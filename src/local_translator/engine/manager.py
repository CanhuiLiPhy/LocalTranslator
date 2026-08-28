"""
中文：翻译任务调度器——队列、子进程生命周期、进度广播。

Job scheduling and supervision.

This is the component that makes the web UI stateless. Every fact about a
translation — queued, running, its progress, its outputs, its token usage — lives
in SQLite and in this manager, never in a page session. A browser refresh
re-renders from :meth:`JobManager.snapshot`; closing the browser entirely does
not touch the worker process. Only an explicit cancel, or shutting the server
down, stops a translation.

Responsibilities:
    - Admit jobs to a FIFO queue and start them subject to a concurrency limit.
    - Own each worker :class:`multiprocessing.Process` and drain its event queue.
    - Persist every state transition to the database.
    - Broadcast events to subscribers so the UI can stream progress.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import multiprocessing as mp
import uuid
from collections import deque
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from local_translator.config import AppPaths, Settings
from local_translator.db import (
    JOB_CANCELLED,
    JOB_COMPLETED,
    JOB_FAILED,
    JOB_RUNNING,
    Database,
    utcnow,
)
from local_translator.engine.worker import (
    EV_ERROR,
    EV_FINISH,
    EV_PROGRESS,
    EV_STARTED,
    JobSpec,
    run_job,
)

logger = logging.getLogger(__name__)

# How long to wait for graceful cancellation before terminating the process.
# 优雅取消的等待时间，超时后强制终止进程。
GRACEFUL_CANCEL_TIMEOUT = 20.0

# Queue polling interval. multiprocessing queues have no awaitable interface, so
# the drain loop polls; 100 ms is imperceptible next to per-paragraph latency.
# multiprocessing 队列没有异步接口，只能轮询；100ms 相对段落延迟可忽略。
POLL_INTERVAL = 0.1


@dataclass
class RunningJob:
    """
    中文：一个正在运行的任务及其子进程资源。

    Bookkeeping for an in-flight job.

    Attributes:
        job_id (str): Job identifier.
        spec (JobSpec): The specification handed to the worker.
        process (mp.Process): The worker process.
        queue (mp.Queue): Event channel from the worker.
        cancel_event (mp.Event): Set to request graceful cancellation.
        drain_task (asyncio.Task | None): Task consuming ``queue``.
        progress (float): Latest overall progress percentage, 0–100.
        stage (str): Latest pipeline stage name.
        stage_current (int): Items completed within the current stage.
        stage_total (int): Total items in the current stage.
        cancelling (bool): Whether a cancel has been requested.
    """

    job_id: str
    spec: JobSpec
    process: mp.Process
    queue: Any
    cancel_event: Any
    drain_task: asyncio.Task | None = None
    progress: float = 0.0
    stage: str = ""
    stage_current: int = 0
    stage_total: int = 0
    cancelling: bool = False


@dataclass
class QueuedJob:
    """
    中文：排队中的任务。

    A job admitted to the queue but not yet started.

    Attributes:
        job_id (str): Job identifier.
        spec (JobSpec): Specification to hand to the worker when it starts.
        doc_title (str): Display title, cached to avoid a database hit per poll.
    """

    job_id: str
    spec: JobSpec
    doc_title: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


class JobManager:
    """
    中文：管理翻译任务的排队、执行、取消与进度广播。

    Owns the lifecycle of every translation job in the process.

    Attributes:
        db (Database): Durable store for jobs and documents.
        paths (AppPaths): Path registry.
        max_concurrent (int): How many workers may run simultaneously.
    """

    def __init__(self, db: Database, paths: AppPaths, max_concurrent: int = 1,
                 ensure_ready=None):
        self.db = db
        self.paths = paths
        self.max_concurrent = max(1, max_concurrent)
        # Called just before a worker is spawned, to make sure the local model
        # is loaded. It lives here rather than at the submission sites because
        # there are six of those in the interface and two of them had forgotten
        # — the library's "translate again" button among them, which produced a
        # job that ran for ten and a half minutes against a server that was
        # never started. Starting the job is the one place every path passes
        # through, and it is also the right moment: the gap between submitting
        # and starting can hold a download.
        # 在派生工作进程之前调用，用于确保本地模型已加载。它放在这里而不是各个提交点，
        # 是因为界面上有六个提交点，其中两个漏了——包括文库里的「重新翻译」按钮，那次
        # 产生的任务对着一个从未启动的服务跑了十分半钟。任务开跑是所有路径的必经之处，
        # 时机也正确：提交到开跑之间可能夹着一次下载。
        self.ensure_ready = ensure_ready

        self._queue: deque[QueuedJob] = deque()
        self._running: dict[str, RunningJob] = {}
        self._subscribers: set[asyncio.Queue] = set()
        self._scheduler_task: asyncio.Task | None = None
        self._wake = asyncio.Event()
        self._closing = False
        self._ctx = mp.get_context("spawn")

    # ------------------------------------------------------------------ #
    # Lifecycle / 生命周期
    # ------------------------------------------------------------------ #

    async def start(self) -> None:
        """
        中文：启动调度循环，并把上次遗留的运行中任务标记为中断。

        Start the scheduler.

        Any job still marked queued or running belongs to a previous server
        process whose workers are gone, so it is reaped to ``interrupted``
        before the scheduler begins.
        """
        self.db.reap_stale_jobs()
        self._closing = False
        self._scheduler_task = asyncio.create_task(self._scheduler_loop())
        logger.info("Job manager started (max concurrent: %d)", self.max_concurrent)

    async def stop(self) -> None:
        """
        中文：停止调度并取消所有在跑的任务。

        Shut down the manager, cancelling every running worker.

        Called on server shutdown. Jobs are cancelled rather than orphaned so no
        worker outlives the process that records its result.
        """
        self._closing = True
        self._wake.set()

        if self._scheduler_task:
            self._scheduler_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._scheduler_task
            self._scheduler_task = None

        for job_id in list(self._running):
            await self.cancel(job_id)

        logger.info("Job manager stopped")

    # ------------------------------------------------------------------ #
    # Submission / 提交
    # ------------------------------------------------------------------ #

    def submit(
        self,
        *,
        doc_id: str,
        input_file: Path,
        output_dir: Path,
        settings: Settings,
        api_key: str | None = None,
        glossary_terms: list[tuple[str, str, str]] | None = None,
        glossary_csv: str | None = None,
        pages: str | None = None,
        doc_title: str = "",
        extract_terms: bool = False,
        latex_source: Path | None = None,
    ) -> str:
        """
        中文：提交一个翻译任务并入队，立即返回任务 id。

        Enqueue a translation job.

        Returns as soon as the job is persisted; execution happens later in a
        worker process. The settings are snapshotted into the job row so that
        changing preferences afterwards does not rewrite history.

        Args:
            doc_id (str): Document identifier; must already exist in the
                ``documents`` table.
            input_file (Path): Absolute path to the source PDF.
            output_dir (Path): Directory to write translated PDFs into.
            settings (Settings): Settings snapshot to run with.
            api_key (str | None): API key for the cloud backend, held in memory
                only and never persisted.
            glossary_terms (list[tuple[str, str, str]] | None): Glossary entries
                as ``(source, target, policy)`` for the local backend.
            glossary_csv (str | None): Glossary CSV path for BabelDOC's own
                glossary mechanism, used by the API backend.
            pages (str | None): BabelDOC page range expression, e.g. ``"1-5"``.
            doc_title (str): Display title for the queue UI.
            extract_terms (bool): Run API-backed term extraction for this
                document and absorb the result into the glossary.
            latex_source (Path | None): An unpacked LaTeX source tree. When
                given, the job translates the paper from its source rather than
                from ``input_file``, which then serves only to identify the
                document in the library.

        Returns:
            str: The new job's identifier.
        """
        job_id = uuid.uuid4().hex
        working_dir = self.paths.meta_dir_for(doc_id)

        spec = JobSpec(
            job_id=job_id,
            doc_id=doc_id,
            input_file=str(input_file),
            output_dir=str(output_dir),
            working_dir=str(working_dir),
            settings=settings.model_dump(mode="json"),
            api_key=api_key,
            glossary_csv=glossary_csv,
            glossary_terms=list(glossary_terms or []),
            pages=pages,
            extract_terms=extract_terms,
            latex_source=str(latex_source) if latex_source else "",
        )

        self.db.create_job(
            job_id,
            doc_id,
            backend=settings.translation.backend,
            config={
                "translation": settings.translation.model_dump(mode="json"),
                "pdf": settings.pdf.model_dump(mode="json"),
                "layout_filter": settings.layout_filter.model_dump(mode="json"),
                "pages": pages,
                "latex": bool(latex_source),
            },
        )

        self._queue.append(QueuedJob(job_id=job_id, spec=spec, doc_title=doc_title))
        self._wake.set()
        self._broadcast({"type": "queued", "job_id": job_id, "doc_id": doc_id})
        logger.info("Queued job %s for document %s", job_id, doc_id)
        return job_id

    async def cancel(self, job_id: str) -> bool:
        """
        中文：取消任务；排队中的直接移除，运行中的先优雅停止再强杀。

        Cancel a queued or running job.

        A queued job is simply dropped. A running job is first asked to stop
        gracefully through its cancel event, which lets BabelDOC unwind between
        paragraphs and release file handles; if it has not exited within
        :data:`GRACEFUL_CANCEL_TIMEOUT` seconds the process is terminated, and
        killed if it still resists.

        Args:
            job_id (str): Job to cancel.

        Returns:
            bool: ``True`` if a job was found and cancelled.
        """
        for i, queued in enumerate(self._queue):
            if queued.job_id == job_id:
                del self._queue[i]
                self.db.update_job(
                    job_id, status=JOB_CANCELLED, finished_at=utcnow(), error="Cancelled while queued"
                )
                self._broadcast({"type": "cancelled", "job_id": job_id})
                logger.info("Removed queued job %s", job_id)
                return True

        running = self._running.get(job_id)
        if running is None:
            return False

        running.cancelling = True
        logger.info("Cancelling running job %s", job_id)
        running.cancel_event.set()

        deadline = asyncio.get_running_loop().time() + GRACEFUL_CANCEL_TIMEOUT
        while running.process.is_alive() and asyncio.get_running_loop().time() < deadline:
            await asyncio.sleep(0.2)

        if running.process.is_alive():
            logger.warning("Job %s ignored graceful cancel; terminating", job_id)
            running.process.terminate()
            await asyncio.sleep(1.0)
            if running.process.is_alive():
                logger.warning("Job %s survived terminate; killing", job_id)
                running.process.kill()

        return True

    def reorder(self, job_id: str, delta: int) -> bool:
        """
        中文：在队列中上移或下移一个任务。

        Move a queued job earlier or later in the queue.

        Only queued jobs can move; a running job has a worker attached and
        reordering it would mean nothing.

        Args:
            job_id (str): Job to move.
            delta (int): Negative to move earlier, positive to move later.

        Returns:
            bool: ``True`` if the job was found and moved.
        """
        for index, queued in enumerate(self._queue):
            if queued.job_id != job_id:
                continue
            target = max(0, min(len(self._queue) - 1, index + delta))
            if target == index:
                return False
            del self._queue[index]
            self._queue.insert(target, queued)
            self._broadcast({"type": "reordered", "job_id": job_id})
            return True
        return False

    async def clear_queue(self) -> int:
        """
        中文：清空排队中的任务，不影响正在运行的。

        Cancel every queued job, leaving running ones alone.

        Running jobs are deliberately untouched: abandoning work that is already
        most of the way through wastes far more than it saves, and the user can
        still stop those individually.

        Returns:
            int: Number of queued jobs removed.
        """
        job_ids = [q.job_id for q in self._queue]
        for job_id in job_ids:
            await self.cancel(job_id)
        return len(job_ids)

    # ------------------------------------------------------------------ #
    # Scheduling / 调度
    # ------------------------------------------------------------------ #

    async def _scheduler_loop(self) -> None:
        """
        中文：调度循环——有空位就从队列取任务启动。

        Continuously admit queued jobs up to the concurrency limit.

        Waits on an event rather than busy-polling, and is woken by submission,
        completion and shutdown.
        """
        while not self._closing:
            while self._queue and len(self._running) < self.max_concurrent:
                await self._start_job(self._queue.popleft())

            self._wake.clear()
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(self._wake.wait(), timeout=5.0)

    async def _start_job(self, queued: QueuedJob) -> None:
        """
        中文：为排队任务创建子进程并开始接收进度事件。

        Spawn a worker process for a queued job.

        Args:
            queued (QueuedJob): The job to start.
        """
        job_id = queued.job_id

        if self.ensure_ready is not None:
            try:
                # Off the event loop: loading several gigabytes of weights takes
                # tens of seconds and would otherwise freeze every open window.
                # 放到线程里：加载数 GB 权重要几十秒，否则会冻住所有打开的窗口。
                await asyncio.to_thread(self.ensure_ready)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Could not make the model ready for job %s", job_id)
                self.db.update_job(
                    job_id,
                    status=JOB_FAILED,
                    error=f"本地模型服务无法启动，这一趟没有开始：{exc}",
                    finished_at=utcnow(),
                )
                self._broadcast({"type": "failed", "job_id": job_id})
                return

        event_queue = self._ctx.Queue()
        cancel_event = self._ctx.Event()

        process = self._ctx.Process(
            target=run_job,
            args=(queued.spec, event_queue, cancel_event),
            name=f"translate-{job_id[:8]}",
            daemon=False,
        )
        process.start()

        running = RunningJob(
            job_id=job_id,
            spec=queued.spec,
            process=process,
            queue=event_queue,
            cancel_event=cancel_event,
        )
        self._running[job_id] = running
        running.drain_task = asyncio.create_task(self._drain(running))

        self.db.update_job(job_id, status=JOB_RUNNING, started_at=utcnow(), pid=process.pid)
        self._broadcast({"type": "started", "job_id": job_id})
        logger.info("Started job %s in pid %d", job_id, process.pid)

    async def _drain(self, running: RunningJob) -> None:
        """
        中文：消费子进程的事件队列，更新数据库并广播进度。

        Consume a worker's event queue until the job settles.

        Also acts as the watchdog for workers that die without reporting: if the
        process exits and the queue is drained without a terminal event, the job
        is failed explicitly rather than being left running forever.

        Args:
            running (RunningJob): The job to drain.
        """
        job_id = running.job_id
        settled = False

        try:
            while True:
                drained_any = False
                while True:
                    try:
                        event = running.queue.get_nowait()
                    except Exception:
                        break
                    drained_any = True
                    if self._handle_event(running, event):
                        settled = True

                if settled:
                    break

                if not running.process.is_alive() and not drained_any:
                    # Give the queue a moment to flush anything written just
                    # before exit, then decide.
                    # 给队列一点时间冲刷退出前写入的事件，再做判断。
                    await asyncio.sleep(0.3)
                    try:
                        event = running.queue.get_nowait()
                    except Exception:
                        event = None
                    if event is not None:
                        if self._handle_event(running, event):
                            settled = True
                            break
                    else:
                        self._settle_dead_worker(running)
                        settled = True
                        break

                await asyncio.sleep(POLL_INTERVAL)

        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Drain loop failed for job %s", job_id)
            self.db.update_job(
                job_id,
                status=JOB_FAILED,
                error="Internal error while tracking job progress",
                finished_at=utcnow(),
            )
        finally:
            with contextlib.suppress(Exception):
                running.process.join(timeout=5)
            with contextlib.suppress(Exception):
                running.queue.close()
            self._running.pop(job_id, None)
            self._wake.set()

    def _handle_event(self, running: RunningJob, event: dict[str, Any]) -> bool:
        """
        中文：处理一条来自子进程的事件，返回该任务是否已终结。

        Apply one worker event to database and subscribers.

        Args:
            running (RunningJob): The job the event belongs to.
            event (dict): Event payload from the worker.

        Returns:
            bool: ``True`` if this event terminates the job.
        """
        etype = event.get("type")
        job_id = running.job_id

        if etype == EV_STARTED:
            self._broadcast({"type": "running", "job_id": job_id})
            return False

        if etype == EV_PROGRESS:
            running.progress = event.get("overall", 0.0)
            running.stage = event.get("stage", "")
            running.stage_current = event.get("stage_current", 0)
            running.stage_total = event.get("stage_total", 0)
            self.db.update_job(
                job_id,
                progress=running.progress,
                stage=running.stage,
                stage_current=running.stage_current,
                stage_total=running.stage_total,
            )
            self._broadcast(
                {
                    "type": "progress",
                    "job_id": job_id,
                    "progress": running.progress,
                    "stage": running.stage,
                    "stage_current": running.stage_current,
                    "stage_total": running.stage_total,
                }
            )
            return False

        if etype == EV_FINISH:
            usage = event.get("usage") or {}
            self._absorb_glossary(event.get("glossary_path"))
            self.db.update_job(
                job_id,
                status=JOB_COMPLETED,
                progress=100.0,
                stage="completed",
                finished_at=utcnow(),
                duration_ms=event.get("duration_ms"),
                dual_path=event.get("dual_path"),
                mono_path=event.get("mono_path"),
                glossary_path=event.get("glossary_path"),
                tracking_path=event.get("tracking_path"),
                **_usage_columns(usage),
            )
            self._broadcast(
                {
                    "type": "completed",
                    "job_id": job_id,
                    "dual_path": event.get("dual_path"),
                    "mono_path": event.get("mono_path"),
                    "usage": usage,
                    "skipped": event.get("skipped", {}),
                }
            )
            logger.info("Job %s completed in %sms", job_id, event.get("duration_ms"))
            return True

        if etype == EV_ERROR:
            cancelled = event.get("cancelled") or running.cancelling
            status = JOB_CANCELLED if cancelled else JOB_FAILED
            usage = event.get("usage") or {}
            self.db.update_job(
                job_id,
                status=status,
                error=event.get("error"),
                finished_at=utcnow(),
                duration_ms=event.get("duration_ms"),
                **_usage_columns(usage),
            )
            self._broadcast(
                {"type": status, "job_id": job_id, "error": event.get("error")}
            )
            if cancelled:
                logger.info("Job %s cancelled", job_id)
            else:
                logger.error("Job %s failed: %s\n%s", job_id, event.get("error"), event.get("traceback"))
            return True

        logger.debug("Ignoring unknown worker event %r", etype)
        return False

    def _absorb_glossary(self, csv_path: str | None) -> None:
        """
        中文：把本次自动提取的术语并入术语库，完成累积闭环。

        Fold a job's automatically extracted terms into the cumulative glossary.

        This is what closes the accumulation loop: terms the API extracted for
        one document become part of the glossary that every later document is
        checked against, so coverage rises and the API is consulted less often.

        Existing entries are never overwritten, so a term the user corrected by
        hand survives every subsequent extraction. Pairs where the extractor
        returned the source unchanged — which it does for proper nouns like
        model and dataset names — are stored as whitelist entries, since that is
        exactly the judgement the whitelist encodes.

        What the deterministic cleaning rules condemn is dropped here rather
        than absorbed and cleaned later. Extraction runs after every document,
        so without this the glossary refills with what the last cleaning
        removed: five test runs put 2,087 entries back, author names and journal
        abbreviations among them.

        Args:
            csv_path (str | None): Path to BabelDOC's extracted glossary CSV.
                ``None`` or a missing file is a no-op.
        """
        if not csv_path:
            return
        path = Path(csv_path)
        if not path.is_file():
            return

        try:
            import csv as _csv

            from local_translator.glossary.store import (
                POLICY_KEEP,
                POLICY_TRANSLATE,
                GlossaryStore,
                TermRecord,
            )

            content = path.read_bytes().decode("utf-8-sig", errors="replace")
            rows = list(_csv.DictReader(content.splitlines()))

            from local_translator.glossary.clean import is_junk

            records = []
            rejected = 0
            for row in rows:
                src = (row.get("source") or "").strip()
                tgt = (row.get("target") or "").strip()
                if not src or not tgt:
                    continue
                keep = src.casefold() == tgt.casefold()
                candidate = {
                    "src": src,
                    "tgt": src if keep else tgt,
                    "policy": POLICY_KEEP if keep else POLICY_TRANSLATE,
                    "source": "auto-api",
                }
                if is_junk(candidate):
                    rejected += 1
                    continue
                records.append(
                    TermRecord(
                        src=candidate["src"],
                        tgt=candidate["tgt"],
                        policy=candidate["policy"],
                        domain=None,
                        source="auto-api",
                        confidence=0.8,
                    )
                )

            if not records:
                if rejected:
                    logger.info("Extraction produced %d term(s), all rejected by the "
                                "cleaning rules", rejected)
                return

            result = GlossaryStore(self.db).upsert_many(records)
            logger.info(
                "Absorbed %d new term(s) from automatic extraction "
                "(%d already known, %d rejected by the cleaning rules)",
                result["inserted"],
                result["skipped"],
                rejected,
            )
            self._broadcast({"type": "glossary_grew", "inserted": result["inserted"]})
        except Exception:
            logger.warning("Could not absorb extracted glossary %s", csv_path, exc_info=True)

    def _settle_dead_worker(self, running: RunningJob) -> None:
        """
        中文：子进程异常退出且未上报结果时，把任务标记为失败或已取消。

        Settle a job whose worker exited without a terminal event.

        This covers hard kills, out-of-memory kills and native crashes inside
        PDF parsing — cases where no Python-level exception handler ever runs.

        Args:
            running (RunningJob): The job whose worker died.
        """
        code = running.process.exitcode
        if running.cancelling:
            status, message = JOB_CANCELLED, "Cancelled by user"
        else:
            status = JOB_FAILED
            message = f"Worker process exited unexpectedly (exit code {code})"
            logger.error("Job %s: %s", running.job_id, message)

        self.db.update_job(
            running.job_id, status=status, error=message, finished_at=utcnow()
        )
        self._broadcast({"type": status, "job_id": running.job_id, "error": message})

    # ------------------------------------------------------------------ #
    # Observation / 状态观察
    # ------------------------------------------------------------------ #

    def snapshot(self) -> dict[str, Any]:
        """
        中文：返回当前队列与运行中任务的实时状态，供前端渲染。

        Capture the live scheduler state.

        The UI renders entirely from this plus database history, which is why a
        page refresh loses nothing: no progress state exists in the browser
        session to begin with.

        Returns:
            dict[str, Any]: Keys ``running`` (list of dicts with ``job_id``,
                ``progress``, ``stage``, ``stage_current``, ``stage_total``,
                ``cancelling``), ``queued`` (list of ``job_id``/``title``), and
                ``capacity``.
        """
        return {
            "running": [
                {
                    "job_id": r.job_id,
                    "doc_id": r.spec.doc_id,
                    "progress": r.progress,
                    "stage": r.stage,
                    "stage_current": r.stage_current,
                    "stage_total": r.stage_total,
                    "cancelling": r.cancelling,
                }
                for r in self._running.values()
            ],
            "queued": [
                {"job_id": q.job_id, "doc_id": q.spec.doc_id, "title": q.doc_title}
                for q in self._queue
            ],
            "capacity": self.max_concurrent,
        }

    def is_active(self, job_id: str) -> bool:
        """
        中文：判断任务是否在队列中或正在运行。

        Args:
            job_id (str): Job identifier.

        Returns:
            bool: ``True`` if the job is queued or running.
        """
        return job_id in self._running or any(q.job_id == job_id for q in self._queue)

    # ------------------------------------------------------------------ #
    # Event broadcasting / 事件广播
    # ------------------------------------------------------------------ #

    def _broadcast(self, event: dict[str, Any]) -> None:
        """
        中文：向所有订阅者推送事件；订阅者积压过多时丢弃最旧事件。

        Publish an event to every subscriber.

        Subscriber queues are bounded and drop their oldest entry when full, so
        a stalled or disconnected client can never grow memory without bound or
        block the scheduler.

        Args:
            event (dict): Event payload.
        """
        for queue in list(self._subscribers):
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                with contextlib.suppress(asyncio.QueueEmpty):
                    queue.get_nowait()
                with contextlib.suppress(asyncio.QueueFull):
                    queue.put_nowait(event)

    async def subscribe(self) -> AsyncIterator[dict[str, Any]]:
        """
        中文：订阅任务事件流，用于 SSE 推送。

        Subscribe to the job event stream.

        Yields:
            dict[str, Any]: Events as they occur — ``queued``, ``started``,
                ``running``, ``progress``, ``completed``, ``failed``,
                ``cancelled``.
        """
        queue: asyncio.Queue = asyncio.Queue(maxsize=256)
        self._subscribers.add(queue)
        try:
            while True:
                yield await queue.get()
        finally:
            self._subscribers.discard(queue)


def _usage_columns(usage: dict[str, Any]) -> dict[str, int]:
    """
    中文：把翻译器的用量汇总映射为数据库列。

    Map a translator's usage summary onto job table columns.

    Args:
        usage (dict): Output of ``translator.usage_summary()``; may be empty if
            the job failed before a translator was constructed.

    Returns:
        dict[str, int]: Column/value pairs, defaulting to zero.
    """
    return {
        "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
        "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
        "total_tokens": int(usage.get("total_tokens", 0) or 0),
        "para_total": int(usage.get("para_total", 0) or 0),
        "para_cache_hit": int(usage.get("para_cache_hit", 0) or 0),
        "placeholder_retries": int(usage.get("placeholder_retries", 0) or 0),
        "placeholder_failures": int(usage.get("placeholder_failures", 0) or 0),
    }
