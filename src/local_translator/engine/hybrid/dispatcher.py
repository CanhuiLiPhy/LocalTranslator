"""
中文：API 侧的并发分发器——多线程、按字符量成批，取代原先的单线程聚合器。

Concurrent request dispatch for the API side of hybrid translation.

This replaces an earlier design that collected paragraphs into batches on a
single background thread. That design optimised the wrong quantity. Measured
against an OpenAI-compatible cloud endpoint:

    concurrency   throughput      mean latency
    1             116 chars/s     1.71s
    4             325 chars/s     1.95s
    16            749 chars/s     1.66s

Throughput scales almost linearly with concurrency while latency stays flat,
which means the endpoint is genuinely parallel and the only way to go faster is
to have more requests in flight. Batching does the opposite: packing twelve
paragraphs into one request converts twelve potentially concurrent requests
into one serial one. Instrumenting the old design confirmed it — seven batched
requests averaging 10.45 seconds each, running at a measured concurrency of
exactly 1.00, for 73 seconds of pure serial time.

So batching is kept, but subordinated to concurrency and bounded differently:

* **Several worker threads**, so requests overlap. This is the change that
  matters; everything else is refinement.
* **Batches bounded by characters, not paragraph count.** A twelve-paragraph
  limit packed the document's longest paragraphs into single enormous requests
  — precisely the ones that most needed to run alongside each other. A
  character bound keeps every request roughly the same size, which keeps
  latency uniform and the pipeline full.
* **Long paragraphs travel alone**, since a batch containing one is already at
  its bound.

Batching still earns its place for short paragraphs: a 5-character paragraph
costs ~2.0s as its own request and almost nothing as part of a batch, and the
document's 234 short paragraphs would otherwise spend that overhead 234 times.
"""

from __future__ import annotations

import logging
import queue
import re
import threading
from concurrent.futures import Future

logger = logging.getLogger(__name__)

# Segment delimiter. Chosen to be something no PDF paragraph would contain and
# no model would emit spontaneously, so a split cannot fire on real content.
# 段落分隔符：选一个 PDF 正文不会出现、模型也不会自发产出的标记，避免误切。
SEGMENT_MARK = "⟦§{n}⟧"
_SEGMENT_RE = re.compile(r"⟦§\s*(\d+)\s*⟧")

BATCH_PROMPT = """Translate each numbered segment below into {language}.

Rules:
- Output every segment, prefixed by its own ⟦§n⟧ marker, in the same order.
- Translate only the text. Copy tokens like {{v1}} and <style id='0'> verbatim.
- No commentary, no explanations, nothing outside the segments.

{segments}"""


class ApiDispatcher:
    """
    中文：多线程分发 API 请求，按字符量攒批。

    Dispatch API requests across several worker threads.

    Attributes:
        threads (int): Worker threads, and therefore the ceiling on how many
            requests are in flight at once.
        max_chars (int): Character bound for one batched request.
        max_batch (int): Paragraph bound for one batched request, a secondary
            guard so a request full of tiny fragments stays splittable.
        max_wait (float): Seconds to wait for a batch to fill. Kept short: a
            worker waiting to fill a batch is a worker not issuing requests,
            and idle concurrency costs more than a half-empty batch.
    """

    def __init__(
        self,
        translate_batch,
        translate_single,
        threads: int = 12,
        max_chars: int = 1200,
        max_batch: int = 10,
        max_wait: float = 0.15,
    ):
        """
        中文：构造分发器并启动工作线程。

        Args:
            translate_batch (Callable[[list[str]], list[str] | None]): Sends one
                batched request, returning one translation per input, or
                ``None`` when the reply could not be split reliably.
            translate_single (Callable[[str], str]): Translates one paragraph.
            threads (int): Number of worker threads.
            max_chars (int): Character bound per request.
            max_batch (int): Paragraph bound per request.
            max_wait (float): Seconds to wait for a batch to fill.
        """
        self._translate_batch = translate_batch
        self._translate_single = translate_single
        self.threads = max(1, threads)
        self.max_chars = max_chars
        self.max_batch = max_batch
        self.max_wait = max_wait

        self._queue: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._batches = 0
        self._batched_items = 0
        self._single_items = 0
        self._split_failures = 0
        self._peak_inflight = 0
        self._inflight = 0

        self._workers = [
            threading.Thread(target=self._run, name=f"api-dispatch-{i}", daemon=True)
            for i in range(self.threads)
        ]
        for worker in self._workers:
            worker.start()

    def submit(self, text: str) -> str:
        """
        中文：提交一个段落，阻塞直到译文返回。

        Submit a paragraph and wait for its translation.

        Args:
            text (str): Paragraph text.

        Returns:
            str: The translation.

        Raises:
            Exception: Whatever the underlying request raised.
        """
        future: Future = Future()
        self._queue.put((text, future))
        return future.result()

    def _collect(self) -> list[tuple[str, Future]]:
        """
        中文：收集一批段落，受字符量、条数与等待时间三重约束。

        Gather one batch, bounded by characters, count and wait time.

        A paragraph that would push the batch past its character bound is left
        in the queue for another worker rather than making this request larger:
        with several workers running, leaving it behind means it starts sooner,
        not later.

        Returns:
            list[tuple[str, Future]]: The batch, empty if nothing arrived.
        """
        try:
            first = self._queue.get(timeout=0.2)
        except queue.Empty:
            return []

        batch = [first]
        chars = len(first[0])
        # A paragraph at or over the bound is already a full request.
        # 超过上限的段落本身就构成一次完整请求。
        if chars >= self.max_chars:
            return batch

        deadline = threading.Event()
        timer = threading.Timer(self.max_wait, deadline.set)
        timer.start()
        try:
            while len(batch) < self.max_batch and not deadline.is_set():
                try:
                    item = self._queue.get(timeout=0.02)
                except queue.Empty:
                    continue
                if chars + len(item[0]) > self.max_chars and len(batch) > 1:
                    # Hand it back for a peer worker to pick up immediately.
                    # 放回队列，让其他工作线程立刻取走。
                    self._queue.put(item)
                    break
                batch.append(item)
                chars += len(item[0])
                if chars >= self.max_chars:
                    break
        finally:
            timer.cancel()
        return batch

    def _run(self) -> None:
        """
        中文：工作线程主循环——收批、翻译、回填 Future。

        Worker loop: collect a batch, translate it, resolve its futures.
        """
        while not self._stop.is_set():
            batch = self._collect()
            if not batch:
                continue

            texts = [text for text, _ in batch]
            futures = [future for _, future in batch]

            with self._lock:
                self._inflight += 1
                self._peak_inflight = max(self._peak_inflight, self._inflight)
            try:
                try:
                    results = self._translate_batch(texts) if len(texts) > 1 else None
                except Exception:
                    logger.warning("Batched request failed; falling back to individual",
                                   exc_info=True)
                    results = None
            finally:
                with self._lock:
                    self._inflight -= 1

            if results is not None and len(results) == len(texts):
                with self._lock:
                    self._batches += 1
                    self._batched_items += len(texts)
                for future, result in zip(futures, results, strict=True):
                    future.set_result(result)
                continue

            # Either the reply could not be split, or this was a single item.
            # Translating individually is slower but never mispairs a paragraph
            # with someone else's translation.
            # 无法可靠切分或本就单条时逐条翻译：慢一些，但绝不会张冠李戴。
            if len(texts) > 1:
                with self._lock:
                    self._split_failures += 1
            for text, future in zip(texts, futures, strict=True):
                with self._lock:
                    self._single_items += 1
                    self._inflight += 1
                    self._peak_inflight = max(self._peak_inflight, self._inflight)
                try:
                    future.set_result(self._translate_single(text))
                except Exception as exc:  # noqa: BLE001
                    future.set_exception(exc)
                finally:
                    with self._lock:
                        self._inflight -= 1

    def stop(self) -> None:
        """中文：停止所有工作线程。 / Stop every worker thread."""
        self._stop.set()

    def stats(self) -> dict[str, int]:
        """
        中文：返回分发统计，含实测并发峰值。

        Report dispatch effectiveness.

        Returns:
            dict[str, int]: Batched request count, paragraphs served by them,
                paragraphs sent individually, split failures, and the peak
                number of requests observed in flight — the figure that says
                whether concurrency was actually achieved.
        """
        with self._lock:
            return {
                "api_batches": self._batches,
                "api_batched_paragraphs": self._batched_items,
                "api_single_paragraphs": self._single_items,
                "api_split_failures": self._split_failures,
                "api_peak_inflight": self._peak_inflight,
                "api_dispatch_threads": self.threads,
            }


def build_batch_prompt(texts: list[str], language: str) -> str:
    """
    中文：把多个段落拼成一个带编号的批量提示词。

    Compose a numbered multi-segment prompt.

    Args:
        texts (list[str]): Paragraphs to translate.
        language (str): Target language name.

    Returns:
        str: The prompt.
    """
    segments = "\n\n".join(
        f"{SEGMENT_MARK.format(n=i)}\n{text}" for i, text in enumerate(texts, 1)
    )
    return BATCH_PROMPT.format(language=language, segments=segments)


def split_batch_response(reply: str, expected: int) -> list[str] | None:
    """
    中文：按编号标记切分批量响应，数量对不上就返回 None。

    Split a batched reply back into per-paragraph translations.

    Returns ``None`` rather than a best guess when the segment count does not
    match. A wrong split would pair paragraphs with the wrong translations
    throughout the rest of the batch, which is far worse than the cost of
    retrying them one at a time.

    Args:
        reply (str): The model's reply.
        expected (int): Number of segments that were sent.

    Returns:
        list[str] | None: One translation per input, or ``None`` if the reply
            could not be split into exactly ``expected`` non-empty segments.
    """
    if not reply:
        return None

    matches = list(_SEGMENT_RE.finditer(reply))
    if len(matches) != expected:
        logger.debug("Batch reply had %d markers, expected %d", len(matches), expected)
        return None

    parts: list[str] = []
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(reply)
        parts.append(reply[start:end].strip())

    if any(not part for part in parts):
        logger.debug("Batch reply contained an empty segment")
        return None

    # Markers must be numbered 1..n in order; anything else means the model
    # reordered or renumbered, and positional pairing can no longer be trusted.
    # 编号必须是 1..n 且有序；否则说明模型重排或改号，按位置配对不再可信。
    if [int(m.group(1)) for m in matches] != list(range(1, expected + 1)):
        logger.debug("Batch reply markers were out of order")
        return None

    return parts
