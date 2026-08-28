"""
中文：应用服务层——把数据库、任务调度、模型服务收拢成一个门面。

Application service layer.

A single :class:`TranslatorService` owns every long-lived resource: the
database, the job manager, and the local inference server supervisor. Both the
REST API and the NiceGUI pages talk to this object rather than to those
components directly.

This is what keeps the front end stateless. A page handler never holds
translation state of its own — it asks the service what is happening right now
and renders that. Refreshing the browser therefore re-renders from live truth
instead of losing anything, and closing the browser has no effect at all on a
running job.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from local_translator import edition
from local_translator.config import AppPaths, Settings, get_paths, get_settings, save_settings
from local_translator.db import Database, get_db
from local_translator.engine.manager import JobManager
from local_translator.library import (
    SOURCE_ARXIV,
    SOURCE_LOCAL,
    SOURCE_UPLOAD,
    already_translated,
    base_title,
    next_copy_index,
    register_document,
)
from local_translator.llm.idle import IdleUnloader
from local_translator.llm.supervisor import LlamaServerError, LlamaServerSupervisor
from local_translator.secrets import get_api_key

logger = logging.getLogger(__name__)

# PDF discovery for folder import / 文件夹导入时识别的扩展名
PDF_SUFFIXES = {".pdf"}


@dataclass
class PendingSubmission:
    """
    中文：一次已准备好、但因标题重复而等待用户确认的提交。

    A submission that is ready to run but is waiting on the user's decision.

    The source is already on disk by the time this exists — an upload has been
    staged, an arXiv paper downloaded — so confirming costs nothing more than a
    database write, and declining leaves the file where it is for next time.

    Attributes:
        source (Path): The PDF on disk, staged or downloaded.
        source_kind (str): ``"local"``, ``"upload"`` or ``"arxiv"``.
        title (str): Title without any copy suffix.
        source_ref (str | None): Original reference, e.g. an arXiv abs URL.
        authors (str | None): Comma-separated authors, when known.
        pages (str | None): BabelDOC page range expression.
        latex_source (Path | None): An unpacked LaTeX source tree to translate
            instead of the PDF. The PDF is still what identifies the document,
            so a paper fetched from arXiv is the same library entry whichever
            way it is translated.
    """

    source: Path
    source_kind: str
    title: str
    source_ref: str | None = None
    authors: str | None = None
    pages: str | None = None
    latex_source: Path | None = None


class DuplicateDocument(Exception):
    """
    中文：库里已有同标题文档，提交被暂停等待确认。

    Raised when a submission's title already exists in the library.

    Carries everything needed to go ahead anyway, so the caller can put the
    question to the user and then complete the submission without redoing the
    download or the upload.

    Attributes:
        pending (PendingSubmission): The submission awaiting a decision.
        existing (list[dict]): The library rows that share this title, oldest
            first.
        copy_index (int): The number the copy would be given.
    """

    def __init__(self, pending: PendingSubmission, existing: list, copy_index: int):
        self.pending = pending
        self.existing = [dict(row) for row in existing]
        self.copy_index = copy_index
        super().__init__(f"《{pending.title}》已在文库中")


def build_translator_for_cli(settings, backend: str | None = None):
    """
    中文：按设置造一个纯文本翻译器，供不经过任务队列的命令使用。

    Build a text translator for commands that do not go through the job queue.

    The LaTeX path translates blocks of source directly rather than submitting
    a document to the scheduler, so it needs a translator without a job around
    it. This is the same construction :meth:`TranslatorService._build_text_engine`
    performs, reduced to what a one-shot command needs — no glossary matcher,
    which depends on a running service, and no hybrid pool, which exists to
    share a local model between concurrent documents.

    Args:
        settings (Settings): The resolved settings.
        backend (str | None): Override for ``translation.backend``.

    Returns:
        BaseTranslator: Ready to use.

    Raises:
        ValueError: If the cloud backend is chosen without a key in the
            keychain, which is a setup problem the reader has to fix.
    """
    from local_translator.engine.translators import ApiTranslator, LocalMTTranslator
    from local_translator.secrets import get_api_key

    tr = settings.translation
    which = backend or tr.backend
    # Hybrid exists to share one local model between concurrent documents.
    # A single command has no concurrency to share, so it takes the local half.
    # 混合模式是为了让多个并发文档共用一个本地模型。单条命令没有并发可共用，取其本地那半。
    if which == "hybrid":
        which = "local"

    if which == "api":
        api = settings.api_model
        key = get_api_key(api.keyring_account)
        if not key:
            raise ValueError(f"没有找到 {api.display_name} 的 API key")
        return ApiTranslator(
            lang_in=tr.lang_in, lang_out=tr.lang_out, model=api.model_name,
            base_url=api.base_url or None, api_key=key,
            ignore_cache=tr.ignore_cache, skip_reasoning=api.skip_reasoning,
            prompt_template=api.chat_prompt_template,
        )
    return LocalMTTranslator(
        lang_in=tr.lang_in, lang_out=tr.lang_out, settings=settings.local_model,
        ignore_cache=tr.ignore_cache,
        glossary_rev=settings.glossary.glossary_rev,
        custom_hint=tr.custom_system_prompt,
    )


def build_glossary_for_cli(settings):
    """
    中文：为不经过服务进程的命令构造术语匹配器。

    Build a glossary matcher for commands that do not run the service.

    Args:
        settings (Settings): The resolved settings.

    Returns:
        GlossaryMatcher | None: ``None`` when the glossary is switched off or
            has nothing in it — which the callers read as "no terms apply",
            the same thing an empty glossary means inside the service.
    """
    if not settings.glossary.enabled:
        return None
    from local_translator.config import get_paths
    from local_translator.db import Database
    from local_translator.glossary.matcher import GlossaryMatcher

    database = Database(get_paths().db_path)
    rows = database.query(
        "SELECT src, tgt, policy FROM glossary_terms WHERE enabled = 1")
    terms = [(row["src"], row["tgt"], row["policy"]) for row in rows]
    if not terms:
        return None
    return GlossaryMatcher(terms, max_terms=settings.glossary.max_terms_per_prompt)


def recognise_and_translate(image: bytes, translate):
    """
    中文：识别图片里的文字并逐行翻译，返回识别结果、译文缓存与逐行对照。

    Recognise the text in an image and translate it line by line.

    Args:
        image (bytes): Image data in any format the recogniser reads.
        translate (Callable[[str], str]): Translates one line.

    Returns:
        tuple | None: ``(result, cache, pairs)`` — the OCR result, a
            text-to-translation cache, and the per-line pairing the overlay
            renderer expects. ``None`` when the models are not installed or
            nothing was recognised.

    Note:
        Each distinct line is translated once, so a header repeated across the
        image costs one call rather than one per occurrence.
        每一种不同的文本只翻一次，图中重复出现的表头因此只花一次调用，而不是每出现一次
        就花一次。
    """
    from local_translator.ocr import get_ocr

    ocr = get_ocr()
    if not ocr.available():
        return None
    result = ocr.read(image)
    if not result.lines:
        return None

    cache: dict[str, str] = {}
    pairs = []
    for line in result.lines:
        text = line.text.strip()
        if text and text not in cache:
            try:
                cache[text] = translate(text)
            except Exception:  # noqa: BLE001
                logger.warning("Line translation failed", exc_info=True)
                cache[text] = ""
        pairs.append({
            "text": line.text,
            "translation": cache.get(text, ""),
            "confidence": round(line.confidence, 3),
            "box": line.box,
        })
    return result, cache, pairs


def with_protected_terms(translate, glossary):
    """
    中文：给逐行翻译套一层白名单保护，让术语原样穿过。

    Wrap a line translator so whitelisted terms survive verbatim.

    Args:
        translate (Callable[[str], str]): Translates one line.
        glossary (Callable | None): The glossary matcher, or ``None``.

    Returns:
        Callable[[str], str]: The same function when there is no glossary, and
            otherwise one that holds every whitelisted term behind a marker for
            the duration of the call.

    Note:
        Figures are the strict case, and always have been: there is no prose in
        an axis label for a marker to disturb, and a 「基线」 where the paper says
        「baseline」 is exactly what a whitelist exists to prevent. The PDF path
        protects figure text unconditionally for the same reason.
        插图是严格保护的场景，一向如此：坐标轴标签里没有会被标记打乱的行文，而论文写
        baseline、图上却写「基线」，正是白名单要防的事。PDF 那条路出于同样的理由对图内
        文字无条件保护。
    """
    if glossary is None:
        return translate
    from local_translator.engine.translators import protect_keeps, restore_keeps

    def call(text: str) -> str:
        """中文：保护、翻译、还原。 / Protect, translate, restore."""
        _, keeps = glossary(text)
        guarded, held = protect_keeps(text, keeps)
        return restore_keeps(translate(guarded), held)

    return call


def render_translated_image(image: bytes, translate) -> bytes | None:
    """
    中文：把一张图里的文字译过来并原位画回去，返回 PNG。

    Translate the text in an image and draw it back in place.

    Args:
        image (bytes): Image data.
        translate (Callable[[str], str]): Translates one line.

    Returns:
        bytes | None: A PNG with the translation over the original text, or
            ``None`` when the recogniser is not installed, nothing was
            recognised, or nothing came back translated. Each of those means
            "keep the original picture", which is why they share an answer.

    Note:
        This is what the LaTeX path calls for a figure. It is deliberately the
        same recognition and the same overlay the image page uses, so a figure
        inside a paper and a screenshot pasted into the box are treated alike.
        这是 LaTeX 那条路处理插图时调用的东西。它有意与图片页用的是同一套识别和同一套
        回填，好让论文里的插图与粘贴进输入框的截图得到一样的对待。
    """
    from local_translator.ocr.overlay import render_overlay

    found = recognise_and_translate(image, translate)
    if found is None:
        return None
    _, _, pairs = found
    if not any(pair["translation"] for pair in pairs):
        return None
    try:
        return render_overlay(image, pairs)
    except Exception:  # noqa: BLE001
        logger.warning("Overlay rendering failed", exc_info=True)
        return None


def _trash(path: Path) -> None:
    """
    中文：把文件移入废纸篓；没有废纸篓的平台上才真正删除。

    Move a file to the trash rather than unlinking it.

    A translated PDF can be an hour of local inference or a few dollars of API
    time. Deleting the library row is reversible — translate it again — but
    deleting the file is not, so it goes somewhere recoverable when the platform
    has one.

    Args:
        path (Path): File to remove.

    Raises:
        OSError: When the file could be neither moved nor deleted.
    """
    import shutil
    import sys
    import time

    if sys.platform == "darwin":
        trash = Path.home() / ".Trash"
        if trash.is_dir():
            target = trash / path.name
            if target.exists():
                stamp = time.strftime("%Y%m%d-%H%M%S")
                target = trash / f"{path.stem}-{stamp}{path.suffix}"
            shutil.move(str(path), str(target))
            return
    path.unlink()


class TranslatorService:
    """
    中文：应用服务门面，管理数据库、任务调度与本地模型服务的生命周期。

    Facade over the application's long-lived components.

    Attributes:
        paths (AppPaths): Path registry.
        settings (Settings): Current user settings.
        db (Database): Durable store.
        manager (JobManager): Job scheduler and worker supervisor.
        supervisor (LlamaServerSupervisor): Local inference server lifecycle.
        idle_unloader (IdleUnloader): Releases the local model after a spell
            with nothing to translate. Inert unless configured.
    """

    def __init__(self, paths: AppPaths | None = None, settings: Settings | None = None):
        self.paths = paths or get_paths()
        self.settings = settings or get_settings()
        self.db: Database = get_db(self.paths.db_path)
        self.manager = JobManager(
            self.db,
            self.paths,
            max_concurrent=self.settings.server.max_concurrent_jobs,
            ensure_ready=self.ensure_model_ready,
        )
        self.supervisor = LlamaServerSupervisor(self.paths, self.settings.local_model)
        self.idle_unloader = IdleUnloader(self, self.settings.local_model.idle_unload_seconds)
        self.idle_unloader.start()

    # ------------------------------------------------------------------ #
    # Lifecycle / 生命周期
    # ------------------------------------------------------------------ #

    async def startup(self) -> None:
        """
        中文：启动服务；本地模型按需惰性启动，不阻塞 Web 服务上线。

        Start background components.

        The inference server is deliberately *not* started here. Loading 6 GB of
        weights takes long enough that doing it eagerly would delay the web UI
        from becoming reachable, and a user who only wants to browse their
        library never needs the model at all. It is started on demand by
        :meth:`ensure_model_ready` when the first local job is submitted.
        """
        _install_bundled_glossary()
        await self.manager.start()
        logger.info("Translator service ready")

    async def shutdown(self) -> None:
        """
        中文：关闭服务，取消所有在跑任务并停止本地模型服务。

        Stop background components and release resources.
        """
        await self.manager.stop()
        self.idle_unloader.stop()
        # An explicit shutdown should release the model's memory even when this
        # run adopted the server rather than starting it.
        # 显式退出时应释放模型内存，即便本次运行是接管而非启动了该服务。
        self.supervisor.stop(include_adopted=True)

        # After the model server is down and before the database closes: the
        # stores that grow with use rather than per job get their tidy-up here,
        # once, instead of on a timer nobody would notice.
        # 在模型服务停下之后、数据库关闭之前：那些「随使用增长」而非「随任务产生」的存储
        # 在这里收拾一次，而不是靠一个没人会注意的定时器。
        try:
            from local_translator.engine import intermediates

            intermediates.on_exit(
                self.paths,
                keep_intermediates=self.settings.translation.keep_intermediates,
                cache_mb=self.settings.translation.translation_cache_mb)
        except Exception:  # noqa: BLE001
            logger.debug("Exit cleanup did not run", exc_info=True)

        self.db.close()
        logger.info("Translator service stopped")

    def ensure_model_ready(self) -> None:
        """
        中文：确保本地推理服务可用（仅在使用本地后端时）。

        Make sure the local inference server is reachable before a local job runs.

        Raises:
            LlamaServerError: If the server is unavailable and cannot be started.
        """
        if self.settings.translation.backend not in ("local", "hybrid"):
            return
        # Starting a job is activity even before the manager reports it running,
        # so the timer is reset here rather than only in the watchdog's own poll.
        # 任务开始就算活动，此时管理器还没把它标成 running，因此在这里清零计时，
        # 而不是只靠看门狗自己的轮询。
        self.idle_unloader.note_activity()
        self.supervisor.ensure_running()
        # And again once the weights are actually in memory. Loading is not
        # instantaneous — measured at 29 seconds for the 1.8B on a MacBook Air —
        # and the watchdog keeps ticking throughout, so a timer reset only
        # before the load leaves the model eligible for unloading the moment it
        # becomes ready. That is not hypothetical: on that machine the server
        # was stopped five seconds after reporting ready, before the caller had
        # finished fetching the PDF, and the job then translated nothing.
        # 加载完成后再清一次。加载并非瞬时（1.8B 在 MacBook Air 上实测 29 秒），而看门狗
        # 全程照常计时，因此只在加载前清零，会让模型在就绪的那一刻就已具备被卸载的资格。
        # 这不是假设：在那台机器上，服务报告就绪五秒后即被停止——此时调用方还在下载 PDF——
        # 随后那次任务一个字也没译出来。
        self.idle_unloader.note_activity()

    def set_backend(self, backend: str) -> None:
        """
        中文：切换翻译后端（本地 / 云端 / 混合）并持久化。

        Switch the translation backend and persist the choice.

        Queued jobs pick up the new backend when they start; a job already
        running keeps the one it began with, since its worker process was
        given the settings at launch.

        Args:
            backend (str): ``"local"``, ``"api"`` or ``"hybrid"``.

        Raises:
            ValueError: If the backend is not recognised, or is not available in
                the running edition — the lite edition has no local model, so
                neither ``local`` nor ``hybrid`` can be selected.
        """
        if backend not in ("local", "api", "hybrid"):
            raise ValueError(f"Unknown backend: {backend!r}")
        if not edition.supports(backend):
            raise ValueError(
                f"{edition.display_name()}不支持「{backend}」后端，只能使用云端 API"
            )
        if self.settings.translation.backend == backend:
            return

        self.settings.translation.backend = backend
        save_settings(self.settings)
        logger.info("Translation backend switched to %s", backend)

    def set_speculative(self, enabled: bool) -> None:
        """
        中文：开关投机解码并在必要时重启推理服务。

        Turn speculative decoding on or off.

        The draft model is a command-line argument to llama-server, so unlike
        sampling parameters it cannot be applied to a running process and the
        server has to restart. Speculative decoding is lossless — it changes
        speed, never output — so switching it costs nothing but the restart.

        Args:
            enabled (bool): Whether to run with a draft model.

        Raises:
            LlamaServerError: If the server fails to restart.
        """
        if self.settings.local_model.speculative == enabled:
            return

        self.settings.local_model.speculative = enabled
        save_settings(self.settings)
        self.supervisor.settings = self.settings.local_model

        if self.supervisor.health():
            logger.info("Restarting inference server with speculative=%s", enabled)
            self.supervisor.restart_for_model_change()

    def model_status(self) -> dict[str, Any]:
        """
        中文：返回本地模型服务状态，供 UI 显示指示灯。

        Report the local model server's status for display.

        Returns:
            dict[str, Any]: Keys ``healthy`` (bool), ``endpoint`` (str),
                ``model`` (str) and ``installed`` (bool).
        """
        model_path = self.supervisor.model_path()
        draft = self.supervisor.draft_model_path()
        return {
            "healthy": self.supervisor.health(),
            "endpoint": self.supervisor.base_url,
            "model": model_path.name,
            "installed": model_path.exists(),
            "backend": self.settings.translation.backend,
            "speculative": draft.name if draft else None,
            "speculative_enabled": self.settings.local_model.speculative,
            # Which cloud service the 「云端」 backend would use, so the header can
            # name the model actually doing the work rather than always naming
            # the local one.
            # 「云端」后端会用哪家服务，以便页首标出真正在干活的那个模型，而不是永远标本地。
            "cloud_provider": self.settings.translation.cloud_provider,
            "cloud_model": self.settings.api_model.model_name,
        }

    # ------------------------------------------------------------------ #
    # Settings / 配置
    # ------------------------------------------------------------------ #

    def update_settings(self, settings: Settings) -> None:
        """
        中文：保存配置并同步到运行中的组件。

        Persist new settings and propagate them to live components.

        Concurrency limits and local-model parameters are applied immediately;
        note that changing the llama-server *launch* parameters (context size,
        parallel slots) only takes effect after the server restarts, since those
        are command-line arguments rather than per-request ones.

        Args:
            settings (Settings): The settings to adopt.
        """
        before = (self.settings.translation.lang_out, self.settings.translation.lang_in)
        self.settings = settings
        save_settings(settings)
        after = (settings.translation.lang_out, settings.translation.lang_in)
        if after != before:
            self._fetch_fonts(*after)
        self.manager.max_concurrent = max(1, settings.server.max_concurrent_jobs)
        self.supervisor.settings = settings.local_model
        self.idle_unloader.seconds = settings.local_model.idle_unload_seconds
        self.idle_unloader.start()

    @staticmethod
    def _fetch_fonts(lang_out: str, lang_in: str) -> None:
        """
        中文：换了语言就把该语言的字体先取下来，在后台进行。

        Fetch the faces a newly chosen language needs, in the background.

        The alternative is fetching them at translation time, which is worse in
        the way that matters: a reader who picks Japanese in the settings and
        starts a translation would sit through a 26 MB download inside the run,
        attributed to nothing in particular. Doing it on the language change
        puts the wait where the decision was made, and the reader is not waiting
        on it — nothing blocks, and a translation started before it finishes
        simply fetches whatever is still missing itself.

        Args:
            lang_out (str): The newly chosen target language.
            lang_in (str): The newly chosen source language.

        Returns:
            None. Failures are logged and left for the run to retry: the network
            being down while someone edits their settings says nothing about
            whether it will be down when they translate.
        """
        import threading

        def fetch() -> None:
            """中文：后台取字体。 / Fetch the faces off the request thread."""
            try:
                from local_translator.engine import fontset

                fontset.ensure(lang_out, lang_in)
            except Exception:  # noqa: BLE001
                logger.info("换语言后预取字体没有成功，留到翻译时再取", exc_info=True)

        threading.Thread(target=fetch, name="font-prefetch", daemon=True).start()

    # ------------------------------------------------------------------ #
    # Submission / 提交翻译
    # ------------------------------------------------------------------ #

    def _glossary_terms(self) -> list[tuple[str, str, str]]:
        """
        中文：从术语库读取启用中的条目，供本地翻译器匹配。

        Load enabled glossary entries for injection into local prompts.

        Returns:
            list[tuple[str, str, str]]: ``(source, target, policy)`` triples.
                Empty when the glossary is disabled or still unpopulated.
        """
        if not self.settings.glossary.enabled:
            return []
        rows = self.db.query(
            "SELECT src, tgt, policy FROM glossary_terms WHERE enabled = 1"
        )
        return [(r["src"], r["tgt"], r["policy"]) for r in rows]

    def _glossary_matcher(self):
        """
        中文：构造当前术语库的匹配器（供覆盖率评估用）。

        Build a matcher over the current glossary.

        Returns:
            GlossaryMatcher | None: ``None`` when the glossary is empty or
                disabled, which the coverage gate reads as zero coverage.
        """
        terms = self._glossary_terms()
        if not terms:
            return None
        from local_translator.glossary.matcher import GlossaryMatcher

        return GlossaryMatcher(terms, max_terms=self.settings.glossary.max_terms_per_prompt)

    def translate_text(self, text: str, lang_in: str | None = None,
                       lang_out: str | None = None) -> dict[str, object]:
        """
        中文：在服务进程内直接翻译一段文本，模型跟随当前后端设置。

        Translate a piece of text in-process using the configured backend.

        Document translation runs in a worker subprocess because BabelDOC is
        CPU-heavy and must be killable. A pasted paragraph is neither: it is one
        HTTP request, so paying for process startup would dominate the work and
        make an interactive box feel slow.

        The glossary applies here exactly as it does to a PDF, so a term reads
        the same whether it was met in a document or pasted into this box.

        Args:
            text (str): Text to translate. Blank lines separate paragraphs, each
                translated on its own so that structure survives.
            lang_in (str | None): Source language; falls back to the setting.
            lang_out (str | None): Target language; falls back to the setting.

        Returns:
            dict[str, object]: ``text`` (the translation), ``backend``,
                ``paragraphs`` translated, ``tokens`` used and ``seconds``
                elapsed.

        Raises:
            ValueError: If the text is empty, or the API backend is selected
                without a key.
        """
        import time

        body = (text or "").strip()
        if not body:
            raise ValueError("没有要翻译的内容")

        engine, backend = self._build_text_engine(lang_in, lang_out)

        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
        started = time.monotonic()
        out = [engine.translate(p) for p in paragraphs]
        elapsed = time.monotonic() - started

        tokens = 0
        if hasattr(engine, "usage_summary"):
            summary = engine.usage_summary()
            tokens = int(summary.get("total_tokens", 0) or 0)

        return {
            "text": "\n\n".join(out),
            "backend": backend,
            "paragraphs": len(paragraphs),
            "tokens": tokens,
            "seconds": round(elapsed, 1),
        }

    def _build_text_engine(self, lang_in: str | None, lang_out: str | None):
        """
        中文：为文本/图片翻译构造翻译引擎，返回（引擎，实际后端）。

        Build a translation engine for free-text and image translation.

        Shared by :meth:`translate_text` and :meth:`translate_image` so both
        honour the same backend selection, glossary injection and sampling
        settings. Hybrid collapses to the API side: hybrid overlaps two backends
        across a whole document, and a handful of lines cannot fill both, so
        routing some to the slower side would only hurt latency.

        Args:
            lang_in (str | None): Source language; falls back to the setting.
            lang_out (str | None): Target language; falls back to the setting.

        Returns:
            tuple: ``(engine, backend)`` with the backend actually used.

        Raises:
            ValueError: If the API backend is selected without a key.
            LlamaServerError: If the local model cannot be started.
        """
        from local_translator.engine.translators import ApiTranslator, LocalMTTranslator

        tr = self.settings.translation
        lang_in = lang_in or tr.lang_in
        lang_out = lang_out or tr.lang_out
        backend = tr.backend

        def local_engine():
            engine = LocalMTTranslator(
                lang_in=lang_in, lang_out=lang_out, settings=self.settings.local_model,
                ignore_cache=tr.ignore_cache,
                glossary_rev=self.settings.glossary.glossary_rev,
                custom_hint=tr.custom_system_prompt,
            )
            matcher = self._glossary_matcher()
            if matcher is not None:
                engine.set_glossary_provider(matcher)
            return engine

        def api_engine():
            from local_translator.secrets import get_api_key

            api = self.settings.api_model
            key = get_api_key(api.keyring_account)
            if not key:
                raise ValueError(f"没有找到 {api.display_name} 的 API key")
            return ApiTranslator(
                lang_in=lang_in, lang_out=lang_out, model=api.model_name,
                base_url=api.base_url or None, api_key=key,
                ignore_cache=tr.ignore_cache, skip_reasoning=api.skip_reasoning,
                prompt_template=api.chat_prompt_template,
            )

        if backend == "api":
            return api_engine(), "api"
        if backend == "hybrid":
            return api_engine(), "api"
        self.ensure_model_ready()
        return local_engine(), "local"

    def translate_image(self, image: bytes | Path, lang_in: str | None = None,
                        lang_out: str | None = None) -> dict[str, object]:
        """
        中文：识别图片中的文字并翻译，返回逐行对照与整段译文。

        Recognise the text in an image and translate it.

        Recognition is local. The configured API model is text-only — it rejects
        an image request outright — and running OCR here rather than remotely
        also means the picture itself never leaves the machine, which matters
        when the image is a page of something unpublished.

        Each recognised line is translated on its own so the translation can be
        drawn back over the box it came from, producing an overlay image: the
        original picture with Chinese in place of the source language. A line is
        short and often a fragment, so a per-line translation reads a little
        worse than one over the joined text — but placement is what a figure or
        screenshot needs, and a line pulled out into a list loses the spatial
        meaning the layout carried. A joined translation is returned as well for
        reading, and the recognised line beside its confidence still lets a
        reader tell an OCR misread from a mistranslation.

        Args:
            image (bytes | Path): Image data or a path to an image file.
            lang_in (str | None): Source language; falls back to the setting.
            lang_out (str | None): Target language; falls back to the setting.

        Returns:
            dict[str, object]: ``text`` (joined translation), ``source``
                (recognised text), ``lines`` (per-line
                ``{text, translation, confidence, box}``), ``overlay`` (a base64
                PNG of the in-place translation, or ``None``), ``backend``,
                ``tokens``, ``ocr_seconds``, ``seconds``, ``width`` and
                ``height``.

        Raises:
            FileNotFoundError: If the OCR models are not installed.
            ValueError: If nothing could be recognised.
        """
        import base64
        import time

        from local_translator.ocr import get_ocr
        from local_translator.ocr.overlay import render_overlay

        ocr = get_ocr()
        if not ocr.available():
            raise FileNotFoundError("OCR 模型尚未安装")

        image_bytes = image if isinstance(image, bytes) else Path(image).read_bytes()
        started = time.monotonic()

        engine, backend = self._build_text_engine(lang_in, lang_out)

        found = recognise_and_translate(image_bytes, engine.translate)
        if found is None:
            raise ValueError("图片里没有识别到文字")
        result, cache, pairs = found
        source_text = result.as_text()

        elapsed = time.monotonic() - started

        tokens = 0
        if hasattr(engine, "usage_summary"):
            tokens = int(engine.usage_summary().get("total_tokens", 0) or 0)

        # The reading translation is the per-line translations reassembled into
        # paragraphs, following the same line grouping as the recognised text,
        # so no extra model call is spent on a second joined translation.
        # 阅读用译文由逐行译文按识别文本相同的分段规则重组而成，不再为整段译文多花
        # 一次模型调用。
        joined = self._reflow_translation(result.lines, cache)

        overlay = None
        try:
            png = render_overlay(image_bytes, pairs)
            if png:
                overlay = base64.b64encode(png).decode("ascii")
        except Exception:  # noqa: BLE001
            logger.warning("Overlay rendering failed", exc_info=True)

        return {
            "text": joined,
            "source": source_text,
            "lines": pairs,
            "overlay": overlay,
            "backend": backend,
            "tokens": tokens,
            "ocr_seconds": result.seconds,
            "seconds": round(elapsed, 1),
            "width": result.width,
            "height": result.height,
        }

    @staticmethod
    def _reflow_translation(lines, cache: dict[str, str]) -> str:
        """
        中文：把逐行译文按与识别文本相同的分段规则重组成阅读文本。

        Reassemble per-line translations into reading paragraphs.

        Uses the same vertical-gap paragraph rule as
        :meth:`OcrResult.as_text`, so the reading translation breaks into
        paragraphs where the recognised text did.

        Args:
            lines (list[TextLine]): Recognised lines in reading order.
            cache (dict[str, str]): Map from a line's recognised text to its
                translation.

        Returns:
            str: Translation with single newlines inside a paragraph and blank
                lines between paragraphs.
        """
        if not lines:
            return ""
        chunks: list[list[str]] = [[cache.get(lines[0].text.strip(), "")]]
        for previous, line in zip(lines, lines[1:], strict=False):
            reference = max(previous.height, line.height, 1.0)
            piece = cache.get(line.text.strip(), "")
            if (line.top - previous.top) > 1.6 * reference:
                chunks.append([piece])
            else:
                chunks[-1].append(piece)
        return "\n\n".join(" ".join(part for part in chunk if part) for chunk in chunks)

    def assess_coverage(self, source: Path) -> dict[str, object]:
        """
        中文：评估一篇文档被现有术语库覆盖的程度。

        Measure existing glossary coverage for a document.

        Args:
            source (Path): Path to the source PDF.

        Returns:
            dict[str, object]: As returned by
                :func:`local_translator.glossary.coverage.assess`.
        """
        from local_translator.glossary import coverage

        return coverage.assess(source, self._glossary_matcher())

    def _should_extract_terms(self, source: Path) -> tuple[bool, dict[str, object]]:
        """
        中文：判断这篇文档是否值得动用 API 提取术语。

        Decide whether this document warrants an API term-extraction call.

        Three conditions must all hold: extraction is enabled, an API key is
        actually available, and the existing glossary does not already cover the
        document's terminology. The last one is what makes the loop converge —
        every absorbed term raises coverage for the next paper, so the API is
        consulted progressively less often rather than once per document forever.

        Args:
            source (Path): Path to the source PDF.

        Returns:
            tuple[bool, dict[str, object]]: Whether to extract, and the coverage
                report behind the decision.
        """
        settings = self.settings.glossary
        if not settings.enabled or not settings.auto_extract:
            return False, {"coverage": None, "reason": "disabled"}

        if not get_api_key(self.settings.api_model.keyring_account):
            return False, {"coverage": None, "reason": "no_api_key"}

        report = self.assess_coverage(source)
        threshold = settings.coverage_threshold
        should = float(report["coverage"]) < threshold

        logger.info(
            "Coverage %.0f%% vs threshold %.0f%% for %s -> %s",
            float(report["coverage"]) * 100,
            threshold * 100,
            source.name,
            "extracting terms via API" if should else "skipping API extraction",
        )
        report["reason"] = "below_threshold" if should else "covered"
        return should, report

    def submit_path(
        self,
        source: Path,
        *,
        source_kind: str = SOURCE_LOCAL,
        title: str | None = None,
        source_ref: str | None = None,
        authors: str | None = None,
        pages: str | None = None,
        copy_index: int | None = None,
        latex_source: Path | None = None,
    ) -> str:
        """
        中文：为一个本地 PDF 建档并提交翻译任务。

        Register a PDF and enqueue a translation job for it.

        Args:
            source (Path): Path to the source PDF.
            source_kind (str): ``"local"``, ``"upload"`` or ``"arxiv"``; decides
                whether outputs land beside the source or in the library.
            title (str | None): Known title; falls back to PDF metadata.
            source_ref (str | None): Original reference such as an arXiv URL.
            authors (str | None): Comma-separated authors.
            pages (str | None): BabelDOC page range expression.
            copy_index (int | None): Register as numbered copy ``N`` of a
                document already in the library instead of reusing its entry.
            latex_source (Path | None): An unpacked LaTeX source tree to
                translate instead of the PDF. ``source`` still identifies the
                document, so the library entry, the title and the output
                directory are decided the same way whichever path runs.

        Returns:
            str: The new job's identifier.

        Raises:
            FileNotFoundError: If ``source`` does not exist.
        """
        source = Path(source).expanduser().resolve()
        doc_id, output_dir = register_document(
            self.db,
            source,
            source_kind,
            self.paths.library_dir,
            title=title,
            source_ref=source_ref,
            authors=authors,
            copy_index=copy_index,
        )

        # The key is needed both for API-backed translation and for term
        # extraction, so it is fetched whenever either could use it.
        # 无论是 API 翻译还是术语提取都需要密钥，两者任一可能用到就取。
        api_key = get_api_key(self.settings.api_model.keyring_account)
        needs_key = (
            self.settings.translation.backend in ("api", "hybrid")
            or self.settings.glossary.auto_extract
        )
        if not needs_key:
            api_key = None

        extract_terms, _ = self._should_extract_terms(source)

        return self.manager.submit(
            doc_id=doc_id,
            input_file=source,
            output_dir=output_dir,
            settings=self.settings,
            api_key=api_key,
            glossary_terms=self._glossary_terms(),
            pages=pages,
            doc_title=title or source.stem,
            extract_terms=extract_terms,
            latex_source=latex_source,
        )

    # ------------------------------------------------------------------ #
    # Duplicate handling / 重复提交处理
    # ------------------------------------------------------------------ #

    def duplicates_of(self, title: str) -> list:
        """
        中文：查库里是否已有同标题文档（含其副本）。

        Find library entries that already carry this title.

        Args:
            title (str): Title to look for; a copy suffix on it is ignored.

        Returns:
            list: Matching document rows, oldest first.
        """
        stripped = base_title(title)
        return self.db.documents_titled(stripped) if stripped else []

    def submit_pending(self, pending: PendingSubmission, *, as_copy: bool = False) -> str:
        """
        中文：提交一次已准备好的任务；标题重复且未确认时抛出 DuplicateDocument。

        Complete a prepared submission, asking about duplicates first.

        The check is on the title rather than on the file's content hash,
        because the two submissions a reader actually wants to distinguish are
        the same paper fetched twice — once uploaded, once from arXiv — and
        those are two different files with one title.

        Args:
            pending (PendingSubmission): The prepared submission.
            as_copy (bool): Go ahead even though the title exists, registering
                the run as a numbered copy.

        Returns:
            str: The new job's identifier.

        Raises:
            DuplicateDocument: When the title is already in the library and
                ``as_copy`` is false. The exception carries what is needed to
                retry with ``as_copy=True``.
        """
        existing = self.duplicates_of(pending.title)
        copy_index = next_copy_index(existing) if existing else None

        if existing and not as_copy:
            raise DuplicateDocument(pending, existing, copy_index or 1)

        return self.submit_path(
            pending.source,
            source_kind=pending.source_kind,
            title=pending.title,
            source_ref=pending.source_ref,
            authors=pending.authors,
            pages=pending.pages,
            copy_index=copy_index if as_copy else None,
            latex_source=pending.latex_source,
        )

    def submit_folder(
        self,
        folder: Path,
        *,
        recursive: bool = True,
        skip_translated: bool = True,
        pages: str | None = None,
    ) -> dict[str, Any]:
        """
        中文：批量提交一个文件夹里的所有 PDF。

        Enqueue every PDF in a folder.

        Args:
            folder (Path): Directory to scan.
            recursive (bool): Descend into subdirectories.
            skip_translated (bool): Skip PDFs that already have translations
                next to them, and skip translation outputs themselves.
            pages (str | None): Page range applied to every document.

        Returns:
            dict[str, Any]: Keys ``job_ids`` (list[str]), ``skipped`` (int) and
                ``total_found`` (int).

        Raises:
            NotADirectoryError: If ``folder`` is not a directory.
        """
        folder = Path(folder).expanduser().resolve()
        if not folder.is_dir():
            raise NotADirectoryError(folder)

        pattern = "**/*" if recursive else "*"
        candidates = sorted(
            p
            for p in folder.glob(pattern)
            if p.is_file() and p.suffix.lower() in PDF_SUFFIXES
        )

        lang_out = self.settings.translation.lang_out
        job_ids: list[str] = []
        skipped = 0

        for pdf in candidates:
            # A previous run's own outputs live in the same folder; translating
            # a translation would be both wasteful and confusing.
            # 上一轮的产物就在同一目录，翻译译文既浪费又混乱。
            if _is_translation_output(pdf, lang_out):
                skipped += 1
                continue
            if skip_translated and already_translated(pdf, pdf.parent, lang_out):
                skipped += 1
                continue
            try:
                job_ids.append(self.submit_path(pdf, source_kind=SOURCE_LOCAL, pages=pages))
            except Exception:
                logger.warning("Could not enqueue %s", pdf, exc_info=True)
                skipped += 1

        logger.info(
            "Folder import from %s: %d queued, %d skipped", folder, len(job_ids), skipped
        )
        return {"job_ids": job_ids, "skipped": skipped, "total_found": len(candidates)}

    def submit_upload(self, filename: str, content: bytes, pages: str | None = None,
                      *, as_copy: bool = False) -> str:
        """
        中文：保存上传的 PDF 到文库并提交翻译；标题重复时抛出 DuplicateDocument。

        Store an uploaded PDF in the library and enqueue it.

        The title is taken from the PDF's own metadata where it has one, because
        a browser filename is often ``2608.13040v1.pdf`` and would never match
        the same paper submitted from arXiv.

        Args:
            filename (str): Original filename supplied by the browser.
            content (bytes): Raw PDF bytes.
            pages (str | None): Page range expression.
            as_copy (bool): Proceed despite a duplicate title, as a copy.

        Returns:
            str: The new job's identifier.

        Raises:
            DuplicateDocument: When the title is already in the library and
                ``as_copy`` is false.
        """
        safe_name = Path(filename).name or "upload.pdf"
        inbox = self.paths.library_dir / "_inbox"
        inbox.mkdir(parents=True, exist_ok=True)

        from local_translator.library import read_pdf_metadata, unique_path

        staged = unique_path(inbox / safe_name)
        staged.write_bytes(content)

        title = str(read_pdf_metadata(staged).get("title") or "").strip() or staged.stem
        return self.submit_pending(
            PendingSubmission(source=staged, source_kind=SOURCE_UPLOAD,
                              title=title, pages=pages),
            as_copy=as_copy,
        )

    async def submit_arxiv(self, reference: str, pages: str | None = None,
                           *, as_copy: bool = False) -> str:
        """
        中文：解析 arXiv 链接或编号，下载 PDF 并提交翻译。

        Resolve an arXiv reference, download the paper and enqueue it.

        Args:
            reference (str): An arXiv id (``2510.04704``), an abs/pdf URL, or a
                versioned id.
            pages (str | None): Page range expression.

            as_copy (bool): Proceed despite a duplicate title, as a copy.

        Returns:
            str: The new job's identifier.

        Raises:
            ValueError: If the reference cannot be parsed as an arXiv paper.
            DuplicateDocument: When the title is already in the library and
                ``as_copy`` is false. The PDF has been downloaded by then, so
                confirming does not fetch it again.
        """
        from local_translator.sources.arxiv import fetch_arxiv

        paper = await fetch_arxiv(reference, self.paths.library_dir)
        return self.submit_pending(
            PendingSubmission(
                source=paper.pdf_path,
                source_kind=SOURCE_ARXIV,
                title=paper.title,
                source_ref=paper.abs_url,
                authors=paper.authors,
                pages=pages,
                latex_source=await self._arxiv_latex_source(paper.arxiv_id),
            ),
            as_copy=as_copy,
        )

    def latex_available(self) -> tuple[bool, str]:
        """
        中文：本机能不能做 LaTeX 源码翻译，以及用的是哪个引擎。

        Report whether source translation is possible here, and with what.

        Returns:
            tuple[bool, str]: Whether an engine was found, and a sentence
                naming it — or saying what is missing — for the settings page.
        """
        from local_translator.latex import engine as tex_engine

        configured = self.settings.latex.engine_path or None
        found = tex_engine.resolve(configured, self.paths.bin_dir)
        return found is not None, tex_engine.describe(configured, self.paths.bin_dir)

    async def _arxiv_latex_source(self, arxiv_id: str) -> Path | None:
        """
        中文：按设置尝试取回 arXiv 的 LaTeX 源码；取不到就回到 PDF 那条路。

        Fetch a paper's LaTeX source when the settings ask for it.

        Args:
            arxiv_id (str): Identifier without a version suffix.

        Returns:
            Path | None: The unpacked source tree, or ``None`` — which means
                translate the PDF instead. Every reason to give up returns
                ``None`` rather than raising: arXiv has no source for some
                papers, the archive is occasionally something this cannot
                unpack, and a paper that cannot be translated from source is
                still a paper that can be translated.
                每一种放弃的理由都返回 None 而不是抛异常：arXiv 上有些论文没有源码，压缩
                包偶尔是这里解不开的格式，而一篇不能从源码翻译的论文仍然是一篇能翻译的
                论文。
        """
        if not self.settings.latex.prefer_for_arxiv:
            return None
        available, _ = self.latex_available()
        if not available:
            logger.info("设置为 arXiv 优先用源码翻译，但本机没有 TeX 引擎，改用 PDF")
            return None

        from local_translator.latex import source as tex_source

        workspace = self.paths.library_dir / "_latex" / arxiv_id
        try:
            workspace.mkdir(parents=True, exist_ok=True)
            archive = await tex_source.download(arxiv_id, workspace)
            root, _ = tex_source.prepare_workspace(archive, workspace / "source")
        except Exception as problem:  # noqa: BLE001
            logger.info("arXiv %s 没有可用的 LaTeX 源码（%s），改用 PDF",
                        arxiv_id, problem)
            return None
        logger.info("arXiv %s 将从 LaTeX 源码翻译", arxiv_id)
        return root

    async def submit_arxiv_batch(self, references, pages: str | None = None,
                                 *, as_copy: bool = True) -> dict[str, Any]:
        """
        中文：一次提交多篇 arXiv 论文，逐篇按当前设置决定走源码还是走 PDF。

        Enqueue several arXiv papers in one go.

        Args:
            references (Iterable[str]): arXiv ids or URLs, one per entry.
            pages (str | None): Page range applied to every paper. Ignored by
                the source path, which translates the whole document.
            as_copy (bool): Go ahead with papers the library already holds,
                registering each as a numbered copy. Default here rather than
                in the single-paper path: a batch that stops halfway to ask a
                question is worse than a batch that makes copies.
                这里默认为真，而单篇提交默认为假：一批任务跑到一半停下来问问题，比多出
                几个副本更糟。

        Returns:
            dict[str, Any]: ``job_ids``, ``failed`` as ``(reference, reason)``
                pairs, and ``total``. A paper that could not be fetched does
                not stop the rest — the batch reports it and carries on.

        Note:
            Each paper goes through :meth:`submit_arxiv`, so the settings decide
            per paper: source translation when the setting asks for it and the
            source exists, the PDF otherwise. The batch makes no choice of its
            own.
            每一篇都走 submit_arxiv，所以是设置逐篇决定：设置要求且有源码就走源码，否则
            走 PDF。批量本身不做任何选择。
        """
        job_ids: list[str] = []
        failed: list[tuple[str, str]] = []
        wanted = [str(item).strip() for item in references if str(item).strip()]

        for reference in wanted:
            try:
                job_ids.append(await self.submit_arxiv(reference, pages=pages,
                                                       as_copy=as_copy))
            except Exception as problem:  # noqa: BLE001
                logger.warning("批量提交失败：%s", reference, exc_info=True)
                failed.append((reference, str(problem)))
        logger.info("arXiv 批量：%d 篇入队，%d 篇失败", len(job_ids), len(failed))
        return {"job_ids": job_ids, "failed": failed, "total": len(wanted)}

    def submit_latex_folder(self, folder: Path, *, title: str | None = None,
                            as_copy: bool = False) -> str:
        """
        中文：提交一个本地 LaTeX 源码文件夹。

        Enqueue a LaTeX source tree already on disk.

        Args:
            folder (Path): Directory holding the sources.
            title (str | None): Display title; defaults to the folder's name.
            as_copy (bool): Proceed despite a duplicate title, as a copy.

        Returns:
            str: The new job's identifier.

        Raises:
            NotADirectoryError: If ``folder`` is not a directory.
            ValueError: If nothing in it declares a document class.
            DuplicateDocument: When the title is already in the library and
                ``as_copy`` is false.

        Note:
            The sources are copied into the library before anything touches
            them. Translation writes the translated ``.tex``, the compiled PDF
            and — when figures are being translated — a Chinese copy of each
            image, all beside the originals; doing that inside a reader's own
            folder would be helping myself to their directory.
            源码会先复制进文库再动手。翻译会在原件旁边写出译文 .tex、编译好的 PDF，以及
            （开了图片翻译时）每张插图的中文版——在读者自己的目录里这么做，等于擅自动了
            他们的文件夹。
        """
        folder = Path(folder).expanduser().resolve()
        if not folder.is_dir():
            raise NotADirectoryError(folder)

        import shutil

        from local_translator.latex import source as tex_source
        from local_translator.library import unique_path

        if tex_source.find_main(folder) is None:
            raise ValueError(f"{folder.name} 里没有找到带 \\documentclass 的主文件")

        name = title or folder.name
        workspace = unique_path(self.paths.library_dir / "_latex" / folder.name)
        workspace.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(folder, workspace)
        return self._submit_latex(workspace, name, as_copy=as_copy)

    def submit_latex_archive(self, filename: str, content: bytes,
                             *, as_copy: bool = False) -> str:
        """
        中文：提交一个上传的 LaTeX 源码压缩包。

        Enqueue an uploaded archive of LaTeX sources.

        Args:
            filename (str): Original filename supplied by the browser.
            content (bytes): The archive's bytes — ``.zip``, ``.tar.gz`` or
                ``.tar``, which is what arXiv hands out and what a reader is
                therefore most likely to have.
            as_copy (bool): Proceed despite a duplicate title, as a copy.

        Returns:
            str: The new job's identifier.

        Raises:
            ValueError: If the archive cannot be unpacked or holds no main file.
            DuplicateDocument: When the title is already in the library and
                ``as_copy`` is false.
        """
        from local_translator.latex import source as tex_source
        from local_translator.library import unique_path

        safe_name = Path(filename).name or "sources.zip"
        stem = safe_name
        for suffix in (".tar.gz", ".tgz", ".tar", ".zip", ".gz"):
            if stem.lower().endswith(suffix):
                stem = stem[: -len(suffix)]
                break

        staging = self.paths.library_dir / "_latex" / "_inbox"
        staging.mkdir(parents=True, exist_ok=True)
        archive = unique_path(staging / safe_name)
        archive.write_bytes(content)

        workspace = unique_path(self.paths.library_dir / "_latex" / stem)
        try:
            root, _ = tex_source.prepare_workspace(archive, workspace)
        except Exception as problem:  # noqa: BLE001
            raise ValueError(f"这个压缩包里没有可用的 LaTeX 源码：{problem}") from problem
        return self._submit_latex(root, stem, as_copy=as_copy)

    def _submit_latex(self, root: Path, title: str, *, as_copy: bool) -> str:
        """
        中文：为一棵源码树建档并提交。

        Register a source tree and enqueue it.

        Args:
            root (Path): The unpacked sources, already inside the library.
            title (str): Display title.
            as_copy (bool): Proceed despite a duplicate title.

        Returns:
            str: The new job's identifier.

        Note:
            A document row wants a file to point at, and there is no PDF here
            yet — the paper has not been compiled. The main ``.tex`` stands in:
            it is a real file, it is the thing being translated, and it names
            the document as well as a PDF would.
            文档表需要一个文件作为指向，而这里还没有 PDF——论文还没编译。主 .tex 顶上：
            它是一个真实存在的文件，是被翻译的那个东西，用来标识文档不比 PDF 差。
        """
        from local_translator.latex import source as tex_source

        main = tex_source.find_main(root)
        if main is None:
            raise ValueError(f"{root.name} 里没有找到带 \\documentclass 的主文件")
        return self.submit_pending(
            PendingSubmission(source=main, source_kind=SOURCE_UPLOAD,
                              title=title, latex_source=root),
            as_copy=as_copy,
        )

    # ------------------------------------------------------------------ #
    # Queries / 查询
    # ------------------------------------------------------------------ #

    def dashboard(self) -> dict[str, Any]:
        """
        中文：返回首页需要的全部状态（队列 + 最近任务 + 模型状态）。

        Gather everything the main page renders, in one call.

        Returning a single consistent snapshot keeps the UI from interleaving
        reads and showing, say, a job as both running and finished.

        Returns:
            dict[str, Any]: Keys ``running``, ``queued``, ``capacity`` (from the
                scheduler), ``recent`` (recent job rows as dicts) and ``model``.
        """
        snapshot = self.manager.snapshot()
        live = {r["job_id"]: r for r in snapshot["running"]}

        recent = []
        for row in self.db.list_jobs(limit=30):
            item = dict(row)
            # The database lags the scheduler by up to one progress tick; prefer
            # the in-memory value so the bar never appears to stall or rewind.
            # 数据库比调度器最多滞后一个进度周期，优先用内存值避免进度条卡顿或回退。
            if row["id"] in live:
                item.update(
                    progress=live[row["id"]]["progress"],
                    stage=live[row["id"]]["stage"],
                    stage_current=live[row["id"]]["stage_current"],
                    stage_total=live[row["id"]]["stage_total"],
                    cancelling=live[row["id"]]["cancelling"],
                )
            recent.append(item)

        return {
            "running": snapshot["running"],
            "queued": snapshot["queued"],
            "capacity": snapshot["capacity"],
            "recent": recent,
            "model": self.model_status(),
        }

    def documents(self, limit: int = 200) -> list[dict[str, Any]]:
        """
        中文：列出文档库中的所有文档。

        List library documents, most recently updated first.

        Args:
            limit (int): Maximum rows to return.

        Returns:
            list[dict[str, Any]]: Document rows with their latest job summary.
        """
        return [dict(r) for r in self.db.list_documents(limit=limit)]

    def document_files(self, doc_ids) -> list[Path]:
        """
        中文：列出这些文档在磁盘上还存在的产物文件。

        List the output files these documents still have on disk.

        Only the produced files — the bilingual PDF, the mono PDF, the mined
        glossary. The *source* is never included: a reader dropped that file in
        themselves and it is theirs, not this program's to remove.

        Args:
            doc_ids (Iterable[str]): Document identifiers.

        Returns:
            list[Path]: Existing files, de-duplicated, in a stable order.
        """
        wanted = {str(i) for i in doc_ids}
        found: list[Path] = []
        for row in self.db.list_documents(limit=100000):
            item = dict(row)
            if item["id"] not in wanted:
                continue
            for key in ("dual_path", "mono_path", "glossary_path"):
                value = item.get(key)
                if not value:
                    continue
                path = Path(value)
                if path.is_file() and path not in found:
                    found.append(path)
        return found

    def delete_documents(self, doc_ids, *, remove_files: bool = False) -> dict[str, int]:
        """
        中文：从历史里删掉这些文档，可选连同产物文件一起删。

        Remove documents from the library, optionally with their outputs.

        The two are separate on purpose. A library entry is a record; the PDF it
        points at may be the only copy of an hour's translation. Removing the
        record is cheap and reversible by translating again; removing the file
        is not, so the caller has to ask for it explicitly.

        Files go to the trash rather than being unlinked, on the same reasoning.

        Args:
            doc_ids (Iterable[str]): Document identifiers.
            remove_files (bool): Also remove the produced files.

        Returns:
            dict[str, int]: ``documents`` removed, ``files`` removed, and
                ``failed`` files that could not be removed.
        """
        ids = [str(i) for i in doc_ids]
        files = self.document_files(ids) if remove_files else []

        removed = failed = 0
        for path in files:
            try:
                _trash(path)
                removed += 1
            except OSError:
                logger.warning("删不掉 %s", path, exc_info=True)
                failed += 1

        count = self.db.delete_documents(ids)
        logger.info("从历史删除 %d 个文档，产物文件 %d 个（%d 个失败）",
                    count, removed, failed)
        return {"documents": count, "files": removed, "failed": failed}

    def job(self, job_id: str) -> dict[str, Any] | None:
        """
        中文：查询单个任务的完整状态。

        Args:
            job_id (str): Job identifier.

        Returns:
            dict[str, Any] | None: Job row merged with live progress, or ``None``.
        """
        row = self.db.get_job(job_id)
        if row is None:
            return None
        item = dict(row)
        for running in self.manager.snapshot()["running"]:
            if running["job_id"] == job_id:
                item.update(
                    progress=running["progress"],
                    stage=running["stage"],
                    stage_current=running["stage_current"],
                    stage_total=running["stage_total"],
                )
        return item

    async def cancel(self, job_id: str) -> bool:
        """
        中文：取消一个任务。

        Args:
            job_id (str): Job to cancel.

        Returns:
            bool: ``True`` if a queued or running job was cancelled.
        """
        return await self.manager.cancel(job_id)

    def retry(self, job_id: str) -> str | None:
        """
        中文：重试一个失败或被中断的任务。

        Re-enqueue a finished job's document.

        Retrying is cheap even for a long document: BabelDOC caches translations
        per paragraph, so a job interrupted near the end mostly replays from
        cache rather than re-running inference.

        Args:
            job_id (str): The job to retry.

        Returns:
            str | None: The new job's id, or ``None`` if the original job or its
                source file no longer exists.
        """
        row = self.db.get_job(job_id)
        if row is None:
            return None
        source = Path(row["source_path"])
        if not source.exists():
            logger.warning("Cannot retry job %s: %s is gone", job_id, source)
            return None
        return self.submit_path(source, source_kind=row["source_kind"], title=row["title"])


def _is_translation_output(path: Path, lang_out: str) -> bool:
    """
    中文：判断某个 PDF 是否是本工具生成的译文产物。

    Detect whether a PDF is one of this tool's own outputs.

    BabelDOC names results ``<stem>.<lang>.dual.pdf`` / ``<stem>.<lang>.mono.pdf``,
    optionally with a ``.vN`` version segment added by the no-clobber rule.

    Args:
        path (Path): PDF path to test.
        lang_out (str): Target language code currently configured.

    Returns:
        bool: ``True`` if the filename matches a translation output pattern.
    """
    parts = path.name.lower().split(".")
    return lang_out.lower() in parts and ("dual" in parts or "mono" in parts)


_service: TranslatorService | None = None


def get_service() -> TranslatorService:
    """
    中文：返回进程内共享的服务单例。

    Return the process-wide service instance, creating it on first use.

    Returns:
        TranslatorService: Shared service facade.
    """
    global _service
    if _service is None:
        _service = TranslatorService()
    return _service


__all__ = ["LlamaServerError", "TranslatorService", "get_service"]


def _install_bundled_glossary() -> None:
    """
    中文：把随源码发布的术语库导入本机；已经导入过就什么都不做。

    Load the glossary that ships with the source into this installation.

    This runs on every start because it is a file read that takes tens of
    milliseconds and skips rows that already exist. It replaces a first-run
    wizard step that contacted two hosts, took twenty-five minutes on the run it
    was measured on, and began without anyone pressing anything — which was the
    real complaint, and the reason the step is gone rather than reworded.

    A failure is logged and swallowed. A missing or damaged snapshot must not
    stop the application from starting: translation works without a glossary,
    just less well.

    Returns:
        None.
    """
    try:
        from local_translator.db import get_db
        from local_translator.glossary.seed import seed
        from local_translator.glossary.store import GlossaryStore

        report = seed(GlossaryStore(get_db()))
        added = sum(r.get("inserted", 0) for r in report.values())
        if added:
            logger.info("Installed %d terms from the bundled glossary", added)
    except Exception:  # noqa: BLE001 - starting up matters more than the glossary
        logger.warning("Could not install the bundled glossary", exc_info=True)
