"""
中文：翻译任务的子进程执行体。

Child-process entry point for a single translation job.

Every job runs in its own ``multiprocessing`` process rather than in the web
server's event loop, for three reasons:

1. BabelDOC is CPU-bound — ONNX layout inference plus its own worker pools —
   and would stall the event loop, freezing the UI for every other client.
2. A process can actually be killed. Cancelling a coroutine mid-``await`` inside
   a third-party library is not reliable; terminating a process is.
3. A crash in PDF parsing takes down the worker, not the server, so the job list
   and history survive.

Progress flows back to the parent over a :class:`multiprocessing.Queue` as plain
dictionaries. Nothing here writes to the database: the parent owns all writes,
which keeps SQLite free of cross-process write contention.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # Annotations only; the runtime imports stay function-local.
    from local_translator.config import LocalModelSettings

logger = logging.getLogger(__name__)

# Event type constants exchanged with the parent / 与父进程交换的事件类型
EV_STARTED = "started"
EV_PROGRESS = "progress"
EV_FINISH = "finish"
EV_ERROR = "error"


@dataclass
class JobSpec:
    """
    中文：传给子进程的任务描述，必须可 pickle。

    Fully self-contained description of one translation job.

    Every field is a plain picklable value because ``spawn`` start method (the
    default and only safe option on macOS) serialises this across the process
    boundary. In particular the settings are passed as a dict rather than a
    pydantic model instance.

    Attributes:
        job_id (str): Job identifier, echoed back in every event.
        doc_id (str): Document identifier.
        input_file (str): Absolute path to the source PDF.
        output_dir (str): Directory that receives the translated PDFs.
        working_dir (str): Directory for BabelDOC intermediates. Setting this
            also makes BabelDOC emit ``translate_tracking.json``, which is the
            source of paragraph-level bilingual text for the reader.
        settings (dict): ``Settings.model_dump()`` snapshot.
        api_key (str | None): API key for the cloud backend, read from the
            keychain by the parent and passed in memory only — never persisted.
        glossary_csv (str | None): Path to a glossary CSV to load, if any.
        glossary_terms (list): Serialised glossary entries for the local
            translator, as ``(source, target, policy)`` triples.
        pages (str | None): BabelDOC page range expression, e.g. ``"1-5"``.
        extract_terms (bool): Run BabelDOC's automatic term extraction using the
            API model, and emit the result as a CSV for the parent to fold back
            into the glossary. Decided by the coverage gate in the service, not
            here, so the worker never makes a network-cost decision on its own.
        latex_source (str): Directory holding an unpacked LaTeX source tree.
            When set, the job translates the paper from its source and every
            PDF-specific stage below is skipped — there is no page to lay out,
            because TeX lays it out.
    """

    job_id: str
    doc_id: str
    input_file: str
    output_dir: str
    working_dir: str
    settings: dict[str, Any]
    api_key: str | None = None
    glossary_csv: str | None = None
    glossary_terms: list[tuple[str, str, str]] = field(default_factory=list)
    pages: str | None = None
    extract_terms: bool = False
    latex_source: str = ""


def _watch_cancel(cancel_event, config, poll_interval: float = 0.5) -> None:
    """
    中文：后台线程，监听取消信号并请求 BabelDOC 优雅停止。

    Watch for a cancellation request and ask BabelDOC to stop gracefully.

    BabelDOC exposes ``TranslationConfig.cancel_translation()``, which sets an
    internal threading event that its stages check via ``raise_if_cancelled()``.
    Driving that from a watcher thread lets an in-flight job unwind cleanly —
    finishing the current paragraph, releasing file handles — instead of being
    killed mid-write. The parent still terminates the process as a backstop if
    this does not take effect in time.

    Args:
        cancel_event (multiprocessing.Event): Set by the parent to request a stop.
        config (TranslationConfig): Config whose translation should be cancelled.
        poll_interval (float): Seconds between checks.
    """
    while True:
        if cancel_event.wait(timeout=poll_interval):
            logger.info("Cancellation requested; asking BabelDOC to stop")
            try:
                config.cancel_translation()
            except Exception:
                logger.warning("cancel_translation() failed", exc_info=True)
            return


# Requests a second Tencent Cloud's TMT allows by default. Documented per
# account and raisable in the console; five is what a new account gets.
# 腾讯云 TMT 默认允许的每秒请求数。该值按账号计、可在控制台申请提升，新账号是五。
_TMT_QPS = 5


def _build_translator(spec: JobSpec, glossary_matcher):
    """
    中文：构造翻译器，并把「标题是否送模型」的设置带上。

    Build the translator and carry the heading setting onto it.

    The setting lives in ``PdfSettings`` and the decision is made inside
    :class:`TranslationGuards`, which has no access to the job. Rather than
    thread it through four construction branches, it is attached here, once,
    to whichever translator those branches returned.

    Args:
        spec (JobSpec): Job description carrying settings and credentials.
        glossary_matcher: Glossary matcher, or ``None``.

    Returns:
        BaseTranslator: The translator for this job.
    """
    from local_translator.config import Settings

    translator = _build_translator_inner(spec, glossary_matcher)
    try:
        translator.translate_headings = (
            Settings.model_validate(spec.settings).pdf.translate_headings)
    except Exception:  # noqa: BLE001
        logger.debug("Could not read the heading setting; using the default",
                     exc_info=True)
    return translator


def _build_translator_inner(spec: JobSpec, glossary_matcher):
    """
    中文：根据后端设置构造翻译器，并接入术语匹配。

    Instantiate the translator backend selected in the job's settings.

    Args:
        spec (JobSpec): Job description carrying settings and credentials.
        glossary_matcher (Callable | None): Callable mapping paragraph text to
            ``(translate_pairs, keep_terms)``, attached to the local translator.

    Returns:
        BaseTranslator: A ready-to-use translator instance.

    Raises:
        ValueError: If the API backend is selected without a usable API key.
    """

    from local_translator.config import (
        ApiModelSettings,
        Settings,
    )
    from local_translator.engine.translators import (
        ApiTranslator,
        LocalMTTranslator,
    )

    settings = Settings.model_validate(spec.settings)
    tr = settings.translation

    # The cloud backend is one destination with two possible services behind
    # it. Which one is a setting, not a backend, so that "where does this
    # paragraph go" and "which vendor serves it" stay separate questions.
    # 云端后端是一个去向，背后可能是两家服务中的一家。是哪一家由设置项决定而非由后端
    # 决定，这样「段落去哪」和「谁来服务」才是两个独立的问题。
    if tr.backend == "api" and tr.cloud_provider == "tencent":
        return _build_tencent(settings, glossary_matcher)

    if tr.backend == "api":
        api: ApiModelSettings = settings.api_model
        if not spec.api_key:
            raise ValueError(
                f"API backend selected but no key found for account "
                f"{api.keyring_account!r} in the system keychain."
            )
        return ApiTranslator(
            lang_in=tr.lang_in,
            lang_out=tr.lang_out,
            model=api.model_name,
            base_url=api.base_url or None,
            api_key=spec.api_key,
            ignore_cache=tr.ignore_cache,
            skip_reasoning=api.skip_reasoning,
            prompt_template=api.chat_prompt_template,
        )

    if tr.backend == "hybrid":
        return _build_hybrid(spec, settings, glossary_matcher)

    local: LocalModelSettings = settings.local_model
    _require_local_server(local)
    translator = LocalMTTranslator(
        lang_in=tr.lang_in,
        lang_out=tr.lang_out,
        settings=local,
        ignore_cache=tr.ignore_cache,
        glossary_rev=settings.glossary.glossary_rev,
        custom_hint=tr.custom_system_prompt,
    )
    if glossary_matcher is not None:
        translator.set_glossary_provider(glossary_matcher)
        translator.hard_protect = settings.glossary.hard_protect_whitelist
    return translator


def _build_tencent(settings, glossary_matcher):
    """
    中文：构造腾讯云翻译器，并把限流调到 TMT 的配额上。

    Build the Tencent Cloud translator.

    Args:
        settings (Settings): Resolved settings, for the language pair, the rate
            limit and the glossary revision.
        glossary_matcher (Callable | None): Glossary lookup, applied the same
            way as for every other backend.

    Returns:
        BaseTranslator: A translator that sends each paragraph to TMT.

    Raises:
        ValueError: When the credentials are missing, naming the file they are
            read from so the reader knows where to put them.
    """
    from babeldoc.translator.translator import set_translate_rate_limiter

    from local_translator.config import TranslationSettings
    from local_translator.engine.translators import TencentTranslator
    from local_translator.secrets import KEY_DIR, read_key_values

    tr = settings.translation

    # TMT's default quota is five requests a second, and BabelDOC throttles
    # every backend through one process-wide bucket that also happens to sit at
    # five. Leaving it there is right for TMT and wrong to rely on by accident,
    # so it is set deliberately here.
    #
    # `qps` still carries 1000 by default because it is meaningless for local
    # inference, and applying that number to a metered service would spend the
    # quota in the first second. A value the reader actually chose — anything
    # below that sentinel — is honoured, so raising the quota in the Tencent
    # console and lowering `qps` to match both work.
    # TMT 的默认配额是每秒五次，而 BabelDOC 用一个进程级漏桶限流所有后端，默认也恰好是
    # 五。恰好对上不等于可以依赖，所以这里显式设定。
    #
    # qps 的默认值仍是 1000——它对本地推理没有意义；把这个数用在计费服务上，第一秒就会把
    # 配额打光。读者真正选过的值（低于这个哨兵值的）会被采用，因此在腾讯云控制台提配额、
    # 再把 qps 调到相应数值，两边都能生效。
    sentinel = TranslationSettings.model_fields["qps"].default
    set_translate_rate_limiter(max(1, int(tr.qps if tr.qps < sentinel else _TMT_QPS)))

    # Tencent Cloud signs every request with a SecretId and a SecretKey, so the
    # single-key path that carries `spec.api_key` cannot express it. The pair is
    # read here, in the worker, and never travels through the job spec or any
    # log line.
    # 腾讯云每次请求都要用 SecretId 与 SecretKey 一起签名，携带单个 api_key 的那条路径
    # 表达不了。这一对在 worker 里直接读取，不经过 job spec，也不进任何日志。
    keys = read_key_values("tencent")
    if not (keys.get("SECRET_ID") and keys.get("SECRET_KEY")):
        raise ValueError(
            f"云端服务选的是腾讯云，但 {KEY_DIR}/tencent.env 里没有 SECRET_ID 与 "
            "SECRET_KEY。请到设置页的「云端翻译服务」里填写并测试。"
        )
    translator = TencentTranslator(
        lang_in=tr.lang_in,
        lang_out=tr.lang_out,
        secret_id=keys["SECRET_ID"],
        secret_key=keys["SECRET_KEY"],
        region=keys.get("REGION") or "ap-guangzhou",
        project_id=int(keys.get("PROJECT_ID") or 0),
        term_repos=[x for x in (keys.get("TERM_REPO_IDS") or "").split(",") if x.strip()],
        sent_repos=[x for x in (keys.get("SENT_REPO_IDS") or "").split(",") if x.strip()],
        ignore_cache=tr.ignore_cache,
        glossary_rev=settings.glossary.glossary_rev,
        settings=settings.local_model,
    )
    if glossary_matcher is not None:
        translator.set_glossary_provider(glossary_matcher)
        translator.hard_protect = settings.glossary.hard_protect_whitelist
    return translator


def _require_local_server(local: LocalModelSettings) -> None:
    """
    中文：开跑前确认本地推理服务真的在监听；不在就立刻报错，不要跑完一整篇才发现。

    Fail immediately when the local inference server is not listening.

    This process cannot start the server — the supervisor lives in the parent —
    so the only useful thing it can do is refuse early. Without this check a
    dead endpoint is invisible: every paragraph's request fails, BabelDOC keeps
    the English source for each one, and the job runs to completion looking
    healthy. On one MacBook Air run that cost fourteen and a half minutes and
    252 untranslated paragraphs before anything said so. A refused connection
    answers in milliseconds, so the check is free.

    Args:
        local (LocalModelSettings): Settings carrying the host and port the
            supervisor binds the server to.

    Raises:
        RuntimeError: When nothing answers ``/health`` at that address.
    """
    import httpx

    url = f"http://{local.host}:{local.port}/health"
    try:
        response = httpx.get(url, timeout=_PROBE_TIMEOUT, trust_env=False)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise RuntimeError(
            f"本地模型服务没有在 {local.host}:{local.port} 上响应，这一趟不会译出任何内容。"
            "请回到主界面重新开始翻译——程序会重新加载模型。"
            f"（探测失败：{type(exc).__name__}）"
        ) from exc


def _build_term_translator(spec: JobSpec, fallback):
    """
    中文：为术语提取构造 API 翻译器；不可用时回退为不提取。

    Build the translator used for automatic term extraction.

    Term extraction needs structured JSON output, which the local
    translation-only model cannot produce — so it is always the API model or
    nothing. Returning the fallback (the main translator) with extraction
    disabled is handled by the caller.

    Args:
        spec (JobSpec): Job description carrying the API credentials.
        fallback: Translator to return when extraction is not possible.

    Returns:
        tuple: ``(translator, enabled)`` where ``enabled`` is ``False`` when no
            usable API model is configured.
    """
    if not spec.extract_terms or not spec.api_key:
        return fallback, False

    from babeldoc.format.pdf.translation_config import TranslationConfig  # noqa: F401

    from local_translator.config import Settings
    from local_translator.engine.translators import ApiTranslator

    settings = Settings.model_validate(spec.settings)
    api = settings.api_model
    try:
        engine = ApiTranslator(
            lang_in=settings.translation.lang_in,
            lang_out=settings.translation.lang_out,
            model=api.model_name,
            base_url=api.base_url or None,
            api_key=spec.api_key,
            ignore_cache=False,
            skip_reasoning=api.skip_reasoning,
            prompt_template=api.chat_prompt_template,
        )
    except Exception:
        logger.warning("Could not build the API term extractor; skipping extraction", exc_info=True)
        return fallback, False

    logger.info("Automatic term extraction enabled via %s", api.model_name)
    return engine, True


def _build_hybrid(spec: JobSpec, settings, glossary_matcher):
    """
    中文：构造混合翻译器，本地与 API 并行分担段落。

    Build the hybrid translator.

    Falls back to the local model alone when no API key is available, so
    selecting hybrid without credentials degrades rather than fails.

    Args:
        spec (JobSpec): Job description carrying credentials.
        settings (Settings): Resolved settings.
        glossary_matcher (Callable | None): Glossary lookup for the local side.

    Returns:
        BaseTranslator: A hybrid translator, or a plain local one when the API
            half cannot be constructed.
    """
    from babeldoc.translator.translator import set_translate_rate_limiter

    from local_translator.engine.hybrid import HybridTranslator
    from local_translator.engine.hybrid.router import RouterPolicy
    from local_translator.engine.translators import ApiTranslator, LocalMTTranslator

    tr = settings.translation
    _require_local_server(settings.local_model)
    local = LocalMTTranslator(
        lang_in=tr.lang_in, lang_out=tr.lang_out, settings=settings.local_model,
        ignore_cache=tr.ignore_cache, glossary_rev=settings.glossary.glossary_rev,
        custom_hint=tr.custom_system_prompt,
    )
    if glossary_matcher is not None:
        local.set_glossary_provider(glossary_matcher)
        local.hard_protect = settings.glossary.hard_protect_whitelist

    if not spec.api_key:
        logger.warning("Hybrid requested without an API key; using the local model alone")
        return local

    api = settings.api_model
    remote = ApiTranslator(
        lang_in=tr.lang_in, lang_out=tr.lang_out, model=api.model_name,
        base_url=api.base_url or None, api_key=spec.api_key,
        ignore_cache=tr.ignore_cache,
        skip_reasoning=api.skip_reasoning, prompt_template=api.chat_prompt_template,
    )
    # BabelDOC throttles every translation through a process-wide leaky bucket
    # fixed at 5 QPS, and never wires the `qps` setting to it — TranslationConfig
    # uses that value only to size its thread pool. Left alone it caps this
    # document at 298 paragraphs / 5 = 60 seconds no matter how many threads or
    # how fast either backend is. Raising it here is scoped to this subprocess,
    # and therefore to this job and this backend; the local and API modes keep
    # BabelDOC's behaviour unchanged.
    # BabelDOC 用一个固定 5 QPS 的进程级漏桶限流所有翻译，且从不把 `qps` 设置接到
    # 它上面——TranslationConfig 只拿该值决定线程池大小。不动它的话，本文 298 段
    # 会被卡在 60 秒，与线程数和后端速度无关。此处仅在本子进程内提高，因此只影响
    # 本任务与本后端；local 与 api 模式仍保持 BabelDOC 原样。
    set_translate_rate_limiter(max(1, int(tr.qps)))

    logger.info(
        "Hybrid mode: %d local slots, API strategy %s, %d worker threads, %g QPS",
        tr.hybrid_local_slots, tr.hybrid_api_strategy, tr.hybrid_pool_workers, tr.qps,
    )
    return HybridTranslator(
        local, remote, tr.lang_in, tr.lang_out,
        policy=RouterPolicy(
            local_slots=tr.hybrid_local_slots,
            local_share_cap=tr.hybrid_local_share_cap,
            api_marker_threshold=tr.hybrid_api_marker_threshold,
            api_length_threshold=tr.hybrid_api_length_threshold,
        ),
        api_strategy=tr.hybrid_api_strategy,
        batch_threads=max(1, tr.hybrid_pool_workers - tr.hybrid_local_slots),
        max_batch=tr.hybrid_max_batch,
        ignore_cache=tr.ignore_cache,
    )


def _build_config(spec: JobSpec, translator, term_translator, extract_terms: bool = False):
    """
    中文：把应用设置翻译成 BabelDOC 的 TranslationConfig。

    Map application settings onto BabelDOC's ``TranslationConfig``.

    Args:
        spec (JobSpec): Job description.
        translator (BaseTranslator): Main paragraph translator.
        term_translator (BaseTranslator): Translator used for glossary term
            extraction; may be the same object as ``translator``.

    Returns:
        TranslationConfig: Configured for this job, with output written next to
            the source file and intermediates isolated under ``working_dir``.
    """
    from babeldoc.format.pdf.translation_config import (
        TranslationConfig,
        WatermarkOutputMode,
    )

    from local_translator.config import Settings

    settings = Settings.model_validate(spec.settings)
    pdf = settings.pdf
    tr = settings.translation

    watermark = {
        "watermarked": WatermarkOutputMode.Watermarked,
        "no_watermark": WatermarkOutputMode.NoWatermark,
        "both": WatermarkOutputMode.Both,
    }.get(pdf.watermark_mode, WatermarkOutputMode.NoWatermark)

    glossaries = []
    if spec.glossary_csv:
        from babeldoc.glossary import Glossary

        path = Path(spec.glossary_csv)
        if path.exists():
            try:
                g = Glossary.from_csv(path, tr.lang_out)
                if g.entries:
                    glossaries.append(g)
            except Exception:
                logger.warning("Failed to load glossary %s", path, exc_info=True)

    from babeldoc.docvision.doclayout import DocLayoutModel

    from local_translator.engine import scanned

    # BabelDOC picks its own output filenames and will happily overwrite an
    # existing translation. Writing to a staging directory first lets us move
    # the results out under non-clobbering names afterwards.
    # BabelDOC 自行决定输出文件名且会覆盖同名文件；先写暂存目录，事后按不覆盖规则搬运。
    staging = Path(spec.working_dir) / "out"
    staging.mkdir(parents=True, exist_ok=True)

    return TranslationConfig(
        input_file=spec.input_file,
        font=None,
        pages=spec.pages,
        output_dir=str(staging),
        working_dir=spec.working_dir,
        translator=translator,
        term_extraction_translator=term_translator,
        debug=False,
        lang_in=tr.lang_in,
        lang_out=tr.lang_out,
        no_dual=not pdf.output_dual,
        no_mono=not pdf.output_mono,
        qps=tr.qps,
        min_text_length=tr.min_text_length,
        doc_layout_model=DocLayoutModel.load_onnx(),
        watermark_output_mode=watermark,
        # False selects create_side_by_side_dual_pdf(): one wide page holding the
        # original and the translation next to each other.
        # False 走 create_side_by_side_dual_pdf()：单页并排原文与译文。
        use_alternating_pages_dual=pdf.use_alternating_pages_dual,
        dual_translate_first=pdf.dual_translate_first,
        enhance_compatibility=pdf.enhance_compatibility,
        skip_clean=pdf.skip_clean,
        skip_form_render=pdf.skip_form_render,
        # Vector curves are always rendered. A scientific PDF draws its plots,
        # diagrams and callout boxes as vector paths, so skipping them strips
        # the artwork out of the translation — one paper lost the rounded box
        # around every "Takeaway" note that way. This was briefly a setting;
        # a value of True left in a settings file went on doing that damage
        # long after the default was corrected, so the choice is gone.
        # 矢量曲线一律渲染。科技论文的图表、示意图与提示框都是矢量路径，跳过它们
        # 等于把插图从译文里抹掉——某篇论文每个 "Takeaway" 提示框的圆角边框就是这样
        # 没的。这曾经是一个可选项；设置文件里残留的 True 在默认值改回之后仍在继续
        # 造成同样的破坏，因此这个选择被取消了。
        skip_curve_render=False,
        auto_enable_ocr_workaround=pdf.auto_enable_ocr_workaround,
        # Scanned-page detection renders every page twice and compares the two
        # through OpenCV and scikit-image — 0.24 GB on a 19-page paper. It is
        # skipped when the reader turned the OCR workaround off, and when no
        # page is covered enough by images to be a picture of a page.
        # 扫描页检测会把每页渲染两次并经 OpenCV 与 scikit-image 比对——19 页论文要
        # 0.24 GB。读者关掉 OCR 兜底时跳过；没有任何一页被图像大面积覆盖、不可能是
        # 整页图片时也跳过。
        skip_scanned_detection=(
            not pdf.auto_enable_ocr_workaround
            or not scanned.could_be_scanned(spec.input_file)
        ),
        primary_font_family=pdf.primary_font_family,
        custom_system_prompt=tr.custom_system_prompt or None,
        # Hybrid needs its own thread budget: a worker is occupied until its
        # backend answers, so the local slots and the API side each need their
        # own supply. The other modes keep their existing setting untouched.
        # 混合模式需要独立的线程预算：线程会一直被占用到后端返回，本地槽位与 API
        # 侧各需一份。其他模式的设置保持不变。
        pool_max_workers=(
            # Cloud work gets the generous thread count, local work the modest
            # one, because the two resources scale oppositely. The local model
            # is one GPU: past its slot count, extra threads only queue.
            # A hosted endpoint is concurrency-elastic — measured on this
            # project at 116 chars/s at concurrency 1, 325 at 4 and 749 at 16,
            # still climbing at 64 — so the number that suits the GPU starves
            # it.
            #
            # Pure API mode used to inherit `pool_max_workers`, which the
            # edition validator derives from the *local* slot count
            # (n_parallel × 2 = 16). That number has no meaning when no local
            # model is involved, and it was the reason API mode ran far below
            # what the endpoint would give.
            #
            # Tencent is the exception: its default quota is five requests a
            # second, so threads beyond a handful sit in the rate limiter
            # rather than in flight.
            # 云端用宽裕的线程数，本地用保守的，因为两种资源的伸缩方向相反。本地模型只有
            # 一块 GPU：超过槽位数之后，多出来的线程只是排队。托管接口则是并发弹性的
            # ——本项目实测并发 1 时 116 字符/秒、并发 4 时 325、并发 16 时 749，到 64
            # 仍在上升——因此适合 GPU 的那个数字会把它饿着。
            #
            # 纯云端模式此前沿用 pool_max_workers，而该值由版本校验器从「本地」槽位数
            # 推导而来（n_parallel × 2 = 16）。在完全不涉及本地模型时这个数字没有意义，
            # 而它正是云端模式远低于接口所能提供的速度的原因。
            #
            # 腾讯云是例外：它的默认配额是每秒五次请求，因此超出少数几个之后，线程只是
            # 停在限流器里而不是在途。
            tr.hybrid_pool_workers
            if tr.backend == "hybrid"
            or (tr.backend == "api" and tr.cloud_provider != "tencent")
            else tr.pool_max_workers
        ),
        # Extraction runs only when the coverage gate asked for it and an API
        # model is available; the local translation-only model cannot emit the
        # structured JSON it requires.
        # 仅当覆盖率闸门要求且 API 模型可用时才提取；本地纯翻译模型产不出所需的结构化 JSON。
        auto_extract_glossary=extract_terms,
        glossaries=glossaries,
        save_auto_extracted_glossary=extract_terms,
    )


async def _translate_with_events(config, spec: JobSpec, queue) -> dict[str, Any]:
    """
    中文：驱动 BabelDOC 异步翻译并把进度事件转发给父进程。

    Run BabelDOC's async translation, forwarding progress to the parent.

    Args:
        config (TranslationConfig): Prepared translation configuration.
        spec (JobSpec): Job description, for stamping ``job_id`` on events.
        queue (multiprocessing.Queue): Channel to the parent process.

    Returns:
        dict[str, Any]: Result paths under keys ``dual_path``, ``mono_path``
            and ``glossary_path``; values are strings or ``None``.
    """
    import babeldoc.format.pdf.high_level as high_level

    result: dict[str, Any] = {"dual_path": None, "mono_path": None, "glossary_path": None}
    last_emit = 0.0
    last_stage = ""
    finished = False

    # Iterate the generator explicitly so it can be closed deterministically.
    # BabelDOC's async_translate ends with `await finish_event.wait()`, and that
    # event is only set from a worker thread via call_soon_threadsafe — a path
    # that was observed not to fire, leaving the generator suspended forever
    # after the PDFs had already been written. Since the `finish` event carries
    # the complete TranslateResult, there is nothing left to wait for once it
    # arrives, so we stop pulling and close the generator ourselves.
    # 显式持有生成器以便确定性关闭。async_translate 末尾的 await finish_event.wait()
    # 曾观察到永不返回（PDF 已写完却仍挂起）；finish 事件已带回完整结果，拿到即可收工。
    agen = high_level.async_translate(config)
    try:
        async for event in agen:
            etype = event.get("type")

            if etype == "progress_update":
                stage = event.get("stage", "")
                if stage != last_stage:
                    last_stage = stage
                    logger.debug("Stage %-42s own memory %.0f MiB",
                                 stage, _own_memory_mib())
                # Throttle: BabelDOC emits an event per paragraph, which would
                # swamp both the queue and the UI on a long document.
                # 限流：BabelDOC 每段都发事件，长文档会淹没队列与前端。
                now = time.monotonic()
                if now - last_emit < 0.2:
                    continue
                last_emit = now
                queue.put(
                    {
                        "type": EV_PROGRESS,
                        "job_id": spec.job_id,
                        "overall": float(event.get("overall_progress", 0.0)),
                        "stage": event.get("stage", ""),
                        "stage_current": int(event.get("stage_current", 0)),
                        "stage_total": int(event.get("stage_total", 0)),
                    }
                )

            elif etype == "error":
                # An error emitted after a successful finish is BabelDOC's
                # shutdown bookkeeping, not a failure of the translation.
                # finish 之后再来的 error 属于 BabelDOC 的收尾记账，不是翻译失败。
                if finished:
                    break
                raise RuntimeError(str(event.get("error", "Unknown BabelDOC error")))

            elif etype == "finish":
                tres = event["translate_result"]
                result["dual_path"] = str(tres.dual_pdf_path) if tres.dual_pdf_path else None
                result["mono_path"] = str(tres.mono_pdf_path) if tres.mono_pdf_path else None
                result["glossary_path"] = (
                    str(tres.auto_extracted_glossary_path)
                    if tres.auto_extracted_glossary_path
                    else None
                )
                finished = True
                break
    finally:
        try:
            await asyncio.wait_for(agen.aclose(), timeout=15)
        except (TimeoutError, Exception):
            logger.debug("async_translate did not close cleanly; continuing", exc_info=True)

    return result


# What ``footprint`` calls a process's own memory: dirty pages, compressed pages
# and IOKit allocations, without the shared clean pages a memory-mapped file
# contributes. This is the number macOS itself uses for memory pressure and
# shows in Activity Monitor.
# footprint 所说的进程自身内存：脏页、压缩页与 IOKit 分配，不含 mmap 文件带来的共享
# 干净页。这正是 macOS 自己用于内存压力、也是活动监视器显示的数字。
_FOOTPRINT = re.compile(rb"phys_footprint:\s*([\d.]+)\s*(\w+)")
_FOOTPRINT_UNITS = {b"B": 1 / 1048576, b"KB": 1 / 1024, b"MB": 1.0, b"GB": 1024.0}


def _own_memory_mib() -> float:
    """
    中文：返回本进程当前自身内存（MiB），用于按阶段归因。

    Report this process's own memory right now, for step-by-step attribution.

    Not ``ru_maxrss``, which this used to be and which is wrong twice over: it
    is a high-water mark rather than a current reading, so every checkpoint
    after the peak reports the peak; and resident set size counts the
    memory-mapped model file, which is shared and evictable and not a cost this
    process pays.

    Cheap enough to leave in at debug level — the next person asking where the
    memory goes gets the answer from a log line instead of a day's
    instrumentation.

    Returns:
        float: ``phys_footprint`` in MiB, or ``0.0`` where ``footprint`` is not
            available, which is every platform but macOS.
    """
    import subprocess

    try:
        raw = subprocess.run(["/usr/bin/footprint", "-p", str(os.getpid())],
                             capture_output=True, timeout=10).stdout
    except Exception:  # noqa: BLE001
        return 0.0
    match = _FOOTPRINT.search(raw)
    if not match:
        return 0.0
    return float(match.group(1)) * _FOOTPRINT_UNITS.get(match.group(2), 1.0)


def _latex_settings(spec: JobSpec):
    """
    中文：从任务快照里取出 LaTeX 那条路要用的东西。

    Unpack what the source path needs from the job's settings snapshot.

    Args:
        spec (JobSpec): Job description.

    Returns:
        tuple: ``(settings, selection, matcher)`` — the resolved settings, the
            reader's region choices and a glossary matcher or ``None``.
    """
    from local_translator.config import Settings
    from local_translator.latex.selection import Selection

    settings = Settings.model_validate(spec.settings)
    selection = Selection.from_settings(settings)

    matcher = None
    if spec.glossary_terms and settings.glossary.enabled:
        from local_translator.glossary.matcher import GlossaryMatcher

        matcher = GlossaryMatcher(
            spec.glossary_terms,
            max_terms=settings.glossary.max_terms_per_prompt,
        )
    return settings, selection, matcher


def run_latex_job(spec: JobSpec, queue, cancel_event) -> None:
    r"""
    中文：子进程主函数的 LaTeX 分支——从源码翻译一篇论文并编译。

    Translate a paper from its LaTeX source. The source branch of the worker.

    Args:
        spec (JobSpec): Job description, with ``latex_source`` set.
        queue (multiprocessing.Queue): Event channel to the parent.
        cancel_event (multiprocessing.Event): Set by the parent to request a
            stop. Checked between blocks, which is the only place this path can
            stop cleanly — a compile is a subprocess that runs to completion.

    Returns:
        None. Every outcome travels through ``queue``.

    Note:
        The bilingual PDF costs a second compile: the same assembled document
        with the translations left out, so both halves come through the same
        preamble and the same engine and differ only in language. The pairing
        is by page index and drifts, because Chinese sets shorter than English
        — see :mod:`local_translator.latex.dual`.
        双语版要多编译一次：同一份装配好的文档、不放译文，于是两边经过同样的前言、同样
        的引擎，只差语言这一件事。配对按页序，会渐渐错开，因为中文排得比英文短——见
        local_translator.latex.dual。
    """
    from local_translator.latex import engine as tex_engine
    from local_translator.latex import pipeline
    from local_translator.latex import source as tex_source
    from local_translator.service import render_translated_image, with_protected_terms

    began = time.monotonic()
    settings, selection, matcher = _latex_settings(spec)
    root = Path(spec.latex_source)
    main = tex_source.find_main(root)
    if main is None:
        raise RuntimeError(f"{root.name} 里没有找到带 \\documentclass 的主文件")

    from local_translator.config import get_paths

    chosen = tex_engine.resolve(settings.latex.engine_path or None,
                                get_paths().bin_dir)
    if chosen is None:
        raise RuntimeError(
            "找不到 LaTeX 编译器。请安装含 TeX 功能的版本，或在设置里指定 TeX 路径。")
    engine_file, is_system = chosen

    translator = _build_translator(spec, matcher)

    image_translator = None
    if selection.translate_images:
        # A second translator: the first one's prompt is replaced with the one
        # written for LaTeX source, and a line read off a chart is not LaTeX.
        # 图片用另一个翻译器：第一个的提示词已换成为 LaTeX 源码写的那一套，而从图表上
        # 读出来的一行字不是 LaTeX。
        lines = with_protected_terms(_build_translator(spec, matcher).translate, matcher)
        image_translator = lambda data: render_translated_image(data, lines)  # noqa: E731

    def progress(done: int, total: int) -> None:
        """中文：把逐块进度报给父进程。 / Report per-block progress."""
        if cancel_event.is_set():
            raise RuntimeError("已取消")
        queue.put({
            "type": EV_PROGRESS,
            "job_id": spec.job_id,
            # Compiling is the last tenth: it is one step, but on an
            # eighty-page paper it is a visible one.
            # 编译占最后一成：它只是一步，但在八十页的论文上是看得见的一步。
            "overall": 0.9 * (done / max(total, 1)),
            "stage": "翻译源码",
            "stage_current": done,
            "stage_total": total,
        })

    outcome = pipeline.translate_tree(
        root, main, translator,
        workers=settings.latex.workers,
        ignore_cache=settings.translation.ignore_cache,
        font=settings.latex.cjk_font or None,
        tectonic=None if is_system else engine_file,
        system_engine=engine_file if is_system else None,
        selection=selection,
        glossary=matcher,
        hard_protect=settings.glossary.hard_protect_whitelist,
        translate_image=image_translator,
        # The same three output switches the PDF path reads. A reader who chose
        # "bilingual only" chose it about their papers.
        # 与 PDF 那条路读的是同样三个输出开关。读者选「只要双语」，选的是自己的论文。
        output_mono=settings.pdf.output_mono,
        output_dual=settings.pdf.output_dual,
        alternating_dual=settings.pdf.use_alternating_pages_dual,
        dual_translate_first=settings.pdf.dual_translate_first,
        on_progress=progress,
    )

    if not outcome.ok and outcome.dual is None:
        raise RuntimeError(
            f"LaTeX 编译失败：{outcome.error}。译文源码留在 {outcome.tex}，可据此排查。")

    published = _publish_outputs(spec, {
        "mono_path": str(outcome.pdf) if outcome.pdf else None,
        "dual_path": str(outcome.dual) if outcome.dual else None,
    })
    stats = outcome.stats
    logger.info("LaTeX 翻译完成：%d/%d 块被采用，%d 块保留原文",
                stats.get("translated", 0), stats.get("total", 0),
                stats.get("refused", 0))

    usage = translator.usage_summary() if hasattr(translator, "usage_summary") else {}
    queue.put({
        "type": EV_FINISH,
        "job_id": spec.job_id,
        "dual_path": published.get("dual_path"),
        "mono_path": published.get("mono_path"),
        "glossary_path": None,
        "tracking_path": None,
        "usage": usage,
        "images_restored": 0,
        "skipped": _latex_skipped(stats, selection),
        "duration_ms": int((time.monotonic() - began) * 1000),
    })


def _latex_skipped(stats: dict, selection) -> dict[str, int]:
    """
    中文：把这一趟保留原文的情况写成界面能显示的计数。

    Summarise what was left in the original, in the shape the UI already reads.

    Args:
        stats (dict): Counts from :func:`translate.translate_document`.
        selection (Selection): The reader's region choices.

    Returns:
        dict[str, int]: Reason to count, matching what the PDF path reports so
            one panel can render either.
    """
    summary: dict[str, int] = {}
    refused = int(stats.get("refused", 0) or 0)
    if refused:
        summary["译文未通过检查"] = refused
    partial = int(stats.get("partial", 0) or 0)
    if partial:
        summary["部分保留原文"] = partial
    return summary


def _latex_or_fall_back(spec: JobSpec, queue, cancel_event) -> bool:
    """
    中文：先走源码；源码编不过、而手上又有 PDF 的话，退回去翻 PDF。

    Try the source path, and fall back to the PDF when it cannot compile.

    Args:
        spec (JobSpec): Job description, with ``latex_source`` set.
        queue (multiprocessing.Queue): Event channel to the parent.
        cancel_event (multiprocessing.Event): The parent's cancel request.

    Returns:
        bool: ``True`` when the source path finished the job — successfully or
            by an error the caller should report. ``False`` when the caller
            should carry on into the PDF path instead.

    Note:
        Some papers do not compile from their own sources. Not "do not compile
        once translated" — do not compile at all: arXiv:2608.12209 loads
        ``xcolor`` from its class and then three more times with three
        different option sets, and stops at "Option clash" before a word is
        translated. A reader who pasted an arXiv link wants the paper, not a
        verdict on its build; where a PDF was downloaded alongside the source,
        translating that is the answer.
        有些论文从自己的源码就编不过。不是「译完之后编不过」，是**根本编不过**：
        arXiv:2608.12209 的文档类加载了 xcolor，前言里又用三组不同选项加载了三次，一个字
        还没译就停在「Option clash」。粘 arXiv 链接的读者要的是这篇论文，不是一份关于它
        构建状况的判决书；既然源码旁边就下好了 PDF，翻那个就是答案。
    """
    try:
        run_latex_job(spec, queue, cancel_event)
        return True
    except Exception as problem:  # noqa: BLE001
        if cancel_event.is_set():
            raise
        source = Path(spec.input_file)
        if source.suffix.lower() != ".pdf" or not source.is_file():
            # Nothing to fall back to: this job was a folder of sources, and
            # there never was a PDF of the paper.
            # 没有可退的东西：这个任务提交的是一个源码文件夹，从来就没有这篇论文的 PDF。
            raise
        logger.warning("源码翻译没有成功（%s），改为翻译 PDF", str(problem)[:200])
        return False


def run_job(spec: JobSpec, queue, cancel_event) -> None:
    """
    中文：子进程主函数——执行一次完整的 PDF 翻译并回传进度与结果。

    Execute one translation job. This is the child process's ``target``.

    All outcomes are communicated through ``queue``; the function itself returns
    ``None`` and never raises into the process boundary, so the parent always
    receives either a ``finish`` or an ``error`` event and can settle the job.

    Args:
        spec (JobSpec): Job description.
        queue (multiprocessing.Queue): Event channel to the parent. Receives
            ``started``, throttled ``progress``, and exactly one terminal
            ``finish`` or ``error`` event.
        cancel_event (multiprocessing.Event): Set by the parent to request
            graceful cancellation.
    """
    # The worker is a fresh process, so it does not inherit the parent's log
    # level; without this, someone who asked for debug output gets it from
    # everything except the part doing the work.
    # worker 是全新进程，不继承父进程的日志级别；没有这一段，要求输出调试日志的人
    # 会从除了真正干活的那部分之外的所有地方拿到日志。
    logging.basicConfig(
        level=os.environ.get("LOCALTRANS_LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s  %(levelname)-7s %(name)s: %(message)s",
    )
    for noisy in ("httpx", "httpcore", "openai", "pdfminer", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    started = time.monotonic()
    translator = None
    filter_cls = None

    try:
        if spec.latex_source:
            # Translating from source shares the translator, the glossary and
            # the region switches with the path below, and nothing else: there
            # is no page to lay out, so none of the repairs that follow have
            # anything to repair.
            # 从源码翻译与下面那条路共用翻译器、术语库和区域开关，此外别无共同之处：这里
            # 没有页面需要排版，所以下面那些修复步骤没有任何东西可修。
            if _latex_or_fall_back(spec, queue, cancel_event):
                return

        # Before BabelDOC, and it has to be: every module of it that touches an
        # image does `import cv2` at the top, and once one of them has bound the
        # real package there is nothing left to substitute. OpenCV is 118 MB
        # installed for nine functions, 78 MB of which is ffmpeg and video
        # encoders this program never calls into — see engine/cvshim.py.
        # 必须早于 BabelDOC：它凡是碰图像的模块都在文件开头 `import cv2`，只要有一个
        # 绑定了真包，就没有什么可替换的了。OpenCV 装出来 118 MB 只为九个函数，其中
        # 78 MB 是本程序从不调用的 ffmpeg 与视频编码器——见 engine/cvshim.py。
        if os.environ.get("LOCALTRANS_USE_OPENCV") != "1":
            from local_translator.engine import cvshim, scishim

            cvshim.install()
            # Same reasoning, same timing: BabelDOC does
            # `from sklearn.cluster import DBSCAN` and
            # `from skimage.metrics import structural_similarity` at module
            # scope. Between them those two packages drag in 45 MB — scikit-learn,
            # scikit-image and the SciPy neither BabelDOC nor this program calls
            # into directly — for one clustering call and one image comparison.
            # 同样的道理、同样的时机：BabelDOC 在模块开头就写了那两个 import。这两个包
            # 合起来拖进 45 MB——scikit-learn、scikit-image，以及 BabelDOC 和本程序都不
            # 直接调用的 SciPy——只为一次聚类和一次图像比对。
            scishim.install()

        import babeldoc.format.pdf.high_level as high_level

        from local_translator.config import Settings
        from local_translator.engine import (
            accents,
            boldface,
            brackets,
            captionlabel,
            crosspage,
            dropcap,
            fonts,
            fontset,
            formulas,
            frontmatter,
            hairlines,
            inlinemath,
            interleaved,
            layout_boxes,
            layout_filter,
            layout_model,
            lists,
            longtoken,
            notation,
            oblique,
            placeholders,
            reclaim,
            scaling,
            smallcaps,
            titlebox,
        )

        # A comma-separated list of passes to leave uninstalled, for bisecting a
        # defect down to the pass that causes it. Unset in normal use.
        # 以逗号分隔的待停用处理步骤，用于把某个缺陷二分定位到具体的步骤。
        # 正常使用时不设置。
        disabled = {
            name.strip() for name in os.environ.get("LOCALTRANS_DISABLE", "").split(",")
            if name.strip()
        }
        if disabled:
            logger.warning("Passes disabled for this run: %s", ", ".join(sorted(disabled)))

        def enabled(name: str) -> bool:
            """中文：该步骤是否启用。 / Whether a pass should be installed."""
            return name not in disabled

        # Layout boxes are repaired first: everything below reads the paragraphs
        # they produce.
        # 先修版面框：下面所有步骤读的都是它切出来的段落。
        # Earliest of all: a form XObject's BBox is a clipping boundary, and
        # content it draws outside that boundary is not painted. The IL took it
        # as a hint and projected everything, so text and paths a reader never
        # sees arrived as ordinary page content — in one measured case, a
        # figure's labels appeared a second time in the facing column. Every
        # stage below reads these elements, so the clip is applied before any
        # of them.
        # 最先执行：Form XObject 的 BBox 是裁剪边界，它在界外绘制的内容不会被画出来。
        # 中间表示把它当成了提示，照单全收，于是读者根本看不到的文字与路径以普通页面内容
        # 的身份进入——在实测的一个案例里，某张图的标签在对面那一栏又出现了一份。下面每
        # 一步都要读这些元素，因此裁剪放在它们之前。
        # A subset font that records its weight only in its name reads as
        # regular, and every bold heading is set in the body face. Belongs with
        # the IL, which is where the font metadata is built.
        # 只在名字里记录字重的子集字体会被当成正体，于是每个加粗小标题都用正文字体排了出来。
        # 归到中间表示这一层，字体元数据就是在这里构建的。
        if enabled("bold_faces"):
            boldface.install()

        if enabled("xobject_clip"):
            from local_translator.engine import xobjclip

            xobjclip.install()

        if enabled("layout_boxes"):
            layout_boxes.install()

        # A figure that is stamped back from the source carries its own labels
        # with it, so the copies the pipeline would otherwise translate and
        # typeset are covered by the stamp — and their boxes distort the layout
        # of the paragraphs around them until they are gone. Only when the
        # reader asks for figures to be translated do those labels have work to
        # do, and then this stays out of the way.
        # 从原文整块搬回的图片自带标签，因此管线另行翻译并排版的那份会被搬回的图盖住
        # ——而在它们消失之前，它们的框会一直干扰周围段落的版面计算。只有当读者要求翻译
        # 图片时，那些标签才有意义，那时本步骤不介入。
        preserved = (spec.settings or {}).get("pdf", {}).get("preserve_kinds") or []
        if enabled("figure_text") and "figure" in preserved:
            from local_translator.engine import figuretext

            figuretext.install()

        # A paragraph cut in two by something drawn in the middle of it — a
        # margin stamp, a rotated header — is rejoined immediately after the
        # boxes are repaired and before anything reads a paragraph's text.
        # 被中途绘制的水印、旋转页眉切开的段落，在版面框修好之后、任何一步读到段落
        # 文本之前立刻接回去。
        if enabled("interleaved"):
            interleaved.install()

        # Numbered lists split before anything else looks at the paragraphs.
        # 编号列表的切分要早于其它一切对段落的处理。
        if enabled("lists"):
            lists.install()

        # Titles, authors, affiliations and footnotes are recognised from the
        # page, because the layout model has no class for any of them.
        # 文章标题、作者、单位与脚注由页面识别——版面模型没有这些类别。
        if enabled("frontmatter"):
            frontmatter.reset()
            frontmatter.install()

        # Accents are folded back onto their base letters before formulas are
        # parsed, because formula grouping runs one line at a time.
        # 重音要在公式识别之前并回基字母，因为公式分组是逐行进行的。
        if enabled("accents"):
            accents.install()

        # Reference brackets are rejoined after formula parsing, so a citation
        # reaches the translator as one placeholder.
        # 引用方括号在公式识别之后合并，使一条引文以单个占位符送进翻译。
        if enabled("brackets"):
            brackets.install()

        # A comma inside a subscript is part of the symbol, not punctuation the
        # translator should localise. Installed alongside the bracket merge,
        # because both decide what a formula placeholder covers.
        # 下标里的逗号属于符号本身，不是该由译者本地化的标点。与引用括号合并装在
        # 一起，因为两者都在决定公式占位符的范围。
        if enabled("notation"):
            notation.install()

        # A formula's fraction and radical rules are zero-height strokes, which
        # the containment test scores at zero and never assigns to the formula.
        # Installed before typesetting, which is where the consequence shows.
        # 公式里的分数线与根号线是零高度的描边，归属判定给它们打零分，于是永远不被算进
        # 公式。装在排版之前，因为后果是在排版阶段显现的。
        if enabled("hairlines"):
            hairlines.install()

        # A shattered inline formula is put back together before the curves are
        # assigned to formulas, because the assignment reads the boxes this
        # produces.
        # 被切碎的行内公式在给曲线定归属之前重新拼合，因为归属判定读的正是这里产出的框。
        # An enlarged opening letter reads as a superscript to the character
        # after it, which preserves that whole word untranslated beside its own
        # translation. Must precede the formula stage, which is where the
        # classification happens.
        # 放大的首字母会让它后面那个字符被判成上标，于是那整个词以原文形态和它自己的译文
        # 并排留在页面上。必须排在公式步骤之前，分类就发生在那里。
        if enabled("drop_caps"):
            dropcap.install()

        # Small capitals are two sizes of one font, and the smaller one wins the
        # mode, so a heading set in them is translated a fifth too small. Same
        # stage as the initials above, and for the same reason.
        # 小型大写是同一字体的两种字号，较小的那种会赢得众数，于是用它排的标题译过来会小
        # 五分之一。和上面的首字母同处一个步骤，理由相同。
        if enabled("small_caps"):
            smallcaps.install()


        # Inline mathematics arrives as one-character runs of ordinary text —
        # italic variables the formula detector never claims, and marks a line
        # break separated from their base. Must precede the formula stage,
        # which is where the tagging happens.
        # 行内数学是以一个个单字符普通文字的形式到达的——公式识别从不认领的斜体变量，
        # 以及被换行与基字母切开的角标。必须排在公式步骤之前，标记就发生在那里。
        if enabled("inline_math"):
            inlinemath.install()

        # A formula the model forgets to return is gone, and nothing checks.
        # The more of a formula the step above protects, the more placeholders
        # a paragraph carries, so this guard is what makes that safe.
        # 模型忘了送回来的公式就没了，而且没有任何一处会检查。上面那一步保护得越多，
        # 一段里的占位符就越多，因此正是这道关卡让那一步变得安全。
        if enabled("placeholder_guard"):
            placeholders.install()


        if enabled("formulas"):
            formulas.install()

        # One cycle collection as each stage begins, so the native buffers behind
        # freed wrappers are handed back before the next stage allocates.
        # 每进入一个阶段先回收一次循环垃圾，把已释放包装对象背后的原生缓冲还回去，
        # 免得下一阶段在它们之上继续增长。
        reclaim.install()

        # Growing a paragraph's box is tried before shrinking its type.
        # 缩字号之前先尝试把段落框扩开。
        if enabled("scaling"):
            scaling.install()

        # A title's box is only as wide as its English needed, so a translation
        # six points too long wraps for one character. Must be installed after
        # scaling: both wrap the same stage, and the box has to be widened
        # before scaling reads it, which means being the outer wrapper.
        # 标题框只有英文所需的那么宽，于是译文多出六个点就会为了一个字换行。必须装在
        # scaling 之后：两者包的是同一个步骤，而框要在 scaling 读它之前就加宽，也就是
        # 说本模块得在外层。
        if enabled("title_boxes"):
            titlebox.install()

        # An ideographic full stop after a figure number is an em wide with the
        # dot at its left, so it leaves a six-point hole between the label and
        # the caption. Runs as the translation is written back.
        # 图注编号后的全角句号占满一个字宽而点在左侧，于是标签与正文之间留下六个点的空洞。
        # 在译文写回段落时执行。
        if enabled("caption_labels"):
            captionlabel.install()


        # A word with no break points inside it — a URL — is wider than the
        # column it has to go in, and the line-break look-ahead then wraps once
        # per character until the remainder fits.
        # 一个内部没有断点的词（比如网址）比它所在的栏还宽，换行前瞻就会每放一个字符
        # 换一次行，直到剩下的部分放得下为止。
        if enabled("long_tokens"):
            longtoken.install()

        # Needs the resolved settings, so it is installed after them.
        # 依赖已解析的设置，因此放在设置之后安装。

        settings = Settings.model_validate(spec.settings)

        # Before BabelDOC constructs the layout model. On macOS it builds that
        # model on CoreML, which costs about a gigabyte resident for a 72 MB
        # file — see engine/layout_model.py for the measurements.
        # 必须早于 BabelDOC 构造版面模型。在 macOS 上它会把模型建在 CoreML 上，
        # 一个 72 MB 的文件要占掉约 1 GB 常驻内存——测量数据见 engine/layout_model.py。
        layout_model.install(settings.low_memory)

        # Before the font mapper is constructed, which reads the family table
        # once and keeps what it found.
        # 必须早于字体映射器构造：它只读一次字体族表，之后就用读到的那份。
        # Italic is emphasis, not calligraphy: unless a script font was named,
        # the body face is registered a second time and slanted at render time.
        # 斜体表示的是强调而非书法：除非指定了手写体字体，否则把正文字体再注册一份，
        # 在绘制时倾斜写出。
        if settings.pdf.script_font:
            fonts.install(settings.pdf.script_font,
                          lang_out=settings.translation.lang_out,
                          serif=settings.pdf.script_font_serif)
        elif enabled("oblique"):
            # The face is registered after high_level.init(), which is what puts
            # the fonts in the cache; the second registration needs a real file
            # of its own and therefore needs the first one to be there.
            # 该字体档在 high_level.init() 之后注册——正是它把字体放进缓存的；第二次
            # 注册需要自己的实体文件，也就必须等第一份先落地。
            oblique.install()

        # A sentence the page break cut in two is rebalanced across the break
        # before either half is translated.
        # 被分页切断的句子，在两半各自翻译之前先重新划分分界。
        crosspage.install(settings.pdf.rebalance_page_breaks)

        queue.put({"type": EV_STARTED, "job_id": spec.job_id})

        high_level.init()

        # Now that the fonts are in the cache, the body face can be registered a
        # second time as the slanted one. Before init there is nothing to copy.
        # 字体已进入缓存，此时才能把正文字体再注册一份作为倾斜档。init 之前无从拷贝。
        if not settings.pdf.script_font and enabled("oblique"):
            oblique.install_face(settings.translation.lang_out)

        # Last thing before the font mapper is built, because it has to see the
        # script tier as this run finally left it.
        #
        # BabelDOC's family table is per-language, but it then appends every
        # other language's faces to every family so that any character at all
        # can be drawn by something. A Simplified Chinese target therefore lists
        # 34 files, and the mapper opens all 34: 323 MB fetched to disk, 253 MB
        # resident, of which the Japanese, Korean, Hong Kong and Traditional
        # faces are never consulted — every character they could draw is already
        # in the Simplified faces ahead of them in the list.
        # 必须放在字体映射器构造之前的最后一步，因为它要看到本次运行最终确定的 script 档。
        #
        # BabelDOC 的字体族表本来是按语言分的，但它随后又把每一种其它语言的字体追加进
        # 每一个族，好让任何字符都有东西能画。于是简体中文目标会列出 34 个文件，映射器
        # 把 34 个全部打开：磁盘取 323 MB，常驻 253 MB，其中日文、韩文、港版和繁体那些
        # 从来轮不到——它们能画的字符，排在前面的简体字体早就有了。
        fontset.install_fallback()
        fontset.restrict(settings.translation.lang_out,
                         settings.translation.lang_in)

        # Must be installed before BabelDOC constructs its translator pipeline.
        # 必须在 BabelDOC 构建翻译管线之前安装。
        # The source document's own named destinations say where its
        # bibliography entries and list items are, which is more reliable than
        # inferring either from text.
        # 源文档自带的命名锚点直接给出文献条目与列表项的位置，比从文本推断可靠。
        from local_translator.engine.anchors import DocumentAnchors

        anchors = DocumentAnchors.load(spec.input_file)
        # The master switch has to reach the content-based rules too. Gated only
        # on the label set, turning selective translation off still left every
        # bibliography entry, display equation and pseudocode line in English.
        # 主开关必须同时管到基于内容的判据。此前只按标签集合判断，关掉选择性翻译后
        # 参考文献、行间公式与伪代码仍然保持英文。
        skip_labels = settings.layout_filter.effective_skip_labels()
        filter_cls = None if not enabled("layout_filter") else layout_filter.install(
            skip_labels,
            # The bibliography switch used to be hardcoded on. It is now the
            # "reference" label, which is in both presets, so the behaviour is
            # unchanged — but a reader who wants translated references can now
            # say so, and say it once for both the PDF and the LaTeX path.
            # 参考文献开关以前是写死打开的。现在它就是 reference 这个标签，两个预设里都
            # 有，所以行为不变——但想要翻译参考文献的读者现在说得出口了，而且说一次，
            # PDF 与 LaTeX 两条路都算数。
            skip_bibliography="reference" in skip_labels,
            skip_pseudocode="algorithm" in skip_labels,
            anchors=anchors,
            skip_content=settings.layout_filter.enabled,
        )

        matcher = None
        if spec.glossary_terms and settings.glossary.enabled:
            from local_translator.glossary.matcher import GlossaryMatcher

            matcher = GlossaryMatcher(
                spec.glossary_terms,
                max_terms=settings.glossary.max_terms_per_prompt,
            )

        translator = _build_translator(spec, matcher)
        term_translator, extracting = _build_term_translator(spec, translator)
        config = _build_config(spec, translator, term_translator, extract_terms=extracting)

        watcher = threading.Thread(
            target=_watch_cancel, args=(cancel_event, config), daemon=True
        )
        watcher.start()

        result = asyncio.run(_translate_with_events(config, spec, queue))
        logger.debug("Own memory after BabelDOC: %.0f MiB", _own_memory_mib())
        if not settings.pdf.script_font and enabled("oblique"):
            logger.info("Slanted %d character(s) that were italic in the source",
                        oblique.slanted_count())

        # Repair before publishing, so the file that lands in the user's folder
        # is already complete.
        # 先修复再发布，保证落到用户目录里的就是完整文件。
        from local_translator.engine.images import restore_all

        image_report = {"restored": 0, "pages": 0} if not settings.pdf.restore_images \
            else restore_all(
            spec.input_file, result,
            translate_first=settings.pdf.dual_translate_first,
        )
        logger.debug("Own memory after image restore: %.0f MiB", _own_memory_mib())
        if image_report["restored"]:
            logger.info(
                "Restored %d figure region(s) that page regeneration had dropped",
                image_report["restored"],
            )

        # Stamped before links, so a link rectangle is not covered by a region
        # that lands on top of it afterwards.
        # 在链接修复之前搬运区块，避免链接矩形随后被覆盖上去的区块盖住。
        protected: dict = {}
        if settings.pdf.preserve_regions:
            from local_translator.engine.regions import (
                find_protected_regions,
                stamp_regions,
            )

            # Found once and used twice: the region pass stamps them, and the
            # link pass needs to know which areas are verbatim copies, where a
            # link's original geometry is still exact.
            # 只找一次、用两处：区块流程负责搬运，链接流程需要知道哪些区域是原样复制的
            # ——在那些区域里，链接的原始几何位置依然精确。
            protected = find_protected_regions(
                spec.input_file, settings.pdf.preserve_kinds
            )
            region_report = {"stamped": 0, "pages": 0}
            for kind, path in (("mono_path", False), ("dual_path", True)):
                target = result.get(kind)
                if not target:
                    continue
                counts = stamp_regions(
                    spec.input_file, target, protected, dual=path,
                    translate_first=settings.pdf.dual_translate_first,
                )
                for key, value in counts.items():
                    region_report[key] += value
            logger.debug("Own memory after region stamping: %.0f MiB", _own_memory_mib())
            if region_report["stamped"]:
                logger.info(
                    "Preserved %d pseudocode/table/equation region(s) verbatim",
                    region_report["stamped"],
                )

        # After the verbatim stamp and before the links: the stamp provides the
        # exact source rows to rewrite, and the links are then placed over the
        # rewritten titles.
        # 在原样搬运之后、链接修复之前：搬运提供了待改写的源文行，链接随后覆盖到
        # 改写后的标题上。
        if settings.pdf.translate_contents and translator is not None:
            from local_translator.engine.contents import translate_all as toc_all

            toc_report = toc_all(
                spec.input_file, result, translator.translate,
                translate_first=settings.pdf.dual_translate_first,
            )
            if toc_report["rows"]:
                logger.info("Translated %d contents row(s) across %d page(s)",
                            toc_report["rows"], toc_report["pages"])

        if settings.pdf.restore_links:
            from local_translator.engine.links import restore_all as restore_link_all

            link_report = restore_link_all(
                spec.input_file, result,
                translate_first=settings.pdf.dual_translate_first,
                regions=protected,
            )
            logger.debug("Own memory after link restore: %.0f MiB", _own_memory_mib())
            if link_report["links"]:
                logger.info(
                    "Restored %d hyperlink(s) that annotation stripping had removed "
                    "(%d could not be located, %d glyph(s) painted link blue)",
                    link_report["links"], link_report["dropped"],
                    link_report.get("coloured", 0),
                )

        # Last, because every stage before it adds objects to the file and
        # compressing before they are all in would leave the additions
        # uncompressed.
        # 放在最后：之前每个阶段都会往文件里添加对象，提前压缩会让后加入的部分未经处理。
        if settings.pdf.compress_dpi > 0:
            from local_translator.engine.compress import compress

            for key in ("mono_path", "dual_path"):
                produced = result.get(key)
                if not produced:
                    continue
                report = compress(Path(produced), settings.pdf.compress_dpi)
                if report["images"]:
                    logger.info(
                        "Compressed %s: %.1f MB to %.1f MB (%.0f%% smaller, "
                        "%d image(s) resampled to %d DPI)",
                        Path(produced).name,
                        report["before"] / 1048576, report["after"] / 1048576,
                        report["saved"] * 100, report["images"],
                        settings.pdf.compress_dpi,
                    )

        # Selective translation that was installed but never used is the
        # failure this checks for. It is invisible in the output — the document
        # comes back complete, just with the author list translated and the
        # citations rewritten — and the install log says 「active」 either way,
        # which is how the cloud path stayed broken while reporting success.
        # 「装上了但从未被用到」正是这里要检查的故障。它在产物里不可见——文档看起来是
        # 完整的，只是作者名被译了、引文被改写了——而安装日志无论如何都会说「已启用」，
        # 云端那条路径正是这样一边坏着一边报告成功的。
        if filter_cls is not None and not layout_filter.was_active():
            logger.warning(
                "选择性翻译已配置但整趟没有生效：BabelDOC 没有使用打过补丁的翻译器。"
                "标题、作者、单位与引文可能都被翻译了。"
            )

        # A run that translated nothing must not report success.
        #
        # Every stage can succeed while the text comes back untouched: the
        # window opens, the job runs its two minutes, the progress bar reaches
        # the end, an output PDF appears — and it is the English original with
        # new fonts embedded. Nothing in the log says so, because nothing went
        # wrong in any one step. The reader finds out by opening the file.
        # 一趟什么都没译的运行不能报成成功。
        #
        # 每一个阶段都可以成功而文字原封不动：窗口打开了，任务跑满两分钟，进度条走到头，
        # 产物 PDF 也出来了——里面是嵌了新字体的英文原文。日志里一句提示都没有，因为单看
        # 每一步都没出错。读者只能靠打开文件才发现。
        prose, translated = _translation_reached(spec)
        if prose >= _ENOUGH_TO_JUDGE and translated == 0:
            raise RuntimeError(
                f"这一趟没有译出任何内容：{prose} 个成句段落全部与原文相同。"
                "产物已生成但等于原文，因此判为失败。"
                "常见原因是本地模型没有真正应答（检查 logs/llama-server.log 里有没有请求记录），"
                "或者同时装了多份程序、启动到了旧的那一份。"
            )
        if prose >= _ENOUGH_TO_JUDGE and translated < prose // 4:
            logger.warning(
                "Only %d of %d prose paragraphs came back translated; the output "
                "is mostly the source text", translated, prose,
            )

        result = _publish_outputs(spec, result)

        usage = translator.usage_summary() if hasattr(translator, "usage_summary") else {}
        queue.put(
            {
                "type": EV_FINISH,
                "job_id": spec.job_id,
                **result,
                "tracking_path": _tracking_path(spec),
                "usage": usage,
                "images_restored": image_report.get("restored", 0),
                "skipped": layout_filter.skipped_summary(filter_cls),
                "duration_ms": int((time.monotonic() - started) * 1000),
            }
        )

    except Exception as exc:  # noqa: BLE001 - the boundary must not leak
        cancelled = cancel_event.is_set()
        queue.put(
            {
                "type": EV_ERROR,
                "job_id": spec.job_id,
                "cancelled": cancelled,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "usage": translator.usage_summary()
                if translator is not None and hasattr(translator, "usage_summary")
                else {},
                "duration_ms": int((time.monotonic() - started) * 1000),
            }
        )

    finally:
        _clean_up(spec)
        _exit_now(queue)


def _clean_up(spec: JobSpec) -> None:
    """
    中文：任务结束后清掉中间产物，除非设置里要求保留。

    Clear the job's intermediates on the way out.

    Runs in the worker, after the outputs have been moved to their destination
    and before the process exits, which is the one moment when BabelDOC is
    demonstrably finished and nothing else has started. Doing it in the parent
    instead would race a worker that has reported its result but not yet
    released its files.

    Every failure is swallowed. This runs after the job's outcome has already
    been reported, so there is no longer anything it could usefully change: a
    translation that succeeded did not fail because a temporary file could not
    be unlinked.

    Args:
        spec (JobSpec): The job that has just ended, successfully or not.

    Returns:
        None.
    """
    try:
        from local_translator.config import Settings

        if Settings.model_validate(spec.settings).translation.keep_intermediates:
            return
    except Exception:  # noqa: BLE001
        logger.debug("Could not read the cleanup setting; cleaning up", exc_info=True)

    try:
        from local_translator.engine import intermediates

        intermediates.sweep(spec.working_dir)
        if spec.latex_source:
            workspace = intermediates.workspace_of(spec.latex_source)
            if workspace is not None:
                intermediates.sweep_latex(workspace)
    except Exception:  # noqa: BLE001
        logger.debug("Could not clean up after job %s", spec.job_id, exc_info=True)


def _exit_now(queue) -> int:
    """
    中文：冲刷事件队列后强制退出子进程。

    Flush the event queue and terminate the worker process immediately.

    BabelDOC leaves behind non-daemon threads and an internal process pool that
    keep the interpreter alive well after the translation itself is done, so a
    worker that simply returns from :func:`run_job` can linger indefinitely
    holding a scheduler slot. The worker has no state worth unwinding once its
    terminal event is reported, so it flushes the queue and calls ``os._exit``.

    Flushing first is essential: ``multiprocessing.Queue`` hands items to a
    background feeder thread, and exiting before that thread has written them to
    the pipe would drop the very event the parent is waiting for.

    Args:
        queue (multiprocessing.Queue): The event channel to flush.

    Returns:
        int: Never returns; the annotation exists only to document intent.
    """
    try:
        queue.close()
        queue.join_thread()
    except Exception:
        logger.debug("Event queue flush failed on exit", exc_info=True)

    try:
        import babeldoc.format.pdf.high_level as high_level

        high_level.close_process_pool()
    except Exception:
        logger.debug("close_process_pool failed on exit", exc_info=True)

    os._exit(0)


def _publish_outputs(spec: JobSpec, result: dict[str, Any]) -> dict[str, Any]:
    """
    中文：把暂存目录里的产物搬到目标目录，同名时自动加版本后缀。

    Move staged outputs into their final destination without overwriting.

    BabelDOC writes into a staging directory under ``working_dir`` (see
    :func:`_build_config`) precisely so this step can apply the project's
    no-clobber rule: if ``paper.zh.dual.pdf`` already exists, the new file
    becomes ``paper.zh.dual.v2.pdf`` rather than replacing an earlier
    translation that may have taken an hour of local inference to produce.

    Args:
        spec (JobSpec): Job description providing the real output directory.
        result (dict[str, Any]): Staged paths under ``dual_path``, ``mono_path``
            and ``glossary_path``; values may be ``None``.

    Returns:
        dict[str, Any]: The same keys, updated to the published locations. A key
            whose file could not be moved keeps its staged path so the output is
            still reachable rather than reported as missing.
    """
    import shutil

    from local_translator.library import unique_path

    destination = Path(spec.output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    published = dict(result)
    for key in ("dual_path", "mono_path", "glossary_path"):
        staged = result.get(key)
        if not staged:
            continue
        staged_path = Path(staged)
        if not staged_path.exists():
            published[key] = None
            continue

        target = unique_path(destination / staged_path.name)
        try:
            # shutil.move handles the cross-filesystem case, which happens
            # whenever the source PDF lives on an external volume.
            # shutil.move 能处理跨文件系统的情况（源 PDF 在外置卷时会发生）。
            shutil.move(str(staged_path), str(target))
            published[key] = str(target)
            if target.name != staged_path.name:
                logger.info("Output %s already existed; wrote %s", staged_path.name, target.name)
        except OSError:
            logger.warning("Could not move %s to %s", staged_path, target, exc_info=True)
            published[key] = str(staged_path)

    return published


# A paragraph shorter than this is a caption, a label or an axis title, and is
# skipped or left alone often enough that it says nothing about whether the run
# worked. Only paragraphs of running prose are counted.
# 短于这个长度的段落是图题、标签或坐标轴文字，本来就常被跳过或原样保留，说明不了这一趟跑
# 得对不对。只统计成句的段落。
_PROSE = 40

# How long to wait for the local server's health endpoint before giving up on
# it. A server that is up answers this in single-digit milliseconds and one that
# is down refuses the connection instantly; the only case this bound covers is a
# host that neither answers nor refuses.
# 等待本地服务健康检查的上限。服务在跑时答复只需个位数毫秒，服务没跑时连接会被立即拒绝；
# 这个上限唯一覆盖的情形是主机既不答复也不拒绝。
_PROBE_TIMEOUT = 3.0

# How many prose paragraphs a document needs before silence is evidence. A
# slide deck or a page of figures legitimately has almost nothing to translate,
# and must not be reported as a failure.
# 一份文档要有多少个成句段落，"一段没译"才算得上证据。一份幻灯片或一页全是插图的稿子本
# 来就几乎没有可翻的东西，不能报成失败。
_ENOUGH_TO_JUDGE = 10


def _translation_reached(spec: JobSpec) -> tuple[int, int]:
    """
    中文：数一数这一趟到底译了多少段——用 BabelDOC 自己的逐段记录。

    Count how many prose paragraphs actually came back changed.

    Reads the tracking file BabelDOC writes, which records every paragraph's
    input and output. A paragraph counts as translated when its output is
    non-empty and differs from its input.

    Args:
        spec (JobSpec): Job description, used to locate the tracking file.

    Returns:
        tuple[int, int]: ``(prose, translated)`` — how many paragraphs held
            running prose, and how many of those came back different. Both zero
            when no tracking file was written, which is not evidence of
            anything and is handled by the caller.
    """
    path = _tracking_path(spec)
    if not path:
        return 0, 0
    try:
        record = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        logger.debug("Could not read the tracking file at %s", path, exc_info=True)
        return 0, 0

    prose = translated = 0
    for page in record.get("page") or []:
        for paragraph in page.get("paragraph") or []:
            source = (paragraph.get("input") or "").strip()
            output = (paragraph.get("output") or "").strip()
            if len(source) < _PROSE:
                continue
            prose += 1
            if output and output != source:
                translated += 1
    return prose, translated


def _tracking_path(spec: JobSpec) -> str | None:
    """
    中文：定位 BabelDOC 生成的 translate_tracking.json（含逐段原文/译文）。

    Locate the paragraph-level bilingual tracking file BabelDOC writes.

    BabelDOC emits ``translate_tracking.json`` whenever ``working_dir`` is set
    (see ``ILTranslator.translate``), placing it inside a subdirectory named
    after the input file's stem. It records the input and output text of every
    paragraph, which is what powers full-text search and glossary mining.

    Args:
        spec (JobSpec): Job description providing the working directory.

    Returns:
        str | None: Absolute path to the tracking file, or ``None`` if BabelDOC
            did not produce one (for example when translation was skipped).
    """
    base = Path(spec.working_dir)
    candidates = sorted(base.rglob("translate_tracking.json"))
    return str(candidates[0]) if candidates else None
