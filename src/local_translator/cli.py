"""
中文：命令行入口，提供环境安装、模型管理与服务启动等子命令。

Command line interface for Local Translator.

Subcommands:
    setup    Download the llama.cpp runtime and Hy-MT2 model weights.
    doctor   Report installation status and resolved paths.
    model    Start, stop, probe and smoke-test the local inference server.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sys
import time

import typer
from tqdm import tqdm

from local_translator import edition
from local_translator.config import LAYOUT_PRESETS, get_paths, get_settings, save_settings
from local_translator.llm.downloader import (
    SUPPORTED_QUANTS,
    find_installed_models,
    install_llama_cpp,
    install_model,
    llama_server_path,
    resolve_model_file,
)
from local_translator.llm.supervisor import LlamaServerError, LlamaServerSupervisor

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Local Translator — 本地 PDF 翻译与双语阅读器",
)
model_app = typer.Typer(no_args_is_help=True, help="本地模型服务管理 / Manage the local model server")
app.add_typer(model_app, name="model")


def _setup_logging(verbose: bool = False) -> None:
    """
    中文：配置日志级别并抑制第三方库噪声。

    Configure root logging for CLI use.

    Args:
        verbose (bool): Emit DEBUG-level records instead of INFO.
    """
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ("httpx", "httpcore", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    # Worker processes are spawned fresh and read this to match the level asked
    # for here; see engine/worker.run_job.
    # worker 是新起的进程，靠这个环境变量对齐这里要求的级别，见 worker.run_job。
    os.environ["LOCALTRANS_LOG_LEVEL"] = "DEBUG" if verbose else "INFO"


def _progress_bar(desc: str):
    """
    中文：构造一个用于下载进度的回调，返回 (回调, 关闭函数)。

    Build a tqdm-backed download progress callback.

    The bar's total is set lazily on the first callback, because the total size
    is only known once the HTTP response headers arrive.

    Args:
        desc (str): Label shown to the left of the bar.

    Returns:
        tuple[Callable[[int, int], None], Callable[[], None]]: The progress
            callback to hand to the downloader, and a closer to call afterwards.
    """
    bar = tqdm(unit="B", unit_scale=True, unit_divisor=1024, desc=desc, leave=True)

    def callback(done: int, total: int) -> None:
        if total and bar.total != total:
            bar.total = total
        bar.n = done
        bar.refresh()

    return callback, bar.close


# --------------------------------------------------------------------------- #
# setup / doctor
# --------------------------------------------------------------------------- #


@app.command()
def init(
    check: bool = typer.Option(False, "--check",
                               help="只报告配置状态，不做改动 / Report only"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """
    中文：首次运行的配置向导——云端 API、语言、后端与模型下载，一次问完。

    Configure a fresh installation interactively.

    Asks for the OpenAI-compatible endpoint and its key, verifies the pair with
    one real request, stores the key in the system keychain and writes the rest
    to ``settings.json``. Re-runnable: every question defaults to the current
    setting.
    """
    _setup_logging(verbose)
    from local_translator import wizard

    raise typer.Exit(code=wizard.run(non_interactive=check))


@app.command()
def setup(
    quant: str = typer.Option(
        "Q6_K",
        "--quant",
        "-q",
        help=f"模型量化档位 / Model quantisation. One of: {', '.join(SUPPORTED_QUANTS)}",
    ),
    skip_llama: bool = typer.Option(False, "--skip-llama", help="跳过 llama.cpp 安装"),
    skip_model: bool = typer.Option(False, "--skip-model", help="跳过模型权重下载"),
    skip_draft: bool = typer.Option(False, "--skip-draft", help="跳过投机解码 draft 模型"),
    force: bool = typer.Option(False, "--force", help="即使已安装也重新下载 / Reinstall"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """
    中文：一键安装 llama.cpp 运行时与 Hy-MT2 模型权重。

    Install the local inference runtime and model weights.

    Downloads are resumable, so an interrupted setup can simply be re-run. The
    chosen quantisation is written to settings so later commands pick it up.
    """
    _setup_logging(verbose)

    try:
        edition.require_local("下载本地模型")
    except RuntimeError as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(code=2) from exc

    paths = get_paths()
    settings = get_settings()

    if quant not in SUPPORTED_QUANTS:
        typer.secho(
            f"Unknown quantisation {quant!r}. Choose from: {', '.join(SUPPORTED_QUANTS)}",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=2)

    typer.secho(f"Data directory: {paths.data_dir}", fg=typer.colors.CYAN)

    spec = edition.local_model() or {}
    model_repo = spec.get("repo", "tencent/Hy-MT2-7B-GGUF")
    draft_repo = spec.get("draft_repo")
    if quant == "Q6_K" and spec.get("quant"):
        # Q6_K is the option default, so it being set means the caller did not
        # choose — take the edition's own quantisation instead. The 1.8B is not
        # published at Q6_K at all, so keeping the default would 404.
        # Q6_K 是选项的默认值，它出现即表示调用方没有选——改用本版本自己的量化档位。
        # 1.8B 根本没有 Q6_K 这一档，沿用默认会 404。
        quant = spec["quant"]
    typer.secho(f"Edition: {edition.display_name()}", fg=typer.colors.CYAN)

    if not skip_llama:
        typer.secho("\n[1/3] Installing llama.cpp runtime…", fg=typer.colors.CYAN, bold=True)
        cb, close = _progress_bar("llama.cpp")
        try:
            binary = install_llama_cpp(paths, progress=cb, force=force)
        finally:
            close()
        typer.secho(f"  ✓ {binary}", fg=typer.colors.GREEN)

    if not skip_model:
        typer.secho(f"\n[2/3] Downloading weights from {model_repo} ({quant})…",
                    fg=typer.colors.CYAN, bold=True)
        filename, size = resolve_model_file(quant, repo=model_repo)
        typer.echo(f"  {filename}  ({size / 1024**3:.2f} GB)")
        cb, close = _progress_bar(quant)
        try:
            weights = install_model(paths, quant, repo=model_repo, progress=cb, force=force)
        finally:
            close()
        typer.secho(f"  ✓ {weights}", fg=typer.colors.GREEN)

        settings.local_model.model_file = weights.name
        save_settings(settings)

    if not skip_draft and draft_repo:
        # The 1.8B sibling acts as a speculative-decoding draft model: it
        # proposes tokens that the 7B verifies, which is lossless and worth a
        # few percent. At ~1.1 GB it is cheap enough to install by default.
        # In the mini edition the 1.8B *is* the model, so there is no draft to
        # download and `draft_repo` is None.
        # 1.8B 同族模型用作投机解码的 draft，无损且能省几个百分点；1.1GB 值得默认安装。
        # 本地轻量版里 1.8B 本身就是主模型，没有 draft 可下，draft_repo 为 None。
        typer.secho("\n[3/3] Downloading 1.8B draft model (speculative decoding)…",
                    fg=typer.colors.CYAN, bold=True)
        cb, close = _progress_bar("draft")
        try:
            draft = install_model(paths, spec["draft_quant"], repo=draft_repo,
                                  progress=cb, force=force)
        finally:
            close()
        typer.secho(f"  ✓ {draft}", fg=typer.colors.GREEN)
        settings.local_model.draft_model_file = draft.name
        save_settings(settings)

    typer.secho("\nSetup complete. Next: `localtrans model test`", fg=typer.colors.GREEN, bold=True)


@app.command()
def doctor(verbose: bool = typer.Option(False, "--verbose", "-v")) -> None:
    """
    中文：检查安装状态、路径与服务可用性。

    Report resolved paths and the status of every external dependency.
    """
    _setup_logging(verbose)
    paths = get_paths()
    settings = get_settings()

    def row(label: str, value: str, ok: bool | None = None) -> None:
        mark = "" if ok is None else ("✓ " if ok else "✗ ")
        color = None if ok is None else (typer.colors.GREEN if ok else typer.colors.RED)
        typer.secho(f"  {mark}{label:<22} {value}", fg=color)

    typer.secho("Paths", bold=True)
    row("data dir", str(paths.data_dir), paths.data_dir.exists())
    row("library dir", str(paths.library_dir), paths.library_dir.exists())
    row("model dir", str(paths.models_dir), paths.models_dir.exists())
    row("database", str(paths.db_path), paths.db_path.exists())
    row("settings", str(paths.settings_path), paths.settings_path.exists())

    typer.secho("\nRuntime", bold=True)
    binary = llama_server_path(paths)
    row("llama-server", str(binary) if binary else "not installed", binary is not None)

    def sup_draft_path() -> str:
        d = LlamaServerSupervisor(paths, settings.local_model).draft_model_path()
        return d.name if d else ""

    models = find_installed_models(paths)
    if models:
        for m in models:
            row("model", f"{m.name}  ({m.stat().st_size / 1024**3:.2f} GB)", True)
    else:
        row("model", "none downloaded — run `localtrans setup`", False)

    draft = sup_draft_path()
    row("draft model", draft or "not installed (speculative decoding off)", bool(draft))

    typer.secho("\nInference server", bold=True)
    sup = LlamaServerSupervisor(paths, settings.local_model)
    healthy = sup.health()
    row("endpoint", sup.base_url, healthy)
    row("status", "healthy" if healthy else "not running", healthy)

    typer.secho("\nConfiguration", bold=True)
    row("backend", settings.translation.backend)
    row("languages", f"{settings.translation.lang_in} → {settings.translation.lang_out}")
    row("selected model", settings.local_model.model_file)
    row("dual layout", "alternating" if settings.pdf.use_alternating_pages_dual else "side-by-side")
    row("layout preset", settings.layout_filter.preset)

    if not binary or not models:
        typer.secho("\nRun `localtrans setup` to finish installation.", fg=typer.colors.YELLOW)
        raise typer.Exit(code=1)


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #


@model_app.command("start")
def model_start(
    wait: float = typer.Option(300.0, help="等待模型加载的秒数 / Startup timeout in seconds"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """
    中文：启动本地 llama-server 并等待就绪（前台运行，Ctrl-C 退出）。

    Start the local inference server in the foreground and keep it running.
    """
    _setup_logging(verbose)
    paths = get_paths()
    settings = get_settings()
    sup = LlamaServerSupervisor(paths, settings.local_model)

    try:
        sup.start(wait_timeout=wait)
    except LlamaServerError as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc

    typer.secho(f"llama-server ready at {sup.base_url}", fg=typer.colors.GREEN)
    if not sup.owns_process:
        typer.echo("(adopted an already-running server; leaving it alone)")
        return

    typer.echo("Press Ctrl-C to stop.")
    try:
        while sup.is_running():
            time.sleep(1.0)
    except KeyboardInterrupt:
        typer.echo("\nStopping…")
    finally:
        sup.stop()


@model_app.command("status")
def model_status() -> None:
    """
    中文：查询本地推理服务是否就绪。

    Report whether the local inference server is reachable and ready.
    """
    _setup_logging()
    settings = get_settings()
    sup = LlamaServerSupervisor(get_paths(), settings.local_model)
    if sup.health():
        typer.secho(f"healthy — {sup.base_url}", fg=typer.colors.GREEN)
    else:
        typer.secho(f"not running — {sup.base_url}", fg=typer.colors.RED)
        raise typer.Exit(code=1)


@model_app.command("test")
def model_test(
    text: str = typer.Option(
        "The attention mechanism {v1} scales as O(n^2), where n is the sequence length.",
        "--text",
        "-t",
        help="待翻译的测试文本 / Text to translate",
    ),
    lang_out: str = typer.Option("zh", "--to", help="目标语言 / Target language code"),
    start: bool = typer.Option(True, "--start/--no-start", help="必要时自动启动服务"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """
    中文：冒烟测试本地模型，检查译文质量与 {v1} 占位符是否被保留。

    Smoke-test the local model end to end.

    Beyond printing the translation, this checks that BabelDOC-style formula
    placeholders such as ``{v1}`` survive the round trip — the single most
    important property for a translator feeding a PDF layout engine, since a
    dropped placeholder means a lost formula in the rendered output.
    """
    _setup_logging(verbose)
    paths = get_paths()
    settings = get_settings()
    sup = LlamaServerSupervisor(paths, settings.local_model)

    try:
        if start:
            sup.ensure_running()
        elif not sup.health():
            typer.secho("Server not running (use --start to launch it)", fg=typer.colors.RED)
            raise typer.Exit(code=1)
    except LlamaServerError as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc

    # Imported lazily so `localtrans setup` works before BabelDOC is installed.
    # 延迟导入，保证 BabelDOC 尚未安装时 setup 命令仍可用。
    from local_translator.engine.translators import LocalMTTranslator

    translator = LocalMTTranslator(
        lang_in=settings.translation.lang_in,
        lang_out=lang_out,
        settings=settings.local_model,
        ignore_cache=True,
    )

    typer.secho("\nInput:", bold=True)
    typer.echo(f"  {text}")

    t0 = time.monotonic()
    result = translator.translate(text)
    elapsed = time.monotonic() - t0

    typer.secho("\nOutput:", bold=True)
    typer.echo(f"  {result}")

    placeholders = re.findall(r"\{v\d+\}", text)
    missing = [p for p in placeholders if p not in result]

    typer.secho("\nChecks:", bold=True)
    if placeholders:
        if missing:
            typer.secho(f"  ✗ placeholders lost: {', '.join(missing)}", fg=typer.colors.RED)
        else:
            typer.secho(f"  ✓ placeholders preserved: {', '.join(placeholders)}", fg=typer.colors.GREEN)
    typer.echo(f"  · latency        {elapsed:.2f}s")
    typer.echo(
        f"  · tokens         in {translator.prompt_token_count.value} / "
        f"out {translator.completion_token_count.value}"
    )
    if elapsed > 0 and translator.completion_token_count.value:
        typer.echo(
            f"  · throughput     {translator.completion_token_count.value / elapsed:.1f} tok/s"
        )

    if missing:
        raise typer.Exit(code=1)


# --------------------------------------------------------------------------- #
# translate
# --------------------------------------------------------------------------- #


def _format_usage(usage: dict, duration_ms: int | None) -> list[str]:
    """
    中文：把 token 用量格式化成可读的小字行。

    Render a usage summary as human-readable lines.

    Args:
        usage (dict): Output of ``translator.usage_summary()``.
        duration_ms (int | None): Wall-clock duration in milliseconds.

    Returns:
        list[str]: Formatted lines, ready to print or show under a result card.
    """

    def k(n: int) -> str:
        return f"{n / 1000:.1f}k" if n >= 1000 else str(n)

    lines = [
        f"输入 {k(usage.get('prompt_tokens', 0))} / "
        f"输出 {k(usage.get('completion_tokens', 0))} / "
        f"合计 {k(usage.get('total_tokens', 0))} tokens"
    ]

    total = usage.get("para_total", 0)
    hits = usage.get("para_cache_hit", 0)
    parts = [f"段落缓存命中 {hits}/{total}"] if total else []

    if duration_ms:
        seconds = duration_ms / 1000
        parts.append(
            f"耗时 {int(seconds // 60)}分{int(seconds % 60)}秒"
            if seconds >= 60
            else f"耗时 {seconds:.1f}秒"
        )

    failures = usage.get("placeholder_failures", 0)
    retries = usage.get("placeholder_retries", 0)
    if retries or failures:
        parts.append(f"占位符重试 {retries}，回退原文 {failures}")

    dropped = usage.get("style_dropped", 0)
    if dropped:
        parts.append(f"样式标记丢失 {dropped}（译文保留）")

    implausible = usage.get("implausible_output", 0)
    if implausible:
        parts.append(f"输出不合理 {implausible}（已弃用，保留原文）")

    if parts:
        lines.append(" · ".join(parts))
    return lines


async def _run_translate(
    files: list, pages: str | None, backend: str | None, wait_model: bool,
    translate_contents: bool | None = None,
    translate_figures: bool | None = None,
    preset: str | None = None,
) -> int:
    """
    中文：注册文档、提交任务并等待全部完成，返回失败数量。

    Register documents, submit jobs and wait for them to settle.

    Runs the same :class:`JobManager` the web server uses, so the CLI exercises
    the real execution path rather than a simplified one.

    Args:
        files (list[Path]): Source PDFs to translate.
        pages (str | None): BabelDOC page range expression.
        backend (str | None): Override the configured backend.
        wait_model (bool): Start the local inference server if it is not running.
        translate_contents (bool | None): Override the contents-page setting for
            this run only; ``None`` keeps whatever the settings say.
        translate_figures (bool | None): Translate the text inside figures for
            this run only, by taking ``"figure"`` out of the preserve list.
            ``None`` keeps whatever the settings say.
        preset (str | None): Selective-translation preset for this run —
            ``"pdfmath"``, ``"conservative"`` or ``"everything"``. ``everything``
            also clears the verbatim-region list, so tables, figures and
            algorithm blocks are translated rather than put back untouched.

    Returns:
        int: Number of jobs that did not complete successfully.
    """
    from pathlib import Path as _Path

    from local_translator.db import JOB_COMPLETED
    from local_translator.service import get_service

    # Submit through the service, not straight to the manager. The service is
    # where the glossary is loaded and the coverage gate decides whether to
    # extract terms; bypassing it silently translated without terminology at all.
    # 必须走 service 提交：术语库加载与覆盖率闸门都在那里，
    # 直接调 manager 会静默地在没有术语约束的情况下翻译。
    service = get_service()
    if backend:
        service.settings.translation.backend = backend
    if translate_contents is not None:
        service.settings.pdf.translate_contents = translate_contents
    if preset:
        service.settings.layout_filter.preset = preset
        service.settings.layout_filter.skip_labels = list(LAYOUT_PRESETS.get(preset, ()))
        # The label preset alone is not the whole switch. Tables, figures and
        # algorithm blocks are put back verbatim by region stamping, which does
        # not consult it, so "translate everything" has to clear that list too.
        # 只切标签预设并不是完整的开关。表、图与算法块是由区块原样搬回处理的，那条路径
        # 不看这个预设，因此「全部翻译」还必须把那份清单清空。
        if preset == "everything":
            # Pseudocode stays verbatim even here. An algorithm block is not
            # prose in a box: it is numbered lines whose indentation carries the
            # control flow, and re-typesetting it destroys both. Measured on
            # page 4 of arXiv:2608.18004, 「ODEStep」 came back as 「NODES做」,
            # 「end if」 as 「结束 if 语句」 and 「else」 landed on a line of its own
            # in the middle of the block. No setting should make that the right
            # answer, which is the same rule the numeric-cell check follows.
            # 即便在这里，伪代码也保持原样。算法块不是装在框里的散文：它是带编号的行，
            # 缩进本身承载控制流，重排会把两者一起毁掉。arXiv:2608.18004 第 4 页实测：
            # 「ODEStep」回来变成「NODES做」，「end if」变成「结束 if 语句」，「else」
            # 单独落在块中间。没有哪个设置能让这成为正确答案——与数字单元格那条规则同理。
            service.settings.pdf.preserve_kinds = [
                kind for kind in service.settings.pdf.preserve_kinds
                if kind == "algorithm"
            ] or ["algorithm"]
            service.settings.pdf.translate_contents = True
    if translate_figures is not None:
        kinds = [kind for kind in service.settings.pdf.preserve_kinds if kind != "figure"]
        if not translate_figures:
            kinds.append("figure")
        service.settings.pdf.preserve_kinds = kinds

    # Hybrid drives the local model too, so it needs the server up just as much
    # as local mode does. Checking only for "local" left hybrid runs to fail on
    # a connection error the moment a paragraph was routed locally.
    # 混合模式同样会调用本地模型，需要与 local 模式一样先确保服务已启动。只判断
    # "local" 会让混合模式在第一个段落被分流到本地时直接连接失败。
    if service.settings.translation.backend in ("local", "hybrid") and wait_model:
        typer.echo("Ensuring the local model server is running…")
        service.ensure_model_ready()

    database = service.db
    manager = service.manager
    await manager.start()

    job_ids: list[str] = []
    for f in files:
        source = _Path(f).expanduser().resolve()
        if not source.exists():
            typer.secho(f"  ✗ not found: {source}", fg=typer.colors.RED)
            continue
        report = service.assess_coverage(source)
        job_id = service.submit_path(source, pages=pages)
        job_ids.append(job_id)
        typer.echo(
            f"  → queued {source.name}  (job {job_id[:8]}, "
            f"术语覆盖 {float(report['coverage']) * 100:.0f}%)"
        )

    if not job_ids:
        await manager.stop()
        return 1

    bar = tqdm(total=100, desc="translating", unit="%", leave=True)
    failures = 0
    pending = set(job_ids)

    try:
        while pending:
            await asyncio.sleep(0.5)
            snap = manager.snapshot()
            if snap["running"]:
                bar.n = int(snap["running"][0]["progress"])
                bar.set_description(snap["running"][0]["stage"] or "translating")
                bar.refresh()

            for job_id in list(pending):
                if manager.is_active(job_id):
                    continue
                pending.discard(job_id)
                row = database.get_job(job_id)
                if row is None:
                    failures += 1
                    continue
                bar.n = 100
                bar.refresh()
                if row["status"] == JOB_COMPLETED:
                    typer.secho(f"\n  ✓ {row['title']}", fg=typer.colors.GREEN, bold=True)
                    for key, label in (("dual_path", "双语"), ("mono_path", "译文")):
                        if row[key]:
                            typer.echo(f"      {label}  {row[key]}")
                    usage = {
                        "prompt_tokens": row["prompt_tokens"],
                        "completion_tokens": row["completion_tokens"],
                        "total_tokens": row["total_tokens"],
                        "para_total": row["para_total"],
                        "para_cache_hit": row["para_cache_hit"],
                        "placeholder_retries": row["placeholder_retries"],
                        "placeholder_failures": row["placeholder_failures"],
                    }
                    for line in _format_usage(usage, row["duration_ms"]):
                        typer.secho(f"      {line}", fg=typer.colors.BRIGHT_BLACK)
                else:
                    failures += 1
                    typer.secho(
                        f"\n  ✗ {row['title']}: {row['status']} — {row['error']}",
                        fg=typer.colors.RED,
                    )
    finally:
        bar.close()
        await manager.stop()
        # Release the model this command loaded. Only a server this process
        # started is stopped, so a translation run beside an open window leaves
        # that window's server alone — and a run on its own does not leave
        # several gigabytes resident with no watchdog left to release them,
        # since the watchdog lives in the process that is now exiting.
        # 释放本次命令加载的模型。只会停掉本进程启动的服务，因此与已打开的窗口
        # 并行运行时不会动窗口那一个；而单独运行时也不会留下几个 GB 常驻——负责
        # 卸载的看门狗就在这个即将退出的进程里。
        service.supervisor.stop()

    return failures


@app.command()
def translate(
    files: list[str] = typer.Argument(..., help="要翻译的 PDF 文件 / PDF files to translate"),
    pages: str = typer.Option(None, "--pages", "-p", help="页码范围，如 1-5 / Page range"),
    backend: str = typer.Option(None, "--backend", "-b", help="local、api 或 hybrid / Override backend"),
    translate_contents: bool = typer.Option(
        None, "--translate-contents/--keep-contents",
        help="目录页逐行原位翻译 / Rewrite contents rows in place",
    ),
    translate_figures: bool = typer.Option(
        None, "--translate-figures/--keep-figures",
        help="翻译图片内的文字 / Translate the text inside figures",
    ),
    preset: str = typer.Option(
        None, "--preset",
        help="选择性翻译预设：pdfmath / conservative / everything",
    ),
    no_start: bool = typer.Option(False, "--no-start", help="不自动启动本地模型服务"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """
    中文：从命令行翻译 PDF，产物写回源文件所在目录。

    Translate one or more PDFs, writing results next to each source file.
    """
    _setup_logging(verbose)
    failures = asyncio.run(_run_translate(files, pages, backend, wait_model=not no_start,
                                          translate_contents=translate_contents,
                                          translate_figures=translate_figures,
                                          preset=preset))
    if failures:
        raise typer.Exit(code=1)


# --------------------------------------------------------------------------- #
# latex
# --------------------------------------------------------------------------- #


@app.command()
def latex(
    target: str = typer.Argument(
        ..., help="arXiv 编号/链接，或一个 LaTeX 源码文件夹 / arXiv id or a source folder"),
    output: str = typer.Option(None, "--output", "-o", help="产物写到哪个目录"),
    backend: str = typer.Option(None, "--backend", "-b", help="local、api 或 hybrid"),
    font: str = typer.Option(None, "--font", help="中文字体族，默认自动挑选"),
    engine: str = typer.Option(None, "--engine", help="TeX 引擎路径，默认用随包的"),
    workers: int = typer.Option(None, "--workers", "-w", help="同时翻译多少块"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """
    中文：从 LaTeX 源码翻译一篇论文，而不是从印出来的 PDF。

    Translate a paper from its LaTeX source rather than from its printed page.

    Source translation removes a whole class of defect at once: nothing is
    clipped, rescaled or overlapped, because TeX lays the page out from
    scratch. It needs the source, which arXiv provides for most papers, and a
    TeX engine, which the edition either bundles or finds on the machine.
    """
    _setup_logging(verbose)
    if asyncio.run(_run_latex(target, output, backend, font, engine, workers)):
        raise typer.Exit(code=1)


async def _run_latex(target: str, output: str | None, backend: str | None,
                     font: str | None, engine_path: str | None,
                     workers: int | None) -> bool:
    """
    中文：跑一次 LaTeX 源码翻译，返回是否失败。

    Run one source translation.

    Args:
        target (str): An arXiv reference or a local folder.
        output (str | None): Where to put the result.
        backend (str | None): Backend override.
        font (str | None): CJK font family override.
        engine_path (str | None): TeX engine override.
        workers (int | None): Concurrency override.

    Returns:
        bool: ``True`` when the run failed, matching the exit-code convention
            of the other commands.
    """
    from pathlib import Path

    from local_translator.config import get_paths, get_settings
    from local_translator.latex import engine as tex_engine
    from local_translator.latex import pipeline, source
    from local_translator.latex.selection import Selection
    from local_translator.service import (
        build_glossary_for_cli,
        build_translator_for_cli,
        render_translated_image,
        with_protected_terms,
    )

    settings = get_settings()
    paths = get_paths()
    latex_settings = settings.latex
    chosen = tex_engine.resolve(engine_path or latex_settings.engine_path or None,
                                paths.bin_dir)
    if chosen is None:
        typer.echo("✗ 没有找到 TeX 引擎。请安装含 TeX 的版本，或用 --engine 指定路径。")
        return True
    engine_file, is_system = chosen
    typer.echo(f"→ 引擎：{'系统 TeX' if is_system else '随包 Tectonic'} {engine_file}")

    translator = build_translator_for_cli(settings, backend)
    destination = Path(output).expanduser() if output else Path.cwd() / "latex-out"
    destination.mkdir(parents=True, exist_ok=True)

    selection = Selection.from_settings(settings)
    typer.echo(f"→ {selection.describe()}")

    # A second translator for the figures. The first one has had its prompt
    # replaced with the one written for LaTeX source; a recognised line off a
    # chart is not LaTeX and should not be told that it is.
    # 图片用的是另一个翻译器。第一个的提示词已经换成了为 LaTeX 源码写的那一套；从图表上
    # 识别出来的一行字不是 LaTeX，不该被告知它是。
    matcher = build_glossary_for_cli(settings)
    image_translator = None
    if selection.translate_images:
        lines = with_protected_terms(
            build_translator_for_cli(settings, backend).translate, matcher)
        image_translator = lambda data: render_translated_image(data, lines)  # noqa: E731

    options = dict(
        workers=workers or latex_settings.workers,
        font=font or latex_settings.cjk_font or None,
        tectonic=None if is_system else engine_file,
        system_engine=engine_file if is_system else None,
        selection=selection,
        glossary=matcher,
        hard_protect=settings.glossary.hard_protect_whitelist,
        translate_image=image_translator,
        on_progress=lambda done, total: typer.echo(f"\r  翻译 {done}/{total}", nl=False),
    )

    folder = Path(target).expanduser()
    try:
        if folder.is_dir():
            typer.echo(f"→ 本地源码：{folder}")
            outcome = pipeline.translate_folder(folder, destination, translator, **options)
        else:
            from local_translator.sources.arxiv import parse_arxiv_id

            arxiv_id = parse_arxiv_id(target)
            typer.echo(f"→ arXiv {arxiv_id}")
            outcome = await pipeline.translate_arxiv(arxiv_id, destination,
                                                     translator, **options)
    except source.NoSource as problem:
        typer.echo(f"\n✗ {problem}")
        return True

    stats = outcome.stats
    partial = stats.get("partial", 0)
    typer.echo(f"\n  翻译：{stats.get('translated', 0)}/{stats.get('total', 0)} 块被采用"
               f"，{stats.get('refused', 0)} 块保留原文"
               + (f"，{partial} 块部分保留" if partial else ""))
    for reason, count in (stats.get("reasons") or {}).items():
        typer.echo(f"    {count} 块：{reason}")
    if outcome.ok:
        typer.echo(f"✓ {outcome.pdf}（编译 {outcome.seconds:.1f} 秒）")
        return False
    typer.echo(f"✗ 编译失败：{outcome.error}")
    typer.echo(f"  译文源码留在 {outcome.tex}，可据此排查")
    return True


# --------------------------------------------------------------------------- #
# serve
# --------------------------------------------------------------------------- #


glossary_app = typer.Typer(no_args_is_help=True, help="术语库管理 / Manage the cumulative glossary")
app.add_typer(glossary_app, name="glossary")


@glossary_app.command("seed")
def glossary_seed(
    limit: int = typer.Option(3000, "--limit", help="每个 Wikidata 类目的上限"),
    skip_wikidata: bool = typer.Option(False, "--skip-wikidata"),
    skip_jiqizhixin: bool = typer.Option(False, "--skip-jiqizhixin"),
    refresh: bool = typer.Option(
        False, "--refresh",
        help="先删除上次导入的条目再重新导入（手工编辑与自动积累的术语不受影响）",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """
    中文：从 Wikidata、机器之心术语库与内置白名单导入种子术语。

    Populate the glossary from its seed sources. Safe to re-run: existing
    entries are kept, so hand-corrections and learned terms survive.

    ``--refresh`` additionally clears what the previous run of the same source
    imported, which is what makes a stricter filter or a narrowed subject class
    take effect on terms that are already in the glossary.
    """
    _setup_logging(verbose)
    from local_translator.db import get_db
    from local_translator.glossary.seed import seed
    from local_translator.glossary.store import GlossaryStore

    store = GlossaryStore(get_db(get_paths().db_path))
    report = seed(store)

    typer.secho("\n导入结果", bold=True)
    for source, counts in report.items():
        line = (
            f"  {source:<14} 新增 {counts.get('inserted', 0):>6}"
            f"  更新 {counts.get('updated', 0):>5}  跳过 {counts.get('skipped', 0):>6}"
        )
        if counts.get("removed"):
            line += f"  清除 {counts['removed']:>6}"
        typer.echo(line)

    stats = store.stats()
    typer.secho(f"\n术语库共 {stats['total']} 条", fg=typer.colors.GREEN, bold=True)
    typer.echo(f"  按策略  {stats['by_policy']}")
    typer.echo(f"  按学科  {stats['by_domain']}")


@glossary_app.command("snapshot")
def glossary_snapshot(
    output: str = typer.Option("", "--output", "-o",
                               help="写到别处，默认写进源码树 / Write elsewhere"),
) -> None:
    """
    中文：把本机术语库导出成随源码发布的快照，取代所有版本的术语库。

    Export this machine's glossary as the snapshot every edition ships.

    This is the developer half of the arrangement described in
    :mod:`local_translator.glossary.bundled`: a person curates the glossary in
    the interface, runs this, and commits the file. From then on every
    installation of every edition gets exactly that corpus, in tens of
    milliseconds and without touching the network.

    Retired terms are exported too. A row a person disabled is a decision, and
    shipping it is what stops a later import from bringing the term back.

    Args:
        output (str): Destination path. Empty writes to the shipped location
            inside the source tree.

    Raises:
        typer.Exit: With code 1 when run from a packaged application, which has
            no source tree to write into.
    """
    import sys as _sys
    from pathlib import Path as _Path

    from local_translator.db import get_db
    from local_translator.glossary import bundled

    if getattr(_sys, "frozen", False) and not output:
        typer.echo("✗ 打包后的程序没有源码树可写，请在源码仓库里运行，或用 --output 指定路径。")
        raise typer.Exit(code=1)

    target = _Path(output).expanduser() if output else bundled.SNAPSHOT
    rows = get_db(get_paths().db_path).query(
        "SELECT src, tgt, policy, domain, source, confidence, enabled FROM glossary_terms"
    )
    stats = bundled.export(rows, target)
    typer.echo(f"✓ 已导出 {stats['rows']} 条（启用 {stats['enabled']}）到 {target}")
    typer.echo(f"  {stats['bytes'] / 1024:.0f} KB · 摘要 {bundled.digest(target)}")
    typer.echo("  提交这个文件，所有版本的术语库即随之更新。")


@glossary_app.command("stats")
def glossary_stats() -> None:
    """
    中文：查看术语库统计。

    Show glossary size and composition.
    """
    _setup_logging()
    from local_translator.db import get_db
    from local_translator.glossary.store import GlossaryStore

    stats = GlossaryStore(get_db(get_paths().db_path)).stats()
    typer.secho(f"术语库启用中 {stats['total']} 条", bold=True)
    if stats.get("disabled"):
        typer.echo(f"  已停用  {stats['disabled']} 条（清洗停用，记录保留，可在术语库页面恢复）")
    for label, group in (("按策略", "by_policy"), ("按来源", "by_source"), ("按学科", "by_domain")):
        typer.echo(f"  {label}  {stats[group]}")


@glossary_app.command("search")
def glossary_search(
    query: str = typer.Argument("", help="搜索关键词"),
    limit: int = typer.Option(30, "--limit", "-n"),
    disabled: bool = typer.Option(False, "--disabled", help="改看已停用的条目"),
    every: bool = typer.Option(False, "--all", help="启用与停用一起看"),
) -> None:
    """
    中文：搜索术语库，默认只看启用中的条目。

    Search the glossary. Retired entries are hidden unless asked for.
    """
    _setup_logging()
    from local_translator.db import get_db
    from local_translator.glossary.store import GlossaryStore

    enabled = None if every else (not disabled)
    rows = GlossaryStore(get_db(get_paths().db_path)).search(
        query, enabled=enabled, limit=limit)
    if not rows:
        typer.echo("没有匹配的术语")
        return
    for r in rows:
        policy = "保持原文" if r["policy"] == "keep" else r["tgt"]
        mark = "" if r["enabled"] else "  [已停用]"
        typer.echo(
            f"  {r['src']:<34} → {policy:<22} "
            f"[{r['domain'] or '-'}/{r['source'] or '-'}]{mark}"
        )


@glossary_app.command("clean")
def glossary_clean(
    corpus: str = typer.Option(None, "--corpus",
                               help="用于衡量命中率的源语 PDF 目录 / Folder of source-language PDFs"),
    apply: bool = typer.Option(False, "--apply",
                               help="真正停用命中的条目；不加则只出报告 / Disable the entries; otherwise report only"),
    judge: bool = typer.Option(False, "--judge",
                               help="把规则判不了的条目交给 API 判别 / Ask the API model about what rules cannot settle"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """
    中文：清洗术语库——先用规则，规则判不了的再交给 API 判别。默认只出报告。

    Clean the glossary: rules first, a judgement only where they cannot decide.

    Nothing is deleted. Entries are disabled, and can be switched back on from
    the glossary page or with SQL.
    """
    _setup_logging(verbose)
    from local_translator.db import get_db
    from local_translator.glossary import clean as cleaner

    db = get_db(get_paths().db_path)
    terms = [dict(r) for r in db.query(
        "SELECT id, src, tgt, policy, domain, source FROM glossary_terms WHERE enabled=1")]
    typer.secho(f"术语库启用中 {len(terms)} 条", bold=True)

    hits = {}
    if corpus:
        from pathlib import Path as _Path

        paragraphs = cleaner.paragraphs_from(_Path(corpus).expanduser())
        typer.echo(f"  语料 {len(paragraphs)} 段")
        hits = cleaner.exposure(terms, paragraphs)
        total = len(paragraphs) or 1
    else:
        total = 1

    doomed: list[dict] = []
    for rule, caught in cleaner.apply_rules(terms).items():
        reach = sum(hits.get(term["src"], 0) for term in caught)
        share = f"，命中 {reach} 段（{reach / total:.0%}）" if corpus else ""
        typer.secho(f"\n[{rule}] {len(caught)} 条{share}", bold=True)
        for term in sorted(caught, key=lambda x: -hits.get(x["src"], 0)):
            count = f"{hits.get(term['src'], 0):>5}  " if corpus else ""
            typer.echo(f"  {count}{term['src']:<24} → {term['tgt']:<20} [{term['source']}]")
        doomed.extend(caught)

    shortlist = cleaner.shortlist(terms)
    typer.secho(f"\n[规则判不了，待判别] {len(shortlist)} 条", bold=True)
    if judge and shortlist:
        complete = _judge_client()
        if complete is None:
            typer.secho("  没有可用的 API 模型，跳过判别", fg=typer.colors.YELLOW)
        else:
            verdicts = cleaner.judge(shortlist, complete)
            dropped = [t for t in shortlist if verdicts.get(t["id"]) == "drop"]
            typer.echo(f"  判为应删除 {len(dropped)} 条 / 已答复 {len(verdicts)} 条")
            for term in dropped:
                typer.echo(f"    {term['src']:<24} → {term['tgt']:<20} [{term['source']}]")
            doomed.extend(dropped)
    elif shortlist:
        typer.echo("  加 --judge 才会送去判别；否则一律保留")

    if not doomed:
        typer.secho("\n没有需要停用的条目", fg=typer.colors.GREEN)
        return
    if not apply:
        typer.secho(f"\n以上 {len(doomed)} 条会被停用。加 --apply 才真正写库。",
                    fg=typer.colors.YELLOW)
        return

    ids = [term["id"] for term in doomed]
    with db.write() as con:
        con.execute(
            f"UPDATE glossary_terms SET enabled=0, updated_at=datetime('now') "
            f"WHERE id IN ({','.join('?' * len(ids))})", ids)
    typer.secho(f"\n已停用 {len(ids)} 条（enabled=0，可随时改回 1）", fg=typer.colors.GREEN)


@glossary_app.command("keep")
def glossary_keep(
    terms: list[str] = typer.Argument(..., help="要保持英文原样的词 / Terms to keep in English"),
    domain: str = typer.Option("", "--domain", "-d",
                               help="学科：ai / cs / physics 等"),
) -> None:
    """
    中文：把若干词设为「保持英文」，并清掉它们已有的中文译法。

    Mark terms as keep-in-English, displacing any rendering they already have.

    A term that already carries a Chinese rendering is converted rather than
    duplicated, and its other rows are retired — otherwise the matcher would
    offer the translator both a keep entry and a translation for the same word.
    """
    _setup_logging()
    from local_translator.db import get_db
    from local_translator.glossary.store import GlossaryStore

    store = GlossaryStore(get_db(get_paths().db_path))
    counts: dict[str, int] = {}
    for term in terms:
        cleaned = term.strip()
        if not cleaned:
            continue
        outcome = store.force_keep(cleaned, domain=domain or None)
        counts[outcome] = counts.get(outcome, 0) + 1
        mark = {"added": "新增", "converted": "已改为保持英文", "unchanged": "本来就是"}
        typer.echo(f"  {cleaned:<20} {mark[outcome]}")

    # Manual edits have to invalidate the cache, or a re-translated document
    # replays the old output and the change appears to have done nothing.
    # 手工改动必须让缓存失效，否则重新翻译会重放旧结果，改动看起来毫无作用。
    settings = get_settings()
    settings.glossary.glossary_rev += 1
    save_settings(settings)

    typer.secho(f"完成：{counts}", fg=typer.colors.GREEN, bold=True)


@glossary_app.command("classify")
def glossary_classify(
    apply: bool = typer.Option(False, "--apply", help="写回数据库；不加则只报告"),
    keep_only: bool = typer.Option(False, "--keep-only",
                                   help="只处理「保持英文」的条目"),
    unclassified: bool = typer.Option(False, "--unclassified",
                                      help="只处理还没有学科的条目（即尚未分类过的）"),
    limit: int = typer.Option(0, "--limit", "-n", help="只处理前 N 条，用于试跑"),
    cache: str = typer.Option("", "--cache",
                              help="判定结果的缓存文件；已有则复用，避免重复请求"),
) -> None:
    """
    中文：重新归类术语库——规则先清垃圾，再请模型判定学科与「翻译/保持英文」。

    Re-sort the glossary into subjects and settle what stays in English.

    Rules condemn what is decidable — a person's name, a journal abbreviation,
    an arbitrary identifier — and a model decides the rest, giving every
    surviving entry a subject. Without ``--apply`` nothing is written.
    """
    _setup_logging()
    from pathlib import Path

    from local_translator.db import get_db
    from local_translator.glossary import classify as classifier
    from local_translator.glossary.store import GlossaryStore

    db = get_db(get_paths().db_path)
    store = GlossaryStore(db)

    where = "WHERE enabled = 1"
    if keep_only:
        where += " AND policy = 'keep'"
    if unclassified:
        # Automatic extraction files nothing under a subject, so "no subject"
        # is exactly the set that has arrived since the last classification.
        # 自动提取不给学科，因此「没有学科」恰好就是上次分类之后新进来的那批。
        where += " AND (domain IS NULL OR domain = '')"
    rows = [dict(r) for r in db.query(
        f"SELECT id, src, tgt, policy, domain, source FROM glossary_terms {where}"
        " ORDER BY id")]
    if limit:
        rows = rows[:limit]
    typer.secho(f"待处理 {len(rows)} 条", bold=True)

    condemned = classifier.junk(rows)
    condemned_ids = {term["id"] for term in condemned}
    typer.echo(f"  规则判定为垃圾：{len(condemned)} 条"
               f"（人名 / 刊物名 / 无意义标记）")
    for term in condemned[:8]:
        typer.echo(f"    - {term['src']}")

    remaining = [term for term in rows if term["id"] not in condemned_ids]

    # The model pass costs an API call per forty entries, so its verdicts are
    # worth keeping: a failure in the write step should never mean paying for
    # the whole classification again.
    # 模型判定每四十条要一次 API 请求，结果值得留存：写库这一步失败不该意味着整轮
    # 分类要重新付一次钱。
    cache_path = Path(cache) if cache else None
    verdicts: dict = {}
    if cache_path and cache_path.exists():
        stored = json.loads(cache_path.read_text(encoding="utf-8"))
        verdicts = {int(key): tuple(value) for key, value in stored.items()}
        typer.echo(f"  复用缓存的判定 {len(verdicts)} 条（{cache_path}）")

    if not verdicts:
        complete = _judge_client()
        if complete is None:
            typer.secho("  未配置 API 模型，跳过模型判定（只应用规则）", fg=typer.colors.YELLOW)
        else:
            calls = (len(remaining) + 39) // 40
            typer.echo(f"  交给模型判定 {len(remaining)} 条，约 {calls} 次请求…")
            verdicts = classifier.classify(remaining, complete)
            typer.echo(f"  模型答复了 {len(verdicts)} 条")
            if cache_path:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(
                    json.dumps({str(key): list(value) for key, value in verdicts.items()},
                               ensure_ascii=False, indent=1),
                    encoding="utf-8")
                typer.echo(f"  判定已缓存到 {cache_path}")

    dropped = list(condemned)
    updates: list[tuple] = []
    moved = relabelled = 0
    for term in remaining:
        answer = verdicts.get(term["id"])
        if answer is None:
            continue
        domain, policy, rendering = answer
        if policy == "drop":
            dropped.append(term)
            continue

        if policy == "keep":
            target = term["src"]
        else:
            # A keep-in-English entry stores its source as its target, so
            # flipping it to "translate" without a rendering would leave an
            # entry that translates the term into itself. Where the model gave
            # no Chinese, the entry stays as it was.
            # 保持英文的条目其译文字段存的就是原文，若翻成 translate 却没有译法，
            # 留下的会是「把术语翻译成它自己」的条目。模型没给中文时保持原样。
            target = rendering or (term["tgt"] if term["tgt"] != term["src"] else "")
            if not target:
                policy = term["policy"]
                target = term["tgt"]

        if term["domain"] != domain:
            moved += 1
        if term["policy"] != policy:
            relabelled += 1
        updates.append((domain, policy, target, term["id"]))

    typer.echo(f"  丢弃合计 {len(dropped)} 条；改学科 {moved} 条；"
               f"改翻译策略 {relabelled} 条")

    if not apply:
        typer.secho("试跑，未写入。加 --apply 生效。", fg=typer.colors.YELLOW)
        return

    import sqlite3

    collisions = 0
    with db.write() as conn:
        for term in dropped:
            conn.execute(
                "UPDATE glossary_terms SET enabled = 0, updated_at = datetime('now') "
                "WHERE id = ?", (term["id"],))
        for update in updates:
            try:
                conn.execute(
                    "UPDATE glossary_terms SET domain = ?, policy = ?, tgt = ?, "
                    "updated_at = datetime('now') WHERE id = ?", update)
            except sqlite3.IntegrityError:
                # The new rendering is one another row already has, so this row
                # became a duplicate of it. Retire it rather than abandoning the
                # whole write.
                # 新译法与另一行重复，这一行就成了它的副本。停用它，而不是让整次写入
                # 前功尽弃。
                collisions += 1
                conn.execute(
                    "UPDATE glossary_terms SET enabled = 0, "
                    "updated_at = datetime('now') WHERE id = ?", (update[3],))
    if collisions:
        typer.echo(f"  与既有条目重复而停用：{collisions} 条")

    stats = store.stats()
    typer.secho(f"已写入。启用中 {stats['total']} 条，已停用 {stats['disabled']} 条。",
                fg=typer.colors.GREEN)


def _judge_client():
    """
    中文：构造一个「发系统提示与用户提示、拿回文本」的可调用对象，配不好就返回 None。

    Build a ``(system, user) -> str`` callable backed by the configured API model.

    Returns:
        Callable | None: The caller, or ``None`` when no API model or key is
            configured — the cleaning then keeps everything the rules did not
            condemn, which is the safe direction.
    """
    from local_translator.config import get_settings
    from local_translator.engine.translators import ApiTranslator
    from local_translator.secrets import get_api_key

    settings = get_settings()
    api = settings.api_model
    key = get_api_key(api.keyring_account) if api.enabled else None
    if not key:
        return None
    try:
        engine = ApiTranslator(
            lang_in=settings.translation.lang_in,
            lang_out=settings.translation.lang_out,
            model=api.model_name,
            base_url=api.base_url or None,
            api_key=key,
            ignore_cache=True,
            skip_reasoning=api.skip_reasoning,
            prompt_template=api.chat_prompt_template,
        )
    except Exception:
        logging.getLogger(__name__).warning(
            "Could not build the judging client", exc_info=True)
        return None
    return lambda system, user: engine._raw_complete(system, user, 2048)


@app.command("setup-reader")
def setup_reader(
    force: bool = typer.Option(False, "--force", help="即使已安装也重新下载"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """
    中文：下载 pdf.js 阅读器组件（本地自带，不依赖 CDN）。

    Download the pdf.js viewer used by the bilingual reader.
    """
    _setup_logging(verbose)
    from local_translator.web import pdfjs

    target = pdfjs.install(force=force)
    typer.secho(f"  ✓ pdf.js installed at {target}", fg=typer.colors.GREEN)


@app.command(name="app")
def desktop_app(
    port: int = typer.Option(None, "--port", help="监听端口 / Port"),
) -> None:
    """
    中文：以桌面应用方式启动——一个窗口，关掉窗口即完全退出。

    Start the desktop application.

    The same interface as ``serve``, in a window of its own rather than a
    browser tab, and with the whole application ending when the window closes.
    This is what the packaged bundle runs.
    """
    from local_translator.desktop import run as run_app

    raise typer.Exit(run_app(port=port))


@app.command()
def serve(
    host: str = typer.Option(None, "--host", help="监听地址 / Bind address"),
    port: int = typer.Option(None, "--port", help="监听端口 / Port"),
    no_browser: bool = typer.Option(False, "--no-browser", help="不自动打开浏览器"),
    native: bool = typer.Option(False, "--native",
                                help="在独立的 macOS 窗口里打开，而不是浏览器标签页"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """
    中文：启动 Web 服务（翻译任务在后台子进程中运行，刷新页面不会中断）。

    Start the web interface.

    Translation runs in worker subprocesses owned by the server, so refreshing
    or closing the browser never interrupts a job — only pressing stop does.
    """
    _setup_logging(verbose)
    for noisy in ("openai", "pdfminer", "PIL", "watchfiles"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    from local_translator.web.app import run as run_web

    run_web(host=host, port=port, show=not no_browser, native=native)


@app.command("fix-titles")
def fix_titles(
    apply: bool = typer.Option(False, "--apply", help="真正写入，默认只预览 / Write the changes"),
) -> None:
    """
    中文：把历史里标题仍是 arXiv 编号的记录补成真标题。

    Repair library entries whose title is still an arXiv identifier.

    An identifier ends up in the title when the arXiv API was unreachable at
    submission time — it rate-limits a burst of submissions, and the metadata
    lookup used to fall straight back to the id. The PDFs are on disk and carry
    their titles, so the repair needs no network. Previews by default; pass
    ``--apply`` to write.
    """
    from pathlib import Path

    from local_translator.config import get_paths as _paths
    from local_translator.db import get_db
    from local_translator.library import SOURCE_ARXIV
    from local_translator.sources.arxiv import title_from

    db = get_db(_paths().db_path)
    rows = [dict(r) for r in db.list_documents(limit=100000)]
    repaired = 0
    for row in rows:
        title = (row.get("title") or "").strip()
        if row.get("source_kind") != SOURCE_ARXIV or not title:
            continue
        # A title that is nothing but an identifier: digits, dots and a version.
        # 纯粹是编号的标题：只有数字、点和版本号。
        if not title.replace(".", "").replace("v", "").isdigit():
            continue
        source = Path(row["source_path"])
        if not source.is_file():
            typer.echo(f"  ? {title}  源文件已不在，跳过")
            continue
        better = title_from({"title": title, "authors": "", "abs_url": ""}, source, title)
        if better == title:
            typer.echo(f"  ? {title}  PDF 里也没有可用标题")
            continue
        repaired += 1
        typer.echo(f"  {'✓' if apply else '·'} {title} → {better}")
        if apply:
            with db.write() as conn:
                conn.execute("UPDATE documents SET title = ? WHERE id = ?",
                             (better, row["id"]))
    if not repaired:
        typer.secho("没有需要修复的记录", fg=typer.colors.GREEN)
    elif apply:
        typer.secho(f"\n已修复 {repaired} 条", fg=typer.colors.GREEN, bold=True)
    else:
        typer.secho(f"\n可修复 {repaired} 条，加 --apply 写入", bold=True)


@app.command()
def stop(
    timeout: float = typer.Option(20.0, help="等待优雅退出的秒数 / Seconds to wait"),
) -> None:
    """
    中文：停止正在运行的服务（同时释放本地模型占用的内存）。

    Stop the running server.

    The server's own shutdown handler cancels running translations, stops
    llama-server and closes the database, so a clean stop also releases the
    several gigabytes the model weights occupy.
    """
    from local_translator.server_control import stop_server

    stopped, message = stop_server(get_paths().pid_path, timeout=timeout)
    typer.echo(message)
    raise typer.Exit(0 if stopped else 1)


def main() -> None:
    """中文：CLI 主入口。 / Console-script entry point."""
    app()


if __name__ == "__main__":
    sys.exit(app())
