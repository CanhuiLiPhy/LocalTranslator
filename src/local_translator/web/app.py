"""
中文：NiceGUI 前端——首页与文库页。

NiceGUI front end.

The defining property of these pages is that they hold **no translation state**.
Every render reads the current truth from :class:`~local_translator.service.TranslatorService`,
and a ``ui.timer`` re-reads it once a second. Consequently:

- Refreshing the browser re-renders from live data instead of losing progress,
  because there was never any per-page progress state to lose.
- Closing the browser entirely does not touch the worker subprocess; reopening
  it shows the job still running, or already finished and in the library.
- Only an explicit "停止" button, or shutting the server down, ends a translation.

Contrast with the previous NiceGUI prototype, which stored progress in a
``PageState`` object constructed inside ``@ui.page`` and awaited the translation
inside the click handler — both of which tie the work's lifetime to one browser
connection.
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

from nicegui import app, events, ui

from local_translator import edition
from local_translator.config import get_settings
from local_translator.library import SOURCE_LOCAL
from local_translator.server_control import clear_pid, write_pid
from local_translator.service import (
    DuplicateDocument,
    LlamaServerError,
    PendingSubmission,
    TranslatorService,
    get_service,
)
from local_translator.web.format import (
    human_bytes,
    short_path,
    status_label,
    usage_line,
)
from local_translator.web.theme import (
    BTN_PRIMARY,
    CARD,
    HEADING,
    INPUT_PROPS,
    META,
    MUTED,
    NOTION_CSS,
    ROW,
    apply_palette,
    badge_style,
)

logger = logging.getLogger(__name__)

# Statuses that still change on their own and therefore need polling.
# 仍会自行变化、需要轮询的状态。
LIVE_STATUSES = {"queued", "running"}


# --------------------------------------------------------------------------- #
# Layout / 布局
# --------------------------------------------------------------------------- #


def _header(active: str) -> None:
    """
    中文：渲染顶部导航栏与本地模型状态指示。

    Render the top navigation bar.

    Args:
        active (str): Key of the current page, used to highlight its nav item.
    """
    service = get_service()
    status = service.model_status()

    ui.add_head_html(NOTION_CSS)
    apply_palette()

    with ui.header().classes("px-5 py-2"):
        with ui.row().classes("w-full items-center gap-1 max-w-6xl mx-auto"):
            ui.label("Local Translator").classes(
                "text-[14.5px] font-semibold mr-4"
            ).style("color: #37352f")

            for key, label, target in (
                ("home", "文件", "/"),
                ("text", "文本", "/text"),
                ("image", "图片", "/image"),
                ("dictionary", "词典", "/dictionary"),
                ("library", "历史", "/library"),
                ("glossary", "术语库", "/glossary"),
                ("settings", "设置", "/settings"),
            ):
                classes = "lt-navlink" + (" lt-navlink-active" if key == active else "")
                ui.link(label, target).classes(classes)

            ui.space()

            _backend_switch(service, status)

            _quit_button(service)

            # The chip reports the local inference server. On the lite edition
            # there is none, and a permanently grey dot beside a model name that
            # was never downloaded reads as a broken installation.
            # 这个状态点报告的是本地推理服务。轻量版没有本地推理服务，一个常灰的圆点
            # 配上从未下载过的模型名，看起来就像装坏了。
            _model_chips(status)


def _model_chips(status: dict) -> None:
    """
    中文：在页首标出真正在干活的模型——本地、云端，或混合模式下两者都标。

    Name the model that will actually do the work.

    The header used to show the local model whatever the backend was, so a run
    on a cloud API displayed the name of a model it was not using. Each chip is
    shown only when its backend is selected; hybrid uses both and shows both.

    Args:
        status (dict): Model status from the service, carrying the active
            backend and both models' names.
    """
    backend = status.get("backend", "local")
    uses_local = backend in ("local", "hybrid") and not edition.is_lite()
    uses_cloud = backend in ("api", "hybrid")

    if uses_local:
        # The dot reports the local inference server: green once it is up, grey
        # before its first load.
        # 这个圆点报告本地推理服务：起来了是绿色，首次加载前是灰色。
        colour = "text-[#0f7b6c]" if status["healthy"] else "text-[#9b9a97]"
        tooltip = (
            f"已就绪 · {status['model']}"
            + (f" · 投机解码 {status['speculative']}" if status.get("speculative") else "")
            if status["healthy"]
            else ("模型未启动（首次翻译时自动加载）" if status["installed"] else "模型未下载")
        )
        with ui.row().classes("items-center gap-2"):
            ui.icon("circle", size="0.5rem").classes(colour)
            ui.label(status["model"].replace(".gguf", "")).classes(MUTED)
        ui.tooltip(tooltip)

    if uses_cloud:
        if status.get("cloud_provider") == "tencent":
            name = "腾讯云翻译"
            full = name
        else:
            full = status.get("cloud_model") or "未配置模型"
            # Vendor prefixes such as 「MiniMaxAI/」 push the name past the width
            # the header has; the whole thing stays in the tooltip.
            # 「MiniMaxAI/」这类厂商前缀会把名字挤出页首的宽度；完整名称留在悬浮说明里。
            name = full.rsplit("/", 1)[-1]
        with ui.row().classes("items-center gap-2"):
            ui.icon("cloud", size="0.9rem").classes("text-[#9b9a97]")
            ui.label(name).classes(MUTED)
        ui.tooltip(f"云端 · {full}")


BACKEND_NOTES = edition.BACKEND_NOTES


def _quit_button(service: TranslatorService) -> None:
    """
    中文：渲染退出按钮，关闭服务并释放本地模型内存。

    Render the quit control.

    The application is launched from an icon and used entirely in a browser, so
    without this there is no way to stop it that does not involve a terminal —
    and a server left running holds a port and, in local or hybrid mode, several
    gigabytes of model weights.

    Quitting goes through the server's own shutdown handler, which cancels
    running translations and stops llama-server. Confirmation is asked for
    because a translation in progress is lost, and the button sits next to
    controls that are used routinely.

    Args:
        service (TranslatorService): Service whose lifecycle is being ended.
    """

    async def confirm() -> None:
        snapshot = service.manager.snapshot()
        busy = len(snapshot.get("running", [])) + len(snapshot.get("queued", []))
        warning = (
            f"有 {busy} 个翻译正在进行或排队，退出会中断它们。"
            if busy
            else "翻译服务将停止，本地模型占用的内存会被释放。"
        )
        with ui.dialog() as dialog, ui.card().classes("gap-3"):
            ui.label("退出本地翻译？").classes(HEADING)
            ui.label(warning).classes(MUTED)
            ui.label("下次从「Local Translator」图标重新启动即可。").classes(MUTED)
            with ui.row().classes("justify-end gap-2 w-full"):
                ui.button("取消", on_click=dialog.close).classes("lt-btn")
                ui.button("退出", on_click=lambda: dialog.submit(True)).classes(
                    "lt-btn lt-btn-danger"
                )
        if not await dialog:
            return

        ui.notify("正在退出…", type="ongoing")
        # Give the notification a moment to reach the browser: shutting the
        # server down first would close the connection before it arrives, and
        # the page would simply go dead with no explanation.
        # 先让提示送达浏览器：直接关服务会在提示到达前断开连接，页面会毫无征兆地失去响应。
        await asyncio.sleep(0.4)
        app.shutdown()

    ui.button(icon="power_settings_new", on_click=confirm).props(
        "flat dense round size=sm"
    ).classes("text-[#9b9a97]")
    ui.tooltip("退出并停止服务")


def _backend_switch(service: TranslatorService, status: dict) -> None:
    """
    中文：渲染本地/云端/混合三种翻译后端的切换。

    Render the translation backend switch.

    The backend decides both where a paragraph is translated and how long the
    document takes, so it belongs in the header rather than buried in settings —
    it is a per-document decision, not a preference set once.

    Hybrid keeps the short paragraphs on the local model and sends the long and
    formula-dense ones to the API, where the local model tends to drop formula
    markers. Both run at once, so a document finishes sooner than it would on
    either backend alone and costs fewer API tokens than the API alone.

    Args:
        service (TranslatorService): Service performing the switch.
        status (dict): Current model status, providing the active backend.
    """

    async def on_change(event) -> None:
        backend = event.value
        if not backend:
            return
        try:
            await asyncio.to_thread(service.set_backend, backend)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Backend switch failed")
            ui.notify(f"切换失败：{exc}", type="negative")
            return
        ui.notify(BACKEND_NOTES[backend], type="positive")

    # The lite edition ships without a local model, so offering the switch at
    # all would be offering two options that cannot work. A single remaining
    # option is shown as a plain label instead of a one-button toggle.
    # 轻量版不含本地模型，把开关整个显示出来等于提供两个用不了的选项。只剩一个选项时
    # 显示为普通标签，而不是只有一个按钮的切换器。
    available = {key: edition.BACKEND_SHORT.get(key, key) for key in edition.backends()}
    if len(available) == 1:
        ui.label(f"{next(iter(available.values()))}翻译").classes(MUTED)
        return

    toggle = ui.toggle(
        available,
        value=status.get("backend", edition.default_backend()),
    ).props("dense unelevated no-caps toggle-color=primary")
    toggle.on_value_change(on_change)


def _page_body():
    """
    中文：返回统一的页面主体容器。

    Create the standard page body container.

    Returns:
        ui.column: A column element to use as a context manager.
    """
    return ui.column().classes("w-full max-w-6xl mx-auto px-6 py-6 gap-5")


# --------------------------------------------------------------------------- #
# Home page / 首页
# --------------------------------------------------------------------------- #


def _source_card(service: TranslatorService, on_submit) -> None:
    """
    中文：渲染「添加翻译来源」卡片——arXiv / 批量 / 本地文件 / 上传 / LaTeX 源码。

    Render the source input card.

    Five intake paths are offered because they have genuinely different
    behaviour, most importantly where the output lands: files already on disk
    get their translations written beside them, while uploads, arXiv papers and
    LaTeX sources go into the managed library.

    Args:
        service (TranslatorService): Service used to submit jobs.
        on_submit (Callable[[], None]): Called after a successful submission so
            the job list refreshes immediately rather than on the next tick.
    """
    with ui.card().classes(CARD):
        with ui.row().classes("items-center gap-2 mb-3"):
            ui.label("添加翻译").classes(HEADING)

        with ui.tabs().classes("w-full") as tabs:
            # arXiv first, because it is the one most papers arrive through and
            # a first tab is the one a reader lands on.
            # arXiv 排第一：多数论文是从这里来的，而第一个标签页就是读者落脚的地方。
            tab_arxiv = ui.tab("arXiv", icon="link")
            tab_batch = ui.tab("批量", icon="folder")
            tab_file = ui.tab("本地文件", icon="description")
            tab_upload = ui.tab("上传", icon="cloud_upload")
            tab_latex = ui.tab("LaTeX 源码", icon="code")

        with ui.tab_panels(tabs, value=tab_arxiv).classes("w-full"):
            with ui.tab_panel(tab_arxiv):
                _arxiv_panel(service, on_submit)
            with ui.tab_panel(tab_batch):
                _batch_panel(service, on_submit)
            with ui.tab_panel(tab_file):
                _file_panel(service, on_submit)
            with ui.tab_panel(tab_upload):
                _upload_panel(service, on_submit)
            with ui.tab_panel(tab_latex):
                _latex_panel(service, on_submit)


def _pages_input():
    """
    中文：渲染页码范围输入框。

    Render the shared page-range input.

    Returns:
        ui.input: The input element; its value is a BabelDOC range expression.
    """
    return ui.input(
        "页码范围（留空翻译全部）", placeholder="例如 1-5, 8, 10-"
    ).classes("w-64").props(INPUT_PROPS)


async def _ensure_model(service: TranslatorService) -> bool:
    """
    中文：确保本地模型就绪；加载放在线程里以免阻塞界面。

    Ensure the local inference server is up before queueing local work.

    Loading several gigabytes of weights takes long enough that doing it on the
    event loop would freeze every connected client, so it runs in a worker thread.

    Args:
        service (TranslatorService): Service owning the supervisor.

    Returns:
        bool: ``True`` if translation can proceed.
    """
    if service.settings.translation.backend != "local":
        return True
    if service.supervisor.health():
        return True

    # ui.notify() is fire-and-forget: it returns nothing to dismiss, and the
    # 「ongoing」 type carries no timeout, so a message posted that way stays on
    # screen for the life of the page no matter what the backend goes on to do.
    # ui.notification() is the object form, and dismissing it in a finally is
    # what ties the message to the work rather than to the moment it started.
    # ui.notify() 是「发完就忘」：它不返回任何可供关闭的东西，而 ongoing 类型本身没有超时，
    # 于是这样发出的消息会在页面存活期间一直挂着，与后端后续做了什么毫无关系。
    # ui.notification() 是它的对象形式，在 finally 里关掉它，才把这条消息绑到那件事上，
    # 而不是绑到它开始的那一刻。
    progress = ui.notification(
        "正在启动本地模型服务，首次加载需要一点时间…", type="ongoing", spinner=True
    )
    try:
        # Dismissed as soon as the await returns, not when this function does,
        # so nothing that follows is drawn underneath a stale message.
        # 在 await 返回时就撤掉，而不是等函数返回，这样后面的内容不会画在一条过期消息底下。
        try:
            await asyncio.to_thread(service.ensure_model_ready)
        finally:
            progress.dismiss()
    except LlamaServerError as exc:
        ui.notify(str(exc), type="negative", timeout=10000)
        return False
    ui.notify("本地模型已就绪", type="positive")
    return True


def _file_panel(service: TranslatorService, on_submit) -> None:
    """
    中文：本地文件路径输入面板。

    Render the "translate a file already on disk" panel.

    Args:
        service (TranslatorService): Service used to submit jobs.
        on_submit (Callable[[], None]): Refresh callback.
    """
    with ui.column().classes("w-full gap-3 pt-2"):
        path_input = (
            ui.input("PDF 路径", placeholder="~/Downloads/paper.pdf")
            .classes("w-full")
            .props(INPUT_PROPS)
        )
        ui.label("译文将写回该 PDF 所在目录").classes(MUTED)
        pages = _pages_input()

        async def submit() -> None:
            raw = (path_input.value or "").strip().strip("'\"")
            if not raw:
                ui.notify("请输入 PDF 路径", type="warning")
                return
            source = Path(raw).expanduser()
            if not source.exists():
                ui.notify(f"文件不存在：{source}", type="negative")
                return
            if source.suffix.lower() != ".pdf":
                ui.notify("只支持 PDF 文件", type="negative")
                return
            if not await _ensure_model(service):
                return
            try:
                service.submit_pending(PendingSubmission(
                    source=source,
                    source_kind=SOURCE_LOCAL,
                    title=_pdf_title(service, source),
                    pages=pages.value or None,
                ))
            except DuplicateDocument as duplicate:
                if await _confirm_copy(duplicate, service):
                    path_input.value = ""
                    on_submit()
                return
            except Exception as exc:
                logger.exception("Submit failed")
                ui.notify(f"提交失败：{exc}", type="negative")
                return
            path_input.value = ""
            ui.notify(f"已加入队列：{source.name}", type="positive")
            on_submit()

        ui.button("开始翻译", on_click=submit).classes(BTN_PRIMARY).props("no-caps unelevated")


def _batch_panel(service: TranslatorService, on_submit) -> None:
    """
    中文：批量面板——一个文件夹里的所有 PDF，或一串 arXiv 编号 / 链接。

    Render the batch import panel.

    Two kinds of batch, one panel, because they are the same request: a reader
    has a list of papers and wants all of them. Where they come from — a folder
    on disk or a column of arXiv links pasted out of a bibliography — changes
    the fetching and nothing else.

    The arXiv half obeys the settings exactly as a single paper does, so with
    「arXiv 优先使用 LaTeX 源码翻译」 on, a batch translates from source
    wherever source exists and falls back to the PDF where it does not. The
    batch makes no choice of its own.

    **中文：** 两种批量放在一个面板里，因为它们是同一个诉求：读者手上有一串论文，想全译。
    来源是磁盘上的文件夹还是从参考文献里粘出来的一列 arXiv 链接，只改变取件方式，别的都
    一样。arXiv 那一半与单篇提交守同一套设置——开着「arXiv 优先使用 LaTeX 源码翻译」时，
    有源码的从源码翻，没有的回到 PDF。批量本身不做任何选择。

    Args:
        service (TranslatorService): Service used to submit jobs.
        on_submit (Callable[[], None]): Refresh callback.
    """
    with ui.column().classes("w-full gap-3 pt-2"):
        with ui.tabs().props("dense no-caps").classes("self-start") as inner:
            by_links = ui.tab("arXiv 批量", icon="link")
            by_folder = ui.tab("文件夹", icon="folder")

        with ui.tab_panels(inner, value=by_links).classes("w-full"):
            with ui.tab_panel(by_links):
                _arxiv_batch(service, on_submit)
            with ui.tab_panel(by_folder):
                _folder_batch(service, on_submit)


def _arxiv_batch(service: TranslatorService, on_submit) -> None:
    """
    中文：一次翻译一串 arXiv 论文。

    Enqueue a list of arXiv papers.

    Args:
        service (TranslatorService): Service used to submit jobs.
        on_submit (Callable[[], None]): Refresh callback.
    """
    references = (
        ui.textarea(
            "arXiv 编号或链接，每行一篇",
            placeholder="2510.04704\narxiv.org/abs/2302.13971\n2312.11805",
        )
        .classes("w-full")
        .props(f"{INPUT_PROPS} autogrow input-style='min-height: 110px'")
    )
    if service.settings.latex.prefer_for_arxiv and service.latex_available()[0]:
        ui.label(
            "按当前设置，每一篇有源码的都会从 LaTeX 源码翻译，没有源码的自动回到 PDF。"
        ).classes(MUTED)
    pages = _pages_input()

    async def submit() -> None:
        """中文：把每一行当成一篇论文提交。 / Submit one paper per line."""
        wanted = [line.strip() for line in (references.value or "").splitlines()
                  if line.strip()]
        if not wanted:
            ui.notify("请每行填一篇 arXiv 编号或链接", type="warning")
            return
        if not await _ensure_model(service):
            return
        progress = ui.notification(
            f"正在获取 {len(wanted)} 篇…", type="ongoing", spinner=True)
        try:
            result = await service.submit_arxiv_batch(
                wanted, pages=pages.value or None)
        except Exception as exc:  # noqa: BLE001
            logger.exception("arXiv batch submit failed")
            ui.notify(f"提交失败：{exc}", type="negative", timeout=8000)
            return
        finally:
            progress.dismiss()

        queued = len(result["job_ids"])
        if queued:
            references.value = ""
        if result["failed"]:
            names = "、".join(name for name, _ in result["failed"][:3])
            ui.notify(
                f"已加入队列 {queued} 篇；{len(result['failed'])} 篇没取到（{names}）",
                type="warning", timeout=8000)
        else:
            ui.notify(f"已加入队列 {queued} 篇", type="positive")
        on_submit()

    ui.button("批量翻译", on_click=submit).classes(BTN_PRIMARY).props(
        "no-caps unelevated")


def _folder_batch(service: TranslatorService, on_submit) -> None:
    """
    中文：一次翻译一个文件夹里的所有 PDF。

    Enqueue every PDF in a folder.

    Args:
        service (TranslatorService): Service used to submit jobs.
        on_submit (Callable[[], None]): Refresh callback.
    """
    folder_input = (
        ui.input("文件夹路径", placeholder="~/Downloads/papers")
        .classes("w-full")
        .props(INPUT_PROPS)
    )
    with ui.row().classes("gap-6"):
        recursive = ui.checkbox("包含子文件夹", value=True)
        skip_done = ui.checkbox("跳过已翻译的", value=True)
    pages = _pages_input()

    async def submit() -> None:
        """中文：扫描文件夹并逐个提交。 / Scan the folder and submit each PDF."""
        raw = (folder_input.value or "").strip().strip("'\"")
        if not raw:
            ui.notify("请输入文件夹路径", type="warning")
            return
        folder = Path(raw).expanduser()
        if not folder.is_dir():
            ui.notify(f"不是有效目录：{folder}", type="negative")
            return
        if not await _ensure_model(service):
            return
        try:
            result = await asyncio.to_thread(
                service.submit_folder,
                folder,
                recursive=recursive.value,
                skip_translated=skip_done.value,
                pages=pages.value or None,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Folder submit failed")
            ui.notify(f"提交失败：{exc}", type="negative")
            return

        queued = len(result["job_ids"])
        if queued:
            ui.notify(
                f"已加入队列 {queued} 个文件，跳过 {result['skipped']} 个", type="positive"
            )
        else:
            ui.notify(
                f"没有需要翻译的文件（共发现 {result['total_found']} 个 PDF）", type="info"
            )
        on_submit()

    ui.button("批量翻译", on_click=submit).classes(BTN_PRIMARY).props(
        "no-caps unelevated")


async def _confirm_copy(exc, service: TranslatorService) -> bool:
    """
    中文：库里已有同名文章时弹窗询问，确认后翻译成一个带序号的副本。

    Ask whether to translate a paper the library already has.

    Re-translating is a legitimate thing to want — settings change, a glossary
    grows, a model is swapped — so the answer is a question, not a refusal. The
    second run is registered as its own document titled 「…（1）」 so that both
    results stay readable side by side instead of one silently replacing the
    other.

    Args:
        exc (DuplicateDocument): The raised duplicate, carrying the prepared
            submission and the entries that share its title.
        service (TranslatorService): Service used to complete the submission.

    Returns:
        bool: ``True`` when the copy was submitted, ``False`` when the user
            cancelled.
    """
    decision = {"go": False}

    with ui.dialog() as dialog, ui.card().classes("w-[460px] gap-3 p-5"):
        ui.label("历史里已经有这篇了").classes("text-[15px] font-semibold")
        ui.label(f"《{exc.pending.title}》").classes("text-[13.5px]").style("color: #37352f")

        existing = exc.existing[0]
        when = (existing.get("created_at") or "")[:10]
        ui.label(f"已有条目：{existing.get('title')}"
                 + (f" · {when}" if when else "")).classes(MUTED)
        ui.label(f"继续翻译会生成副本《{_copy_name(exc)}》，原来的那份不受影响。").classes(MUTED)

        with ui.row().classes("w-full justify-end gap-2 pt-1"):
            ui.button("取消", on_click=dialog.close).props("flat no-caps")

            def go() -> None:
                decision["go"] = True
                dialog.close()

            ui.button("仍然翻译", on_click=go).classes(BTN_PRIMARY).props(
                "no-caps unelevated")

    await dialog
    if not decision["go"]:
        return False

    try:
        service.submit_pending(exc.pending, as_copy=True)
    except Exception as inner:  # noqa: BLE001
        logger.exception("Copy submit failed")
        ui.notify(f"提交失败：{inner}", type="negative")
        return False
    ui.notify(f"已加入队列：{_copy_name(exc)}", type="positive")
    return True


def _copy_name(exc) -> str:
    """
    中文：副本将使用的标题。

    The title the copy will be given.

    Args:
        exc (DuplicateDocument): The raised duplicate.

    Returns:
        str: e.g. ``"Attention Is All You Need（1）"``.
    """
    from local_translator.library import copy_title

    return copy_title(exc.pending.title, exc.copy_index)


def _pdf_title(service: TranslatorService, source: Path) -> str:
    """
    中文：读出 PDF 自己的标题，读不到就用文件名。

    Read a PDF's own title, falling back to its filename.

    The document's metadata title is what makes the duplicate check work across
    intake routes: the same paper arrives as ``2608.13040v1.pdf`` from a
    download and under its real title from arXiv.

    Args:
        service (TranslatorService): Unused beyond keeping the call sites
            uniform; the read needs no service state.
        source (Path): Path to the PDF.

    Returns:
        str: The metadata title where the PDF has one, else the filename stem.
    """
    from local_translator.library import read_pdf_metadata

    title = str(read_pdf_metadata(source).get("title") or "").strip()
    return title or source.stem


def _upload_panel(service: TranslatorService, on_submit) -> None:
    """
    中文：浏览器上传面板。

    Render the browser upload panel.

    Args:
        service (TranslatorService): Service used to submit jobs.
        on_submit (Callable[[], None]): Refresh callback.
    """
    with ui.column().classes("w-full gap-3 pt-2"):
        pages = _pages_input()

        async def handle(event: events.UploadEventArguments) -> None:
            content = await event.file.read()
            if not await _ensure_model(service):
                return
            try:
                service.submit_upload(event.file.name, content, pages=pages.value or None)
            except DuplicateDocument as duplicate:
                if await _confirm_copy(duplicate, service):
                    on_submit()
                uploader.reset()
                return
            except Exception as exc:
                logger.exception("Upload submit failed")
                ui.notify(f"提交失败：{exc}", type="negative")
                uploader.reset()
                return
            ui.notify(f"已加入队列：{event.file.name}", type="positive")
            # The browser fires no change event when the same file is picked
            # twice in a row, so without this the second upload of a paper
            # silently does nothing at all.
            # 连续两次选同一个文件时浏览器不会再触发 change 事件，不重置的话第二次上传
            # 同一篇论文会毫无反应。
            uploader.reset()
            on_submit()

        uploader = ui.upload(
            label="拖拽或点击上传 PDF",
            multiple=True,
            auto_upload=True,
            on_upload=handle,
        ).props('accept=".pdf" flat bordered').classes("w-full")

        ui.label(
            f"上传的文件会保存到文库：{service.paths.library_dir}"
        ).classes(MUTED)


def _latex_panel(service: TranslatorService, on_submit) -> None:
    """
    中文：LaTeX 源码面板——上传压缩包，或指向本机的一个源码文件夹。

    Render the LaTeX source panel.

    Translating a paper from its source removes a whole class of defect at
    once: nothing is clipped, rescaled or overlapped, because TeX lays the page
    out from scratch. What it needs is the source and a TeX engine, so the
    panel says plainly when the engine is missing rather than letting a reader
    upload an archive and find out afterwards.

    Args:
        service (TranslatorService): Service used to submit jobs.
        on_submit (Callable[[], None]): Refresh callback.
    """
    available, description = service.latex_available()

    with ui.column().classes("w-full gap-3 pt-2"):
        if not available:
            ui.label(description).classes(MUTED)
            ui.label(
                "请安装含 TeX 功能的版本，或在设置页填写系统 TeX 引擎的路径。"
            ).classes(MUTED)
            return

        async def submit_folder() -> None:
            """中文：提交一个本地源码文件夹。 / Submit a local source folder."""
            raw = (folder_input.value or "").strip()
            if not raw:
                ui.notify("请输入源码文件夹路径", type="warning")
                return
            if not await _ensure_model(service):
                return
            try:
                service.submit_latex_folder(Path(raw))
            except DuplicateDocument as duplicate:
                if await _confirm_copy(duplicate, service):
                    folder_input.value = ""
                    on_submit()
                return
            except Exception as exc:  # noqa: BLE001
                logger.exception("LaTeX folder submit failed")
                ui.notify(f"提交失败：{exc}", type="negative", timeout=8000)
                return
            folder_input.value = ""
            ui.notify("已加入队列", type="positive")
            on_submit()

        async def handle(event: events.UploadEventArguments) -> None:
            """中文：提交一个上传的源码压缩包。 / Submit an uploaded archive."""
            content = await event.file.read()
            if not await _ensure_model(service):
                return
            try:
                service.submit_latex_archive(event.file.name, content)
            except DuplicateDocument as duplicate:
                if await _confirm_copy(duplicate, service):
                    on_submit()
                uploader.reset()
                return
            except Exception as exc:  # noqa: BLE001
                logger.exception("LaTeX archive submit failed")
                ui.notify(f"提交失败：{exc}", type="negative", timeout=8000)
                uploader.reset()
                return
            ui.notify(f"已加入队列：{event.file.name}", type="positive")
            uploader.reset()
            on_submit()

        uploader = ui.upload(
            label="拖拽或点击上传源码压缩包（.zip / .tar.gz）",
            multiple=False,
            auto_upload=True,
            on_upload=handle,
        ).props('accept=".zip,.tar.gz,.tgz,.tar" flat bordered').classes("w-full")

        with ui.row().classes("w-full items-end gap-3"):
            folder_input = (
                ui.input("或：本机源码文件夹", placeholder="/Users/…/paper-source")
                .classes("flex-1")
                .props(INPUT_PROPS)
            )
            ui.button("翻译该文件夹", on_click=submit_folder).classes(
                BTN_PRIMARY).props("no-caps unelevated")

        ui.label(description).classes(MUTED)
        ui.label(
            "源码会先复制进文库再翻译，不会改动你自己的文件夹。产物是中文单语 PDF——"
            "版面由 TeX 重排，中文的某一页没有一个对应的英文页可以并排。"
        ).classes(MUTED)


def _arxiv_panel(service: TranslatorService, on_submit) -> None:
    """
    中文：arXiv 链接/编号面板。

    Render the arXiv intake panel.

    Args:
        service (TranslatorService): Service used to submit jobs.
        on_submit (Callable[[], None]): Refresh callback.
    """
    with ui.column().classes("w-full gap-3 pt-2"):
        ref_input = (
            ui.input("arXiv 链接或编号", placeholder="2510.04704 或 arxiv.org/abs/2510.04704")
            .classes("w-full")
            .props(INPUT_PROPS)
        )
        ui.label("自动获取标题作者、下载 PDF 到文库并开始翻译").classes(MUTED)
        if service.settings.latex.prefer_for_arxiv and service.latex_available()[0]:
            ui.label(
                "当前设置为优先使用 LaTeX 源码翻译：有源码就从源码翻，没有就回到 PDF。"
            ).classes(MUTED)
        pages = _pages_input()

        async def submit() -> None:
            raw = (ref_input.value or "").strip()
            if not raw:
                ui.notify("请输入 arXiv 链接或编号", type="warning")
                return
            if not await _ensure_model(service):
                return
            progress = ui.notification("正在从 arXiv 下载…", type="ongoing", spinner=True)
            try:
                try:
                    await service.submit_arxiv(raw, pages=pages.value or None)
                finally:
                    progress.dismiss()
            except DuplicateDocument as duplicate:
                if await _confirm_copy(duplicate, service):
                    ref_input.value = ""
                    on_submit()
                return
            except Exception as exc:
                logger.exception("arXiv submit failed")
                ui.notify(f"获取失败：{exc}", type="negative", timeout=8000)
                return
            ref_input.value = ""
            ui.notify("已加入队列", type="positive")
            on_submit()

        ui.button("下载并翻译", on_click=submit).classes(BTN_PRIMARY).props("no-caps unelevated")


def _job_row(service: TranslatorService, job: dict, refresh) -> None:
    """
    中文：渲染任务列表中的一行。

    Render one row of the job list.

    Args:
        service (TranslatorService): Service used for cancel/retry actions.
        job (dict): Job row merged with live scheduler progress.
        refresh (Callable[[], None]): Refresh callback for after an action.
    """
    status = job["status"]
    label, colour = status_label(status)
    is_live = status in LIVE_STATUSES

    with ui.column().classes(f"{ROW} gap-2"):
        with ui.row().classes("w-full items-center gap-3"):
            ui.badge(label).style(badge_style(status))
            ui.label(job.get("title") or Path(job["source_path"]).stem).classes(
                "flex-1 text-[13.5px] font-medium truncate"
            ).style("color: #37352f")

            if is_live:
                # Queued jobs can be reordered; a running one has a worker
                # attached, so moving it would mean nothing.
                # 排队任务可调序；运行中的已绑定 worker，调序没有意义。
                if status == "queued":
                    def move(delta: int, job_id: str = job["id"]) -> None:
                        if service.manager.reorder(job_id, delta):
                            refresh()

                    ui.button(icon="arrow_upward", on_click=lambda: move(-1)).props(
                        "flat dense round size=sm"
                    ).tooltip("上移")
                    ui.button(icon="arrow_downward", on_click=lambda: move(1)).props(
                        "flat dense round size=sm"
                    ).tooltip("下移")

                async def cancel(job_id: str = job["id"]) -> None:
                    await service.cancel(job_id)
                    ui.notify("已请求停止", type="warning")
                    refresh()

                ui.button("停止" if status == "running" else "移除", on_click=cancel).props(
                    "flat dense no-caps"
                ).classes("text-[13px]").style("color: #eb5757")
            elif status in ("failed", "interrupted", "cancelled"):
                def retry(job_id: str = job["id"]) -> None:
                    if service.retry(job_id):
                        ui.notify("已重新加入队列", type="positive")
                        refresh()
                    else:
                        ui.notify("无法重试：源文件已不存在", type="negative")

                ui.button("重试", on_click=retry).props("flat dense no-caps").classes("text-[13px]")

        if status == "running":
            progress = float(job.get("progress") or 0)
            ui.linear_progress(value=progress / 100, show_value=False, color="info").classes("w-full")
            stage = job.get("stage") or "准备中"
            detail = ""
            if job.get("stage_total"):
                detail = f"（{job.get('stage_current', 0)}/{job['stage_total']}）"
            ui.label(f"{progress:.0f}% · {stage}{detail}").classes(f"{META} lt-progress-label")

        elif status == "queued":
            ui.label("等待前面的任务完成").classes(MUTED)

        elif status == "completed":
            with ui.row().classes("items-center gap-3 flex-wrap"):
                if job.get("dual_path"):
                    _output_link("双语对照", job["dual_path"], job["doc_id"],
                                 mode="combined")
                if job.get("mono_path"):
                    _output_link("译文", job["mono_path"], job["doc_id"],
                                 mode="mono")
            ui.label(usage_line(job)).classes(f"{MUTED} lt-usage")

        elif job.get("error"):
            ui.label(job["error"]).classes("text-xs text-red-600")


def _output_link(label: str, path: str, doc_id: str | None,
                 mode: str = "") -> None:
    """
    中文：渲染一个产物的操作按钮组（阅读 / 下载 / 在访达中显示）。

    Render the action buttons for one output file.

    Args:
        label (str): Display label, e.g. ``"双语对照"``.
        path (str): Absolute path to the output PDF.
        doc_id (str | None): Document id; when given, an "open in reader" button
            is shown for this file.
        mode (str): Which reader mode to open — ``combined`` for the bilingual
            file, ``mono`` for the translation on its own. The translation used
            to have no reader button at all, so a reader who wanted the Chinese
            by itself had to download the file and open it elsewhere.
            用哪种阅读模式打开——combined 是双语版，mono 是单独的译文。译文此前根本没有
            阅读按钮，只想看中文的人得先下载、再用别的软件打开。
    """
    exists = Path(path).exists()
    with ui.row().classes("items-center gap-1"):
        ui.icon("picture_as_pdf", size="xs").classes(
            "text-red-500" if exists else "text-gray-300"
        )
        ui.label(label).classes("text-xs font-medium text-gray-700")

        if not exists:
            ui.label("(文件已移除)").classes("text-xs text-gray-400")
            return

        if doc_id:
            target = f"/reader/{doc_id}" + (f"?mode={mode}" if mode else "")
            ui.button(
                icon="menu_book",
                on_click=lambda t=target: ui.navigate.to(t),
            ).props("flat dense round size=sm").tooltip("在阅读器中打开")

        ui.button(
            icon="download",
            on_click=lambda p=path: ui.download.file(p),
        ).props("flat dense round size=sm").tooltip("下载")

        ui.button(
            icon="folder_open",
            on_click=lambda p=path: _reveal(p),
        ).props("flat dense round size=sm").tooltip("在访达中显示")


def _reveal(path: str) -> None:
    """
    中文：在系统文件管理器中显示该文件。

    Reveal a file in the platform file manager.

    Useful because the whole point of writing translations next to the source
    PDF is that the user works with them in their own folders.

    Args:
        path (str): Absolute path to reveal.
    """
    import subprocess
    import sys

    try:
        if sys.platform == "darwin":
            subprocess.run(["open", "-R", path], check=False)
        elif sys.platform.startswith("linux"):
            subprocess.run(["xdg-open", str(Path(path).parent)], check=False)
        else:
            subprocess.run(["explorer", "/select,", path], check=False)
    except Exception:
        logger.warning("Could not reveal %s", path, exc_info=True)
        ui.notify("无法打开文件位置", type="negative")


@ui.page("/")
def home_page() -> None:
    """
    中文：首页——添加翻译来源与任务队列。

    Main page: submit new translations and watch the queue.

    Everything below the source card is rebuilt from
    :meth:`TranslatorService.dashboard` on a timer, so this page is safe to
    refresh at any point during a translation.
    """
    from local_translator.web.setup_page import needs_setup

    if needs_setup():
        # A packaged application opens straight into the wizard rather than
        # into an interface whose every button would fail.
        # 打包后的应用直接进入向导，而不是进入一个每个按钮都会失败的界面。
        ui.navigate.to("/setup")
        return

    service = get_service()
    _header("home")

    with _page_body():

        @ui.refreshable
        def jobs_section() -> None:
            """中文：渲染任务列表。 / Render the live job list."""
            data = service.dashboard()
            jobs = data["recent"]
            active = [j for j in jobs if j["status"] in LIVE_STATUSES]
            finished = [j for j in jobs if j["status"] not in LIVE_STATUSES][:10]

            with ui.card().classes(CARD):
                with ui.row().classes("items-center gap-2 mb-3"):
                    ui.label("任务队列").classes(HEADING)
                    if active:
                        ui.badge(str(len(active))).style(badge_style("running"))

                    queued = [j for j in active if j["status"] == "queued"]
                    if len(queued) > 1:
                        ui.space()

                        async def clear_queue() -> None:
                            count = await service.manager.clear_queue()
                            ui.notify(f"已清空 {count} 个排队任务", type="warning")
                            jobs_section.refresh()

                        ui.button("清空队列", on_click=clear_queue).props(
                            "flat dense no-caps"
                        ).classes("text-[13px]").style("color: #6b7280")

                if not active:
                    ui.label("当前没有进行中的任务").classes("text-sm text-gray-400 py-2")
                else:
                    with ui.column().classes("w-full gap-3"):
                        for job in active:
                            _job_row(service, job, jobs_section.refresh)

            if finished:
                with ui.card().classes(CARD):
                    with ui.row().classes("items-center gap-2 mb-3"):
                        ui.label("最近完成").classes(HEADING)
                    with ui.column().classes("w-full gap-3"):
                        for job in finished:
                            _job_row(service, job, jobs_section.refresh)

        _source_card(service, jobs_section.refresh)
        jobs_section()

        # Polling rather than push: the timer runs per client, so a refreshed or
        # newly opened tab immediately starts from current truth with no
        # subscription state to reconcile.
        # 用轮询而非推送：定时器随客户端存在，刷新或新开标签页都能立刻拿到最新状态。
        ui.timer(1.0, jobs_section.refresh)


# --------------------------------------------------------------------------- #
# Library page / 文库页
# --------------------------------------------------------------------------- #


@ui.page("/library")
def library_page() -> None:
    """
    中文：文库页——所有翻译过的文档及其产物。

    Library page listing every document known to the system.

    This is the durable history the previous prototype lacked entirely: it is
    read from SQLite, so it survives server restarts, not just page refreshes.
    """
    service = get_service()
    _header("library")

    with _page_body():

        # Which rows are ticked, and whether the batch is in selection mode.
        # Held outside the refreshable so a redraw does not lose the selection.
        # 勾选了哪些行、是否处于多选状态。放在 refreshable 之外，重绘时不会丢掉选择。
        state = {"selecting": False, "chosen": set()}

        @ui.refreshable
        def table() -> None:
            """中文：渲染文档列表。 / Render the document list."""
            documents = service.documents()
            known = {d["id"] for d in documents}
            state["chosen"] &= known

            with ui.card().classes(CARD):
                with ui.row().classes("items-center gap-2 mb-4 w-full"):
                    ui.label("历史").classes(HEADING)
                    ui.label(f"{len(documents)} 个文档").classes(MUTED)
                    ui.space()
                    if documents:
                        _library_toolbar(service, documents, state, table.refresh)

                if not documents:
                    with ui.column().classes("w-full items-center py-10 gap-2"):
                        ui.icon("inbox", size="2rem").classes("text-gray-300")
                        ui.label("还没有翻译过的文档").classes("text-sm text-gray-400")
                        ui.button("去翻译一篇", on_click=lambda: ui.navigate.to("/")).props(
                            "flat"
                        )
                    return

                for doc in documents:
                    _document_row(service, doc, table.refresh, state)

        table()


def _library_toolbar(service, documents, state, refresh) -> None:
    """
    中文：历史页顶部的多选与批量删除。

    Render the selection toggle and the batch delete.

    Args:
        service (TranslatorService): Service performing the removal.
        documents (list[dict]): The rows currently listed.
        state (dict): Shared ``{"selecting": bool, "chosen": set[str]}``.
        refresh (Callable[[], None]): Redraw callback.
    """
    if not state["selecting"]:
        def start() -> None:
            """中文：进入多选。 / Enter selection mode."""
            state["selecting"] = True
            refresh()

        ui.button("选择", icon="checklist", on_click=start).props(
            "flat dense no-caps size=sm")
        return

    chosen = state["chosen"]
    ui.label(f"已选 {len(chosen)} 项").classes(MUTED)

    def all_or_none() -> None:
        """中文：全选与全不选之间切换。 / Toggle select-all."""
        if len(chosen) == len(documents):
            chosen.clear()
        else:
            chosen.update(d["id"] for d in documents)
        refresh()

    ui.button("全选" if len(chosen) < len(documents) else "全不选",
              on_click=all_or_none).props("flat dense no-caps size=sm")

    async def remove() -> None:
        """中文：删掉选中的条目。 / Remove the ticked entries."""
        if not chosen:
            ui.notify("还没有选中任何条目", type="info")
            return
        await _delete_documents(service, sorted(chosen), batch=True)
        chosen.clear()
        state["selecting"] = False
        refresh()

    ui.button("删除", icon="delete_outline", on_click=remove).props(
        "flat dense no-caps size=sm").classes("text-[#c0392b]")

    def cancel() -> None:
        """中文：退出多选。 / Leave selection mode."""
        state["selecting"] = False
        chosen.clear()
        refresh()

    ui.button("取消", on_click=cancel).props("flat dense no-caps size=sm")


async def _delete_documents(service, doc_ids: list[str], *, batch: bool) -> None:
    """
    中文：删除历史条目；每一条只要产物文件还在，就问一次要不要连文件一起删。

    Remove library entries, asking about the files each one points at.

    The record and the file are different things and the dialog says so. A
    record costs nothing to recreate — translate the paper again. The PDF it
    points at may be an hour of local inference, so removing it is a separate
    answer, and the files go to the trash rather than being unlinked.

    An entry whose outputs are already gone is removed without a question:
    there is nothing to decide about.

    In a batch the first dialog carries a "stop asking" box. Ticking it applies
    *that same answer* to every remaining entry in this batch — including the
    answer "no, keep the files", which is the one a reader is most likely to
    want repeated.

    Args:
        service (TranslatorService): Service performing the removal.
        doc_ids (list[str]): Documents to remove.
        batch (bool): Whether this came from a multi-select, which is when the
            "stop asking" box is offered.

    Returns:
        None. The outcome is reported through a notification.
    """
    titles = {d["id"]: (d.get("title") or "(无标题)") for d in service.documents()}
    settled: bool | None = None          # the answer to reuse, once one is given
    plan: dict[bool, list[str]] = {True: [], False: []}

    for doc_id in doc_ids:
        files = service.document_files([doc_id])
        if not files:
            plan[False].append(doc_id)
            continue
        if settled is not None:
            plan[settled].append(doc_id)
            continue

        with ui.dialog() as dialog, ui.card().classes("w-[480px] gap-3 p-5"):
            ui.label("删除这条历史记录？").classes(HEADING)
            ui.label(titles.get(doc_id, doc_id)).classes(
                "text-[13.5px] font-medium").style("color: #37352f")
            ui.label(
                f"它指向的 {len(files)} 个产物文件还在磁盘上。"
                "只删记录的话文件会留下。"
            ).classes(MUTED)
            with ui.column().classes("gap-1 py-1 max-h-40 overflow-auto"):
                for path in files[:6]:
                    ui.label(short_path(str(path), keep=2)).classes(MUTED)
                if len(files) > 6:
                    ui.label(f"…另有 {len(files) - 6} 个").classes(MUTED)
            also = ui.checkbox("同时删除这些文件（移入废纸篓）")
            quiet = None
            if batch and len(doc_ids) > 1:
                quiet = ui.checkbox("本次批量删除的其余条目按同样处理，不再询问")
            with ui.row().classes("justify-end gap-2 w-full"):
                ui.button("取消", on_click=lambda: dialog.submit("cancel")).classes("lt-btn")
                ui.button("跳过这条", on_click=lambda: dialog.submit("skip")).props(
                    "flat no-caps")
                ui.button("删除", on_click=lambda: dialog.submit("go")).classes(
                    "lt-btn lt-btn-danger")

        answer = await dialog
        if answer == "cancel":
            break
        if answer != "go":
            continue
        plan[bool(also.value)].append(doc_id)
        if quiet is not None and quiet.value:
            settled = bool(also.value)

    total = {"documents": 0, "files": 0, "failed": 0}
    for remove_files, ids in plan.items():
        if not ids:
            continue
        result = await asyncio.to_thread(
            service.delete_documents, ids, remove_files=remove_files)
        for key in total:
            total[key] += result[key]

    if not total["documents"]:
        ui.notify("没有删除任何条目", type="info")
        return
    note = f"已删除 {total['documents']} 条记录"
    if total["files"]:
        note += f"，{total['files']} 个文件已移入废纸篓"
    if total["failed"]:
        note += f"（{total['failed']} 个文件删不掉）"
    ui.notify(note, type="positive")


def _document_row(service: TranslatorService, doc: dict, refresh,
                  state: dict | None = None) -> None:
    """
    中文：渲染文库中的一行文档。

    Render one document entry in the library.

    Args:
        service (TranslatorService): Service used for re-translation.
        doc (dict): Document row joined with its latest job.
        refresh (Callable[[], None]): Refresh callback.
        state (dict | None): Shared selection state from the library page, or
            ``None`` when the row is drawn somewhere that has no multi-select.
    """
    status = doc.get("last_status")
    label, colour = status_label(status) if status else ("未翻译", "grey")
    selecting = bool(state and state["selecting"])

    with ui.column().classes(f"{ROW} gap-2"):
        with ui.row().classes("w-full items-center gap-3"):
            if selecting:
                def pick(event, doc_id=doc["id"]) -> None:
                    """中文：勾选/取消这一行。 / Tick or untick this row."""
                    chosen = state["chosen"]
                    chosen.add(doc_id) if event.value else chosen.discard(doc_id)

                ui.checkbox(value=doc["id"] in state["chosen"],
                            on_change=pick).props("dense")
            ui.badge(label).style(badge_style(status or "queued"))
            ui.label(doc.get("title") or "(无标题)").classes(
                "flex-1 text-[13.5px] font-medium truncate"
            ).style("color: #37352f")

            def retranslate(path: str = doc["source_path"]) -> None:
                source = Path(path)
                if not source.exists():
                    ui.notify("源文件已不存在", type="negative")
                    return
                service.submit_path(source, source_kind=doc["source_kind"])
                ui.notify("已重新加入队列", type="positive")
                refresh()

            ui.button(icon="refresh", on_click=retranslate).props(
                "flat dense round size=sm"
            ).tooltip("重新翻译")

            if not selecting:
                async def remove(doc_id: str = doc["id"]) -> None:
                    """中文：删除这一条。 / Remove this entry."""
                    await _delete_documents(service, [doc_id], batch=False)
                    refresh()

                ui.button(icon="delete_outline", on_click=remove).props(
                    "flat dense round size=sm"
                ).classes("text-[#c0392b]").tooltip("从历史中删除")

        with ui.row().classes("items-center gap-4 flex-wrap"):
            ui.label(short_path(doc.get("source_path"), keep=3)).classes(MUTED)
            if doc.get("page_count"):
                ui.label(f"{doc['page_count']} 页").classes(MUTED)
            ui.label(human_bytes(doc.get("file_size"))).classes(MUTED)
            ui.label(doc.get("source_kind", "")).classes(MUTED)

        with ui.row().classes("items-center gap-4 flex-wrap"):
            if doc.get("dual_path"):
                _output_link("双语对照", doc["dual_path"], doc["id"],
                             mode="combined")
            if doc.get("mono_path"):
                _output_link("译文", doc["mono_path"], doc["id"], mode="mono")


# --------------------------------------------------------------------------- #
# Server / 服务启动
# --------------------------------------------------------------------------- #


def _register_lifecycle(service: TranslatorService) -> None:
    """
    中文：把服务的启动/关闭挂到 Web 应用生命周期上。

    Bind the service lifecycle to the web application's.

    Args:
        service (TranslatorService): Service to start and stop.
    """

    async def on_startup() -> None:
        await service.startup()
        write_pid(service.paths.pid_path)

    async def on_shutdown() -> None:
        clear_pid(service.paths.pid_path)
        await service.shutdown()

    app.on_startup(on_startup)
    app.on_shutdown(on_shutdown)


def _native_available() -> bool:
    """
    中文：判断是否装了 pywebview（原生窗口所需）。

    Report whether a desktop window can be opened.

    Returns:
        bool: ``True`` when ``pywebview`` imports, which is what NiceGUI's
            ``native`` mode is built on.
    """
    try:
        import webview  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


def _window_size() -> tuple[int, int]:
    """
    中文：读取桌面窗口尺寸，可用 LOCALTRANS_WINDOW 覆盖（如 1440x960）。

    Read the desktop window size.

    Returns:
        tuple[int, int]: Width and height in points, taken from
            ``$LOCALTRANS_WINDOW`` formatted as ``WIDTHxHEIGHT`` when it parses,
            and from the default otherwise.
    """
    raw = (os.getenv("LOCALTRANS_WINDOW") or "").lower().split("x")
    if len(raw) == 2 and all(part.strip().isdigit() for part in raw):
        return int(raw[0]), int(raw[1])
    return 1440, 960


def run(host: str | None = None, port: int | None = None, show: bool = True,
        native: bool = False) -> None:
    """
    中文：启动 Web 服务；native=True 时开一个独立的 macOS 窗口而不是浏览器标签页。

    Start the web server.

    Args:
        host (str | None): Bind address; defaults to the configured value.
        port (int | None): Port; defaults to the configured value.
        show (bool): Open a browser window on startup. Ignored when ``native``
            is set, which opens its own window.
        native (bool): Render into a desktop window (a WKWebView on macOS)
            instead of the default browser. Requires ``pywebview``, which is an
            optional dependency — the application is fully usable without it,
            so a missing package degrades to the browser with a warning rather
            than refusing to start.
    """
    settings = get_settings()
    service = get_service()
    _register_lifecycle(service)

    # Serve library and output PDFs to the reader. add_media_files supports HTTP
    # range requests, which pdf.js relies on to fetch a large PDF page by page
    # instead of downloading all of it before showing anything.
    # 供阅读器读取 PDF；add_media_files 支持 Range 请求，pdf.js 依赖它按页取数据。
    app.add_media_files("/files/library", service.paths.library_dir)

    from local_translator.web import pdfjs

    if pdfjs.is_installed():
        app.add_static_files("/static/pdfjs", pdfjs.pdfjs_dir())
    else:
        logger.warning("pdf.js is not installed; the reader will prompt to run setup-reader")

    # Imported for its @ui.page side effect / 导入以触发 @ui.page 注册
    from local_translator.api import register_api
    from local_translator.web import (  # noqa: F401
        dictionary_page,
        glossary_page,
        image_page,
        reader,
        settings_page,
        setup_page,
        text_page,
    )

    register_api(app, service)

    if native and not _native_available():
        logger.warning(
            "Native window requested but pywebview is not installed; "
            "falling back to the browser. Install it with: uv sync --extra native"
        )
        native = False

    options: dict = {}
    if native:
        # A desktop window binds to a private loopback port chosen by NiceGUI;
        # forcing the configured one keeps `localtrans stop`, the app bundle's
        # health check and the reader's media URLs all pointing at one place.
        # 桌面窗口模式下 NiceGUI 会自行挑一个本地端口；固定用配置里的端口，
        # 才能让 `localtrans stop`、应用包的健康检查与阅读器的媒体地址指向同一处。
        # ``show`` is passed by the call below for both modes; repeating it here
        # made NiceGUI raise TypeError before a window ever opened.
        # 下面的调用两种模式都会传 show；在这里重复一次会让 NiceGUI 在窗口打开之前
        # 就抛出 TypeError。
        options = {"native": True, "window_size": _window_size()}

    ui.run(
        title="Local Translator",
        favicon="📘",
        host=host or settings.server.host,
        port=port or settings.server.port,
        show=show and not native,
        reload=False,
        dark=False,
        **options,
    )
