"""
中文：双语阅读器页面——同步浏览（并排单文件）与自由浏览（双栏各自独立）两种模式。

Bilingual reader.

Two viewing modes, because the two useful ways to read a translated paper have
different requirements:

**同步浏览 (side-by-side file)** — the default.
    Opens the dual PDF that BabelDOC produces with
    ``use_alternating_pages_dual=False``: one wide page carrying the original and
    the translation next to each other, with the same page count as the source.
    Original and translation are physically locked together in the file, so a
    paragraph is always opposite its own translation and no mechanism can put
    them out of step. The same file can be copied to a tablet or any other
    reader and still works.

**自由浏览 (two panes)** — original on one side, translation-only PDF on the other.
    Two independent viewers: each pane has its own zoom, its own scroll position
    and its own search, which is what makes it the free one — worth having when
    the original carries small figures that need magnifying on their own. A
    「同步翻页」 switch can keep the two page numbers in step; turning it off
    leaves the panes fully independent.

The names describe how tightly the two halves are tied together, which is the
only thing that actually differs. Two earlier attempts named them after their
layout (拼版对照 / 双栏同步) and after their mechanism (自由浏览 for the combined
file, because it needs no synchronising) — both read backwards to a reader, who
sees one file whose halves can never drift and one pair of panes that can be
driven separately.

Both modes use the vendored pdf.js viewer (see :mod:`local_translator.web.pdfjs`)
rather than the browser's built-in PDF plugin, because page synchronisation
requires reading and setting the viewer's current page — which a native viewer
iframe does not expose.
"""

from __future__ import annotations

import logging
from pathlib import Path
from urllib.parse import quote

from nicegui import ui

from local_translator.service import get_service
from local_translator.web import pdfjs
from local_translator.web.theme import NOTION_CSS, apply_palette

logger = logging.getLogger(__name__)

MODE_COMBINED = "combined"
MODE_PANES = "panes"

# A single viewer on the translation alone. It was missing, and its absence was
# felt in two places: a document translated from LaTeX source has no page-for-
# page original to sit beside, and a reader who only wanted to read the Chinese
# had to open the bilingual file and ignore half of it.
# 只看译文的单窗口模式。以前没有这一项，两处都感觉得到：从 LaTeX 源码翻译出来的文档没有
# 一份逐页对应的原文可以并排；而只想读中文的人得打开双语版、然后无视其中一半。
MODE_MONO = "mono"


def pdf_stream_url(doc_id: str, kind: str) -> str:
    """
    中文：返回阅读器读取某个文档 PDF 的 URL。

    Build the URL the viewer uses to fetch one of a document's PDFs.

    Args:
        doc_id (str): Document identifier.
        kind (str): ``source``, ``dual`` or ``mono``.

    Returns:
        str: A path-only URL, free of percent-encoded separators so it survives
            being embedded in pdf.js's ``file=`` query parameter.
    """
    return f"/api/pdf/{quote(doc_id, safe='')}/{kind}"


def _viewer_src(doc_id: str, kind: str) -> str:
    """
    中文：拼装 pdf.js viewer 的 iframe src。

    Build the viewer iframe source for one of a document's PDFs.

    Args:
        doc_id (str): Document identifier.
        kind (str): ``source``, ``dual`` or ``mono``.

    Returns:
        str: Viewer URL.
    """
    return pdfjs.viewer_url(pdf_stream_url(doc_id, kind))


# JavaScript that keeps two pdf.js instances on the same page.
#
# Two details matter. The reentrancy guard: setting `page` on one viewer fires
# that viewer's own pagechanging event, which without the guard bounces straight
# back and fights the user's scrolling. And the wiring must be *invoked*, not run
# at load time — NiceGUI creates the iframes when the pane mode is rendered,
# which is after this script has already executed, so a one-shot DOMContentLoaded
# hook finds nothing and silently never attaches.
#
# 同步两个 pdf.js 实例。两个关键点：防重入锁（给一侧设页码会触发它自己的事件，
# 不加锁会来回弹跳）；以及接线必须由外部主动调用——iframe 是切换模式时才由
# NiceGUI 创建的，晚于本脚本执行，靠 DOMContentLoaded 挂钩会什么都找不到。
_SYNC_SCRIPT = """
<script>
(function () {
  let syncing = false;
  let enabled = true;
  const wired = new WeakSet();

  window.__ltSetSync = function (value) { enabled = value; };

  function attach(source, target) {
    if (wired.has(source)) { return true; }
    const app = source.contentWindow && source.contentWindow.PDFViewerApplication;
    if (!app || !app.initializedPromise) { return false; }
    wired.add(source);
    app.initializedPromise.then(function () {
      app.eventBus.on('pagechanging', function (evt) {
        if (!enabled || syncing) { return; }
        const other = target.contentWindow && target.contentWindow.PDFViewerApplication;
        if (!other || !other.pdfViewer) { return; }
        if (other.page === evt.pageNumber) { return; }
        syncing = true;
        try { other.page = evt.pageNumber; } catch (e) { /* viewer not ready */ }
        setTimeout(function () { syncing = false; }, 80);
      });
    });
    return true;
  }

  // Poll briefly: the iframes exist as soon as this is called, but their
  // pdf.js application object appears only once the viewer document loads.
  // 轮询一小段时间：iframe 已存在，但其 pdf.js 应用对象要等 viewer 文档加载完才有。
  window.__ltWire = function () {
    let tries = 0;
    const timer = setInterval(function () {
      const left = document.getElementById('lt-pane-left');
      const right = document.getElementById('lt-pane-right');
      const done = left && right && attach(left, right) && attach(right, left);
      if (done || ++tries > 60) { clearInterval(timer); }
    }, 250);
  };
})();
</script>
"""


def _resolve_files(doc_id: str) -> dict[str, str | None]:
    """
    中文：查出该文档最近一次成功翻译的产物路径。

    Find the most recent successful translation outputs for a document.

    Args:
        doc_id (str): Document identifier.

    Returns:
        dict[str, str | None]: Keys ``title``, ``source``, ``dual`` and ``mono``.
            Path values are ``None`` when the artefact was never produced or has
            since been deleted from disk.
    """
    service = get_service()
    document = service.db.get_document(doc_id)
    if document is None:
        return {"title": None, "source": None, "dual": None, "mono": None}

    job = service.db.query_one(
        """
        SELECT dual_path, mono_path FROM jobs
        WHERE doc_id = ? AND status = 'completed'
        ORDER BY finished_at DESC LIMIT 1
        """,
        (doc_id,),
    )

    def existing(value: str | None) -> str | None:
        """中文：文件还在就返回它。 / Return the path when the file is still there."""
        return value if value and Path(value).is_file() else None

    def viewable(value: str | None) -> str | None:
        """
        中文：只有 PDF 才能给阅读器看。

        Return the path only when it is something the viewer can open.

        A document translated from LaTeX source records its main ``.tex`` as
        the source file — there is no original PDF, because the paper was never
        printed before this. Handing that to pdf.js produces an empty viewer
        and nothing that says why.
        从 LaTeX 源码翻译的文档，记录的源文件是主 .tex——没有原始 PDF，因为这篇论文在
        此之前根本没有被印出来过。把它交给 pdf.js，得到的是一个空白窗口，没有任何东西
        说明原因。
        """
        found = existing(value)
        return found if found and found.lower().endswith(".pdf") else None

    return {
        "title": document["title"],
        "source": viewable(document["source_path"]),
        "dual": existing(job["dual_path"]) if job else None,
        "mono": existing(job["mono_path"]) if job else None,
    }


@ui.page("/reader/{doc_id}")
def reader_page(doc_id: str, mode: str = "") -> None:
    """
    中文：双语阅读器页面。

    Render the bilingual reader for one document.

    Args:
        doc_id (str): Document identifier taken from the URL.
        mode (str): Which view to open in — ``combined``, ``panes`` or
            ``mono``. Empty picks the best available. Passed from the output
            buttons so that clicking the reader icon beside 「译文」 opens the
            translation rather than the bilingual file.
            打开时用哪个视图。留空则按可用的产物挑一个。由产物按钮传入，好让「译文」旁边
            的阅读按钮打开的是译文，而不是双语版。
    """
    files = _resolve_files(doc_id)

    if not pdfjs.is_installed():
        _missing_viewer()
        return

    if not files["dual"] and not files["mono"]:
        _no_translation(files["title"])
        return

    ui.add_head_html(
        "<style>"
        "body { overflow: hidden; }"
        ".lt-viewer { width: 100%; height: 100%; border: 0; }"
        ".lt-stage { height: calc(100vh - 56px); }"
        "</style>"
    )
    ui.add_body_html(_SYNC_SCRIPT)

    available = {
        MODE_COMBINED: bool(files["dual"]),
        MODE_PANES: bool(files["source"] and files["mono"]),
        MODE_MONO: bool(files["mono"]),
    }
    if available.get(mode):
        default_mode = mode
    elif available[MODE_COMBINED]:
        default_mode = MODE_COMBINED
    elif available[MODE_PANES]:
        default_mode = MODE_PANES
    else:
        default_mode = MODE_MONO
    state = {"mode": default_mode, "sync": True}

    ui.add_head_html(NOTION_CSS)
    apply_palette()

    with ui.header().classes("px-4 py-1"):
        with ui.row().classes("w-full items-center gap-3"):
            ui.button(icon="arrow_back", on_click=lambda: ui.navigate.to("/library")).props(
                "flat dense round"
            ).tooltip("返回历史")
            ui.label(files["title"] or "阅读").classes(
                "text-[14px] font-medium truncate max-w-xl"
            ).style("color: #37352f")
            ui.space()

            options: dict[str, str] = {}
            if files["dual"]:
                options[MODE_COMBINED] = "同步浏览"
            if files["source"] and files["mono"]:
                options[MODE_PANES] = "自由浏览"
            if files["mono"]:
                options[MODE_MONO] = "纯译文"

            sync_toggle = ui.switch("同步翻页", value=True).props("dense")

            def on_sync(event) -> None:
                state["sync"] = bool(event.value)
                ui.run_javascript(
                    f"window.__ltSetSync && window.__ltSetSync({str(state['sync']).lower()})"
                )

            sync_toggle.on_value_change(on_sync)
            sync_toggle.visible = state["mode"] == MODE_PANES

            # An inline toggle rather than a dropdown: with only two modes a
            # popup is an extra click for nothing, and the reader sets
            # `body { overflow: hidden }` to keep the viewer full-height, which
            # interferes with how Quasar positions its select menu.
            # 两个模式用内联切换即可；且阅读器为撑满高度设了 body overflow:hidden，
            # 会干扰 Quasar 下拉弹层的定位。
            if len(options) > 1:
                mode_toggle = ui.toggle(options, value=state["mode"]).props(
                    "dense unelevated no-caps toggle-color=primary"
                )
                mode_toggle.tooltip(
                    "同步浏览：双语并排的同一个文件，原文与译文永远对齐；"
                    "自由浏览：原文与译文各开一个阅读器，缩放与滚动互不影响"
                )

                def on_mode(event) -> None:
                    if not event.value:
                        return
                    state["mode"] = event.value
                    sync_toggle.visible = event.value == MODE_PANES
                    stage.refresh()

                mode_toggle.on_value_change(on_mode)

            downloadable = files["dual"] or files["mono"]
            if downloadable:
                ui.button(
                    icon="download",
                    on_click=lambda path=downloadable: ui.download.file(path),
                ).props("flat dense round").tooltip(
                    "下载双语 PDF" if files["dual"] else "下载译文 PDF")

    def _viewer(kind: str, element_id: str | None = None):
        """
        中文：创建一个指向 pdf.js viewer 的 iframe 元素。

        Create an iframe element pointing at the vendored pdf.js viewer.

        Built as a real element rather than through :func:`ui.html`, whose
        default client-side sanitisation strips ``<iframe>`` outright — the
        container renders but stays empty, with nothing logged to explain it.

        Args:
            kind (str): ``source``, ``dual`` or ``mono``.
            element_id (str | None): DOM id, needed by the pane-synchronisation
                script to find the two viewers.

        Returns:
            ui.element: The iframe element.
        """
        frame = ui.element("iframe").classes("lt-viewer").props(
            f'src="{_viewer_src(doc_id, kind)}"'
        )
        if element_id:
            frame.props(f'id={element_id}')
        return frame

    @ui.refreshable
    def stage() -> None:
        """中文：渲染阅读区域。 / Render the viewer area for the current mode."""
        if state["mode"] == MODE_COMBINED and files["dual"]:
            with ui.element("div").classes("w-full lt-stage"):
                _viewer("dual")
            return

        if state["mode"] == MODE_MONO or not (files["source"] and files["mono"]):
            with ui.element("div").classes("w-full lt-stage"):
                _viewer("mono")
            return

        with ui.row().classes("w-full lt-stage gap-0 flex-nowrap"):
            with ui.column().classes("flex-1 h-full min-w-0"):
                _viewer("source", "lt-pane-left")
            with ui.column().classes("flex-1 h-full min-w-0 border-l border-gray-200"):
                _viewer("mono", "lt-pane-right")

        # Wire the panes together once the browser has actually created them.
        # 面板真正建出来之后再接线。
        ui.timer(0.3, lambda: ui.run_javascript("window.__ltWire && window.__ltWire()"), once=True)

    stage()


def _missing_viewer() -> None:
    """
    中文：pdf.js 未安装时的提示页。

    Render guidance when the vendored viewer has not been downloaded yet.
    """
    with ui.column().classes("w-full items-center justify-center gap-3 py-24"):
        ui.icon("extension_off", size="3rem").classes("text-gray-300")
        ui.label("阅读器组件尚未安装").classes("text-lg font-medium text-gray-700")
        ui.label("运行 `localtrans setup-reader` 下载 pdf.js 后即可使用").classes(
            "text-sm text-gray-500"
        )
        ui.button("返回历史", on_click=lambda: ui.navigate.to("/library")).props("flat")


def _no_translation(title: str | None) -> None:
    """
    中文：该文档尚无译文产物时的提示页。

    Render guidance when a document has no usable translation output.

    Args:
        title (str | None): Document title, if known.
    """
    with ui.column().classes("w-full items-center justify-center gap-3 py-24"):
        ui.icon("menu_book", size="3rem").classes("text-gray-300")
        ui.label(title or "该文档").classes("text-lg font-medium text-gray-700")
        ui.label("还没有可阅读的译文，或产物文件已被移动").classes("text-sm text-gray-500")
        ui.button("返回历史", on_click=lambda: ui.navigate.to("/library")).props("flat")
