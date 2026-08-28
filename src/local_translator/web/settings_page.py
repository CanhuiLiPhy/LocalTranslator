"""
中文：设置页——翻译、模型、选择性翻译、输出版式、术语库、API 与累计用量。

Settings page.

Grouped by what the user is deciding rather than by which object stores the
value, so related switches sit together even when they live in different
settings models. Changes are written on save, and the few that need the
inference server restarted say so rather than appearing to have no effect.

API keys never appear here as stored values: the field writes to the system
keychain and afterwards reports only whether a key is configured.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from nicegui import ui

from local_translator import edition
from local_translator.config import LANGUAGES, LAYOUT_PRESETS, SKIPPABLE_LABELS
from local_translator.service import get_service
from local_translator.web.format import display_path, human_tokens
from local_translator.web.theme import (
    BLUE,
    BTN_PRIMARY,
    CARD,
    GREEN,
    HEADING,
    INK_SOFT,
    INPUT_PROPS,
    META,
    MUTED,
    NOTION_CSS,
    SURFACE_ALT,
    apply_palette,
)

logger = logging.getLogger(__name__)

# Human-readable names for the layout regions selective translation can skip.
# 选择性翻译可跳过的版面区域中文名。
# Only the classes the layout model actually emits, named as a reader thinks of
# them. The map used to carry twenty-seven entries inherited from other layout
# backends; seventeen of them could never match anything, so the page offered a
# column of switches that changed nothing.
# 仅列出版面模型确实会输出的类别，并用读者的说法命名。此表原有二十七项，是从其它版面
# 后端沿用来的，其中十七项永远匹配不到任何内容——设置页那一列不起作用的开关正源于此。
LAYOUT_LABEL_NAMES: dict[str, str] = {
    "title": "各级标题",
    "plain text": "正文",
    "abandon": "页眉页脚与页码",
    "figure": "图片",
    "figure_caption": "图注",
    "table": "表格",
    "table_caption": "表注",
    "table_footnote": "表格脚注",
    "isolate_formula": "行间公式",
    "formula_caption": "公式编号",
    # Recognised by reading the page rather than by a layout class; the model
    # has no category for any of these.
    # 以下四项靠阅读页面识别而非版面类别——模型没有对应的类目。
    "document_title": "文章标题",
    "author": "作者",
    "affiliation": "作者单位",
    "footnote": "脚注",
    "reference": "参考文献",
    "algorithm": "伪代码 / 算法块",
}

# The kinds put back from the source verbatim, in the order they are shown.
# 原样搬回的内容类型，按显示顺序排列。
PRESERVE_KIND_NAMES: dict[str, str] = {
    "figure": "图片",
    "table": "表格",
    "formula": "行间公式",
    "algorithm": "伪代码 / 算法块",
    "contents": "目录页",
}

PRESET_NAMES: dict[str, str] = {
    "pdfmath": "PDFMathTranslate 风格",
    "conservative": "保守（只跳页眉页脚与参考文献）",
    "everything": "全部翻译",
    "custom": "自定义",
}



def _section(title: str, hint: str = ""):
    """
    中文：渲染一个设置分组的标题行。

    Render a settings section heading.

    Args:
        title (str): Section title.
        hint (str): Optional one-line explanation shown beside the title.
    """
    with ui.row().classes("items-baseline gap-3 mb-3"):
        ui.label(title).classes(HEADING)
        if hint:
            ui.label(hint).classes(MUTED)


@ui.page("/settings")
def settings_page() -> None:
    """
    中文：设置页面。

    Render the settings page.
    """
    from local_translator.web.app import _header, _page_body

    service = get_service()
    # Edited on a copy. The controls bind straight to the model objects, so
    # binding them to the service's own settings would make every keystroke take
    # effect at once: a half-finished edit would be snapshotted into whatever
    # translation was submitted next, and leaving the page without saving would
    # not undo anything. A running job is unaffected either way — its settings
    # were snapshotted into the job row when it was submitted — but what the next
    # job runs with should be what the reader saved, not what they last touched.
    # 在副本上编辑。控件是直接绑到配置对象上的，若绑服务自身的配置，任何一次改动都会
    # 立即生效：改到一半的设置会被下一个提交的翻译快照走，而离开页面不保存也撤销不了
    # 什么。正在跑的任务两种做法都不受影响——它的设置在提交时已快照进任务行——但下一个
    # 任务应当按读者保存过的设置来跑，而不是按他最后碰过的那一下。
    s = service.settings.model_copy(deep=True)

    _header("settings")
    ui.add_head_html(NOTION_CSS)
    apply_palette()

    # Edits mutate the live settings models directly and are persisted on save,
    # so a mistaken change is undone by leaving without saving.
    # 编辑直接改内存中的配置对象，保存时才落盘；改错了不保存即可作废。
    with _page_body():
        # Ordered by how often a reader touches them: the language pair, what
        # comes out, which regions, the service and its usage, where files live
        # — and then everything that is about the machine rather than the
        # translation, under one heading at the bottom.
        # 按读者动它们的频率排：语言对、出什么、翻哪些区域、用哪个服务及其用量、文件放
        # 哪——然后是所有「关乎机器而非翻译」的东西，集中在最下面一栏。
        _translation_card(service, s)
        _output_card(s)
        _layout_card(s)
        _api_card(s, service)
        _paths_card(service, s)
        _developer_card(service, s)

        with ui.row().classes("w-full justify-end gap-3 pb-6"):
            def save() -> None:
                service.update_settings(s.model_copy(deep=True))
                ui.notify("设置已保存，下次翻译起生效", type="positive")

            ui.label(
                "保存后从下一个任务生效；模型与并发的改动需重启推理服务。"
            ).classes(MUTED)
            ui.button("保存设置", on_click=save).props("no-caps unelevated").classes(
                BTN_PRIMARY
            )


def _translation_card(service, s) -> None:
    """
    中文：翻译基本设置——语言对与后端。

    Render language pair and backend selection.

    Args:
        service (TranslatorService): Service, used to report API readiness.
        s (Settings): Live settings object being edited.
    """
    with ui.card().classes(CARD):
        _section("语言设置")

        with ui.row().classes("w-full gap-4 flex-wrap"):
            ui.select(LANGUAGES, label="源语言", with_input=True).classes("w-52").props(INPUT_PROPS).bind_value(
                s.translation, "lang_in"
            )
            ui.select(LANGUAGES, label="目标语言", with_input=True).classes("w-52").props(INPUT_PROPS).bind_value(
                s.translation, "lang_out"
            )


        if edition.is_lite():
            ui.label(
                f"{edition.display_name()}：全部段落交给下方配置的云端 API。"
            ).classes(MUTED)
        elif edition.current() == edition.MINI:
            ui.label(
                f"{edition.display_name()}：本地 1.8B 模型，约 1.1 GB。"
            ).classes(MUTED)

        ui.textarea(
            "附加指令（可选）", placeholder="例如：保持被动语态；数值单位一律保留原文"
        ).classes("w-full").props(f"{INPUT_PROPS} rows=2").bind_value(
            s.translation, "custom_system_prompt"
        )


async def _confirm_keep(service, s, box, wanted: bool) -> None:
    """
    中文：打开「保留中间产物」之前先说清楚它有多占地方，读者点了确认才真的打开。

    Make the cost of keeping intermediates plain before accepting the choice.

    Turning this on is cheap to click and expensive to live with — BabelDOC
    writes the whole document out again at every stage, so a few dozen papers
    run into gigabytes — and the growth is invisible until a disk fills. So the
    switch asks, and the question quotes the number from this machine rather
    than a general warning: what is already there is the most convincing
    estimate of what will be there next month.

    Turning it *off* asks nothing. It is the default, and it only ever frees
    space.

    Args:
        service (TranslatorService): Service, for the directories to measure.
        s (Settings): Live settings object being edited.
        box (ui.checkbox): The switch itself, reset when the reader declines.
        wanted (bool): The value the reader just clicked.

    Returns:
        None. The settings object is updated in place, either way.
    """
    if not wanted:
        s.translation.keep_intermediates = False
        return

    from local_translator.engine import intermediates

    used = intermediates.size_of(service.paths.meta_dir)
    used += intermediates.size_of(service.paths.library_dir / "_latex")

    with ui.dialog() as dialog, ui.card().classes("w-[460px] gap-3 p-5"):
        ui.label("确定不再自动清理吗？").classes(HEADING)
        ui.label(
            "中间产物是各阶段的文档副本，任务结束后不再被读取。"
            "一篇论文最多留下 705 MB，译文本身只有 12 MB。"
        ).classes(MUTED)
        ui.label(f"当前已占用 {used / 1073741824:.1f} GB。").classes(MUTED)
        with ui.row().classes("justify-end gap-2 w-full"):
            ui.button("取消", on_click=dialog.close).classes("lt-btn")
            ui.button("仍然保留", on_click=lambda: dialog.submit(True)).classes(
                "lt-btn lt-btn-danger"
            )

    if await dialog:
        s.translation.keep_intermediates = True
        return

    s.translation.keep_intermediates = False
    box.value = False


def _developer_card(service, s) -> None:
    """
    中文：开发者选项——本地模型、输出细节、缓存、术语库、LaTeX 配置。

    Render the settings a reader does not normally need.

    Everything here is either about how the machine does the work rather than
    what the reader wants (the local model's sampling and concurrency, the
    paragraph cache, whether intermediates are kept) or a lever that can make
    the output worse if turned the wrong way (skipping the structural clean,
    compressing the output). They are gathered under one heading so the pages
    above stay about the translation itself.

    Args:
        service (TranslatorService): Service, used to show what is installed.
        s (Settings): Live settings object being edited.
    """
    status = service.model_status()

    with ui.card().classes(CARD):
        _section("开发者选项")

        # The lite edition ships no weights, so every control below edits a
        # setting nothing reads. The rest of this card still applies to it.
        # 轻量版不带权重，下面每个控件改的都是无人读取的设置。这张卡片其余部分对它仍然有效。
        if not edition.is_lite():
            ui.label(f"本地模型 · 当前加载：{status['model']}").classes(META)
            with ui.row().classes("gap-3 flex-wrap mb-2"):
                with ui.column().classes("gap-0 px-4 py-2 rounded-md").style(
                    f"background: {SURFACE_ALT}"
                ):
                    ui.label("已就绪" if status["healthy"] else "未启动").classes(
                        "text-[15px] font-semibold"
                    ).style(f"color: {GREEN if status['healthy'] else INK_SOFT}")
                    ui.label("推理服务").classes(MUTED)
                if status.get("speculative"):
                    with ui.column().classes("gap-0 px-4 py-2 rounded-md").style(
                        f"background: {SURFACE_ALT}"
                    ):
                        ui.label("已启用").classes("text-[15px] font-semibold").style(f"color: {BLUE}")
                        ui.label(f"投机解码 · {status['speculative']}").classes(MUTED)

            async def toggle_speculative(event) -> None:
                """中文：开关投机解码并重启服务。 / Toggle speculative decoding."""
                try:
                    await asyncio.to_thread(service.set_speculative, bool(event.value))
                except Exception as exc:  # noqa: BLE001
                    ui.notify(f"切换失败：{exc}", type="negative")
                    return
                ui.notify("已重启推理服务" if status.get("healthy") else "设置已保存",
                          type="positive")

            # Speculative decoding needs a smaller model of the same family to
            # draft with. In the mini edition the 1.8B is already the main model and
            # there is nothing smaller, so the switch would toggle a setting nothing
            # can act on.
            # 投机解码需要一个同族的更小模型来出草稿。本地轻量版里 1.8B 本身就是主模型，
            # 没有更小的可用，这个开关改的是一个无人能执行的设置。
            if edition.supports_speculative():
                spec = ui.switch("投机解码", value=s.local_model.speculative)
                spec.on_value_change(toggle_speculative)

            ui.separator().classes("my-3")
            _section("采样参数")

            with ui.row().classes("w-full gap-4 flex-wrap"):
                ui.number("温度", min=0.0, max=1.5, step=0.05, format="%.2f").classes(
                    "w-32"
                ).props(INPUT_PROPS).bind_value(s.local_model, "temperature")
                ui.number("top_p", min=0.1, max=1.0, step=0.05, format="%.2f").classes(
                    "w-32"
                ).props(INPUT_PROPS).bind_value(s.local_model, "top_p")
                ui.number("top_k", min=1, max=100).classes("w-32").props(INPUT_PROPS).bind_value(
                    s.local_model, "top_k"
                )
                ui.number("重复惩罚", min=1.0, max=1.5, step=0.01, format="%.2f").classes(
                    "w-32"
                ).props(INPUT_PROPS).bind_value(s.local_model, "repeat_penalty")

            ui.label(
                "温度默认 0.2 而非官方推荐的 0.7：高温下 {v1} 这类公式占位符更易被改写。"
            ).classes(MUTED)

            ui.separator().classes("my-3")
            _section("内存")

            ui.checkbox("低内存模式").bind_value(s, "low_memory")

            with ui.row().classes("w-full gap-4 flex-wrap items-center mt-2"):
                ui.number("空闲多久卸载模型（秒，0=不卸载）", min=0, max=3600, step=30).classes(
                    "w-72"
                ).props(INPUT_PROPS).bind_value(s.local_model, "idle_unload_seconds")

            with ui.row().classes("w-full gap-4 flex-wrap items-center mt-2"):
                ui.switch("Flash Attention").bind_value(s.local_model, "flash_attention")
                ui.select({"": "f16（默认）", "q8_0": "q8_0", "q4_0": "q4_0"},
                          label="KV 缓存精度").classes("w-44").props(INPUT_PROPS).bind_value(
                    s.local_model, "kv_cache_type"
                )
                ui.number("批大小 -b（0=默认）", min=0, max=4096, step=128).classes(
                    "w-44"
                ).props(INPUT_PROPS).bind_value(s.local_model, "batch_size")
                ui.number("微批 -ub（0=默认）", min=0, max=1024, step=32).classes(
                    "w-44"
                ).props(INPUT_PROPS).bind_value(s.local_model, "ubatch_size")

            ui.separator().classes("my-3")
            _section("并发")

            with ui.row().classes("w-full gap-4 flex-wrap"):
                ui.number("推理并发槽位", min=1, max=32).classes("w-40").props(
                    INPUT_PROPS
                ).bind_value(s.local_model, "n_parallel")
                ui.number("段落工作线程", min=1, max=64).classes("w-40").props(
                    INPUT_PROPS
                ).bind_value(s.translation, "pool_max_workers")
                ui.number("同时翻译文档数", min=1, max=4).classes("w-40").props(
                    INPUT_PROPS
                ).bind_value(s.server, "max_concurrent_jobs")

        ui.separator().classes("my-4")
        _section("产物细节")

        with ui.row().classes("w-full gap-4 flex-wrap items-center"):
            ui.select(
                {0: "不压缩", 300: "300 DPI（印刷级）", 200: "200 DPI（屏幕阅读）"},
                label="压缩输出 PDF",
            ).classes("w-56").props(INPUT_PROPS).bind_value(s.pdf, "compress_dpi")

        with ui.row().classes("w-full gap-6 flex-wrap mt-2"):
            ui.checkbox("跳过 PDF 结构清理（略快，文件略大）").bind_value(s.pdf, "skip_clean")
            ui.checkbox("自动 OCR 扫描页").bind_value(s.pdf, "auto_enable_ocr_workaround")

        _cache_section(service, s)
        _glossary_section(s)
        _latex_options(service, s)



def _cache_section(service, s) -> None:
    """
    中文：翻译缓存——上限、忽略开关、中间产物保留，以及立即清空。

    Render the cache controls.

    The size shown is the one measured, not a guess: a paper produces a median
    of 689 cache entries at 2,209 bytes each, so about 1.5 MB. What that buys is
    also measured, and it is narrower than it sounds — across *different* papers
    the cache answers 2.68% of blocks by count and none at all of the blocks of
    real prose; re-translating the *same* paper after a settings change, it
    answers 78.9%. So the number here is really "how many papers am I still
    working on", not "how much translation am I keeping".

    Args:
        service (TranslatorService): Service, for the cache's current size.
        s (Settings): Live settings object being edited.
    """
    from local_translator.engine import intermediates

    ui.separator().classes("my-4")
    _section("翻译缓存")

    cache = intermediates.babeldoc_cache_path()
    label = ui.label("").classes(META)

    def describe() -> None:
        """中文：把当前占用写成一句话。 / State the current size."""
        try:
            now = cache.stat().st_size / 1048576 if cache.is_file() else 0.0
        except OSError:
            label.text = "一篇论文约 1.5 MB，主要在重译同一篇时命中。"
            return
        label.text = (f"当前 {now:.0f} MB · 一篇论文约 1.5 MB，"
                      "主要在重译同一篇时命中。")

    describe()

    with ui.row().classes("w-full gap-4 flex-wrap items-end"):
        ui.number("退出时保留上限（MB，0 表示清空）", min=0, max=2048, step=10).classes(
            "w-72"
        ).props(INPUT_PROPS).bind_value(s.translation, "translation_cache_mb")

    ui.checkbox("忽略翻译缓存（每段都重新翻译）").bind_value(s.translation, "ignore_cache")

    keep = ui.checkbox("保留翻译中间产物（不自动清理）",
                       value=s.translation.keep_intermediates)
    keep.on_value_change(lambda event: _confirm_keep(service, s, keep, event.value))
    ui.label("默认每次翻译后清空，只留逐段对照记录。").classes(MUTED)

    async def drop() -> None:
        """中文：立即清空段落缓存。 / Empty the paragraph cache now."""
        freed = await asyncio.to_thread(intermediates.clear_translation_cache, cache)
        describe()
        ui.notify(f"已清空，释放 {freed / 1048576:.0f} MB" if freed else "缓存本来就是空的",
                  type="positive" if freed else "info")

    ui.button("立即清空翻译缓存", on_click=drop).props("no-caps flat")


def _layout_card(s) -> None:
    """
    中文：选择性翻译设置——按版面区域跳过。

    Render selective-translation controls.

    Args:
        s (Settings): Live settings object being edited.
    """
    with ui.card().classes(CARD):
        _section("选择性翻译", "哪些区域保留原文不译")

        ui.checkbox("启用选择性翻译").bind_value(s.layout_filter, "enabled")

        @ui.refreshable
        def controls() -> None:
            """中文：渲染预设与逐项开关。 / Render preset and per-label switches."""
            preset = ui.select(PRESET_NAMES, label="预设").classes("w-72").props(INPUT_PROPS)
            preset.value = s.layout_filter.preset

            def on_preset(event) -> None:
                s.layout_filter.preset = event.value
                if event.value in LAYOUT_PRESETS:
                    s.layout_filter.skip_labels = list(LAYOUT_PRESETS[event.value])
                controls.refresh()

            preset.on_value_change(on_preset)

            active = s.layout_filter.effective_skip_labels()
            ui.label(
                f"当前跳过 {len(active)} 类区域"
                + ("（选择「自定义」后可逐项调整）" if s.layout_filter.preset != "custom" else "")
            ).classes(META)

            editable = s.layout_filter.preset == "custom"
            with ui.row().classes("w-full gap-x-6 gap-y-1 flex-wrap mt-2"):
                for label in SKIPPABLE_LABELS:
                    box = ui.checkbox(LAYOUT_LABEL_NAMES.get(label, label), value=label in active)
                    box.set_enabled(editable)

                    def on_toggle(event, name=label) -> None:
                        labels = set(s.layout_filter.skip_labels)
                        labels.add(name) if event.value else labels.discard(name)
                        s.layout_filter.skip_labels = sorted(labels)

                    box.on_value_change(on_toggle)

        controls()

        ui.separator().classes("my-4")
        _section("原样保留")
        with ui.row().classes("w-full gap-x-6 gap-y-1 flex-wrap"):
            for kind, name in PRESERVE_KIND_NAMES.items():
                selected = kind in s.pdf.preserve_kinds
                box = ui.checkbox(name, value=selected)

                def on_kind(event, key=kind) -> None:
                    kinds = set(s.pdf.preserve_kinds)
                    kinds.add(key) if event.value else kinds.discard(key)
                    s.pdf.preserve_kinds = [
                        k for k in PRESERVE_KIND_NAMES if k in kinds
                    ]

                box.on_value_change(on_kind)

        from local_translator.engine import headings as _headings

        chapter = ui.checkbox("章节词", value=not s.pdf.translate_headings)
        chapter.on_value_change(
            lambda e: setattr(s.pdf, "translate_headings", not e.value))
        ui.label(
            f"「摘要」「引言」等 {len(_headings.known())} 个标准章节标题查表直出，编号保留。"
        ).classes(MUTED)

        toc = ui.checkbox("目录页逐行原位翻译", value=s.pdf.translate_contents)
        toc.on_value_change(lambda e: setattr(s.pdf, "translate_contents", e.value))


def _paths_card(service, s) -> None:
    """
    中文：文件路径设置——译文与文库放哪、模型权重挂在哪。

    Render the directory settings.

    Two of the three are the reader's to choose. The library is where
    translations and downloaded papers land, and a reader with a small startup
    disk wants it on the external one. The model root is where several
    gigabytes of weights live, and a reader who already has them — from this
    application on another disk, or from another tool entirely — wants to point
    at that copy rather than download them again.

    The application's own data directory is shown and not editable. Moving it
    means moving an open database, and the honest way to offer that is not a
    text field on a settings page.

    **中文：** 三个里有两个该由读者定。文库是译文与下载的论文的落脚处，系统盘小的人会想把
    它放到外置盘上；模型根目录里躺着几个 G 的权重，已经有一份的人——不管是本程序装在另一
    块盘上的，还是别的工具下的——会想直接指过去，而不是重下一遍。

    应用自己的数据目录只显示、不可改。挪它等于挪一个正在使用的数据库，而提供这件事的诚实
    做法不是设置页上的一个输入框。

    Args:
        service (TranslatorService): Service, for the paths in use now.
    """
    from local_translator.config import read_directory_choices, write_directory_choices

    paths = service.paths
    chosen = read_directory_choices()

    with ui.card().classes(CARD):
        _section("文件路径", "改完需要重启程序才生效")

        library = (
            ui.input("译文与下载文件的保存位置",
                     placeholder=str(Path.home() / "Documents" / "LocalTranslator" / "library"))
            .classes("w-full").props(INPUT_PROPS)
        )
        library.value = chosen.get("library_dir", "")
        ui.label(f"当前：{paths.library_dir}").classes(META)

        models = (
            ui.input("模型权重挂载位置", placeholder=str(Path.home() / "LLM"))
            .classes("w-full").props(INPUT_PROPS)
        )
        models.value = chosen.get("llm_dir", "")
        ui.label(f"当前：{paths.llm_dir}（翻译模型在 {paths.models_dir.name}/，"
                 f"OCR 模型在 {paths.ocr_models_dir.name}/）").classes(META)

        ui.label(f"程序数据（数据库、配置、日志）：{paths.data_dir}").classes(MUTED)
        ui.label("留空用默认位置。已有译文不搬动，只有新产生的去新位置。").classes(MUTED)

        _intermediates_row(service)
        _latex_paths(service, s)

        def save_paths() -> None:
            """中文：写入目录选择。 / Record the directory choices."""
            wanted = {"library_dir": (library.value or "").strip(),
                      "llm_dir": (models.value or "").strip()}
            for key, value in wanted.items():
                if not value:
                    continue
                target = Path(value).expanduser()
                if target.exists() and not target.is_dir():
                    ui.notify(f"{target} 不是文件夹", type="negative")
                    return
                try:
                    target.mkdir(parents=True, exist_ok=True)
                except OSError as problem:
                    ui.notify(f"建不了这个文件夹：{problem}", type="negative")
                    return
                wanted[key] = str(target.resolve())
            write_directory_choices(wanted)
            ui.notify("路径已保存，重启程序后生效", type="positive")

        with ui.row().classes("gap-3 mt-1"):
            ui.button("保存路径", on_click=save_paths).props(
                "no-caps unelevated outline")
            ui.button("在访达中打开",
                      on_click=lambda: _reveal(paths.library_dir)).props(
                "no-caps flat")


def _intermediates_row(service) -> None:
    """
    中文：显示中间产物占了多少地方，并提供立即清理。

    Show how much the leftovers take and offer to clear them now.

    Automatic cleanup only reaches jobs that run after it was turned on, so a
    reader upgrading into it still has whatever accumulated before — 4.1 GB
    across sixty papers, on the machine this was written on. This is the button
    for that backlog, and for the case where someone kept the intermediates
    deliberately and is now done with them.

    A document that is being translated right now is left alone: its working
    directory is in use, and removing it would fail the run.

    Args:
        service (TranslatorService): Service, for the directories and for which
            jobs are in flight.

    Returns:
        None.
    """
    from local_translator.engine import intermediates

    paths = service.paths
    latex_dir = paths.library_dir / "_latex"

    def measure() -> str:
        """中文：把当前占用写成一句话。 / Describe the current usage in one line."""
        used = intermediates.size_of(paths.meta_dir) + intermediates.size_of(latex_dir)
        if used >= 1073741824:
            return f"翻译中间产物：{used / 1073741824:.1f} GB"
        return f"翻译中间产物：{used / 1048576:.0f} MB"

    # Measured off the render path, never on it. Walking a directory means
    # `opendir`, and on macOS `opendir` under ~/Documents blocks until the user
    # answers the files-and-folders prompt — which a freshly installed build has
    # not been asked yet. Doing it inline hung the settings page indefinitely:
    # the sample showed the main thread parked in `os_scandir → open$NOCANCEL`.
    # A settings page must render whether or not the filesystem answers.
    # 放在渲染路径之外测量，绝不放在路径之上。遍历目录就要 opendir，而在 macOS 上对
    # ~/Documents 底下 opendir 会一直阻塞，直到用户回答「文件与文件夹」授权弹窗——刚装好
    # 的新构建还没被问过。放在渲染里做会把设置页无限期挂住：采样显示主线程停在
    # os_scandir → open$NOCANCEL 上。设置页必须能渲染出来，不管文件系统答不答话。
    label = ui.label("翻译中间产物：统计中…").classes(META)

    async def refresh_size() -> None:
        """中文：后台统计占用并回填。 / Measure in the background and fill in."""
        try:
            label.text = await asyncio.wait_for(asyncio.to_thread(measure), timeout=20)
        except asyncio.TimeoutError:
            label.text = "翻译中间产物：统计超时（可能未授权访问该目录）"
        except Exception:  # noqa: BLE001
            label.text = "翻译中间产物：读不到"

    ui.timer(0.1, refresh_size, once=True)

    async def clear() -> None:
        """中文：清理全部历史中间产物。 / Clear every job's leftovers."""
        busy = {r["doc_id"] for r in service.manager.snapshot()["running"]}
        freed, touched = await asyncio.to_thread(
            intermediates.sweep_all, paths.meta_dir, latex_dir, skip=busy)
        try:
            label.text = await asyncio.wait_for(asyncio.to_thread(measure), timeout=20)
        except Exception:  # noqa: BLE001
            label.text = "翻译中间产物：读不到"
        if not freed:
            ui.notify("没有可清理的中间产物", type="info")
            return
        note = f"已清理 {freed / 1073741824:.1f} GB" if freed >= 1073741824 \
            else f"已清理 {freed / 1048576:.0f} MB"
        if busy:
            note += f"（{len(busy)} 篇正在翻译，已跳过）"
        ui.notify(f"{note}，涉及 {touched} 个文档", type="positive")

    ui.button("立即清理中间产物", on_click=clear).props("no-caps flat")


def _reveal(target) -> None:
    """
    中文：在访达里打开一个目录。

    Open a directory in the system file browser.

    Args:
        target (Path): The directory to show.
    """
    import subprocess
    import sys as _sys

    try:
        if _sys.platform == "darwin":
            subprocess.Popen(["open", str(target)])
        elif _sys.platform.startswith("win"):
            subprocess.Popen(["explorer", str(target)])
        else:
            subprocess.Popen(["xdg-open", str(target)])
    except Exception:  # noqa: BLE001
        logger.warning("打不开 %s", target, exc_info=True)
        ui.notify("打不开这个位置", type="warning")


def _latex_paths(service, s) -> None:
    """
    中文：LaTeX 的路径部分——引擎在哪、用哪款中文字体，画在「文件路径」里。

    Render the LaTeX path settings, inside the file-paths card.

    An engine path and a font family are the same kind of question as where the
    library lives: they say where something is on this machine. The rest of the
    LaTeX settings — whether arXiv prefers source, how many blocks run at once,
    whether the build directory is kept — are about how the work is done, and
    live under the developer heading.

    Args:
        service (TranslatorService): Service, used to report the engine found.
        s (Settings): Live settings object being edited.
    """
    available, description = service.latex_available()

    ui.separator().classes("my-4")
    _section("LaTeX 引擎与字体")
    ui.label(description).classes(META if available else MUTED)
    if not available:
        ui.label("没有引擎时 arXiv 仍走 PDF 翻译；可在下方填系统 xelatex 路径。").classes(MUTED)

    with ui.row().classes("w-full gap-4 flex-wrap items-end"):
        ui.input("TeX 引擎路径（留空用随包的）",
                 placeholder="/Library/TeX/texbin/xelatex").classes("w-96").props(
            INPUT_PROPS
        ).bind_value(s.latex, "engine_path")
        ui.input("中文字体族（留空自动挑选）", placeholder="Songti SC").classes(
            "w-64"
        ).props(INPUT_PROPS).bind_value(s.latex, "cjk_font")


def _latex_options(service, s) -> None:
    """
    中文：LaTeX 的行为部分，画在「开发者选项」里。

    Render the LaTeX behaviour settings, inside the developer card.

    Args:
        service (TranslatorService): Service, used to report the engine found.
        s (Settings): Live settings object being edited.
    """
    available, _ = service.latex_available()

    ui.separator().classes("my-4")
    _section("LaTeX 配置", "从 .tex 源码翻译，版面由 TeX 重排"
             + ("" if available else "（本机没有引擎，这些设置暂不起作用）"))

    ui.checkbox("arXiv 优先使用 LaTeX 源码翻译").bind_value(
        s.latex, "prefer_for_arxiv"
    ).tooltip("有源码就从源码翻，arXiv 上没有源码的论文自动回到 PDF 翻译")

    with ui.row().classes("w-full gap-4 flex-wrap items-end"):
        ui.number("同时翻译块数", min=1, max=32).classes("w-40").props(
            INPUT_PROPS
        ).bind_value(s.latex, "workers")

    ui.checkbox("保留译文源码与工作目录").bind_value(s.latex, "keep_sources").tooltip(
        "编译失败时据此排查；成功后这些文件只是占地方"
    )

    ui.label("输出版式与翻译区域沿用上面两栏的设置。").classes(MUTED)


def _output_card(s) -> None:
    """
    中文：PDF 输出设置。

    Render PDF output options.

    Args:
        s (Settings): Live settings object being edited.
    """
    with ui.card().classes(CARD):
        _section("输出设置")

        with ui.row().classes("w-full gap-6 flex-wrap"):
            ui.checkbox("输出双语对照 PDF").bind_value(s.pdf, "output_dual")
            ui.checkbox("输出纯译文 PDF").bind_value(s.pdf, "output_mono")


        with ui.row().classes("w-full gap-4 flex-wrap mt-2"):
            ui.select(
                {False: "并排拼版（一页左右对照）", True: "交替页（原文页、译文页轮流）"},
                label="双语版式",
            ).classes("w-72").props(INPUT_PROPS).bind_value(s.pdf, "use_alternating_pages_dual")
            ui.select(
                {"no_watermark": "无水印", "watermarked": "有水印", "both": "两者都输出"},
                label="水印",
            ).classes("w-40").props(INPUT_PROPS).bind_value(s.pdf, "watermark_mode")

        with ui.row().classes("w-full gap-6 flex-wrap mt-2"):
            ui.checkbox("译文在左").bind_value(s.pdf, "dual_translate_first")
            ui.checkbox("修复被丢弃的插图与分割线").bind_value(s.pdf, "restore_images").tooltip(
                "重建页面时整块丢失的插图、表格与页眉页脚分割线，按原位从源 PDF 搬回"
            )
            ui.checkbox("保留原文超链接").bind_value(s.pdf, "restore_links").tooltip(
                "BabelDOC 会连同所有注释一起删掉链接；此项把引用、交叉引用与网址重新贴回译文位置"
            )
            ui.checkbox("跨页断句重新划分").bind_value(s.pdf, "rebalance_page_breaks").tooltip(
                "一句话被分页切成两半时，把较短的那半挪到另一页再分别翻译，"
                "避免两边都拿到半截话；代价是某个分句会换到相邻一页上"
            )

        ui.separator().classes("my-3")
        _section("字体")

        with ui.row().classes("w-full gap-4 flex-wrap"):
            ui.select({None: "跟随原文（衬线→宋体，无衬线→黑体）",
                       "serif": "全部用宋体", "sans-serif": "全部用黑体",
                       "script": "全部用手写体"},
                      label="正文字体").classes("w-72").props(INPUT_PROPS).bind_value(
                s.pdf, "primary_font_family"
            )

        with ui.row().classes("w-full gap-4 flex-wrap items-center mt-2"):
            ui.input("手写体字体文件（留空用附带的汇文仿宋）",
                     placeholder="/path/to/YourFont.ttf").classes("w-96").props(
                INPUT_PROPS
            ).bind_value(s.pdf, "script_font")
            ui.switch("该字体是衬线体").bind_value(s.pdf, "script_font_serif")


def _glossary_section(s) -> None:
    """
    中文：术语库行为设置，画在「开发者选项」里。

    Render glossary behaviour settings, inside the developer card.

    Args:
        s (Settings): Live settings object being edited.
    """
    if True:
        ui.separator().classes("my-4")
        _section("术语库设置")

        ui.checkbox("翻译时注入匹配到的术语").bind_value(s.glossary, "enabled")

        with ui.row().classes("w-full gap-4 flex-wrap mt-2"):
            ui.number("每段最多注入术语数", min=4, max=64).classes("w-52").props(
                INPUT_PROPS
            ).bind_value(s.glossary, "max_terms_per_prompt")
            ui.number(
                "跳过 API 提取的覆盖率阈值", min=0.0, max=1.0, step=0.05, format="%.2f"
            ).classes("w-56").props(INPUT_PROPS).bind_value(s.glossary, "coverage_threshold")


        # Withdrawn rather than removed: the switch and its wiring stay, so
        # turning it back on is a one-line change once extraction filters its
        # own output well enough to be worth the API call.
        # 撤下而非删除：开关与接线都保留，等提取端能把自己的输出筛干净、值得为它花
        # API 调用时，改一行就能放回来。
        ui.checkbox("允许在覆盖率不足时调用 API 提取新术语（暂未开放）").props(
            "disable"
        )
        ui.checkbox("正文也用白名单硬保护（译前替换为标记，译后还原）").bind_value(
            s.glossary, "hard_protect_whitelist"
        )


def _tencent_credentials(s) -> None:
    """
    中文：腾讯云密钥的填写与验证，写入 ~/.keys/tencent.env。

    Render the Tencent Cloud credential fields.

    The pair is written to a key file rather than the keychain because the
    application reads it from a worker subprocess, where a keychain prompt
    would have nobody to answer it. The file is created owner-only.

    Verification performs one real translation before saving: a key pair can be
    well-formed and still be refused for reasons no offline check can see — the
    service not activated, the account isolated for non-payment, the pair
    lacking permission for TMT — and each of those looks identical to a typo
    until something asks the service.

    Args:
        s (Settings): Live settings object being edited.
    """
    import os
    import stat

    from local_translator.secrets import KEY_DIR, read_key_values

    path = KEY_DIR / "tencent.env"

    @ui.refreshable
    def rows() -> None:
        """中文：渲染密钥状态与输入。 / Render credential status and entry."""
        stored = read_key_values("tencent")
        configured = bool(stored.get("SECRET_ID") and stored.get("SECRET_KEY"))

        with ui.row().classes("w-full items-center gap-3"):
            ui.label("已配置" if configured else "未配置密钥").classes(
                "text-xs font-medium"
            ).style(f"color: {GREEN if configured else INK_SOFT}")
            if configured:
                # The id is not a secret and seeing its first characters is how
                # a reader tells two accounts apart; the key is never shown.
                # SecretId 不是机密，露出开头几位是读者区分两个账号的方式；SecretKey
                # 任何情况下都不显示。
                ui.label(f"SecretId {stored['SECRET_ID'][:8]}… · "
                         f"地域 {stored.get('REGION') or 'ap-guangzhou'}").classes(META)

        with ui.row().classes("w-full gap-4 flex-wrap"):
            id_input = ui.input("SecretId").classes("flex-1 min-w-[240px]").props(INPUT_PROPS)
            key_input = ui.input("SecretKey", password=True).classes(
                "flex-1 min-w-[240px]"
            ).props(INPUT_PROPS)

        with ui.row().classes("w-full gap-4 flex-wrap items-center"):
            region = ui.select(
                ["ap-guangzhou", "ap-shanghai", "ap-beijing", "ap-chengdu",
                 "ap-hongkong", "ap-singapore"],
                label="地域",
                value=stored.get("REGION") or "ap-guangzhou",
            ).classes("w-44").props(INPUT_PROPS)
            repos = ui.input(
                "术语库 ID（可选，逗号分隔）", value=stored.get("TERM_REPO_IDS") or ""
            ).classes("flex-1 min-w-[200px]").props(INPUT_PROPS)

        async def verify_and_save() -> None:
            """中文：先真的翻一句，通过再写盘。 / Translate once, then save."""
            secret_id = (id_input.value or "").strip()
            secret_key = (key_input.value or "").strip()
            if not secret_id or not secret_key:
                ui.notify("SecretId 和 SecretKey 都要填", type="warning")
                return

            from local_translator.engine.tencent import translate_text

            # An object, not ui.notify(), so it can be taken down again — see
            # the note in web/app.py's _ensure_model.
            # 用对象形式而不是 ui.notify()，这样它才能被撤下——原因见 web/app.py 的
            # _ensure_model 处那段说明。
            progress = ui.notification(
                "正在用这对密钥试译一句话…", type="ongoing", spinner=True
            )
            try:
                try:
                    sample = await asyncio.to_thread(
                        translate_text,
                        "The model was trained on a large corpus.",
                        secret_id=secret_id, secret_key=secret_key,
                        region=region.value, source="en", target="zh",
                    )
                finally:
                    progress.dismiss()
            except Exception as exc:  # noqa: BLE001
                ui.notify(f"验证失败：{exc}", type="negative", timeout=12000)
                return

            lines = [
                "# 腾讯云机器翻译密钥 / Tencent Cloud credentials",
                f"SECRET_ID={secret_id}",
                f"SECRET_KEY={secret_key}",
                f"REGION={region.value}",
            ]
            if (repos.value or "").strip():
                lines.append(f"TERM_REPO_IDS={repos.value.strip()}")

            # Created owner-only before the secret is written, rather than
            # chmod-ed after, which would leave it briefly world-readable.
            # 在写入机密之前就以仅本人可读的权限创建，而不是写完再 chmod——那样会有一段
            # 时间文件是所有人可读的。
            KEY_DIR.mkdir(parents=True, exist_ok=True)
            handle = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
                             stat.S_IRUSR | stat.S_IWUSR)
            try:
                os.write(handle, ("\n".join(lines) + "\n").encode("utf-8"))
            finally:
                os.close(handle)
            os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)

            key_input.value = ""
            id_input.value = ""
            ui.notify(f"验证通过，已保存。试译结果：{sample}", type="positive", timeout=8000)
            rows.refresh()

        def forget() -> None:
            """中文：删除密钥文件。 / Remove the credentials file."""
            path.unlink(missing_ok=True)
            ui.notify("已删除密钥文件", type="warning")
            rows.refresh()

        with ui.row().classes("w-full items-center gap-3"):
            # The same weight as 「保存设置」, because it is the same kind of act:
            # it is the button that commits the credentials. Rendered flat it
            # read as a secondary link next to the fields it belongs to.
            # 与「保存设置」同等份量，因为它们是同一类动作：这是提交密钥的那个按钮。
            # 用扁平样式渲染时，它在所属字段旁边看起来像个次要链接。
            ui.button("测试并保存", on_click=verify_and_save).props(
                "no-caps unelevated"
            ).classes(BTN_PRIMARY)
            if configured:
                ui.button("删除", on_click=forget).props("no-caps flat dense").classes(
                    "text-[#eb5757]"
                )

    rows()
    ui.label(f"密钥存在 {display_path(path)}（仅本人可读）").classes(MUTED)


def _api_card(s, service) -> None:
    """
    中文：云端 API 模型设置，密钥写入系统钥匙串。

    Render cloud API model settings.

    The key field is write-only: it stores into the keychain and then shows only
    whether a key exists, so the page never renders a secret.

    Args:
        s (Settings): Live settings object being edited.
    """
    from local_translator.secrets import KEY_DIR, delete_api_key, key_source, set_api_key

    with ui.card().classes(CARD):
        _section("API 设置")

        # One card, one choice. The two services are alternatives, not layers:
        # a paragraph sent to the cloud goes to whichever is selected here and
        # never to both. Before this they were separate backends sitting beside
        # 「本地」 in the top-right toggle, which put a vendor choice in the same
        # row as a "where does this run" choice and left the Tencent
        # credentials with nowhere in the interface to be entered at all.
        # 一张卡片，一个选择。两家服务是互斥的备选而非叠加的层次：发往云端的段落只会去到
        # 这里选中的那一家，绝不会同时发给两家。在此之前它们是右上角切换器里与「本地」
        # 并列的两个后端，把「用哪家」和「在哪跑」这两个不同性质的选择放进了同一排，而
        # 腾讯云的密钥在界面上根本没有可填之处。
        provider = ui.toggle(
            dict(edition.CLOUD_PROVIDER_NAMES),
            value=s.translation.cloud_provider,
        ).props("dense unelevated no-caps toggle-color=primary").classes("mb-1")
        provider.bind_value(s.translation, "cloud_provider")

        ui.label("").bind_text_from(
            s.translation, "cloud_provider",
            lambda v: edition.CLOUD_PROVIDER_NOTES.get(v, ""),
        ).classes(MUTED)

        openai_fields = ui.column().classes("w-full gap-3")
        openai_fields.bind_visibility_from(
            s.translation, "cloud_provider", lambda v: v == "openai"
        )
        tencent_fields = ui.column().classes("w-full gap-3")
        tencent_fields.bind_visibility_from(
            s.translation, "cloud_provider", lambda v: v == "tencent"
        )

        with tencent_fields:
            _tencent_credentials(s)

    with openai_fields:
        with ui.row().classes("w-full gap-4 flex-wrap"):
            ui.input("显示名称").classes("w-40").props(INPUT_PROPS).bind_value(
                s.api_model, "display_name"
            )
            ui.input("模型名").classes("w-52").props(INPUT_PROPS).bind_value(
                s.api_model, "model_name"
            )
            ui.input("Base URL").classes("flex-1 min-w-[240px]").props(INPUT_PROPS).bind_value(
                s.api_model, "base_url"
            )

        @ui.refreshable
        def key_row() -> None:
            """中文：渲染密钥状态与输入。 / Render key status and entry."""
            account = s.api_model.keyring_account
            origin = key_source(account)
            configured = origin is not None
            where = {
                "env": "环境变量",
                "file": f"{KEY_DIR}/{account}.env",
                "keychain": "系统钥匙串",
            }.get(origin or "", "")

            with ui.row().classes("w-full items-center gap-3"):
                ui.label(f"账户：{account}").classes(META)
                ui.label(f"已配置（来自{where}）" if configured else "未配置密钥").classes(
                    "text-xs font-medium"
                ).style(f"color: {GREEN if configured else INK_SOFT}")

                key_input = ui.input("输入 API Key 以保存", password=True).classes(
                    "flex-1 min-w-[200px]"
                ).props(INPUT_PROPS)

                def store_key() -> None:
                    value = (key_input.value or "").strip()
                    if not value:
                        ui.notify("请输入 API Key", type="warning")
                        return
                    try:
                        set_api_key(account, value)
                    except RuntimeError as exc:
                        ui.notify(str(exc), type="negative")
                        return
                    key_input.value = ""
                    ui.notify("密钥已存入系统钥匙串", type="positive")
                    key_row.refresh()

                def clear_key() -> None:
                    delete_api_key(account)
                    ui.notify("已从钥匙串删除", type="warning")
                    key_row.refresh()

                ui.button("保存密钥", on_click=store_key).props("no-caps flat dense")
                if configured:
                    ui.button("删除", on_click=clear_key).props("no-caps flat dense").classes(
                        "text-[#eb5757]"
                    )

        key_row()
        ui.label(f"查找顺序：环境变量 → {display_path(KEY_DIR)}/<账户>.env → 系统钥匙串").classes(MUTED)

        with ui.row().classes("w-full gap-4 flex-wrap mt-2"):
            ui.number("输入单价（每百万 token）", min=0, step=0.1, format="%.2f").classes(
                "w-56"
            ).props(INPUT_PROPS).bind_value(s.api_model, "price_in_per_1m")
            ui.number("输出单价（每百万 token）", min=0, step=0.1, format="%.2f").classes(
                "w-56"
            ).props(INPUT_PROPS).bind_value(s.api_model, "price_out_per_1m")

        _usage_section(service)


def _usage_section(service) -> None:
    """
    中文：累计用量，画在「API 设置」这张卡片里而不是自成一栏。

    Render cumulative token usage, inside the caller's card.

    Usage and the service that produced it are one subject: the numbers here
    answer "what did that endpoint cost", and reading them a card away from the
    endpoint's own settings made the reader hold two places in mind at once.

    Args:
        service (TranslatorService): Service providing the aggregates.
    """
    totals = service.db.usage_totals()

    ui.separator().classes("my-4")
    _section("累计用量")

    if not totals:
        ui.label("还没有已完成的翻译").classes("text-sm text-[#9b9a97] py-2")
        return

    names = edition.BACKEND_NAMES
    with ui.row().classes("gap-3 flex-wrap"):
        for backend, data in totals.items():
            with ui.column().classes("gap-0 px-4 py-3 rounded-md min-w-[190px]").style(
                f"background: {SURFACE_ALT}"
            ):
                ui.label(names.get(backend, backend)).classes(
                    "text-[13px] font-semibold"
                ).style("color: #37352f")
                ui.label(f"{data['jobs']} 篇").classes(f"{META} lt-num")
                # Two services now share the 「云端」 backend and they meter
                # differently — one by token, one by source character — so
                # the row shows whichever figures it actually has. A
                # backend that has served both kinds of job shows both,
                # which is the truth about that history.
                # 两家服务如今共用「云端」这个后端，而它们的计量方式不同——一个按
                # token，一个按原文字符——因此这一行显示它实际拥有的那些数字。既服务过
                # 两类任务的后端会两者都显示，这正是那段历史的真实情况。
                if data["total_tokens"] or not data["source_chars"]:
                    ui.label(
                        f"输入 {human_tokens(data['prompt_tokens'])}"
                        f" · 输出 {human_tokens(data['completion_tokens'])}"
                    ).classes(f"{MUTED} lt-num")
                if data["source_chars"]:
                    ui.label(
                        f"原文 {human_tokens(data['source_chars'])} 字符"
                        f" · 译文 {human_tokens(data['target_chars'])} 字符"
                    ).classes(f"{MUTED} lt-num")
                if backend in ("api", "hybrid") and data["est_cost"]:
                    ui.label(f"预估费用 ¥{data['est_cost']:.2f}").classes(f"{MUTED} lt-num")
                elif backend == "local":
                    ui.label("无费用").classes(MUTED)


async def _restart_model(service) -> None:
    """
    中文：按当前设置重启推理服务。

    Restart the inference server so launch-time settings take effect.

    Context size, slot count and the draft model are command-line arguments to
    llama-server, so unlike sampling parameters they cannot be applied to a
    running process.

    Args:
    service (TranslatorService): Service owning the supervisor.
    """
    await asyncio.to_thread(service.supervisor.restart_for_model_change)
