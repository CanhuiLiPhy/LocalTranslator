"""
中文：图片翻译页——上传或粘贴图片，本地识别文字后翻译。

Image translation page.

Recognition runs locally on PP-OCRv6 and only the recognised text is sent for
translation, so the picture itself never leaves the machine. That was not a
design preference to begin with — the configured API model is text-only and
rejects images outright — but it is the better arrangement regardless when the
image is a page of something unpublished.

The result is shown two ways. The joined translation is what a reader wants.
The line-by-line pairing beside it is what makes a wrong line diagnosable: once
recognition and translation are merged, a misread character and a mistranslated
word look exactly the same, and only the recognised line tells them apart.
"""

from __future__ import annotations

import asyncio
import base64
import logging

from nicegui import ui

from local_translator.config import LANGUAGES
from local_translator.ocr.install import TIERS
from local_translator.service import get_service
from local_translator.web.format import display_path
from local_translator.web.theme import (
    AMBER,
    CARD,
    GREEN,
    HEADING,
    INK_SOFT,
    INPUT_PROPS,
    META,
    MUTED,
)

logger = logging.getLogger(__name__)

# Below this the recognised line is worth flagging rather than trusting.
# 低于该置信度的行值得标出而不是照单全收。
_UNSURE = 0.85


@ui.page("/image")
def image_page() -> None:
    """
    中文：渲染图片翻译页。

    Render the image translation page.
    """
    from local_translator.web.app import _header, _page_body

    _header("image")
    service = get_service()
    settings = service.settings.translation

    from local_translator.ocr import get_ocr

    engine = get_ocr()

    with _page_body():
        if not engine.available():
            _install_card()
            return

        with ui.card().classes(CARD):
            with ui.row().classes("w-full items-center gap-3 flex-wrap"):
                ui.label("图片翻译").classes(HEADING)
                ui.space()
                source = ui.select(
                    LANGUAGES, value=settings.lang_in, label="源语言", with_input=True
                ).props(INPUT_PROPS).classes("w-40")
                target = ui.select(
                    LANGUAGES, value=settings.lang_out, label="目标语言", with_input=True
                ).props(INPUT_PROPS).classes("w-40")

            ui.label(
                "文字识别在本地完成（PP-OCRv6），只有识别出的文字会送去翻译，"
                "图片本身不外传。翻译使用顶部选择的后端。"
            ).classes(MUTED)

        state: dict = {"data": None}

        with ui.card().classes(CARD):
            preview = ui.image().classes("w-full max-h-72 object-contain rounded")
            preview.visible = False

            async def handle(event) -> None:
                """中文：接收上传的图片并立即识别翻译。 / Take an upload and run it."""
                data = await event.file.read()
                state["data"] = data
                preview.set_source(
                    "data:image/png;base64," + base64.b64encode(data).decode()
                )
                preview.visible = True
                await run()

            ui.upload(
                label="拖拽或点击上传图片（PNG / JPG / WEBP）",
                multiple=False,
                auto_upload=True,
                on_upload=handle,
            ).props('accept="image/*" flat bordered').classes("w-full")

        note = ui.label("").classes(MUTED)

        @ui.refreshable
        def output() -> None:
            """中文：渲染原图与原位译文的对照，以及可展开的文本结果。 / Render the image comparison and text."""
            result = state.get("result")
            if not result:
                return

            original = state.get("data")
            overlay = result.get("overlay")

            # The comparison is the main view: the original on the left, the same
            # image with the translation drawn in place on the right. It is what
            # a figure or screenshot needs — the text kept where the layout put
            # it — so it leads, and the reading text and per-line check follow.
            # 对照是主视图：左原图、右为把译文原位画回的同一张图。图表或截图需要的正是
            # 让文字留在版式原处，因此它居首，阅读文本与逐行核对在其后。
            with ui.row().classes("w-full gap-4 items-start no-wrap"):
                with ui.card().classes("lt-card flex-1 min-w-0"):
                    ui.label("原图").classes(MUTED)
                    if original:
                        ui.image(
                            "data:image/png;base64," + base64.b64encode(original).decode()
                        ).classes("w-full rounded border border-gray-100")
                with ui.card().classes("lt-card flex-1 min-w-0"):
                    ui.label("原位译文").classes(MUTED)
                    if overlay:
                        ui.image("data:image/png;base64," + overlay).classes(
                            "w-full rounded border border-gray-100"
                        )
                    else:
                        ui.label("该图无法生成原位译文，请看下方文本").classes(META).style(
                            f"color: {INK_SOFT}"
                        )

            with ui.expansion("译文全文（便于复制）", icon="translate", value=True).classes(
                "w-full"
            ):
                for block in result["text"].split("\n\n"):
                    if block.strip():
                        ui.label(block).classes("text-[14px] leading-7 mb-2").style(
                            "color: #37352f"
                        )

            unsure = [ln for ln in result["lines"] if ln["confidence"] < _UNSURE]
            with ui.expansion(
                f"逐行识别与译文（{len(result['lines'])} 行"
                + (f"，{len(unsure)} 行置信度偏低）" if unsure else "）"),
                icon="format_list_bulleted",
            ).classes("w-full"):
                for line in result["lines"]:
                    colour = GREEN if line["confidence"] >= _UNSURE else AMBER
                    with ui.row().classes("w-full items-baseline gap-3 py-0.5"):
                        ui.label(f"{line['confidence']:.2f}").classes(META).style(
                            f"color: {colour}; min-width: 2.6rem"
                        )
                        with ui.column().classes("gap-0 min-w-0"):
                            ui.label(line["text"]).classes("text-[13px]").style(
                                "color: #37352f"
                            )
                            if line.get("translation"):
                                ui.label(line["translation"]).classes("text-[13px]").style(
                                    f"color: {INK_SOFT}"
                                )

        async def run() -> None:
            """中文：执行识别与翻译。 / Recognise and translate."""
            if not state.get("data"):
                ui.notify("先上传一张图片", type="warning")
                return

            spinner.visible = True
            note.text = "识别中…"
            try:
                result = await asyncio.to_thread(
                    service.translate_image, state["data"], source.value, target.value
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception("Image translation failed")
                ui.notify(f"失败：{exc}", type="negative")
                note.text = ""
                return
            finally:
                spinner.visible = False

            state["result"] = result
            preview.visible = False
            output.refresh()
            note.text = (
                f"{result['width']}×{result['height']} · 识别 {result['ocr_seconds']} 秒 · "
                f"翻译 {result['seconds']} 秒 · {len(result['lines'])} 行"
                + (f" · {result['tokens']:,} tokens" if result["tokens"] else "")
            )

        with ui.row().classes("items-center gap-3"):
            ui.button("重新翻译", on_click=run).classes("lt-btn")
            spinner = ui.spinner(size="sm")
            spinner.visible = False

        output()


def _install_card() -> None:
    """
    中文：渲染 OCR 模型安装卡片。

    Render the OCR model install prompt.
    """
    with ui.card().classes(CARD):
        ui.label("文字识别模型尚未安装").classes(HEADING)
        ui.label(
            "图片翻译需要本地的文字识别模型。使用 PaddleOCR 的 PP-OCRv6（Apache 2.0），"
            "官方 ONNX 导出，跑在项目已有的 onnxruntime 上，不新增任何依赖。"
            "单个模型覆盖 50 种语言。"
        ).classes(MUTED)

        tier = ui.toggle(
            {name: f"{name}（{spec['size_mb']} MB）" for name, spec in TIERS.items()},
            value="small",
        ).props("dense unelevated no-caps toggle-color=primary").classes("mt-3")
        with ui.column().classes("gap-0 mt-1"):
            for name, spec in TIERS.items():
                ui.label(f"{name} · {spec['note']}").classes(META).style(
                    f"color: {INK_SOFT}"
                )

        bar = ui.linear_progress(value=0, show_value=False).classes("w-full mt-3")
        bar.visible = False
        label = ui.label("").classes(MUTED)

        async def install() -> None:
            from local_translator.ocr.install import install_ocr

            button.disable()
            bar.visible = True

            def report(stage: str, frac: float) -> None:
                bar.value = frac
                label.text = f"{stage} · {frac:.0%}"

            try:
                await asyncio.to_thread(install_ocr, tier.value, None, report)
            except Exception as exc:  # noqa: BLE001
                logger.exception("OCR install failed")
                ui.notify(f"安装失败：{exc}", type="negative")
                button.enable()
                return
            ui.notify("识别模型已安装", type="positive")
            ui.navigate.reload()

        button = ui.button("下载并安装", on_click=install).classes("lt-btn-primary mt-3")
        ui.label(f"模型存放在 {display_path(get_service().paths.ocr_models_dir)}").classes(META).style(
            f"color: {INK_SOFT}"
        )
