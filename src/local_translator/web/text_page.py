"""
中文：段落翻译页——粘贴文本直接翻译，模型跟随顶部的后端选择。

Free-text translation page.

The model is not chosen here. It follows the backend selected in the header, so
the same control governs a pasted paragraph and a PDF, and there is never a
question of which setting won. The glossary applies identically, so a term
reads the same whichever route it arrived by.

Translation runs in the service process rather than a worker subprocess, unlike
document translation. A pasted paragraph is a single HTTP request; spawning a
process for it would cost more than the work.
"""

from __future__ import annotations

import asyncio
import logging

from nicegui import ui

from local_translator import edition
from local_translator.config import LANGUAGES
from local_translator.service import get_service
from local_translator.web.theme import (
    CARD,
    HEADING,
    INK_SOFT,
    INPUT_PROPS,
    MUTED,
)

logger = logging.getLogger(__name__)

BACKEND_LABELS = edition.BACKEND_NAMES


@ui.page("/text")
def text_page() -> None:
    """
    中文：渲染段落翻译页。

    Render the free-text translation page.
    """
    from local_translator.web.app import _header, _page_body

    _header("text")
    service = get_service()
    settings = service.settings.translation

    with _page_body():
        with ui.card().classes(CARD):
            with ui.row().classes("w-full items-center gap-3 flex-wrap"):
                ui.label("段落翻译").classes(HEADING)
                ui.space()
                source = ui.select(
                    LANGUAGES, value=settings.lang_in, label="源语言", with_input=True
                ).props(INPUT_PROPS).classes("w-40")
                target = ui.select(
                    LANGUAGES, value=settings.lang_out, label="目标语言", with_input=True
                ).props(INPUT_PROPS).classes("w-40")

            ui.label(
                f"使用顶部选择的后端（当前：{BACKEND_LABELS.get(settings.backend, settings.backend)}）。"
                "空行分段，每段单独翻译以保留结构。术语库与 PDF 翻译一致。"
            ).classes(MUTED)

        with ui.row().classes("w-full gap-4 items-stretch no-wrap"):
            with ui.card().classes("lt-card flex-1"):
                ui.label("原文").classes(MUTED)
                box = ui.textarea(placeholder="粘贴或输入要翻译的文本…").props(
                    "outlined autogrow"
                ).classes("w-full").style("min-height: 320px")

            with ui.card().classes("lt-card flex-1"):
                ui.label("译文").classes(MUTED)
                out = ui.markdown("").classes("w-full text-[14px] leading-7").style(
                    "min-height: 320px; color: #37352f"
                )

        note = ui.label("").classes(MUTED)

        async def run() -> None:
            """中文：执行翻译并显示用量。 / Translate and report usage."""
            text = (box.value or "").strip()
            if not text:
                ui.notify("先输入要翻译的内容", type="warning")
                return

            go.disable()
            spinner.visible = True
            note.text = ""
            try:
                result = await asyncio.to_thread(
                    service.translate_text, text, source.value, target.value
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception("Text translation failed")
                ui.notify(f"翻译失败：{exc}", type="negative")
                return
            finally:
                spinner.visible = False
                go.enable()

            out.content = result["text"]
            tokens = result["tokens"]
            note.text = (
                f"{BACKEND_LABELS.get(result['backend'], result['backend'])} · "
                f"{result['paragraphs']} 段 · {result['seconds']} 秒"
                + (f" · {tokens:,} tokens" if tokens else "")
            )

        with ui.row().classes("items-center gap-3"):
            go = ui.button("翻译", on_click=run).classes("lt-btn-primary")
            spinner = ui.spinner(size="sm")
            spinner.visible = False

            def swap() -> None:
                """中文：交换源语言与目标语言。 / Swap the language pair."""
                source.value, target.value = target.value, source.value

            ui.button("交换语言", on_click=swap).classes("lt-btn")

            def clear() -> None:
                """中文：清空两侧内容。 / Clear both panes."""
                box.value = ""
                out.content = ""
                note.text = ""

            ui.button("清空", on_click=clear).classes("lt-btn")
            ui.label("⌘/Ctrl + Enter 也可翻译").classes(MUTED).style(f"color: {INK_SOFT}")

        box.on("keydown.ctrl.enter", run)
        box.on("keydown.meta.enter", run)
