"""
中文：段首那个放大的首字母会被当成上标，于是它后面那个词整个不翻译。

An enlarged opening letter is read as a superscript, and the word after it
is then never translated.

**The symptom.** IEEE papers open a section with a two-line initial: a large
``A`` followed by ``CCORDING`` in caps, then ordinary text. Translated, the page
reads::

    A CCORDING        I. 引言
              针对国际电信联盟（ITU）发布的最新
    6G愿景 [1], 未来无线通信系统将...

The English ``A CCORDING`` is still there — and the Chinese says the same thing,
because the sentence *was* translated. One phrase, printed twice, on top of the
section heading.

**Why.** Superscripts are found by comparing each character with the one before
it::

    and char.pdf_style.font_size < previous_char.pdf_style.font_size * 0.79
    ...
    and char.pdf_style.font_size < previous_char.pdf_style.font_size * 1.1
        and in_corner_mark_state

The ``C`` after the initial is 9.96 points against the ``A``'s 28.74 — a third
of it, far under the 0.79 that marks a superscript. So ``C`` is tagged as one,
and the second clause makes the tag sticky: every following character of the
same size stays tagged until a space ends the run. That is the whole word
``CCORDING``, which is then preserved as a formula and never sent to the model —
while the sentence it belongs to is translated with it still in place.

The comparison is only ever between neighbours, and that cannot tell "this
character is small" from "the one before it was big". Both look identical to a
ratio. The code's own comment says it means to allow for an enlarged first
letter; the threshold does not.

**The fix.** A superscript is small relative to *the text around it*, not merely
relative to whatever precedes it. So an opening letter far larger than its own
paragraph is restored to the paragraph's size before the classifier runs. The
``A`` is then the same size as the ``C``, no ratio fires, and nothing is
mistaken for a superscript.

Its face is restored with its size. A drop cap is set in a heavier weight —
``NimbusRomNo9L-Medi`` against the paragraph's ``-Regu`` — and a change of face
starts a new style run, so the paragraph reached the model as ``A`` followed by
``CCORDING to the latest…``. The model translated the sentence and handed the
lone ``A`` back unchanged, which is how a bare ``A`` survived in front of the
Chinese even once the size was right. One run instead of two, and the sentence
is translated once, whole, at body size.

This also settles what a drop cap should become in Chinese, where the device
does not exist: ordinary text, which is what the rest of the paragraph is.
"""

from __future__ import annotations

import logging
import statistics

logger = logging.getLogger(__name__)

# How much larger than its paragraph an opening letter must be to count as an
# enlarged initial. A drop cap runs two or three lines deep, so it is at least
# twice the body; ordinary emphasis inside a paragraph does not reach half again.
# 段首字母要比本段大多少才算放大首字母。下沉首字母有两三行高，至少是正文的两倍；段落
# 内部普通的强调达不到一点五倍。
_ENLARGED = 1.5

# A paragraph shorter than this has no reliable dominant size to compare against.
# 短于这个长度的段落没有可靠的主流字号可供比较。
_MIN_CHARS = 20


def _characters(paragraph):
    """
    中文：按顺序取出段落里的字符。

    Yield a paragraph's characters in reading order.

    Args:
        paragraph (PdfParagraph): The paragraph to walk.

    Yields:
        PdfCharacter: Each character the paragraph holds, whether it sits
            directly in a composition or inside one of the line-like parts.
    """
    for composition in getattr(paragraph, "pdf_paragraph_composition", None) or []:
        for name in ("pdf_character", "pdf_line", "pdf_same_style_characters",
                     "pdf_same_style_unicode_characters", "pdf_formula"):
            part = getattr(composition, name, None)
            if part is None:
                continue
            if name == "pdf_character":
                yield part
                continue
            yield from getattr(part, "pdf_character", None) or []


def _normalise(page) -> int:
    """
    中文：把每段开头那个过大的字母改回本段的字号。

    Reset each paragraph's oversized opening letter to the paragraph's own size.

    Args:
        page (Page): The page, before formulas and styles are classified.

    Returns:
        int: How many initials were reset.
    """
    reset = 0
    for paragraph in getattr(page, "pdf_paragraph", None) or []:
        characters = [c for c in _characters(paragraph)
                      if getattr(c, "pdf_style", None) is not None
                      and getattr(c.pdf_style, "font_size", None)]
        if len(characters) < _MIN_CHARS:
            continue
        try:
            dominant = statistics.mode([c.pdf_style.font_size for c in characters])
        except statistics.StatisticsError:
            continue
        first, second = characters[0], characters[1]
        if not dominant or first.pdf_style.font_size < dominant * _ENLARGED:
            continue
        # Only the style is changed. The character's box is its ink on the source
        # page and is still what the page is measured by; the paragraph is
        # retypeset from the translation regardless, and the style is what
        # decides both the superscript test and where a style run begins.
        # 只改样式。字符的框是它在原文页面上的墨迹，页面度量仍要用它；无论如何这一段都会
        # 按译文重新排版，而样式才是决定上标判定与样式段起点的依据。
        logger.debug("Reset an enlarged initial %r from %.2fpt %s to %.2fpt %s",
                     getattr(first, "char_unicode", "?"),
                     first.pdf_style.font_size, first.pdf_style.font_id,
                     dominant, second.pdf_style.font_id)
        first.pdf_style.font_size = dominant
        first.pdf_style.font_id = second.pdf_style.font_id
        reset += 1
    return reset


def install() -> None:
    """
    中文：在公式与样式分类之前把放大的首字母改回正常字号。

    Reset enlarged initials before formulas and styles are classified.

    Wraps ``StylesAndFormulas.process_page``, which is the stage that reads
    character sizes to find superscripts. Running immediately before it means
    the paragraphs are built and the sizes are final, and nothing has yet been
    tagged from them.

    Idempotent, so a second call is harmless.

    Raises:
        AttributeError: If BabelDOC no longer exposes the stage, so an upgrade
            that moves it fails at startup rather than silently restoring the
            duplicated opening phrase.
    """
    from babeldoc.format.pdf.document_il.midend.styles_and_formulas import (
        StylesAndFormulas,
    )

    if getattr(StylesAndFormulas.process_page, "_dropcap_patched", False):
        return

    original = StylesAndFormulas.process_page

    def process_page(self, page, *args, **kwargs):
        """中文：先归一化首字母，再照常分类。 / Normalise initials, then classify."""
        try:
            reset = _normalise(page)
            if reset:
                logger.debug("Reset %d enlarged initial(s) on page %s",
                             reset, getattr(page, "page_number", "?"))
        except Exception:  # noqa: BLE001
            # A page that cannot be normalised keeps its initials, which is how
            # every page behaved before this existed.
            # 归一化不了的页面保留原样，也就是本模块存在之前所有页面的行为。
            logger.warning("Could not normalise enlarged initials", exc_info=True)
        return original(self, page, *args, **kwargs)

    process_page._dropcap_patched = True
    StylesAndFormulas.process_page = process_page
