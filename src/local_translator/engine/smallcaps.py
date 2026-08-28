"""
中文：小型大写字母排的标题，译文会照着小写体的字号排，于是整行比原文小两成。

A heading set in small capitals is translated at the small-capital size.

**The symptom.** Two headings in the bench come out a fifth too small::

    ICLR   1 INTRODUCTION            11.96pt  →  1 引言        9.56pt
    IEEE   COMPARISON OF PARADIGMS    7.97pt  →  表I 范式比较  6.38pt

Neither is scaled down — the typesetter reports scale 1.0 for both. The type
itself is smaller.

**Why.** Small capitals are not a font, they are two sizes of the same one: the
capital that opens the word is set at full size and the letters after it at
about four fifths, as capitals. Read character by character, the IEEE table
title is ``C`` at 7.97 followed by ``OMPARISON`` at 6.38, and the ICLR heading
is ``I`` at 11.96 followed by ``NTRODUCTION`` at 9.56 — 0.800 and 0.799.

A paragraph carries one size into its translation, and it is the *mode* of its
characters. Small capitals outnumber full ones in every word longer than a
letter, so the mode is the small size, and the Chinese is set at four fifths of
the heading it replaces.

**The fix.** Chinese has no small capitals, so the size that means anything here
is the capital one. The smaller run is raised to it before the style is worked
out, and the mode then lands where it should.

**Why this is not a superscript.** The two look alike from a ratio alone, and
BabelDOC's own superscript test turns on the difference: its comment records
superscripts at 0.76 and small capitals at 0.799, and puts the threshold at
0.79 between them. This reads the same boundary from the other side and adds
what a ratio cannot say — every character in the smaller run must be a capital
letter, and there must be a run of them, which no superscript is.
"""

from __future__ import annotations

import collections
import logging

logger = logging.getLogger(__name__)

# The band a small-capital size occupies relative to its full capital. The floor
# is BabelDOC's own superscript boundary, below which a smaller run is a corner
# mark rather than a small capital; the ceiling leaves room for a face whose
# small capitals are drawn closer to full height.
# 小型大写相对全大写的字号区间。下界取 BabelDOC 自己的上标分界线，低于它的较小字串是角标
# 而不是小型大写；上界为小型大写画得更接近全高的字体留出余地。
_MIN_RATIO = 0.79
_MAX_RATIO = 0.92

# How many characters the smaller run needs before it counts. A superscript is
# one or two characters; a word set in small capitals is longer.
# 较小字串至少要有多少个字符才算数。上标只有一两个字符，用小型大写排的词更长。
_MIN_RUN = 3


def _characters(paragraph):
    """
    中文：按顺序取出段落里的字符。

    Yield a paragraph's characters in reading order.

    Args:
        paragraph (PdfParagraph): The paragraph to walk.

    Yields:
        PdfCharacter: Each character the paragraph holds.
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


def _is_capital(character) -> bool:
    """中文：这是不是一个大写字母。 / Whether this character is a capital letter."""
    text = getattr(character, "char_unicode", None)
    return bool(text) and len(text) == 1 and text.isalpha() and text.isupper()


def _raise_small_capitals(paragraph) -> int:
    """
    中文：把段落里用小型大写排的那一串抬到全大写的字号。

    Raise a paragraph's small capitals to the size of the capitals beside them.

    Args:
        paragraph (PdfParagraph): The paragraph to inspect.

    Returns:
        int: How many characters were raised.
    """
    by_font = collections.defaultdict(lambda: collections.defaultdict(list))
    for character in _characters(paragraph):
        style = getattr(character, "pdf_style", None)
        size = getattr(style, "font_size", None)
        if not size:
            continue
        by_font[getattr(style, "font_id", None)][round(size, 2)].append(character)

    raised = 0
    for sizes in by_font.values():
        if len(sizes) < 2:
            continue
        largest = max(sizes)
        for size, characters in sizes.items():
            if size >= largest:
                continue
            ratio = size / largest
            if not _MIN_RATIO <= ratio <= _MAX_RATIO:
                continue
            letters = [c for c in characters
                       if (getattr(c, "char_unicode", "") or "").strip()]
            if len(letters) < _MIN_RUN or not all(_is_capital(c) for c in letters):
                continue
            for character in characters:
                character.pdf_style.font_size = largest
            raised += len(characters)
    return raised


def install() -> None:
    """
    中文：在样式定型之前把小型大写抬到全大写字号。

    Raise small capitals before the paragraph's style is settled.

    Wraps ``StylesAndFormulas.process_page``, the stage that reduces a
    paragraph's character sizes to the one style its translation inherits.

    Idempotent, so a second call is harmless.

    Raises:
        AttributeError: If BabelDOC no longer exposes the stage, so an upgrade
            that moves it fails at startup rather than silently setting every
            small-capital heading a fifth too small again.
    """
    from babeldoc.format.pdf.document_il.midend.styles_and_formulas import (
        StylesAndFormulas,
    )

    if getattr(StylesAndFormulas.process_page, "_smallcaps_patched", False):
        return

    original = StylesAndFormulas.process_page

    def process_page(self, page, *args, **kwargs):
        """中文：先抬小型大写，再照常处理。 / Raise small capitals, then proceed."""
        try:
            raised = 0
            for paragraph in getattr(page, "pdf_paragraph", None) or []:
                raised += _raise_small_capitals(paragraph)
            if raised:
                logger.debug("Raised %d small capital(s) on page %s",
                             raised, getattr(page, "page_number", "?"))
        except Exception:  # noqa: BLE001
            # A page that cannot be examined keeps its sizes, which is how every
            # page behaved before this existed.
            # 看不了的页面保留原字号，也就是本模块存在之前所有页面的行为。
            logger.warning("Could not raise small capitals", exc_info=True)
        return original(self, page, *args, **kwargs)

    process_page._smallcaps_patched = True
    StylesAndFormulas.process_page = process_page
