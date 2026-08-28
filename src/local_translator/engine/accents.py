"""
中文：把「只有重音的行」并回它下面那一行——修复 \\hat 之类的符号在重排后乱跑。

Fold accent-only lines back into the line they belong to.

**The symptom.** In a translated paper, the wide hats of ``\\widehat{V}`` end up
scattered across the paragraph: one floats in front of the first word, another
sits in the left margin, a third lands in the middle of a sentence. The base
letters are typeset correctly; only the accents move.

**Why it happens.** LaTeX draws a wide hat as a separate glyph from a math
extension font, placed above the letter. In the paper this was reported on, the
hat occupies y 618.7–620.5 while its ``V`` occupies y 610.4–617.4 — the two do
not overlap at all. BabelDOC's paragraph finder groups characters into lines by
their vertical band, so every hat on a text line is collected into a *line of
its own*, sitting immediately above. Formula grouping then runs per line
(``StylesAndFormulas._group_classified_characters``), so the hat becomes its own
formula unit, and ``merge_overlapping_formulas`` refuses to join it to its base
because that check begins with ``formula1.line_id == formula2.line_id``. Two
independent formula units then flow through typesetting independently, and the
accent lands wherever the text flow happens to put it.

Observed on page 7 of arXiv 2608.11612::

    [0] F '(cid:98)'   line=0  box=(203.9, 618.7, 209.5, 620.5)   <- the hat
    [1] T 'Because the dimensions of '
    [2] F 'MV '        line=1  box=(190.0, 610.4, 210.9, 617.4)   <- its base

**The fix.** Before formula grouping runs, move each accent character out of its
own line and into the line below, inserting it directly after the character it
sits above. The accent is then sequence-adjacent to its base, so the existing
grouping puts the two in one formula, and typesetting relocates the pair
together with their relative offsets intact — which is what keeps the hat on the
letter.

PDFMathTranslate solves the same problem in its converter by giving an accent
zero horizontal advance (``adv -= mod``) so it stacks onto the preceding glyph.
That works there because its extractor never separates the two in the first
place. Here the separation has already happened by the time any of our code
runs, so the repair has to put the character back where it belongs instead.

**Why this is conservative.** A line is only dissolved when *every* one of its
inked characters finds a base directly beneath it. If even one does not, the
line is left exactly as it was — the cost of doing nothing is the misplaced
accent we started with, whereas the cost of guessing wrong is deleting a line of
the paper.
"""

from __future__ import annotations

import logging
import re
import unicodedata

logger = logging.getLogger(__name__)

# A glyph the source font never mapped to Unicode. Accents from the TeX math
# extension fonts nearly always come through this way, because those fonts carry
# no usable ToUnicode table.
# 源字体没有映射到 Unicode 的字形。TeX 数学扩展字体里的重音几乎都是这种形式，
# 因为这些字体没有可用的 ToUnicode 表。
_UNMAPPED = re.compile(r"^\(cid:\d+\)$")

# Unicode categories that hold combining marks and spacing modifiers. `Sm`
# (math symbols) is deliberately absent: it would match ⊤, ∈ and ≈, which are
# ordinary formula content and must never be relocated.
# 组合符号与修饰符所在的 Unicode 类别。刻意不含 `Sm`（数学符号）：那会命中 ⊤、∈、≈
# 这些普通公式内容，绝不能搬动。
_ACCENT_CATEGORIES = frozenset({"Mn", "Sk", "Lm"})

# Accents that *do* carry a Unicode mapping, written as standalone characters.
# 带 Unicode 映射、以独立字符出现的重音。
_ACCENT_CHARS = frozenset("^~¯˘˙¨˚˝ˇ´`ˆ˜")

# An accent glyph is a thin stroke. Anything taller than this is a letter, a
# digit or an operator, and the line holding it is real content.
# 重音字形只是一道细笔画。高于此值的是字母、数字或运算符，其所在行是正文内容。
_MAX_ACCENT_HEIGHT = 5.0

# Vertical clearance between the bottom of an accent and the top of its base.
# Measured at 1.3pt on the reported paper; 5pt leaves room for a taller base
# without reaching the line above.
# 重音底部与基字符顶部之间的垂直间隙。实测该论文为 1.3pt；取 5pt 既容得下更高的
# 基字符，又够不到上一行。
_MAX_CLEARANCE = 5.0

# How far outside a base character's horizontal extent the accent's centre may
# fall and still be taken to belong to it. A wide hat is drawn slightly wider
# than the letter it covers.
# 重音中心可以超出基字符水平范围多远仍算属于它。宽帽的绘制宽度略大于所盖字母。
_X_SLACK = 2.0


def _blank(char) -> bool:
    """
    中文：判断字符是否不着墨（空格或程序补的占位符）。

    Report whether a character puts no ink on the page.

    Args:
        char (PdfCharacter): Character from the intermediate layout.

    Returns:
        bool: ``True`` for spaces and for the dummy characters BabelDOC inserts
            with no Unicode value at all.
    """
    text = char.char_unicode
    return text is None or text == "" or text.isspace()


def _is_accent(char) -> bool:
    """
    中文：判断字符是否是重音/上标记号一类的细笔画符号。

    Report whether a character is an accent-shaped mark.

    Three forms are accepted, covering how the same glyph reaches us from
    different fonts: an unmapped ``(cid:NN)`` glyph, a Unicode combining or
    modifier mark, and a standalone accent character such as ``^``.

    Args:
        char (PdfCharacter): Character from the intermediate layout.

    Returns:
        bool: ``True`` when the character may be relocated onto a base letter.
    """
    text = char.char_unicode
    if not text:
        return False
    if _UNMAPPED.match(text):
        return True
    if len(text) != 1:
        return False
    if text in _ACCENT_CHARS:
        return True
    return unicodedata.category(text) in _ACCENT_CATEGORIES


def _visual(char):
    """
    中文：取字符的视觉包围盒，缺失时退回排版盒。

    Return a character's visual bounding box, falling back to its layout box.

    The visual box is what the glyph actually inks; the layout box includes the
    side bearings, which for an accent are wide enough to change the answer.

    Args:
        char (PdfCharacter): Character from the intermediate layout.

    Returns:
        Box: The box to measure this character by.
    """
    bbox = getattr(char, "visual_bbox", None)
    if bbox is not None and getattr(bbox, "box", None) is not None:
        return bbox.box
    return char.box


def _find_base(accent, targets):
    """
    中文：在下一行里找出重音应该落在哪个字符上。

    Locate the character an accent belongs to, in the line below it.

    The accent is matched by its horizontal centre rather than by overlap, so a
    hat drawn slightly wider than its letter still resolves to that letter and
    not to the one beside it. The base must also sit *below* the accent, which
    is what stops a subscript on the line above from being claimed.

    Args:
        accent (PdfCharacter): The accent character to place.
        targets (list[PdfCharacter]): Characters of the candidate line, in
            reading order.

    Returns:
        int | None: Index into ``targets`` of the base character, or ``None``
            when no character on that line can carry this accent.
    """
    box = _visual(accent)
    centre = (box.x + box.x2) / 2

    best = None
    best_distance = None
    for index, candidate in enumerate(targets):
        if _blank(candidate):
            continue
        other = _visual(candidate)
        if other.x - _X_SLACK > centre or other.x2 + _X_SLACK < centre:
            continue
        # The base has to be underneath: its top must not rise above the
        # accent's top, and the gap between them must be a diacritic's gap.
        # 基字符必须在下方：其顶部不得高过重音顶部，且两者间隙要是变音符号的间隙。
        if other.y2 > box.y2:
            continue
        clearance = box.y - other.y2
        if clearance < -_MAX_ACCENT_HEIGHT or clearance > _MAX_CLEARANCE:
            continue
        distance = abs(((other.x + other.x2) / 2) - centre)
        if best_distance is None or distance < best_distance:
            best, best_distance = index, distance
    return best


def _accent_line(line) -> list:
    """
    中文：判断一行是否只由重音组成，是则返回这些重音。

    Return the accents of a line that holds nothing else.

    Args:
        line (PdfLine): A line of the intermediate layout.

    Returns:
        list[PdfCharacter]: The inked characters, when every one of them is an
            accent and the line is thin enough to be a row of diacritics.
            Empty when the line is ordinary content.
    """
    inked = [char for char in line.pdf_character if not _blank(char)]
    if not inked:
        return []
    if not all(_is_accent(char) for char in inked):
        return []
    boxes = [_visual(char) for char in inked]
    height = max(box.y2 for box in boxes) - min(box.y for box in boxes)
    if height > _MAX_ACCENT_HEIGHT:
        return []
    return inked


def _refresh(line) -> None:
    """
    中文：按当前字符重新计算行的包围盒。

    Recompute a line's bounding box from the characters it now holds.

    Args:
        line (PdfLine): Line whose character list has just changed.
    """
    from babeldoc.format.pdf.document_il import il_version_1

    boxes = [_visual(char) for char in line.pdf_character]
    if not boxes:
        return
    line.box = il_version_1.Box(
        min(box.x for box in boxes),
        min(box.y for box in boxes),
        max(box.x2 for box in boxes),
        max(box.y2 for box in boxes),
    )


def merge_accent_lines(page) -> int:
    """
    中文：把页面上所有「只有重音的行」并回它们的基字符所在行。

    Fold every accent-only line on a page into the line beneath it.

    Runs before formula grouping, while each paragraph is still a list of lines.
    For each line that holds nothing but accents, every accent is moved into the
    following line directly after the character it sits above; the emptied line
    is then dropped. A line whose accents cannot all be placed is left untouched.

    Args:
        page (Page): One page of BabelDOC's intermediate layout.

    Returns:
        int: Number of accent characters relocated. Zero when the page has none,
            which is the common case.
    """
    moved = 0
    for paragraph in getattr(page, "pdf_paragraph", None) or []:
        compositions = paragraph.pdf_paragraph_composition
        if not compositions or len(compositions) < 2:
            continue

        keep = []
        for index, composition in enumerate(compositions):
            line = composition.pdf_line
            if line is None or not line.pdf_character:
                keep.append(composition)
                continue

            accents = _accent_line(line)
            if not accents:
                keep.append(composition)
                continue

            # The accent sits above its base, so the base is on the *next* line.
            # 重音在基字符上方，因此基字符位于下一行。
            following = compositions[index + 1] if index + 1 < len(compositions) else None
            target = following.pdf_line if following is not None else None
            if target is None or not target.pdf_character:
                keep.append(composition)
                continue

            placements = []
            for accent in accents:
                position = _find_base(accent, target.pdf_character)
                if position is None:
                    break
                placements.append((position, accent))
            else:
                # Insert from the right so earlier indices stay valid.
                # 从右往左插入，使先前求得的下标保持有效。
                for position, accent in sorted(placements, reverse=True,
                                               key=lambda item: item[0]):
                    target.pdf_character.insert(position + 1, accent)
                _refresh(target)
                moved += len(placements)
                continue

            keep.append(composition)

        if len(keep) != len(compositions):
            paragraph.pdf_paragraph_composition = keep

    return moved


def install() -> None:
    """
    中文：在 BabelDOC 的公式识别之前插入重音归位这一步。

    Run accent folding just before BabelDOC parses formulas.

    Wraps ``StylesAndFormulas.process_page``, which is the first stage that sees
    a paragraph as lines of characters and the last one before those lines are
    cut into formula units. Idempotent, so a second call is harmless.

    Raises:
        AttributeError: If BabelDOC no longer exposes the method, so an upgrade
            that moves it fails loudly at startup instead of silently doing
            nothing.
    """
    from babeldoc.format.pdf.document_il.midend.styles_and_formulas import (
        StylesAndFormulas,
    )

    if getattr(StylesAndFormulas.process_page, "_accents_patched", False):
        return

    original = StylesAndFormulas.process_page

    def process_page(self, page):
        """中文：先把重音并回基字符，再走原逻辑。 / Fold accents, then run the original pass."""
        try:
            moved = merge_accent_lines(page)
        except Exception:  # noqa: BLE001
            # Failing here must not cost the page: the worst outcome of skipping
            # the fold is the misplaced accent we had before.
            # 这里失败不应赔上整页：跳过归位最坏也只是回到原来的错位重音。
            logger.warning("Accent folding failed on a page", exc_info=True)
        else:
            if moved:
                logger.debug("Folded %d accent(s) back onto their base characters",
                             moved)
        return original(self, page)

    process_page._accents_patched = True
    StylesAndFormulas.process_page = process_page
