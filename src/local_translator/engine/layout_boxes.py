"""
中文：不让版面框从单词中间切过去——修复「致谢」这类段落被劈成碎片的问题。

Stop a layout box from cutting a word in half.

**The symptom.** The Acknowledgements block came out as four unrelated pieces:
a bold 「致谢」 on its own, a line starting with a stray ``s``, a fragment set in
6.5pt type, and the last clause on a line of its own. Two of the four begin
mid-word.

**Why it happens.** The layout model proposes boxes, and BabelDOC assigns each
character to the box it overlaps most, then starts a new paragraph whenever the
box changes. On page 17 of the paper this was reported on, the model emitted::

    id=7  plain text  conf=0.63  box=(163.0, 384.0, 536.0, 408.0)

That box covers only the right-hand part of a two-line block that runs from
x=72. Its left edge at x=163 falls **inside the word "Acknowledgements"** — the
final ``s`` is centred at 163.3 — so the word is split between two paragraphs,
and the same edge cuts ``supported`` on the line below into ``su`` and
``pported``. Each fragment is then translated on its own.

**The fix.** Before paragraphs are formed, widen any box whose vertical edge
passes through a word so that it covers the whole word. Here that pulls the left
edge back to x=72.4 and the block becomes one paragraph again.

**A second way the same edge does damage.** A box need not cut a word in two to
break it: clipping the last glyph by a tenth of a point is enough. Characters
are awarded to the layout they overlap most, and on page 2 of the Nature
Communications paper two ``plain text`` boxes overlap — a tight one around the
paragraph and a larger one spanning both paragraphs. Every glyph of
``superconducting`` is fully inside both and goes to the tight one; the final
``g`` runs from x=556.7 to 561.1 against a box ending at 561.0, so it scores
0.972 there and 1.000 in the larger box, and defects. It arrives at the head of
the *next* paragraph, and the page opens with a stray ``g e`` — the last letters
of two lines, marooned:

    g e 随后开始出现超导转变…

So a box is widened not only when it splits a word but whenever it fails to
contain one it overlaps at all. Here that is two tenths of a point, and the
letter stays with its line.

**Why a word, and not a line.** Widening to the whole line would be the obvious
generalisation and is wrong: in a two-column paper a box that legitimately
covers the right column would swallow the left one. A word is never split across
columns, so "this edge is inside a word" is a defect with no legitimate reading,
and repairing exactly that leaves every correct box alone.

**The second symptom.** The same page's closing sentence came out as two
overlapping runs on one baseline — 「一起，这些」 at x=70.9 and 「这些结果提供了…」 at
x=142.1. The layout model had put its box at x=164.1 on a line that starts at
x=70.5, so ``Together, these`` fell outside every box, was given a per-line
``fallback_line`` layout of its own, and became a separate paragraph. Widening
off the word it cut pulls the edge back to 142.1 — enough to keep ``results``
whole, not enough to reach the two words before it.

**The second fix.** After the word repair, a box also swallows the run of
characters immediately beside it on a line it already covers, but only while
those characters belong to **no other box**. That last condition is what makes
this safe where widening to the whole line is not: in a two-column paper the
characters to the left belong to the other column's box, so the walk stops at
the first one and the box does not move.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Layout classes whose boxes carry running prose. A box around a figure or a
# table may legitimately end beside text, and widening it would change which
# region a caption belongs to.
# 承载连续文字的版面类别。插图或表格的框合理地可以紧邻文字结束，扩宽它会改变
# 标题归属的区域。
_REPAIRABLE = frozenset({
    "plain text", "title", "figure_caption", "table_caption", "table_footnote",
    "formula_caption", "abandon", "text", "paragraph", "abstract",
})

# Gap between two characters, in points, below which they belong to the same
# word. An inter-word space in 10pt Times is about 3.3pt, so this stays well
# clear of one.
# 两个字符间距小于该值（点）即属于同一个单词。10pt Times 的词间空格约 3.3pt，
# 该阈值与之保持了足够距离。
_WORD_GAP = 1.6

# Overlap in square points above which a character counts as belonging to a box.
# BabelDOC's own assignment (``get_character_layout``) accepts *any* positive
# overlap and then decides by class priority, and ``plain text`` outranks the
# per-line fallback — so a glyph grazed by one tenth of a point still lands in
# the box, which is exactly how a word ends up split. The threshold here is a
# guard against floating-point noise, nothing more.
# 字符与框的重叠面积超过该值（平方点）即算属于该框。BabelDOC 自己的归属逻辑
# （get_character_layout）接受任何正重叠，再按类别优先级决定，而 plain text 优先于
# 逐行兜底框——所以只被擦到十分之一点的字形也会落进框里，单词正是这样被切开的。
# 这里的阈值只是防浮点噪声，别无他意。
_MIN_OVERLAP = 0.01

# Shortest run of characters treated as a word. Two characters can straddle a
# column gutter by coincidence; three in a row cannot.
# 视为单词的最短字符串长度。两个字符可能碰巧跨在分栏间隙两侧，三个连着不会。
_MIN_WORD = 3

# Vertical tolerance when grouping characters into lines, as a share of the
# median character height.
# 把字符聚成行时的垂直容差，取字符高度中位数的这一比例。
_LINE_SHARE = 0.5

# How far an unclaimed character may sit from a box's edge and still be taken as
# part of the line inside it, in points. A word space in 10pt Times is about
# 3.3pt and justification stretches it, so this clears an ordinary gap while
# staying far short of the margin, where a stray line number would sit.
# 未被任何框认领的字符距框边多远仍算作框内这一行的一部分（点）。10pt Times 的词间
# 空格约 3.3pt，两端对齐会把它拉宽，该阈值容得下正常间隙，又远够不到页边——页边
# 才是零散行号所在的地方。
_ORPHAN_GAP = 6.0

# The same gap as a share of the line's median character height, so a page set in
# large type is measured by its own scale rather than by a body-text constant.
# 同一间隙相对于该行字符高度中位数的比例，使大字号页面按自身尺度衡量，而不是套用
# 正文字号的常数。
_ORPHAN_GAP_SHARE = 0.9

# BabelDOC's own name for the box it invents around a line no model box claimed.
# Such a box is the *absence* of a claim, so it must not stop the walk below —
# otherwise every unclaimed character looks like it belongs somewhere else.
# BabelDOC 给「没有任何模型框认领的行」自造的框所用的名字。这种框代表的正是「无人
# 认领」，因此不能让它中断下面的外扩——否则每个无主字符看上去都像已归属别处。
_FALLBACK = "fallback_line"


def _box(char):
    """
    中文：取字符的视觉包围盒，缺失时退回排版盒。

    Return a character's visual bounding box.

    Args:
        char (PdfCharacter): Character from the intermediate layout.

    Returns:
        Box: The box BabelDOC measures this character by.
    """
    bbox = getattr(char, "visual_bbox", None)
    if bbox is not None and getattr(bbox, "box", None) is not None:
        return bbox.box
    return char.box


def _typographic(char):
    """
    中文：取字符的排版盒——含上下留白，比墨迹盒更能代表它在哪一行。

    Return a character's typographic box.

    The visual box is the ink, which for a comma is a short mark sitting on the
    baseline: its centre is several points below the letters beside it, so
    clustering by ink puts a comma on a line of its own. The typographic box
    spans the full ascent and descent, and is the same for every character set
    in the same face and size.

    Args:
        char (PdfCharacter): Character from the intermediate layout.

    Returns:
        Box: The character's typographic box, falling back to the visual one.
    """
    return char.box if char.box is not None else _box(char)


def _lines(characters) -> list:
    """
    中文：把一页的字符按基线聚成行，每行内按 x 排好序。

    Group a page's characters into lines.

    Args:
        characters (list): Every character on the page.

    Returns:
        list[tuple[list, float]]: One entry per line — its characters sorted
            left to right, and the line's vertical centre.
    """
    inked = [c for c in characters
             if c.char_unicode and not c.char_unicode.isspace()]
    if not inked:
        return []

    heights = sorted(_typographic(c).y2 - _typographic(c).y for c in inked)
    median = heights[len(heights) // 2] or 1.0
    tolerance = median * _LINE_SHARE

    def centre_of(char) -> float:
        """中文：字符排版盒的垂直中心。 / Vertical centre of a character's typographic box."""
        box = _typographic(char)
        return (box.y + box.y2) / 2

    lines: list[list] = []
    for char in sorted(inked, key=lambda c: (-centre_of(c), _box(c).x)):
        centre = centre_of(char)
        if lines and abs(centre - lines[-1][1]) <= tolerance:
            lines[-1][0].append(char)
        else:
            lines.append(([char], centre))

    for line, _ in lines:
        line.sort(key=lambda c: _box(c).x)
    return [(line, centre) for line, centre in lines]


def _words(characters) -> list:
    """
    中文：把一页的字符切成一个个单词（同一行、间距小于阈值的连续字符）。

    Group a page's characters into words.

    A word is a run of non-blank characters that share a line and sit closer
    together than :data:`_WORD_GAP`.

    Args:
        characters (list): Every character on the page.

    Returns:
        list[list[PdfCharacter]]: One list per word, each at least
            :data:`_MIN_WORD` characters long.
    """
    words = []
    for line, _ in _lines(characters):
        run = [line[0]]
        for char in line[1:]:
            if _box(char).x - _box(run[-1]).x2 < _WORD_GAP:
                run.append(char)
            else:
                if len(run) >= _MIN_WORD:
                    words.append(run)
                run = [char]
        if len(run) >= _MIN_WORD:
            words.append(run)
    return words


def _inside(char, layout) -> bool:
    """
    中文：判断字符是否算落在某个版面框内。

    Decide whether a character belongs to a layout box.

    Args:
        char (PdfCharacter): The character to place.
        layout (PageLayout): The box to test against.

    Returns:
        bool: ``True`` when the character overlaps the box at all, which is the
            condition BabelDOC itself uses.
    """
    box, other = _box(char), layout.box
    width = min(box.x2, other.x2) - max(box.x, other.x)
    height = min(box.y2, other.y2) - max(box.y, other.y)
    return width > 0 and height > 0 and width * height > _MIN_OVERLAP


def widen_boxes_off_words(page) -> int:
    """
    中文：把每个从单词中间切过的版面框扩到覆盖整个单词。

    Widen every layout box whose vertical edge passes through a word.

    Runs before paragraphs are formed, while ``page.pdf_character`` still holds
    the page's characters. A box is only ever widened, and only far enough to
    clear the word it was cutting.

    Args:
        page (Page): One page of BabelDOC's intermediate layout.

    Returns:
        int: Number of boxes widened.
    """
    layouts = [layout for layout in (page.page_layout or [])
               if layout.box is not None and layout.class_name in _REPAIRABLE]
    if not layouts:
        return 0

    words = _words(page.pdf_character or [])
    if not words:
        return 0

    limit = page.cropbox.box if page.cropbox and page.cropbox.box else None
    widened = 0
    for layout in layouts:
        left, right = layout.box.x, layout.box.x2
        for word in words:
            centre = sum((_box(c).y + _box(c).y2) / 2 for c in word) / len(word)
            if not (layout.box.y <= centre <= layout.box.y2):
                continue
            flags = [_inside(char, layout) for char in word]
            if not any(flags):
                continue
            word_left = min(_box(c).x for c in word)
            word_right = max(_box(c).x2 for c in word)
            if all(flags) and layout.box.x <= word_left and word_right <= layout.box.x2:
                continue
            left = min(left, word_left)
            right = max(right, word_right)

        if left == layout.box.x and right == layout.box.x2:
            continue
        if limit is not None:
            left, right = max(left, limit.x), min(right, limit.x2)
        logger.debug("Widened %s box from x=%.1f..%.1f to %.1f..%.1f",
                     layout.class_name, layout.box.x, layout.box.x2, left, right)
        layout.box.x, layout.box.x2 = left, right
        widened += 1

    return widened


def widen_boxes_over_orphans(page) -> int:
    """
    中文：把版面框旁边、不属于任何框的同行字符纳入该框。

    Extend every layout box over the unclaimed characters beside it.

    For each line a box already covers, the characters immediately to the left
    of its left edge — and to the right of its right edge — are walked outwards.
    Each one is taken into the box only if no *other* layout box claims it and
    it sits within a word space of what came before, so the walk stops at the
    first character that belongs somewhere else and at the first real gap.

    Meant to run after :func:`widen_boxes_off_words`, which pulls the edge back
    to a word boundary first; from there the remaining distance to the rest of
    the line is an ordinary word space.

    Args:
        page (Page): One page of BabelDOC's intermediate layout.

    Returns:
        int: Number of boxes extended.
    """
    everything = [layout for layout in (page.page_layout or [])
                  if layout.box is not None and layout.class_name != _FALLBACK]
    layouts = [layout for layout in everything if layout.class_name in _REPAIRABLE]
    if not layouts:
        return 0

    lines = _lines(page.pdf_character or [])
    if not lines:
        return 0

    def claimed_elsewhere(char, layout) -> bool:
        """中文：该字符是否被别的框认领。 / Whether another box claims this character."""
        return any(other is not layout and _inside(char, other) for other in everything)

    limit = page.cropbox.box if page.cropbox and page.cropbox.box else None
    widened = 0
    for layout in layouts:
        left, right = layout.box.x, layout.box.x2
        for chars, centre in lines:
            if not (layout.box.y <= centre <= layout.box.y2):
                continue
            covered = [i for i, char in enumerate(chars) if _inside(char, layout)]
            if not covered or len(covered) == len(chars):
                continue

            heights = sorted(_typographic(c).y2 - _typographic(c).y for c in chars)
            reach = max(_ORPHAN_GAP, heights[len(heights) // 2] * _ORPHAN_GAP_SHARE)

            edge = _box(chars[covered[0]]).x
            for index in range(covered[0] - 1, -1, -1):
                char = chars[index]
                if claimed_elsewhere(char, layout) or edge - _box(char).x2 > reach:
                    break
                edge = _box(char).x
                left = min(left, edge)

            edge = _box(chars[covered[-1]]).x2
            for index in range(covered[-1] + 1, len(chars)):
                char = chars[index]
                if claimed_elsewhere(char, layout) or _box(char).x - edge > reach:
                    break
                edge = _box(char).x2
                right = max(right, edge)

        if left == layout.box.x and right == layout.box.x2:
            continue
        if limit is not None:
            left, right = max(left, limit.x), min(right, limit.x2)
        logger.debug("Extended %s box over unclaimed text, x=%.1f..%.1f to %.1f..%.1f",
                     layout.class_name, layout.box.x, layout.box.x2, left, right)
        layout.box.x, layout.box.x2 = left, right
        widened += 1

    return widened


def install() -> None:
    """
    中文：在 BabelDOC 切分段落之前修好版面框。

    Repair layout boxes just before BabelDOC forms paragraphs.

    Wraps ``ParagraphFinder.process_page``, which is the first stage that reads
    ``page.page_layout`` and the last one that still has the page's characters
    in hand. Idempotent, so a second call is harmless.

    Raises:
        AttributeError: If BabelDOC no longer exposes the method, so an upgrade
            that moves it fails loudly at startup instead of silently doing
            nothing.
    """
    from babeldoc.format.pdf.document_il.midend.paragraph_finder import ParagraphFinder

    if getattr(ParagraphFinder.process_page, "_boxes_patched", False):
        return

    original = ParagraphFinder.process_page

    def process_page(self, page):
        """中文：先修版面框，再走原逻辑。 / Repair the boxes, then run the original pass."""
        try:
            widened = widen_boxes_off_words(page)
            # Second, because it measures the remaining distance from an edge
            # that the word repair has already pulled back to a word boundary.
            # 放在第二步：它衡量的是词修复把框边拉回词边界之后剩下的那点距离。
            extended = widen_boxes_over_orphans(page)
        except Exception:  # noqa: BLE001
            # Failing here must not cost the page: the worst outcome of skipping
            # the repair is the split paragraph we had before.
            # 这里失败不应赔上整页：跳过修复最坏也只是回到原来被切碎的段落。
            logger.warning("Layout-box repair failed on a page", exc_info=True)
        else:
            if widened:
                logger.debug("Widened %d layout box(es) off the words they cut",
                             widened)
            if extended:
                logger.debug("Extended %d layout box(es) over unclaimed text",
                             extended)
        return original(self, page)

    process_page._boxes_patched = True
    ParagraphFinder.process_page = process_page
