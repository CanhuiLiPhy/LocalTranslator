"""
中文：编号列表切分——让「1) … 2) … 3) …」这类悬挂缩进的列表各自成段。

Split hanging-indent numbered lists into one paragraph per item.

BabelDOC already breaks a paragraph at a bullet, but only at a *typographic*
one: its test matches a single character against a set of glyphs — ■ • ◆ ‣ † ‡
and the superscript digits (``layout_helper.BULLET_POINT_PATTERN``). A list
numbered "1)", "2)", "3)" carries no such glyph, so its items stay glued into
one paragraph, and the translation comes back as a single run-on block that has
lost the structure the author put there.

**How the items are recognised.** Three conditions have to hold together, and
each rules out a different way of being wrong:

* The line begins with a list marker — an arabic numeral, a letter or a roman
  numeral, followed by ``)`` ``.`` ``、`` and then a space. Requiring it at the
  *start of a line* already excludes the common false positive, an enumeration
  written inline ("…, and 2) the model must…"), which never begins a line except
  by coincidence.
* The previous line ends a sentence, or stops well short of the column width.
  A line that wraps mid-sentence into "2)" fails this.
* The marker sits at or left of where the following lines start. This is what a
  hanging indent *is*, and it is the property that separates a list item from a
  sentence that merely opens with a number. In the paper this was reported on,
  the markers sit at x=73.6 and their continuation lines at x=89.6.

**Why not fix it in the translator instead.** By the time text reaches the
translator the paragraph is one string, and splitting it there would produce
three translations with nowhere to put them: typesetting lays out one paragraph
per :class:`PdfParagraph`, so the items would be re-joined on the way out. The
split has to happen while the document is still a list of lines with
coordinates, which means the paragraph finder.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# A list marker at the start of a line: "1)", "(2)", "3.", "a)", "iv)", "1、".
# The number is capped at two digits because a longer run of digits at the head
# of a line is a measurement or a year, not an item number.
# 行首的列表标号。位数限制为两位：更长的数字串是数值或年份，不是条目编号。
_MARKER = re.compile(
    r"^\s*[\(（]?(?:\d{1,2}|[a-zA-Z]|[ivxIVX]{1,4})[\)）\.．、]\s+\S"
)

# The marker itself, captured so consecutive items can be recognised by their
# numbering. Kept separate from :data:`_MARKER` because that pattern also has to
# see the character after the marker to reject "1." at the end of a sentence.
# 单独捕获标号本身，用于识别连续编号。与 _MARKER 分开是因为后者还要看标号之后的
# 字符，才能排除句末的 "1."。
_MARKER_VALUE = re.compile(r"^\s*[\(（]?(\d{1,2}|[a-zA-Z]|[ivxIVX]{1,4})[\)）\.．、]")

# Roman numerals that can open a list, in order.
# 可用作列表开头的罗马数字，按顺序排列。
_ROMAN = ("i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x")

# Punctuation that ends the sentence before a list item.
# 列表条目之前那句话的结束标点。
_SENTENCE_END = tuple(".:;?!。：；？！")

# A previous line shorter than this share of the paragraph's width is taken to
# have ended deliberately rather than wrapped.
# 前一行短于段落宽度的该比例时，视为主动断行而非自动换行。
_SHORT_LINE = 0.85

# Slack in points when comparing the marker's left edge with the indent of the
# lines that follow it.
# 比较标号左边缘与后续行缩进时允许的容差（点）。
_INDENT_SLACK = 2.0

# The hanging indent has to be at least this wide to count as one. Below it the
# difference is kerning, not layout.
# 悬挂缩进至少要有这么宽才算数，更小的差异是字距而非版式。
_MIN_HANG = 3.0


def _line_text(line) -> str:
    """
    中文：取出一行的文字内容。

    Read a line's text.

    Args:
        line (PdfLine): Line from BabelDOC's intermediate representation.

    Returns:
        str: The line's characters joined, empty when it has none.
    """
    if line is None or not line.pdf_character:
        return ""
    return "".join(char.char_unicode or "" for char in line.pdf_character)


def _lines(paragraph) -> list:
    """
    中文：列出段落中每个组成部分对应的行（没有行的位置为 None）。

    List the line behind each of a paragraph's compositions.

    Args:
        paragraph (PdfParagraph): Paragraph to inspect.

    Returns:
        list: One entry per composition, either a ``PdfLine`` or ``None``.
    """
    return [
        composition.pdf_line if composition.pdf_line else None
        for composition in paragraph.pdf_paragraph_composition
    ]


def _marker_value(text: str):
    """
    中文：把行首标号解析成序号，认不出来返回 None。

    Read the ordinal a line's marker stands for.

    Args:
        text (str): The line's text.

    Returns:
        int | None: 1 for "1.", "a)" and "i.", and so on. ``None`` when the line
            does not open with a marker this function understands.
    """
    match = _MARKER_VALUE.match(text)
    if not match:
        return None
    token = match.group(1)
    if token.isdigit():
        return int(token)
    lowered = token.lower()
    if lowered in _ROMAN:
        return _ROMAN.index(lowered) + 1
    if len(token) == 1 and token.isalpha():
        return ord(lowered) - ord("a") + 1
    return None


def _continues_numbering(lines, index: int) -> bool:
    """
    中文：判断该行的标号是否正好接在同一缩进的上一个标号之后。

    Decide whether a line's marker follows the previous one in sequence.

    This is what identifies a list that the other tests miss. An item whose text
    ends in a formula neither closes a sentence nor stops short of the column —
    on page 7 of one paper, item 2 ends in the superscript of
    ``QT ≈ (XX⊤)^{-1/2}`` and runs the full width of the item block — so the
    line below it, ``3. Output QT X``, was swallowed into item 2 and its marker
    read as part of the exponent. Numbering settles it: 3 follows 2, at the same
    indent, so it is a new item whatever the previous line looks like.

    Args:
        lines (list): Lines of the paragraph, as returned by :func:`_lines`.
        index (int): Position of the candidate line.

    Returns:
        bool: ``True`` when a line above opens with the preceding ordinal at the
            same indent.
    """
    line = lines[index]
    value = _marker_value(_line_text(line))
    if value is None or value < 2:
        return False

    for position in range(index - 1, -1, -1):
        previous = lines[position]
        if previous is None:
            continue
        earlier = _marker_value(_line_text(previous))
        if earlier is None:
            continue
        if abs(previous.box.x - line.box.x) > _INDENT_SLACK:
            # A marker at a different indent belongs to a nested list, which
            # numbers independently of this one.
            # 缩进不同的标号属于嵌套的另一层列表，其编号与本层无关。
            continue
        return earlier == value - 1
    return False


def _starts_item(lines, index: int, width: float) -> bool:
    """
    中文：判断第 ``index`` 行是否是一个新的编号条目的开头。

    Decide whether a line opens a new numbered item.

    Args:
        lines (list): Lines of the paragraph, as returned by :func:`_lines`.
        index (int): Position of the candidate line.
        width (float): Width of the paragraph, used to judge whether the
            previous line stopped short.

    Returns:
        bool: ``True`` when this line should begin a paragraph of its own.
    """
    line = lines[index]
    previous = lines[index - 1] if index else None
    if line is None or previous is None:
        return False

    text = _line_text(line)
    if not _MARKER.match(text):
        return False

    # Numbering that continues the item above settles it on its own; the tests
    # below exist for the first item, where there is no numbering to continue.
    # 编号接续上一条即可定案；下面的判据是为第一条准备的，那里没有编号可接续。
    if _continues_numbering(lines, index):
        return True

    previous_text = _line_text(previous).rstrip()
    if not previous_text:
        return False
    previous_width = previous.box.x2 - previous.box.x
    ended = previous_text.endswith(_SENTENCE_END)
    stopped_short = width > 0 and previous_width < width * _SHORT_LINE
    if not ended and not stopped_short:
        return False

    # A hanging indent: the marker starts at or left of the lines that continue
    # the item. Compared against the next line rather than the paragraph box,
    # which may include the lead-in sentence at a smaller indent still.
    # 悬挂缩进：标号所在位置不右于其后的续行。这里与下一行比较而不与段落框比较，
    # 因为段落框可能还包含缩进更小的引导句。
    following = lines[index + 1] if index + 1 < len(lines) else None
    if following is None:
        # The last line of the paragraph: no continuation to compare against, so
        # the marker and the sentence break carry the decision on their own.
        # 段落最后一行：没有续行可比，判断只能由标号与断句承担。
        return True
    hang = following.box.x - line.box.x
    if hang >= _MIN_HANG:
        return True
    return abs(hang) <= _INDENT_SLACK and _is_flat_list(lines, index)


def _is_flat_list(lines, index: int) -> bool:
    """
    中文：判断这是否是一组单行条目（每条只占一行，因而没有悬挂缩进可比）。

    Decide whether the surrounding lines form a list of single-line items.

    A one-line item has no continuation to hang under, so the indent test cannot
    apply. What identifies it instead is its neighbours: the line after it also
    opens with a marker.

    Args:
        lines (list): Lines of the paragraph.
        index (int): Position of the candidate line.

    Returns:
        bool: ``True`` when the next line is itself a marker line.
    """
    following = lines[index + 1] if index + 1 < len(lines) else None
    return following is not None and bool(_MARKER.match(_line_text(following)))


def _split_paragraphs(finder, paragraphs) -> int:
    """
    中文：把每个含有多个编号条目的段落按条目切开。

    Split every paragraph that holds more than one numbered item.

    Args:
        finder (ParagraphFinder): The finder instance, used for its
            ``update_paragraph_data`` bookkeeping.
        paragraphs (list[PdfParagraph]): Paragraphs of one page, modified in
            place.

    Returns:
        int: Number of new paragraphs created.
    """
    from babeldoc.format.pdf.document_il.il_version_1 import Box, PdfParagraph
    from babeldoc.format.pdf.document_il.midend.paragraph_finder import (
        generate_base58_id,
    )

    created = 0
    index = 0
    while index < len(paragraphs):
        paragraph = paragraphs[index]
        compositions = paragraph.pdf_paragraph_composition
        if not compositions or len(compositions) <= 1:
            index += 1
            continue

        lines = _lines(paragraph)
        width = paragraph.box.x2 - paragraph.box.x if paragraph.box else 0.0

        cut = None
        for position in range(1, len(lines)):
            if _starts_item(lines, position, width):
                cut = position
                break

        if cut is None:
            index += 1
            continue

        tail = PdfParagraph(
            box=Box(0, 0, 0, 0),
            pdf_paragraph_composition=compositions[cut:],
            unicode="",
            debug_id=generate_base58_id(),
            layout_label=paragraph.layout_label,
            layout_id=paragraph.layout_id,
        )
        paragraph.pdf_paragraph_composition = compositions[:cut]
        finder.update_paragraph_data(paragraph, update_unicode=True)
        finder.update_paragraph_data(tail, update_unicode=True)
        paragraphs.insert(index + 1, tail)
        created += 1

        # Re-examine the tail rather than advancing past it: a list of three
        # items needs two cuts, and the second one lives in the piece just made.
        # 不跳过刚生成的尾段而是重新检查它：三条的列表需要切两刀，第二刀就在刚切
        # 出来的这一段里。
        index += 1

    return created


def install() -> None:
    """
    中文：给 BabelDOC 的段落切分补上编号列表这一情形。

    Extend BabelDOC's paragraph splitting to cover numbered lists.

    Wraps ``ParagraphFinder.process_independent_paragraphs`` rather than
    replacing it, so BabelDOC's own rules — table-of-contents dot leaders, short
    trailing lines, typographic bullets — keep working and this only adds a pass
    afterwards.

    Raises:
        AttributeError: If BabelDOC no longer exposes the method, so an upgrade
            that moves it fails loudly at startup instead of silently doing
            nothing.
    """
    from babeldoc.format.pdf.document_il.midend.paragraph_finder import ParagraphFinder

    if getattr(ParagraphFinder.process_independent_paragraphs, "_lists_patched", False):
        return

    original = ParagraphFinder.process_independent_paragraphs

    def process_independent_paragraphs(self, paragraphs, median_width):
        """中文：先跑原逻辑，再补编号列表切分。 / Run the original pass, then split numbered lists."""
        original(self, paragraphs, median_width)
        try:
            created = _split_paragraphs(self, paragraphs)
        except Exception:  # noqa: BLE001
            # A failure here must not cost the document: the worst outcome of
            # skipping the split is the run-on paragraph we had before.
            # 这里失败不应赔上整篇文档：跳过切分最坏也只是回到原来的长段落。
            logger.warning("Numbered-list splitting failed on a page", exc_info=True)
            return
        if created:
            logger.debug("Split %d numbered list item(s) into their own paragraphs",
                         created)

    process_independent_paragraphs._lists_patched = True
    ParagraphFinder.process_independent_paragraphs = process_independent_paragraphs
