"""
中文：目录页逐行原位翻译——只换标题文字，行位置、引线、页码与链接都不动。

Translate a contents page one row at a time, in place.

A contents page cannot go through the ordinary pipeline. Three things break at
once, and they were all measured on the paper this was written against:

* **The page becomes one paragraph.** BabelDOC does split on a dot leader, but
  its test is ``re.search(r"\\.{20,}", prev_text)`` — twenty *consecutive* dots.
  LaTeX sets leaders as ``. . . . .`` with spaces, so the rule never fires and
  the whole page is translated as a single block.
* **The leaders and page numbers come adrift.** A row is title, leader, then a
  right-aligned number. The leader is fixed-length text, so once the title
  changes width the number lands somewhere else and the right margin is ragged.
* **The links lose their anchor.** Every row is a link, and re-anchoring works
  by searching for the text under the source rectangle — text that here is
  interleaved with leader dots and a page number.

**What this does instead.** The contents page is already replayed verbatim from
the source (``regions.stamp_all`` with the ``contents`` kind), so the starting
point is a pixel-exact copy. Each row is then rewritten on its own: the title is
redacted and redrawn at the same baseline and the same left edge, the leader is
refilled to the width the new title leaves, and the page number is never touched.
Nothing reflows, so no row can push another one out of place.

**How the rows are found.** By the page-number column. Every row ends in a bare
integer whose right edge sits on the same column — measured across one contents
page, 38 rows ended in a number and every one of those right edges fell between
540.0 and 540.1. Requiring the right edge to agree with the column's own mode to
within a point is enough to pick the rows out, and it hands over the baseline,
the number, and the x to fill the leader up to at the same time.

**Why not the outline.** ``doc.get_toc()`` gives cleaner titles, and on the test
paper its 37 entries match the page's 37 links one for one. But a contents page
does not always come with an outline, whereas the page numbers are on the page by
definition. The outline is used to check the row count when it is there.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

# A leader is drawn as a run of full stops, one word per stop once the spaces
# between them are taken into account.
# 引线由一串句点构成，考虑到点之间的空格，抽取出来是一个点一个词。
_LEADER_CHAR = "."

# The page number that closes a row. Supplementary material numbers its pages
# with a letter in front — arXiv:2511.03112 runs S1..S30 — and without the
# prefix the walk back from the end stops on the letter, which leaves the
# leader dots inside the title and sends forty of them to the translator.
# 一行末尾的页码。补充材料的页码前面带字母——arXiv:2511.03112 是 S1..S30——不认这个
# 前缀的话，从行尾回溯会停在字母上，引线的点就留在标题里，四十个点一起被送去翻译。
_NUMBER = re.compile(r"^[A-Za-z]?\d{1,4}$")

# How far a row's page number may sit from the column's own right edge, in
# points. The column is set by the same right margin for every row, so this only
# absorbs the difference between glyph widths.
# 一行的页码与该列右边缘的最大偏差（点）。整列由同一个右边界排出，该容差只吸收
# 字形宽度的差异。
_COLUMN_SLACK = 1.5

# A page needs this many aligned rows before it is treated as a contents page.
# Two could be a coincidence in a table; a table's own numbers are not usually
# right-aligned against the text margin.
# 一页要有这么多对齐的行才当作目录页。两行可能是表格里的巧合；表格里的数字通常
# 不会与正文右边界对齐。
_MIN_ROWS = 4

# Dots a row must carry before its leader counts as a real one. Alignment alone
# is not evidence: measured on arXiv:2608.18004, which has no contents page at
# all, four separate pages were read as one — a colorbar tick `0.4` ends in a
# digit whose right edge agrees with the column, and the decimal point in front
# of that digit was taken for a leader. A leader spans the whole gap between the
# title and the page number, so three dots separate the two cases cleanly.
# 一行的引线要有这么多个点才算数。仅凭对齐不足为证：实测 arXiv:2608.18004 根本没有
# 目录页，却有四页被判成目录——色标刻度 `0.4` 以数字结尾且右边缘与列对齐，而那个数字
# 前面的小数点被当成了引线。真正的引线要横跨标题与页码之间的整段空白，三个点足以把
# 两种情况分开。
_MIN_LEADER_DOTS = 3

# Rows carrying a real leader that a page needs before it is read as a contents
# page. A genuine contents page is mostly leader rows; a table is none of them.
# Chapter-level rows that legitimately have no leader are still kept once the
# page has qualified.
# 一页要有这么多带真引线的行，才判定为目录页。真正的目录页绝大多数行都有引线，表格
# 一行都没有。一旦该页过关，本来就没有引线的章级行仍然照收。
_MIN_LEADER_ROWS = 3

# A run of letters long enough to count as a word, and the characters that make
# a row read as data rather than as a title.
# 足以算作一个词的字母串，以及使一行读起来像数据而非标题的字符。
_WORD_RUN = re.compile(r"[A-Za-z\u00c0-\u024f]{2,}|[\u4e00-\u9fff]")
_DATA_CHARS = "0123456789.,%±+-/()[]<>=×"

# Share of a title that may be digits and arithmetic before the row is read as a
# table row rather than a contents entry. A contents entry is a section heading
# with at most a section number in front of it.
# 标题中数字与算式字符的占比上限，超过即视为表格行而非目录条目。目录条目是一个章节
# 标题，最多在前面带一个章节号。
_MAX_DATA_SHARE = 0.5

# Characters whose baselines differ by less than this belong to one row.
# 基线相差小于该值的字符属于同一行。
_ROW_TOLERANCE = 2.0

# Gap kept between the title and the first leader dot, and between the last dot
# and the page number, in points.
# 标题与第一个引线点之间、最后一个点与页码之间保留的间隙（点）。
_LEADER_GAP = 3.0

# Spacing between leader dots, in points. Taken from the source row when it can
# be measured, and this is the fallback.
# 引线点之间的间距（点）。能从源文行量到时就用量到的值，此处是兜底。
_DOT_PITCH = 5.0

# The title may be set down to this share of its original size to fit the space
# the row leaves. Below it the row would no longer match the page.
# 标题最多缩到原字号的这一比例以塞进该行剩余空间。再小就与整页不协调了。
_MIN_SCALE = 0.75

# Fonts BabelDOC itself embeds, used here so a rewritten row matches the rest of
# the translation. Reading the file rather than the output's own embedded copy is
# deliberate: the copy in the output has been subsetted to the glyphs the body
# text needed, and a title may well use others.
# BabelDOC 自己内嵌的字体，这里沿用以保证重写的行与译文其余部分一致。特意读文件而
# 不用产物里已内嵌的那份：那份已按正文用到的字形做了子集化，标题很可能用到别的字。
_FONT_DIR = Path.home() / ".cache" / "babeldoc" / "fonts"
_FONT_FILES = {
    False: "SourceHanSerifCN-Regular.ttf",
    True: "SourceHanSerifCN-Bold.ttf",
}


def _load_fonts():
    """
    中文：载入重写目录行所用的两种字重。

    Load the two weights a rewritten row may be set in.

    A contents page sets chapter entries bold and their subsections regular, and
    a rewritten row keeps whichever weight its source row had.

    Returns:
        dict[bool, pymupdf.Font] | None: Regular under ``False`` and bold under
            ``True``, or ``None`` when the files are not on this machine — in
            which case the caller leaves the page alone rather than drawing in a
            face that does not match the rest of the translation.
    """
    import pymupdf

    fonts = {}
    for bold, name in _FONT_FILES.items():
        path = _FONT_DIR / name
        if not path.is_file():
            logger.warning("Contents translation needs %s; leaving the page as it is",
                           path)
            return None
        try:
            fonts[bold] = pymupdf.Font(fontfile=str(path))
        except Exception:  # noqa: BLE001
            logger.warning("Could not load %s", path, exc_info=True)
            return None
    return fonts


def _lines(page):
    """
    中文：把一页的字符按基线聚成行。

    Group a page's characters into rows by their baseline.

    Args:
        page (pymupdf.Page): Page to read.

    Returns:
        list[list[dict]]: One list of characters per row, each sorted left to
            right, and each character carrying the ``size`` of its span.
    """
    try:
        blocks = page.get_text("rawdict")["blocks"]
    except Exception:  # noqa: BLE001
        return []

    rows: dict[float, list] = {}
    for block in blocks:
        if block.get("type") != 0:
            continue
        for line in block.get("lines", ()):
            for span in line.get("spans", ()):
                size = span.get("size", 0.0)
                flags = span.get("flags", 0)
                colour = span.get("color", 0)
                for char in span.get("chars", ()):
                    baseline = char["origin"][1]
                    key = next(
                        (k for k in rows if abs(k - baseline) <= _ROW_TOLERANCE),
                        baseline,
                    )
                    rows.setdefault(key, []).append(
                        {**char, "size": size, "flags": flags,
                         "colour": colour})

    return [sorted(chars, key=lambda c: c["origin"][0])
            for _, chars in sorted(rows.items())]


def _column(rows) -> float | None:
    """
    中文：找出页码列的右边缘。

    Find the right edge of the page-number column.

    Args:
        rows (list): Rows from :func:`_lines`.

    Returns:
        float | None: The x every row's number ends at, or ``None`` when too few
            rows end in a number for the page to be a contents page.
    """
    edges: dict[float, int] = {}
    for chars in rows:
        trailing = _trailing_number(chars)
        if trailing is None:
            continue
        edge = round(chars[trailing[2] - 1]["bbox"][2], 1)
        edges[edge] = edges.get(edge, 0) + 1
    if not edges:
        return None

    best, count = max(edges.items(), key=lambda item: item[1])
    # Neighbouring edges within the slack belong to the same column.
    # 容差之内的相邻边缘属于同一列。
    total = sum(n for edge, n in edges.items() if abs(edge - best) <= _COLUMN_SLACK)
    return best if total >= _MIN_ROWS else None


def _trailing_number(chars):
    """
    中文：取出一行末尾的裸数字。

    Read the bare number a row ends with.

    Args:
        chars (list[dict]): One row's characters, left to right.

    Returns:
        tuple[str, int, int] | None: The number, the index of its first
            character and the index one past its last, or ``None`` when the row
            does not end in one.
    """
    end = len(chars)
    while end > 0 and (chars[end - 1].get("c") or " ").isspace():
        end -= 1
    start = end
    while start > 0 and (chars[start - 1].get("c") or "").isdigit():
        start -= 1
    if start == end:
        return None
    if start > 0 and (chars[start - 1].get("c") or "").isalpha():
        start -= 1
    text = "".join(chars[i]["c"] for i in range(start, end))
    return (text, start, end) if _NUMBER.match(text) else None


def _split_row(chars, number_start: int):
    """
    中文：把一行切成「标题 / 引线 / 页码」三段。

    Split a row into its title, its leader and its page number.

    Args:
        chars (list[dict]): One row's characters, left to right.
        number_start (int): Index of the first digit of the page number.

    Returns:
        tuple[int, int] | None: ``(title_end, leader_start)`` as indices into
            ``chars``, or ``None`` when the row holds no title.
    """
    leader_start = None
    for index in range(number_start - 1, 0, -1):
        char = chars[index].get("c") or ""
        if char == _LEADER_CHAR:
            leader_start = index
        elif not char.isspace():
            break
    if leader_start is None:
        leader_start = number_start

    title_end = leader_start
    while title_end > 0 and (chars[title_end - 1].get("c") or " ").isspace():
        title_end -= 1
    return (title_end, leader_start) if title_end > 0 else None


def _dot_pitch(chars, leader_start: int, number_start: int) -> float:
    """
    中文：量出源文引线的点距。

    Measure the spacing between the source row's leader dots.

    Args:
        chars (list[dict]): One row's characters.
        leader_start (int): Index of the first leader dot.
        number_start (int): Index of the first digit of the page number.

    Returns:
        float: Distance between consecutive dots in points, falling back to
            :data:`_DOT_PITCH` when the row has fewer than two of them.
    """
    dots = [c["origin"][0] for c in chars[leader_start:number_start]
            if (c.get("c") or "") == _LEADER_CHAR]
    if len(dots) < 2:
        return _DOT_PITCH
    return (dots[-1] - dots[0]) / (len(dots) - 1)


# A gap wider than this share of the font size separates two words. LaTeX often
# positions a contents entry's title with a kern rather than a space glyph, so
# reading the characters straight through yields "1Introduction" — and the
# translator then has no section number to hold back.
# 间距超过字号的这一比例即为词间空白。LaTeX 常用字距而非空格字形来定位目录条目的
# 标题，于是逐字读出来是 "1Introduction"——翻译时也就没有章节号可以保留。
_WORD_GAP = 0.25


def _join(chars, end: int) -> str:
    """
    中文：把一行前 ``end`` 个字符拼成文字，按字距补回丢失的空格。

    Join a row's leading characters, restoring the spaces the kerning stands for.

    Args:
        chars (list[dict]): One row's characters, left to right.
        end (int): One past the last character to take.

    Returns:
        str: The text, stripped.
    """
    out = []
    previous = None
    for index in range(end):
        char = chars[index]
        text = char.get("c") or ""
        if previous is not None and text and not text.isspace():
            gap = char["bbox"][0] - previous["bbox"][2]
            if gap > previous.get("size", 10.0) * _WORD_GAP and not out[-1].isspace():
                out.append(" ")
        out.append(text)
        previous = char
    return "".join(out).strip()


def _reads_as_a_title(text: str) -> bool:
    """
    中文：判断一行的标题读起来像章节名，而不像一行数据。

    Decide whether a row's title reads as a section heading.

    A contents entry is prose: it carries words, and at most a section number in
    front of them. A table row is the opposite — mostly digits, decimal points
    and error bars — and a figure's colorbar labels carry no words at all. Both
    were being rewritten as contents rows on arXiv:2608.18004.

    Args:
        text (str): The row's title as read off the page.

    Returns:
        bool: ``True`` when the title carries at least one word and is not
            mostly arithmetic.
    """
    if not _WORD_RUN.search(text):
        return False
    dense = [char for char in text if not char.isspace()]
    if not dense:
        return False
    data = sum(1 for char in dense if char in _DATA_CHARS)
    return data <= len(dense) * _MAX_DATA_SHARE


def _leader_dots(chars, leader_start: int, number_start: int) -> int:
    """
    中文：数一行引线里的点数。

    Count the leader dots in one row.

    Args:
        chars (list[dict]): The row's characters, left to right.
        leader_start (int): Index of the first leader dot.
        number_start (int): Index of the first digit of the page number.

    Returns:
        int: How many full stops lie between the title and the page number.
    """
    return sum(1 for index in range(leader_start, number_start)
               if (chars[index].get("c") or "") == _LEADER_CHAR)


def rows_of(page) -> list[dict]:
    """
    中文：读出一页目录的每一行。

    Read the rows of a contents page.

    Args:
        page (pymupdf.Page): Page to read, in the source's own layout.

    Returns:
        list[dict]: One entry per row with ``title``, ``left``, ``baseline``,
            ``size``, ``top``, ``bottom``, ``number_x`` and ``pitch``. Empty
            when the page is not a contents page.
    """
    rows = _lines(page)
    column = _column(rows)
    if column is None:
        return []

    found = []
    for chars in rows:
        trailing = _trailing_number(chars)
        if trailing is None:
            continue
        _, number_start, number_end = trailing
        if abs(chars[number_end - 1]["bbox"][2] - column) > _COLUMN_SLACK:
            continue

        split = _split_row(chars, number_start)
        if split is None:
            continue
        title_end, leader_start = split

        title = _join(chars, title_end)
        if not title or not _reads_as_a_title(title):
            continue
        dots = _leader_dots(chars, leader_start, number_start)
        boxes = [chars[i]["bbox"] for i in range(number_start)]
        found.append({
            "dots": dots,
            "title": title,
            "bold": bool(chars[0].get("flags", 0) & 16),
            "colour": chars[0].get("colour", 0),
            "left": chars[0]["bbox"][0],
            "baseline": chars[0]["origin"][1],
            "size": chars[0]["size"],
            "top": min(b[1] for b in boxes),
            "bottom": max(b[3] for b in boxes),
            "number_x": chars[number_start]["bbox"][0],
            # A chapter-level row often has no leader at all — the source sets
            # its title and page number with nothing between them — and adding
            # one would be a change the reader can see.
            # 章级行常常根本没有引线——源文的标题与页码之间是空的——补上一条读者一眼
            # 就能看出是改动。
            "leader": leader_start < number_start,
            "pitch": _dot_pitch(chars, leader_start, number_start),
        })

    if len(found) < _MIN_ROWS:
        return []
    # The page has to look like a contents page as a whole, not merely hold four
    # rows whose last digit lines up. Without this a data table qualifies.
    # 整页必须像目录页，而不只是碰巧有四行末位数字对齐。没有这一条，数据表就会过关。
    if sum(1 for row in found if row["dots"] >= _MIN_LEADER_DOTS) < _MIN_LEADER_ROWS:
        logger.debug("Page has aligned rows but no leaders; not a contents page")
        return []
    return found


# A translation this much longer than its source is not a translation of a
# contents entry. The model echoed one three-word proper noun back as
# "Gram Newton-Schulz=Gram Newton-Schulz"; a title is a few words, so a large
# expansion is a repetition rather than a rendering.
# 比原文长这么多的"译文"不是目录条目的译文。模型曾把一个三词专名回吐成
# "Gram Newton-Schulz=Gram Newton-Schulz"；标题只有几个词，大幅变长说明是复读而非翻译。
_MAX_EXPANSION = 2.0

# A stretch of Latin long enough to be a name rather than an abbreviation.
# 足以构成名称（而非缩写）的拉丁字母串。
_LATIN_RUN = re.compile(r"[A-Za-z][A-Za-z0-9 \-]{5,}[A-Za-z0-9]")


# Sentence-ending punctuation a heading does not carry. The model closed
# 「模型质量得以保持」 with a full stop because it read the entry as a sentence;
# no other row on the page ends in one.
# 标题不会带的句末标点。模型把「Model Quality Is Preserved」当成句子，给
# 「模型质量得以保持」结了个句号，而该页其它任何一行都没有。
_TRAILING = "。．.！!？?；;，,、"


def _tidy(translated: str) -> str:
    """
    中文：去掉译后标题末尾的句末标点。

    Strip the sentence punctuation a translated heading does not need.

    Args:
        translated (str): What the translator returned.

    Returns:
        str: The title without its trailing punctuation.
    """
    return translated.strip().rstrip(_TRAILING).strip()


def _is_usable(title: str, translated: str) -> bool:
    """
    中文：判断译文是否可用——非空、有变化、且没有把原文复读一遍。

    Decide whether a translated title may replace its original.

    Args:
        title (str): The row's original title.
        translated (str): What the translator returned.

    Returns:
        bool: ``False`` when the row should be left in the original language.
    """
    if not translated or translated == title:
        return False
    if title and len(translated) > len(title) * _MAX_EXPANSION:
        return False

    # A Latin run the translation repeats more often than the source did is the
    # model stuttering over a proper noun. "A Alternative Experiments on Gram
    # Newton-Schulz" came back as "Gram Newton-Schulz=Gram Newton-Schulz 的替代
    # 实验" — the whole title never repeats, so a whole-title test misses it.
    # 译文里某个拉丁串出现的次数多于原文，说明模型在专有名词上卡住了复读。
    # "A Alternative Experiments on Gram Newton-Schulz" 回来是
    # "Gram Newton-Schulz=Gram Newton-Schulz 的替代实验"——整条标题并没有重复，
    # 只比对整条标题是发现不了的。
    for run in _LATIN_RUN.findall(translated):
        if translated.count(run) > max(1, title.count(run)):
            return False
    return True


def _rewrite(page, row, translated: str, fonts, offset: float) -> bool:
    """
    中文：把一行的标题换成译文，并把引线补到页码前。

    Replace one row's title with its translation and refill the leader.

    Args:
        page (pymupdf.Page): Target page, already holding a verbatim copy of the
            source row.
        row (dict): The row, as returned by :func:`rows_of`.
        translated (str): Translated title.
        fonts (dict[bool, pymupdf.Font]): Regular and bold faces.
        offset (float): Horizontal shift, for the translated half of a dual page.

    Returns:
        bool: ``True`` when the row was rewritten.
    """
    import pymupdf

    left = row["left"] + offset
    limit = row["number_x"] + offset - _LEADER_GAP
    available = limit - left - _LEADER_GAP
    if available <= 0:
        return False

    font = fonts[row.get("bold", False)]
    size = row["size"]
    width = font.text_length(translated, fontsize=size)
    if width > available:
        size = max(size * _MIN_SCALE, size * available / width)
        width = font.text_length(translated, fontsize=size)
    if width > available:
        # Even at the floor the title does not fit; leaving the English row is
        # better than one that overruns its own page number.
        # 缩到下限仍放不下；保留英文原行也好过让标题压到页码上。
        return False

    baseline = row["baseline"]
    writer = pymupdf.TextWriter(page.rect)
    writer.append((left, baseline), translated, font=font, fontsize=size)

    # Each dot is placed at its own x rather than written as one ". . ." run.
    # A run advances by the font's own width for ". ", which is not the pitch the
    # source used: the first attempt ran the leader of every chapter row past its
    # own page number and off the text block, one row reaching x=600 on a page
    # whose margin is 540.
    # 每个点单独定位，而不是整体写成一串 ". . ."。整串写出时按字体自身的 ". " 宽度步进，
    # 与源文的点距并不相同：第一版让每个章级行的引线越过了自己的页码、冲出版心，
    # 有一行到了 x=600，而该页右边界是 540。
    if row.get("leader", True):
        pitch = max(row["pitch"], 1.0)
        start = left + width + _LEADER_GAP
        position = limit - pitch
        while position >= start:
            writer.append((position, baseline), _LEADER_CHAR, font=font,
                          fontsize=size)
            position -= pitch

    colour = row.get("colour", 0)
    writer.write_text(page, color=(
        ((colour >> 16) & 0xFF) / 255.0,
        ((colour >> 8) & 0xFF) / 255.0,
        (colour & 0xFF) / 255.0,
    ))
    return True


def translate_pages(source_path, target_path, translate, dual: bool = False,
                    translate_first: bool = False) -> dict[str, int]:
    """
    中文：把译文产物里的目录页逐行改成中文。

    Rewrite the contents pages of a translated PDF, one row at a time.

    Args:
        source_path (Path | str): Original PDF, read for the row geometry.
        target_path (Path | str): Translated PDF, modified in place. Its contents
            pages must already hold a verbatim copy of the source's.
        translate (Callable[[str], str]): Translates one title.
        dual (bool): The target is a side-by-side dual PDF.
        translate_first (bool): The dual layout puts the translation on the left.

    Returns:
        dict[str, int]: ``rows`` rewritten and ``pages`` touched.
    """
    import pymupdf

    result = {"rows": 0, "pages": 0}
    source_path, target_path = Path(source_path), Path(target_path)
    if not source_path.is_file() or not target_path.is_file():
        return result

    fonts = _load_fonts()
    if fonts is None:
        return result

    try:
        source = pymupdf.open(source_path)
        target = pymupdf.open(target_path)
    except Exception:  # noqa: BLE001
        logger.warning("Could not open PDFs for contents translation", exc_info=True)
        return result

    try:
        touched = False
        for index in range(min(source.page_count, target.page_count)):
            rows = rows_of(source[index])
            if not rows:
                continue

            dst_page = target[index]
            if dual:
                half = dst_page.rect.width / 2.0
                offset = 0.0 if translate_first else half
            else:
                offset = 0.0

            pending = []
            for row in rows:
                try:
                    translated = _tidy(translate(row["title"]) or "")
                except Exception:  # noqa: BLE001
                    logger.warning("Could not translate a contents row", exc_info=True)
                    continue
                if not _is_usable(row["title"], translated):
                    continue
                area = pymupdf.Rect(
                    row["left"] + offset - 0.5,
                    row["top"] - 0.5,
                    row["number_x"] + offset - _LEADER_GAP,
                    row["bottom"] + 0.5,
                )
                dst_page.add_redact_annot(area, fill=(1, 1, 1))
                pending.append((row, translated))

            if not pending:
                continue

            try:
                # Applied once per page: applying redactions drops every
                # annotation on the page, so a second pass would undo the first.
                # 按页统一应用：应用涂抹会清掉该页所有注释，分两次做会互相抵消。
                dst_page.apply_redactions(
                    images=pymupdf.PDF_REDACT_IMAGE_NONE,
                    graphics=pymupdf.PDF_REDACT_LINE_ART_NONE,
                    text=pymupdf.PDF_REDACT_TEXT_REMOVE,
                )
            except Exception:  # noqa: BLE001
                logger.warning("Could not clear contents rows on page %d", index + 1,
                               exc_info=True)
                continue

            for row, translated in pending:
                if _rewrite(dst_page, row, translated, fonts, offset):
                    result["rows"] += 1
            result["pages"] += 1
            touched = True

        if touched:
            temp = target_path.with_suffix(target_path.suffix + ".tocfix")
            target.save(temp, garbage=4, deflate=True)
            # Close before replacing, not after. POSIX lets a rename overwrite
            # a file another handle still has open; Windows answers
            # PermissionError [WinError 5]. The `finally` below closes only what
            # is still open, because pymupdf raises ValueError on a second
            # close — see engine/images.py for the run this was found on.
            # 覆盖之前先关闭，而不是之后。POSIX 允许改名覆盖一个仍被打开的文件，Windows
            # 会以 PermissionError [WinError 5] 作答。下面的 finally 只关闭仍然打开的那些：
            # pymupdf 对已关闭的文档再次 close() 会抛 ValueError——发现这个问题的那次运行见
            # engine/images.py。
            source.close()
            target.close()
            temp.replace(target_path)
            logger.info("Translated %d contents row(s) across %d page(s) in %s",
                        result["rows"], result["pages"], target_path.name)
    finally:
        for document in (source, target):
            if not document.is_closed:
                document.close()

    return result


def translate_all(source_path, outputs: dict, translate,
                  translate_first: bool = False) -> dict[str, int]:
    """
    中文：对所有产物执行目录页逐行翻译。

    Rewrite the contents pages of every produced PDF.

    Args:
        source_path (Path | str): Original PDF.
        outputs (dict[str, str | None]): Produced paths keyed by kind.
        translate (Callable[[str], str]): Translates one title.
        translate_first (bool): The dual layout puts the translation on the left.

    Returns:
        dict[str, int]: Combined counts across the files rewritten.
    """
    totals = {"rows": 0, "pages": 0}

    mono = outputs.get("mono_path")
    if mono:
        for key, value in translate_pages(source_path, mono, translate).items():
            totals[key] += value

    dual = outputs.get("dual_path")
    if dual:
        for key, value in translate_pages(source_path, dual, translate, dual=True,
                                          translate_first=translate_first).items():
            totals[key] += value

    return totals
