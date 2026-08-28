"""
中文：受保护区块——把伪代码、表格与行间公式整块从源 PDF 原样搬回译文页。

Stamp pseudocode, tables and display equations back from the source, verbatim.

Leaving a region untranslated is not the same as leaving it alone. BabelDOC
rebuilds every page from its intermediate representation regardless of whether a
paragraph was translated, so an untranslated block is still re-typeset: each line
is re-laid with its own font metrics, which is why the steps of an algorithm come
back at inconsistent sizes and no longer share a left edge. Worse, anything in
the block that is not text does not survive at all — the vinculum over a radical
is a drawn line, not a glyph, and nothing in the text pipeline re-creates it.

So these regions are treated the way figures are: the output area is covered and
the source region is replayed into it with ``show_pdf_page``, which reproduces
glyphs, spacing, rules and vector artwork exactly as the author drew them. Text
filtering alone cannot achieve this, and the two mechanisms are complementary —
filtering stops the model wasting calls on notation, stamping is what makes the
result look right.

**Finding the regions.** The layout model that BabelDOC already uses reports
``table`` and ``isolate_formula`` boxes directly. It has no class for pseudocode:
the same paper has its algorithm floats labelled ``table`` on one page and
``plain text`` on another. A block is therefore taken to be pseudocode when it
carries an ``Algorithm N`` caption and numbered step markers, which no ordinary
paragraph does.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# Layout classes the model emits that should be preserved verbatim.
# 版面模型输出的、需原样保留的类别。
# Figures are included for the same reason as tables, and for one more of their
# own. The ink-comparison restoration in :mod:`images` finds a figure by looking
# for areas that carry ink in the source and none in the output, which trims a
# figure to the part that is solidly inked: a column of thin line drawings on
# white at the edge of a plate falls under the ink threshold and is cut off, and
# figure 3 of one paper came back missing its rightmost column of skeletons
# entirely. A layout box does not have that problem — it is the figure's extent
# as the model sees it, sparse regions included.
# 插图纳入的理由与表格相同，另有一条自己的理由。images 模块基于墨迹比较的修复是靠
# 「源页有墨、译文页无墨」找图，这会把图裁到墨迹密实的那部分：图版边缘那一列白底细线
# 稿低于墨迹阈值而被切掉——某篇论文的图 3 就整列骨架图没了。版面框没有这个问题，它就是
# 模型看到的插图完整范围，稀疏部分也在内。
_PROTECTED_LABELS = ("table", "isolate_formula", "figure")

# An algorithm float is identified by its caption and its numbered steps. A
# paragraph of prose can mention "Algorithm 3", but it does not also contain
# lines beginning "4:".
# 算法浮动块由标题与带编号的步骤共同确定。正文可以提到 "Algorithm 3"，但不会同时
# 含有以 "4:" 开头的行。
_ALGORITHM_CAPTION = re.compile(r"\bAlgorithm\s+\d+\b")
_STEP_MARKER = re.compile(r"(?:^|\n)\s*\d{1,3}:\s", re.M)
_MIN_STEPS = 3

# Leader dots in a contents entry. The spaces matter: LaTeX sets the leader as
# ". . . . ." and a pattern demanding consecutive dots matches nothing.
# 目录条目中的点导引。空格很关键：LaTeX 排出来是 ". . . . ."，要求连续点的正则匹配不到。
_LEADER = re.compile(r"(?:\.\s*){6,}")

# A page needs this many dotted entries before it is taken to be a contents page.
# 一页要有这么多带点导引的条目，才认定为目录页。
_MIN_TOC_ENTRIES = 3

# Bottom fraction of the page treated as the footer, excluded from the block.
# 视为页脚的页面底部比例，不纳入区块。
_FOOTER_BAND = 0.07

# Render scale for the layout model, matching what BabelDOC uses.
# 版面模型的渲染倍率，与 BabelDOC 保持一致。
_ZOOM = 2.0

# Regions smaller than this in square points are not worth stamping.
# 小于该面积（平方点）的区域不值得搬运。
_MIN_AREA = 1500.0

# The stamped area is grown by this many points so a box that clips a descender
# or a table rule by a hair still covers it.
# 搬运区域外扩该点数，使框线或下伸笔画被切掉一点点时仍能覆盖到。
_MARGIN = 2.0

# Vertical distance, in points, that ends an algorithm float. Its own lines sit
# within a line height of each other; the prose or page number after it does not.
# 结束算法浮动块的垂直间隔（点）。块内各行彼此相距不到一个行高，其后的正文或页码则不然。
_BLOCK_GAP = 20.0

# A line carrying this many function words has returned to prose.
# 含有这么多虚词的行说明已回到正文。
_PROSE_WORDS = re.compile(
    r"\b(we|our|is|are|was|were|be|this|that|these|those|which|where|when|by"
    r"|and|but|to|of|the|from|than|will|would|can|could|should|may|as|it|has|have)\b",
    re.I,
)
_MAX_PROSE_WORDS = 5


# Which layout class each switchable kind corresponds to.
# 各可选类型对应的版面类别。
_KIND_LABELS = {"figure": "figure", "table": "table", "formula": "isolate_formula"}


def _page_regions(page, model, names, kinds) -> list:
    """
    中文：找出一页上需要原样保留的区块。

    Locate the regions of one page that must be preserved verbatim.

    Args:
        page (pymupdf.Page): Source page.
        model: Loaded DocLayout model.
        names (dict[int, str]): Class index to label name.

    Returns:
        list[pymupdf.Rect]: Regions in page coordinates.
    """
    import numpy as np
    import pymupdf

    pix = page.get_pixmap(matrix=pymupdf.Matrix(_ZOOM, _ZOOM))
    image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:
        image = image[:, :, :3]
    # The model expects BGR, as OpenCV would deliver.
    # 模型按 OpenCV 的习惯接收 BGR。
    result = model.predict(image[:, :, ::-1], imgsz=1024)[0]

    scale_x = page.rect.width / pix.width
    scale_y = page.rect.height / pix.height

    wanted_labels = {
        _KIND_LABELS[kind] for kind in kinds if kind in _KIND_LABELS
    }
    out: list = []
    for box in result.boxes:
        x0, y0, x1, y1 = (float(v) for v in box.xyxy)
        rect = pymupdf.Rect(x0 * scale_x, y0 * scale_y, x1 * scale_x, y1 * scale_y)
        rect = rect & page.rect
        if rect.is_empty or rect.get_area() < _MIN_AREA:
            continue

        label = names.get(int(box.cls), "")
        if label in wanted_labels:
            out.append(rect)
            continue

        # Pseudocode has no class of its own; recognise the float by its content.
        # 伪代码没有专属类别，按内容识别该浮动块。
        try:
            text = page.get_textbox(rect)
        except Exception:  # noqa: BLE001
            continue
        if "algorithm" in kinds and _ALGORITHM_CAPTION.search(text) \
                and len(_STEP_MARKER.findall(text)) >= _MIN_STEPS:
            out.append(rect)

    # The layout model fragments some algorithm floats into boxes too small to
    # recognise, so the text-based search runs as well and the two are merged.
    # 版面模型会把部分算法块切碎成无法识别的小框，因此同时跑基于文字的搜索并合并。
    if "algorithm" in kinds:
        out.extend(_algorithm_blocks(page))
    if "contents" in kinds:
        out.extend(_toc_blocks(page))
    return _merge(out)


def _algorithm_blocks(page) -> list:
    """
    中文：按文字定位算法浮动块（从标题行往下收集步骤行）。

    Locate algorithm floats from their text.

    The layout model cannot be relied on for these. On one page it returns the
    float as a single ``table`` box; on the next it fragments the same kind of
    block into several small ``plain text`` boxes, none of which carries both the
    caption and enough steps to be recognised. Reading the lines is stable
    because the block's own shape is stable: a caption, then a run of lines that
    are numbered steps or algorithm keywords, and nothing else until the prose
    resumes.

    Args:
        page (pymupdf.Page): Source page.

    Returns:
        list[pymupdf.Rect]: One region per algorithm float found.
    """
    import pymupdf

    try:
        data = page.get_text("dict")
    except Exception:  # noqa: BLE001
        return []

    lines = []
    for block in data.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            text = "".join(span.get("text", "") for span in line.get("spans", []))
            if text.strip():
                lines.append((pymupdf.Rect(line["bbox"]), text.strip()))
    lines.sort(key=lambda item: (round(item[0].y0, 1), item[0].x0))

    out: list = []
    index = 0
    while index < len(lines):
        rect, text = lines[index]
        if not _ALGORITHM_CAPTION.match(text):
            index += 1
            continue

        # Walk downwards accepting everything until the block plainly ends.
        # Testing each line for "does this look like a step" and stopping at the
        # first failure does not work: a formula's superscript is emitted as its
        # own line ("t=1"), and a comment gloss sits in the right margin, so the
        # run is broken before a single step has been counted. What actually
        # bounds the float is the vertical gap to whatever follows it.
        # 向下逐行接收，直到该块明显结束。逐行判断「像不像步骤」、一旦不像就停是行不
        # 通的：公式的上标会自成一行（如 "t=1"），注释小字排在右边距，扫描在数到任何
        # 一个步骤之前就被打断了。真正界定这个浮动块的，是它与后文之间的垂直间隔。
        area = pymupdf.Rect(rect)
        steps = 0
        cursor = index + 1
        # Lines seen since the last step, held back rather than absorbed. A
        # formula's superscript and a margin gloss belong to the block and are
        # taken in once a further step proves the block continues; the paragraph
        # that follows the float is never proved and so is never taken in. The
        # region therefore ends at the last step, which is where the float ends.
        # 自上一个步骤行以来看到的行先扣住不并入。公式上标与边注属于本块，等到后面
        # 再出现步骤行即被一并吸收；而浮动块之后的正文段落永远等不到这个证明，因此
        # 不会被并入。于是区域止于最后一个步骤行——那正是浮动块的下边界。
        pending: list = []
        while cursor < len(lines):
            next_rect, next_text = lines[cursor]
            if next_rect.y0 - area.y1 > _BLOCK_GAP:
                break
            if len(_PROSE_WORDS.findall(next_text)) >= _MAX_PROSE_WORDS:
                break
            # The number can be a line by itself: when a step's body is a
            # display formula, the marker and the body are emitted as separate
            # text lines, so requiring something after the colon loses steps 4
            # to 6 of a typical algorithm and cuts the block short.
            # 编号可能自成一行：当步骤内容是行间公式时，编号与内容会被拆成两行输出，
            # 因此要求冒号后必须有内容会漏掉典型算法的第 4–6 步，把区块截断。
            is_step = bool(re.match(r"^\s*\d{1,3}:(?:\s|$)", next_text))
            is_keyword = bool(re.match(
                r"^\s*(Input|Output|Require|Ensure|Repeat|Until|While"
                r"|end\s+for|end\s+while|end\s+if|return)\b",
                next_text, re.I,
            ))
            if is_step or is_keyword:
                for held in pending:
                    area |= held
                pending.clear()
                area |= next_rect
                if is_step:
                    steps += 1
            else:
                pending.append(next_rect)
            cursor += 1

        if steps >= _MIN_STEPS:
            out.append(area)
            index = cursor
        else:
            index += 1

    return out


def _toc_blocks(page) -> list:
    """
    中文：定位目录区块（含点导引的条目行）。

    Locate a table of contents.

    A translated contents list is worse than an untranslated one even when the
    translation itself is fine: its page numbers point into the original
    pagination, and its entries no longer match the headings a reader searches
    for. In practice it also comes back with its line structure destroyed,
    because the entry, its leader dots and its page number are separate runs that
    the paragraph finder glues into one line of prose.

    BabelDOC does have a rule for this — it splits a paragraph at a run of twenty
    or more dots — but the run it looks for is ``.{20,}``, twenty *consecutive*
    dots. LaTeX sets leader dots with spaces between them (``. . . . .``), so the
    pattern matches nothing and the rule never fires. Hence the separate test
    here, which allows the spaces.

    The whole block is taken, not only the lines carrying dots: chapter-level
    entries ("1 Introduction 3") have no leader at all, and protecting the
    dotted lines alone would leave a contents page half in each language.

    Args:
        page (pymupdf.Page): Source page.

    Returns:
        list[pymupdf.Rect]: One region covering the contents, or empty.
    """
    import pymupdf

    try:
        data = page.get_text("dict")
    except Exception:  # noqa: BLE001
        return []

    lines = []
    for block in data.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            text = "".join(span.get("text", "") for span in line.get("spans", []))
            if text.strip():
                lines.append((pymupdf.Rect(line["bbox"]), text.strip()))
    if not lines:
        return []
    lines.sort(key=lambda item: (round(item[0].y0, 1), item[0].x0))

    dotted = [index for index, (_, text) in enumerate(lines) if _LEADER.search(text)]
    if len(dotted) < _MIN_TOC_ENTRIES:
        return []

    # Everything on the page above the footer band, not merely down to the last
    # dotted line. A contents page is dedicated to the contents, and its final
    # entries often carry no leader at all — a top-level appendix whose title is
    # short enough to sit against its page number. Stopping at the last dotted
    # entry left exactly one line outside the region, and that line came back
    # translated in the middle of an otherwise untouched contents page.
    # 取页脚带以上的全部内容，而不是只取到最后一条带点导引的条目。目录页整页都是目录，
    # 而它最后几条常常没有点导引——标题短到直接顶着页码的顶级附录条目就是如此。止于最后
    # 一条带点条目会把恰好一行留在区块之外，那一行便在整页未动的目录中被译成了中文。
    floor = page.rect.height * (1.0 - _FOOTER_BAND)
    area = None
    for rect, _ in lines:
        if rect.y0 >= floor:
            continue
        area = rect if area is None else (area | rect)
    return [area] if area is not None else []


def _merge(rects, gap: float = 4.0):
    """
    中文：合并彼此重叠或紧邻的区域，避免重复搬运。

    Merge overlapping or touching regions.

    An algorithm float is often reported both as a whole and as its inner table,
    and stamping the same area twice is wasted work.

    Args:
        rects (list[pymupdf.Rect]): Regions to merge.
        gap (float): Separation below which two regions are treated as one.

    Returns:
        list[pymupdf.Rect]: Merged regions.
    """
    merged = list(rects)
    changed = True
    while changed:
        changed = False
        for i in range(len(merged)):
            for j in range(i + 1, len(merged)):
                grown = merged[i] + (-gap, -gap, gap, gap)
                if grown.intersects(merged[j]):
                    merged[i] = merged[i] | merged[j]
                    del merged[j]
                    changed = True
                    break
            if changed:
                break
    return merged


def find_protected_regions(source_path, kinds=None) -> dict[int, list]:
    """
    中文：扫描源 PDF，找出每页需要原样保留的区块。

    Scan the source document for regions to preserve verbatim.

    Args:
        source_path (Path | str): Original PDF.
        kinds (Iterable[str] | None): Which of
            :data:`local_translator.config.PRESERVE_KINDS` to look for. ``None``
            means all of them.

    Returns:
        dict[int, list]: Page index to a list of ``pymupdf.Rect`` regions. Empty
            when the layout model is unavailable, so a missing model degrades to
            the previous behaviour rather than failing the translation.
    """
    from local_translator.config import PRESERVE_KINDS

    kinds = set(PRESERVE_KINDS if kinds is None else kinds)
    if not kinds:
        return {}
    from pathlib import Path

    import pymupdf

    try:
        from babeldoc.docvision.doclayout import DocLayoutModel

        model = DocLayoutModel.load_available()
        names = model._names
    except Exception:  # noqa: BLE001
        logger.warning("Layout model unavailable; regions will not be preserved",
                       exc_info=True)
        return {}

    regions: dict[int, list] = {}
    document = pymupdf.open(Path(source_path))
    try:
        for index in range(document.page_count):
            try:
                found = _page_regions(document[index], model, names, kinds)
            except Exception:  # noqa: BLE001
                logger.debug("Region detection failed on page %d", index + 1, exc_info=True)
                continue
            if found:
                regions[index] = found
    finally:
        document.close()

    total = sum(len(v) for v in regions.values())
    logger.info("Found %d region(s) to preserve verbatim across %d page(s)",
                total, len(regions))
    return regions


# A destination text line whose height lies this far inside a stamped region is
# taken to belong to the region and is removed with it. Below the share it is a
# neighbouring line that merely grazes the edge, and removing it would delete a
# sentence of the translation.
# 译文行有这一比例的高度落在被搬运区块内，才算属于该区块、随之删除。低于该比例
# 的只是擦到边缘的相邻行，删掉它就等于删掉译文中的一句话。
_LINE_INSIDE = 0.5


def _text_lines(page) -> list:
    """
    中文：列出一页上所有文字行的包围盒。

    List the bounding box of every text line on a page.

    Read once per page rather than per region: ``get_text`` with a ``clip``
    keeps only characters *contained* in the rectangle, so a line that merely
    overlaps the rectangle — which is exactly the line at risk — would not be
    reported at all.

    Args:
        page (pymupdf.Page): Page to read.

    Returns:
        list[tuple[float, float, float, float]]: ``(x0, y0, x1, y1)`` per line.
    """
    try:
        blocks = page.get_text("dict")["blocks"]
    except Exception:  # noqa: BLE001
        return []
    return [line["bbox"] for block in blocks if block.get("type") == 0
            for line in block.get("lines", ())]


def _redaction_rect(lines, destination):
    """
    中文：把涂抹矩形从相邻文字行上收回来，只删真正落在区块内的译文。

    Pull a redaction rectangle back off the text lines around a region.

    A stamped region is inflated by :data:`_MARGIN` and the layout model's own
    box is approximate, so the rectangle routinely overlaps the line above or
    below by a fraction of a point. PyMuPDF removes every glyph that *intersects*
    a redaction rectangle, so that fraction is enough to delete a whole line of
    the translation: on page 4 of one paper the display formula's box reached
    0.1pt into the line beneath it and took 「算法如何影响奇异值分解来理解它。设」
    with it, leaving a hole in the middle of the sentence.

    Each line that overlaps the rectangle is measured against its own height. A
    line lying mostly inside is the re-typeset version of the region and must go;
    a line lying mostly outside is a neighbour, and the rectangle is pulled back
    to clear it.

    Args:
        lines (list): Line boxes of the destination page, from
            :func:`_text_lines`.
        destination (pymupdf.Rect): Where the region is going, in that page's
            coordinates.

    Returns:
        pymupdf.Rect: The rectangle to redact. Never taller than ``destination``;
            equal to it when no neighbouring line is in the way.
    """
    import pymupdf

    top, bottom = destination.y0, destination.y1
    middle = (top + bottom) / 2.0
    for x0, y0, x1, y1 in lines:
        height = y1 - y0
        if height <= 0 or x1 <= destination.x0 or x0 >= destination.x1:
            continue
        overlap = min(bottom, y1) - max(top, y0)
        if overlap <= 0 or overlap / height >= _LINE_INSIDE:
            continue
        if (y0 + y1) / 2.0 < middle:
            top = max(top, y1)
        else:
            bottom = min(bottom, y0)

    if bottom - top < _MARGIN:
        # Pulling back both edges left nothing to redact. Fall back to the
        # region as given: covering the region matters more than sparing a line
        # that overlaps it this heavily.
        # 两边都收回后已无可涂抹的区域。退回原矩形：盖住区块比放过一条与它重叠
        # 到这种程度的行更重要。
        return destination
    return pymupdf.Rect(destination.x0, top, destination.x1, bottom)


def stamp_regions(source_path, target_path, regions: dict[int, list],
                  dual: bool = False, translate_first: bool = False) -> dict[str, int]:
    """
    中文：把受保护区块从源 PDF 覆盖搬运到译文产物上。

    Copy the protected regions from the source over the translated output.

    Args:
        source_path (Path | str): Original PDF.
        target_path (Path | str): Translated PDF, modified in place.
        regions (dict[int, list]): Page index to regions, from
            :func:`find_protected_regions`.
        dual (bool): The target is a side-by-side dual PDF.
        translate_first (bool): The dual layout puts the translation on the left.

    Returns:
        dict[str, int]: ``stamped`` regions copied and ``pages`` touched.
    """
    from pathlib import Path

    import pymupdf

    result = {"stamped": 0, "pages": 0}
    source_path, target_path = Path(source_path), Path(target_path)
    if not regions or not source_path.is_file() or not target_path.is_file():
        return result

    try:
        source = pymupdf.open(source_path)
        target = pymupdf.open(target_path)
    except Exception:  # noqa: BLE001
        logger.warning("Could not open PDFs for region stamping", exc_info=True)
        return result

    try:
        touched = False
        for index, rects in sorted(regions.items()):
            if index >= target.page_count or index >= source.page_count:
                continue
            src_page, dst_page = source[index], target[index]

            if dual:
                half = dst_page.rect.width / 2.0
                offset = 0.0 if translate_first else half
            else:
                offset = 0.0

            # Read the destination's lines once: every region on this page is
            # measured against the same set.
            # 译文页的行只读一次：本页所有区块都对着同一份行表来衡量。
            lines = _text_lines(dst_page)

            pending: list = []
            for rect in rects:
                area = (rect + (-_MARGIN, -_MARGIN, _MARGIN, _MARGIN)) & src_page.rect
                if area.is_empty:
                    continue
                destination = area + (offset, 0, offset, 0)
                try:
                    # Redact rather than merely paint over. A white rectangle
                    # hides the re-typeset version but leaves its text objects in
                    # the content stream, so the page still yields the discarded
                    # translation to copy, search and screen readers — the
                    # contents page came back reading as English while still
                    # holding 262 invisible Chinese characters underneath.
                    # Redaction removes the text itself, and the replayed source
                    # is then the only thing there.
                    # 用涂抹（redaction）而不只是覆盖。白色矩形只是遮住重排版本，其文字
                    # 对象仍留在内容流里，页面依然会把被弃用的译文交给复制、搜索与读屏
                    # 工具——目录页看起来已是英文，底下却还压着 262 个不可见的汉字。
                    # 涂抹会真正删掉这些文字，回放的源区域于是成为该处唯一的内容。
                    dst_page.add_redact_annot(
                        _redaction_rect(lines, destination), fill=(1, 1, 1),
                    )
                    pending.append((destination, area, index))
                    result["stamped"] += 1
                except Exception:  # noqa: BLE001
                    logger.debug("Could not stamp region %s on page %d", area, index + 1,
                                 exc_info=True)
            # Redactions are applied once per page, because applying them drops
            # every annotation on the page — including content already stamped.
            # The source regions therefore go on afterwards.
            # 涂抹按页统一应用：应用涂抹会清掉该页所有注释，包括已经搬运上去的内容，
            # 因此源区域要在此之后再放上去。
            if pending:
                try:
                    # text=REMOVE deletes the covered glyphs; images and line
                    # art are left alone, since a figure or a table rule that
                    # merely overlaps the region must survive.
                    # text=REMOVE 删除被覆盖的字形；图像与线条图保持不动，因为与该区域
                    # 仅有重叠的插图或表格线必须保留。
                    dst_page.apply_redactions(
                        images=pymupdf.PDF_REDACT_IMAGE_NONE,
                        graphics=pymupdf.PDF_REDACT_LINE_ART_NONE,
                        text=pymupdf.PDF_REDACT_TEXT_REMOVE,
                    )
                except Exception:  # noqa: BLE001
                    logger.debug("Could not apply redactions on page %d", index + 1,
                                 exc_info=True)
                for destination, area, page_index in pending:
                    try:
                        dst_page.show_pdf_page(destination, source, page_index,
                                               clip=area, overlay=True)
                    except Exception:  # noqa: BLE001
                        logger.debug("Could not replay region %s on page %d", area,
                                     page_index + 1, exc_info=True)

            if rects:
                result["pages"] += 1
                touched = True

        if touched:
            temp = target_path.with_suffix(target_path.suffix + ".regionfix")
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
            logger.info("Stamped %d protected region(s) across %d page(s) in %s",
                        result["stamped"], result["pages"], target_path.name)
    finally:
        for document in (source, target):
            if not document.is_closed:
                document.close()

    return result


def stamp_all(source_path, outputs: dict, translate_first: bool = False,
              kinds=None) -> dict[str, int]:
    """
    中文：对所有产物执行受保护区块的原样搬运。

    Stamp the protected regions onto every produced PDF.

    Args:
        source_path (Path | str): Original PDF.
        outputs (dict[str, str | None]): Produced paths keyed by kind.
        translate_first (bool): The dual layout puts the translation on the left.
        kinds (Iterable[str] | None): Kinds to preserve; ``None`` means all.

    Returns:
        dict[str, int]: Combined counts across the repaired files.
    """
    totals = {"stamped": 0, "pages": 0}
    regions = find_protected_regions(source_path, kinds)
    if not regions:
        return totals

    mono = outputs.get("mono_path")
    if mono:
        for key, value in stamp_regions(source_path, mono, regions).items():
            totals[key] += value

    dual = outputs.get("dual_path")
    if dual:
        for key, value in stamp_regions(source_path, dual, regions, dual=True,
                                        translate_first=translate_first).items():
            totals[key] += value

    return totals
