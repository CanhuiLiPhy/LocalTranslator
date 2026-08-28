"""
中文：图区修复——把 BabelDOC 重绘时整块丢失的插图按原位搬回（含双语版译文半边）。

Restore figure regions that page regeneration blanks out.

BabelDOC rebuilds each translated page from its intermediate representation and
installs the result with ``page.set_contents(...)``, replacing the original
content stream outright. Whatever the IR does not model is gone. For composite
figures that means everything at once — bitmaps, vector artwork and the labels
inside the figure — leaving a blank rectangle where the illustration was.

**Why this works on regions rather than on individual images.** The obvious
repair is to extract each bitmap and re-place it at the rectangle the source
reports. That was tried and is unsound, for two independent reasons:

* Placement rectangles reported for images nested inside Form XObjects are
  unreliable. This paper's page 6 reports a thumbnail at ``(-20.7, 0.9,
  328.8, 132.0)`` — 349 points wide, starting off the page.
* Those numbers are not even wrong: the image really is placed that large and
  then *clipped* to a small window. Re-placing it without the clip paints a
  full-width picture across the body text. Clipping lives in the content
  stream, not in the image object, so no amount of care with rectangles
  recovers it.

Re-drawing pixels also cannot restore the vector arrows and text labels that
made up most of the figure, so even a perfect bitmap repair would leave the
illustration broken.

Copying the *region* sidesteps all of it. ``show_pdf_page`` replays the source
page's own content stream into a target rectangle, so clips, vector paths,
labels and bitmaps all arrive exactly as the author drew them.

**Finding the regions.** The regions are identified by appearance rather than
by parsing structure, because appearance is what actually went missing: render
both pages, and look for areas that carry ink in the source and are blank in
the output. Translated text does not qualify — it may have moved or reflowed,
but it is still ink on the page. Only genuinely dropped content leaves a region
that is full in one and empty in the other.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Resolution for the comparison render, in pixels per point. One is enough to
# tell a figure from blank paper and keeps a 19-page comparison well under a
# second; the copy itself is vector, so this never limits output quality.
# 比较渲染的分辨率（像素/点）。1 已足以区分插图与空白，19 页比较耗时不到一秒；
# 搬运本身是矢量的，所以这个精度不影响成品质量。
_ZOOM = 1.0

# Grid cell size in points. Large enough that a stray antialiased pixel cannot
# create a region, small enough to trace a figure's outline closely.
# 网格单元边长（点）。足够大以免单个抗锯齿像素形成区域，又足够小以贴合插图轮廓。
_CELL = 6

# A cell counts as inked when this fraction of its pixels are not white.
# 单元内非白像素达到该比例即视为有内容。
_INK_PRESENT = 0.02

# A cell counts as blank when its ink falls below this. The gap between the two
# thresholds is deliberate: cells in between are ambiguous and are left alone.
# 低于该比例视为空白。两个阈值之间留有间隔：处于其间的单元含义不明，不予处理。
_INK_ABSENT = 0.004

# Regions smaller than this in square points are bullets or artefacts.
# 小于该面积（平方点）的区域是项目符号或噪点。
_MIN_REGION_AREA = 900.0

# A region must reach this along its longer side, in points. Below it there is
# nothing but specks.
# 区域较长的一边需达到该长度（点）。低于此值的只是噪点。
_MIN_REGION_SIDE = 20.0

# A region thin in one direction is a rule rather than a figure, and is kept
# when it runs at least this far. Header and footer dividers measure about
# 474 x 6: an earlier version rejected everything thin on the assumption that
# the rebuilt page drew its own rules, which turned out to be false — the
# output has none — so the assumption cost every divider in the document.
# 某一方向很窄的区域是分割线而非插图，只要长度达到该值就予以保留。页眉页脚的
# 分割线实测约 474×6：早先的版本以「重建后的页面会自己画线」为由拒绝了所有细长
# 区域，而该假设是错的——输出里一条都没有——于是全文的分割线都因此丢失。
_MIN_RULE_LENGTH = 60.0

# A drawing thinner than this, and long, is a rule rather than a filled shape.
# 比该厚度更薄且够长的图形是分割线，而不是填充块。
_RULE_THICKNESS = 6.0

# Along a drawn rule, this share of the source's cells must carry ink for the
# rule to be considered visible in the original at all.
# 沿一条画出的线，源文中须有这一比例的单元有墨迹，才认为该线在原文里确实可见。
_RULE_ZOOM = 4.0

# Half-height of the strip sampled around a rule, in points.
# 围绕分割线采样的细带半高（点）。
_RULE_STRIP = 1.2

# The source strip must be at least this inked for the rule to be visible at all.
# 源文细带至少要有这么多墨迹，该线才算在原文中可见。
_RULE_PRESENT = 0.55

# The rule counts as damaged when the output keeps less than this share of the
# source's ink along it. Set high on purpose: a rule is not only lost outright,
# it is also broken. A running head that reflows sits across the header rule and
# leaves it 54% covered on every page of one paper — plainly damaged to a reader,
# yet nowhere near absent. Redrawing a partially present rule is harmless, since
# what is drawn is the line the source itself draws, in its own colour and width.
# 输出沿该线保留的墨迹低于源文的这一比例时，判定该线受损。阈值刻意取高：分割线不只会
# 整条消失，也会被打断。重排后的页眉文字会压在页眉线上，某篇论文每页都只剩 54%——读者
# 一眼就看得出坏了，却远谈不上「消失」。补画一条尚存部分的线没有风险，因为画的就是源文
# 自己那条线，颜色与线宽都取自它。
_RULE_FADED = 0.85


# Share of a source line's height that has to fall inside a candidate band for
# the line to be considered part of the region rather than clipped by it. A
# figure's own labels lie wholly inside; a line the band merely slices does not.
# 源文行有这一比例的高度落在候选带内，才算属于该区块而不是被它切到。插图自带的
# 标注整行都在里面，被带子切到的行则不是。
_LINE_WHOLE = 0.5

# The shortest band worth copying once translated text has been trimmed away,
# in points. A sliver thinner than this holds no figure and no rule that
# :func:`missing_rules` would not redraw more faithfully anyway.
# 裁掉译文文字后仍值得搬运的最小带高（点）。比这更窄的碎带里既没有插图，也没有
# missing_rules 无法更忠实地补画的分割线。
_MIN_FREE_HEIGHT = 3.0

# A region is only copied when this share of its cells are blank in the output,
# which distinguishes a dropped figure from text that merely moved.
# 仅当输出中该比例的单元为空白时才搬运，用以区分「插图被丢弃」与「文字位移」。
_MIN_BLANK_SHARE = 0.90

# At least this share of a region's ink must fall outside the source page's text
# boxes for the region to count as artwork. A paragraph is ink that is entirely
# accounted for by its own characters; a figure has frames, arrows, plots and
# photographs that no character box covers. Labels inside a figure are text, so
# the test asks for a minority of non-text ink rather than a majority.
# 区域内至少这一比例的墨迹须落在源页文字框之外，才算作插图。一段正文的墨迹完全
# 由其字符解释，而插图含有字符框覆盖不到的线框、箭头、图表与照片。图内标注本身
# 是文字，因此该判据只要求少数非文字墨迹，而非多数。
_MIN_NONTEXT_INK = 0.18

# Largest share of a region that extractable text may cover before the region is
# refused as a transplant, whatever else is true of it.
#
# The ink-versus-text-cell test above is arithmetic on a six-point grid, and on
# a region one line tall the quantisation is most of the answer: descenders,
# italics and half-covered edge cells all count as non-text, so a line of pure
# prose measures 0.22 to 0.44 non-text and sails past a 0.18 floor. Four such
# lines were copied back into arXiv:2608.18004 — 「samples to the source,」,
# 「so the conditional path is Gaussian, pt(」 — where they read exactly like a
# translation failure: a line of English among the Chinese.
#
# Asking the text extractor directly is not arithmetic and does not care about
# the grid. Measured on that paper, the four wrongly copied lines cover 0.28 to
# 0.49 of their region; three real figures cover 0.01 to 0.03. Fifteen percent
# sits an order of magnitude away from both.
# 一块区域中可提取文字最多可以覆盖多大比例，超过就拒绝搬运，不论其他条件如何。
#
# 上面那道「墨迹 vs 文字单元」的检查是在六点格上做算术，而对只有一行高的区域，量化误差就
# 是答案的大头：下伸笔画、斜体、被切了一半的边缘格全都算作非文字，于是一行纯正文能量出
# 0.22 到 0.44 的非文字占比，轻松越过 0.18 的下限。arXiv:2608.18004 里就有四行这样被搬了
# 回去——「samples to the source,」、「so the conditional path is Gaussian, pt(」——它们
# 在产物上读起来与翻译失败一模一样：中文里夹着一行英文。
#
# 直接问文字提取器不是算术，也不受格点影响。在那篇论文上量得：被误搬的四行覆盖率
# 0.28–0.49，三张真插图覆盖率 0.01–0.03。取百分之十五，与两边都隔着一个数量级。
_MAX_TEXT_SHARE = 0.15


def _ink_grid(page, clip=None):
    """
    中文：把页面渲染成粗网格上的墨迹比例矩阵。

    Render a page into a coarse grid of ink fractions.

    Args:
        page (pymupdf.Page): Page to render.
        clip (pymupdf.Rect | None): Restrict rendering to this rectangle; the
            grid is then relative to its top-left corner.

    Returns:
        numpy.ndarray: 2-D array of ink fractions per grid cell, values in
            ``[0, 1]``, indexed ``[row, column]``.
    """
    import numpy as np
    import pymupdf

    pix = page.get_pixmap(matrix=pymupdf.Matrix(_ZOOM, _ZOOM), clip=clip,
                          colorspace=pymupdf.csGRAY, alpha=False)
    if pix.width == 0 or pix.height == 0:
        return np.zeros((0, 0), dtype=float)

    buffer = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)
    ink = buffer < 250

    step = max(1, int(round(_CELL * _ZOOM)))
    rows = -(-pix.height // step)
    cols = -(-pix.width // step)
    # Pad to a whole number of cells so the reshape below is exact.
    # 补齐到整数个单元，使下面的 reshape 精确成立。
    padded = np.zeros((rows * step, cols * step), dtype=bool)
    padded[: pix.height, : pix.width] = ink
    return padded.reshape(rows, step, cols, step).mean(axis=(1, 3))


def _regions(mask) -> list[tuple[int, int, int, int]]:
    """
    中文：把布尔网格中相连的单元聚成矩形区域。

    Group connected cells of a boolean grid into bounding rectangles.

    Connectivity includes diagonals and is computed iteratively rather than
    recursively, so a figure spanning a whole page cannot exhaust the stack.

    Args:
        mask (numpy.ndarray): 2-D boolean array; ``True`` marks a cell of
            interest.

    Returns:
        list[tuple[int, int, int, int]]: One ``(row0, col0, row1, col1)`` per
            region, with the end indices exclusive.
    """
    import numpy as np

    seen = np.zeros_like(mask, dtype=bool)
    out: list[tuple[int, int, int, int]] = []
    rows, cols = mask.shape

    for start_r in range(rows):
        for start_c in range(cols):
            if not mask[start_r, start_c] or seen[start_r, start_c]:
                continue
            stack = [(start_r, start_c)]
            seen[start_r, start_c] = True
            r0 = r1 = start_r
            c0 = c1 = start_c
            while stack:
                r, c = stack.pop()
                r0, r1 = min(r0, r), max(r1, r)
                c0, c1 = min(c0, c), max(c1, c)
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols \
                                and mask[nr, nc] and not seen[nr, nc]:
                            seen[nr, nc] = True
                            stack.append((nr, nc))
            out.append((r0, c0, r1 + 1, c1 + 1))
    return out


def _text_share(page, rect) -> float:
    """
    中文：区域里能提取出的文字覆盖了多大面积。

    Measure how much of a region extractable text covers.

    Args:
        page (pymupdf.Page): Source page.
        rect (pymupdf.Rect): Region to measure, in page coordinates.

    Returns:
        float: Covered area over region area, in ``[0, 1]``. Overlapping word
            boxes are counted twice, which can only push the answer up and
            therefore only makes the guard stricter — never laxer.
    """
    import pymupdf

    area = rect.get_area()
    if area <= 0:
        return 0.0
    try:
        words = page.get_text("words", clip=rect)
    except Exception:  # noqa: BLE001
        return 0.0
    covered = 0.0
    for x0, y0, x1, y1, *_ in words:
        piece = pymupdf.Rect(x0, y0, x1, y1) & rect
        if not piece.is_empty:
            covered += piece.get_area()
    return min(1.0, covered / area)


def _text_cells(page, shape):
    """
    中文：标出被源页文字包围盒覆盖的网格单元。

    Mark grid cells covered by the source page's text boxes.

    Word rectangles come from text extraction, which is exact — unlike the
    placement rectangles reported for images nested in form XObjects, which
    cannot be trusted for this document.

    Args:
        page (pymupdf.Page): Source page.
        shape (tuple[int, int]): Grid shape as ``(rows, columns)``.

    Returns:
        numpy.ndarray: 2-D boolean array; ``True`` where a word sits.
    """
    import numpy as np

    mask = np.zeros(shape, dtype=bool)
    rows, cols = shape
    try:
        words = page.get_text("words")
    except Exception:  # noqa: BLE001
        return mask

    for x0, y0, x1, y1, *_ in words:
        c0 = max(0, int(x0 // _CELL))
        r0 = max(0, int(y0 // _CELL))
        c1 = min(cols, int(-(-x1 // _CELL)))
        r1 = min(rows, int(-(-y1 // _CELL)))
        if r1 > r0 and c1 > c0:
            mask[r0:r1, c0:c1] = True
    return mask


def _drawn_rules(page):
    """
    中文：读出源页上实际画出来的细长线段（分割线、脚注线、表格线）。

    Read the thin, long lines the source page actually draws.

    The distinction matters because a region that is blank in the output and
    inked in the source can be either a dropped rule or a line of text that
    merely moved. Both look identical as a one-cell sliver, and 62 of 64 such
    slivers in one paper were shifted text. Asking the source's own content
    stream settles it: a rule is a drawing, a line of prose is not.

    Args:
        page (pymupdf.Page): Source page.

    Returns:
        list[tuple]: One ``(rect, colour, width)`` per rule-like drawing, with
            the colour as an RGB triple in ``[0, 1]``.
    """
    try:
        drawings = page.get_drawings()
    except Exception:  # noqa: BLE001
        return []
    rules = []
    for item in drawings:
        rect = item["rect"]
        if rect.width < _MIN_RULE_LENGTH or rect.height >= _RULE_THICKNESS:
            continue
        colour = item.get("color") or item.get("fill") or (0, 0, 0)
        width = item.get("width") or 0.5
        rules.append((rect, tuple(colour), float(width) or 0.5))
    return rules


def _merge(rects, gap: float = 8.0):
    """
    中文：把彼此靠近的矩形合并，使一幅插图的各碎片成为一块。

    Merge rectangles that lie within ``gap`` points of each other.

    A figure is usually several disconnected pieces — panels, arrows, a caption
    strip — and each arrives as its own component. Merging them into one
    transplant keeps the copy count low and avoids seams between adjacent
    pieces.

    Args:
        rects (list[pymupdf.Rect]): Rectangles to merge.
        gap (float): Separation below which two rectangles are treated as one.

    Returns:
        list[pymupdf.Rect]: Merged rectangles.
    """
    merged = list(rects)
    changed = True
    while changed:
        changed = False
        for i in range(len(merged)):
            for j in range(i + 1, len(merged)):
                a = merged[i] + (-gap, -gap, gap, gap)
                if a.intersects(merged[j]):
                    merged[i] = merged[i] | merged[j]
                    del merged[j]
                    changed = True
                    break
            if changed:
                break
    return merged


def _peel(blank, bounds):
    """
    中文：把区域四边不够空白的边逐条剥掉，只留下真正丢失的核心。

    Shrink a region to the part that is genuinely blank in the output.

    A connected group of candidate cells does not always coincide with what was
    lost. Diagonal connectivity lets a single noisy cell in a neighbouring line
    of text attach a whole extra row to the region, and judging the enlarged
    box as a whole then rejects a real loss on the strength of its uninvited
    neighbour. That is not hypothetical: on a side-by-side dual page the right
    half is rasterised from an origin 0.28 points off the pixel grid, so a
    sparse row of running-head text renders a few hundredths lighter than the
    same row in the mono file. Enough of its cells fall below the blank
    threshold to join the header rule below them, and the merged box scores
    0.80 blank against a 0.90 requirement — which is why every page but the
    first lost its header divider in the dual output while the mono kept it.

    Peeling removes the borrowed edge instead of the finding. Only edges below
    the blank requirement are removed, so a region that is already coherent
    comes back untouched; a region whose non-blank mass sits in its interior
    cannot be rescued this way and is rejected.

    Args:
        blank (numpy.ndarray): 2-D boolean grid, ``True`` where the output cell
            carries no ink.
        bounds (tuple[int, int, int, int]): ``(row0, col0, row1, col1)`` with
            exclusive end indices.

    Returns:
        tuple[int, int, int, int] | None: Tightened bounds, or ``None`` when no
            sufficiently blank core remains.
    """
    r0, c0, r1, c1 = bounds
    while r1 > r0 and c1 > c0:
        window = blank[r0:r1, c0:c1]
        if window.mean() >= _MIN_BLANK_SHARE:
            return r0, c0, r1, c1
        share, edge = min(
            (float(window[0, :].mean()), "top"),
            (float(window[-1, :].mean()), "bottom"),
            (float(window[:, 0].mean()), "left"),
            (float(window[:, -1].mean()), "right"),
        )
        # Every edge already meets the requirement, so the ink sits inside the
        # region and peeling cannot reach it.
        # 四条边都已达标，说明墨迹在区域内部，剥边够不着。
        if share >= _MIN_BLANK_SHARE:
            return None
        if edge == "top":
            r0 += 1
        elif edge == "bottom":
            r1 -= 1
        elif edge == "left":
            c0 += 1
        else:
            c1 -= 1
    return None


def _occupied_bands(dst_page, clip):
    """
    中文：取译文页上已有文字所占的横带，换算回源页坐标。

    Report the horizontal bands the translated page already fills with text.

    Args:
        dst_page (pymupdf.Page): Translated page.
        clip (pymupdf.Rect): The half of a dual page under consideration; its
            left edge is the offset between destination and source coordinates.

    Returns:
        list[tuple[float, float, float, float]]: ``(y0, y1, x0, x1)`` per word,
            in source page coordinates.
    """
    try:
        words = dst_page.get_text("words", clip=clip)
    except Exception:  # noqa: BLE001
        return []
    shift = clip.x0
    return [(word[1], word[3], word[0] - shift, word[2] - shift) for word in words]


def _source_lines(page):
    """
    中文：列出源页上所有文字行的包围盒。

    List the bounding box of every text line on the source page.

    Args:
        page (pymupdf.Page): Source page.

    Returns:
        list[tuple[float, float, float, float]]: ``(y0, y1, x0, x1)`` per line,
            ordered as :func:`_occupied_bands` returns its bands.
    """
    try:
        blocks = page.get_text("dict")["blocks"]
    except Exception:  # noqa: BLE001
        return []
    return [(line["bbox"][1], line["bbox"][3], line["bbox"][0], line["bbox"][2])
            for block in blocks if block.get("type") == 0
            for line in block.get("lines", ())]


def _clear_of_text(rect, occupied, sliced=()):
    """
    中文：把矩形在垂直方向裁到既不压译文、也不切到源文文字行的那一段。

    Trim a rectangle vertically off translated text and off sliced source lines.

    A region is only a candidate because the output is blank there, but "blank"
    is measured over the whole width: a band that is empty across most of the
    page still passes while its left end sits under a translated heading. Copying
    such a band replays whatever the source drew underneath — on the title page
    of one paper, the top 2.8pt of the title's ascenders, which reads as stray
    strokes above the translated title.

    Two different things therefore block the band, and both are measured here:

    * **Translated text under it.** Any of it is disqualifying: the region would
      be stamped on top of the translation.
    * **A source line the band only clips.** A band whose edge falls inside a
      line of the source replays a horizontal slice of that line's glyphs. The
      translation of that line lives elsewhere on the page, so the slice is a
      ghost — on page 6 of one paper the bottom 3pt of a callout's last English
      line showed through beneath its Chinese replacement. A line lying *mostly*
      inside is different: that is a figure's own label, and it must come with
      the figure, so only lines the band clips are treated as obstacles.

    For a real figure the output is empty throughout and the source lines inside
    it are whole, so nothing blocks and the rectangle is returned unchanged. This
    only ever narrows.

    Args:
        rect (pymupdf.Rect): Candidate region in source page coordinates.
        occupied (list[tuple[float, float, float, float]]): Translated-text
            bands from :func:`_occupied_bands`.
        sliced (list[tuple[float, float, float, float]]): Source text lines from
            :func:`_source_lines`.

    Returns:
        pymupdf.Rect | None: The tallest sub-band free of both, or ``None`` when
            no usable part remains.
    """
    import pymupdf

    blocked = [
        (y0, y1) for y0, y1, x0, x1 in occupied
        if x1 > rect.x0 and x0 < rect.x1 and y1 > rect.y0 and y0 < rect.y1
    ]
    for y0, y1, x0, x1 in sliced:
        height = y1 - y0
        if height <= 0 or x1 <= rect.x0 or x0 >= rect.x1:
            continue
        overlap = min(rect.y1, y1) - max(rect.y0, y0)
        if 0 < overlap and overlap / height < _LINE_WHOLE:
            blocked.append((y0, y1))
    blocked.sort()
    if not blocked:
        return rect

    free: list[tuple[float, float]] = []
    cursor = rect.y0
    for y0, y1 in blocked:
        if y0 > cursor:
            free.append((cursor, min(y0, rect.y1)))
        cursor = max(cursor, y1)
        if cursor >= rect.y1:
            break
    if cursor < rect.y1:
        free.append((cursor, rect.y1))
    if not free:
        return None

    top, bottom = max(free, key=lambda span: span[1] - span[0])
    if bottom - top < _MIN_FREE_HEIGHT:
        return None
    return pymupdf.Rect(rect.x0, top, rect.x1, bottom)


def _lost_regions(src_page, dst_page, clip):
    """
    中文：找出源页有内容而译文页空白的区域。

    Locate areas that carry content in the source and none in the output.

    Each connected group of candidate cells is validated on its own before any
    merging. Validating after merging does not work: translated text also
    produces scattered candidate cells, because the Chinese lines do not fall
    exactly where the English ones did, and merging early joins those specks to
    the genuine figures until a single region covers the page and fails the
    test as a whole.

    Args:
        src_page (pymupdf.Page): Source page.
        dst_page (pymupdf.Page): Translated page.
        clip (pymupdf.Rect): Region of the translated page to compare against
            the whole source page, used to address one half of a dual page.

    Returns:
        list[pymupdf.Rect]: Rectangles in source page coordinates.
    """
    import pymupdf

    src = _ink_grid(src_page)
    dst = _ink_grid(dst_page, clip=clip)
    if src.size == 0 or dst.size == 0:
        return []

    rows = min(src.shape[0], dst.shape[0])
    cols = min(src.shape[1], dst.shape[1])
    src, dst = src[:rows, :cols], dst[:rows, :cols]

    candidate = (src >= _INK_PRESENT) & (dst <= _INK_ABSENT)
    if not candidate.any():
        return []

    blank = dst <= _INK_ABSENT
    inked = src >= _INK_PRESENT
    text = _text_cells(src_page, src.shape)
    accepted: list[pymupdf.Rect] = []

    for bounds in _regions(candidate):
        tightened = _peel(blank, bounds)
        if tightened is None:
            continue
        r0, c0, r1, c1 = tightened
        rect = pymupdf.Rect(c0 * _CELL, r0 * _CELL, c1 * _CELL, r1 * _CELL) & src_page.rect
        if rect.is_empty:
            continue
        # Rules are exempt from the area floor: a divider is a thin sliver whose
        # area says nothing about whether the reader will miss it.
        # 分割线不受面积下限约束：它本就是细长的一条，面积说明不了读者会否察觉缺失。
        is_rule = min(rect.width, rect.height) < _MIN_REGION_SIDE
        if not is_rule and rect.get_area() < _MIN_REGION_AREA:
            continue
        long_side, short_side = max(rect.width, rect.height), min(rect.width, rect.height)
        if long_side < _MIN_REGION_SIDE:
            continue
        if short_side < _MIN_REGION_SIDE and long_side < _MIN_RULE_LENGTH:
            continue
        window_ink = inked[r0:r1, c0:c1]
        total_ink = int(window_ink.sum())
        if not total_ink:
            continue
        nontext = int((window_ink & ~text[r0:r1, c0:c1]).sum()) / total_ink
        if nontext < _MIN_NONTEXT_INK:
            continue
        # What the source draws here is text, so the translation already carries
        # it somewhere else on the page. Copying it back would print it twice,
        # once in each language. See :data:`_MAX_TEXT_SHARE`.
        # 源文在这里画的是文字，那么译文已经把它安置在页面的别处了。搬回去等于把它印两遍、
        # 每种语言一遍。理由见 _MAX_TEXT_SHARE。
        if _text_share(src_page, rect) > _MAX_TEXT_SHARE:
            continue
        accepted.append(rect)

    if not accepted:
        return []

    occupied = _occupied_bands(dst_page, clip)
    sliced = _source_lines(src_page)
    final: list[pymupdf.Rect] = []
    for rect in _merge(accepted):
        rect = rect & src_page.rect
        if rect.is_empty:
            continue
        # Never copy over ground the translation already occupies.
        # 译文已占据的地方，绝不搬运覆盖。
        rect = _clear_of_text(rect, occupied, sliced)
        if rect is None or rect.is_empty:
            continue
        # Re-check the merged rectangle: joining neighbours can pull in cells
        # that hold translated text, and copying over those would double it.
        # 复核合并后的矩形：连接相邻块可能纳入含译文的单元，覆盖上去会造成重影。
        c0, r0 = int(rect.x0 // _CELL), int(rect.y0 // _CELL)
        c1, r1 = min(cols, int(-(-rect.x1 // _CELL))), min(rows, int(-(-rect.y1 // _CELL)))
        if r1 <= r0 or c1 <= c0:
            continue
        if blank[r0:r1, c0:c1].mean() < _MIN_BLANK_SHARE:
            continue
        final.append(rect)

    return final


def _strip_ink(page, rect, offset: float = 0.0) -> float:
    """
    中文：测量一条细长带内的墨迹比例（按行取最大，以免被空白段稀释）。

    Measure ink along a thin horizontal strip.

    The maximum over the strip's pixel rows is taken rather than the mean,
    because a rule occupies one or two rows of a strip that is deliberately
    taller than the line, and averaging over the blank rows around it would put
    a present rule and an absent one at nearly the same score.

    Args:
        page (pymupdf.Page): Page to sample.
        rect (pymupdf.Rect): Strip in page coordinates.
        offset (float): Horizontal shift applied to the strip, for addressing
            one half of a dual page.

    Returns:
        float: Fraction of dark pixels in the darkest row, in ``[0, 1]``.
    """
    import numpy as np
    import pymupdf

    clip = (rect + (offset, 0, offset, 0)) & page.rect
    if clip.is_empty or clip.width < 1 or clip.height < 0.5:
        return 0.0
    try:
        pix = page.get_pixmap(matrix=pymupdf.Matrix(_RULE_ZOOM, _RULE_ZOOM), clip=clip,
                              colorspace=pymupdf.csGRAY, alpha=False)
    except Exception:  # noqa: BLE001
        return 0.0
    if pix.width == 0 or pix.height == 0:
        return 0.0
    buffer = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)
    return float((buffer < 250).mean(axis=1).max())


def missing_rules(src_page, dst_page, offset: float = 0.0) -> list:
    """
    中文：列出源页画的所有分割线，用于在译文页上无条件重画。

    List every rule the source draws, to be redrawn unconditionally.

    Every rule is redrawn, whether or not the output still has it. Deciding
    which ones are damaged does not work, and the reason is structural rather
    than a matter of picking a better threshold. A rule is not lost all at once:
    a reflowed running head sits across the header rule and leaves 87% of it, a
    translated title covers the left tenth of another. Both are plainly broken
    to a reader while scoring far above any threshold that does not also fire on
    ordinary antialiasing differences, and the numbers move with every document.

    PDFMathTranslate does not have this problem because it never re-emits the
    graphics: it replaces the text operators in the original content stream and
    leaves every drawing operator untouched. BabelDOC rebuilds the page instead,
    so the closest equivalent available here is to put all of the source's rules
    back afterwards. Redrawing one that survived is invisible — same coordinates,
    same colour, same width — so the cost of being unconditional is nil and the
    threshold disappears with it.

    A rule the source draws in white or too faintly to see is skipped, since
    drawing it would be the one case where a redraw is not a no-op.

    Args:
        src_page (pymupdf.Page): Source page.
        dst_page (pymupdf.Page): Translated page, unused; kept in the signature
            because the caller pairs the two and a future check may need it.
        offset (float): Horizontal shift for the translated half of a dual page,
            unused for the same reason.

    Returns:
        list[tuple]: One ``(rect, colour, width)`` per visible rule.
    """
    import pymupdf

    out = []
    for rect, colour, width in _drawn_rules(src_page):
        strip = pymupdf.Rect(rect.x0, rect.y0 - _RULE_STRIP, rect.x1, rect.y1 + _RULE_STRIP)
        if _strip_ink(src_page, strip) < _RULE_PRESENT:
            continue
        out.append((rect, colour, width))
    return out


def restore_images(source_path: Path | str, target_path: Path | str,
                   dual: bool = False, translate_first: bool = False) -> dict[str, int]:
    """
    中文：把译文页上整块丢失的插图区域从源 PDF 搬回原位。

    Copy dropped figure regions from the source into the translated PDF.

    Args:
        source_path (Path | str): Original PDF.
        target_path (Path | str): Translated PDF, modified in place.
        dual (bool): The target is a side-by-side dual PDF, whose pages are
            twice as wide and hold the original beside the translation.
        translate_first (bool): The translation occupies the left half.

    Returns:
        dict[str, int]: ``restored`` regions copied, ``rules`` redrawn and
            ``pages`` touched.
    """
    import pymupdf

    result = {"restored": 0, "pages": 0, "rules": 0}
    source_path, target_path = Path(source_path), Path(target_path)
    if not source_path.is_file() or not target_path.is_file():
        return result

    try:
        source = pymupdf.open(source_path)
        target = pymupdf.open(target_path)
    except Exception:  # noqa: BLE001
        logger.warning("Could not open PDFs for figure restoration", exc_info=True)
        return result

    try:
        touched = False
        for index in range(min(source.page_count, target.page_count)):
            src_page, dst_page = source[index], target[index]

            if dual:
                half = dst_page.rect.width / 2.0
                offset = 0.0 if translate_first else half
                clip = pymupdf.Rect(offset, 0, offset + half, dst_page.rect.height)
            else:
                offset = 0.0
                clip = dst_page.rect

            regions = _lost_regions(src_page, dst_page, clip)
            for rect in regions:
                target_rect = rect + (offset, 0, offset, 0)
                try:
                    # The region is blank in the output, so drawing on top of it
                    # covers nothing; overlaying guarantees the transplant is
                    # visible even if the page paints its own background.
                    # 该区域在输出中是空白，覆盖上去不会遮住任何内容；用前景绘制
                    # 可确保即使页面自绘背景，搬运的内容也可见。
                    dst_page.show_pdf_page(target_rect, source, index, clip=rect,
                                           overlay=True)
                    result["restored"] += 1
                    # Say what was transplanted and from where. A region copied
                    # back onto a page it should not have been is invisible in
                    # the log otherwise, and looks in the output exactly like a
                    # translation failure — a line of English among the Chinese.
                    # 把搬了什么、从哪搬的写出来。否则一次不该发生的搬运在日志里毫无痕迹，
                    # 而它在产物上看起来与翻译失败一模一样：中文里夹着一行英文。
                    logger.info("Restored region p%d (%.0f,%.0f)-(%.0f,%.0f): %r",
                                index + 1, rect.x0, rect.y0, rect.x1, rect.y1,
                                src_page.get_textbox(rect).replace("\n", " ")[:40])
                except Exception:  # noqa: BLE001
                    logger.debug("Could not copy region %s on page %d", rect, index + 1,
                                 exc_info=True)
            # Rules are redrawn rather than copied; see :func:`missing_rules`.
            # 分割线用重画而非搬运，理由见 missing_rules。
            for rect, colour, width in missing_rules(src_page, dst_page, offset):
                try:
                    y = (rect.y0 + rect.y1) / 2.0
                    dst_page.draw_line(
                        pymupdf.Point(rect.x0 + offset, y),
                        pymupdf.Point(rect.x1 + offset, y),
                        color=colour, width=max(width, 0.4), overlay=True,
                    )
                    result["rules"] += 1
                except Exception:  # noqa: BLE001
                    logger.debug("Could not redraw rule on page %d", index + 1,
                                 exc_info=True)

            if regions or result["rules"]:
                result["pages"] += 1
                touched = True

        if touched:
            temp = target_path.with_suffix(target_path.suffix + ".imgfix")
            # garbage=4, not 3. Level 3 merges duplicate *objects*; only level 4
            # merges duplicate *streams*, and streams are where the weight is:
            # these stages stamp original page content back over the rebuilt
            # page, so every figure ends up in the file twice. Measured on
            # arXiv:2510.04704 — 137 images became 274, and the output grew from
            # 14.3 MB to 27.5 MB. Level 4 brings it back to 14.9 MB and costs
            # 0.1 s.
            # 用 garbage=4 而不是 3。级别 3 合并重复的「对象」，只有级别 4 才合并重复的
            # 「流」，而体积正是在流上：这些阶段会把原页内容重新盖回重排后的页面，于是每张
            # 图在文件里存了两份。在 arXiv:2510.04704 上实测——137 张图变成 274 张，产物
            # 从 14.3 MB 涨到 27.5 MB。级别 4 把它压回 14.9 MB，代价是 0.1 秒。
            target.save(temp, garbage=4, deflate=True)
            # Close before replacing, not after.
            #
            # POSIX lets a rename overwrite a file another handle still has
            # open; Windows does not, and answers PermissionError [WinError 5]
            # instead. Both documents are closed here rather than in the
            # `finally` below, which is where they used to be closed and which
            # runs after this line: on Windows the whole translation reached
            # this point and then failed at the last step, with the finished PDF
            # sitting beside its destination under a .imgfix suffix.
            #
            # The `finally` below closes whatever is still open; it checks
            # `is_closed` because pymupdf raises ValueError on a second close.
            # 覆盖之前先关闭，而不是之后。
            #
            # POSIX 允许改名覆盖一个仍被其他句柄打开的文件，Windows 不允许，会以
            # PermissionError [WinError 5] 作答。这里把两个文档就地关掉，而不是留给下面的
            # finally——那才是它们原先关闭的地方，且在本行之后才执行：于是 Windows 上整趟
            # 翻译走到这一步才失败，而译好的 PDF 就带着 .imgfix 后缀躺在目标文件旁边。
            #
            # 下面的 finally 只关闭仍然打开的那些：pymupdf 对已关闭的文档再次调用 close()
            # 会抛 ValueError。
            source.close()
            target.close()
            temp.replace(target_path)
            logger.info(
                "Restored %d figure region(s) and %d rule(s) across %d page(s) in %s",
                result["restored"], result["rules"], result["pages"], target_path.name,
            )
    finally:
        for document in (source, target):
            if not document.is_closed:
                document.close()

    return result


def restore_all(source_path: Path | str, outputs: dict[str, str | None],
                translate_first: bool = False) -> dict[str, int]:
    """
    中文：对所有产物执行图区修复。

    Repair every produced PDF against the source.

    Both outputs need it, for the same underlying reason. The mono PDF loses its
    figures outright. The dual PDF keeps the original page beside the
    translation, so a count of its images looks correct while the translated
    half is in fact blank — a figure the reader can see on the left and not on
    the right.

    Args:
        source_path (Path | str): Original PDF.
        outputs (dict[str, str | None]): Produced paths keyed by kind.
        translate_first (bool): The dual layout puts the translation on the left.

    Returns:
        dict[str, int]: Combined counts across repaired files.
    """
    totals = {"restored": 0, "pages": 0, "rules": 0}

    mono = outputs.get("mono_path")
    if mono:
        for key, value in restore_images(source_path, mono).items():
            totals[key] += value

    dual = outputs.get("dual_path")
    if dual:
        for key, value in restore_images(source_path, dual, dual=True,
                                         translate_first=translate_first).items():
            totals[key] += value

    return totals
