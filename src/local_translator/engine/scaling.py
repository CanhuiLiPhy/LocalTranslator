"""
中文：先向下扩框，再考虑缩字号——大多数段落下方本来就有空地。

Grow a paragraph's box before shrinking its type.

**The symptom.** Most of a translated paper comes out slightly small. Measured
over one 37-page document, only 5,026 characters were set at the body size of
10pt, against 7,862 at 9.5pt, 7,769 at 9pt and 4,428 at 8.5pt. Several
paragraphs sit at 8pt with an inch of blank paper underneath them.

**Why it happens.** Chinese needs more width than the English it replaces — a
full-width glyph per character — so a translated paragraph rarely fits the box
its original occupied. BabelDOC responds by shrinking: ``scale`` starts at 1.0
and steps down 0.05 at a time until the text fits. It *can* grow the box
instead, but only once ``scale`` has already fallen below 0.7, and it then
re-tries at that reduced scale rather than starting over — so a paragraph that
would have fitted at full size in a taller box is still typeset at 0.65. A
paragraph that fits at 0.9 never even reaches the growing code.

**The fix.** Ask for the scale twice: once as BabelDOC computes it, and once
with the box grown downwards as far as ``get_max_bottom_space`` allows. If the
taller box gives a better scale, keep it. Growing is free — the space is empty
by construction, and the layout only uses as many lines as it needs — so the
paragraph ends up at full size wherever the page has room, and shrinks only
where it genuinely has none.

BabelDOC's own bottom-space function is used rather than a new one, so what
counts as free space is unchanged; only the order of the two remedies is.

**Inside a drawing the rule is the opposite.** A chart label's box is the label,
not a slot on the page: below an axis label is the plot, below a legend entry is
the next entry, and to the right of both is the artwork. BabelDOC grows a box in
both directions once ``scale`` falls below 0.7, which on page 8 of
arXiv:2608.19072 pushed the legend out of its white panel and across the bars.
So for a paragraph that belongs to a figure, a table or a display formula this
module:

1. holds the box fixed — neither BabelDOC's growth nor ours;
2. lets the type shrink, which is the only remedy a fixed box has;
3. puts the source text back when even the smallest usable size does not fit.

Step 3 needs the untranslated paragraph, which BabelDOC overwrites in
``ILTranslator.post_translate_paragraph``; a copy is taken there and released
when the document is done.
"""

from __future__ import annotations

import logging
import os
import re
import statistics
import threading

logger = logging.getLogger(__name__)

_SCALE_PROBE = bool(os.getenv("LOCALTRANS_SCALE_PROBE"))

# A scale at or above this is already full size, so there is nothing to gain.
# Left slightly below 1.0 because the search steps in hundredths.
# 达到该缩放比例即视为原大，无可再争取。略低于 1.0 是因为搜索以百分位步进。
_FULL = 0.99

# Clearance kept between the grown box and whatever sits below it, in points.
# The same value BabelDOC uses when it grows a box itself.
# 扩框后与下方元素之间保留的间隙（点）。与 BabelDOC 自行扩框时所用的值相同。
_CLEARANCE = 2.0

# Smallest gain worth taking. Below this the two layouts are the same size and
# growing the box only moves text down the page for nothing.
# 值得采纳的最小收益。低于此值两种排版字号相同，扩框只是白白把文字下移。
_MIN_GAIN = 0.02


def _curve_floor(box, page) -> float:
    """
    中文：算出段落下方最近的矢量图形顶部，扩框不得越过它。

    Find the top of the nearest vector drawing below a paragraph.

    ``Typesetting.get_max_bottom_space`` looks at paragraphs, loose characters
    and figures, and at nothing else — a callout box is a filled path, so it is
    invisible to that check and a paragraph grew straight down through the
    rounded box under it, clipping its own last line against the border.

    Args:
        box (Box): The paragraph's current box.
        page (Page): Its page.

    Returns:
        float: The highest ``y2`` among the drawings below the paragraph that
            overlap it horizontally, or negative infinity when there are none.
    """
    floor = float("-inf")
    for curve in getattr(page, "pdf_curve", None) or []:
        other = getattr(curve, "box", None)
        if other is None:
            continue
        if other.y2 >= box.y:
            continue
        if other.x >= box.x2 or other.x2 <= box.x:
            continue
        floor = max(floor, other.y2)
    return floor


# Layout labels whose box is part of a drawing rather than part of the page.
# 其框属于图形内部而非页面版心的版面标签。
_DRAWING_LABELS = frozenset({"figure", "table", "table_footnote", "isolate_formula"})


# Smallest share of its original size a drawing's text may be set at before the
# source is put back instead. A chart label starts at 5–8pt, so 0.6 already
# means 3–5pt: below that the Chinese is unreadable, and unreadable Chinese is a
# worse outcome for the reader than the English that was there before.
# 图内文字可缩到原字号的最小比例，再小就改为放回原文。图表标签本来就只有 5–8pt，
# 0.6 已经是 3–5pt：再小中文就看不清了，而看不清的中文对读者比原本的英文更糟。
_DRAWING_FLOOR = 0.6

# There is deliberately no absolute point-size floor here, only the relative one
# above. An absolute floor cannot be enforced from inside the pipeline: the
# intermediate layout's boxes are not always in page coordinates. On page 7 of
# arXiv:2608.13520 — three panels drawn as *nested* Form XObjects — a label whose
# source measures 6.26pt on the page is recorded as 6.26 in the layout, is
# rendered at that same 6.26, and comes out of BabelDOC at 3.69pt on the page.
# The translation and the source are identical in every number this module can
# read, and differ by 1.7× in the file. A check written against those numbers
# could never fire, so none is written; `scripts/audit-output.py` reports the
# outcome from the produced PDF instead, which is where it is visible.
# 这里刻意没有绝对字号下限，只有上面那条相对下限。绝对下限在管线内部无法执行：中间版式
# 的框并不总是页面坐标。arXiv:2608.13520 第 7 页——三个面板画成*嵌套的* Form
# XObject——某个标签在页面上量得 6.26pt，版式里记的也是 6.26，渲染用的还是 6.26，而
# BabelDOC 输出到页面上是 3.69pt。译文与原文在本模块能读到的每一个数上都相同，在文件里
# 却差 1.7 倍。按这些数字写的检查永远不会触发，所以不写；改由
# scripts/audit-output.py 从产物 PDF 上报告，那里才看得见。

# Smallest size ratio treated as a real measurement rather than a mistake. A
# label drawn at less than a quarter of what its translated run claims is not a
# font-metrics difference, so that paragraph is left to BabelDOC.
# 小到这个比例以下就不当作有效测量。原文实际字号不到译文串所报字号的四分之一，那不是
# 字体度量的差异，这一段交还给 BabelDOC。
_MIN_CAP = 0.25

# Chinese, for telling a translated label from one left in English.
# 汉字，用于区分已翻译的标签与保留英文的标签。
_CJK = re.compile(r"[\u4e00-\u9fff]")

# Set while a paragraph that belongs to a drawing is being laid out. Per thread
# because BabelDOC typesets pages from a worker pool.
# 排版图内段落期间置位。按线程保存，因为 BabelDOC 用工作线程池排版各页。
_LAYING_OUT_A_DRAWING = threading.local()

# Untranslated paragraphs held for the length of one document, keyed by
# ``debug_id`` — the base58 identity ``paragraph_finder`` gives every paragraph
# when it forms them, before anything is translated.
#
# Object identity does not work here. BabelDOC does not carry the same
# ``PdfParagraph`` objects from the translation stage into typesetting, so a
# table keyed by ``id()`` was written 31 times and read back 35 misses on a
# single page. ``debug_id`` survives whatever copying happens in between, and
# ``PdfParagraph`` is a slotted dataclass, so hanging the copy off the paragraph
# itself is not available either.
# 按 debug_id 保存的未翻译段落，生命周期为一篇文档。debug_id 是 paragraph_finder
# 组段时分配的 base58 标识，早于任何翻译。
#
# 这里不能用对象标识：BabelDOC 并不把翻译阶段的 PdfParagraph 原样带进排版阶段，
# 用 id() 做键在同一页上写进 31 条、读出 35 次全落空。debug_id 能跨越中间的复制，
# 而 PdfParagraph 是带 slots 的 dataclass，也无法把副本挂在段落自己身上。
_SOURCE_TEXT: dict[str, tuple] = {}


def _key(paragraph) -> str | None:
    """
    中文：取段落的稳定标识。

    Return the paragraph's stable identity.

    Args:
        paragraph (PdfParagraph): Paragraph to identify.

    Returns:
        str | None: Its ``debug_id``, or ``None`` when it has none — in which
            case it cannot be tracked across the copy and is left alone.
    """
    return getattr(paragraph, "debug_id", None) or None


def _inside_a_drawing(paragraph) -> bool:
    """
    中文：判断段落是否位于图、表或公式内部——那里下方没有空地可扩。

    Decide whether a paragraph sits inside a drawing rather than on the page.

    Args:
        paragraph (PdfParagraph): Paragraph about to be laid out.

    Returns:
        bool: ``True`` for text belonging to a figure, a table or a display
            formula, including text drawn inside a Form XObject.
    """
    label = (getattr(paragraph, "layout_label", "") or "").strip().lower()
    if label in _DRAWING_LABELS:
        return True
    return bool(getattr(paragraph, "xobj_id", None))


def install() -> None:
    """
    中文：让 BabelDOC 在缩字号之前先尝试把段落框往下扩。

    Make BabelDOC try growing a paragraph's box before it shrinks the type.

    Wraps ``Typesetting._find_optimal_scale_and_layout``. The wrapper measures
    the scale in the original box, and — only when that came out below full size
    — measures it again in a box grown downwards. The better of the two is the
    one the original method is then asked to apply, so all of BabelDOC's own
    layout logic still does the work.

    The probe runs with ``apply_layout=False``, which BabelDOC guarantees not to
    mutate the paragraph, so a probe that does not pay off leaves nothing behind.

    Idempotent, so a second call is harmless.

    Raises:
        AttributeError: If BabelDOC no longer exposes the method, so an upgrade
            that moves it fails loudly at startup instead of silently doing
            nothing.
    """
    from babeldoc.format.pdf.document_il.il_version_1 import Box
    from babeldoc.format.pdf.document_il.midend.typesetting import Typesetting

    if getattr(Typesetting._find_optimal_scale_and_layout, "_scaling_patched", False):
        return

    original = Typesetting._find_optimal_scale_and_layout

    def _find_optimal_scale_and_layout(self, paragraph, page, typesetting_units,
                                       initial_scale=1.0,
                                       use_english_line_break=True,
                                       apply_layout=False):
        """中文：先探一次，能靠扩框保住字号就扩；图表内不扩。 / Probe first, except inside a figure."""
        if _inside_a_drawing(paragraph):
            # Growing is free on a page — the space below a paragraph is empty
            # by construction. Inside a chart it is not: below an axis label is
            # the plot, below a legend entry is the next one, and to the right
            # of both is the artwork. Neither our growth nor BabelDOC's own may
            # run here, and the flag is what stops BabelDOC's: it makes the two
            # space functions report a box that is already as large as it gets.
            # 在页面上扩框是免费的——段落下方按构造就是空的。在图表里并非如此：坐标轴
            # 标签下面是绘图区，图例条目下面是下一个条目，两者右边都是图形本身。这里
            # 我们的扩框和 BabelDOC 自己的扩框都不能跑，标志位挡住的是后者：它让两个
            # 空间函数报告「这个框已经不能再大了」。
            previous = getattr(_LAYING_OUT_A_DRAWING, "active", False)
            _LAYING_OUT_A_DRAWING.active = True
            try:
                return original(self, paragraph, page, typesetting_units,
                                initial_scale, use_english_line_break, apply_layout)
            finally:
                _LAYING_OUT_A_DRAWING.active = previous
        try:
            grown = _grow(self, paragraph, page, typesetting_units, initial_scale,
                          use_english_line_break, original, Box)
        except Exception:  # noqa: BLE001
            # Failing here must not cost the paragraph: without the probe the
            # original method still lays it out, just smaller.
            # 这里失败不应赔上这一段：没有探测，原方法照样排版，只是字号小一些。
            logger.debug("Box growing failed for a paragraph", exc_info=True)
            grown = None

        if grown is not None:
            paragraph.box = grown
        _before = getattr(paragraph, "box", None)
        _h0 = (_before.y2 - _before.y) if _before else -1
        if _SCALE_PROBE and _before is not None:
            logger.info("[对象] apply=%-5s id=%d debug_id=%s 框=%.0fpt %r",
                        apply_layout, id(paragraph),
                        getattr(paragraph, "debug_id", "?"), _h0,
                        (getattr(paragraph, "unicode", "") or "")[:20])
        result = original(self, paragraph, page, typesetting_units, initial_scale,
                          use_english_line_break, apply_layout)
        if _SCALE_PROBE:
            _after = getattr(paragraph, "box", None)
            _h1 = (_after.y2 - _after.y) if _after else -1
            if abs(_h1 - _h0) > 0.5:
                logger.info("[框变化] apply=%s  %.0fpt → %.0fpt  %r",
                            apply_layout, _h0, _h1,
                            (getattr(paragraph, "unicode", "") or "")[:24])
        if _SCALE_PROBE and apply_layout:
            scale = result[0] if isinstance(result, tuple) else result
            text = (getattr(paragraph, "unicode", "") or "")[:34]
            box = getattr(paragraph, "box", None)
            logger.info("[scale] %.2f 扩框=%s 高=%.0f %r",
                        scale if isinstance(scale, (int, float)) else -1,
                        "是" if grown is not None else "否",
                        (box.y2 - box.y) if box else -1, text)
        return result

    def _grow(self, paragraph, page, units, initial_scale, use_english_line_break,
              original, Box):
        """
        中文：算出向下扩后的段落框；不值得扩则返回 None。

        Work out a taller box for the paragraph, or ``None`` to leave it alone.

        Args:
            self (Typesetting): The typesetting stage.
            paragraph (PdfParagraph): Paragraph being laid out.
            page (Page): Its page, used to find the free space below it.
            units (list): Typesetting units for the paragraph.
            initial_scale (float): Scale the search starts from.
            use_english_line_break (bool): Passed through to the search.
            original (callable): BabelDOC's unwrapped search.
            Box (type): The intermediate layout's box class.

        Returns:
            Box | None: A box grown downwards, when doing so lets the paragraph
                be set larger. ``None`` when the paragraph already fits at full
                size, when there is no free space below it, or when the extra
                space buys nothing.
        """
        box = paragraph.box
        if box is None or initial_scale < _FULL:
            return None

        before, _ = original(self, paragraph, page, units, initial_scale,
                             use_english_line_break, False)
        if before is None or before >= _FULL:
            return None

        bottom = self.get_max_bottom_space(box, page) + _CLEARANCE
        curve = _curve_floor(box, page) + _CLEARANCE
        floor = max(bottom, curve)
        if _SCALE_PROBE:
            logger.info("[grow] 框 x=%.0f..%.0f y=%.0f..%.0f  BabelDOC底=%.0f 曲线底=%.0f → %s",
                        box.x, box.x2, box.y, box.y2, bottom, curve,
                        "不扩" if floor >= box.y else f"扩到 {floor:.0f}")
        if floor >= box.y:
            return None
        taller = Box(x=box.x, y=floor, x2=box.x2, y2=box.y2)

        paragraph.box = taller
        try:
            after, _ = original(self, paragraph, page, units, initial_scale,
                                use_english_line_break, False)
        finally:
            paragraph.box = box

        if after is None or after - before < _MIN_GAIN:
            return None
        logger.debug("Grew a paragraph box downwards by %.1fpt, scale %.2f -> %.2f",
                     box.y - floor, before, after)
        return taller

    _find_optimal_scale_and_layout._scaling_patched = True
    Typesetting._find_optimal_scale_and_layout = _find_optimal_scale_and_layout

    _hold_the_box(Typesetting)
    _fall_back_to_the_source(Typesetting)
    install_source_keeper()


def _hold_the_box(Typesetting) -> None:
    """
    中文：图内排版期间让 BabelDOC 的两个「可扩空间」函数报告没有空间。

    Make BabelDOC's two space functions report no room while a drawing is set.

    ``_find_optimal_scale_and_layout`` grows the box once ``scale`` drops below
    0.7 — downwards first, then to the right — and mutates ``paragraph.box``
    while doing it. Both directions are wrong inside a chart, and the growth is
    buried in the middle of a loop that cannot be reached from outside. What can
    be reached are the two functions the loop asks for room, so they answer
    ``already at the edge`` while the flag is set: ``min_y = bottom + 2`` is then
    not below ``box.y`` and ``max_x = right - 5`` is not beyond ``box.x2``, so
    both attempts decline and the search goes on shrinking the type instead.

    Args:
        Typesetting (type): BabelDOC's typesetting stage.

    Returns:
        None.
    """
    if getattr(Typesetting.get_max_bottom_space, "_scaling_patched", False):
        return

    original_bottom = Typesetting.get_max_bottom_space
    original_right = Typesetting.get_max_right_space

    def get_max_bottom_space(self, current_box, page):
        """中文：图内时报告框底即极限。 / Report the box's own floor inside a drawing."""
        if getattr(_LAYING_OUT_A_DRAWING, "active", False):
            return current_box.y
        return original_bottom(self, current_box, page)

    def get_max_right_space(self, current_box, page):
        """中文：图内时报告框右即极限。 / Report the box's own right edge inside a drawing."""
        if getattr(_LAYING_OUT_A_DRAWING, "active", False):
            return current_box.x2
        return original_right(self, current_box, page)

    get_max_bottom_space._scaling_patched = True
    Typesetting.get_max_bottom_space = get_max_bottom_space
    Typesetting.get_max_right_space = get_max_right_space


def keep_source(paragraph) -> None:
    """
    中文：在译文覆盖之前留一份图内段落的原文。

    Keep a copy of a drawing's paragraph before the translation overwrites it.

    Called from the translation stage, which replaces both ``unicode`` and
    ``pdf_paragraph_composition`` in place. Only paragraphs that belong to a
    drawing are kept, because only those can be reverted — a paragraph on the
    page has room to shrink into and never needs its source back.

    Keyed by ``debug_id`` rather than by object identity; see
    :data:`_SOURCE_TEXT`.

    Args:
        paragraph (PdfParagraph): Paragraph about to be translated.

    Returns:
        None.
    """
    key = _key(paragraph)
    if key is None or not _inside_a_drawing(paragraph):
        return
    _SOURCE_TEXT[key] = (
        paragraph.unicode,
        list(paragraph.pdf_paragraph_composition or []),
    )


def _characters_of(composition):
    """
    中文：取出一个组成单元里的字符。

    List the characters a paragraph composition holds.

    A composition is a union of five shapes and only one of them is set, so this
    walks them in turn rather than guessing from the paragraph.

    Args:
        composition (PdfParagraphComposition): One element of a paragraph.

    Returns:
        list: Its ``PdfCharacter`` objects, empty when it holds none.
    """
    for name in ("pdf_line", "pdf_same_style_characters", "pdf_formula"):
        holder = getattr(composition, name, None)
        if holder is not None:
            return holder.pdf_character or []
    char = getattr(composition, "pdf_character", None)
    return [char] if char is not None else []


def _run_size(paragraph) -> float | None:
    """
    中文：读出译文串的字号——BabelDOC 就是按这个数值把新文字画上去的。

    Read the size the translated run will be drawn at.

    Args:
        paragraph (PdfParagraph): Paragraph carrying the translated runs.

    Returns:
        float | None: The largest run's ``font_size``, because that is the one
            that will overflow, or ``None`` when no run carries one.
    """
    sizes = []
    for composition in paragraph.pdf_paragraph_composition or []:
        run = getattr(composition, "pdf_same_style_unicode_characters", None)
        style = getattr(run, "pdf_style", None) if run is not None else None
        size = getattr(style, "font_size", None)
        if size:
            sizes.append(size)
    return max(sizes) if sizes else None


def _drawn_size(compositions) -> float | None:
    """
    中文：量出原文字符实际画出来的字号——取字符框的高度，而不是记录的字号。

    Measure the size the source characters were actually drawn at.

    ``pdf_style.font_size`` is not that size. Measured on page 8 of
    arXiv:2608.19072, an axis label that PyMuPDF reports as 5.712pt type carries
    ``font_size`` 9.687 — which is the height of its *line box*, not its type
    size. The two agree for the serif face the body is set in, whose glyph box
    is about one em tall, and disagree by a factor of 1.7 for the DejaVu Sans
    that matplotlib embeds in every figure. BabelDOC draws the translation at
    ``font_size × scale``, so a 5.7pt axis label comes back at 9.7pt: two thirds
    larger than the numbers beside it, over the axis and over the label below.

    The character's own box does hold the type size — 5.712 against PyMuPDF's
    5.712, checked at three sizes across two figures — so that is what this
    reads.

    Args:
        compositions (list): The paragraph's compositions before translation.

    Returns:
        float | None: Median height of the source characters' boxes, or ``None``
            when none of them carries one. The median rather than the mean, so a
            single superscript does not speak for the label.
    """
    heights = []
    for composition in compositions or []:
        for char in _characters_of(composition):
            box = getattr(char, "box", None)
            if box is not None and box.y2 > box.y:
                heights.append(box.y2 - box.y)
    return statistics.median(heights) if heights else None


def _size_cap(paragraph, source) -> float | None:
    """
    中文：算出让图内译文与原文一样大的缩放上限。

    Work out the scale that sets a drawing's translation in the size it replaces.

    A chart is drawn to a size and every label in it shares one small face. The
    translated run does not inherit that size — see :func:`_drawn_size` — so it
    is capped here at the ratio between the size the source was drawn at and the
    size the run would otherwise be drawn at. The fit search then works
    downwards from there.

    Args:
        paragraph (PdfParagraph): Paragraph carrying the translated run.
        source (tuple): The entry :func:`keep_source` stored for it.

    Returns:
        float | None: The largest scale the paragraph may be set at, or ``None``
            when either size cannot be measured, when the run is already no
            larger than its source, or when the ratio is too small to be a
            font-metrics difference and is therefore not trusted.
    """
    was = _drawn_size(source[1])
    now = _run_size(paragraph)
    if not was or not now:
        return None
    cap = was / now
    if cap >= 1.0 or cap < _MIN_CAP:
        return None
    return cap


# How far a drawing's translated text may sit below the text it replaced before
# it is lifted back, in points. Below this the difference is rounding.
# 图内译文可以比它替换掉的原文低多少而不被抬回，单位为点。小于此值属于舍入误差。
_DROP_SLACK = 0.2

# Largest lift applied, as a share of the source text's own height. One line's
# worth: a Chinese glyph fills its em box where a Latin one leaves the descender
# space empty, so the drift between the two can be most of a line and still be
# nothing but that difference. Measured on the tick labels of page 8 of
# arXiv:2608.19072, 「Context」 → 「上下文」 drifted 2.29pt at 3.18pt type — 72% of
# a line, and refusing to lift it left the label sitting on the one below.
# Beyond a full line the discrepancy is a different fault, which this must not
# paper over.
# 最大抬升量，按原文自身高度的比例计，取一行的高度：汉字填满字框，而拉丁字母的下伸区
# 是空的，两者的漂移可以接近一整行而仍然只是这个差异。实测 arXiv:2608.19072 第 8 页的
# 刻度标签，3.18pt 字号下「Context」→「上下文」漂移了 2.29pt——占一行的 72%，不抬就会
# 压在下面那一行上。超过一整行则是别的毛病，不能用这里掩盖过去。
_MAX_LIFT = 1.0


def _floor_of(compositions) -> float | None:
    """
    中文：取一组组成里所有字符的最低边。

    Find the lowest edge among a set of compositions' characters.

    Args:
        compositions (list): Paragraph compositions to measure.

    Returns:
        float | None: The smallest ``box.y``, or ``None`` when nothing carries a
            box.
    """
    floors = [char.box.y for composition in compositions or []
              for char in _characters_of(composition)
              if getattr(char, "box", None) is not None]
    return min(floors) if floors else None


def _lift_into_place(paragraph, source) -> None:
    """
    中文：把图内译文抬回原文所占的高度范围，避免下缘被图形边界裁掉。

    Lift a drawing's translated text back onto the band its source occupied.

    A Chinese glyph fills its em box where a Latin one leaves the descender
    space empty, and BabelDOC lays a paragraph out from the top of its box, so
    the translated line ends up sitting lower than the line it replaces. Inside
    a figure there is nothing below it but the figure's own clipping boundary:
    on page 8 of arXiv:2608.19072 the axis title 「AIME 2025得分（pass@8，%）」
    came out 2.2pt low and the descender of 「p」 was cut off by the form's edge.

    Only upwards, and only by a little: a paragraph that sits higher than its
    source is not a problem, and a large discrepancy is a different fault that
    this must not paper over.

    Args:
        paragraph (PdfParagraph): Paragraph as it has just been laid out.
        source (tuple): The entry :func:`keep_source` stored for it.

    Returns:
        None. The characters' boxes are moved in place.
    """
    was = _floor_of(source[1])
    now = _floor_of(paragraph.pdf_paragraph_composition)
    if was is None or now is None:
        return
    lift = was - now
    height = _drawn_size(source[1]) or 0.0
    if lift <= _DROP_SLACK or not height or lift > height * _MAX_LIFT:
        logger.debug("Not lifting %.24s: was=%.2f now=%.2f lift=%.2f height=%.2f",
                     paragraph.unicode or "", was, now, lift, height)
        return
    for composition in paragraph.pdf_paragraph_composition or []:
        for char in _characters_of(composition):
            box = getattr(char, "box", None)
            if box is not None:
                box.y += lift
                box.y2 += lift
            visual = getattr(getattr(char, "visual_bbox", None), "box", None)
            if visual is not None:
                visual.y += lift
                visual.y2 += lift
    logger.debug("Lifted a drawing's text %.2fpt back onto its own line", lift)


def _lost_glyphs(paragraph) -> bool:
    """
    中文：判断排版后是否有字被丢掉——图内小标签常见。

    Decide whether the layout dropped characters it was given.

    BabelDOC's fit test can report success on a box that is in fact too small,
    and what it then produces is the text minus the glyphs that did not make it:
    on page 8 of arXiv:2608.19072 a tick label came out as 「上下文-富有」 with the
    last character gone and its two lines overlapping. A label missing a
    character is worse than the English it replaced, so this is the last check
    before the translation is accepted.

    Skipped for a paragraph carrying a formula, where a placeholder in the text
    stands for several characters on the page and the counts cannot be compared.

    Args:
        paragraph (PdfParagraph): Paragraph as it has just been laid out.

    Returns:
        bool: ``True`` when fewer characters were drawn than the text held.
    """
    text = paragraph.unicode or ""
    if "{v" in text:
        return False
    wanted = sum(1 for char in text if not char.isspace())
    if not wanted:
        return False
    drawn = sum(1 for composition in paragraph.pdf_paragraph_composition or []
                for char in _characters_of(composition)
                if not (getattr(char, "char_unicode", "") or " ").isspace())
    return drawn < wanted


# Rows whose baselines differ by less than this are the same row.
# 基线相差小于该值的字符属于同一行。
_ROW_TOLERANCE = 0.5

# How much more the translated rows may overlap each other than the source's did
# before the translation is refused, in points. A Latin line box carries padding
# its glyphs do not use, so a small overlap is normal and invisible; a Chinese
# glyph fills its box, so the same overlap is two characters on top of each
# other. Measured on the tick labels of page 8 of arXiv:2608.19072, the source's
# two rows overlap by 1.8pt and read cleanly, and the translation's by 3.3pt and
# does not.
# 译文各行之间比原文多出多少重叠即拒绝该译文，单位为点。拉丁字母的行框留有字形用不到
# 的空白，因此小幅重叠正常且看不出来；汉字填满字框，同样的重叠就是两个字叠在一起。
# 实测 arXiv:2608.19072 第 8 页的刻度标签：原文两行重叠 1.8pt 而清晰可读，译文重叠
# 3.3pt 则不可读。
_OVERLAP_SLACK = 1.0


def _max_overlap(compositions) -> float:
    """
    中文：量出一段文字相邻两行之间最大的重叠量。

    Measure the largest overlap between consecutive rows of a paragraph.

    Args:
        compositions (list): Paragraph compositions to measure.

    Returns:
        float: The largest amount by which a row reaches into the one above it,
            in points. Zero when the paragraph has fewer than two rows.
    """
    rows: list[list[float]] = []
    for composition in compositions or []:
        for char in _characters_of(composition):
            box = getattr(char, "box", None)
            if box is None:
                continue
            for row in rows:
                if abs(row[0] - box.y) <= _ROW_TOLERANCE:
                    row[1] = max(row[1], box.y2)
                    break
            else:
                rows.append([box.y, box.y2])
    if len(rows) < 2:
        return 0.0
    rows.sort(key=lambda row: row[0], reverse=True)
    return max(lower[1] - upper[0] for upper, lower in zip(rows, rows[1:], strict=False))


def _fits(stage, paragraph, page, fonts, ceiling: float = 1.0) -> bool:
    """
    中文：判断译文能否在原框内以可读字号排下。

    Decide whether the translation fits its box at a size still worth reading.

    Args:
        stage (Typesetting): The typesetting stage, used for its own measuring.
        paragraph (PdfParagraph): Paragraph about to be rendered.
        page (Page): Its page.
        fonts (dict): Fonts available to the paragraph.
        ceiling (float): Largest scale the paragraph may use — the cap that puts
            it in the figure's own type size. The relative floor is taken
            against this, so 「shrunk to 60%」 means 60% of the size the label
            had rather than 60% of whatever size the translated run arrived
            with.

    Returns:
        bool: ``True`` when the text fits at :data:`_DRAWING_FLOOR` of the
            ceiling, or when it needs no re-typesetting at all. This is the
            ratio floor only; whether the result is large enough to read is
            settled afterwards by :func:`_too_small`, because what the search
            says here and what the layout actually does are not always the same
            number. A measurement that raises is reported as fitting, so a
            failure here costs nothing beyond leaving the paragraph as BabelDOC
            would have set it.
    """
    try:
        units = stage.create_typesetting_units(paragraph, fonts)
        if all(unit.can_passthrough for unit in units):
            return True
        scale, _ = stage._find_optimal_scale_and_layout(
            paragraph, page, units, ceiling, True, False)
        return scale >= _DRAWING_FLOOR * ceiling
    except Exception:  # noqa: BLE001
        logger.debug("Could not measure a drawing's paragraph", exc_info=True)
        return True


def _fall_back_to_the_source(Typesetting) -> None:
    """
    中文：图内文字若缩到可读下限仍放不下，就把原文放回去。

    Put the source text back when a drawing's translation will not fit.

    A fixed box has exactly one remedy, and once the type has shrunk as far as
    it is worth shrinking there is nothing left to try. BabelDOC's answer is to
    lay the text out anyway, which either overflows the box or — when no scale
    down to 0.1 fits — drops the characters that did not make it, leaving
    「clean」 rendered as 「an」. Restoring the untranslated paragraph gives units
    that can all pass through, so the source characters are stamped back exactly
    where they were, at their original size.

    Args:
        Typesetting (type): BabelDOC's typesetting stage.

    Returns:
        None.
    """
    if getattr(Typesetting.render_paragraph, "_scaling_patched", False):
        return

    original_render = Typesetting.render_paragraph
    original_document = Typesetting.typesetting_document

    def render_paragraph(self, paragraph, page, fonts):
        """中文：图内放不下就还原原文，再交给 BabelDOC 排版。 / Revert, then render."""
        source = _SOURCE_TEXT.get(_key(paragraph)) if _inside_a_drawing(paragraph) else None
        if source is not None and (paragraph.unicode or "") == (source[0] or ""):
            # The text came back unchanged — a label that is one whitelisted
            # term, or one the numeric rule held back. BabelDOC does not notice:
            # its own check compares the output string against the *input
            # object*, so it re-typesets the identical text as a fresh run and a
            # 5.7pt label is redrawn at 4.8pt, smaller than everything around
            # it. Putting the original composition back makes it pass through
            # instead, so the glyphs are the ones that were there.
            # 文字原样返回——可能是一个白名单术语构成的标签，也可能是被数字规则拦下的。
            # BabelDOC 察觉不到：它自己的判断是把输出字符串与*输入对象*相比，于是把
            # 一模一样的文字当成新串重排，5.7pt 的标签被重画成 4.8pt，比周围一切都小。
            # 把原组成放回去，它就走直通路径，画出来的还是原来那些字形。
            paragraph.pdf_paragraph_composition = source[1]
            paragraph.scale = 1.0
            paragraph.optimal_scale = 1.0
            return original_render(self, paragraph, page, fonts)
        if source is not None:
            cap = _size_cap(paragraph, source)
            if cap is not None and (paragraph.optimal_scale or 1.0) > cap:
                logger.debug("Capping a drawing's type at %.2f of its run size", cap)
                paragraph.optimal_scale = cap
            ceiling = paragraph.optimal_scale if paragraph.optimal_scale else 1.0
            if not _fits(self, paragraph, page, fonts, ceiling):
                unicode, composition = source
                logger.debug("Keeping the source inside a drawing: %.60s", unicode or "")
                paragraph.unicode = unicode
                paragraph.pdf_paragraph_composition = composition
                paragraph.scale = 1.0
                paragraph.optimal_scale = 1.0
                return original_render(self, paragraph, page, fonts)
        rendered = original_render(self, paragraph, page, fonts)
        if source is None:
            return rendered
        crowded = _max_overlap(paragraph.pdf_paragraph_composition) > (
            _max_overlap(source[1]) + _OVERLAP_SLACK)
        if _lost_glyphs(paragraph) or crowded:
            logger.debug("The layout %s inside a drawing; keeping %.40s",
                         "crowded the lines" if crowded else "dropped glyphs",
                         source[0] or "")
            paragraph.unicode = source[0]
            paragraph.pdf_paragraph_composition = source[1]
            paragraph.scale = 1.0
            paragraph.optimal_scale = 1.0
            return original_render(self, paragraph, page, fonts)
        _lift_into_place(paragraph, source)
        return rendered

    def typesetting_document(self, document):
        """中文：一篇文档排完即释放留存的原文。 / Release the kept sources per document."""
        try:
            return original_document(self, document)
        finally:
            _SOURCE_TEXT.clear()

    render_paragraph._scaling_patched = True
    Typesetting.render_paragraph = render_paragraph
    Typesetting.typesetting_document = typesetting_document


def install_source_keeper() -> None:
    """
    中文：在翻译写回段落之前留一份图内原文。

    Take a copy of a drawing's paragraph before the translation overwrites it.

    Wraps ``ILTranslator.post_translate_paragraph``, which is the last moment the
    untranslated ``unicode`` and composition still exist — its first act is to
    replace both. Only drawings are copied, so the cost is one list per figure
    label rather than one per paragraph in the document.

    Kept as its own function because it patches the translation stage rather
    than the typesetting stage; :func:`install` calls it.

    Idempotent, so a second call is harmless.

    Returns:
        None.
    """
    from babeldoc.format.pdf.document_il.midend.il_translator import ILTranslator

    if getattr(ILTranslator.post_translate_paragraph, "_scaling_patched", False):
        return

    original = ILTranslator.post_translate_paragraph

    def post_translate_paragraph(self, paragraph, tracker, translate_input,
                                 translated_text):
        """中文：先留档再交给 BabelDOC 写回。 / Keep the source, then delegate."""
        keep_source(paragraph)
        return original(self, paragraph, tracker, translate_input, translated_text)

    post_translate_paragraph._scaling_patched = True
    ILTranslator.post_translate_paragraph = post_translate_paragraph
