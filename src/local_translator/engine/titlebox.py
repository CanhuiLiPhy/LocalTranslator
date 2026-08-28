"""
中文：标题框的宽度是英文碰巧断行的地方，译文因此为了一个字换行。

A title's box is as wide as the English happened to need, and no wider.

**The symptom.** The Gemini report's title comes out as::

    Gemini：能力卓越的多模态模型系
    列

One character on a line of its own. Nothing is wrong with the type — the
paragraph is set at full size — and there is a third of the page free to the
right of it.

**Why.** A paragraph's box is the bounding box of the lines it had in the
source, and the English title broke after ``Highly Capable``, at x=406.3 on a
page 595 points wide. That break point is an accident of English word lengths,
but it becomes the measure the Chinese must fit: the translation needs 412
points, overruns by six, and wraps.

Growing the box is the remedy, and BabelDOC has one — but it only grows
*downwards*, and only once the type has already been shrunk below 0.7. Here the
type never shrinks: two lines at full size is a perfectly good fit as far as the
search is concerned. The paragraph never reaches the growing code.

**When the widening has to happen.** Every paragraph's scale is chosen in a pass
over the whole document before anything is rendered, so a box widened at render
time would arrive after the scale it should have changed. The widening therefore
sits in the scale search itself.

**The fix.** A title's measure is the text column, not the width of its English.
So a title that would fit on one line in the space the page actually has is
given that space. The test is arithmetic on the units the typesetter has
already built — their widths summed against the free space to the right — and
the box is widened only when the whole title then fits on one line. A title too
long for one line is left exactly as it was, so a three-line title does not turn
into a two-line one with a different shape.

The free space comes from BabelDOC's own ``get_max_right_space``, which stops at
anything that overlaps the box vertically, so the room taken is provably empty.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Which layout labels this applies to. A running head is not a title in this
# sense, and a caption's measure really is its figure.
# 适用于哪些版面标签。页眉不算这个意义上的标题，图注的宽度也确实该由它的图决定。
_LABELS = frozenset({"title", "document_title"})

# Points left between the widened box and whatever stops it, so a title never
# runs right up against the element that bounded it.
# 加宽后的框与拦住它的东西之间留出的点数，免得标题紧贴着限制它的元素。
_CLEARANCE = 2.0


def _one_line_width(units, scale: float) -> float:
    """
    中文：把全部排版单元排成一行需要多宽。

    Measure how wide the units would be set all on one line.

    Args:
        units (list): The paragraph's typesetting units.
        scale (float): The scale they would be set at.

    Returns:
        float: Their total width in points.
    """
    return sum(unit.width for unit in units) * scale


def install() -> None:
    """
    中文：排版标题之前，若右边的空地足够放下一整行，就把框加宽到那么宽。

    Widen a title's box to the page's free width when one line then fits.

    Wraps ``_get_optimal_scale``, not the later ``render_paragraph``: the scale
    every paragraph is finally set at is chosen in an earlier pass over the whole
    document, so a box widened at render time would come too late to undo a
    reduction the narrow box had already caused. The widened box persists from
    here into the rendering.

    The box is only ever widened, and only when the whole title then fits on a
    single line.

    Idempotent, so a second call is harmless.

    Raises:
        AttributeError: If BabelDOC no longer exposes the stage, so an upgrade
            that moves it fails at startup rather than silently leaving titles
            to wrap for the sake of one character.
    """
    from babeldoc.format.pdf.document_il.midend.typesetting import Typesetting

    if getattr(Typesetting._get_optimal_scale, "_titlebox_patched", False):
        return

    original = Typesetting._get_optimal_scale

    def _get_optimal_scale(self, paragraph, page, typesetting_units,
                           initial_scale=1.0, *args, **kwargs):
        """中文：需要且放得下才加宽，然后照常求缩放。 / Widen if it helps, then proceed."""
        try:
            _widen(self, paragraph, page, typesetting_units, initial_scale)
        except Exception:  # noqa: BLE001
            # A title that cannot be measured keeps its box, which is how every
            # title behaved before this existed.
            # 量不了的标题保留原框，也就是本模块存在之前所有标题的行为。
            logger.warning("Could not widen a title's box", exc_info=True)
        return original(self, paragraph, page, typesetting_units, initial_scale,
                        *args, **kwargs)

    def _widen(self, paragraph, page, units, scale) -> None:
        """
        中文：判断该标题是否值得加宽，值得就改框。

        Widen this paragraph's box when doing so puts the title on one line.

        Args:
            self (Typesetting): The typesetting stage.
            paragraph (PdfParagraph): The paragraph about to be measured.
            page (Page): Its page, used to find the free space to the right.
            units (list): The paragraph's typesetting units.
            scale (float): The scale the search is starting from.
        """
        if getattr(paragraph, "layout_label", None) not in _LABELS:
            return
        box = getattr(paragraph, "box", None)
        if box is None or not units:
            return

        needed = _one_line_width(units, scale)
        if needed <= box.x2 - box.x:
            return  # Already fits on one line. / 本来就放得下一行。

        limit = self.get_max_right_space(box, page) - _CLEARANCE
        if box.x + needed > limit:
            # Even the whole free width will not hold it. Leaving the box alone
            # keeps a long title's existing shape rather than half-changing it.
            # 就算把右边的空地全用上也放不下。保持原框，长标题的既有版式就不会被改成
            # 一个不上不下的样子。
            return

        from babeldoc.format.pdf.document_il import il_version_1

        logger.debug("Widened a title's box from x2=%.1f to %.1f to fit one line",
                     box.x2, box.x + needed)
        paragraph.box = il_version_1.Box(
            x=box.x, y=box.y, x2=box.x + needed, y2=box.y2)

    _get_optimal_scale._titlebox_patched = True
    Typesetting._get_optimal_scale = _get_optimal_scale
