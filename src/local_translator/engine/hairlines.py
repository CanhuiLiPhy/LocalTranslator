"""
中文：分数线与根号线是零高度的描边路径，按面积判归属永远算不进公式，于是留在原地不动。

Keep a formula's rules with the formula they belong to.

**The symptom.** Fraction bars and radical bars land in the wrong place — under
Chinese prose halfway down the page, floating in the right margin, drawn across
a line that has nothing to do with them. Frequent enough to affect most pages of
a mathematical paper: measured on arXiv:2608.13520, every inline ``\\frac`` and
``\\sqrt`` on pages 11, 15 and 23.

**Why.** A rule in a LaTeX PDF is a *stroked* path — a horizontal line with no
thickness of its own, painted with a line width. Its bounding box is therefore
zero points high. BabelDOC decides which formula a curve belongs to by
containment, and containment is measured as intersection over the element's own
area:

    if x_right <= x_left or y_top <= y_bottom:
        return 0.0
    ...
    if first_box_area <= 0:
        return 0.0

A zero-height box takes both of those exits, so the score is 0 against a
threshold of 0.95 and the rule is never assigned. The formula's *characters* are
carried along when the paragraph reflows; the rule stays in ``page.pdf_curve``
and is drawn at the coordinates it had in the English original.

**The fix.** Give a degenerate box a hairline's worth of thickness before the
containment test, and only for that test. A rule 14 points long and 0 high
becomes 14 by :data:`_HAIRLINE`, which is entirely inside the formula that drew
it, so it scores 1.0 and travels with it. Nothing else sees the padded box: the
curve is stored, relocated and painted with its own geometry throughout.

**Why this cannot catch something else.** Containment is a strict test — the
element has to sit inside the formula's box. A page-wide rule under a heading, a
table's ruling line and a footnote separator are all far wider than any formula
on their line, so none of them is contained in one whatever their height. What
the padding admits is exactly the class it was written for: a short horizontal
line inside a formula's own bounding box.

**What it does not reach on its own.** A fraction whose numerator and
denominator BabelDOC made into *two* formulas leaves its bar between them and
inside neither, so no containment test can claim it. That is not a scoring
problem and is not solved here: :mod:`local_translator.engine.formulas` joins
the two formulas back into one first, and the padding below then claims the bar
for the formula that results. The two are only useful together.

**How this is measured.** By rendering, not by geometry.
``page.get_drawings()`` reports every path the page references, including the
ones inside the clipped full-page XObjects that the region stamping leaves
behind — geometry that is never painted. Counting those made a clean page read
as a broken one, and made this module look far more effective than it was.
``scripts/audit-rules.py`` renders each candidate and keeps only the rules whose
ink is actually on the page. On the 45 pages of arXiv:2608.13520 it counts 378
visible rules in the English original, of which 80 sit over nothing — a diagram
on page 8 draws dozens of short rules by design. Against that baseline the
translated output carried 21 further stranded rules before the formula joining
and the overhang allowance, and 16 after. Rendering those 16 shows several to be
bars that are exactly where they belong and that the audit mis-reads, so the
real residue is smaller than the number — see that script on why it errs
towards reporting.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Thickness given to a degenerate box for the containment test, in points. A
# hairline is 0.4pt at its thinnest and the formula box has a tolerance of 2pt
# around it, so anything in this range behaves the same; half a point is chosen
# for being unambiguously non-zero while staying below the tolerance.
# 为退化的框在归属判定中补上的厚度（点）。最细的发丝线是 0.4pt，而公式框本身有 2pt 的
# 容差，因此这个量级内取值效果一致；取半点，既确定不是零，又小于容差。
_HAIRLINE = 0.5

# How far a rule may reach past the formula it belongs to, in points.
#
# A rule that decorates or divides a formula is drawn slightly longer than the
# thing it sits on — that is what makes it read as a rule rather than as part of
# the glyph. The overline of 「q̄」 on page 9 of arXiv:2608.13520 spans 4.7 points
# where the formula's own ink spans 11.2, and 2.5 of those points stick out past
# its right edge: containment scores 0.46 against a threshold of 0.95, so the
# overline is never claimed and stays where the English had it.
#
# Allowing the formula three points of slack — about a side bearing at body size
# — admits the decoration and nothing else. A table's ruling line or a footnote
# separator is an order of magnitude longer than any formula on its line, so
# three points does not bring it near the threshold.
# 一条线可以比它所属的公式伸出多少（点）。
#
# 装饰或分隔公式的线，画出来总比它所依附的东西略长——正是这一点让它读起来像一条线而不
# 是字形的一部分。arXiv:2608.13520 第 9 页上「q̄」的上划线宽 4.7 点，而公式自身的墨迹
# 宽 11.2 点，其中 2.5 点伸到了公式右边界之外：归属得分 0.46，阈值 0.95，于是这条上划
# 线从不被认领，留在英文原稿的位置上。
#
# 给公式框三个点的余量——大约是正文字号下的一个边距——正好收下这类装饰，别的收不进来。
# 表格的横线或脚注分隔线比同一行上的任何公式长一个数量级，三个点不会让它接近阈值。
_OVERHANG = 3.0

# How much a rule's score may be reduced for sitting far from a formula's top
# edge, and the distance over which that reduction is spread.
#
# Two formulas stacked one line apart both contain a rule that lies between
# them, once the padding and the overhang above are applied. Both then score a
# flat 1.0, the assignment picks whichever the page happened to list first, and
# half the time the rule travels with the wrong one: on page 14 of
# arXiv:2608.11612 the vinculum of 「η√n」 went with the formula on the line above
# and landed twenty-two points to the left of the 「n」 it belongs over, leaving a
# bare radical and a stray bar across 「M)」.
#
# A vinculum is drawn from the top of its own formula, so distance from that top
# edge is exactly the evidence needed to break the tie. The reduction is kept
# small on purpose: the same score gates containment at 0.95, so it must never
# push a genuinely contained rule below that. Four hundredths leaves the whole
# range inside [0.96, 1.0] — enough to order two candidates, never enough to
# disqualify one.
# 一条线因为离公式顶边较远，其得分最多可以被压低多少，以及这个压低分摊在多长的距离上。
#
# 相隔一行、上下叠放的两条公式，在上面的补厚与外扩之后，都会「包含」夹在它们中间的那条
# 线。两者于是同得 1.0，归属就取页面上先列出的那一条，有一半的机会跟错：arXiv:2608.11612
# 第 14 页上，「η√n」的横线跟着上一行的公式走了，落在它本该覆盖的「n」左边二十二点处，留下
# 一个光秃秃的根号，和一条横在「M)」上的游离横线。
#
# 根号横线是从它自己那条公式的顶端画出去的，因此「离顶边多远」正是打破平局所需要的证据。
# 压低幅度刻意取得很小：同一个得分还在 0.95 处充当包含判定的闸门，绝不能把一条真正被包含
# 的线压到闸门以下。四个百分点把整个区间留在 [0.96, 1.0] 内——足以给两个候选排序，永远不
# 足以让谁出局。
_TOP_BIAS = 0.04
_TOP_SPREAD = 300.0

# Set ``LOCALTRANS_RULE_PROBE`` to a curve's x coordinate to have every
# containment score for that curve written to the log. Nothing is computed when
# it is unset. This is how each of the rules in this module's docstring was
# diagnosed, and it is left in because the next one will be diagnosed the same
# way.
# 把 LOCALTRANS_RULE_PROBE 设成某条线的 x 坐标，即可把该线的每一次归属打分写进日志。
# 不设置时不做任何计算。本模块文档里的每一条都是这样查出来的，留着是因为下一条也会这样查。
_PROBE = float(_probe) if (_probe := __import__("os").environ.get(
    "LOCALTRANS_RULE_PROBE")) else None

def _padded(box):
    """
    中文：把零宽或零高的框补成有厚度的框，只用于判定，不改原对象。

    Return a box with no zero-length side, for the containment test alone.

    Args:
        box (Box | None): The element's own bounding box.

    Returns:
        Box | None: ``box`` itself when both sides already have length, a padded
            copy when either does not, and ``None`` for ``None``.
    """
    if box is None:
        return box
    flat_x = box.x2 - box.x < _HAIRLINE
    flat_y = box.y2 - box.y < _HAIRLINE
    if not (flat_x or flat_y):
        return box

    from babeldoc.format.pdf.document_il.il_version_1 import Box

    half = _HAIRLINE / 2
    return Box(
        x=box.x - half if flat_x else box.x,
        y=box.y - half if flat_y else box.y,
        x2=box.x2 + half if flat_x else box.x2,
        y2=box.y2 + half if flat_y else box.y2,
    )


def _slack(box):
    """
    中文：把公式框放大一点，容下比它略长的那条线。

    Return the formula's box with room for a rule that overhangs it.

    Args:
        box (Box | None): The formula's own box.

    Returns:
        Box | None: A copy grown by :data:`_OVERHANG` on every side, or ``None``
            for ``None``. Only ever used to score a flat element — see
            :data:`_OVERHANG` for why that is safe.
    """
    if box is None:
        return box

    from babeldoc.format.pdf.document_il.il_version_1 import Box

    return Box(x=box.x - _OVERHANG, y=box.y - _OVERHANG,
               x2=box.x2 + _OVERHANG, y2=box.y2 + _OVERHANG)


def install() -> None:
    """
    中文：让公式归属的打分把零高度的线也算进去。

    Make the curve-to-formula containment score see a hairline.

    The whole assignment — the exact predicate, the tolerant one, and the score
    that ranks the candidates — is built on one function,
    ``calculate_iou_for_boxes``, which divides the intersection by the *first*
    box's own area. Patching that one function is therefore enough, and is
    better than patching the predicates: a predicate that said yes while the
    score still said zero would leave the curve with a candidate it could never
    win with.

    The padded box is used for the arithmetic and nowhere else. It is rebound on
    the two modules that do the assigning rather than on the module that defines
    it, so any other caller keeps the original.

    Idempotent, so a second call is harmless.

    Raises:
        AttributeError: If BabelDOC no longer exposes the function on either
            module, so an upgrade that moves it fails at startup rather than
            silently leaving every fraction bar behind.
    """
    from babeldoc.format.pdf.document_il.midend import styles_and_formulas as stage
    from babeldoc.format.pdf.document_il.utils import spatial_analyzer

    if getattr(stage.calculate_iou_for_boxes, "_hairlines_patched", False):
        return

    original = stage.calculate_iou_for_boxes

    def calculate_iou_for_boxes(box1, box2):
        """中文：零厚度的框补厚，并允许它比公式伸出一点。 / Pad the rule, allow overhang."""
        padded = _padded(box1)
        if padded is box1:
            return original(box1, box2)
        score = original(padded, _slack(box2))
        # Only fully contained candidates are ranked this way; a partial score
        # is already telling the assignment what it needs to know.
        # 只对完全被包含的候选做这种排序；部分包含的得分本身已经说明了问题。
        if score >= 0.99 and box2 is not None:
            distance = abs((box1.y + box1.y2) / 2 - box2.y2)
            score *= 1.0 - min(_TOP_BIAS, distance / _TOP_SPREAD)
        if _PROBE is not None and box1 is not None and abs(box1.x - _PROBE) < 1.0:
            logger.warning("RULE %.1f,%.1f w=%.1f vs formula %.1f,%.1f-%.1f,%.1f = %.3f",
                           box1.x, box1.y, box1.x2 - box1.x,
                           box2.x, box2.y, box2.x2, box2.y2, score)
        return score

    calculate_iou_for_boxes._hairlines_patched = True
    stage.calculate_iou_for_boxes = calculate_iou_for_boxes
    spatial_analyzer.calculate_iou_for_boxes = calculate_iou_for_boxes
    logger.debug("Hairline rules will now be assigned to the formulas that drew them")
