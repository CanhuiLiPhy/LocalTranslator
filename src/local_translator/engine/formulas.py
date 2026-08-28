"""
中文：把被切碎的行内公式在中间表示层重新拼成一个整体，再交给后面的每一步。

Put a shattered inline formula back together before anything else sees it.

**The symptom, three ways.** BabelDOC cuts an inline formula wherever its own
segmentation finds a boundary, and every stage downstream then treats the pieces
as unrelated:

* **The translator sees a list.** ``Y_ℓ/2τ ∈ [0,1]`` arrives as
  ``{v3}({v4}){v5}({v6}) {v7}`` — five placeholders and some punctuation — and a
  translation model reads five opaque tokens in a row as an enumeration:
  「{v2}、{v3}、{v4}、{v5}、{v6}」. The formula is printed with ideographic commas
  through the middle of it.
* **The fraction bar is orphaned.** A ``\\frac`` whose numerator and denominator
  became two formulas leaves its rule between them and inside neither, so no
  containment test can claim it. The characters move with the reflowed
  paragraph; the rule stays at the coordinates the English original had.
* **A citation is torn in half.** ``[Theorem 4, MP09]`` reaches the translator as
  ``{v1}Theorem 4 MP{v2}`` — the brackets are formulas, the words between them
  are text — and the model is free to put the two halves in different sentences.
  It does: 「定理4 MP{v1}中提出的…变量{v2}」.

All three are the same defect seen from three stages. Repairing it once, here,
fixes all of them: the placeholder count drops because there is one formula, the
bar is contained because the box now spans the whole fraction, and the citation
cannot be split because it is a single unit.

**When this runs.** Immediately after BabelDOC's own
``merge_overlapping_formulas`` and before ``collect_contained_elements``, which
is the stage that decides which curves belong to which formula. Merging first is
what lets that stage see a box big enough to contain the rule.

**What is merged.** Two formulas in the same paragraph join when either holds:

* **Nothing but glue between them** — spaces, brackets, operators, a comma;
  anything that is not a letter or a Chinese character. This is the ordinary
  case, and it is conservative: ``{v1} and {v2}`` is two formulas in a sentence
  and stays that way.
* **An unclosed bracket** — the text so far has an open ``(``, ``[`` or ``{``
  that the next formula closes. Words in between are then absorbed too, because
  a citation's words belong to the citation.

Both are bounded: the pieces must be near each other on the page, and a merged
formula stops growing at :data:`_MAX_CHARS`. A pass that could run away would
swallow a paragraph.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# Characters that may sit between two formulas without breaking them apart. In
# practice: whitespace, brackets, operators, a comma. Anything with a letter or
# a Chinese character in it is prose, and prose separates formulas.
# 可以夹在两个公式之间而不算把它们分开的字符。实际上就是：空白、括号、运算符、逗号。
# 只要含字母或汉字就是行文，而行文是把公式分开的。
_WORD = re.compile(r"[A-Za-zÀ-ɏ一-鿿]")
_WORDS = re.compile(r"[A-Za-zÀ-ɏ一-鿿]+")

# Operator names TeX sets in upright roman *inside* a formula: 「log」, 「max」,
# 「ker」, 「lim sup」. Being roman is exactly why they end up outside it — the
# formula detector reads the italic maths face as its signal, so an operator
# name is filed as ordinary prose and the formula around it is cut in two.
#
# 「\sqrt{d \log d}」 on page 45 of arXiv:2608.13520 arrives as a formula 「√d」,
# the text 「log」 and a formula 「d」. The radical's own bar spans all three and
# is therefore contained in none of them, so it is never assigned to a formula
# and stays at its English coordinates while the Chinese reflows out from under
# it: the output reads 「√d log  d, , ‾我们因此有」 with the bar stranded over the
# prose. Gluing the operator back in makes one formula whose box covers the bar.
#
# Only these exact names, and only when the gap between two formula pieces is
# nothing but them. 「log」 between two formulas is an operator; 「log of the」 is
# a sentence.
# TeX 在公式内部用正体罗马排的算子名。正体恰恰就是它们被排除在公式之外的原因——公式识别
# 以斜体数学字面为信号，于是算子名被当作普通正文，它两侧的公式也被切成两半。
# arXiv:2608.13520 第 45 页的 \sqrt{d \log d} 就切成了公式「√d」、文字「log」、公式
# 「d」三段，根号的横线横跨三者、哪一段都不包含它，于是永远分配不到公式上，中文重排之后
# 横线仍留在英文原处，输出成了「√d log  d, , ‾我们因此有」。把算子名粘回去，合成的这一条
# 公式的框就盖住了这根横线。
# 只认这些名字，且只在两段公式之间除它们之外别无他物时才算数：两个公式之间的「log」是算
# 子，「log of the」是句子。
_OPERATORS = frozenset("""
    arccos arcsin arctan arg cos cosh cot coth csc deg det dim exp gcd hom inf
    injlim ker lg lim liminf limsup ln log max min mod bmod pmod projlim sec sin
    sinh sup tan tanh tr rank sgn diag span erf argmax argmin
""".split())


def _operators_only(text: str) -> bool:
    """
    中文：判断一段文字是否只由数学算子名组成。

    Decide whether a gap between two formula pieces is nothing but operator names.

    Args:
        text (str): The text lying between two formula fragments.

    Returns:
        bool: ``True`` when it contains at least one word and every word in it
            is in :data:`_OPERATORS`. Digits, spaces and punctuation around them
            are ignored — they are already glue by the letter test.
    """
    words = _WORDS.findall(text)
    return bool(words) and all(word.lower() in _OPERATORS for word in words)

# Brackets whose halves belong to the same semantic unit.
# 两半属于同一个语义单元的括号。
_OPENERS = "([{"
_CLOSERS = ")]}"

# How far apart two pieces may be and still be one formula, in points. Generous
# enough for a fraction's numerator and denominator, which are a line apart, and
# far short of the distance to the next formula in a sentence.
# 两块相距多远仍算作一个公式（点）。足够容纳分数的分子与分母——它们相隔一行——又远小于
# 一个句子里到下一个公式的距离。
_NEAR = 28.0

# Set LOCALTRANS_GAP_PROBE=1 to log the text between every pair of formula
# pieces, and which of the three gates below turned it away. Nothing is computed
# when it is unset.
#
# Both faults this module carries a fix for were found this way and could not
# have been found any other: the log line
# 「GAP 'log ' after 'c1/4)√d' before 'd' glued=True」 said the operator rule was
# working, and the 「STOP not near」 on the very next line said the run was still
# measuring from the wrong place. Reading the output PDF shows only that a bar
# is in the wrong spot, never which gate put it there.
# 设 LOCALTRANS_GAP_PROBE=1 可把每两段公式之间的文字、以及是下面三道关卡中的哪一道把它
# 挡回去，都写进日志。不设置时不做任何计算。
#
# 本模块修的两处毛病都是这样查出来的，也只能这样查：日志里
# 「GAP 'log ' after 'c1/4)√d' before 'd' glued=True」说明算子规则已经放行，紧接着的
# 「STOP not near」说明这一串仍在按错误的位置量距离。看产物 PDF 只能看出横线摆错了，
# 看不出是哪一道关卡摆错的。
_GAPS = bool(__import__("os").environ.get("LOCALTRANS_GAP_PROBE"))

# Set LOCALTRANS_RULE_EXTENT=1 to log, for every paragraph, each formula's box
# beside any flat curve that begins near its right edge.
#
# Every constant in the vinculum rule below was read off this probe's output.
# The numbers it printed for page 15 of arXiv:2608.20061 —
#   formula box=(115.5,362.5)-(137.5,374.4) | rule x=137.5-183.6 y=374.2
#   gap=-0.1 dy_top=-0.2
# are what say a vinculum touches its formula's top-right corner, which is the
# whole basis for telling one apart from every other rule on a page. Reading
# the output PDF shows only that a bar is in the wrong place.
# 设 LOCALTRANS_RULE_EXTENT=1，可为每个段落打印每条公式的框，以及紧邻其右边界起始的扁平
# 曲线。
#
# 下面根号规则里的每一个常数都是从这个探针的输出上读出来的。它为 arXiv:2608.20061 第 15
# 页打出的那几个数——公式框 (115.5,362.5)-(137.5,374.4)、横线 x=137.5-183.6 y=374.2、
# gap=-0.1、dy_top=-0.2——正是「根号横线贴着公式右上角起笔」这一判据的全部依据，而这一判
# 据又是把它与页面上其他任何线区分开的唯一办法。看产物 PDF 只能看出横线摆错了。
_EXTENT = bool(__import__("os").environ.get("LOCALTRANS_RULE_EXTENT"))
# Set LOCALTRANS_PLACE_PROBE to a substring of a formula's text to have every
# placement of that formula logged: its own box, the position and scale the
# layout engine chose, and its offsets.
#
# This is what identified the second fault this module fixes. The output PDF
# showed a tangle of overprinted characters and suggested a baseline error; the
# probe showed
#   PLACE 'µP)(Yang & Hu,2021' box=(108.2,374.8)-(296.9,393.2)
# — a formula 18.4 points tall, which is two lines — and the real fault was the
# merge that built it, not the placement of it.
# 把 LOCALTRANS_PLACE_PROBE 设成某条公式文字的一个子串，即可把该公式的每一次落位写进
# 日志：它自己的框、排版引擎选定的位置与缩放，以及它的偏移量。
#
# 本模块修的第二个毛病就是这样认出来的。产物 PDF 上只看到一团互相压住的字符，像是基线算
# 错了；而探针打出的是
#   PLACE 'µP)(Yang & Hu,2021' box=(108.2,374.8)-(296.9,393.2)
# ——一条高 18.4 点、即两行高的公式——真正的毛病在造出它的那次合并，不在把它摆到哪里。
_PLACE = __import__("os").environ.get("LOCALTRANS_PLACE_PROBE")

# The same bound for the bracket rule, where the two halves of a citation are a
# phrase apart rather than a space apart: 「[」 and 「09]」 in 「[Theorem 4, MP09]」
# sit about 55 points from one another, and a citation that wraps puts them on
# two lines. Generous horizontally, still tight vertically, and both far short of
# the width of a column.
# 括号规则用的同一组界限。引用的两半之间隔的是一个短语而不是一个空格：
# 「[Theorem 4, MP09]」里的「[」与「09]」相距约 55 点，换行的引用还会把它们分到两行。
# 横向放宽、纵向仍然收紧，两者都远小于一栏的宽度。
_NEAR_BRACKET_X = 190.0

# How many words of ordinary prose the bracket rule may swallow in one run.
#
# An open bracket says the run is still inside a unit, so the rule lets it take
# words as well as formulas until the closer arrives. That is right for the unit
# it was written for and catastrophic without a bound: prose is full of
# parentheses, the fragment boundaries cut through them, and a count that never
# returns to zero swallows sentence after sentence. On page 19 of
# arXiv:2608.20061 it made 「8K (Cobbe et al., 2021)), and Code (MBPP (Austin et
# al., 2021), HumanEval …」 — a line of citations 396 points wide — into a single
# formula. A formula is kept verbatim and set in the source's own glyphs, so
# that line was never translated *and* was laid out as if it were a symbol: 60
# points of x-advance lost and its baseline 7 points below the Chinese it landed
# on. The page reads as two paragraphs printed on top of each other.
#
# The bound is in characters, and counted across the whole run rather than per
# gap. Characters because words cannot be counted here: the text arrives from
# the PDF with most of its spaces missing, so 「promptsper benchmark
# (exceptforHumanEval, whereall」 — a whole clause — reads as four words and
# slips under any word bound worth having. Across the run because every
# individual gap in the bad case is short on its own; it is their sum that runs
# away.
#
# Measured on both: 「[Theorem 4, MP09]」 needs ten characters (「Theorem 」 and
# 「MP」) and spans 79 points; the citation lines need fifty in a single gap
# and span four hundred. Twenty-four keeps the first — with room for
# 「[see Theorem 4.1 in MP09]」 and its like — and stops the second on its first
# gap.
# 括号规则在一整串里最多可以吞下多少个词的普通文字。
#
# 左括号一旦打开就说明这一串还在某个单元内部，于是规则允许它连文字一起吞，直到闭合。对它
# 本来要保住的那种单元这是对的，没有上限则是灾难：散文里到处是括号，切分边界又会把括号切
# 断，一个永远回不到零的计数会把句子一句接一句吞进去。arXiv:2608.20061 第 19 页上，它把
# 「8K (Cobbe et al., 2021)), and Code (MBPP (Austin et al., 2021), HumanEval …」
# ——一整行宽 396 点的引文——并成了一条公式。公式一律原样保留、用源文档自己的字形排，于是
# 那一行既没被翻译，又被当成一个符号来摆放：x 少进了 60 点，基线落在它压住的中文下方 7 点。
# 整页读起来就是两段文字印在了一起。
#
# 上限按字符数计，且按整串累计而不是逐个间隙计。按字符是因为这里根本数不了词：文字从 PDF
# 里提出来时大部分空格是缺的，「promptsper benchmark (exceptforHumanEval, whereall」——
# 一整个从句——只能数出四个词，任何像样的词数上限都拦不住它。按整串累计是因为坏例子里每一
# 个间隙单看都不长，跑掉的是它们的和。
#
# 两边都量过：「[Theorem 4, MP09]」需要十个字符（「Theorem 」与「MP」）、宽 79 点；那几行引文
# 单是第一个间隙就要五十个字符、宽四百点。取二十四，收得下前者（并给
# 「[see Theorem 4.1 in MP09]」这类留有余地），又能在后者的第一个间隙上就把它拦住。
_BRACKET_CHARS = 24
_NEAR_BRACKET_Y = 34.0

# A merged formula stops growing here. An inline formula is short; something
# this long is a runaway merge, and stopping leaves the ordinary segmentation in
# place rather than damaging the paragraph.
# 合并到这个长度就停。行内公式很短；长到这个地步说明合并跑飞了，就此停手会保留原有的
# 切分方式，而不是把整段弄坏。
_MAX_CHARS = 240



def _characters(composition) -> list:
    """
    中文：取出一个组成单元里的字符，不管它是哪一种形状。

    List the characters a paragraph composition holds, whichever shape it is.

    Args:
        composition (PdfParagraphComposition): One element of a paragraph.

    Returns:
        list: Its ``PdfCharacter`` objects, in order, empty when it holds none.
    """
    for name in ("pdf_formula", "pdf_line", "pdf_same_style_characters"):
        holder = getattr(composition, name, None)
        if holder is not None:
            return list(holder.pdf_character or [])
    char = getattr(composition, "pdf_character", None)
    return [char] if char is not None else []


def _text(characters) -> str:
    """
    中文：把一串字符拼成文字。

    Join a run of characters into text.

    Args:
        characters (list): ``PdfCharacter`` objects.

    Returns:
        str: Their Unicode, in order.
    """
    return "".join(char.char_unicode or "" for char in characters)


def _open_brackets(text: str) -> int:
    """
    中文：数出还没闭合的左括号个数。

    Count the brackets a piece of text opens and does not close.

    Args:
        text (str): Text to scan.

    Returns:
        int: How many openers are still waiting for their closer. Never
            negative — a stray closer belongs to something outside this text.
    """
    depth = 0
    for char in text:
        if char in _OPENERS:
            depth += 1
        elif char in _CLOSERS and depth:
            depth -= 1
    return depth


def _union(first, second):
    """
    中文：求两个框的外接框，用来跟踪一串公式已经长到哪里。

    Return the smallest box containing both.

    Args:
        first (Box | None): One box.
        second (Box | None): The other.

    Returns:
        Box | None: Their union, or whichever one is not ``None``.
    """
    if first is None:
        return second
    if second is None:
        return first

    from babeldoc.format.pdf.document_il.il_version_1 import Box

    return Box(x=min(first.x, second.x), y=min(first.y, second.y),
               x2=max(first.x2, second.x2), y2=max(first.y2, second.y2))


def _near(first, second, *, bracketed: bool = False) -> bool:
    """
    中文：判断两个框是否近到可以属于同一条公式。

    Decide whether two boxes are close enough to be one formula.

    Args:
        first (Box | None): Box of the piece on the left.
        second (Box | None): Box of the piece on the right.
        bracketed (bool): Whether the two are the halves of a bracketed unit,
            which are allowed to be a phrase apart — see
            :data:`_NEAR_BRACKET_X`.

    Returns:
        bool: ``True`` when the gap in both axes is within the bound. A missing
            box answers ``False``, because a merge that cannot be checked must
            not happen.
    """
    if first is None or second is None:
        return False
    gap_x = max(second.x - first.x2, first.x - second.x2, 0.0)
    gap_y = max(second.y - first.y2, first.y - second.y2, 0.0)
    if bracketed:
        # The bracket rule may reach across a phrase, never across a line break.
        #
        # A formula is laid out as one inline unit, so one that spans two source
        # lines cannot be placed: it is as tall as two lines and as wide as one,
        # and wherever the engine puts it, it overprints its neighbours. On page
        # 2 of arXiv:2608.20061 this rule merged 「µ」 on one line with 「2021」 on
        # the next, across the fourteen characters 「P) (Yang & Hu,」 between
        # them, into a formula whose box was 188.7 by 18.4 points. It was then
        # placed at the start of a line that already had Chinese on it, and the
        # two printed on top of each other.
        #
        # Requiring the two boxes to overlap vertically is what confines the
        # rule to one line. 「[Theorem 4, MP09]」, the unit it was written for,
        # sits on one line — its box is ten points tall — so it is unaffected. A
        # citation that does wrap stays in two pieces, which reads correctly
        # even when it is not ideal.
        # 括号规则可以跨越一个短语，绝不可以跨越换行。
        #
        # 公式是作为一个行内单元排版的，因此横跨源文两行的公式根本没法摆：它有两行那么高、
        # 一行那么宽，引擎放到哪里都会压住旁边的东西。arXiv:2608.20061 第 2 页上，这条规则
        # 把一行的「µ」和下一行的「2021」——中间隔着「P) (Yang & Hu,」十四个字符——合成了一
        # 条 188.7 × 18.4 点的公式，然后被摆到一行已经有中文的位置上，两者印在了一起。
        #
        # 要求两个框在纵向上有重叠，就把这条规则限制在了一行之内。它本来要保住的
        # 「[Theorem 4, MP09]」在同一行上、框高十点，不受影响。真正换行的引用会保持两段，
        # 虽不理想，但读起来是对的。
        return gap_x <= _NEAR_BRACKET_X and gap_y <= 0.0
    return gap_x <= _NEAR and gap_y <= _NEAR


def _mergeable(characters) -> bool:
    """
    中文：给缺少视觉框的字符补一个，让它们能并入公式。

    Make sure every character carries the visual box the merge will need.

    ``update_formula_data`` computes the merged box from every character's
    ``visual_bbox``. A character that has always been part of a formula has one;
    a character taken out of a text line often does not, and refusing the merge
    on that ground is what kept 「[Theorem 4, MP09]」 in two halves — the words
    between the brackets are ordinary text.

    Where it is missing, the layout box stands in. For a text character the two
    are the same rectangle to within the glyph's own side bearings, which is far
    finer than anything the containment tests measure.

    Args:
        characters (list): Characters about to be absorbed. Modified in place.

    Returns:
        bool: ``True`` when every character now has a usable visual box.
    """
    from babeldoc.format.pdf.document_il.il_version_1 import Box, VisualBbox

    for char in characters:
        visual = getattr(char, "visual_bbox", None)
        if visual is not None and getattr(visual, "box", None) is not None:
            continue
        box = getattr(char, "box", None)
        if box is None:
            return False
        char.visual_bbox = VisualBbox(box=Box(x=box.x, y=box.y, x2=box.x2, y2=box.y2))
    return True


# How close a rule's left edge must be to a formula's right edge to be that
# formula's, in points. A vinculum is drawn from the top of the radical sign, so
# the two touch: measured on page 15 of arXiv:2608.20061 the gap is -0.1.
# 一条线的左端要离公式右边界多近，才算这条公式的（点）。根号的横线是从根号顶端画出去的，
# 两者相接：在 arXiv:2608.20061 第 15 页上量到的间隙是 -0.1。
_TOUCH = 2.0

# And how close to the formula's top edge, in points. Same measurement: -0.2.
# Both bounds together are what separates a vinculum from every other rule on a
# page. A table's ruling line does not begin at a formula's right edge; a
# fraction bar inside a formula is already contained and never reaches here; an
# underline sits below the text, not level with its top.
# 以及要离公式顶边多近（点）。同一次测量：-0.2。正是这两个界限合在一起，把根号横线与页面
# 上其他任何线区分开：表格线不会正好从某条公式的右边界起笔，公式内部的分数线本来就已被包
# 含、根本走不到这里，而下划线在文字下方、不与其顶端齐平。
_TOP = 2.5

# Widest rule that can be a radical's vinculum, in points. The one measured is
# 46. A rule wider than this spans something that is not an inline expression.
# 可以是根号横线的最宽值（点）。量到的那条是 46。比这更宽的线，跨的就不是一个行内表达式了。
_MAX_RULE = 150.0


def _flat_rules(page) -> list:
    """
    中文：取出页面上所有可能是根号横线的扁平曲线。

    Collect the page's flat curves that could be a vinculum.

    Args:
        page (Page): The page being processed.

    Returns:
        list: Curves with a box, no measurable height, and a width in the range
            an inline radical can span.
    """
    return [c for c in (page.pdf_curve or [])
            if c.box and (c.box.y2 - c.box.y) < 1.0
            and 3.0 < (c.box.x2 - c.box.x) <= _MAX_RULE]


def _vinculum_for(formula, rules):
    """
    中文：找出属于这条公式的根号横线。

    Find the rule that belongs to this formula, if any.

    Args:
        formula (PdfFormula): Formula to test.
        rules (list): Candidates from :func:`_flat_rules`.

    Returns:
        PdfCurve | None: The curve starting at the formula's top-right corner.
    """
    box = formula.box
    if box is None:
        return None
    for curve in rules:
        if (abs(curve.box.x - box.x2) <= _TOUCH
                and abs(curve.box.y - box.y2) <= _TOP
                and curve.box.x2 > box.x2):
            return curve
    return None


def _under_rule(characters, curve, formula_box) -> int:
    """
    中文：数出这一串里有多少个前缀字符落在横线底下。

    Count how many leading characters of a run sit under the rule.

    A character counts when its horizontal centre falls within the rule's span
    and it is on the same line as the formula. Counting by centre rather than by
    overlap stops the first character *after* the radicand — the comma in
    「√max(A, B),」 — from being pulled in with it.

    Args:
        characters (list): Characters of the composition following the formula.
        curve (PdfCurve): The vinculum.
        formula_box (Box): The formula's own box, used to stay on one line.

    Returns:
        int: How many characters from the front belong under the rule. Zero when
            the first character is already past it.
    """
    taken = 0
    for char in characters:
        if char.box is None:
            break
        middle = (char.box.x + char.box.x2) / 2
        if middle > curve.box.x2:
            break
        # Same line: the character's ink must overlap the formula's vertical
        # extent. Without this the line below, which starts at the same x, is
        # read as part of the radicand.
        # 同一行：字符的墨迹必须与公式的纵向范围有重叠。没有这一条，下一行——它起点的 x
        # 与本行相同——会被当成被开方数的一部分。
        if char.box.y2 < formula_box.y or char.box.y > formula_box.y2:
            break
        taken += 1
    return taken


def _extend_over_rules(stage, page, paragraph) -> int:
    """
    中文：让以根号收尾的公式把它的被开方数一起吞进来。

    Give a formula that ends in a radical the operand its rule spans.

    **The fault.** BabelDOC decides formula-ness one character at a time, from
    the font name and the unicode category, with no lookahead and no notion of
    an operand. 「0.2 · √max(A, B)」 therefore becomes the formula 「0.2 · √」 —
    the coefficient and the radical *sign* — followed by ordinary text, because
    「m」 is a Latin letter in a text font and ends the run. The vinculum spans
    「max(A, B)」, which is outside the formula, so no containment test can claim
    it: on page 15 of arXiv:2608.20061 the formula box ends at x 137.4 and the
    rule runs 137.5 to 183.6. The characters relocate with the paragraph and the
    rule stays where the English had it — a bare radical with nothing over it,
    and a stray bar lying across the Chinese two lines below.

    **The repair.** Read the rule as the statement of extent it is. A vinculum
    is drawn from the top of the radical sign across exactly what is under the
    root, so a flat curve that begins at a formula's top-right corner says
    precisely where that formula should have ended. Absorb the characters it
    covers and the formula is whole; the rule is then inside its box, gets
    assigned by the existing machinery, and travels.

    This is narrower than reconstructing LaTeX, and it is the part of that idea
    which the available information supports: the PDF says where the root ends
    because it drew a line there.

    Args:
        stage (StylesAndFormulas): BabelDOC's stage, for its ``update_formula_data``.
        page (Page): Page being processed, for its curves.
        paragraph (PdfParagraph): Paragraph to repair, modified in place.

    Returns:
        int: How many formulas were extended.
    """
    rules = _flat_rules(page)
    if not rules:
        return 0

    compositions = list(paragraph.pdf_paragraph_composition or [])
    extended = 0

    for index, composition in enumerate(compositions):
        formula = composition.pdf_formula
        if formula is None or formula.box is None:
            continue
        curve = _vinculum_for(formula, rules)
        if curve is None:
            continue

        following = compositions[index + 1] if index + 1 < len(compositions) else None
        if following is None or following.pdf_line is None:
            continue

        characters = list(following.pdf_line.pdf_character or [])
        taken = _under_rule(characters, curve, formula.box)
        if not taken:
            continue

        absorbed, remainder = characters[:taken], characters[taken:]
        if not _mergeable(absorbed):
            continue

        formula.pdf_character = list(formula.pdf_character or []) + absorbed
        try:
            stage.update_formula_data(formula)
        except Exception:  # noqa: BLE001
            # Put them back rather than leave a formula whose box is a lie.
            # 放回去，而不是留下一条框与内容不符的公式。
            formula.pdf_character = formula.pdf_character[:-taken]
            logger.debug("Could not recompute a formula extended over its rule",
                         exc_info=True)
            continue

        following.pdf_line.pdf_character = remainder
        if remainder:
            following.pdf_line.box = _union_of(remainder)
        else:
            compositions[index + 1] = None
        extended += 1
        logger.debug("Extended %r over its rule, taking %r",
                     _text(formula.pdf_character)[:24], _text(absorbed))

    if extended:
        paragraph.pdf_paragraph_composition = [c for c in compositions if c is not None]
    return extended


def _union_of(characters):
    """
    中文：算出一串字符的外接框。

    Return the box enclosing a run of characters.

    Args:
        characters (list): Characters with boxes.

    Returns:
        Box | None: Their union, or ``None`` when none has a box.
    """
    boxed = [c.box for c in characters if c.box is not None]
    if not boxed:
        return None

    from babeldoc.format.pdf.document_il.il_version_1 import Box

    return Box(x=min(b.x for b in boxed), y=min(b.y for b in boxed),
               x2=max(b.x2 for b in boxed), y2=max(b.y2 for b in boxed))


def _probe_extent(page, paragraph) -> None:
    """中文：打印公式框与近旁扁平曲线的几何。 / Log formula boxes beside nearby flat curves."""
    flats = [c for c in (page.pdf_curve or [])
             if c.box and (c.box.y2 - c.box.y) < 1.0 and 3 < (c.box.x2 - c.box.x) < 200]
    for comp in (paragraph.pdf_paragraph_composition or []):
        if comp.pdf_formula is None or comp.pdf_formula.box is None:
            continue
        fb = comp.pdf_formula.box
        text = _text(comp.pdf_formula.pdf_character or [])
        for c in flats:
            if abs(c.box.x - fb.x2) < 30 and abs(c.box.y - fb.y2) < 20:
                logger.warning(
                    "EXTENT formula %r box=(%.1f,%.1f)-(%.1f,%.1f) | rule x=%.1f-%.1f y=%.1f "
                    "| gap=%.1f dy_top=%.1f",
                    text[:30], fb.x, fb.y, fb.x2, fb.y2,
                    c.box.x, c.box.x2, c.box.y, c.box.x - fb.x2, c.box.y - fb.y2)


def _join(stage, paragraph) -> int:
    """
    中文：把一个段落里相邻的公式片段合并成整条公式。

    Join a paragraph's adjacent formula fragments into whole formulas.

    Walks the compositions once. A formula starts a run; the run swallows the
    pieces after it for as long as one of the two rules in this module's
    docstring holds, and is written back as a single formula.

    Args:
        stage (StylesAndFormulas): BabelDOC's stage, used for its own
            ``merge_formulas`` so the merged object is built the way every other
            formula on the page is.
        paragraph (PdfParagraph): Paragraph to repair, modified in place.

    Returns:
        int: How many merges were performed, for the run's log line.
    """
    from babeldoc.format.pdf.document_il.il_version_1 import (
        PdfFormula,
        PdfParagraphComposition,
    )

    compositions = list(paragraph.pdf_paragraph_composition or [])
    if len(compositions) < 2:
        return 0


    out: list = []
    merges = 0
    index = 0
    while index < len(compositions):
        head = compositions[index]
        if head.pdf_formula is None:
            out.append(head)
            index += 1
            continue

        taken = [head]
        # How far the run has grown so far. The distance to the next piece has
        # to be measured from here and not from ``head``: a run reads
        # left to right, so every piece it swallows puts the head's right edge
        # further from the piece after it, and a run long enough eventually
        # breaks on a gap that is not there. 「T− := k − (c1/4)√d log d」 broke
        # exactly that way — by the time 「d」 came up the run reached x=236 but
        # was still being measured from the head's x=219, making a 17-point gap
        # look like 34 and putting it over the bound. The radical bar spans the
        # whole thing, so the break left it belonging to nothing.
        # 这一串到目前为止长到了哪里。到下一段的距离必须从这里量，不能从 head 量：一串是
        # 从左往右读的，每吞进一段，head 的右边界离后面的段就更远一点，串够长时必然会在一
        # 个并不存在的间隙上断开。「T− := k − (c1/4)√d log d」正是这样断的——轮到「d」时
        # 这一串已经到 x=236，却仍按 head 的 x=219 去量，17 点的间隙被算成 34 点，超出
        # 上限。而根号的横线横跨整条，一断就无所属了。
        frontier = head.pdf_formula.box
        # Prose admitted by the bracket rule alone, counted across the run.
        # 仅凭括号规则收进来的文字，按整串累计。
        borrowed = 0
        characters = list(head.pdf_formula.pdf_character or [])
        curves = list(head.pdf_formula.pdf_curve or [])
        forms = list(head.pdf_formula.pdf_form or [])
        look = index + 1
        pending: list = []

        while look < len(compositions) and len(characters) < _MAX_CHARS:
            candidate = compositions[look]
            if candidate.pdf_formula is None:
                pending.append(candidate)
                look += 1
                continue

            gap = [char for entry in pending for char in _characters(entry)]
            gap_text = _text(gap)
            glued = not _WORD.search(gap_text) or _operators_only(gap_text)
            # A bracket that is open means the run is inside a unit, and
            # everything up to its closer belongs to that unit — words
            # included. 「[Theorem 4, MP09]」 arrives as five pieces, and the
            # closer is two formulas past the opener, so asking only whether the
            # *next* piece closes it never finds the end.
            # 括号一旦打开就说明这一串还在某个单元内部，直到闭合为止的一切都属于它，
            # 文字也不例外。「[Theorem 4, MP09]」是五段，闭合括号在开括号之后的第二个
            # 公式里，只问「下一段是否闭合」永远找不到结尾。
            # The bracket must already be open before this gap — opened by what
            # the run has taken so far, not by the gap's own last character.
            #
            # Counting the gap in too makes an opener at its END say the run is
            # inside a unit, when that opener belongs to the unit *after* it.
            # 「achieving the largest improvement (−0.027) at the largest scale
            # (14B).」 is a formula 「−0.027」, the twenty-three characters
            # 「) at the largest scale (」, and a formula 「14B」 — round brackets
            # are not formula characters, digits and 「−」 are. The trailing 「(」
            # opened a bracket that was never this run's, the gap slipped under
            # the character cap, and all three merged into one formula: never
            # translated, and printed with the brackets on either side of it
            # lost. That shape — a number in brackets, some words, another
            # number in brackets — is everywhere in a results section.
            #
            # 「[Theorem 4, MP09]」 is unaffected: its opener is 「[」, a formula
            # character, so it is among the run's own characters from the first
            # step.
            # 括号必须在这个间隙之前就已经打开——由这一串已经吞下的东西打开，而不是由间隙
            # 自己的最后一个字符打开。
            #
            # 把间隙一并计入，会让位于间隙**末尾**的左括号宣称「这一串还在某个单元内部」，
            # 而那个左括号属于它**后面**的单元。
            # 「achieving the largest improvement (−0.027) at the largest scale (14B).」
            # 切出来是：公式「−0.027」、二十三个字符的「) at the largest scale (」、公式
            # 「14B」——圆括号不是公式字符，而数字与「−」是。末尾那个「(」打开了一个根本
            # 不属于这一串的括号，间隙又恰好没超过字符上限，于是三段并成一条公式：既不会
            # 被翻译，印出来时两侧的括号还都丢了。而「括号里一个数、几个词、括号里另一个
            # 数」这种形状，结果章节里到处都是。
            #
            # 「[Theorem 4, MP09]」不受影响：它的开括号是「[」，本身就是公式字符，从第一步
            # 起就在这一串自己的字符里。
            inside = _open_brackets(_text(characters)) > 0
            if _GAPS:
                logger.warning("GAP %r after %r before %r glued=%s inside=%s",
                               gap_text, _text(characters)[-14:],
                               _text(candidate.pdf_formula.pdf_character or [])[:14],
                               glued, inside)
            if not (glued or inside):
                break
            if not glued:
                borrowed += len(gap_text)
                if borrowed > _BRACKET_CHARS:
                    if _GAPS:
                        logger.warning("  STOP bracket rule at %d characters", borrowed)
                    break
            if not _near(frontier, candidate.pdf_formula.box,
                         bracketed=inside and not glued):
                if _GAPS:
                    logger.warning("  STOP not near: %s vs %s",
                                   frontier, candidate.pdf_formula.box)
                break
            if not _mergeable(gap):
                if _GAPS:
                    logger.warning("  STOP not mergeable")
                break

            characters.extend(gap)
            characters.extend(candidate.pdf_formula.pdf_character or [])
            curves.extend(candidate.pdf_formula.pdf_curve or [])
            forms.extend(candidate.pdf_formula.pdf_form or [])
            frontier = _union(frontier, candidate.pdf_formula.box)
            taken.extend(pending)
            taken.append(candidate)
            pending = []
            look += 1
            merges += 1

        if len(taken) == 1:
            out.append(head)
            index += 1
            continue

        merged = PdfFormula(pdf_character=characters,
                            line_id=head.pdf_formula.line_id)
        merged.pdf_curve = curves
        merged.pdf_form = forms
        try:
            stage.update_formula_data(merged)
        except Exception:  # noqa: BLE001
            # A character the box calculation cannot use leaves the paragraph
            # exactly as BabelDOC segmented it.
            # 有字符算不出框时，该段落原样保留 BabelDOC 的切分结果。
            logger.debug("Could not rebuild a merged formula", exc_info=True)
            out.append(head)
            index += 1
            continue

        if _GAPS:
            logger.warning("MERGED %r box=%s", _text(characters)[:70], merged.box)
        out.append(PdfParagraphComposition(pdf_formula=merged))
        index += len(taken)
        # Anything trailing that was collected but never absorbed goes back.
        # 收集了但最终没并进去的尾巴要放回去。
        out.extend(pending)
        index += len(pending)

    paragraph.pdf_paragraph_composition = out
    return merges


def install() -> None:
    """
    中文：安装公式合并——在 BabelDOC 自己合并之后、给曲线定归属之前。

    Install the neighbour-merging pass.

    Wraps ``StylesAndFormulas.merge_overlapping_formulas``, which BabelDOC calls
    once per page between forming the formulas and assigning the curves to them.
    That is the only point where both are true: the formulas exist, and nothing
    has yet been decided on the basis of their boxes.

    Idempotent, so a second call is harmless.

    Raises:
        AttributeError: If BabelDOC no longer exposes the method, so an upgrade
            that renames it fails at startup instead of silently leaving every
            inline formula in pieces.
    """
    from babeldoc.format.pdf.document_il.midend.styles_and_formulas import (
        StylesAndFormulas,
    )

    if getattr(StylesAndFormulas.merge_overlapping_formulas,
               "_formulas_patched", False):
        return

    original = StylesAndFormulas.merge_overlapping_formulas

    def merge_overlapping_formulas(self, page):
        """中文：先跑原有合并，再合并相邻碎片。 / Merge overlaps, then neighbours."""
        result = original(self, page)
        merged = 0
        for paragraph in page.pdf_paragraph or []:
            try:
                if _EXTENT:
                    _probe_extent(page, paragraph)
                merged += _extend_over_rules(self, page, paragraph)
                merged += _join(self, paragraph)
            except Exception:  # noqa: BLE001
                logger.debug("Neighbour merging failed for a paragraph", exc_info=True)
        if merged:
            logger.debug("Joined %d formula fragment(s) on page %s",
                         merged, getattr(page, "page_number", "?"))
        return result

    if _PLACE:
        from babeldoc.format.pdf.document_il.midend import typesetting as _ts

        _relocate = _ts.TypesettingUnit.relocate

        def relocate(self, x, y, scale):
            """中文：打印公式单元的落位参数。 / Log where a formula unit is put."""
            f = self.formular
            if f is not None and f.box is not None:
                text = _text(f.pdf_character or [])
                if _PLACE in text:
                    logger.warning(
                        "PLACE %r box=(%.1f,%.1f)-(%.1f,%.1f) -> x=%.1f y=%.1f scale=%.3f "
                        "| x_off=%.2f y_off=%.2f adv=%.2f",
                        text[:26], f.box.x, f.box.y, f.box.x2, f.box.y2,
                        x, y, scale, f.x_offset or 0.0, f.y_offset or 0.0,
                        f.x_advance or 0.0)
            return _relocate(self, x, y, scale)

        _ts.TypesettingUnit.relocate = relocate

    merge_overlapping_formulas._formulas_patched = True
    StylesAndFormulas.merge_overlapping_formulas = merge_overlapping_formulas
