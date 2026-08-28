"""
中文：超链接修复——把源 PDF 的链接按译文位置重新贴回产物。

Restore the source document's hyperlinks onto the translated output.

BabelDOC removes them on purpose. Its pdfminer-based parser chokes on some
annotation dictionaries, so the pipeline blanks ``/Annots`` on every object
before parsing (``format/pdf/high_level.py``) — and since links *are* annotations,
a translated paper comes out with none. A 19-page source with 354 links produces
an output with zero, so every citation marker, cross-reference and URL in it is
dead text.

**Why the links cannot simply be copied where they were.** A link is a rectangle
plus a target. The rectangle is only meaningful over the text it covers, and the
translated text is not where the English was: a Chinese line is shorter, so a
citation marker halfway down a paragraph lands somewhere else entirely. Copying
the geometry unchanged would scatter clickable rectangles across unrelated words.

The one place geometry *is* exact is the original half of a side-by-side dual
page, which is a verbatim copy of the source. Those links are copied as they are.

**Re-anchoring everywhere else.** What a link covers is nearly always something
translation leaves alone: a citation number, a section number, a figure label, a
URL. So the text under the source rectangle is read, stripped of the brackets and
commas that surround citation markers, and searched for on the translated page.

Searching alone is not enough, because the needle is often a bare number that
occurs many times — 194 of this document's 354 links match more than three places
on their page. The occurrence *index* disambiguates them: the marker that is the
ninth "2" on the English page is the ninth "2" on the Chinese one, because the
tokens that carry these links are exactly the tokens translation does not touch.
Counts agree for 94% of the links here; where they do not, the nearest candidate
to the original position is used, and a link whose text cannot be found at all is
dropped rather than guessed at.

Named destinations are resolved to explicit page-and-point targets, because the
name tree does not survive either and a link to a name the document no longer
defines goes nowhere.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from pathlib import Path

import pymupdf

logger = logging.getLogger(__name__)

# Punctuation that surrounds a citation marker without being part of it.
# Stripping it lifts the match rate on this document from 91% to 99%: the source
# rectangle of "[23," covers the bracket and comma, and the same marker in the
# translation is bracketed differently or not at all.
# 引用标记周围的标点，本身不属于标记内容。去掉它可把本文档的匹配率从 91% 提到
# 99%：源矩形 "[23," 连括号和逗号一起覆盖，而译文中同一标记的括号形式可能不同。
_STRIP = " \t\r\n[](){}<>,.;:，。；：、"

# A needle longer than this is a sentence, not a label, and searching for it on a
# translated page is pointless.
# 超过该长度的检索串是句子而非标签，在译文页上搜索没有意义。
_MAX_NEEDLE = 60

# A word counts as inside the link when this share of it lies within the
# rectangle. Link rectangles are drawn a little loose around their text, so a
# neighbouring word often clips the edge; requiring most of the word excludes it.
# 词的这一比例落在矩形内才算属于该链接。链接矩形通常比文字略宽松，邻词常会擦到边缘；
# 要求词的大部分落入即可排除它们。
_WORD_OVERLAP = 0.6

# The same test at character level, used when no whole word qualifies.
# 字符级的同一判据，在没有整词合格时使用。
_CHAR_OVERLAP = 0.5


# A citation marker as the source draws it, brackets and separator included:
# "[28]", "[37,", "38]". The brackets are what make it unambiguous — the bare
# number 28 also occurs inside 「26.8提升至28.6」 and in a dozen other places on
# the same page, and picking the wrong occurrence is how sixteen of
# arXiv:2608.12209's links landed on the wrong text.
# 源文绘制的引用标记，含方括号与分隔符："[28]"、"[37,"、"38]"。方括号正是消歧的关键
# ——光是数字 28 在同一页的「26.8提升至28.6」里也有，还有十来处别的地方；选错出现位置
# 正是 arXiv:2608.12209 有十六个链接落错文字的原因。
_NUMERIC_CITATION = re.compile(r"^[\[\(]?\d{1,3}[\],;\)]?$")


def _citation_fragment(page, rect) -> str:
    """
    中文：取出链接下的引用标记原样（含方括号），用作更唯一的检索串。

    Read a numeric citation marker under a link, brackets and all.

    The stripped needle for a citation link is a bare number, which is rarely
    unique on a page. The bracket that the source draws around it is, and it
    survives translation because a numeric citation is not translated.

    Args:
        page (pymupdf.Page): Source page.
        rect (pymupdf.Rect): The link's rectangle.

    Returns:
        str: The marker with its punctuation, e.g. ``"[28]"`` or ``"[37,"``, or
            ``""`` when the link is not a bracketed numeric citation.
    """
    words = _words_inside(page, rect)
    if not words:
        return ""
    compact = "".join("".join(word.split()) for _, _, word in sorted(words))
    if not _NUMERIC_CITATION.match(compact):
        return ""
    # A bare "28" with no bracket is no more findable than the needle already
    # is, so there is nothing to gain from trying it separately.
    # 不带括号的光秃秃 "28" 并不比现有待查串更好找，单独试它没有意义。
    return compact if any(ch in compact for ch in "[](),;") else ""


def _words_inside(page, rect) -> list:
    """
    中文：取出真正落在矩形内的词及其位置。

    Collect the words that genuinely sit inside a rectangle.

    Built from words rather than from :meth:`Page.get_textbox`, which returns
    the text of every *line* the rectangle touches: in a two-column paper a link
    in the left column touches the line running through the right one, and the
    needle for a citation marker came out as ``"13, xpen"`` or ``"Th [5] res"``.

    Args:
        page (pymupdf.Page): Page to read.
        rect (pymupdf.Rect): Area to read inside.

    Returns:
        list[tuple[float, float, str]]: ``(y0, x0, word)`` per word, unsorted.
    """
    try:
        words = page.get_text("words")
    except Exception:  # noqa: BLE001
        return []

    inside = []
    for x0, y0, x1, y1, word, *_ in words:
        box = pymupdf.Rect(x0, y0, x1, y1)
        overlap = box & rect
        if overlap.is_empty:
            continue
        area = box.get_area()
        if area <= 0 or overlap.get_area() / area < _WORD_OVERLAP:
            continue
        inside.append((y0, x0, word))
    return inside


def _needle(page, rect) -> str:  # noqa: D401
    """
    中文：取出链接矩形下的文字，作为在译文页上定位的检索串。

    Read the text under a link rectangle and reduce it to a search key.

    Args:
        page (pymupdf.Page): Source page.
        rect (pymupdf.Rect): The link's rectangle.

    Returns:
        str: Text stripped of surrounding punctuation, empty when unusable.
    """
    # Built from the words that actually sit inside the rectangle, not from
    # get_textbox. That method returns the text of every *line* the rectangle
    # touches, and in a two-column paper a link in the left column touches the
    # line running through the right one: the needle for a citation marker came
    # out as "13, xpen", "21] sive", "Th [5] res". Anchoring found 99% of the
    # links in a single-column paper and 38% in a two-column one for this reason
    # alone.
    # 由真正落在矩形内的词构成，而不是用 get_textbox。后者返回矩形所触及的每一「行」的
    # 文字，而在双栏论文里，左栏的链接会触及贯穿右栏的那一行：引用标记的检索串因此变成
    # "13, xpen"、"21] sive"、"Th [5] res"。单栏论文能锚定 99% 的链接、双栏只有 38%，
    # 仅此一个原因。
    try:
        words = page.get_text("words")
    except Exception:  # noqa: BLE001
        return ""

    inside = []
    included_right: list[float] = []
    for x0, y0, x1, y1, word, *_ in words:
        box = pymupdf.Rect(x0, y0, x1, y1)
        overlap = box & rect
        if overlap.is_empty:
            continue
        area = box.get_area()
        if area <= 0 or overlap.get_area() / area < _WORD_OVERLAP:
            continue
        inside.append((y0, x0, word))
        included_right.append(x1)

    if not inside:
        # Nothing survived the word test, which happens when the rectangle is
        # drawn tightly around a fragment: an equation reference "(1)" has its
        # link over the digit alone, so the word "(1)" is only half inside and is
        # rejected. Falling back to characters keeps those.
        # 没有整词通过判据。当矩形紧贴片段绘制时就会这样：公式引用 "(1)" 的链接只盖住
        # 数字本身，整词 "(1)" 仅一半在内而被排除。退回按字符提取即可保住这些。
        inside = _chars_inside(page, rect)
        if not inside:
            return ""
        text = "".join(word for _, _, word in sorted(inside)).strip(_STRIP)
        return text if 0 < len(text) <= _MAX_NEEDLE else ""
    inside.sort()
    text = " ".join(word for _, _, word in inside).strip(_STRIP)
    if not text:
        return ""

    # hyperref draws the rectangle over the word alone and leaves the number
    # outside it, so the text under a reference link is the bare "Section",
    # "Table" or "Algorithm" while the source plainly reads "(Section 4)". The
    # bare word is untranslatable as a search key — every page has sections —
    # and it is the number that makes the reference findable in Chinese. It is
    # picked up from just past the rectangle's right edge.
    # hyperref 只在单词上画矩形、把编号留在框外，因此引用类链接下的文字是光秃秃的
    # "Section"、"Table"、"Algorithm"，而源文明明写着 "(Section 4)"。裸词作为检索串
    # 没有意义——每页都有小节——真正能在中文页上定位的是编号。这里从矩形右边缘之外把它
    # 取回来。
    if _BARE_REFERENCE.fullmatch(text):
        # Measured from the right edge of the last word taken, not from the
        # rectangle's. The number is usually *inside* the rectangle and only
        # overhangs it — "Table 1." has the rectangle ending mid-way through the
        # "1." — so it fails the overlap test that keeps a neighbouring word out,
        # and starting the search at the rectangle's edge would miss it too.
        # 以已纳入的最后一个词的右端为起点，而不是矩形的右端。编号通常就在矩形*内部*、
        # 只是略微探出——"Table 1." 的矩形正好切在 "1." 中间——因此它既过不了用于排除
        # 邻词的重叠判据，从矩形边缘开始找也会错过它。
        right = max(included_right)
        number = _number_after(right, rect, words)
        if number:
            text = f"{text} {number}"

    return text if 0 < len(text) <= _MAX_NEEDLE else ""


# How a cross-reference is written in the translation. The English word is
# translated but the number never is, so the number is the invariant and the
# Chinese word is what the search has to supply. Several renderings are listed
# per term because the model does not settle on one: a section reference comes
# back as 第4节 or 章节4 depending on the sentence.
# 交叉引用在译文中的写法。英文词被翻译而编号不变，因此编号是不变量，中文词则需要由
# 搜索侧补上。每个词列出多种写法，因为模型并不固定用一种：小节引用视句子而定，可能是
# 第4节，也可能是章节4。
_CROSS_REFERENCE = re.compile(
    r"^(Sections?|Figures?|Figs?\.?|Tables?|Algorithms?|Appendi(?:x|ces)"
    r"|Equations?|Eqs?\.?|Theorems?|Lemmas?|Corollar(?:y|ies)|Definitions?"
    r"|Lines?|Chapters?|Parts?|Steps?|Remarks?|Propositions?|Assumptions?"
    r"|Examples?)"
    # The number may be parenthesised — "Eq. (3.31)" is how most papers write an
    # equation reference — and may carry a trailing stop, because the needle is
    # cut from the page and takes whatever punctuation followed the link.
    # 编号可以带括号——"Eq. (3.31)" 是绝大多数论文写公式引用的方式——也可以带句末标点，
    # 因为待查串是从页面上截下来的，链接后面跟着什么标点就带上什么。
    r"\s*[\(（]?\s*([0-9A-Z]+(?:\.[0-9]+)*)\s*[\)）]?\s*[.,;:]?\s*$",
    re.I,
)

_CROSS_REFERENCE_FORMS: dict[str, tuple[str, ...]] = {
    # 「A 部分」 is what the model writes for an appendix referred to as a
    # section — measured twice in arXiv:2608.13520, which is where every other
    # form in this list also came from.
    # 「A 部分」是模型对以 section 形式引用的附录的写法——在 arXiv:2608.13520 上实测到
    # 两次，本表其余各式也都来自实测。
    "section": ("第{n}节", "第 {n} 节", "章节{n}", "节{n}", "第{n}章", "小节{n}",
                "第{n}小节", "Section {n}", "{n}节", "{n}部分", "第{n}部分"),
    "figure": ("图{n}", "图 {n}", "图表{n}", "Figure {n}", "Fig. {n}"),
    "fig": ("图{n}", "图 {n}", "Fig. {n}"),
    "table": ("表{n}", "表 {n}", "表格{n}", "Table {n}"),
    "algorithm": ("算法{n}", "算法 {n}", "Algorithm {n}"),
    "appendix": ("附录{n}", "附录 {n}", "Appendix {n}", "{n}部分", "第{n}部分",
                 "{n}节", "第{n}节"),
    # Parenthesised first: measured over a 51-page paper the translation wrote
    # 方程(N) 146 times and 式(N) 35, against a bare 式N not once. The bare forms
    # stay behind them for documents that do drop the brackets.
    # 带括号的排在前面：在一篇 51 页论文上实测，译文写 方程(N) 146 次、式(N) 35 次，
    # 而不带括号的 式N 一次也没有。不带括号的形式留在后面，供确实省略括号的文档使用。
    "equation": ("方程({n})", "式({n})", "公式({n})", "方程 ({n})", "式 ({n})",
                 "方程（{n}）", "式（{n}）", "公式（{n}）", "等式({n})",
                 "式{n}", "公式{n}", "方程{n}", "等式{n}", "Equation {n}", "({n})"),
    "eq": ("方程({n})", "式({n})", "公式({n})", "方程 ({n})", "式 ({n})",
           "方程（{n}）", "式（{n}）", "等式({n})",
           "式{n}", "公式{n}", "等式{n}", "Eq. {n}", "({n})"),
    "theorem": ("定理{n}", "定理 {n}", "Theorem {n}"),
    "lemma": ("引理{n}", "引理 {n}", "Lemma {n}"),
    "corollary": ("推论{n}", "推论 {n}", "Corollary {n}"),
    "definition": ("定义{n}", "定义 {n}", "Definition {n}"),
    "line": ("第{n}行", "行{n}", "Line {n}"),
    "chapter": ("第{n}章", "章{n}", "Chapter {n}"),
    "part": ("第{n}部分", "Part {n}"),
    "step": ("第{n}步", "步骤{n}", "Step {n}"),
    "remark": ("注{n}", "备注{n}", "Remark {n}"),
    "proposition": ("命题{n}", "Proposition {n}"),
    "assumption": ("假设{n}", "Assumption {n}"),
    "example": ("例{n}", "示例{n}", "Example {n}"),
}

# Words a link is anchored on that carry no number at all. A URI on the word
# "here" is written 此处 or 这里 in the translation, so the English is unfindable
# and there is no number to fall back on.
# 不带任何编号的链接锚词。锚在 "here" 上的网址，译文写作「此处」或「这里」，英文既找不
# 到、也没有编号可退。
_ANCHOR_WORDS: dict[str, tuple[str, ...]] = {
    "here": ("此处", "这里", "此", "点击此处", "见此"),
    "this": ("此", "这", "本文"),
    "see": ("参见", "见", "详见"),
    "link": ("链接", "此链接"),
    "code": ("代码", "此代码", "代码库"),
    "repository": ("代码库", "仓库", "存储库"),
    "website": ("网站", "网页"),
    "page": ("页面", "网页"),
    "paper": ("论文", "该论文"),
    "available": ("可获取", "可在", "已公开"),
}


# A citation written as author-and-year rather than as a bracketed number.
# natbib hyperlinks the two halves separately, so "(Song & Zheng, 2026)" arrives
# as two links: one over "Song & Zheng" and one over "2026". The year half needs
# nothing — a year survives translation — but the author half does, because the
# translation writes 「Song 和 Zheng」 or 「Watson等人」 and the English string is
# nowhere on the page. This is the single largest class of link on an
# author-year paper: 2608.13040 has 242 links and 131 of them are author halves.
# 作者-年份式引用，而非方括号编号式。natbib 会把两半分别加链接，因此
# "(Song & Zheng, 2026)" 到手时是两个链接：一个盖 "Song & Zheng"，一个盖 "2026"。
# 年份那半不用管——年份不随翻译改变——作者那半必须处理，因为译文写作「Song 和 Zheng」
# 或「Watson等人」，英文串在页面上根本不存在。在作者-年份体例的论文里这是最大的一类
# 链接：2608.13040 共 242 个链接，其中 131 个是作者半边。
_ET_AL = re.compile(r"^(?P<name>.+?)[\s,]*\bet\s*\.?\s*al\.?$", re.I)
_TWO_AUTHORS = re.compile(
    r"^(?P<first>.+?)\s*(?:&|\band\b)\s*(?P<second>.+?)$", re.I)

# Spacing accent characters as a PDF's text extraction reports them, and the
# combining mark each stands for. A source needle reads "H¨ubotter" because the
# diaeresis is a separate glyph placed before the letter; the translated page
# reads "Hübotter" because parsing merges the two (see ``engine/accents``). The
# needle has to be composed the same way or the name never matches.
# PDF 文本抽取给出的间隔重音符，及其对应的组合符号。源文待查串是 "H¨ubotter"，因为
# 分音符是独立字形且排在字母之前；译文页上是 "Hübotter"，因为解析阶段已把两者合并
# （见 engine/accents）。待查串必须同样合成，否则这个姓氏永远匹配不上。
_SPACING_ACCENTS = {
    "\u00a8": "\u0308", "\u00b4": "\u0301", "`": "\u0300", "\u02c6": "\u0302",
    "^": "\u0302", "\u02dc": "\u0303", "~": "\u0303", "\u02c7": "\u030c",
    "\u02da": "\u030a", "\u00af": "\u0304", "\u02d8": "\u0306",
    "\u02d9": "\u0307", "\u02dd": "\u030b", "\u00b8": "\u0327",
}


def _compose_accents(text: str) -> str:
    """
    中文：把「重音符 + 字母」合成为带重音的字母，与解析阶段的合并保持一致。

    Compose a spacing accent with the letter it precedes.

    Args:
        text (str): Text as extracted from the source, e.g. ``"H¨ubotter"``.

    Returns:
        str: The same text with each spacing accent merged into the letter that
            follows it, e.g. ``"Hübotter"``. Text without such accents is
            returned unchanged.
    """
    if not any(char in _SPACING_ACCENTS for char in text):
        return text
    out: list[str] = []
    index = 0
    while index < len(text):
        char = text[index]
        mark = _SPACING_ACCENTS.get(char)
        if mark and index + 1 < len(text) and text[index + 1].isalpha():
            out.append(unicodedata.normalize("NFC", text[index + 1] + mark))
            index += 2
            continue
        out.append(char)
        index += 1
    return "".join(out)


def _author_forms(needle: str) -> list[str]:
    """
    中文：把英文作者引用换成译文里可能的写法（等人 / 和 / 与）。

    Render an author citation the way the translation would write it.

    Three shapes cover the field. ``"Watson et al."`` becomes 「Watson等人」;
    ``"Song & Zheng"`` becomes 「Song 和 Zheng」; a lone surname is left as it is,
    because a name is not translated. Every candidate keeps the Latin surname —
    that is what makes the search reliable — and spacing is irrelevant because
    :func:`_find_text` searches text with its spaces removed.

    Args:
        needle (str): Text under the link, e.g. ``"Li et al"``.

    Returns:
        list[str]: Candidate strings to search for on the translated page, most
            likely first. Empty when the needle is not an author citation.
    """
    stripped = _compose_accents(needle.strip().strip(",;"))
    if not stripped or not any(char.isalpha() for char in stripped):
        return []

    match = _ET_AL.match(stripped)
    if match:
        name = match.group("name").strip()
        return [f"{name}等人", f"{name}等", f"{name}其他人", f"{name}et al.",
                f"{name}etal", name]

    match = _TWO_AUTHORS.match(stripped)
    if match:
        first, second = match.group("first").strip(), match.group("second").strip()
        # Both separators are offered whichever the source used: a paper that
        # writes 「Ho and Salimans」 and one that writes 「Song & Zheng」 are
        # rendered the same way in Chinese, and the untranslated form has to
        # stay available for a citation the translation left in English.
        # 无论源文用哪个连接词，两种都给出：写「Ho and Salimans」的论文与写
        # 「Song & Zheng」的论文，中文写法相同；同时保留未翻译的形式，供译文原样保留
        # 的引用使用。
        return [f"{first}和{second}", f"{first}与{second}", f"{first}及{second}",
                f"{first}&{second}", f"{first}and{second}", first]

    # A lone surname, or a name the accent composition just changed. Worth
    # searching only when the text is not plain ASCII already — a plain one has
    # been searched verbatim by the caller and failed.
    # 单个姓氏，或刚被重音合成改写过的姓名。仅当文本已非纯 ASCII 时才值得再搜一次
    # ——纯 ASCII 的那种调用方已按原样搜过且落空了。
    if stripped != needle.strip():
        return [stripped]
    return []


def _both_orders(forms: tuple[str, ...]) -> list[str]:
    """
    中文：给每个「第…」的写法补上去掉「第」的那一种。

    Add the bare form of each 「第…」 form.

    Chinese writes a cross-reference either way round — 「第 4 节」 and 「4 节」 are
    both ordinary — and which one a translation picks is not predictable. Every
    form measured on a real document is listed by hand in
    :data:`_CROSS_REFERENCE_FORMS`, in the order they were seen; this adds the
    bare variant of each so that an unseen ordering is not a missed link.

    Only that direction. Putting 「第」 *in front of* an arbitrary form invents
    strings no one writes — 「第图 2」, 「第Section 4」 — which cost a search each
    and could in principle match something they should not.

    Args:
        forms (tuple[str, ...]): The measured forms, most likely first.

    Returns:
        list[str]: Those forms with the bare variants appended, duplicates
            removed and the measured order preserved.
    """
    out: list[str] = []
    for form in forms:
        for candidate in (form, form[1:] if form.startswith("第") else ""):
            if candidate and candidate not in out:
                out.append(candidate)
    return out


def _translated_forms(needle: str) -> list[str]:
    """
    中文：把英文交叉引用换成译文里可能的写法。

    Render an English cross-reference the way the translation would write it.

    Args:
        needle (str): Text under the link, e.g. ``"Section 4"``.

    Returns:
        list[str]: Candidate strings to search for on the translated page, most
            likely first. Empty when the needle is not a cross-reference.
    """
    stripped = needle.strip()
    plain = _ANCHOR_WORDS.get(stripped.lower().strip(".,;:"))
    if plain:
        return list(plain)
    match = _CROSS_REFERENCE.match(stripped)
    if not match:
        return _author_forms(stripped)
    word = match.group(1).lower().rstrip("s.").rstrip("e") or match.group(1).lower()
    number = match.group(2)
    for key, forms in _CROSS_REFERENCE_FORMS.items():
        if word.startswith(key[:3]):
            return [form.format(n=number) for form in _both_orders(forms)]
    return _author_forms(stripped)


# A link's text is nearly always coloured — blue in almost every paper — and the
# colour survives translation: the citation marker that is blue "13" in the
# source is blue "13" on the Chinese page, moved to wherever the sentence put it.
# That makes colour a better index than a page-wide text search, because it
# restricts the candidates to text that is itself a link.
# 链接文字几乎总是带颜色的——绝大多数论文里是蓝色——而颜色会随翻译保留：源文里蓝色的
# 引用标记 "13"，在中文页上仍是蓝色的 "13"，只是移到了句子安排的位置。因此颜色比全页
# 文字搜索是更好的索引，它把候选限制在本身就是链接的文字上。
_MIN_BLUE = 110
_BLUE_MARGIN = 40

# Blue is only the *usual* link colour, not the only one. A LaTeX preamble sets
# hyperref's ``citecolor``, ``linkcolor`` and ``urlcolor`` independently, and
# 2608.13040 uses green #009a55 for its 131 citation links and blue #0072bc for
# its URLs. Assuming blue made every one of those green citations invisible to
# the colour-based anchoring, and would have repainted the survivors the wrong
# colour. So the palette is read off the source's own links rather than assumed.
# 蓝色只是链接的「常见」颜色，不是唯一颜色。LaTeX 导言区里 hyperref 的 citecolor、
# linkcolor、urlcolor 是分开设的，2608.13040 的 131 个引用链接用绿色 #009a55，网址
# 用蓝色 #0072bc。假定蓝色会让那些绿色引用在按颜色定位时完全不可见，侥幸活下来的也
# 会被染成错误的颜色。因此调色板改为从源文自己的链接上读出来，而不是假定。
_DEFAULT_LINK_COLOUR = 0x0000FF

# Pages scanned to learn the palette. A paper sets its colours in the preamble,
# so the first few pages already show all of them; the cap keeps a long document
# from paying for a full extra text pass.
# 为学习调色板而扫描的页数。论文的颜色在导言区设定，前几页就能看全；设上限是为了
# 避免长文档为此多付一次整篇文本抽取的代价。
_PALETTE_PAGES = 12

# How many spans must carry a colour before it counts as a link colour rather
# than a one-off. A heading that happens to sit under a link should not enrol
# its colour.
# 一个颜色要出现在多少个 span 上才算链接色而非偶然。恰好压在链接下的标题不应把它的
# 颜色登记进来。
_PALETTE_MIN_HITS = 3

# Channel spread below which a colour is grey — black text under a link (the
# comma between two citations) must never be taken for the link colour.
# 通道差小于此值即视为灰阶——链接下的黑色文字（两个引用之间的逗号）绝不能被当成链接色。
_GREY_SPREAD = 24

# Per-channel tolerance when testing a span against the palette. Anti-aliasing
# and colour-space conversion move a channel by a few units, not by forty.
# 判断 span 是否属于调色板时的单通道容差。抗锯齿与色彩空间转换会让通道偏移几个单位，
# 而不是四十个。
_PALETTE_TOLERANCE = 12


def _is_grey(colour: int) -> bool:
    """
    中文：判断颜色是否为灰阶（含黑白）。

    Decide whether a colour is grey, black or white.

    Args:
        colour (int): Packed ``0xRRGGBB`` colour.

    Returns:
        bool: ``True`` when the three channels are within ``_GREY_SPREAD`` of
            each other.
    """
    red, green, blue = (colour >> 16) & 255, (colour >> 8) & 255, colour & 255
    return max(red, green, blue) - min(red, green, blue) <= _GREY_SPREAD


def link_palette(document) -> tuple[int, ...]:
    """
    中文：从源文自身的链接上读出它使用的链接颜色集合。

    Learn the colours this document draws its links in.

    Every link rectangle is intersected with the page's text spans and the
    non-grey colours found underneath are counted. Colours seen on at least
    ``_PALETTE_MIN_HITS`` spans are the document's link palette, most used
    first.

    Args:
        document (pymupdf.Document): Source document, read only.

    Returns:
        tuple[int, ...]: Packed ``0xRRGGBB`` colours, most frequent first.
            Empty when the document has no coloured links, in which case
            callers fall back to the built-in blue test.
    """
    counts: dict[int, int] = {}
    for number in range(min(document.page_count, _PALETTE_PAGES)):
        try:
            page = document[number]
            links = page.get_links()
            if not links:
                continue
            blocks = page.get_text("dict")["blocks"]
        except Exception:  # noqa: BLE001
            continue

        spans = [(pymupdf.Rect(span["bbox"]), span.get("color", 0))
                 for block in blocks if block.get("type") == 0
                 for line in block.get("lines", ())
                 for span in line.get("spans", ())
                 if (span.get("text") or "").strip()]

        for link in links:
            rect = pymupdf.Rect(link["from"])
            for box, colour in spans:
                if _is_grey(colour) or not box.intersects(rect):
                    continue
                counts[colour] = counts.get(colour, 0) + 1

    palette = [colour for colour, hits in counts.items() if hits >= _PALETTE_MIN_HITS]
    palette.sort(key=lambda colour: -counts[colour])
    if palette:
        logger.info("Link palette: %s",
                    ", ".join(f"#{colour:06x}({counts[colour]})" for colour in palette))
    return tuple(palette)


def _is_link_colour(colour: int, palette: tuple[int, ...] = ()) -> bool:
    """
    中文：判断 span 颜色是否为链接色；有调色板时按调色板判定，否则退回偏蓝判据。

    Decide whether a span colour is the colour links are drawn in.

    Args:
        colour (int): Packed ``0xRRGGBB`` colour from a text span.
        palette (tuple[int, ...]): Colours learned from the source's own links.
            When empty, the built-in "distinctly blue" test is used instead.

    Returns:
        bool: ``True`` when the colour is one the document uses for links.
    """
    red, green, blue = (colour >> 16) & 255, (colour >> 8) & 255, colour & 255
    if palette:
        for other in palette:
            if (abs(red - ((other >> 16) & 255)) <= _PALETTE_TOLERANCE
                    and abs(green - ((other >> 8) & 255)) <= _PALETTE_TOLERANCE
                    and abs(blue - (other & 255)) <= _PALETTE_TOLERANCE):
                return True
        return False
    return blue >= _MIN_BLUE and blue >= red + _BLUE_MARGIN and blue >= green + _BLUE_MARGIN


def _coloured_runs(page, clip=None, palette: tuple[int, ...] = ()) -> list:
    """
    中文：按阅读顺序列出页面上所有链接色文字，并保留各自的颜色。

    List the link-coloured text on a page, in reading order.

    Args:
        page (pymupdf.Page): Page to scan.
        clip (pymupdf.Rect | None): Restrict to this area, for one half of a
            side-by-side page.
        palette (tuple[int, ...]): Link colours learned from the source. Empty
            falls back to the built-in blue test.

    Returns:
        list[tuple]: ``(rect, text, colour)`` per coloured run, top-to-bottom
            then left-to-right. ``colour`` is packed ``0xRRGGBB``.
    """
    out: list = []
    try:
        blocks = (page.get_text("dict", clip=clip) if clip
                  else page.get_text("dict"))["blocks"]
    except Exception:  # noqa: BLE001
        return out

    for block in blocks:
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = (span.get("text") or "").strip()
                colour = span.get("color", 0)
                if not text or not _is_link_colour(colour, palette):
                    continue
                out.append((pymupdf.Rect(span["bbox"]), text, colour))
    out.sort(key=lambda item: (round(item[0].y0, 1), item[0].x0))
    return out


def _anchor_by_colour(src_page, dst_page, rect, cache, clip=None):
    """
    中文：用链接色文字把链接对应到译文页上的位置。

    Locate a link by matching the link-coloured text around it.

    The source's coloured run under the link is identified, then the run with the
    same text is looked for among the translated page's coloured runs. Where a
    text occurs several times the occurrence index carries over, as it does for
    the plain text search — but the candidate set is now only text that is itself
    a link.

    Args:
        src_page (pymupdf.Page): Source page.
        dst_page (pymupdf.Page): Translated page.
        rect (pymupdf.Rect): Link rectangle in source coordinates.
        cache (dict): Per-page memo.
        clip (pymupdf.Rect | None): Restrict the destination scan, for a dual
            page's translated half.

    Returns:
        pymupdf.Rect | None: Where the link belongs, or ``None``.
    """
    palette = cache.get("palette", ())
    if "src_colour" not in cache:
        cache["src_colour"] = _coloured_runs(src_page, palette=palette)
    if "dst_colour" not in cache:
        cache["dst_colour"] = _coloured_runs(dst_page, clip, palette)
    source_runs, target_runs = cache["src_colour"], cache["dst_colour"]
    if not source_runs or not target_runs:
        return None

    index = next((i for i, (box, _, _) in enumerate(source_runs) if box.intersects(rect)),
                 None)
    if index is None:
        return None
    text = source_runs[index][1]

    same_text = [box for box, other, _ in target_runs if other == text]
    if same_text:
        # Which occurrence of this text among the source's coloured runs is ours.
        # 本链接对应的是源文彩色文字中该文本的第几次出现。
        rank = sum(1 for i in range(index) if source_runs[i][1] == text)
        return same_text[rank] if rank < len(same_text) else _nearest(same_text, rect)

    # No run with the same text. When both pages carry the same number of
    # coloured runs, the sequence itself is the mapping.
    # 没有同文本的候选。若两页彩色文字数量一致，序列本身就是对应关系。
    if len(source_runs) == len(target_runs):
        return target_runs[index][0]
    return None


# A number as a reference writes it: 28, 2.1, 3.31, A.4 is handled by the
# letter branch. Compared as a whole token, never digit by digit.
# 引用里写出的编号：28、2.1、3.31；A.4 走字母分支。整体比对，绝不逐位比。
_NUMBER_TOKEN = re.compile(r"\d+(?:\.\d+)*")


def _plausible(dst_page, rect, needle: str, cache: dict) -> bool:
    """
    中文：检查按颜色定位到的位置，其文字是否与待查串对得上。

    Check that an anchor actually landed on the text it was looking for.

    Colour matching is tried first because it searches only text that is itself
    a link, but when it is wrong nothing else notices. On page 9 of
    arXiv:2608.12209 it answered a search for citation 「28」 with a rectangle one
    glyph wide reading 「82」, and did the same for fifteen more — every mis-placed
    link on that document came through this path. Reading the text back and
    comparing is what turns a wrong answer into a fall-through to the text
    search, which gets it right.

    Args:
        dst_page (pymupdf.Page): Translated page.
        rect (pymupdf.Rect): The candidate rectangle on it.
        needle (str): What was being looked for.
        cache (dict): Per-page memo for :func:`_link_glyphs`.

    Returns:
        bool: ``True`` when the landed text is consistent with the needle, or
            when the needle offers nothing to check against.
    """
    landed = fold("".join(char.get("c") or ""
                          for char, _ in _link_glyphs(dst_page, rect, cache)))
    if not landed:
        return False

    wanted = fold(needle)
    numbers = _NUMBER_TOKEN.findall(wanted)
    if numbers:
        # A cross-reference is translated — "Section 4" becomes 「第4节」 — so the
        # number is the only part that has to survive, and it has to survive as
        # the number it is: 「28」 must not be satisfied by the 「2」 of 「82」, and
        # 「2.1」 must not be reduced to 「21」 before comparing, or every equation
        # reference is rejected.
        # 交叉引用会被翻译——"Section 4" 变成「第4节」——因此只有编号必须留存，而且要作为
        # 它本身的那个数留存：「28」不能被「82」里的「2」蒙混过关，而「2.1」也不能先被
        # 压成「21」再比对，否则每个公式引用都会被判否。
        return any(number in landed for number in numbers)

    head = next((word.strip(".,;:()[]") for word in wanted.split()
                 if len(word.strip(".,;:()[]")) >= 2), "")
    return head.lower() in landed.lower() if head else True


def _source_colour(src_page, rect, cache) -> int:
    """
    中文：读出源文该链接文字的实际颜色，作为译文重新着色的目标色。

    Read the colour the source draws this particular link in.

    A document uses several link colours at once — citations green, URLs blue —
    so the colour has to be taken per link rather than per document.

    Args:
        src_page (pymupdf.Page): Source page.
        rect (pymupdf.Rect): The link's rectangle on it.
        cache (dict): Per-page memo, carrying ``palette``.

    Returns:
        int: Packed ``0xRRGGBB`` colour of the coloured run under the link, or
            :data:`_DEFAULT_LINK_COLOUR` when the source draws it in no colour
            of its own.
    """
    palette = cache.get("palette", ())
    if "src_colour" not in cache:
        cache["src_colour"] = _coloured_runs(src_page, palette=palette)
    for box, _, colour in cache["src_colour"]:
        if box.intersects(rect):
            return colour
    return _DEFAULT_LINK_COLOUR


# A reference word with no number of its own.
# 不带编号的引用词。
_BARE_REFERENCE = re.compile(
    r"(?:Sections?|Figures?|Figs?\.?|Tables?|Algorithms?|Appendi(?:x|ces)"
    r"|Equations?|Eqs?\.?|Theorems?|Lemmas?|Corollar(?:y|ies)|Definitions?"
    r"|Lines?|Chapters?|Parts?|Steps?|Remarks?|Propositions?|Assumptions?"
    r"|Examples?)",
    re.I,
)

# The number belonging to such a word, as it is written just after it.
# 紧跟其后的编号写法。
# Trailing punctuation is a run, not a single character: a reference at the end
# of a clause is written "Table 4)," and "(Algorithm 2).", so allowing one closing
# mark matched neither.
# 尾部标点是一串而不是一个：句中的引用写作 "Table 4)," 与 "(Algorithm 2)."，只允许
# 一个收尾符号两者都匹配不到。
_REFERENCE_NUMBER = re.compile(r"^[\(（]?([0-9A-Z]+(?:\.[0-9]+)*)[\)）\.,;:，。；：]*$")

# How far past the rectangle's right edge to look for that number, in points.
# 在矩形右边缘之外寻找该编号的距离（点）。
_NUMBER_REACH = 26.0

# Baseline difference below which two characters are on the same line.
# 基线差小于此值即认为两个字符在同一行。
_LINE_SLACK = 1.0


def _number_after(right: float, rect, words) -> str:
    """
    中文：取出紧跟在链接矩形右侧的编号。

    Read the reference number sitting just past a link rectangle.

    Args:
        right (float): Right edge of the last word already taken for the needle.
        rect (pymupdf.Rect): The link's rectangle, used for the line test.
        words (list): Words of the page, as :meth:`Page.get_text` returns them.

    Returns:
        str: The number, or an empty string when the next thing along is not one.
    """
    best = None
    for x0, y0, _x1, y1, word, *_ in words:
        if x0 < right - 1 or x0 > right + _NUMBER_REACH:
            continue
        # Same line, judged by the word's vertical centre rather than by any
        # overlap. The line below grazes the rectangle's lower edge by a fraction
        # of a point, and a mere-overlap test let its first word — "the", sitting
        # slightly left of the number — win as the nearest candidate.
        # 同一行，以词的垂直中心判定而非只要有重叠。下一行会以零点几个点的幅度擦到矩形
        # 下缘，只要求重叠会让它的首词（"the"，位置略在编号左侧）被当成最近候选。
        middle = (y0 + y1) / 2.0
        if middle <= rect.y0 or middle >= rect.y1:
            continue
        if best is None or x0 < best[0]:
            best = (x0, word)
    if best is None:
        return ""
    match = _REFERENCE_NUMBER.match(best[1].strip())
    return match.group(1) if match else ""


# Unicode categories that carry no ink: control characters and format
# characters. A PDF's space glyph is sometimes mapped to one of these.
# 不着墨的 Unicode 类别：控制字符与格式字符。PDF 的空格字形有时会被映射到其中之一。
_SPACING_CATEGORIES = frozenset({"Cc", "Cf", "Zs", "Zl", "Zp"})


def _is_spacing(glyph: str) -> bool:
    """
    中文：判断一个字符是否只起间隔作用（控制符、格式符、各类空格）。

    Decide whether a character is spacing rather than text.

    Args:
        glyph (str): A single extracted character.

    Returns:
        bool: ``True`` for control, format and separator characters.
    """
    return unicodedata.category(glyph) in _SPACING_CATEGORIES


def fold(text: str) -> str:
    """
    中文：把文本折成「无重音」形式，供两侧比对使用。

    Reduce text to an accent-free form for matching.

    A surname reaches the two sides in different shapes. The source PDF draws
    the diaeresis as its own glyph, so extraction yields ``H¨ubotter``; parsing
    merges the pair, so the translated page carries ``Hübotter``. Neither string
    contains the other and the link is lost. Folding both to ``Hubotter`` makes
    them meet. Non-Latin text is untouched — a CJK character has no accent to
    remove.

    Args:
        text (str): Text as extracted from a page, or a needle.

    Returns:
        str: The same text with spacing accents dropped and combining marks
            removed from the letters they sit on.
    """
    if not text:
        return text
    stripped = "".join(ch for ch in text if ch not in _SPACING_ACCENTS)
    decomposed = unicodedata.normalize("NFD", stripped)
    return unicodedata.normalize(
        "NFC", "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn"))


def _char_index(page, clip=None):
    """
    中文：把一页的字符连成一串（去掉换行与空格），并保留每个字符的位置。

    Build a page's characters as one string, with each character's rectangle.

    Line breaks and spaces are dropped from the string. A reference that wraps —
    「定」 at the end of one line and 「理2」 at the start of the next — reads as
    定理2 on the page but is 定\n理2 in the extracted text, so neither
    :meth:`Page.search_for` nor a regex over :meth:`Page.get_text` finds it. Two
    of this document's links pointed at exactly such a wrapped 定理2.

    中文说明：断行会把「定理2」拆成「定」和「理2」，页面上读起来没问题，但抽取出的文本
    里夹着换行，search_for 与正则都匹配不到——本文档就有两个链接指向这样被拆开的定理2。

    Args:
        page (pymupdf.Page): Page to index.
        clip (pymupdf.Rect | None): Restrict to this area.

    Returns:
        tuple[str, list]: The concatenated text and one rectangle per character.
    """
    import pymupdf

    try:
        blocks = (page.get_text("rawdict", clip=clip) if clip
                  else page.get_text("rawdict"))["blocks"]
    except Exception:  # noqa: BLE001
        return "", []

    text: list[str] = []
    boxes: list = []
    for block in blocks:
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                for char in span.get("chars", []):
                    glyph = char.get("c", "")
                    # Control characters count as spacing, not as text. The
                    # embedded font's ToUnicode maps its space glyph to U+0003
                    # in this document, so the extracted line reads
                    # "Song\x03&\x03Zheng" — and a needle of "Song&Zheng" then
                    # matches nothing, leaving the link anchored on the single
                    # word "Song" and only that word clickable and coloured.
                    # 控制字符属于间隔而非文字。本文档内嵌字体的 ToUnicode 把空格字形
                    # 映射到了 U+0003，抽取出的行是 "Song\x03&\x03Zheng"——于是待查串
                    # "Song&Zheng" 什么也匹配不到，链接只能落在单词 "Song" 上，可点和
                    # 着色的也就只有这一个词。
                    if not glyph or glyph.isspace() or _is_spacing(glyph):
                        continue
                    glyph = fold(glyph)
                    if not glyph:
                        continue
                    # Normalised because the embedded font's ToUnicode maps some
                    # characters to CJK compatibility ideographs: 定 comes out as
                    # U+FA08 rather than U+5B9A, so it renders correctly and is
                    # unfindable. NFC folds those back to the unified form.
                    # 需要规范化：内嵌字体的 ToUnicode 会把部分汉字映射到兼容汉字区，
                    # 「定」抽取出来是 U+FA08 而非 U+5B9A——显示正常却搜不到。NFC 可把
                    # 它们折回统一汉字。
                    text.append(unicodedata.normalize("NFC", glyph))
                    boxes.append(pymupdf.Rect(char["bbox"]))
    return "".join(text), boxes


def _find_text(page, needle: str, cache: dict, key: str, clip=None) -> list:
    """
    中文：在页面上查找一个字符串，跨越换行与空格。

    Find every occurrence of a string on a page, across line breaks.

    Args:
        page (pymupdf.Page): Page to search.
        needle (str): Text to find; its spaces are ignored.
        cache (dict): Per-page memo holding the character index.
        key (str): Cache key for this page's index.
        clip (pymupdf.Rect | None): Restrict to this area.

    Returns:
        list[pymupdf.Rect]: One rectangle per occurrence, covering its
            characters. A match that wraps across a line break is returned as
            the bounding box of both lines; :func:`_fragments` splits it back
            into the pieces actually covered, which is what callers must use
            for anything that touches the page rather than merely locating it.
    """
    return [_span(parts) for parts in _find_fragments(page, needle, cache, key, clip)]


def _find_fragments(page, needle: str, cache: dict, key: str, clip=None) -> list:
    """
    中文：查找字符串，按行返回每次出现所覆盖的矩形（换行处拆成多段）。

    Find every occurrence of a string, as one rectangle per line it covers.

    A citation often wraps: 「Song &」 ends one line and 「Zheng, 2026」 begins the
    next. The bounding box of the two is a rectangle that also covers everything
    between them — the whole tail of the first line and head of the second — so
    using it would make unrelated text clickable and, since a link's text is
    recoloured, would paint a stretch of Chinese in the citation colour. Keeping
    the pieces apart is what avoids both.

    Args:
        page (pymupdf.Page): Page to search.
        needle (str): Text to find; its spaces are ignored.
        cache (dict): Per-page memo holding the character index.
        key (str): Cache key for this page's index.
        clip (pymupdf.Rect | None): Restrict to this area.

    Returns:
        list[list[pymupdf.Rect]]: Per occurrence, one rectangle per line the
            occurrence covers, in reading order.
    """
    if key not in cache:
        cache[key] = _char_index(page, clip)
    text, boxes = cache[key]
    target = fold(unicodedata.normalize("NFC", "".join(needle.split())))
    if not target or not text:
        return []

    out = []
    start = text.find(target)
    while start != -1:
        out.append(_lines(boxes[start:start + len(target)]))
        start = text.find(target, start + 1)
    return out


def _lines(boxes: list):
    """
    中文：把一次匹配的字符按行切开。

    Split a match's characters into one rectangle per line.

    A citation often wraps: 「Song &」 ends one line and 「Zheng, 2026」 begins the
    next. The bounding box of the two also covers everything between them — the
    whole tail of the first line and head of the second — so using it would make
    unrelated text clickable and, since a link's text is recoloured, would paint
    a stretch of Chinese in the citation colour.

    Requiring the characters to be *contiguous* on the page as well was tried,
    to stop a search for ``28`` matching the ``2`` ending one bibliography entry
    and the ``8`` starting the next. Measured over 1,607 links in seven papers
    it dropped ten more links than it saved and corrected none, so the rule is
    not here.

    中文说明：曾试过再要求这些字符在页面上「连续」，以阻止搜索 ``28`` 时匹配到上一条
    参考文献结尾的 ``2`` 与下一条开头的 ``8``。在七篇论文 1,607 个链接上实测，它比原来
    多丢十个链接、一个也没修好，因此没有保留。

    Args:
        boxes (list[pymupdf.Rect]): One rectangle per matched character, in
            index order.

    Returns:
        list[pymupdf.Rect]: One rectangle per line the match covers.
    """
    parts = [pymupdf.Rect(boxes[0])]
    previous = boxes[0]
    for box in boxes[1:]:
        if abs(box.y0 - previous.y0) < _LINE_SLACK and box.x0 >= previous.x0:
            parts[-1] = parts[-1] | box
        else:
            parts.append(pymupdf.Rect(box))
        previous = box
    return parts


def _span(parts: list):
    """
    中文：把多段矩形合并成一个外接矩形。

    Combine rectangles into their bounding box.

    Args:
        parts (list[pymupdf.Rect]): One or more rectangles.

    Returns:
        pymupdf.Rect: The rectangle covering all of them.
    """
    box = pymupdf.Rect(parts[0])
    for other in parts[1:]:
        box = box | other
    return box


def _chars_inside(page, rect) -> list:
    """
    中文：取出方框内的单个字符（整词判据落空时使用）。

    Collect the individual characters inside a rectangle.

    Args:
        page (pymupdf.Page): Page to read.
        rect (pymupdf.Rect): Rectangle to read under.

    Returns:
        list[tuple]: ``(y, x, character)`` triples, unsorted.
    """
    import pymupdf

    out = []
    try:
        blocks = page.get_text("rawdict")["blocks"]
    except Exception:  # noqa: BLE001
        return out
    for block in blocks:
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                for char in span.get("chars", []):
                    box = pymupdf.Rect(char["bbox"])
                    overlap = box & rect
                    area = box.get_area()
                    if overlap.is_empty or area <= 0:
                        continue
                    if overlap.get_area() / area >= _CHAR_OVERLAP:
                        out.append((box.y0, box.x0, char.get("c", "")))
    return out


def _rank(hits, rect) -> int | None:
    """
    中文：确定该链接对应源页上同一检索串的第几次出现。

    Find which occurrence of the needle this link covers.

    Args:
        hits (list[pymupdf.Rect]): All occurrences on the source page, in the
            order :meth:`Page.search_for` returns them.
        rect (pymupdf.Rect): The link's rectangle.

    Returns:
        int | None: Index into ``hits``, or ``None`` when the link's rectangle
            overlaps none of them.
    """
    for index, hit in enumerate(hits):
        if hit.intersects(rect):
            return index
    return None


def _nearest(hits, rect):
    """
    中文：在候选矩形中取与原位置最近的一个。

    Pick the candidate closest to where the link used to be.

    Used only when the needle occurs a different number of times in the
    translation than in the source, so the occurrence index cannot be trusted.

    Args:
        hits (list[pymupdf.Rect]): Candidate rectangles on the translated page.
        rect (pymupdf.Rect): The link's rectangle on the source page.

    Returns:
        pymupdf.Rect: The closest candidate.
    """
    def distance(candidate) -> float:
        dx = (candidate.x0 + candidate.x1) / 2 - (rect.x0 + rect.x1) / 2
        dy = (candidate.y0 + candidate.y1) / 2 - (rect.y0 + rect.y1) / 2
        return dx * dx + dy * dy

    return min(hits, key=distance)


def _anchor(src_page, dst_page, rect, cache, needle: str | None = None):
    """
    中文：在译文页上为一个链接矩形找到对应位置。

    Locate a link's rectangle on the translated page.

    Args:
        src_page (pymupdf.Page): Source page.
        dst_page (pymupdf.Page): Translated page.
        rect (pymupdf.Rect): Link rectangle in source coordinates.
        cache (dict): Per-page memo of search results, keyed
            ``(side, needle)``; searching is the expensive part and the same
            needle recurs across links on a page.
        needle (str | None): Text to look for, overriding what sits under the
            rectangle. Used for a reference a line break split in two, where
            the rectangle covers only ``"Sec-"``.

    Returns:
        list[pymupdf.Rect] | None: Where the link belongs on the translated
            page, as one rectangle per line it covers — a citation that wraps
            gets two — or ``None`` when its text cannot be found there.
    """
    # Colour first. A link's text is drawn in the link colour and keeps it
    # through translation, so matching on colour searches only text that is
    # itself a link — a far smaller and cleaner candidate set than the page.
    # 先按颜色匹配。链接文字带链接色且颜色随翻译保留，因此按颜色匹配只在本身就是
    # 链接的文字里找——候选集比整页小得多也干净得多。
    needle = needle if needle is not None else _needle(src_page, rect)

    by_colour = _anchor_by_colour(src_page, dst_page, rect, cache)
    if by_colour is not None and (
            not needle or _plausible(dst_page, by_colour, needle, cache.setdefault(
                "glyphs", {}))):
        return [by_colour]

    if not needle:
        # No needle to search with, so an unverified colour match is the only
        # candidate there is.
        # 没有待查串可搜，未经校验的颜色匹配是唯一的候选。
        return [by_colour] if by_colour is not None else None

    # Both sides are searched through the page's own character index rather
    # than through :meth:`Page.search_for`. The index ignores spacing, folds
    # accents, and crosses line breaks, so a URL broken over two lines and a
    # surname whose diaeresis is a separate glyph both match whole. search_for
    # matched them partially instead, which anchored 「Yang et al」 on the two
    # letters 「al」 and a forty-character URL on 「https:」.
    # 两侧都改用页面自己的字符索引来搜，而不用 Page.search_for。索引忽略空格、折掉
    # 重音、且能跨行，因此断成两行的网址与分音符独立成字的姓氏都能整体匹配。
    # search_for 只能部分匹配，结果把「Yang et al」锚在「al」两个字母上、把四十字符
    # 的网址锚在「https:」上。
    # A bracketed citation marker is tried before the bare number it reduces to.
    # 「[28]」 occurs once on a page; 「28」 occurs inside every decimal and every
    # other citation, and the occurrence index then picks the wrong one.
    # 先试带方括号的引用标记，再试它化简成的裸数字。「[28]」在一页上通常只出现一次，
    # 而「28」在每个小数和其它引用里都有，按出现序号取就会取错。
    candidates = []
    fragment = _citation_fragment(src_page, rect)
    if fragment and fragment != needle:
        candidates.append(fragment)
    candidates.append(needle)

    src_parts = dst_parts = []
    for candidate in candidates:
        src_parts = _find_fragments(src_page, candidate, cache, "src_chars")
        dst_parts = _find_fragments(dst_page, candidate, cache, "dst_chars")
        if dst_parts:
            break
    src_hits = [_span(one) for one in src_parts]
    dst_hits = [_span(one) for one in dst_parts]

    if not dst_hits:
        # A cross-reference is the one kind of link whose text *is* translated,
        # so searching for the English is guaranteed to fail: 96 of this
        # document's 260 links point at "Section 4", "Algorithm 1" or "Figure 3"
        # and none of those strings exists on the Chinese page. The number
        # survives translation, so the reference is looked up in the form the
        # translation would have written it.
        # 交叉引用是唯一文字本身会被翻译的链接，用英文去搜必然落空：本文档 260 个链接
        # 中有 96 个指向 "Section 4"、"Algorithm 1"、"Figure 3"，这些串在中文页上都
        # 不存在。编号不随翻译改变，因此改用译文会采用的写法去查。
        for form in _translated_forms(needle):
            parts = _find_fragments(dst_page, form, cache, "dst_chars")
            if parts:
                dst_parts = parts
                dst_hits = [_span(one) for one in parts]
                break
        if not dst_hits:
            # Nothing was found by text either. A colour match that failed the
            # plausibility check is imprecise, not useless — on a bibliography
            # entry it lands on the journal name rather than on the year, and
            # the link still works. Dropping it loses the link outright.
            # 文本也没找到。未通过校验的颜色匹配只是不够精确，并非无用——在参考文献
            # 条目上它落在期刊名而不是年份上，链接照样能点。丢弃反而是彻底失去它。
            return [by_colour] if by_colour is not None else None

    rank = _rank(src_hits, rect)
    if rank is not None and len(src_hits) == len(dst_hits):
        return dst_parts[rank]
    # Counts disagree, so the index is a guess; the nearest candidate is a
    # better one than an index into a list of a different length.
    # 出现次数不一致，序号已不可靠；取距离最近的候选比按序号硬取更合理。
    nearest = _nearest(dst_hits, rect)
    if nearest is None:
        return None
    return dst_parts[dst_hits.index(nearest)]


# A needle that is only the tail of a reference the line break cut in two.
# hyperref emits one link per piece, so 「Section 3.3」 broken as 「Sec-」 /
# 「tion 3.3」 arrives as two links whose text is separately unfindable. The same
# happens to 「Yang et al.」 when the name and the 「et al.」 fall on either side of
# a break, and to a lone 「et al」 whose author sits in the previous link.
# 只是被换行截断的引用的后半段。hyperref 会给每一段各发一个链接，于是「Section 3.3」
# 断成「Sec-」/「tion 3.3」后成了两个各自都搜不到的链接。「Yang et al.」的姓名与
# 「et al.」分处断行两侧时、以及作者在前一个链接里的孤零零一个「et al」，情况相同。
_CONTINUATION = re.compile(r"^(?:et\s*\.?\s*al\.?|and|&)$", re.I)

# The arXiv stamp printed down the margin of a preprint's first page. Its link
# text comes out in reading order that does not match the page, so the stable
# part — the identifier itself — is what gets searched for.
# 预印本首页页边那条 arXiv 印记。它的链接文字抽出来的顺序与页面不一致，因此改用其中
# 稳定的那部分——编号本身——去搜。
_ARXIV_STAMP = re.compile(r"(arXiv:\d{4}\.\d{4,5}(?:v\d+)?)", re.I)


def effective_needles(page, links) -> list[str]:
    """
    中文：算出每个链接实际该拿去搜索的文字，把被换行截断的引用重新接上。

    Work out the text each link should actually be searched for.

    Two repairs, both for references a line break split into separate links.
    A needle ending in a hyphen is joined with the one after it, and a needle
    that is only a continuation — 「et al.」, 「tion 3.3」 — is joined with the one
    before it. Both links in a joined pair get the whole reference, so both land
    on it.

    Args:
        page (pymupdf.Page): Source page the links belong to.
        links (list[dict]): The page's links, in document order.

    Returns:
        list[str]: One needle per link, aligned with ``links``.
    """
    needles = [_needle(page, link["from"]) for link in links]

    for index, text in enumerate(needles):
        stamp = _ARXIV_STAMP.search(text)
        if stamp:
            needles[index] = stamp.group(1)

    joined = list(needles)
    for index, text in enumerate(needles):
        stripped = text.strip()
        if index + 1 >= len(needles):
            continue
        following = needles[index + 1].strip()
        if not stripped or not following:
            continue
        # Only two joins are safe. A hyphen at the end of a needle is the
        # line-break kind, so the two pieces are one word: "Sec-" + "tion 3.3".
        # And a needle that is nothing but "et al." belongs to the name before
        # it. Anything looser joins things that merely sit next to each other —
        # an earlier version merged two adjacent URLs because the second began
        # with a lowercase letter, and turned 「iii」 and 「v」 into 「iii v」.
        # 只有两种合并是安全的。待查串末尾的连字符是换行断字，两段本是一个词：
        # 「Sec-」+「tion 3.3」。而只由「et al.」构成的待查串属于它前面的姓名。判据再
        # 松就会把仅仅相邻的东西粘起来——早先的版本因为第二个网址以小写字母开头就把两
        # 个网址合并了，还把「iii」和「v」变成了「iii v」。
        if stripped.endswith("-") and not stripped.lower().startswith("http"):
            whole = stripped[:-1] + following
        elif _CONTINUATION.match(following) and not stripped.lower().startswith("http"):
            whole = f"{stripped} {following}"
        else:
            continue
        joined[index] = joined[index + 1] = whole
    return joined


def _target(link, names, offset: float) -> dict | None:
    """
    中文：把源链接的目标转换成产物里可用的目标。

    Translate a link's destination into one the output can express.

    Named destinations are resolved to a page and a point, because the name tree
    is not carried over: a link to an undefined name silently does nothing.

    Args:
        link (dict): Link as returned by :meth:`Page.get_links`.
        names (dict): Result of :meth:`Document.resolve_names` on the source.
        offset (float): Horizontal shift applied to an internal destination, so
            a link on the translated half of a dual page lands on that half.

    Returns:
        dict | None: A link dictionary ready for :meth:`Page.insert_link`,
            without its ``from`` rectangle, or ``None`` when the destination
            cannot be expressed.
    """
    import pymupdf

    kind = link.get("kind")

    if kind == pymupdf.LINK_URI:
        return {"kind": pymupdf.LINK_URI, "uri": link.get("uri", "")}

    if kind == pymupdf.LINK_GOTO:
        point = link.get("to") or pymupdf.Point(0, 0)
        return {
            "kind": pymupdf.LINK_GOTO,
            "page": int(link.get("page", 0)),
            "to": pymupdf.Point(point.x + offset, point.y),
        }

    if kind == pymupdf.LINK_NAMED:
        entry = names.get(link.get("name") or link.get("nameddest") or "")
        if not entry:
            return None
        page = entry.get("page")
        if page is None or page < 0:
            return None
        to = entry.get("to")
        x = (to.x if hasattr(to, "x") else to[0]) if to else 0.0
        y = (to.y if hasattr(to, "y") else to[1]) if to else 0.0
        return {
            "kind": pymupdf.LINK_GOTO,
            "page": int(page),
            "to": pymupdf.Point(x + offset, y),
        }

    return None


# The blue LaTeX's hyperref gives a cross-reference, as PDF fill and stroke
# colour operators. Matching the source's own colour is the point, and every
# paper in the test set uses this one.
# hyperref 给交叉引用上的那种蓝色，写成 PDF 的填充与描边颜色算子。目的就是与源文
# 一致，测试集中的论文用的都是这个颜色。
def _colour_operators(colour: int) -> bytes:
    """
    中文：把打包的 RGB 颜色写成 PDF 的填充与描边算子。

    Render a packed colour as PDF fill and stroke colour operators.

    Args:
        colour (int): Packed ``0xRRGGBB`` colour.

    Returns:
        bytes: e.g. ``b"0 0.604 0.333 rg 0 0.604 0.333 RG "`` for #009a55.
    """
    red, green, blue = (colour >> 16) & 255, (colour >> 8) & 255, colour & 255
    triple = f"{red / 255:.4g} {green / 255:.4g} {blue / 255:.4g}"
    return f"{triple} rg {triple} RG ".encode("latin-1")

# One glyph as BabelDOC writes it. Each is its own ``q … Q`` block:
#
#     q /TRP1 gs 0 g 0 G \nBT /F74 9.962600 Tf 1 0 0 1 108.0 756.631 Tm <4C> Tj ET Q
#
# What is matched is the text-positioning part, and the colour is inserted just
# in front of ``BT``. Inserting rather than replacing is what makes this robust:
# the graphic state in between is a passthrough string copied verbatim from the
# source and may hold an ``ExtGState`` reference, a colour, both or neither, and
# PDF colour operators are state — the last one before the text wins. The
# enclosing ``q … Q`` is what keeps the injected colour from leaking onto the
# next glyph.
#
# An earlier version matched an optional colour right after ``q`` and a baseline
# in ``TD``. Neither holds any more: BabelDOC writes ``/Name gs`` ahead of the
# colour and positions with a full ``Tm`` matrix, so the pattern stopped
# matching entirely and link recolouring silently painted nothing.
# BabelDOC 写出的单个字形，每个都是独立的 q … Q 块（见上）。这里匹配的是定位部分，
# 颜色插在 BT 之前。用插入而不是替换才够稳：中间那段图形状态是从源文原样抄来的字符
# 串，可能含 ExtGState 引用、可能含颜色、也可能都没有；而 PDF 的颜色算子是状态，文字
# 之前最后一个生效。外层的 q … Q 保证注入的颜色不会漏到下一个字形。
#
# 旧版本匹配的是 q 之后的可选颜色和 TD 里的基线，这两点如今都不成立：BabelDOC 会在
# 颜色前写 /Name gs，并且用完整的 Tm 矩阵定位——于是整条正则再也匹配不上，链接着色
# 无声地什么也没做。
_GLYPH_BLOCK = re.compile(
    rb"BT\s*/(?P<font>[^\s/]+)\s+(?P<size>[\d.]+)\s+Tf\s+"
    rb"(?P<matrix>(?:-?[\d.]+\s+){2,6})(?P<op>Tm|TD)\s"
)


def _glyph_origin(match) -> tuple[float, float]:
    """
    中文：从字形块的定位算子里取出基线坐标。

    Read a glyph block's baseline out of its positioning operator.

    ``Tm`` carries a full matrix (``1 0 0 1 x y`` for horizontal text,
    ``0 1 -1 0 x y`` for vertical) and ``TD`` carries the offset alone, so in
    both cases the position is the last two numbers.

    Args:
        match (re.Match): A :data:`_GLYPH_BLOCK` match.

    Returns:
        tuple[float, float]: The baseline ``(x, y)`` in stream coordinates.
    """
    numbers = match.group("matrix").split()
    return float(numbers[-2]), float(numbers[-1])

# How close a block's baseline has to be to the glyph's, in points. The two come
# from the same number written twice, so this only absorbs rounding.
# 块的基线与字形基线的允许偏差（点）。两者本是同一个数写了两遍，该容差只吸收舍入。
_BASELINE_SLACK = 0.2

# Allowed difference between the block's font size and the span's.
# 块的字号与 span 字号的允许偏差。
_SIZE_SLACK = 0.05

# Punctuation that stays inside a number run, so "7.18" and "4–5" are coloured
# whole rather than as digits with a black mark between them.
# 保留在数字串内部的标点，使 "7.18"、"4–5" 整体着色，而不是数字变蓝、中间的符号仍黑。
_NUMBER_GLUE = ".-–—/"


def _normalise_font(name: str) -> str:
    """
    中文：把字体名归一化，便于跨「资源名」与「span 名」比较。

    Reduce a font name to a form comparable across naming conventions.

    A content stream names the font by its resource name
    (``NotoSerif-Regular.ttf``); text extraction reports it by its display name
    (``Noto Serif Regular``). Stripping everything but letters and digits, and
    dropping a subset prefix, makes the two comparable.

    Args:
        name (str): Either form of the name.

    Returns:
        str: Lower-case alphanumerics only.
    """
    if "+" in name:
        name = name.split("+", 1)[1]
    return "".join(ch for ch in name.lower() if ch.isalnum())


def _translation_streams(document, page) -> list:
    """
    中文：列出该页可编辑的内容流，以及每个流到页面坐标的平移量。

    List the page's editable content streams with their offset to page space.

    A mono page holds its glyphs in its own content stream. A side-by-side dual
    page holds nothing but two ``Do`` operators: the translated half is a Form
    XObject placed with ``Matrix [1 0 0 1 612 0]``, and the glyphs live inside
    it — one more level down, because that form wraps another. Walking the forms
    and accumulating their matrices is what lets the same code recolour both.

    Forms whose matrix is not a pure translation are skipped rather than guessed
    at; none of the placements BabelDOC produces needs one.

    Args:
        document (pymupdf.Document): The translated document.
        page (pymupdf.Page): The page to walk.

    Returns:
        list[dict]: One entry per stream with ``xref``, ``dx``, ``dy`` — the
            translation from the stream's coordinates into page space — and
            ``data``, the stream's bytes.
    """
    def _read(xref):
        try:
            return document.xref_stream(xref)
        except Exception:  # noqa: BLE001
            return None

    streams = []
    for xref in page.get_contents():
        data = _read(xref)
        if data:
            streams.append({"xref": xref, "dx": 0.0, "dy": 0.0, "data": data})

    try:
        entries = page.get_xobjects()
    except Exception:  # noqa: BLE001
        return streams

    offsets = {0: (0.0, 0.0)}
    for _ in range(4):
        progressed = False
        for xref, _name, invoker, _bbox in entries:
            if xref in offsets or invoker not in offsets:
                continue
            kind, value = document.xref_get_key(xref, "Matrix")
            if kind == "array":
                numbers = value.strip("[] ").split()
                if len(numbers) != 6:
                    continue
                try:
                    a, b, c, d, e, f = (float(n) for n in numbers)
                except ValueError:
                    continue
                if (a, b, c, d) != (1.0, 0.0, 0.0, 1.0):
                    continue
            else:
                # A form with no /Matrix is placed unchanged.
                # 没有 /Matrix 的 form 按原样放置。
                e = f = 0.0
            base = offsets[invoker]
            offsets[xref] = (base[0] + e, base[1] + f)
            progressed = True
            data = _read(xref)
            if data:
                streams.append({"xref": xref, "dx": base[0] + e,
                                "dy": base[1] + f, "data": data})
        if not progressed:
            break
    return streams


def _link_glyphs(page, rect, chars_cache) -> list:
    """
    中文：取出链接矩形内的全部字符（空白除外），用于整体着色。

    Select the characters inside a link rectangle.

    Every glyph is taken, not only the digits. The rectangle handed in is always
    one that a text search returned, so it covers exactly the string the link was
    re-anchored on — colouring all of it reproduces what hyperref does in the
    source, where the whole of "Section 4" is coloured and not just the 4. The
    earlier digits-only rule left 「第4节」 with one coloured character, and left
    an author citation such as 「Watson等人」 with none at all.

    Args:
        page (pymupdf.Page): Translated page.
        rect (pymupdf.Rect): The link's rectangle on that page.
        chars_cache (dict): Per-page memo of ``(char, span)`` pairs.

    Returns:
        list[tuple[dict, dict]]: The character and its span, in reading order.
    """
    key = page.number
    if key not in chars_cache:
        collected = []
        try:
            blocks = page.get_text("rawdict")["blocks"]
        except Exception:  # noqa: BLE001
            blocks = []
        for block in blocks:
            if block.get("type") != 0:
                continue
            for line in block.get("lines", ()):
                for span in line.get("spans", ()):
                    for char in span.get("chars", ()):
                        collected.append((char, span))
        chars_cache[key] = collected

    inside = []
    for char, span in chars_cache[key]:
        box = pymupdf.Rect(char["bbox"])
        if box.is_empty:
            continue
        overlap = box & rect
        if overlap.is_empty or overlap.get_area() < box.get_area() * _CHAR_OVERLAP:
            continue
        inside.append((char, span))
    inside.sort(key=lambda pair: (round(pair[0]["origin"][1], 1), pair[0]["origin"][0]))

    return [(char, span) for char, span in inside
            if (char.get("c") or "") and not (char.get("c") or "").isspace()]


def _paint_link(document, page, rect, colour, chars_cache, stream_cache) -> int:
    """
    中文：把链接矩形内的字符染成源文该链接自己的颜色。

    Colour a link's text the colour the source draws that link in.

    In the source, hyperref paints every reference, which is how a reader sees
    that it is clickable. BabelDOC writes the translated glyphs with no colour at
    all, so the restored links were invisible: 「（见第4节）」 carried a working
    link and looked like ordinary text. The colour is the source link's own —
    a paper that cites in green and links URLs in blue keeps both.

    Each glyph BabelDOC emits is its own ``q … Q`` block with its baseline
    written out in ``TD``, so the colour is set by inserting two operators into
    the block that draws the glyph — no glyph is moved, no text is re-inserted,
    and a mistaken match can only miscolour one character.

    Args:
        document (pymupdf.Document): The translated document, edited in place.
        page (pymupdf.Page): The page the link is on.
        rect (pymupdf.Rect): The link's rectangle on that page.
        colour (int): Packed ``0xRRGGBB`` colour to paint, from the source link.
        chars_cache (dict): Per-page memo for :func:`_link_glyphs`.
        stream_cache (dict): Per-page memo of content streams and their offsets.

    Returns:
        int: Number of glyphs recoloured.
    """
    glyphs = _link_glyphs(page, rect, chars_cache)
    if not glyphs:
        return 0

    if page.number not in stream_cache:
        stream_cache[page.number] = _translation_streams(document, page)
    streams = stream_cache[page.number]
    if not streams:
        return 0

    height = page.rect.height
    operators = _colour_operators(colour)
    painted = 0

    for char, span in glyphs:
        wanted = _normalise_font(span.get("font") or "")
        size = float(span.get("size") or 0)
        origin_x, origin_y = char["origin"]

        found = None
        ambiguous = False
        for stream in streams:
            target_x = origin_x - stream["dx"]
            target_y = height - origin_y - stream["dy"]
            for match in _GLYPH_BLOCK.finditer(stream["data"]):
                block_x, block_y = _glyph_origin(match)
                if abs(block_x - target_x) > _BASELINE_SLACK:
                    continue
                if abs(block_y - target_y) > _BASELINE_SLACK:
                    continue
                if abs(float(match.group("size")) - size) > _SIZE_SLACK:
                    continue
                name = _normalise_font(match.group("font").decode("latin-1"))
                if wanted and not (name.startswith(wanted) or wanted.startswith(name)):
                    continue
                if found is not None:
                    # Two blocks answer to the same description; recolouring
                    # either would be a guess.
                    # 有两个块符合同一描述，改哪个都是猜。
                    ambiguous = True
                    break
                found = (stream, match.start())
            if ambiguous or found is not None:
                break

        if ambiguous or found is None:
            continue
        stream, start = found
        if stream["data"][max(0, start - len(operators)):start] == operators:
            # Already carrying the link colour; nothing to do.
            # 已经带着链接色了，不必处理。
            continue
        stream["data"] = stream["data"][:start] + operators + stream["data"][start:]
        stream["dirty"] = True
        painted += 1

    for stream in streams:
        if not stream.pop("dirty", False):
            continue
        try:
            document.update_stream(stream["xref"], stream["data"])
        except Exception:  # noqa: BLE001
            logger.debug("Could not recolour glyphs in stream %d", stream["xref"],
                         exc_info=True)

    return painted


def restore_links(source_path: Path | str, target_path: Path | str,
                  dual: bool = False, translate_first: bool = False,
                  regions: dict | None = None) -> dict[str, int]:
    """
    中文：把源 PDF 的超链接重新贴到译文产物上。

    Reinstate the source document's hyperlinks on a translated PDF.

    Args:
        source_path (Path | str): Original PDF.
        target_path (Path | str): Translated PDF, modified in place.
        dual (bool): The target is a side-by-side dual PDF, twice as wide, with
            the original beside the translation.
        translate_first (bool): The dual layout puts the translation on the left.
        regions (dict | None): Page index to regions copied verbatim from the
            source, as returned by
            :func:`local_translator.engine.regions.find_protected_regions`. A
            link inside one of these is placed by geometry rather than searched
            for.

    Returns:
        dict[str, int]: ``links`` inserted, ``dropped`` links whose text could
            not be located, ``coloured`` glyphs painted the source's own link
            colour, and ``pages`` touched.
    """
    import pymupdf

    result = {"links": 0, "dropped": 0, "pages": 0, "coloured": 0}
    source_path, target_path = Path(source_path), Path(target_path)
    if not source_path.is_file() or not target_path.is_file():
        return result

    try:
        source = pymupdf.open(source_path)
        target = pymupdf.open(target_path)
    except Exception:  # noqa: BLE001
        logger.warning("Could not open PDFs for link restoration", exc_info=True)
        return result

    try:
        try:
            names = source.resolve_names()
        except Exception:  # noqa: BLE001
            logger.debug("Could not resolve named destinations", exc_info=True)
            names = {}

        # Read off this document's link colours once. They differ per paper and
        # per link kind, and both the anchoring and the recolouring need them.
        # 先把本文档的链接颜色读出来，只做一次。它随论文、随链接种类而不同，定位与
        # 重新着色两处都要用。
        palette = link_palette(source)

        touched = False
        for index in range(min(source.page_count, target.page_count)):
            src_page, dst_page = source[index], target[index]
            links = src_page.get_links()
            if not links:
                continue

            width = src_page.rect.width
            if dual:
                half = dst_page.rect.width / 2.0
                original_offset = half if translate_first else 0.0
                translated_offset = 0.0 if translate_first else half
            else:
                original_offset = None
                translated_offset = 0.0

            cache: dict = {"palette": palette}
            chars_cache: dict = {}
            stream_cache: dict = {}
            before = result["links"]
            page_regions = regions.get(index, ()) if regions else ()

            needles = effective_needles(src_page, links)

            for position, link in enumerate(links):
                rect = link["from"]

                # Inside a region copied from the source verbatim, the geometry
                # is exact by construction — the pixels there *are* the source's
                # — so the link belongs exactly where it was. This is how the
                # contents page keeps its links: its entries carry no separate
                # text in the output to search for, and 26 of this document's
                # links sit on such regions.
                # 在原样搬回的区块内，几何关系按构造就是精确的——那里的画面就是源文本身
                # ——链接理应留在原位。目录页的链接正是靠这一条保住的：它的条目在输出中
                # 没有可供搜索的独立文字，本文档有 26 个链接落在这类区块上。
                in_region = any(area.intersects(rect) for area in page_regions)

                # The original half of a dual page is the source page itself, so
                # its links need no re-anchoring at all — only a shift.
                # 双语页的原文半边就是源页本身，链接无需重定位，平移即可。
                if original_offset is not None:
                    target_spec = _target(link, names, original_offset)
                    if target_spec is not None:
                        target_spec["from"] = rect + (original_offset, 0,
                                                      original_offset, 0)
                        try:
                            dst_page.insert_link(target_spec)
                            result["links"] += 1
                        except Exception:  # noqa: BLE001
                            logger.debug("Could not insert original-half link on page %d",
                                         index + 1, exc_info=True)

                if in_region:
                    anchored = [rect + (translated_offset, 0, translated_offset, 0)]
                elif dual:
                    anchored = _anchor_dual(src_page, dst_page, rect, cache,
                                            translated_offset, width,
                                            needles[position])
                else:
                    anchored = _anchor(src_page, dst_page, rect, cache,
                                       needles[position])
                if not anchored:
                    result["dropped"] += 1
                    continue

                target_spec = _target(link, names, translated_offset)
                if target_spec is None:
                    result["dropped"] += 1
                    continue

                # A citation that wraps gets one link per line. A single
                # rectangle spanning both would also cover the unrelated text
                # between them, making it clickable and — since a link's text is
                # recoloured — painting a stretch of Chinese in the citation
                # colour.
                # 换行的引用按行各给一个链接。用一个跨行的大矩形会把两行之间无关的
                # 文字也框进去，既让它变成可点，又因为链接文字要重新着色而把一段中文
                # 染成引用色。
                inserted = 0
                for part in anchored:
                    spec = dict(target_spec)
                    spec["from"] = part
                    try:
                        dst_page.insert_link(spec)
                        inserted += 1
                    except Exception:  # noqa: BLE001
                        logger.debug("Could not insert link on page %d", index + 1,
                                     exc_info=True)
                if not inserted:
                    result["dropped"] += 1
                    continue
                result["links"] += 1

                # A restored link that looks like ordinary text is half a
                # repair: in the source the reader sees it is clickable because
                # hyperref painted it in the document's link colour.
                # 恢复了却看不出是链接，等于只修了一半：源文里读者之所以知道能点，
                # 是因为 hyperref 把它染成了该文档的链接色。
                colour = _source_colour(src_page, rect, cache)
                for part in anchored:
                    try:
                        result["coloured"] += _paint_link(
                            target, dst_page, part, colour, chars_cache, stream_cache,
                        )
                    except Exception:  # noqa: BLE001
                        logger.debug("Could not recolour link text on page %d",
                                     index + 1, exc_info=True)

            if result["links"] > before:
                result["pages"] += 1
                touched = True

        if touched:
            temp = target_path.with_suffix(target_path.suffix + ".linkfix")
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
            logger.info(
                "Restored %d link(s) across %d page(s) in %s "
                "(%d dropped, %d glyph(s) coloured)",
                result["links"], result["pages"], target_path.name,
                result["dropped"], result["coloured"],
            )
    finally:
        for document in (source, target):
            if not document.is_closed:
                document.close()

    return result


def _anchor_dual(src_page, dst_page, rect, cache, offset: float, width: float,
                 needle: str | None = None):
    """
    中文：在双语页的译文半边上定位链接。

    Locate a link on the translated half of a side-by-side page.

    The search has to be confined to that half: the other half is the original,
    where every needle also occurs, and an unconfined search would return the
    two halves interleaved and make the occurrence index meaningless.

    Args:
        src_page (pymupdf.Page): Source page.
        dst_page (pymupdf.Page): Dual page.
        rect (pymupdf.Rect): Link rectangle in source coordinates.
        cache (dict): Per-page search memo.
        offset (float): Left edge of the translated half.
        width (float): Width of one half.
        needle (str | None): Text to look for, overriding what sits under the
            rectangle.

    Returns:
        list[pymupdf.Rect] | None: One rectangle per line the link covers on
            the dual page, or ``None``.
    """
    import pymupdf

    clip = pymupdf.Rect(offset, 0, offset + width, dst_page.rect.height)

    needle = needle if needle is not None else _needle(src_page, rect)

    by_colour = _anchor_by_colour(src_page, dst_page, rect, cache, clip)
    if by_colour is not None and (
            not needle or _plausible(dst_page, by_colour, needle, cache.setdefault(
                "glyphs", {}))):
        return [by_colour]

    if not needle:
        # No needle to search with, so an unverified colour match is the only
        # candidate there is.
        # 没有待查串可搜，未经校验的颜色匹配是唯一的候选。
        return [by_colour] if by_colour is not None else None

    candidates = []
    fragment = _citation_fragment(src_page, rect)
    if fragment and fragment != needle:
        candidates.append(fragment)
    candidates.append(needle)

    src_parts = dst_parts = []
    for candidate in candidates:
        src_parts = _find_fragments(src_page, candidate, cache, "src_chars")
        dst_parts = _find_fragments(dst_page, candidate, cache, "dst_chars_half", clip)
        if dst_parts:
            break
    src_hits = [_span(one) for one in src_parts]
    dst_hits = [_span(one) for one in dst_parts]
    if not dst_hits:
        # Same fallback as the single-page path: a cross-reference is written in
        # Chinese on the translated half, so the English never matches there.
        # 与单页路径相同的回退：交叉引用在译文半边是中文写法，英文串在那里永远匹配不到。
        for form in _translated_forms(needle):
            parts = _find_fragments(dst_page, form, cache, "dst_chars_half", clip)
            if parts:
                dst_parts = parts
                dst_hits = [_span(one) for one in parts]
                break
        if not dst_hits:
            # Nothing was found by text either. A colour match that failed the
            # plausibility check is imprecise, not useless — on a bibliography
            # entry it lands on the journal name rather than on the year, and
            # the link still works. Dropping it loses the link outright.
            # 文本也没找到。未通过校验的颜色匹配只是不够精确，并非无用——在参考文献
            # 条目上它落在期刊名而不是年份上，链接照样能点。丢弃反而是彻底失去它。
            return [by_colour] if by_colour is not None else None

    shifted = rect + (offset, 0, offset, 0)
    rank = _rank(src_hits, rect)
    if rank is not None and len(src_hits) == len(dst_hits):
        return dst_parts[rank]
    nearest = _nearest(dst_hits, shifted)
    if nearest is None:
        return None
    return dst_parts[dst_hits.index(nearest)]


def restore_all(source_path: Path | str, outputs: dict[str, str | None],
                translate_first: bool = False,
                regions: dict | None = None) -> dict[str, int]:
    """
    中文：对所有产物执行超链接修复。

    Reinstate links on every produced PDF.

    Args:
        source_path (Path | str): Original PDF.
        outputs (dict[str, str | None]): Produced paths keyed by kind.
        translate_first (bool): The dual layout puts the translation on the left.

    Returns:
        dict[str, int]: Combined counts across repaired files, with the same
            keys :func:`restore_links` reports.
    """
    totals = {"links": 0, "dropped": 0, "pages": 0, "coloured": 0}

    mono = outputs.get("mono_path")
    if mono:
        for key, value in restore_links(source_path, mono, regions=regions).items():
            totals[key] += value

    dual = outputs.get("dual_path")
    if dual:
        for key, value in restore_links(source_path, dual, dual=True,
                                        translate_first=translate_first,
                                        regions=regions).items():
            totals[key] += value

    return totals
