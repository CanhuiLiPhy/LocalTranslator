"""
中文：把引用方括号整体并成一个公式——`[17, Eq. 7.18]` 里的内容不该被翻译。

Keep a reference bracket whole, so nothing inside it is translated.

**The symptom.** ``(see also Higham [17, Eq. 7.18])`` came back as
「亦可参考Higham [17, 的方程7 18]）」: the abbreviation was translated, the decimal
point vanished, and the closing bracket ended up outside the citation.

**Why it happens.** BabelDOC classifies characters as formula or text one at a
time, and a citation is only a formula while it is made of brackets, digits and
punctuation. ``Eq.`` is ordinary text, so the citation is cut into five
alternating pieces::

    [14] F '[17, '
    [15] T 'Eq. '
    [16] F '7'
    [17] T '.'
    [18] F '18]'

The two text pieces are sent to the translator on their own, with no sign that
they are the inside of a citation. ``Eq. 7.`` becomes 「的方程7 」 and the pieces
are then laid out independently, which is what breaks the alignment as well.

**The fix.** After formula parsing, a formula that opens a bracket absorbs the
pieces that follow it until the bracket closes, provided the whole thing reads
like a reference: it opens with a number, it is short, and every word in it is a
reference abbreviation (``Eq.``, ``Sec.``, ``Thm.``, ``pp.``, …). The citation
then reaches the translator as a single placeholder and comes back untouched.

PDFMathTranslate does the same thing with a bracket counter (``vbkt``) while it
extracts, which keeps ``(1)`` from being split. This runs a step later, on
compositions rather than characters, because that is where BabelDOC leaves the
seam.

**Why the word list.** Without it, ``[3, see the discussion of convergence]``
would be absorbed and left in English. Requiring every word to be a known
reference abbreviation keeps the rule to what it is meant for; a bracket holding
a sentence still goes to the translator.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# Words allowed inside a reference bracket. Anything else means the bracket
# holds prose, which must still be translated.
# 引用方括号内允许出现的词。出现其它词说明括号里是正文，仍需翻译。
_REFERENCE_WORDS = frozenset({
    "alg", "algorithm", "algorithms", "and", "app", "appendix", "appendices",
    "ch", "chap", "chapter", "chapters", "cf", "cor", "corollary",
    "def", "definition", "eq", "eqn", "eqs", "equation", "equations",
    "et", "al", "ex", "example", "examples", "fig", "figs", "figure", "figures",
    "ff", "item", "lemma", "lemmas", "line", "lines", "note", "p", "pp",
    "page", "pages", "part", "prop", "proposition", "remark", "sec", "secs",
    "section", "sections", "see", "also", "step", "steps",
    "table", "tables", "thm", "theorem", "theorems",
})

# Words in the bracket, for the check above.
# 括号内的单词，供上面的判据使用。
_WORD = re.compile(r"[A-Za-z]+")

# A reference bracket opens with a number: "[17, Eq. 7.18]", "[3]". A bracket
# opening with a word is a note or an editorial insertion.
# 引用方括号以数字开头："[17, Eq. 7.18]"、"[3]"。以单词开头的是注解或编者插入语。
_OPENS_REFERENCE = re.compile(r"^\[\s*\d")

# Longest bracket worth absorbing, in characters. A citation list runs to maybe
# thirty characters; past this it is prose.
# 值得吸收的方括号长度上限（字符）。一条引文最多三十来个字符，超出就是正文。
_MAX_LENGTH = 48

# Most compositions a bracket may span. Each alternation of formula and text
# costs two, so this allows several abbreviations inside one citation.
# 一个方括号最多跨越的组成部分数。公式与文字每交替一次占两个，这个上限容得下一条
# 引文里的若干个缩写。
_MAX_PIECES = 9

# Vertical distance between two pieces that still counts as the same line.
# 两个部分之间仍算同一行的垂直距离。
_SAME_LINE = 4.0


def _characters(composition):
    """
    中文：取出一个组成部分的字符列表，取不到返回 None。

    Return a composition's characters.

    Args:
        composition (PdfParagraphComposition): One piece of a paragraph.

    Returns:
        list | None: Its characters, or ``None`` when the composition is
            neither a formula nor a line and therefore cannot be merged.
    """
    if composition.pdf_formula is not None:
        return composition.pdf_formula.pdf_character
    if composition.pdf_line is not None:
        return composition.pdf_line.pdf_character
    return None


def _text(characters) -> str:
    """
    中文：把字符列表还原成字符串。

    Join characters back into a string.

    Args:
        characters (list): Characters to read.

    Returns:
        str: Their Unicode values in order, with unmapped glyphs dropped.
    """
    return "".join(char.char_unicode or "" for char in characters)


def _centre(char) -> float:
    """
    中文：取字符的垂直中心。

    Return a character's vertical centre.

    Args:
        char (PdfCharacter): The character to measure.

    Returns:
        float: Midpoint of its box in the y direction.
    """
    box = char.box
    return (box.y + box.y2) / 2.0


def _is_reference(text: str) -> bool:
    """
    中文：判断括号内容是否是一条引用（而非需要翻译的正文）。

    Decide whether a bracket's contents are a reference.

    Args:
        text (str): The bracket including both delimiters.

    Returns:
        bool: ``True`` when the bracket may be kept verbatim.
    """
    if len(text) > _MAX_LENGTH or not _OPENS_REFERENCE.match(text):
        return False
    return all(word.lower() in _REFERENCE_WORDS for word in _WORD.findall(text))


def merge_reference_brackets(page) -> int:
    """
    中文：把页面上每个被切开的引用方括号并回一个公式。

    Rejoin every reference bracket that formula parsing cut apart.

    Args:
        page (Page): One page of BabelDOC's intermediate layout, after
            ``process_page_formulas`` has run.

    Returns:
        int: Number of brackets merged.
    """
    from babeldoc.format.pdf.document_il.il_version_1 import (
        PdfFormula,
        PdfParagraphComposition,
    )
    from babeldoc.format.pdf.document_il.utils.formular_helper import (
        update_formula_data,
    )

    merged = 0
    for paragraph in getattr(page, "pdf_paragraph", None) or []:
        compositions = paragraph.pdf_paragraph_composition
        if not compositions:
            continue

        result = []
        index = 0
        while index < len(compositions):
            composition = compositions[index]
            formula = composition.pdf_formula
            opened = _text(formula.pdf_character) if formula is not None else ""
            if formula is None or opened.count("[") <= opened.count("]"):
                result.append(composition)
                index += 1
                continue

            characters = list(formula.pdf_character)
            end = None
            for position in range(index + 1, min(index + _MAX_PIECES, len(compositions))):
                following = compositions[position]
                more = _characters(following)
                if not more:
                    break
                if abs(_centre(more[0]) - _centre(characters[-1])) > _SAME_LINE:
                    break
                characters.extend(more)
                text = _text(characters)
                if text.count("[") <= text.count("]"):
                    # The bracket closes here. Only a formula may close it: a
                    # text composition would carry the sentence after the
                    # citation into the placeholder along with it.
                    # 括号在此闭合。只允许公式收尾：若是文字组成部分，会把引文之后的
                    # 句子一并卷进占位符。
                    if following.pdf_formula is not None:
                        end = position
                    break

            if end is None or end == index:
                result.append(composition)
                index += 1
                continue

            text = _text(characters)
            start = text.find("[")
            stop = text.rfind("]")
            if start < 0 or stop < start or not _is_reference(text[start:stop + 1]):
                result.append(composition)
                index += 1
                continue

            replacement = PdfFormula(pdf_character=characters,
                                     line_id=formula.line_id)
            update_formula_data(replacement)
            result.append(PdfParagraphComposition(pdf_formula=replacement))
            merged += 1
            index = end + 1

        if len(result) != len(compositions):
            paragraph.pdf_paragraph_composition = result

    return merged


def install() -> None:
    """
    中文：在 BabelDOC 的公式识别之后补上引用方括号的合并。

    Rejoin reference brackets right after BabelDOC parses formulas.

    Wraps ``StylesAndFormulas.process_page_formulas`` so the merge happens
    before offsets, styles and the translator see the paragraph. Idempotent.

    Raises:
        AttributeError: If BabelDOC no longer exposes the method, so an upgrade
            that moves it fails loudly at startup instead of silently doing
            nothing.
    """
    from babeldoc.format.pdf.document_il.midend.styles_and_formulas import (
        StylesAndFormulas,
    )

    if getattr(StylesAndFormulas.process_page_formulas, "_brackets_patched", False):
        return

    original = StylesAndFormulas.process_page_formulas

    def process_page_formulas(self, page):
        """中文：先跑原逻辑，再合并引用方括号。 / Run the original pass, then merge brackets."""
        original(self, page)
        try:
            merged = merge_reference_brackets(page)
        except Exception:  # noqa: BLE001
            # Failing here must not cost the page: the worst outcome of skipping
            # the merge is the split citation we had before.
            # 这里失败不应赔上整页：跳过合并最坏也只是回到原来被切开的引文。
            logger.warning("Reference-bracket merging failed on a page", exc_info=True)
        else:
            if merged:
                logger.debug("Merged %d reference bracket(s)", merged)

    process_page_formulas._brackets_patched = True
    StylesAndFormulas.process_page_formulas = process_page_formulas
