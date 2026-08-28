"""
中文：数学记号里的逗号不是顿号——不要把下标里的逗号交给翻译模型。

Keep the commas inside a piece of notation out of the translator's hands.

**The symptom.** ``Let θd,N,α,e denote the model obtained from configuration
(d, N, α, e)`` came back as 「令 θd、N、α、e 表示由配置 (d、N、α、e)得到的模型」.
Each 2.4pt Latin comma had become a 10pt ideographic one, so a subscript that
occupied 35pt in the source occupied 60pt in the translation and no longer read
as a subscript at all.

**Why it happens.** BabelDOC's ``process_comma_formulas`` splits any formula
that contains a comma *and* anything other than digits and brackets, emitting
each comma as an ordinary text line between the pieces. The translator therefore
receives ``{v1},{v2},{v3},{v4}`` and, quite reasonably, punctuates the Chinese
sentence it is writing the Chinese way.

The rule makes sense for an enumeration of symbols in running prose, where 「、」
is what a Chinese reader expects. It is wrong for notation, where the comma is
part of the symbol.

**Telling them apart.** Two signals separate the two, both taken from the source
geometry rather than from the sentence:

* **A comma with no space after it.** ``θd,N,α,e`` has none — comma and letter
  are adjacent characters. An enumeration in prose is set with a space:
  ``d, N, α, e`` carries a space glyph after every comma.
* **A formula wrapped in brackets by the text around it.** ``(d, N, α, e)`` is a
  tuple, not a list of three things the sentence is talking about. BabelDOC
  already refuses to split at a comma inside a bracket; it simply cannot see
  these brackets, because they are text compositions on either side of the
  formula rather than characters within it.

Formulas matching either signal are left whole, so their commas are restored
exactly as they were drawn. Everything else keeps BabelDOC's behaviour.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Bracket pairs, in the order BabelDOC lists them, so "the text opens with the
# bracket this formula closes with" is decided the same way it decides it.
# 括号对，顺序与 BabelDOC 一致，使「文字里的开括号与公式后的闭括号配对」这一判断
# 与它自身的判断保持一致。
_PAIRS: dict[str, str] = {}


def _pairs() -> dict[str, str]:
    """
    中文：返回左括号到右括号的映射，首次调用时从 BabelDOC 读取。

    Map each opening bracket to its closing one.

    Returns:
        dict[str, str]: Opening bracket to closing bracket, taken from
            BabelDOC's own ``LEFT_BRACKET`` and ``RIGHT_BRACKET`` tuples so the
            two never disagree.
    """
    if not _PAIRS:
        from babeldoc.format.pdf.document_il.midend.styles_and_formulas import (
            LEFT_BRACKET,
            RIGHT_BRACKET,
        )

        _PAIRS.update(zip(LEFT_BRACKET, RIGHT_BRACKET, strict=False))
    return _PAIRS


def _text(characters) -> str:
    """
    中文：把字符列表还原成字符串。

    Join characters back into a string.

    Args:
        characters (list): Characters to read.

    Returns:
        str: Their Unicode values in order.
    """
    return "".join(char.char_unicode or "" for char in characters)


def _neighbour_text(composition) -> str:
    """
    中文：取相邻组成部分的文字，取不到返回空串。

    Read the text of a neighbouring composition.

    Args:
        composition (PdfParagraphComposition | None): The neighbour.

    Returns:
        str: Its characters, or ``""`` when it holds none.
    """
    if composition is None:
        return ""
    if composition.pdf_line is not None:
        return _text(composition.pdf_line.pdf_character)
    if composition.pdf_formula is not None:
        return _text(composition.pdf_formula.pdf_character)
    if composition.pdf_character is not None:
        return composition.pdf_character.char_unicode or ""
    return ""


def _bracketed(before: str, after: str) -> bool:
    """
    中文：判断公式是否被两侧文字里的一对括号夹住。

    Decide whether the text around a formula brackets it.

    Args:
        before (str): Text of the composition immediately before the formula.
        after (str): Text of the composition immediately after it.

    Returns:
        bool: ``True`` when ``before`` ends with an opening bracket whose
            closing partner starts ``after``. The two are checked as a pair, so
            a sentence that merely happens to end in ``(`` does not qualify.
    """
    if not before or not after:
        return False
    closing = _pairs().get(before[-1])
    return closing is not None and after[0] == closing


def _commas_are_notation(formula) -> bool:
    """
    中文：判断公式里的逗号是否都属于记号本身（后面不跟空格）。

    Decide whether every comma in a formula belongs to the notation.

    Commas inside brackets are ignored, because BabelDOC does not split at those
    either. A comma counts as notation when the character after it is not a
    space — ``θd,N`` rather than ``d, N``.

    Args:
        formula (PdfFormula): The formula to inspect.

    Returns:
        bool: ``True`` when the formula has at least one top-level comma and
            none of them is followed by a space.
    """
    pairs = _pairs()
    characters = formula.pdf_character or []
    depth = 0
    found = False
    for index, char in enumerate(characters):
        value = char.char_unicode
        if value in pairs:
            depth += 1
        elif value in pairs.values():
            depth = max(0, depth - 1)
        elif value == "," and depth == 0:
            found = True
            following = characters[index + 1].char_unicode if index + 1 < len(characters) else ""
            if following.isspace():
                return False
    return found


def protected_formulas(stage, page) -> set[int]:
    """
    中文：列出本页中不应按逗号拆开的公式。

    Identify the formulas on a page whose commas must not be split out.

    Args:
        stage (StylesAndFormulas): The stage, used for its own
            ``should_split_formula`` so only formulas it would actually split
            are considered.
        page (Page): One page of BabelDOC's intermediate layout.

    Returns:
        set[int]: ``id()`` of each formula to leave whole.
    """
    from babeldoc.format.pdf.document_il.midend.styles_and_formulas import (
        StylesAndFormulas,
    )

    should_split = StylesAndFormulas.should_split_formula
    protected: set[int] = set()
    for paragraph in getattr(page, "pdf_paragraph", None) or []:
        compositions = paragraph.pdf_paragraph_composition or []
        for index, composition in enumerate(compositions):
            formula = composition.pdf_formula
            if formula is None or not should_split(stage, formula):
                continue
            before = _neighbour_text(compositions[index - 1] if index else None)
            after = _neighbour_text(
                compositions[index + 1] if index + 1 < len(compositions) else None
            )
            if _bracketed(before, after) or _commas_are_notation(formula):
                protected.add(id(formula))
                logger.debug("Keeping notation whole: %r", _text(formula.pdf_character))
    return protected


def install() -> None:
    """
    中文：让按逗号拆公式的那一步跳过数学记号。

    Make BabelDOC's comma splitting skip notation.

    Wraps ``process_comma_formulas`` to work out which formulas to protect
    before it runs, and ``should_split_formula`` to answer ``False`` for those
    while it does. Going through the predicate rather than reimplementing the
    split leaves BabelDOC in charge of every formula this does not protect.

    Idempotent, so a second call is harmless.

    Raises:
        AttributeError: If BabelDOC no longer exposes either method, so an
            upgrade that moves them fails at startup rather than silently doing
            nothing.
    """
    from babeldoc.format.pdf.document_il.midend.styles_and_formulas import (
        StylesAndFormulas,
    )

    if getattr(StylesAndFormulas.process_comma_formulas, "_notation_patched", False):
        return

    original_process = StylesAndFormulas.process_comma_formulas
    original_should = StylesAndFormulas.should_split_formula

    def should_split_formula(self, formula):
        """中文：受保护的记号一律不拆。 / Never split a formula that was marked as notation."""
        if id(formula) in getattr(self, "_notation_protected", ()):
            return False
        return original_should(self, formula)

    def process_comma_formulas(self, page):
        """中文：先挑出记号，再跑原逻辑。 / Pick out the notation, then run the original pass."""
        try:
            self._notation_protected = protected_formulas(self, page)
        except Exception:  # noqa: BLE001
            # Failing here must not cost the page: without the protection the
            # commas are split out, which is how they were before this existed.
            # 这里失败不应赔上整页：没有保护就是把逗号拆出来，也就是本模块出现之前的样子。
            logger.warning("Protecting notation failed on a page", exc_info=True)
            self._notation_protected = set()
        try:
            return original_process(self, page)
        finally:
            self._notation_protected = set()

    process_comma_formulas._notation_patched = True
    StylesAndFormulas.should_split_formula = should_split_formula
    StylesAndFormulas.process_comma_formulas = process_comma_formulas
