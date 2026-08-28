"""
中文：行内公式的碎片各走各的——斜体变量当普通文字送进模型，角标被换行切断后干脆没被认出来。

Keep the pieces of an inline formula together.

**The symptom.** Page 2 of the Nature Communications paper carries the
Halperin-Nelson equation and the variables around it. Translated, it reads::

    g e 随后在 Tonsetc 为 14.8 K 处开始出现超导转变…
    R = R₀ exp[−2 ((T′−T_BKT)/(−T_BKT))^{1/2}] b cT（R₀ 和 b

Three things went wrong. Stray letters ``g e`` opened the paragraph; the
coefficient ``b`` that belongs *inside* the exponent is sitting after the
bracket with a stray ``cT``; and ``T_c^onset`` was flattened to ``Tonsetc``.

**Why — two causes, both about what reaches the model.** Read as the
intermediate representation has it, the paragraph is sprinkled with
one-character runs of ordinary text::

    文字 'T'                      italic     ← the T of T_c^onset
    文字 'onsetc of 14.8 K (Fig. ' upright    ← its own marks, untagged
    文字 'R'  '('  'T'  ')'                   ← R(T), four separate runs
    文字 'b '                     italic     ← the coefficient of the exponent

*Italic variables are not protected at all.* Formulas are recognised by font
name, and this journal's italic face is the ordinary text italic, so every
variable set in it — 332 of them in this paper — is handed to the model as a
bare letter in the middle of English prose, where it is moved or dropped.

*A raised mark is never recognised at all.* Superscripts are found by size: a
character smaller than the one before it is a mark. But this paper does not set
them smaller — it raises the text matrix and leaves the font alone, so the
intermediate representation records ``T`` and its superscript at the *same*
8.22 points and the ratio test can never fire. What separates them is where
they sit::

    'a' 't' ' ' 'T'       box.y = 668.30   ← the line's baseline
    'o' 'n' 's' 'e' 't'   box.y = 671.98   ← raised 3.68, a superscript
    'c'                   box.y = 666.03   ← dropped 2.27, a subscript
    ' ' 'o' 'f'           box.y = 668.30   ← baseline again

A mark is a character off its line's baseline, whatever the file did to put it
there, and that is what is measured here.

**The fix, in the order it has to happen.**

1. A character sitting clear of its line's baseline is a mark, however the file
   raised it. This is what the size test misses.
2. A run of italic that is a single letter is a variable, and is protected.
   Length is what separates it from italic prose: this paper's italic face
   carries 332 single letters, every one a variable, and its longer runs are
   journal names in the bibliography.
3. A lone italic character with formula on both sides belongs to that formula.
   This is the exponent's coefficient — ``exp[−2b(…)]`` — which nothing else
   recognises and which the model otherwise moves past the closing bracket.
4. A mark's base joins it, so ``T_c`` travels as one placeholder rather than a
   bare ``T`` and a placeholder with nothing to connect them.

**Why the base rule cannot simply take whatever precedes a mark.**
``superconductivity²⁶⁻²⁹`` has the identical shape — a protected mark after
ordinary text — and pulling its base in would keep that word in English
forever. A variable's base is one to three characters and a word is not, so the
base is taken only when the run before the mark, bounded by a space or by
punctuation, is short enough to be a variable.
"""

from __future__ import annotations

import collections
import logging
import threading

logger = logging.getLogger(__name__)

# The longest run of characters that may be taken as a mark's base. A variable
# is T, R, Nd, NiO — one to three characters. A word carrying a citation is
# longer, and taking it would stop the word being translated.
# 可以当作角标基字母的最长字符串。变量是 T、R、Nd、NiO 这种一到三个字符的东西；带引文
# 的单词比这长，把它算进来会让这个词不再被翻译。
_MAX_BASE = 3

# How far off its line's baseline a character must sit to count as a mark, as a
# fraction of its own size. The measured offsets are 0.45 and 0.28 of the em;
# body text on the same line shares one baseline exactly, so anything this far
# off is deliberate.
# 一个字符要偏离本行基线多远才算角标，按它自身字号的比例计。实测的偏移是 0.45 和 0.28 个
# em；同一行的正文共用同一条基线，因此偏到这个程度必是有意为之。
_MARK_OFFSET = 0.2

# How much of a line must share one baseline before the offsets of the rest mean
# anything. A display formula has characters at many heights and no baseline to
# speak of; measuring deviation there would call most of the line a mark.
# 一行里要有多大比例共用同一条基线，其余字符的偏移才有意义。行间公式的字符高低错落，
# 根本谈不上基线，在那里量偏移会把整行大半都判成角标。
_BASELINE_SHARE = 0.6

# Fonts whose italic form is used for emphasis rather than for mathematics keep
# their multi-character runs; only a run this short is taken as a variable.
# 有些字体的斜体是用来强调而不是排数学的，它们的多字符串保持原样；只有短到这个程度的
# 字串才当作变量。
_MAX_VARIABLE = 1

# Italic font ids for the page being classified, kept per thread.
# 正在分类的这一页的斜体字体编号，按线程保存。
_state = threading.local()


def _text_of(item) -> str:
    """中文：取出被标记项的字符。 / The unicode of a tagged item."""
    return getattr(item[0], "char_unicode", None) or ""


def _font_of(item):
    """中文：取出被标记项的字体编号。 / The font id of a tagged item."""
    return getattr(getattr(item[0], "pdf_style", None), "font_id", None)


def _size_of(item):
    """中文：取出被标记项的字号。 / The font size of a tagged item."""
    return getattr(getattr(item[0], "pdf_style", None), "font_size", None)


def _box_of(item):
    """中文：取出被标记项的框。 / The box of a tagged item."""
    return getattr(item[0], "box", None)


def _is_base_char(text: str) -> bool:
    """
    中文：这个字符能不能算作变量名的一部分。

    Report whether a character can be part of a variable's name.

    Args:
        text (str): The character's unicode, which may be empty or a multi-
            character stand-in such as ``(cid:1)``.

    Returns:
        bool: ``True`` for a single alphanumeric character that is not CJK.
            Chinese is excluded because a mark following it is a citation on
            translated text, never a subscript on a variable.
    """
    if not text or len(text) != 1 or not text.isalnum():
        return False
    return not ("⺀" <= text <= "鿿" or "豈" <= text <= "﫿")


def offset_marks(tagged: list) -> list:
    """
    中文：把偏离本行基线的字符标成角标——不管文件是用小字号还是抬高基线做出来的。

    Tag characters that sit off their line's baseline as corner marks.

    Args:
        tagged (list): ``(character, is_formula, is_corner_mark)`` triples for
            one line.

    Returns:
        list: The same triples with off-baseline characters tagged. Unchanged
            when the line has no baseline most of it agrees on, which is what a
            display formula looks like.
    """
    heights = [(_box_of(item), _size_of(item), _text_of(item)) for item in tagged]
    grounded = [(round(box.y, 1), size) for box, size, text in heights
                if box is not None and size and text.strip()]
    if len(grounded) < 6:
        return tagged
    counts = collections.Counter(y for y, _ in grounded)
    baseline, share = counts.most_common(1)[0]
    if share / len(grounded) < _BASELINE_SHARE:
        return tagged

    flags = [bool(item[1]) for item in tagged]
    marks = [bool(item[2]) for item in tagged]
    changed = False
    for index, (box, size, text) in enumerate(heights):
        if box is None or not size or not text.strip():
            continue
        if abs(box.y - baseline) < size * _MARK_OFFSET:
            continue
        if not marks[index] or not flags[index]:
            flags[index] = marks[index] = True
            changed = True

    if not changed:
        return tagged
    return [(item[0], flags[i], marks[i]) for i, item in enumerate(tagged)]


def italic_variables(tagged: list, italic_ids: set) -> list:
    """
    中文：把只有一个字母的斜体串标成公式——那是数学变量。

    Protect single-letter italic runs, which are mathematical variables.

    Args:
        tagged (list): ``(character, is_formula, is_corner_mark)`` triples for
            one line.
        italic_ids (set): Font ids the page reports as italic.

    Returns:
        list: The same triples with single-letter italic runs tagged as
            formulas. A longer italic run is left alone: in the paper this was
            measured on, every one-character italic run is a variable and every
            longer one is a journal name.
    """
    if not italic_ids:
        return tagged
    flags = [bool(item[1]) for item in tagged]
    changed = False

    index = 0
    while index < len(tagged):
        if _font_of(tagged[index]) not in italic_ids:
            index += 1
            continue
        start = index
        while index < len(tagged) and _font_of(tagged[index]) in italic_ids:
            index += 1
        run = tagged[start:index]
        letters = [_text_of(item) for item in run if _text_of(item).strip()]
        if len(letters) <= _MAX_VARIABLE and all(_is_base_char(c) for c in letters):
            for position in range(start, index):
                if not flags[position]:
                    flags[position] = True
                    changed = True

    if not changed:
        return tagged
    return [(item[0], flags[i], item[2]) for i, item in enumerate(tagged)]


def formula_islands(tagged: list, italic_ids: set) -> list:
    """
    中文：被公式字符夹在中间的单个斜体字母，属于那个公式。

    Absorb a lone italic character that has formula on both sides.

    The exponent's coefficient in the Halperin-Nelson equation is an italic
    ``b`` sitting between two stretches of the same display formula::

        公式  '…exp (cid:1)2'
        文字  'b '                ← this
        公式  'T0 c(cid:1)TBKT…'

    Nothing recognises it: it is not in a formula font, it is not a symbol, and
    the run of italic it belongs to continues into the variables beside it, so
    the single-letter test does not see it either. Left as text it is sent to
    the model, which puts it after the closing bracket.

    Args:
        tagged (list): ``(character, is_formula, is_corner_mark)`` triples for
            one line.
        italic_ids (set): Font ids the page reports as italic.

    Returns:
        list: The same triples with lone italic characters between formulas
            tagged. One character only, and italic only: an upright island is
            ordinary prose — ``and`` between two formulas is a word, not a
            coefficient.
    """
    if not italic_ids:
        return tagged
    flags = [bool(item[1]) for item in tagged]
    changed = False

    index = 0
    while index < len(tagged):
        if flags[index]:
            index += 1
            continue
        start = index
        while index < len(tagged) and not flags[index]:
            index += 1
        run = list(range(start, index))
        letters = [i for i in run if _text_of(tagged[i]).strip()]
        if start == 0 or index >= len(tagged):
            continue  # 一端没有公式。 / No formula on one side.
        if len(letters) != 1:
            continue
        position = letters[0]
        if _font_of(tagged[position]) not in italic_ids:
            continue
        if not _is_base_char(_text_of(tagged[position])):
            continue
        for i in run:
            flags[i] = True
        changed = True

    if not changed:
        return tagged
    return [(item[0], flags[i], item[2]) for i, item in enumerate(tagged)]


def attach_bases(tagged: list) -> list:
    """
    中文：把每串角标前面的基字母也标成公式。

    Tag the base of every corner-mark run as part of the formula.

    Args:
        tagged (list): ``(character, is_formula, is_corner_mark)`` triples for
            one line, as BabelDOC's classifier returns them.

    Returns:
        list: The same triples with some ``is_formula`` raised to ``True``.
            The list is rebuilt only when something changed, so a line with no
            marks is returned untouched.
    """
    if not tagged:
        return tagged
    flags = [bool(item[1]) for item in tagged]
    marks = [bool(item[2]) for item in tagged]
    text = [_text_of(item) for item in tagged]
    changed = False

    index = 0
    while index < len(tagged):
        if not marks[index]:
            index += 1
            continue
        start = index
        while index < len(tagged) and marks[index]:
            index += 1

        # Walk back from the mark over the run it attaches to, stopping at the
        # first character that cannot belong to a variable's name.
        # 从角标往回走过它所依附的那串字符，遇到第一个不能构成变量名的字符就停。
        run = []
        position = start - 1
        while position >= 0 and _is_base_char(text[position]):
            if flags[position]:
                # Already part of a formula — an earlier mark on the same base,
                # or a variable protected above. Grouping joins the two.
                # 已经属于某个公式了——同一基字母上更早的角标，或上面已保护的变量。
                # 分组会把两者并到一起。
                run = []
                break
            run.append(position)
            position -= 1

        if run and len(run) <= _MAX_BASE:
            for position in run:
                flags[position] = True
            changed = True

    if not changed:
        return tagged
    return [(item[0], flags[i], marks[i]) for i, item in enumerate(tagged)]


def _italic_ids(page) -> set:
    """
    中文：收集本页所有斜体字体的编号。

    Collect the ids of every italic font on a page.

    Args:
        page (Page): The page about to be classified.

    Returns:
        set: Font ids the page reports as italic, including those declared
            inside its XObjects.
    """
    found = set()
    for font in getattr(page, "pdf_font", None) or []:
        if getattr(font, "italic", None):
            found.add(font.font_id)
    for xobject in getattr(page, "pdf_xobject", None) or []:
        for font in getattr(xobject, "pdf_font", None) or []:
            if getattr(font, "italic", None):
                found.add(font.font_id)
    return found


def install() -> None:
    """
    中文：分类角标时补上三条规则，让行内公式的碎片留在一起。

    Add three rules to the character classifier so an inline formula holds.

    Two methods are wrapped. ``process_page`` publishes the page's italic font
    ids, because the classifier is given a line and has no way to reach the
    page's font table. ``_classify_characters_in_composition`` then re-tags its
    own result: off-baseline marks first, so the bases can find them; italic
    variables next; bases last.

    Idempotent, so a second call is harmless.

    Raises:
        AttributeError: If BabelDOC no longer exposes either method, so an
            upgrade that renames them fails at startup rather than silently
            going back to scattering variables through the prose.
    """
    from babeldoc.format.pdf.document_il.midend.styles_and_formulas import (
        StylesAndFormulas,
    )

    if getattr(StylesAndFormulas._classify_characters_in_composition,
               "_inlinemath_patched", False):
        return

    original_page = StylesAndFormulas.process_page
    original_classify = StylesAndFormulas._classify_characters_in_composition

    def process_page(self, page, *args, **kwargs):
        """中文：先记下本页的斜体字体，再照常处理。 / Publish italic fonts, then proceed."""
        previous = getattr(_state, "italic", None)
        try:
            _state.italic = _italic_ids(page)
        except Exception:  # noqa: BLE001
            _state.italic = set()
        try:
            return original_page(self, page, *args, **kwargs)
        finally:
            _state.italic = previous

    def _classify_characters_in_composition(self, *args, **kwargs):
        """中文：先照原逻辑分类，再补上三条规则。 / Classify, then apply the rules."""
        tagged, first_is_bullet = original_classify(self, *args, **kwargs)
        try:
            tagged = offset_marks(tagged)
            italic = getattr(_state, "italic", None) or set()
            tagged = italic_variables(tagged, italic)
            tagged = formula_islands(tagged, italic)
            tagged = attach_bases(tagged)
        except Exception:  # noqa: BLE001
            # A line that cannot be examined keeps its tags, which is how every
            # line behaved before this existed.
            # 看不了的行保留原有标记，也就是本模块存在之前所有行的行为。
            logger.warning("Could not hold an inline formula together", exc_info=True)
        return tagged, first_is_bullet

    _classify_characters_in_composition._inlinemath_patched = True
    StylesAndFormulas.process_page = process_page
    StylesAndFormulas._classify_characters_in_composition = (
        _classify_characters_in_composition
    )
