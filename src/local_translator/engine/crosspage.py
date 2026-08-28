"""
中文：分页把句子切断时，把较短的半句挪到另一页，再分别翻译。

Move the shorter half of a sentence across a page break, then translate each
page's paragraph on its own.

**Why this exists.** BabelDOC has two translation pipelines. The LLM-only one,
used when the engine implements ``llm_translate``, pairs the last body paragraph
of each page with the first of the next. The classic one — the path every local
and hybrid backend takes, because a translation model cannot follow the LLM
pipeline's JSON protocol — does not: its ``paragraph_finder`` runs strictly page
by page, and ``il_translator`` carries a ``new_cross_page()`` tracker that
nothing ever calls.

The result is visible in any paper. Measured on one 19-page document, the source
splits ``…for high-`` / ``quality domains.`` across pages 2 and 3. Translated a
page at a time, ``for high-`` became 「高」 and ``quality domains.`` became a
free-standing 「高质量领域。」 at the top of the next page.

**How this one works.** Not by translating the two halves together and cutting
the answer in two — that asks a model to place a boundary it was never told
about, and a bad cut is worse than the split it repairs. Instead the *source* is
rebalanced before either half is translated:

1. Take the last body paragraph of a page and the first of the next.
2. If the first does not end a sentence and the second reads as a continuation,
   there is a break to repair.
3. ``tail`` is what follows the last sentence terminator on the earlier page;
   ``head`` is the first sentence on the later one.
4. If ``tail`` is shorter than half of ``head``, move ``tail`` forward onto the
   later page. Otherwise move ``head``'s first clause back onto the earlier one.
5. Translate each paragraph separately, as usual.

Moving the shorter piece keeps the amount of text that ends up on a page it did
not come from as small as the break allows, and either way both halves reach the
model as something that reads like language rather than like a fragment.

**What it refuses to touch.** The piece that moves must carry no formula
placeholder. Placeholders are numbered from ``{v1}`` within each paragraph and
resolved against that paragraph's own table, so a placeholder carried across
would name the wrong formula on arrival. The rest of either paragraph may hold
as much mathematics as it likes.

Any failure — a pair that cannot be split cleanly, an exception anywhere in the
pass — leaves both paragraphs exactly as they were, which is the behaviour this
module replaces.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# Bound by :func:`_install_helpers` so BabelDOC is imported lazily.
# 由 _install_helpers 绑定，使 BabelDOC 延迟导入。
_ILTranslator = None
_DocumentTranslateTracker = None

# Sentence-final punctuation. A paragraph ending in one of these is finished, and
# whatever starts the next page begins something new.
# 句末标点。以此结尾的段落是完整的，下一页开头属于新的内容。
_ENDS_SENTENCE = re.compile(r"[.!?。！？」』）\)]\s*$")

# A continuation usually starts lower-case, because it is the middle of a
# sentence. A page whose first body paragraph starts with a capital is nearly
# always a new one.
# 续行通常以小写开头，因为它是句子的中段。首字母大写的段落几乎都是新段落。
_STARTS_LOWER = re.compile(r"^[a-z(\[]")

# Formula placeholders, as they appear once a translate input has been built.
# 公式占位符在翻译输入里的样子。
_PLACEHOLDER = re.compile(r"\{\s*v\s*\d+\s*\}")

# End of a sentence, mid-string: the mark plus the space that follows it. The
# lookahead keeps an abbreviation such as "Fig. 3" or "et al." from counting,
# since what follows those is not a capital starting a new sentence.
# 句中的句末位置：标点加其后的空格。前瞻可避免把 "Fig. 3"、"et al." 这类缩写当成
# 句子结束——它们后面跟的并不是新句子的大写首字母。
_SENTENCE_END = re.compile(r"[.!?。！？](?:[\"'」』）\)]\s*)?\s+(?=[A-Z(\[])|[。！？]")

# Clause boundaries inside a sentence, used when the later page has to give some
# of its first sentence back to the earlier one.
# 句内的分句边界。当后页需要把首句的一部分还给前页时使用。
_CLAUSE_END = re.compile(r"[,;:，；：]\s+")

# Shortest fragment worth moving. Below this the two paragraphs are already
# divided at something close enough to a boundary.
# 值得搬动的最短片段。短于此值说明两段的分界已经足够接近边界。
_MIN_MOVE = 3

# Longest fragment that may move, in characters of source. This bounds how far
# the page break can shift regardless of how long either paragraph is: a page's
# text has to fit the box it is laid out in, and a clause is a repair while a
# couple of sentences is a relocation.
# 可搬动片段的最大长度（源文字符）。无论两段各有多长，它都限定分页位置能挪多远：
# 一页的文字要排进它自己的框里，搬一个分句是修补，搬两三句就是搬家了。
_MAX_MOVE = 160

# Longest tail that may move *forward*, as a share of its paragraph. A tail
# longer than this means the paragraph is mostly one unfinished sentence, and
# sending it on would empty the page rather than tidy its edge.
# 可向后搬动的段末残句占其段落的最大比例。超过此值说明该段基本上就是一个未完成的
# 句子，把它整个送走是搬空这一页，而不是修整页边。
_MAX_SHARE = 0.5


def _tail(text: str) -> str:
    """
    中文：取出段末那半句——最后一个句末标点之后的部分。

    Return the unfinished sentence at the end of a paragraph.

    Args:
        text (str): The paragraph's translate input.

    Returns:
        str: Everything after the last sentence end, or the whole string when
            the paragraph contains no sentence end at all — a paragraph that is
            one long unfinished sentence is entirely a tail.
    """
    last = None
    for match in _SENTENCE_END.finditer(text):
        last = match
    return text if last is None else text[last.end():]


def _first_sentence(text: str) -> str:
    """
    中文：取出段首第一句。

    Return the first sentence of a paragraph.

    Args:
        text (str): The paragraph's translate input.

    Returns:
        str: Up to and including the first sentence end, or the whole string
            when there is none.
    """
    match = _SENTENCE_END.search(text)
    return text if match is None else text[:match.end()]


def _first_clause(text: str) -> str:
    """
    中文：取出段首第一个「半句」——到第一个分句边界为止，没有就取整个第一句。

    Return the first clause of a paragraph.

    Args:
        text (str): The paragraph's translate input.

    Returns:
        str: Up to and including the first clause boundary within the first
            sentence, falling back to the whole first sentence when the sentence
            has no internal boundary.
    """
    sentence = _first_sentence(text)
    match = _CLAUSE_END.search(sentence)
    return sentence if match is None else sentence[:match.end()]


def _movable(fragment: str, source: str, share: float) -> bool:
    """
    中文：判断一个片段能不能搬——不含公式占位符，长度既够也不过分。

    Decide whether a fragment may be moved to the other page.

    Args:
        fragment (str): The piece proposed for moving.
        source (str): The paragraph it would be taken from.
        share (float): Largest share of ``source`` the fragment may be. Used
            when sending a tail forward, where taking most of a paragraph means
            emptying it; a clause brought back is bounded by its own length
            instead, so callers pass 1.0.

    Returns:
        bool: ``True`` when the fragment carries no formula placeholder and its
            length is within both bounds. A placeholder is numbered within its
            own paragraph and resolved against that paragraph's table, so
            carrying one across would name the wrong formula on arrival.
    """
    fragment = fragment.strip()
    if not _MIN_MOVE <= len(fragment) <= _MAX_MOVE:
        return False
    if _PLACEHOLDER.search(fragment):
        return False
    return len(fragment) <= len(source) * share


def _continues(first: str, second: str) -> bool:
    """
    中文：判断后一段是不是前一段的续行。

    Decide whether the second paragraph continues the first.

    Args:
        first (str): Text of the earlier page's last body paragraph.
        second (str): Text of the later page's first body paragraph.

    Returns:
        bool: ``True`` when the first does not finish its sentence and the
            second reads like the middle of one — it starts lower-case, or the
            first ends on a hyphen, which is a word broken across the break.
    """
    first, second = (first or "").strip(), (second or "").strip()
    if not first or not second:
        return False
    if _ENDS_SENTENCE.search(first):
        return False
    return bool(first.endswith("-") or _STARTS_LOWER.match(second))


def _join(left: str, right: str) -> str:
    """
    中文：拼接两个片段；前半以连字符结尾时不加空格，因为那是被分页切断的一个词。

    Join two fragments.

    Args:
        left (str): The earlier fragment.
        right (str): The later one.

    Returns:
        str: The two with a space between them, or directly concatenated when
            the first ends on a hyphen — that hyphen is a word the page break
            cut in half, and a space would make it two words.
    """
    left, right = left.rstrip(), right.lstrip()
    if left.endswith("-"):
        return left + right
    return f"{left} {right}"


def _move_forward(first: str, second: str) -> tuple[str, str] | None:
    """
    中文：把前页末尾那半句挪到后页开头。

    Send the earlier page's trailing fragment forward.

    Args:
        first (str): Translate input of the earlier paragraph.
        second (str): Translate input of the later one.

    Returns:
        tuple[str, str] | None: The rebalanced pair, or ``None`` when the tail
            cannot move — it carries a formula, it is the whole paragraph, or
            it is too long a piece to relocate.
    """
    tail = _tail(first).strip()
    if not tail or not _movable(tail, first, _MAX_SHARE):
        return None
    remainder = first[:len(first) - len(tail)].rstrip()
    if not remainder.strip():
        return None
    return remainder, _join(tail, second)


def _pull_back(first: str, second: str) -> tuple[str, str] | None:
    """
    中文：把后页首句的第一个「半句」拿回前页末尾。

    Bring the later page's opening clause back.

    Args:
        first (str): Translate input of the earlier paragraph.
        second (str): Translate input of the later one.

    Returns:
        tuple[str, str] | None: The rebalanced pair, or ``None`` when the clause
            cannot move.
    """
    clause = _first_clause(second)
    if not clause.strip() or not _movable(clause, second, 1.0):
        return None
    remainder = second[len(clause):].lstrip()
    if not remainder.strip():
        return None
    return _join(first, clause.strip()), remainder


def rebalance(first: str, second: str) -> tuple[str, str] | None:
    """
    中文：按「把较短的半句挪过去」重新划分两段的分界；首选不成就试另一个方向。

    Move the shorter half of the broken sentence to the other side.

    Which way it goes is decided by length: when the earlier page holds less
    than half of what the later page's first sentence holds, it is holding a
    scrap and the scrap moves forward. Otherwise the sentence mostly belongs to
    the earlier page and enough of the later one is brought back to finish it.

    Whichever direction is chosen, the other is tried if it does not work out,
    because a fragment can be unmovable for reasons that have nothing to do with
    which side is shorter — it may carry a formula, or be the whole paragraph.

    Args:
        first (str): Translate input of the earlier page's last body paragraph.
        second (str): Translate input of the later page's first body paragraph.

    Returns:
        tuple[str, str] | None: The rebalanced pair, or ``None`` when neither
            direction is safe, in which case both paragraphs are left alone.
    """
    tail = _tail(first).strip()
    head = _first_sentence(second).strip()
    if not tail or not head:
        return None

    forward_first = len(tail) * 2 < len(head)
    order = (_move_forward, _pull_back) if forward_first else (_pull_back, _move_forward)
    for move in order:
        result = move(first, second)
        if result is not None:
            return result
    return None


def install(enabled: bool = True) -> None:
    """
    中文：给经典管线加上跨页合并；失败时静默退回逐页翻译。

    Add cross-page merging to BabelDOC's classic translation pipeline.

    Wraps ``ILTranslator.translate`` so a cross-page pass runs before the
    per-page loop. Paragraphs it has handled are recorded on the instance and
    skipped by the wrapper around ``translate_paragraph``.

    Args:
        enabled (bool): When false, does nothing, so a caller can pass the
            setting straight through.

    Raises:
        AttributeError: If BabelDOC no longer exposes ``ILTranslator.translate``,
            so an upgrade that moves it fails at startup rather than silently
            reverting to page-at-a-time translation.
    """
    if not enabled:
        return

    from babeldoc.format.pdf.document_il.midend.il_translator import (
        DocumentTranslateTracker,
        ILTranslator,
    )

    if getattr(ILTranslator.translate, "_crosspage_patched", False):
        return

    original_translate = ILTranslator.translate
    original_translate_paragraph = ILTranslator.translate_paragraph

    def translate(self, docs):
        """中文：先跑跨页合并，再走原有逐页流程。 / Merge across pages, then translate as usual."""
        self._crosspage_done = set()
        try:
            _merge_pages(self, docs)
        except Exception:  # noqa: BLE001
            # A failure here must not cost the document: without the pass every
            # paragraph is simply translated on its own, as before.
            # 这里失败不应赔上整篇文档：没有这一步，每段照旧各自翻译。
            logger.warning("Cross-page merging failed; translating page by page",
                           exc_info=True)
        return original_translate(self, docs)

    def translate_paragraph(self, paragraph, *args, **kwargs):
        """中文：跨页合并已处理过的段落直接跳过。 / Skip what the cross-page pass already did."""
        done = getattr(self, "_crosspage_done", None)
        if done and id(paragraph) in done:
            pbar = args[1] if len(args) > 1 else kwargs.get("pbar")
            if pbar is not None:
                pbar.advance()
            return None
        return original_translate_paragraph(self, paragraph, *args, **kwargs)

    translate._crosspage_patched = True
    ILTranslator.translate = translate
    ILTranslator.translate_paragraph = translate_paragraph
    _install_helpers(ILTranslator, DocumentTranslateTracker)


def _install_helpers(ILTranslator, DocumentTranslateTracker) -> None:
    """
    中文：把跨页合并的具体步骤挂到 ILTranslator 上（延迟绑定 BabelDOC 类型）。

    Attach the merging helpers, binding BabelDOC's types lazily.

    Args:
        ILTranslator (type): BabelDOC's classic translator class.
        DocumentTranslateTracker (type): Its tracker, needed to build the
            per-paragraph trackers ``post_translate_paragraph`` writes into.
    """
    global _ILTranslator, _DocumentTranslateTracker
    _ILTranslator = ILTranslator
    _DocumentTranslateTracker = DocumentTranslateTracker


def _body_paragraphs(page) -> list:
    """
    中文：取出一页里可以参与跨页合并的正文段落。

    List the body-text paragraphs on a page.

    Only ``plain text`` qualifies. A heading, caption or table cell that happens
    to sit at a page edge is not a sentence continued from somewhere else, and
    joining one to its neighbour would merge two unrelated things.

    Args:
        page (Page): Page to read.

    Returns:
        list: Paragraphs in reading order, headings and captions excluded.
    """
    out = []
    for paragraph in page.pdf_paragraph:
        if getattr(paragraph, "vertical", False):
            continue
        if (paragraph.layout_label or "").strip() != "plain text":
            continue
        if not (paragraph.unicode or "").strip():
            continue
        out.append(paragraph)
    return out


def _font_maps(page):
    """
    中文：构造该页的字体映射，与 BabelDOC 逐页流程里的一致。

    Build the page's font maps exactly as ``process_page`` does.

    Args:
        page (Page): Page to read.

    Returns:
        tuple[dict, dict]: Fonts by id, and fonts by XObject id.
    """
    page_font_map = {font.font_id: font for font in page.pdf_font}
    xobj_font_map = {}
    for xobj in page.pdf_xobject:
        fonts = page_font_map.copy()
        for font in xobj.pdf_font:
            fonts[font.font_id] = font
        xobj_font_map[xobj.xobj_id] = fonts
    return page_font_map, xobj_font_map


def _merge_pages(self, docs) -> None:
    """
    中文：遍历相邻页对，把被切开的段落合并翻译。

    Walk adjacent page pairs and translate the split paragraphs together.

    Args:
        self (ILTranslator): The translator running the document.
        docs (Document): The document being translated.
    """
    tracker = _DocumentTranslateTracker()
    merged = skipped_formula = 0

    for index in range(len(docs.page) - 1):
        page_first, page_second = docs.page[index], docs.page[index + 1]
        first_list = _body_paragraphs(page_first)
        second_list = _body_paragraphs(page_second)
        if not first_list or not second_list:
            continue

        first, second = first_list[-1], second_list[0]
        if id(first) in self._crosspage_done or id(second) in self._crosspage_done:
            continue
        if not _continues(first.unicode, second.unicode):
            continue

        outcome = _merge_pair(self, tracker, page_first, first, page_second, second)
        if outcome == "formula":
            skipped_formula += 1
        elif outcome == "merged":
            merged += 1
            self._crosspage_done.add(id(first))
            self._crosspage_done.add(id(second))

    if merged or skipped_formula:
        logger.info("Cross-page merging: %d paragraph pairs joined, "
                    "%d left alone because they carry formulas", merged, skipped_formula)


def _merge_pair(self, tracker, page_first, first, page_second, second) -> str:
    """
    中文：把一对跨页段落的分界挪到句子边界上，再分别翻译写回。

    Rebalance one cross-page pair and translate each half on its own.

    Args:
        self (ILTranslator): The translator running the document.
        tracker (DocumentTranslateTracker): Document-level tracker.
        page_first (Page): Page the earlier half sits on.
        first (PdfParagraph): Last body paragraph of that page.
        page_second (Page): Page the later half sits on.
        second (PdfParagraph): First body paragraph of that page.

    Returns:
        str: ``"merged"`` when the pair was rebalanced and both halves written
            back, ``"formula"`` when a placeholder in the piece that would have
            moved made it unsafe, or ``"skip"`` for anything else — in every
            case other than ``"merged"`` the paragraphs are untouched and the
            ordinary per-page pass will translate them.
    """
    page_tracker = tracker.new_cross_page()
    tracker_first = page_tracker.new_paragraph()
    tracker_second = page_tracker.new_paragraph()

    fonts_first, xobj_first = _font_maps(page_first)
    fonts_second, xobj_second = _font_maps(page_second)

    text_first, input_first = self.pre_translate_paragraph(
        first, tracker_first, fonts_first, xobj_first)
    text_second, input_second = self.pre_translate_paragraph(
        second, tracker_second, fonts_second, xobj_second)
    if not text_first or not text_second:
        return "skip"

    balanced = rebalance(text_first, text_second)
    if balanced is None:
        # Either nothing needed moving or the piece that would have moved
        # carries a formula; both mean the pair is left as it is.
        # 要么无须搬动，要么该搬的片段带着公式；两种情况都保持原样。
        return "formula" if _PLACEHOLDER.search(text_first + text_second) else "skip"

    logger.debug("Rebalanced a page break: %r | %r  ->  %r | %r",
                 text_first[-40:], text_second[:40],
                 balanced[0][-40:], balanced[1][:40])

    # Both halves are translated before either is written back. Writing one and
    # then failing would leave that paragraph translated but unmarked, and the
    # ordinary per-page pass would translate it a second time on top of itself.
    # 两半都译完之后才写回。写了一半再失败，会让那一段已译却未标记，随后的逐页流程
    # 会在它上面再翻译一次。
    halves = []
    for source in balanced:
        translated = self.translate_engine.translate(
            source,
            rate_limit_params={"paragraph_token_count": self.calc_token_count(source)},
        )
        translated = re.sub(r"[. 。…，]{20,}", ".", translated or "")
        if not translated.strip():
            return "skip"
        halves.append(translated)

    self.post_translate_paragraph(first, tracker_first, input_first, halves[0])
    self.post_translate_paragraph(second, tracker_second, input_second, halves[1])
    return "merged"
