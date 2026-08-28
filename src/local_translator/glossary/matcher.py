"""
中文：段落级术语匹配——在提示词中注入译法约束与英文保留白名单。

Paragraph-level terminology matching.

BabelDOC only injects glossary blocks on its LLM-only pipeline, which the local
Hy-MT2 backend deliberately avoids (see :mod:`local_translator.engine.translators`).
Terminology is therefore matched here and folded into the model's native prompt.

Why not reuse ``babeldoc.glossary.Glossary``:
    That implementation compiles ``re.escape(term)`` with ``HS_FLAG_CASELESS``
    and **no word boundaries**, so a whitelist entry like ``token`` also fires
    inside ``tokenizer``, ``tokenised`` and ``TokenType``. For terms we merely
    suggest a translation for, an over-eager match is harmless noise; for
    whitelist terms it is actively wrong, because it would instruct the model to
    leave an unrelated word untranslated. This module compiles the same
    Hyperscan engine with explicit ``\\b`` anchors instead.
"""

from __future__ import annotations

import logging
import re
import threading

logger = logging.getLogger(__name__)

POLICY_TRANSLATE = "translate"
POLICY_KEEP = "keep"

# Terms shorter than this are skipped: single letters and two-character tokens
# match far too much text to be useful constraints.
# 过短的术语命中面太广，没有约束价值，直接跳过。
MIN_TERM_LENGTH = 3

_WORD_START_RE = re.compile(r"^\w")
_WORD_END_RE = re.compile(r"\w$")

# Endings that take "-es" rather than a bare "-s" / 需要加 -es 而非 -s 的词尾
_ES_ENDINGS = ("s", "x", "z", "ch", "sh")
_VOWELS = "aeiou"

# Irregular plurals that matter in scientific writing. Regular suffixing would
# produce "matrixes" and "analysises", neither of which occurs in a real paper,
# so the whitelist would silently miss the form that is actually used.
# 科技文献常见的不规则复数；按规则加后缀会得到论文里根本不会出现的形式。
_IRREGULAR_PLURALS: dict[str, str] = {
    "matrix": "matrices",
    "index": "indices",
    "vertex": "vertices",
    "vortex": "vortices",
    "simplex": "simplices",
    "analysis": "analyses",
    "basis": "bases",
    "hypothesis": "hypotheses",
    "thesis": "theses",
    "axis": "axes",
    "criterion": "criteria",
    "phenomenon": "phenomena",
    "spectrum": "spectra",
    "datum": "data",
    "medium": "media",
    "maximum": "maxima",
    "minimum": "minima",
    "formula": "formulae",
    "nucleus": "nuclei",
    "radius": "radii",
    "locus": "loci",
    "stimulus": "stimuli",
    "lattice": "lattices",
}


def plural_variant(term: str) -> str | None:
    """
    中文：生成术语的英文复数形式，供匹配时一并覆盖。

    Derive the English plural of a term, for matching purposes.

    Word-boundary anchoring is what stops the whitelist entry ``token`` from
    firing inside ``tokenizer``, but it also stops it firing on ``tokens`` — so
    a sentence like "truncated to 512 tokens" would slip past the whitelist and
    come back as 「512个标记」. Registering the plural as an additional surface
    form closes that gap without weakening the anchoring.

    Only the final word is inflected, so ``ablation study`` becomes
    ``ablation studies`` rather than ``ablations study``.

    Args:
        term (str): Source term, one or more words.

    Returns:
        str | None: The plural form, or ``None`` when it would be identical to
            the input or the term does not look like an inflectable word (for
            example an acronym such as ``LLM``, or anything ending in a
            non-letter such as ``C++``).
    """
    head, _, last = term.rpartition(" ")
    if not last or not last[-1].isalpha():
        return None

    # Acronyms pluralise unpredictably in technical prose (LLMs, but also LLM's
    # and plain LLM), so leave them to the base form rather than guessing.
    # 缩写的复数写法不统一，不做猜测。
    if last.isupper() and len(last) <= 5:
        return None

    lowered = last.lower()
    if lowered in _IRREGULAR_PLURALS:
        inflected = _IRREGULAR_PLURALS[lowered]
    elif lowered.endswith("y") and len(lowered) > 1 and lowered[-2] not in _VOWELS:
        inflected = last[:-1] + "ies"
    elif lowered.endswith(_ES_ENDINGS):
        inflected = last + "es"
    else:
        inflected = last + "s"

    plural = f"{head} {inflected}" if head else inflected
    return plural if plural.lower() != term.lower() else None


def _char_before(data: bytes, index: int) -> str:
    """
    中文：取字节位置 index 之前的那个完整字符。

    Decode the character ending just before a byte offset.

    Args:
        data (bytes): UTF-8 encoded paragraph being scanned.
        index (int): Byte offset the match starts at.

    Returns:
        str: The preceding character, or ``""`` at the start of the text.
    """
    if index <= 0:
        return ""
    start = index - 1
    while start > 0 and 0x80 <= data[start] < 0xC0:
        start -= 1
    return data[start:index].decode("utf-8", "ignore")


def _char_at(data: bytes, index: int) -> str:
    """
    中文：取字节位置 index 开始的那个完整字符。

    Decode the character beginning at a byte offset.

    Args:
        data (bytes): UTF-8 encoded paragraph being scanned.
        index (int): Byte offset just past the match.

    Returns:
        str: The following character, or ``""`` at the end of the text.
    """
    if index >= len(data):
        return ""
    end = index + 1
    while end < len(data) and 0x80 <= data[end] < 0xC0:
        end += 1
    return data[index:end].decode("utf-8", "ignore")


# Scripts that do not put spaces between words. A Latin term butted straight up
# against one of these is at a word boundary in every sense that matters —
# 「使用Transformer模型」 names Transformer as plainly as "use the Transformer
# model" does, and there is nowhere for a space to go.
# 不用空格分词的文种。拉丁术语紧贴这类字符时，从任何有意义的角度看都处在词边界上
# ——「使用Transformer模型」提到 Transformer 的程度与英文写法毫无二致，而这里根本
# 没有插入空格的位置。
_SPACELESS_RANGES: tuple[tuple[int, int], ...] = (
    (0x1100, 0x11FF),    # Hangul Jamo
    (0x3040, 0x30FF),    # Hiragana, Katakana
    (0x3100, 0x312F),    # Bopomofo
    (0x31F0, 0x31FF),    # Katakana phonetic extensions
    (0x3400, 0x4DBF),    # CJK unified ideographs extension A
    (0x4E00, 0x9FFF),    # CJK unified ideographs
    (0xAC00, 0xD7AF),    # Hangul syllables
    (0xF900, 0xFAFF),    # CJK compatibility ideographs
    (0xFF00, 0xFFEF),    # Halfwidth and fullwidth forms
    (0x20000, 0x3FFFF),  # CJK extensions B and beyond
)


def _is_spaceless_script(char: str) -> bool:
    """
    中文：判断字符是否属于不用空格分词的文种（中日韩等）。

    Test whether a character belongs to a script written without spaces.

    Args:
        char (str): A single character.

    Returns:
        bool: ``True`` for Han, Kana, Hangul and the full-width forms.
    """
    code = ord(char)
    return any(low <= code <= high for low, high in _SPACELESS_RANGES)


def _is_word_char(char: str) -> bool:
    r"""
    中文：判断一个字符能否构成词边界的「单词一侧」；中日韩字符不算。

    Test whether a character blocks a word boundary.

    The test is done per character rather than per byte on purpose. Hyperscan
    scanned the paragraph as bytes, so its ``\b`` only ever saw ASCII word
    characters, and that was wrong in both directions:

    * A term beginning with a Greek letter, such as ``εtotal``, could never
      match at all — the pattern asked for a boundary before the byte 0xCE, and
      a byte that is not a word byte never has one on its left. Measured over
      2,104 real paragraphs, that silently lost 19 matches.
    * A term against a Han character *did* match, because no byte of 「用」 is
      an ASCII word byte. That one was right by accident, and taking Python's
      ``\w`` at face value would have broken it: ``'用'.isalnum()`` is true, so
      「使用Transformer模型」 would stop matching ``Transformer`` — exactly the
      shape of text the zh→en direction and the free-text page are full of.

    So a character counts as word-forming when it is alphanumeric *and* written
    in a script that separates words with spaces.

    Args:
        char (str): A single character, or ``""`` for the edge of the text.

    Returns:
        bool: ``True`` when the character would make this side of the match the
            middle of a word.
    """
    if not char:
        return False
    if char == "_":
        return True
    return char.isalnum() and not _is_spaceless_script(char)


def _anchored_pattern(term: str) -> bytes:
    """
    中文：为术语生成带词边界的匹配模式。

    Build a word-boundary-anchored Hyperscan pattern for a term.

    ``\\b`` is only added on a side where the term actually begins or ends with
    a word character. Anchoring a term such as ``C++`` on its right would never
    match, because ``+`` is not a word character and there is no word boundary
    after it.

    Args:
        term (str): Source term as stored in the glossary.

    Returns:
        bytes: UTF-8 encoded regular expression suitable for Hyperscan.
    """
    escaped = re.escape(term)
    left = r"\b" if _WORD_START_RE.search(term) else ""
    right = r"\b" if _WORD_END_RE.search(term) else ""
    return f"{left}{escaped}{right}".encode()


def _is_subphrase(candidate: str, container: str) -> bool:
    """
    中文：判断一个术语是否作为完整词组被另一个术语包含。

    Test whether one term appears inside another at word boundaries.

    Boundary-aware so that ``net`` is not treated as covered by ``network``,
    while ``batch`` genuinely is covered by ``batch normalization``.

    Args:
        candidate (str): The possibly-redundant term, lower-cased.
        container (str): The longer term, lower-cased.

    Returns:
        bool: ``True`` if ``candidate`` is a whole-word sub-phrase of
            ``container`` and not the term itself.
    """
    if candidate == container or len(candidate) >= len(container):
        return False
    words = container.split()
    needle = candidate.split()
    return any(
        words[i : i + len(needle)] == needle for i in range(len(words) - len(needle) + 1)
    )


class GlossaryMatcher:
    """
    中文：用 Hyperscan 在段落文本中匹配术语，返回译法约束与白名单。

    Match glossary terms against paragraph text.

    Instances are built once per job and then called once per paragraph from
    BabelDOC's worker threads, so matching must be both fast and thread-safe.
    Hyperscan scratch space is not thread-safe, so each thread gets its own via
    thread-local storage rather than allocating one per call.

    Attributes:
        max_terms (int): Upper bound on terms returned for a single paragraph.
    """

    def __init__(self, terms: list[tuple[str, str, str]], max_terms: int = 24):
        """
        中文：编译术语匹配库。

        Compile the term database.

        Args:
            terms (list[tuple[str, str, str]]): Entries as
                ``(source, target, policy)``. ``policy`` is ``"translate"`` for a
                suggested rendering or ``"keep"`` for a term that must stay in
                the source language. Entries shorter than
                :data:`MIN_TERM_LENGTH`, or with an empty source, are dropped.
            max_terms (int): Maximum number of terms injected into one prompt.
                A translation-specialised 7B model stops honouring a term list
                that grows too long, so this stays small by default.
        """
        self.max_terms = max_terms
        self._entries: list[tuple[str, str, str]] = []
        self._db = None
        self._local = threading.local()

        # Index-aligned with ``_entries``: the literal being searched for, and
        # whether a word boundary is required on each side. The boundary test
        # used to be inside the pattern; see :meth:`_compile` for why it moved.
        # 与 _entries 一一对应：待查的字面量，以及左右两侧是否需要词边界。
        # 词边界原本写在模式里，移出来的原因见 _compile。
        self._literals: list[bytes] = []
        self._anchors: list[tuple[bool, bool]] = []

        seen: set[str] = set()
        patterns: list[bytes] = []

        def register(surface: str, target: str, policy: str) -> None:
            """
            中文：登记一个待匹配的表面形式。

            Register one matchable surface form.

            Args:
                surface (str): Text to match in the paragraph.
                target (str): Rendering to instruct the model to use.
                policy (str): ``translate`` or ``keep``.
            """
            key = surface.lower()
            if key in seen or len(surface) < MIN_TERM_LENGTH:
                return
            seen.add(key)
            patterns.append(_anchored_pattern(surface))
            self._literals.append(surface.encode("utf-8"))
            self._anchors.append((bool(_WORD_START_RE.search(surface)),
                                  bool(_WORD_END_RE.search(surface))))
            self._entries.append((surface, target, policy))

        for src, tgt, policy in terms:
            src = (src or "").strip()
            tgt = (tgt or "").strip()
            if len(src) < MIN_TERM_LENGTH or not tgt:
                continue
            policy = policy or POLICY_TRANSLATE
            register(src, tgt, policy)

            # Register the plural as its own surface form. For a whitelist entry
            # the plural must map to *itself* (tokens=tokens, not tokens=token),
            # otherwise the instruction would ask the model to change the number.
            # 复数单独登记；白名单必须映射到复数本身，否则会让模型改变数。
            plural = plural_variant(src)
            if plural:
                register(plural, plural if policy == POLICY_KEEP else tgt, policy)

        if not self._entries:
            logger.info("Glossary matcher built with no usable terms")
            return

        self._compile(patterns)

    def _compile(self, patterns: list[bytes]) -> None:
        """
        中文：编译 Hyperscan 数据库；失败则降级为不匹配。

        Compile the Hyperscan database over all anchored patterns.

        A compilation failure degrades to "no glossary" rather than aborting the
        job: losing terminology hints is a quality regression, not a reason to
        fail an otherwise valid translation.

        **Why the patterns are literals rather than regexes.** Each term used to
        be compiled as ``\bterm\b``. A word-boundary assertion is not a literal,
        so Hyperscan had to build its general pattern machinery for all of them:
        measured on this project's own glossary, 26,231 anchored patterns cost
        **1.78 GB and 20.1 seconds to compile**, on every translation, in a
        process that otherwise needs a fraction of that. The same terms compiled
        through Hyperscan's pure-literal API cost **0.07 GB and 0.2 seconds**.

        The boundary test moves to :meth:`__call__`, which decodes the character
        on either side of each hit — see :func:`_is_word_char` for why it is
        done per character rather than per byte, and for the nineteen matches
        the old form was silently losing.

        ``HS_FLAG_SINGLEMATCH`` had to go with it, and that matters. It reports
        only the first hit per pattern, which was harmless when the pattern
        itself enforced the boundary: the first hit was by construction a real
        one. Against a bare literal the first hit may well be inside a longer
        word — ``net`` inside ``internet`` — and rejecting it would have hidden
        the genuine ``a net`` later in the same paragraph. Every occurrence is
        reported instead and the ids are deduplicated in Python, which is what
        the caller did anyway.

        Args:
            patterns (list[bytes]): Anchored regex patterns, kept for reference;
                the literals in ``self._literals`` are what is compiled.
        """
        try:
            import hyperscan

            db = hyperscan.Database()
            db.compile(
                expressions=self._literals,
                ids=list(range(len(self._literals))),
                elements=len(self._literals),
                # CASELESS so "Transformer" matches "transformer". SINGLEMATCH
                # is deliberately absent — see the note above.
                # CASELESS 用于大小写不敏感。刻意不加 SINGLEMATCH，原因见上文。
                flags=hyperscan.HS_FLAG_CASELESS,
                literal=True,
            )
            self._db = db
            logger.info("Glossary matcher compiled with %d terms", len(self._entries))
        except Exception:
            logger.warning(
                "Failed to compile glossary matcher; continuing without terminology",
                exc_info=True,
            )
            self._db = None

    def _scratch(self):
        """
        中文：返回当前线程专属的 Hyperscan scratch 空间。

        Return this thread's Hyperscan scratch buffer.

        Scratch space cannot be shared between concurrently scanning threads,
        and BabelDOC translates paragraphs from a thread pool.

        Returns:
            hyperscan.Scratch: Per-thread scratch allocated on first use.
        """
        scratch = getattr(self._local, "scratch", None)
        if scratch is None:
            import hyperscan

            scratch = hyperscan.Scratch(self._db)
            self._local.scratch = scratch
        return scratch

    def __call__(self, text: str) -> tuple[list[tuple[str, str]], list[str]]:
        """
        中文：匹配段落文本，返回 (译法约束对, 保持原文的术语列表)。

        Find glossary terms occurring in a paragraph.

        Results are ordered by descending term length so that when both
        ``attention`` and ``attention mechanism`` match, the more specific term
        survives the ``max_terms`` cap.

        Args:
            text (str): Paragraph text to scan.

        Returns:
            tuple[list[tuple[str, str]], list[str]]: A pair of
                ``(translate_pairs, keep_terms)``. ``translate_pairs`` holds
                ``(source_term, target_term)`` suggestions; ``keep_terms`` holds
                source terms that must remain untranslated. Both are capped
                jointly at ``max_terms`` and may be empty.
        """
        if self._db is None or not text:
            return [], []

        hits: set[int] = set()
        data = text.encode("utf-8")

        def on_match(idx: int, _from: int, to: int, _flags: int, _ctx=None):
            """中文：命中后补做词边界检查。 / Apply the word-boundary test to a hit."""
            if idx in hits:
                return False
            left, right = self._anchors[idx]
            if left:
                start = to - len(self._literals[idx])
                if _is_word_char(_char_before(data, start)):
                    return False
            if right and _is_word_char(_char_at(data, to)):
                return False
            hits.add(idx)
            return False

        try:
            self._db.scan(data, on_match, scratch=self._scratch())
        except Exception:
            logger.debug("Glossary scan failed for a paragraph", exc_info=True)
            return [], []

        if not hits:
            return [], []

        ordered = sorted(
            (self._entries[i] for i in hits),
            key=lambda e: len(e[0]),
            reverse=True,
        )

        # Drop terms already covered by a longer match. A paragraph mentioning
        # "batch normalization" matches that term and also "batch",
        # "normalization" and "layer"; instructing the model on all four wastes
        # the prompt budget a 7B model can actually track, and the fragments add
        # nothing the compound term has not already pinned down.
        # 丢弃被更长匹配包含的术语。"batch normalization" 会同时命中 batch、
        # normalization、layer，全部注入既浪费预算，也提供不了额外信息。
        matched: list[tuple[str, str, str]] = []
        for entry in ordered:
            lowered = entry[0].lower()
            if any(_is_subphrase(lowered, kept[0].lower()) for kept in matched):
                continue
            matched.append(entry)
            if len(matched) >= self.max_terms:
                break

        pairs: list[tuple[str, str]] = []
        keeps: list[str] = []
        for src, tgt, policy in matched:
            if policy == POLICY_KEEP:
                keeps.append(src)
            else:
                pairs.append((src, tgt))
        return pairs, keeps

    def __len__(self) -> int:
        """
        中文：返回已编译的术语条数。

        Returns:
            int: Number of usable terms in this matcher.
        """
        return len(self._entries)
