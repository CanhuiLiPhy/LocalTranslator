"""
中文：BabelDOC 翻译器实现——本地 Hy-MT2 与云端 API 两种后端。

Translator backends plugged into BabelDOC's translation pipeline.

Two implementations live here:

:class:`LocalMTTranslator`
    Talks to the locally hosted Hy-MT2 model through llama-server. Hy-MT2 is a
    *translation-specialised* model: it follows a short, fixed instruction and
    emits a translation, but it cannot produce the batched JSON structures that
    BabelDOC's LLM-only pipeline expects. This class therefore deliberately
    leaves ``do_llm_translate`` unimplemented, which makes
    ``babeldoc.format.pdf.high_level.translator_supports_llm`` return ``False``
    and routes the document through the classic per-paragraph ``ILTranslator``.

    Because that classic path does not inject glossaries or document context
    (BabelDOC only builds those blocks on the LLM path), glossary handling is
    done here instead — matched terms are folded into the model's native prompt.

:class:`ApiTranslator`
    A thin subclass of BabelDOC's own ``OpenAITranslator``, kept so the cloud
    API option retains the full-quality LLM-only pipeline unchanged.
"""

from __future__ import annotations

import logging
import re
import threading

import httpx
import openai
from babeldoc.translator.translator import BaseTranslator, OpenAITranslator
from babeldoc.utils.atomic_integer import AtomicInteger

from local_translator.config import LocalModelSettings
from local_translator.engine.reinsert import (
    build_placeholder_grammar,
    looks_hallucinated,
    reinsert,
)

logger = logging.getLogger(__name__)

# A Chinese full stop between two digits is never right: it is a decimal point
# the model rewrote as sentence punctuation. Observed on the 1.8B model, which
# returned 「有80。7%的」 for "80.7% of", and inside table cells on the 7B.
# 两个数字之间的中文句号永远是错的：那是模型把小数点当成了句读。在 1.8B 上观察到
# 「有80。7%的」（原文 "80.7% of"），在 7B 的表格单元格里也出现过。
_DECIMAL_STOP = re.compile(r"(?<=\d)\s*。\s*(?=\d)")

# The same slip with a full-width comma, which a thousands separator turns into.
# 千分位逗号被写成全角逗号，是同一类错误。
_THOUSANDS_COMMA = re.compile(r"(?<=\d)\s*，\s*(?=\d{3}\b)")


# Marker used to hold a whitelist term through translation. Deliberately shaped
# like BabelDOC's own ``{vN}`` formula placeholders, which this model copies
# through reliably — measured across five documents, zero were lost — but in a
# separate ``k`` namespace so the two can never be confused.
# 用于让白名单术语原样穿过翻译的标记。形状刻意模仿 BabelDOC 自己的 {vN} 公式占位符
# ——本模型对它们的复制很可靠，五篇文档实测零丢失——但用独立的 k 命名空间，两者绝不会
# 混淆。
_KEEP_MARKER = "{{k{index}}}"
_KEEP_MARKER_RE = re.compile(r"\{k(\d+)\}")

# Whether the paragraph currently being translated is one where a whitelist
# term must survive no matter what. Held per thread because BabelDOC translates
# paragraphs from a worker pool, and set by the layout filter, which is the only
# place that knows what region a paragraph came from.
# 当前正在翻译的段落是否属于「白名单术语必须原样保留」的场景。按线程保存，因为
# BabelDOC 用线程池翻译段落；由版面过滤器设置，那里是唯一知道段落来自哪个区域的地方。
_CONTEXT = threading.local()


def set_strict_terms(strict: bool) -> None:
    """
    中文：标记当前线程正在翻译的段落是否需要严格保护白名单术语。

    Mark whether the paragraph this thread is translating needs strict terms.

    Args:
        strict (bool): ``True`` inside a figure or a table.

    Returns:
        None.
    """
    _CONTEXT.strict = strict


def strict_terms() -> bool:
    """
    中文：当前线程是否处于严格保护场景。

    Whether this thread is translating a paragraph that needs strict terms.

    Returns:
        bool: ``False`` unless :func:`set_strict_terms` said otherwise.
    """
    return getattr(_CONTEXT, "strict", False)


def protect_keeps(text: str, keeps: list[str]) -> tuple[str, dict[str, str]]:
    """
    中文：把白名单术语替换成标记，使其不可能被翻译。

    Replace whitelist terms with markers so they cannot be translated.

    Writing the term as an identity mapping in the prompt — ``baseline=baseline``
    — is what the model mostly obeys, but not always: measured on one page of
    arXiv:2608.19072 it kept 「Agent」, 「token」 and 「pass@」 every time and still
    rendered 「baseline」 as 「基线」 five times out of seven. A marker removes the
    choice.

    Args:
        text (str): Paragraph text as it will be sent.
        keeps (list[str]): Source terms that must survive in English.

    Returns:
        tuple[str, dict[str, str]]: The text with markers substituted in, and
            the marker-to-term mapping needed to undo it. The mapping is empty
            when nothing matched, in which case the text is returned unchanged.
    """
    if not text or not keeps:
        return text, {}

    mapping: dict[str, str] = {}
    # Longest first, so 「flow matching」 is protected as one term rather than
    # having 「matching」 taken out from under it.
    # 先长后短，让「flow matching」作为一个术语被保护，而不是被「matching」抢先拆掉。
    for term in sorted({t for t in keeps if t.strip()}, key=len, reverse=True):
        pattern = re.compile(
            (r"\b" if term[:1].isalnum() else "") + re.escape(term)
            + (r"\b" if term[-1:].isalnum() else ""),
            re.IGNORECASE)
        if not pattern.search(text):
            continue
        marker = _KEEP_MARKER.format(index=len(mapping) + 1)
        # The term is written back in the glossary's own spelling, which is the
        # point of the whitelist: 「chatgpt」 in the source becomes 「ChatGPT」.
        # 术语按术语库里的写法还原，这正是白名单的意义：源文的「chatgpt」还原为
        # 「ChatGPT」。
        mapping[marker] = term
        text = pattern.sub(marker, text)

    return text, mapping


def restore_keeps(text: str, mapping: dict[str, str]) -> str:
    """
    中文：把标记换回白名单术语；模型弄丢的标记不做补救。

    Put the whitelist terms back in place of their markers.

    A marker the model dropped is simply absent, which leaves that one term
    translated — the same outcome as before protection existed, so the repair
    can only improve on it.

    Args:
        text (str): The model's output.
        mapping (dict[str, str]): Marker to term, from :func:`protect_keeps`.

    Returns:
        str: The output with every marker it still carries replaced.
    """
    if not mapping or not text:
        return text
    for marker, term in mapping.items():
        text = text.replace(marker, term)
    # A marker the model reshaped — spacing inside the braces — is still
    # recoverable by its number.
    # 被模型改了形状（花括号里多了空格）的标记，仍可按编号还原。
    return _KEEP_MARKER_RE.sub(
        lambda m: mapping.get(_KEEP_MARKER.format(index=m.group(1)), m.group(0)), text)


# A run of letters long enough to be a word rather than an initial.
# 足以算作一个词而非缩写首字母的字母串。
_WORD_RUN = re.compile(r"[A-Za-z\u00c0-\u024f]{2,}|[\u4e00-\u9fff]")

# What counts as one unit of source: a run of letters, or a number with its
# decimal point and thousands separators.
# 源文中的一个计量单位：一串字母，或一个含小数点与千位分隔符的数字。
_SOURCE_TOKEN = re.compile(r"[A-Za-z\u00c0-\u024f]+|\d[\d.,]*")

# Markers and style tags, which belong to neither side's length.
# 占位符与样式标记，两边的长度都不该算它们。
_MARKUP = re.compile(r"\{\s*[vk]\d+\s*\}|</?style[^>]*>")

# A punctuation mark repeated this many times over is not language. Six is above
# the three of a mathematical ellipsis and far below the twenty of a contents
# leader, both of which appear in the source as well and are excused by the
# comparison against it.
# 一个标点重复到这个次数就不是语言了。六高于数学省略号的三，远低于目录引线的二十，
# 而那两者源文里也有，会被与源文的比较放行。
_REPEATED_MARK = re.compile(r"([^\w\s])(?:\s*\1){5,}")

# Characters of output allowed per unit of source, plus a constant that keeps
# the rule off short fragments. English-to-Chinese is normally *shorter* in
# characters than its source, so five is loose on purpose: this is a guard
# against output that is not a translation at all, not a style check.
# 每单位源文允许的译文字符数，外加一个常数以免短片段被误伤。英译中的字符数通常比源文
# *更少*，五是有意放宽的：这道闸拦的是「根本不是译文」的输出，不是文风检查。
_CHARS_PER_TOKEN = 5.0
_LENGTH_SLACK = 16


def _terms_were_lost(translated: str, mapping: dict[str, str]) -> bool:
    """
    中文：判断被保护的术语是否真的丢了——标记和原词都不在，才算丢。

    Decide whether a protected term failed to survive.

    A missing marker is not by itself a failure. Models routinely drop the
    marker and write the English word in its place, which is the outcome the
    whitelist wanted; rejecting those would leave a whole axis label in English
    for no gain. What counts as lost is a term that appears neither as its
    marker nor as itself.

    Args:
        translated (str): The model's output, markers not yet restored.
        mapping (dict[str, str]): Marker to term, as :func:`protect_keeps` built
            it.

    Returns:
        bool: ``True`` when at least one term is gone both ways.
    """
    lowered = translated.lower()
    return any(marker not in translated and term.lower() not in lowered
               for marker, term in mapping.items())


# A run of formula placeholders separated by nothing a reader would call a word:
# spaces, brackets, an operator, a comma. Two or more of them in a row.
# 一串公式占位符，中间只隔着算不上「词」的东西：空格、括号、运算符、逗号。至少两个。
_FORMULA_RUN = re.compile(r"\{v\d+\}(?:[^\w\u4e00-\u9fff{}]{0,4}\{v\d+\})+")

# The first placeholder of a run, which is what the run collapses to.
# 一串占位符里的第一个，整串合并后就用它代表。
_FIRST_PLACEHOLDER = re.compile(r"\{v\d+\}")


# Brackets a run may have opened and not yet closed.
# 一串占位符里可能打开却尚未闭合的括号。
_OPENERS = "([{"
_CLOSERS = ")]}"


def _closed(text: str, start: int, run: str) -> str:
    """
    中文：把紧跟在占位符串后面、用于闭合串内括号的右括号一并收进这一串。

    Extend a run to cover the closing brackets it opened.

    The run pattern has to end on a placeholder, so ``{v8}({v9}){v10}({v11})``
    matches only up to ``{v11}`` and leaves the final ``)`` behind as loose text
    the model is free to lose. Taking the closers back makes the run whole.

    Args:
        text (str): The paragraph the run was found in.
        start (int): Where the run begins in it.
        run (str): The run as matched.

    Returns:
        str: The run, extended over any immediately following closers that
            balance a bracket opened inside it.
    """
    depth = sum(1 for char in run if char in _OPENERS) - sum(
        1 for char in run if char in _CLOSERS)
    index = start + len(run)
    while depth > 0 and index < len(text) and text[index] in _CLOSERS:
        run += text[index]
        depth -= 1
        index += 1
    return run


def merge_formula_runs(text: str) -> tuple[str, dict[str, str]]:
    """
    中文：把连成一片的公式占位符合并成一个，别让模型把一条公式读成一串列表。

    Collapse a run of adjacent formula placeholders into a single placeholder.

    **What goes wrong without this.** BabelDOC cuts an inline formula wherever
    its own segmentation finds a boundary, so ``Y_ℓ/2τ ∈ [0,1]`` reaches the
    translator as ``{v3}({v4}){v5}({v6}) {v7}`` — five placeholders and some
    punctuation. A translation model reads five opaque tokens in a row as a
    list and writes them out as one: 「{v2}、{v3}、{v4}、{v5}、{v6}」, and the
    formula is printed on the page with ideographic commas through the middle
    of it. Measured over 6,484 cached paragraphs, 19% carry such a run and 6%
    of the outputs join one with 、 or ，.

    Collapsing the run to its first placeholder gives the model one atom where
    it had several, so there is nothing to punctuate between. The run is put
    back by :func:`restore_formula_runs` after the translation returns.

    Only runs of two or more are touched, and only where the characters between
    them are neither letters nor Chinese — 「{v1} and {v2}」 is two formulas in a
    sentence and stays that way.

    Args:
        text (str): Paragraph as BabelDOC built it, placeholders included.

    Returns:
        tuple[str, dict[str, str]]: The text with each run collapsed, and a map
            from the surviving placeholder to the run it stands for. The map is
            empty when there was nothing to merge.
    """
    runs: dict[str, str] = {}
    if not text:
        return text, runs

    def collapse(match: re.Match) -> str:
        """中文：整串换成它的第一个占位符，并把没闭合的括号一起收进来。 / Collapse the run."""
        run = _closed(text, match.start(), match.group(0))
        first = _FIRST_PLACEHOLDER.search(run)
        if first is None or first.group(0) in runs:
            return run
        runs[first.group(0)] = run
        return first.group(0)

    out = []
    end = 0
    for match in _FORMULA_RUN.finditer(text):
        if match.start() < end:
            continue
        run = _closed(text, match.start(), match.group(0))
        out.append(text[end:match.start()])
        out.append(collapse(match))
        end = match.start() + len(run)
    out.append(text[end:])
    return "".join(out), runs


def restore_formula_runs(translated: str, runs: dict[str, str]) -> str:
    """
    中文：把合并掉的占位符串展开回去。

    Put each merged run back in place of the placeholder that stood for it.

    Args:
        translated (str): The model's output, in merged form.
        runs (dict[str, str]): Map from :func:`merge_formula_runs`.

    Returns:
        str: The output with every run expanded. A placeholder the model lost
            simply leaves its run out, which is the same loss the ordinary
            placeholder checks already handle.
    """
    for first, run in runs.items():
        if first in translated:
            translated = translated.replace(first, run, 1)
    return translated


def output_is_implausible(source: str, translated: str) -> bool:
    """
    中文：判断一段输出是否根本不是这段源文的译文。

    Decide whether an output cannot be a translation of its input.

    Two signals, both chosen so that legitimate text cannot trip them:

    * **A repeated mark the source does not have.** A run of one punctuation
      character is the shape a degenerate decode takes. Comparing against the
      source excuses the ellipsis in ``a1, a2, . . . , aN`` and the dot leader on
      a contents row, which arrive already in the text to be translated.
    * **Length far beyond what the source can justify.** Counted as characters of
      output against units of source, where a unit is a word or a number — the
      two things a technical fragment is made of. A faithful Chinese rendering is
      normally shorter than its English source, so the allowance is several times
      what is needed.

    Args:
        source (str): The paragraph as it was sent, markers and all.
        translated (str): The model's output.

    Returns:
        bool: ``True`` when the output should be discarded and the source kept.
    """
    src = _MARKUP.sub(" ", source or "")
    out = _MARKUP.sub(" ", translated or "")
    if not out.strip():
        return False
    if _REPEATED_MARK.search(out) and not _REPEATED_MARK.search(src):
        return True
    units = len(_SOURCE_TOKEN.findall(src))
    if not units:
        return False
    payload = sum(1 for char in out if not char.isspace())
    return payload > units * _CHARS_PER_TOKEN + _LENGTH_SLACK


def repair_decimals(text: str) -> str:
    """
    中文：把数字之间被写成中文标点的小数点与千分位逗号改回半角。

    Restore a decimal point or thousands separator the model made Chinese.

    Args:
        text (str): Translated text.

    Returns:
        str: The same text with 「。」 between digits turned back into ``.`` and
            「，」 before a group of three digits turned back into ``,``.
    """
    if not text:
        return text
    return _THOUSANDS_COMMA.sub(",", _DECIMAL_STOP.sub(".", text))


# Hy-MT2 is prompted with a language *name*, not an ISO code.
# Hy-MT2 的提示词里用语言全名而非 ISO 代码。
LANGUAGE_NAMES: dict[str, str] = {
    "zh": "Chinese",
    "zh-cn": "Chinese",
    "zh-hans": "Chinese",
    "zh-tw": "Traditional Chinese",
    "zh-hant": "Traditional Chinese",
    "en": "English",
    "ja": "Japanese",
    "ko": "Korean",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "pt": "Portuguese",
    "ru": "Russian",
    "ar": "Arabic",
    "it": "Italian",
}

# Formula placeholders BabelDOC embeds in paragraph text, e.g. "{v3}".
# BabelDOC 嵌入段落文本的公式占位符，如 "{v3}"。
FORMULA_PLACEHOLDER_RE = re.compile(r"\{\s*v\s*\d+\s*\}")

# Rich-text style markers, e.g. "<style id='2'>" … "</style>".
# 富文本样式标记。
STYLE_OPEN_RE = re.compile(r"<\s*style\s+id\s*=\s*'\s*\d+\s*'\s*>")
STYLE_CLOSE_RE = re.compile(r"<\s*/\s*style\s*>")

# The closing tag, written the way BabelDOC's restoration expects to find it.
# 结束标记，写法与 BabelDOC 还原时的预期一致。
STYLE_CLOSE = "</style>"


def language_name(code: str) -> str:
    """
    中文：把语言代码转成 Hy-MT2 提示词使用的语言全名。

    Convert a language code into the natural-language name used in the prompt.

    Args:
        code (str): ISO-ish language code such as ``"zh"`` or ``"zh-TW"``.

    Returns:
        str: Human-readable language name, e.g. ``"Chinese"``. Unknown codes are
            returned unchanged so an unusual target language still produces a
            sensible instruction.
    """
    return LANGUAGE_NAMES.get(code.lower(), code)


def extract_placeholders(text: str) -> list[str]:
    """
    中文：提取文本中所有需要原样保留的占位符。

    Collect every placeholder token that must survive translation untouched.

    Args:
        text (str): Source paragraph text as handed over by BabelDOC.

    Returns:
        list[str]: Distinct placeholder strings, in order of first appearance.
            Includes formula placeholders (``{v1}``) and rich-text style tags
            (``<style id='0'>``). The closing ``</style>`` is not tracked
            separately because it always pairs with an opening tag.
    """
    found: list[str] = []
    for match in FORMULA_PLACEHOLDER_RE.finditer(text):
        token = match.group(0)
        if token not in found:
            found.append(token)
    for match in STYLE_OPEN_RE.finditer(text):
        token = match.group(0)
        if token not in found:
            found.append(token)
    return found


# Any occurrence of the opening delimiter, well formed or not. Comparing this
# count with the number of well-formed tags is what catches a shredded tag.
# 起始定界符的任何出现，无论格式是否正确。用它的数量与格式正确的标记数相比，
# 就能发现被打碎的标记。
_STYLE_FRAGMENT_RE = re.compile(r"<\s*style\b")

# Anything that looks like either half of a style tag, however malformed.
# 形似样式标记任一半的内容，无论格式如何损坏。
_STYLE_ANY_RE = re.compile(r"<\s*/?\s*style\b[^>]*>?")


def _is_style_marker(token: str) -> bool:
    """
    中文：判断一个标记是不是富文本样式标记（而非公式占位符）。

    Report whether a marker is a rich-text style tag rather than a formula.

    Args:
        token (str): A marker from :func:`missing_placeholders`.

    Returns:
        bool: ``True`` for ``<style id='N'>`` and ``</style>``.
    """
    return bool(STYLE_OPEN_RE.fullmatch(token) or STYLE_CLOSE_RE.fullmatch(token))


def strip_style_markers(text: str) -> str:
    """
    中文：删掉文本里所有富文本样式标记。

    Remove every rich-text style tag from a string.

    Used when a style run could not be carried through translation: a lone
    opening tag left in the output would be rendered as literal text by
    BabelDOC's restoration, which is worse than losing the styling it stood for.

    Args:
        text (str): Model output, possibly holding unpaired tags.

    Returns:
        str: The same text with the tags taken out.
    """
    # Sweeps malformed openings too. This runs only once the tags are known to
    # be unusable, and what is left of a shredded ``<style <style id='5'>id='3'>``
    # is exactly what would otherwise print on the page.
    # 连格式损坏的起始标记一并清除。此函数只在标记已判定不可用时才运行，而被打碎的
    # <style <style id='5'>id='3'> 剩下的部分，正是不清就会印在页面上的东西。
    return _STYLE_ANY_RE.sub("", text)


def _spans(text: str) -> list[tuple[int, int]]:
    """
    中文：找出文本里每个样式段的范围（含首尾标记）。

    Locate every style span in a string.

    Args:
        text (str): Text holding ``<style id='N'>`` … ``</style>`` pairs.

    Returns:
        list[tuple[int, int]]: One ``(start, end)`` per closed span, where
            ``start`` is the index of the opening tag and ``end`` the index just
            past the closing one. An opening tag with no closer is skipped:
            nothing can be said about where the run it opened was meant to end.
    """
    spans: list[tuple[int, int]] = []
    for opening in STYLE_OPEN_RE.finditer(text):
        closing = STYLE_CLOSE_RE.search(text, opening.end())
        if closing is None:
            continue
        if any(start < opening.start() < end for start, end in spans):
            continue
        spans.append((opening.start(), closing.end()))
    return spans


def _enclosing_span(text: str, index: int) -> tuple[int, int] | None:
    """
    中文：返回包含指定位置的样式段，没有则返回 None。

    Return the style span containing a position.

    Args:
        text (str): The text to search.
        index (int): Position of interest.

    Returns:
        tuple[int, int] | None: The innermost span that strictly contains
            ``index``, or ``None`` when the position is outside every span.
    """
    for start, end in _spans(text):
        if start < index < end:
            return (start, end)
    return None


def hoist_placeholders(source: str, translated: str) -> str:
    """
    中文：把被模型挪进样式段里的公式占位符挪回段外。

    Move a formula placeholder back out of a style span it does not belong in.

    **Why this is necessary.** BabelDOC restores a translation by scanning it
    for a single combined pattern that alternates formula placeholders with
    whole style spans. Alternation takes the leftmost match, so a ``{v1}`` that
    the model has moved *inside* ``<style id='2'>…</style>`` is consumed as part
    of that span, and the span's inner text then goes through
    ``remove_placeholder`` — which deletes it. The formula disappears from the
    page with no trace and no warning.

    A bullet is the case that showed this: a list item begins ``{v1}<style
    id='2'>At a fixed…``, the model answered ``<style id='2'>{v1}在固定的…``, and
    the item lost its 「•」.

    Only placeholders that were outside every span in the source are moved, and
    they are moved to just outside the span that swallowed them — before it when
    the source had the placeholder ahead of that span's opening tag, after it
    otherwise. A placeholder the source itself put inside a span is left alone.

    Args:
        source (str): The request as BabelDOC built it.
        translated (str): The model's output.

    Returns:
        str: The output with any swallowed placeholder lifted back out.
    """
    if not translated or STYLE_OPEN_RE.search(translated) is None:
        return translated

    outside = {
        match.group(0)
        for match in FORMULA_PLACEHOLDER_RE.finditer(source)
        if _enclosing_span(source, match.start()) is None
    }
    if not outside:
        return translated

    # One placeholder at a time, re-scanning after each move: every move shifts
    # the offsets of everything after it.
    # 一次只搬一个，搬完重新扫描：每搬一次，其后所有位置都会移动。
    for _ in range(len(outside)):
        moved = False
        for match in FORMULA_PLACEHOLDER_RE.finditer(translated):
            token = match.group(0)
            if token not in outside:
                continue
            span = _enclosing_span(translated, match.start())
            if span is None:
                continue
            start, end = span
            body = translated[:match.start()] + translated[match.end():]
            shift = match.end() - match.start()
            target = start if _leads(source, token) else end - shift
            translated = body[:target] + token + body[target:]
            moved = True
            break
        if not moved:
            break
    return translated


def _leads(source: str, token: str) -> bool:
    """
    中文：判断该占位符在原文中是否位于第一个样式标记之前。

    Report whether a placeholder precedes the first style tag in the source.

    Args:
        source (str): The request as BabelDOC built it.
        token (str): The placeholder to locate, e.g. ``"{v1}"``.

    Returns:
        bool: ``True`` when the placeholder appears before any style tag — the
            list-bullet case, where it has to come back to the front.
    """
    where = source.find(token)
    opening = STYLE_OPEN_RE.search(source)
    return where >= 0 and (opening is None or where < opening.start())


# A style span, with its id and its contents. Used to put back a span whose
# contents were only punctuation — see :func:`restore_symbol_spans`.
# 一个样式段，连同其 id 与内容。用于还原内容仅为标点的样式段——见 restore_symbol_spans。
_STYLE_SPAN = re.compile(
    r"<\s*style\s+id\s*=\s*'\s*(?P<id>\d+)\s*'\s*>(?P<body>.*?)<\s*/\s*style\s*>",
    re.DOTALL)

# Contents that carry no word and are therefore the same in every language.
# 不含任何词、因而在任何语言里都一样的内容。
_SYMBOL_ONLY = re.compile(r"^[^\w\s]{1,3}$")


def restore_symbol_spans(source: str, translated: str) -> str:
    """
    中文：样式段里如果原本只是标点，就把原样点回去——别让模型把小数点译成句号。

    Put back a style span whose source contents were nothing but punctuation.

    A decimal point set in its own font arrives as a span of its own:
    ``a pass@8 score of 13<style id='1'>.</style>33%``. The model reads the
    contents as text to translate and answers ``13<style id='1'>。</style>33%``
    — a Chinese full stop where the decimal point was, which the ordinary
    decimal repair cannot see because the two digits are on the far sides of a
    tag. Counted over one 23-page paper, ten numbers were broken this way.

    A span holding no word needs no translation at all, so whatever came back is
    replaced by what went in. Length is capped at three characters so that this
    only ever restores a mark, never a stretch of text the model rewrote.

    Args:
        source (str): The request as BabelDOC built it, style tags included.
        translated (str): The model's output.

    Returns:
        str: The output with each punctuation-only span holding its source
            contents again. Spans the model dropped are left alone; that is the
            business of :func:`style_tags_intact`.
    """
    wanted = {match.group("id"): match.group("body")
              for match in _STYLE_SPAN.finditer(source or "")
              if _SYMBOL_ONLY.match(match.group("body") or "")}
    if not wanted:
        return translated

    def restore(match: re.Match) -> str:
        """中文：把该样式段的内容换回原文。 / Put the source contents back."""
        body = wanted.get(match.group("id"))
        if body is None or body == match.group("body"):
            return match.group(0)
        return match.group(0).replace(match.group("body"), body, 1)

    return _STYLE_SPAN.sub(restore, translated or "")


def style_tags_intact(source: str, translated: str) -> bool:
    """
    中文：判断译文里的样式标记是否与原文一一对应且格式完好。

    Decide whether the style tags survived translation intact.

    Three things have to hold, and a real failure broke each of them: the same
    opening tags in the same order, the same number of closing tags, and no
    stray ``<style`` beyond those — the model returned
    ``</style><style <style id='5'>id='3'>``, which passes a naive
    "is the token present" test and renders as literal angle brackets on the
    page.

    Args:
        source (str): The request as BabelDOC built it.
        translated (str): The model's output.

    Returns:
        bool: ``True`` when the tags can be handed to BabelDOC's restoration.
    """
    if STYLE_OPEN_RE.search(source) is None:
        # Nothing was asked for, so nothing may appear.
        # 原文没有要求任何标记，译文里也就不该出现。
        return _STYLE_FRAGMENT_RE.search(translated) is None

    opens = STYLE_OPEN_RE.findall(translated)
    if opens != STYLE_OPEN_RE.findall(source):
        return False
    if len(_STYLE_FRAGMENT_RE.findall(translated)) != len(opens):
        return False
    return len(STYLE_CLOSE_RE.findall(translated)) == len(
        STYLE_CLOSE_RE.findall(source)
    )


def missing_placeholders(source: str, translated: str) -> list[str]:
    """
    中文：找出译文里丢失的占位符。

    Determine which placeholders present in the source vanished from the output.

    Whitespace inside placeholders is normalised before comparison, since a
    model may render ``{v1}`` as ``{ v1 }`` without changing its meaning to the
    downstream regex-based restoration in BabelDOC.

    Args:
        source (str): Original paragraph text.
        translated (str): Model output.

    Returns:
        list[str]: Placeholders that appear in ``source`` but not in
            ``translated``. Empty when the round trip is clean.
    """

    def normalise(s: str) -> str:
        return re.sub(r"\s+", "", s)

    norm_out = normalise(translated)
    lost = [p for p in extract_placeholders(source) if normalise(p) not in norm_out]

    # A style run needs both halves. The closing tag carries no id, so it cannot
    # be listed like the others — an output holding fewer of them than the
    # source is missing one, and BabelDOC then has no run to apply the style to.
    # 一个样式段需要首尾两半。结束标记不带编号，无法像其它标记那样逐个列出——
    # 输出里的数量少于原文即为丢失，BabelDOC 也就找不到该套用样式的那一段。
    if len(STYLE_CLOSE_RE.findall(translated)) < len(STYLE_CLOSE_RE.findall(source)):
        lost.append(STYLE_CLOSE)
    return lost


# A section number or list marker at the head of a paragraph: "4", "3.2", "A.1",
# "1)", "(2)", "iv.". Bounded to three digits and three dot-separated levels,
# because a longer run is a measurement or a year rather than a number that
# labels something.
# 段首的章节号或列表标号。位数限制为三位、层级限制为三级：更长的串是数值或年份，
# 而不是给某物编号的标号。
_LEADING_NUMBER = re.compile(
    r"^(\s*[\(（]?(?:\d{1,3}(?:\.\d{1,3}){0,3}|[A-Za-z](?:\.\d{1,3}){1,3}"
    r"|[A-Za-z]|[ivxIVX]{1,4})[\)）\.．、]?\s+)(?=\S)"
)

# The remainder has to begin like a title or an item: a letter, a digit or a
# CJK character. Requiring this keeps the split off "1 000 samples" and similar.
# 剩余部分须以字母、数字或汉字开头。这条限制可避免切开 "1 000 samples" 之类。
_AFTER_NUMBER = re.compile(r"^[A-Za-z\u4e00-\u9fff\d]")


# Revision of what this class stores against a paragraph, folded into the cache
# key. Covers how the request is built *and* how the answer is repaired, because
# what the cache holds is the repaired answer.
# 1 -> 2: leading section numbers and list markers are no longer sent.
# 2 -> 3: a placeholder the model moved into a style span is lifted back out,
#         and style markers are no longer reinserted at guessed positions.
# 3 -> 4: a run of adjacent formula placeholders is sent as one placeholder, and
#         a style span holding only punctuation comes back as its source did.
# 本类针对一个段落所存内容的版本号，计入缓存键。既涵盖请求的构造方式，也涵盖答案的
# 修复方式——因为缓存里存的正是修复后的答案。
# 1 → 2：不再把段首章节号与列表标号送进模型。
# 2 → 3：模型挪进样式段里的占位符会被取回段外。
# 3 → 4：连成一片的公式占位符合并为一个再送出；仅含标点的样式段按原文还原。
_REQUEST_REVISION = 4


def split_leading_number(text: str) -> tuple[str, str]:
    """
    中文：把段首的章节号/列表标号与正文拆开。

    Split a leading section number or list marker off a paragraph.

    Args:
        text (str): Paragraph text.

    Returns:
        tuple[str, str]: ``(marker, remainder)``. The marker keeps its trailing
            whitespace so re-joining reproduces the original spacing exactly;
            it is empty when the paragraph does not start with one, in which
            case the remainder is the input unchanged.
    """
    if not text:
        return "", text
    match = _LEADING_NUMBER.match(text)
    if not match:
        return "", text
    marker = match.group(1)
    remainder = text[match.end():]
    # Something has to be left to translate, and it has to look like prose or a
    # title rather than the continuation of a number.
    # 必须还有可译的内容，且其开头要像正文或标题，而不是数字的延续。
    if len(remainder.strip()) < 2 or not _AFTER_NUMBER.match(remainder):
        return "", text
    # A bare number with no punctuation after it — "1 000 samples" — is a value
    # whose digits continue, not a label. A marker that closes with punctuation
    # ("1)", "3.2.") may be followed by a digit, as in "1) 3D reconstruction".
    # 后面不带标点的裸数字——"1 000 samples"——是数值的延续而非标号。带闭合标点的标号
    # （"1)"、"3.2."）后面可以跟数字，例如 "1) 3D reconstruction"。
    if marker.strip()[-1].isdigit() and remainder[0].isdigit():
        return "", text
    # A marker with no punctuation is only a marker when a heading follows it.
    # "A Alternative Experiments" labels an appendix; "A first line of work
    # studies data repetition" is a sentence whose first word is an article, and
    # splitting it left a bare "A" sitting in front of the translation. Headings
    # capitalise; sentences continue in lower case, and that is the whole of the
    # difference visible here.
    # 不带标点的标号，只有当其后是标题时才算标号。"A Alternative Experiments" 是附录
    # 编号；"A first line of work studies data repetition" 则是一个以冠词开头的句子，
    # 切开它会在译文前留下一个孤零零的 "A"。标题首字母大写，句子接着小写往下走——
    # 在这里能看到的区别仅此一条。
    if marker.strip()[-1].isalnum() and remainder[0].islower():
        return "", text
    return marker, remainder


class TranslationGuards:
    """
    中文：每一次翻译前后各做一件事——标题查表直接返回，返回的译文修掉替换字符。

    The two things done around every translation.

    **Before.** A standard section heading is answered from a table instead of
    being sent (see :mod:`engine.headings`). Across 33 papers those headings are
    the only text different papers share, the model renders them inconsistently
    — 「参考文献」 and 「参考资料」 for the same word — and sometimes wrongly:
    ``Epochs`` came back as 「时代」, ``Final`` as 「决赛」. A landmark a reader
    scrolls back to should not change its name between papers. Switched off by
    ``PdfSettings.translate_headings``.

    **After.** :func:`repair_replacement` on the result.

    Putting the repair in ``do_translate`` alone is not enough, and the gap is
    easy to miss. ``BaseTranslator.translate`` answers a cache hit by returning
    the cached string directly::

        cache = self.cache.get(text)
        if cache is not None:
            self.translate_cache_call_count += 1
            return cache

    ``do_translate`` is never called on that path, so a corrupted translation
    written to the cache once would be served from it forever — and the cache is
    exactly where a paragraph that has been translated before comes from.

    Overriding both entry points closes it from the other side: the text is
    repaired on the way out whether it was just produced or just recalled, and
    ``do_translate`` still repairs it on the way *in* so nothing corrupt is
    written to the cache in the first place.
    在 do_translate 里挂修复是不够的，而这个缺口很容易漏掉：BaseTranslator.translate 命中
    缓存时直接返回缓存字符串（见上），根本不走 do_translate。于是一条写坏过一次的译文会被
    永远地供出来——而缓存恰恰就是「这段以前译过」时取译文的地方。

    覆盖两个入口即可从另一侧堵上：无论文本是刚生成的还是刚取回的，出门时都过一遍修复；
    而 do_translate 里那一道仍然保留，好让写进缓存的东西本身就是干净的。
    """

    # Set by :func:`worker._build_translator` from the job's settings. Default
    # False so a translator built without settings — a test, a probe — behaves
    # the way the shipped default does.
    # 由 worker._build_translator 依任务设置写入。默认 False，使得不带设置构造出来的
    # 翻译器（测试、探针）与发布默认值行为一致。
    translate_headings: bool = False

    def _settled_heading(self, text: str) -> str | None:
        """
        中文：这一块若是标准章节标题，返回固定译法。

        Return the table's rendering when this block is a standard heading.

        Args:
            text (str): Source block.

        Returns:
            str | None: The settled rendering for a known heading, the source
                unchanged for a label that must not be translated at all (see
                :func:`headings.verbatim`), or ``None`` when the block should go
                to the model.
        """
        if self.translate_headings:
            return None
        from local_translator.engine import headings

        if headings.verbatim(text):
            return text
        return headings.rendering(text)

    def translate(self, text, ignore_cache=False, rate_limit_params: dict = None):
        """
        中文：翻译一段；标题查表直接返回，其余照常，出门前修掉替换字符。

        Translate one paragraph.

        Args:
            text (str): Source paragraph.
            ignore_cache (bool): Passed through to the base implementation.
            rate_limit_params (dict): Passed through unchanged.

        Returns:
            str: The translation, with any spurious replacement character gone.
        """
        settled = self._settled_heading(text)
        if settled is not None:
            return settled
        return repair_replacement(
            super().translate(text, ignore_cache, rate_limit_params), text)

    def llm_translate(self, text, ignore_cache=False, rate_limit_params: dict = None):
        """
        中文：LLM 路径上的同一套。

        The same on the LLM path.

        Args:
            text (str): Source paragraph.
            ignore_cache (bool): Passed through to the base implementation.
            rate_limit_params (dict): Passed through unchanged.

        Returns:
            str: The translation, repaired.
        """
        settled = self._settled_heading(text)
        if settled is not None:
            return settled
        return repair_replacement(
            super().llm_translate(text, ignore_cache, rate_limit_params), text)


class LocalMTTranslator(TranslationGuards, BaseTranslator):
    """
    中文：调用本地 llama-server 上的 Hy-MT2 模型进行段落翻译。

    Paragraph translator backed by the local Hy-MT2 model.

    The class name registered with BabelDOC's cache must stay within 20
    characters (see ``babeldoc.translator.cache``), hence the short ``name``.

    Attributes:
        name (str): Cache key prefix identifying this engine.
        settings (LocalModelSettings): Endpoint and sampling configuration.
        glossary_provider (Callable | None): Optional callable mapping a
            paragraph's text to ``(translate_pairs, keep_terms)``; see
            :meth:`set_glossary_provider`.
        token_count (AtomicInteger): Total tokens reported by the server.
        prompt_token_count (AtomicInteger): Prompt tokens consumed.
        completion_token_count (AtomicInteger): Tokens generated.
        placeholder_retry_count (AtomicInteger): Paragraphs retried because a
            placeholder was dropped.
        placeholder_failure_count (AtomicInteger): Paragraphs that fell back to
            the original text after retrying.
    """

    name = "hy-mt2"

    def __init__(
        self,
        lang_in: str,
        lang_out: str,
        settings: LocalModelSettings,
        ignore_cache: bool = False,
        glossary_rev: int = 1,
        custom_hint: str = "",
    ):
        """
        中文：初始化本地翻译器并建立到 llama-server 的连接。

        Args:
            lang_in (str): Source language code, forwarded to BabelDOC's cache key.
            lang_out (str): Target language code; also drives the prompt wording.
            settings (LocalModelSettings): Endpoint, sampling and timeout config.
            ignore_cache (bool): Bypass BabelDOC's paragraph translation cache.
            glossary_rev (int): Glossary revision folded into the cache key so
                that manually corrected terminology invalidates stale entries on
                retranslation.
            custom_hint (str): Extra instruction line appended to every prompt,
                surfaced in the UI as "custom system prompt".
        """
        super().__init__(lang_in, lang_out, ignore_cache)
        self.settings = settings
        self.custom_hint = custom_hint.strip()
        self.target_language = language_name(lang_out)
        self.glossary_provider = None
        # Set by the worker from ``glossary.hard_protect_whitelist``. Kept as a
        # plain attribute so a translator built without settings for a test
        # still has a defined value.
        # 由 worker 依据 glossary.hard_protect_whitelist 设置。用普通属性保存，使得
        # 测试里不带该设置构造出的翻译器也有确定的取值。
        self.hard_protect = False

        self.client = openai.OpenAI(
            base_url=settings.base_url,
            # llama-server ignores the key but the SDK requires a non-empty one.
            # llama-server 不校验密钥，但 SDK 要求非空。
            api_key="local",
            http_client=httpx.Client(
                # Match the supervisor's direct connection: macOS system
                # proxies may intercept loopback even without proxy env vars.
                trust_env=False,
                limits=httpx.Limits(max_connections=None, max_keepalive_connections=None),
                timeout=settings.request_timeout,
            ),
        )

        self.token_count = AtomicInteger()
        self.prompt_token_count = AtomicInteger()
        self.completion_token_count = AtomicInteger()
        self.placeholder_retry_count = AtomicInteger()
        self.style_dropped_count = AtomicInteger()
        self.grammar_retry_count = AtomicInteger()
        self.implausible_count = AtomicInteger()
        self.reinsert_count = AtomicInteger()
        self.placeholder_failure_count = AtomicInteger()
        self._warn_lock = threading.Lock()

        # Anything that changes translation output must be part of the cache key,
        # otherwise a settings change would silently return stale translations.
        # 任何影响译文的参数都要进缓存键，否则改配置后会取到旧译文。
        self.add_cache_impact_parameters("engine", "hy-mt2-local")
        self.add_cache_impact_parameters("model", settings.model_file)
        self.add_cache_impact_parameters("temperature", settings.temperature)
        self.add_cache_impact_parameters("top_p", settings.top_p)
        self.add_cache_impact_parameters("top_k", settings.top_k)
        self.add_cache_impact_parameters("glossary_rev", glossary_rev)
        # Bumped whenever the text this class actually sends to the model changes
        # for the same input. The paragraph cache is keyed on the source text, so
        # a behaviour change that alters the request — such as splitting a
        # heading's section number off before sending it — is invisible to the
        # key, and every previously translated heading would keep returning the
        # old, damaged rendering from cache without the model ever being asked.
        # 每当本类对同一输入实际发给模型的文本发生变化时递增。段落缓存以原文为键，因此
        # 改变请求内容的行为变更——例如把标题的章节号剥离后再发送——对键不可见，所有此前
        # 译过的标题都会继续从缓存返回旧的错误结果，模型根本不会被问到。
        self.add_cache_impact_parameters("request_rev", _REQUEST_REVISION)
        if self.custom_hint:
            self.add_cache_impact_parameters("hint", self.custom_hint)

    # ------------------------------------------------------------------ #
    # Glossary wiring / 术语接入
    # ------------------------------------------------------------------ #

    def set_glossary_provider(self, provider) -> None:
        """
        中文：注入术语匹配回调，用于在提示词里加入术语约束。

        Attach a glossary lookup used to constrain terminology per paragraph.

        BabelDOC only injects glossary blocks on its LLM-only pipeline, which
        this translator deliberately avoids, so terminology has to be applied
        here instead.

        Args:
            provider (Callable[[str], tuple[list[tuple[str, str]], list[str]]]):
                Called with the paragraph text; returns
                ``(translate_pairs, keep_terms)`` where ``translate_pairs`` is a
                list of ``(source_term, target_term)`` and ``keep_terms`` is a
                list of source terms that must remain in the source language.
                Both lists may be empty.
        """
        self.glossary_provider = provider

    def _terminology_line(self, text: str) -> str | None:
        """
        中文：为当前段落生成术语约束行，白名单写成 X=X 恒等映射。

        Build the terminology constraint line for one paragraph.

        Both term policies are expressed the same way, as ``source=target``
        pairs, with whitelist entries rendered as identity mappings
        (``transformer=transformer``). That is not merely tidy — it is the only
        formulation Hy-MT2 actually obeys. Measured against the running model:

        - ``Terminology: attention mechanism=专注机理`` → honoured, even for a
          rendering the model would never choose on its own.
        - ``Keep in English as-is: transformer, prompt`` → ignored; the model
          still produced 变压器 and 提示词.
        - ``Terminology: transformer=transformer; prompt=prompt`` → honoured,
          both terms left in English.

        The number of injected terms is capped so the instruction stays short
        enough for a 7B translation model to keep track of.

        Args:
            text (str): Paragraph text being translated.

        Returns:
            str | None: The terminology line, or ``None`` when no term matched.
        """
        if self.glossary_provider is None:
            return None
        try:
            pairs, keep = self.glossary_provider(text)
        except Exception:
            logger.warning("Glossary lookup failed; continuing without terms", exc_info=True)
            return None

        mappings = [f"{src}={tgt}" for src, tgt in pairs]
        mappings += [f"{term}={term}" for term in keep]
        if not mappings:
            return None

        return "Terminology (use exactly these renderings): " + "; ".join(mappings) + "."

    # ------------------------------------------------------------------ #
    # Prompting / 提示词
    # ------------------------------------------------------------------ #

    def build_messages(self, text: str, *, strict: bool = False) -> list[dict[str, str]]:
        """
        中文：构造 Hy-MT2 的消息列表——约束放 system，正文走官方单行格式。

        Compose the chat messages for one paragraph.

        The split between the two messages is essential rather than stylistic.
        Hy-MT2 treats everything in the user message after its instruction line
        as material to translate, so any constraint written there is rendered
        into the output. Measured against the running model, a user message of::

            Translate the following segment into Chinese, without additional
            explanation. Keep {v1} unchanged; keep these terms in English: LLM.

            The attention mechanism {v1} scales as O(n^2) …

        produced a translation prefixed with 「保持{v1}不变；这些术语仍用英文
        表示：LLM」 — the instructions themselves, translated into the result.

        Moving every constraint into the system message removes the leakage
        entirely while keeping the constraints in force, so the user message
        holds exactly Tencent's documented format: one instruction line, a blank
        line, then the text.

        Args:
            text (str): Paragraph text to translate.
            strict (bool): Emphasise placeholder preservation. Used on the retry
                after a placeholder was dropped.

        Returns:
            list[dict[str, str]]: Chat messages, a system message followed by
                the user message. The system message is omitted when there is
                nothing to constrain.
        """
        system_lines = ["You are a professional, faithful translation engine."]

        placeholders = extract_placeholders(text)
        if placeholders and STYLE_OPEN_RE.search(text):
            # The closing tag has no id of its own, so it is not one of the
            # tokens collected above; the model still has to reproduce it.
            # 结束标记本身不带编号，不在上面收集的标记之列，但模型同样要照抄。
            placeholders = [*placeholders, STYLE_CLOSE]
        if placeholders:
            shown = ", ".join(placeholders[:8])
            if strict:
                system_lines.append(
                    f"CRITICAL: reproduce these tokens verbatim in the output, unchanged: "
                    f"{shown}. Never translate, reformat or drop them."
                )
            else:
                system_lines.append(f"Copy these tokens verbatim into the output: {shown}.")

        terminology = self._terminology_line(text)
        if terminology:
            system_lines.append(terminology)

        if self.custom_hint:
            system_lines.append(self.custom_hint)

        user = (
            f"Translate the following segment into {self.target_language}, "
            f"without additional explanation.\n\n{text}"
        )

        messages = []
        if len(system_lines) > 1:
            messages.append({"role": "system", "content": "\n".join(system_lines)})
        messages.append({"role": "user", "content": user})
        return messages

    # ------------------------------------------------------------------ #
    # BaseTranslator interface / 基类接口实现
    # ------------------------------------------------------------------ #

    def do_llm_translate(self, text, rate_limit_params: dict | None = None):
        """
        中文：故意不实现，以强制 BabelDOC 走经典逐段翻译管线。

        Intentionally unimplemented.

        ``babeldoc.format.pdf.high_level.translator_supports_llm`` probes this
        method and treats ``NotImplementedError`` as "this engine is a plain
        translator". Raising here is what selects the classic ``ILTranslator``
        pipeline, which sends one clean paragraph at a time, instead of
        ``ILTranslatorLLMOnly``, which batches paragraphs into a structured JSON
        prompt that Hy-MT2 cannot produce.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError("Hy-MT2 is a translation-only model; use do_translate")

    def do_translate(self, text: str, rate_limit_params: dict | None = None) -> str:
        """
        中文：翻译单个段落；段首的章节号/列表标号不送进模型，原样保留。

        Translate one paragraph, keeping any leading number out of the model.

        A heading's section number is not language, but a model reading
        "4 Gram Newton-Schulz" has no way to know that and treats the 4 as a
        quantity: the observed outputs were 「4 个 Gram Newton-Schulz」,
        「8项实验」 and, for "5 Symmetric GEMM Kernels in CuteDSL",
        「CuteDSL中的5个对称GEMM内核」 — the section number swallowed into the
        sentence and gone from the front, where a reader and every cross-
        reference expect it. The same happens to list markers.

        Splitting the number off and re-attaching it afterwards costs nothing
        and removes the whole class of damage, and it does so for headings and
        list items alike.

        Args:
            text (str): Paragraph text.
            rate_limit_params (dict | None): Supplied by BabelDOC.

        Returns:
            str: Translated paragraph with its leading number intact.
        """
        marker, body = split_leading_number(text)
        if not marker:
            return self._translate_paragraph(text, rate_limit_params)
        return marker + self._translate_paragraph(body, rate_limit_params)

    def _translate_paragraph(self, text: str, rate_limit_params: dict | None = None) -> str:
        """
        中文：翻译一个段落，并在输出明显不是译文时退回原文。

        Translate one paragraph, discarding an output that is not a translation.

        Two things happen here around the translation itself. Adjacent formula
        placeholders are merged into one before it and expanded after — see
        :func:`merge_formula_runs` for why a model must not be shown five of
        them in a row. And the output is checked for having come apart: a
        punctuation mark repeated across the line, or an output several times
        longer than its source can account for. Keeping the English costs the
        reader one paragraph; printing the wreckage costs them the page it is
        drawn across.

        Args:
            text (str): Paragraph text.
            rate_limit_params (dict | None): Passed through unchanged.

        Returns:
            str: Translated paragraph, or the source text when the output cannot
                be a translation of it.
        """
        merged, runs = merge_formula_runs(text)
        translated = self._translate_protected(merged, rate_limit_params)
        # A local model decodes its own tokens, so it can produce the same
        # byte-fallback artefact the cloud one does; see repair_replacement.
        # 本地模型同样要自己解码词元，因此会产生与云端一样的 byte-fallback 残留；
        # 见 repair_replacement。
        translated = repair_replacement(translated, merged)
        if translated is not merged and output_is_implausible(merged, translated):
            self.implausible_count.inc()
            with self._warn_lock:
                logger.warning(
                    "Discarding an implausible translation of %d chars for a source of "
                    "%d. Source starts with: %.60s",
                    len(translated), len(text), text,
                )
            return text
        return restore_formula_runs(translated, runs)

    def _translate_protected(self, text: str, rate_limit_params: dict | None = None) -> str:
        """
        中文：翻译单个段落；图表内的段落先把白名单术语换成标记，译后还原。

        Translate one paragraph, holding whitelist terms out of the model's
        reach where they must not be lost.

        Protection is applied inside figures and tables, and elsewhere only if
        the setting asks for it. The split is deliberate: a marker is a token
        where a word was, so it can disturb word order, and in running prose a
        rare leak reads better than a sentence built around a placeholder. In a
        chart's axis label or a table cell there is no prose to disturb, and a
        translated 「baseline」 is exactly the failure the whitelist exists to
        prevent.

        Every exit of the inner method is covered, including the one that gives
        up and returns the source text — that text carries markers too by then,
        and returning it with them in would print ``{k1}`` on the page.

        Args:
            text (str): Paragraph text.
            rate_limit_params (dict | None): Passed through unchanged.

        Returns:
            str: Translated paragraph, with whitelist terms in English.
        """
        if not ((self.hard_protect or strict_terms())
                and self.glossary_provider is not None):
            return self._translate_checked(text, rate_limit_params)

        try:
            _, keeps = self.glossary_provider(text)
        except Exception:  # noqa: BLE001
            logger.warning("Glossary lookup failed; translating unprotected", exc_info=True)
            return self._translate_checked(text, rate_limit_params)

        protected, mapping = protect_keeps(text, keeps)
        if not mapping:
            return self._translate_checked(text, rate_limit_params)

        # Once the terms are held back there may be nothing left to translate —
        # a chart's axis label is often one whitelisted word and nothing else.
        # Sending 「{k1}」 to a translation model asks it to render a token it has
        # never seen, and on 「Baseline」 it answered 「基线」: the marker gone and
        # the term translated, which is the one outcome the whitelist exists to
        # prevent. Returning the source is exact, free, and keeps its casing —
        # the glossary's own spelling of a term is not always the figure's.
        # 术语被扣下之后，可能已经没有需要翻译的内容——图表的坐标轴标签往往就是一个
        # 白名单词，别无他物。把「{k1}」送给翻译模型，是要它翻一个从未见过的记号；
        # 「Baseline」那次它答的是「基线」：标记没了，术语被翻了，恰恰是白名单要防的
        # 那一种结果。直接退回原文，既准确又不花钱，而且连大小写都与源文一致。
        if not _WORD_RUN.search(_KEEP_MARKER_RE.sub(" ", protected)):
            return text

        translated = self._translate_checked(protected, rate_limit_params)
        if strict_terms() and _terms_were_lost(translated, mapping):
            # In a figure or a table the whitelist is not advisory, and a term
            # that is neither its marker nor itself was translated after all.
            # The fragment goes back as it came.
            # 图与表里白名单不是建议：一个术语既不以标记也不以本来面目出现，就说明它终究
            # 被翻了。这一段原样退回。
            self.implausible_count.inc()
            logger.debug("A whitelist term was lost inside a drawing; keeping %.40s", text)
            return text
        return restore_keeps(translated, mapping)

    def _translate_checked(self, text: str, rate_limit_params: dict | None = None) -> str:
        """
        中文：翻译单个段落，并校验占位符是否完整保留。

        Translate one paragraph, verifying placeholder integrity.

        Strategy:
            1. Translate at the configured temperature.
            2. If any placeholder from the source is missing in the output,
               retry once at temperature 0 with an emphatic instruction — greedy
               decoding is markedly better at copying literal tokens.
            3. If the retry still loses a placeholder, return the *original*
               text. A paragraph left in English is a far smaller defect than a
               formula silently deleted from the rendered PDF.

        Args:
            text (str): Paragraph text, possibly containing ``{vN}`` formula
                placeholders and ``<style id='N'>`` rich-text markers.
            rate_limit_params (dict | None): Supplied by BabelDOC; unused here
                because local inference is bounded by the server's slot count
                rather than by a request rate.

        Returns:
            str: Translated paragraph, or the untouched source text if
                placeholders could not be preserved.
        """
        translated = self._sanitise(text, self._request(
            self.build_messages(text), self.settings.temperature))
        missing = missing_placeholders(text, translated)
        if not missing:
            return translated

        # Step 2: greedy retry. Temperature 0 copies literal tokens markedly
        # better than sampling does.
        # 第二级：贪心重试。温度 0 复制字面标记的能力明显强于采样。
        self.placeholder_retry_count.inc()
        logger.debug("Markers %s lost; retrying at temperature 0", missing)
        retried = self._sanitise(
            text, self._request(self.build_messages(text, strict=True), 0.0))
        if not missing_placeholders(text, retried):
            return retried

        # Step 3: constrained decoding. The grammar makes omission impossible,
        # but can push the model to pad text until it can satisfy the rule, so
        # the result is accepted only if its length stays plausible.
        # 第三级：约束解码。语法让遗漏不可能发生，但可能诱使模型凑字数，
        # 因此只在长度仍然合理时接受。
        grammar = build_placeholder_grammar(extract_placeholders(text))
        if grammar:
            self.grammar_retry_count.inc()
            constrained = self._sanitise(text, self._request(
                self.build_messages(text, strict=True), 0.0, grammar=grammar
            ))
            if constrained and not missing_placeholders(text, constrained):
                if not looks_hallucinated(text, constrained):
                    return constrained
                logger.debug("Constrained output looks padded; falling through to reinsertion")

        # Step 4: reinsertion, formula placeholders only. Keeping the
        # translation and putting a lost formula marker back at an aligned
        # position beats both alternatives — discarding the translation leaves
        # the paragraph unreadable, and letting the marker vanish deletes a
        # formula the reader cannot know was there.
        #
        # Style markers are deliberately **not** reinserted. A style run needs a
        # matched pair, and the missing list can only ever name ``</style>``
        # once however many runs lost their closer, so putting them back yields
        # spans that do not close. BabelDOC then consumes everything after an
        # unmatched opening tag as that span's contents and deletes any
        # placeholder inside it — which is how a list item whose text begins
        # ``{v1}`` lost its 「•」 while the paragraph around it looked fine.
        # A run that loses its weight costs the reader nothing; a bullet deleted
        # without trace costs them the structure of the list.
        # 第四级：只重插公式占位符。保留译文并把丢失的公式标记补回对齐位置，优于另两种
        # 做法——丢弃译文会让整段不可读，放任标记消失则会删掉读者无从察觉的公式。
        #
        # 样式标记刻意不重插。一个样式段需要成对的首尾标记，而缺失清单里 </style>
        # 无论丢了几个都只能出现一次，补回去必然得到不闭合的段。BabelDOC 会把未闭合
        # 开标记之后的全部内容当成该段的内容，并删掉其中的占位符——正文看似无恙，但
        # 以 {v1} 开头的列表项就是这样丢掉了它的「•」。丢掉字重对读者毫无损失，
        # 无声删掉一个项目符号则让读者失去列表的结构。
        best = retried if retried else translated
        missing = missing_placeholders(text, best)
        formulas = [marker for marker in missing if not _is_style_marker(marker)]

        if not formulas:
            # Only styling was lost, so the translation is kept and the
            # half-open tags are cleaned out of it.
            # 只丢了样式，保留译文并清掉其中残缺的标记。
            self.style_dropped_count.inc()
            logger.debug("Style markers %s lost; keeping the translation", missing)
            return strip_style_markers(best)

        repaired, ok = reinsert(text, best, formulas)
        if ok:
            self.reinsert_count.inc()
            if style_tags_intact(text, repaired):
                return repaired
            self.style_dropped_count.inc()
            return strip_style_markers(repaired)

        # Step 5: give up and keep the source. Reached only when the paragraph
        # is mostly markup, where there is no prose to anchor a guess to.
        # 第五级：放弃并保留原文。只有段落几乎全是标记、无从锚定时才会到这里。
        self.placeholder_failure_count.inc()
        with self._warn_lock:
            logger.warning(
                "Keeping original text: markers %s could not be preserved or reinserted. "
                "Source starts with: %.80s",
                missing,
                text,
            )
        return text

    def _sanitise(self, source: str, translated: str) -> str:
        """
        中文：样式标记一旦残缺或被打碎，就整体清掉，绝不让它印到页面上。

        Remove the style tags from an output whose tags did not survive whole.

        BabelDOC's restoration consumes the tags it recognises and leaves the
        rest as text, so a mangled tag becomes literal angle brackets in the
        middle of a heading. When the tags are not exactly what was asked for,
        none of them can be trusted, and dropping the styling is the smaller
        loss.

        Args:
            source (str): The request as BabelDOC built it.
            translated (str): The model's output.

        Returns:
            str: The output, with style tags removed when they are unusable.
        """
        if not translated:
            return translated
        translated = repair_decimals(translated)
        # A placeholder the model tucked inside a style span is deleted by
        # BabelDOC's restoration, so it is lifted out before anything else looks
        # at the output — including the check below, which may keep the tags.
        # 被模型塞进样式段里的占位符会被 BabelDOC 的还原逻辑删掉，因此在其它任何一步
        # 读到这段输出之前先把它取出来——包括下面那道可能保留标记的检查。
        translated = hoist_placeholders(source, translated)
        translated = restore_symbol_spans(source, translated)
        if style_tags_intact(source, translated):
            return translated
        self.style_dropped_count.inc()
        logger.debug("Style tags came back malformed; dropping them")
        return strip_style_markers(translated)

    def _request(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        grammar: str | None = None,
    ) -> str:
        """
        中文：向 llama-server 发一次补全请求并累计 token 用量。

        Issue a single chat-completion request and accumulate token usage.

        Args:
            messages (list[dict[str, str]]): Chat messages from
                :meth:`build_messages`.
            temperature (float): Sampling temperature for this request. Passed
                per-request so the UI can change it without restarting the server.
            grammar (str | None): Optional GBNF grammar. llama.cpp enforces it as
                a per-token logit mask, which is how a marker can be made
                impossible to omit.

        Returns:
            str: Stripped assistant message content, or an empty string if the
                server returned no content.
        """
        s = self.settings
        response = self.client.chat.completions.create(
            model=s.model_file,
            messages=messages,
            temperature=temperature,
            top_p=s.top_p,
            max_tokens=s.max_tokens,
            # top_k and repeat_penalty are llama.cpp extensions rather than part
            # of the OpenAI schema, so they travel in extra_body.
            # top_k 与 repeat_penalty 是 llama.cpp 的扩展参数，不在 OpenAI 规范内。
            extra_body={
                "top_k": s.top_k,
                "repeat_penalty": s.repeat_penalty,
                **({"grammar": grammar} if grammar else {}),
            },
        )
        self._update_token_count(response)
        content = response.choices[0].message.content
        return content.strip() if content else ""

    def _update_token_count(self, response) -> None:
        """
        中文：从响应的 usage 字段累计 token 计数。

        Accumulate token counters from an OpenAI-compatible ``usage`` block.

        llama-server reports the same ``usage`` fields as the hosted API, so the
        local and API backends produce directly comparable statistics.

        Args:
            response: Chat completion response object.
        """
        try:
            usage = getattr(response, "usage", None)
            if not usage:
                return
            if usage.total_tokens:
                self.token_count.inc(usage.total_tokens)
            if usage.prompt_tokens:
                self.prompt_token_count.inc(usage.prompt_tokens)
            if usage.completion_tokens:
                self.completion_token_count.inc(usage.completion_tokens)
        except Exception:
            logger.debug("Could not read token usage from response", exc_info=True)

    # ------------------------------------------------------------------ #
    # Placeholder formats / 占位符格式
    # ------------------------------------------------------------------ #
    # These must match what BabelDOC expects when it restores formulas and rich
    # text after translation; the formats mirror OpenAITranslator's.
    # 必须与 BabelDOC 还原公式和富文本时的预期一致，格式沿用 OpenAITranslator。

    def get_formular_placeholder(self, placeholder_id: int | str):
        """
        中文：返回公式占位符及其匹配正则。

        Args:
            placeholder_id (int | str): Formula index assigned by BabelDOC.

        Returns:
            tuple[str, str]: ``(placeholder_text, regex_pattern)``.
        """
        return "{v" + str(placeholder_id) + "}", rf"{{\s*v\s*{placeholder_id}\s*}}"

    def get_rich_text_left_placeholder(self, placeholder_id: int | str):
        """
        中文：返回富文本左标记及其匹配正则。

        Args:
            placeholder_id (int | str): Style index assigned by BabelDOC.

        Returns:
            tuple[str, str]: ``(opening_tag, regex_pattern)``.
        """
        return (
            f"<style id='{placeholder_id}'>",
            rf"<\s*style\s*id\s*=\s*'\s*{placeholder_id}\s*'\s*>",
        )

    def get_rich_text_right_placeholder(self, placeholder_id: int | str):
        """
        中文：返回富文本右标记及其匹配正则。

        Args:
            placeholder_id (int | str): Style index assigned by BabelDOC.

        Returns:
            tuple[str, str]: ``(closing_tag, regex_pattern)``.
        """
        return "</style>", r"<\s*/\s*style\s*>"

    def usage_summary(self) -> dict[str, int]:
        """
        中文：汇总本次运行的 token 用量与缓存命中情况。

        Snapshot the counters accumulated over this translator's lifetime.

        Returns:
            dict[str, int]: Keys ``prompt_tokens``, ``completion_tokens``,
                ``total_tokens``, ``para_total`` (paragraphs requested),
                ``para_cache_hit`` (served from BabelDOC's cache),
                ``placeholder_retries``, ``placeholder_failures`` and
                ``implausible_output`` (outputs discarded as not being a
                translation of their input).
        """
        return {
            "prompt_tokens": self.prompt_token_count.value,
            "completion_tokens": self.completion_token_count.value,
            "total_tokens": self.token_count.value,
            "para_total": self.translate_call_count,
            "para_cache_hit": self.translate_cache_call_count,
            "placeholder_retries": self.placeholder_retry_count.value,
            "style_dropped": self.style_dropped_count.value,
            "placeholder_grammar": self.grammar_retry_count.value,
            "placeholder_reinserted": self.reinsert_count.value,
            "placeholder_failures": self.placeholder_failure_count.value,
            "implausible_output": self.implausible_count.value,
        }


# Reasoning models wrap their deliberation in <think>…</think> before the answer.
# 推理模型会把思考过程包在 <think>…</think> 里，正文在其后。
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_UNCLOSED_THINK = re.compile(r"^.*?</think>", re.DOTALL | re.IGNORECASE)


# The replacement character. It reaches this program inside the API's own
# response body — not from anything here: the completion is requested without
# streaming, so there is no chunk boundary on this side that could split a
# multi-byte sequence, and the body is decoded once, whole.
#
# Observed twice on the same phrase in two unrelated papers: 「本研究」 came back
# as 「本研�究」 from MiniMax-M2.5, with the replacement character sitting
# between two intact Han characters. That signature — both characters present,
# one extra U+FFFD wedged between them — is what a byte-fallback token decodes
# to when the model emits a lone continuation byte, and it is the same word each
# time because that is where the tokenisation happens to land.
#
# It has to be repaired rather than merely reported. Left alone it reaches the
# PDF, where the font mapper has nothing to draw it with, and the reader gets a
# black box in the middle of a word.
# 替换字符。它是随 API 响应体本身进来的，不是这里产生的：补全请求没有开流式，本侧不存在
# 能把多字节序列切开的分块边界，响应体是一次性整体解码的。
#
# 在两篇毫不相干的论文里、同一个词上观察到两次：MiniMax-M2.5 把「本研究」返回成
# 「本研�究」，替换字符夹在两个完好的汉字之间。这个特征——两个字都在，中间多出一个
# U+FFFD——正是模型吐出一个孤立的续接字节时，byte-fallback 词元解码出来的样子；每次都
# 落在同一个词上，是因为分词恰好断在那里。
#
# 必须修而不能只报。放着不管它会一路进到 PDF，字体映射器没有字形可画，读者就在一个词
# 中间看到一个黑块。
_REPLACEMENT = "\ufffd"


def repair_replacement(translated: str, source: str = "") -> str:
    """
    中文：去掉译文里凭空多出来的替换字符（U+FFFD）。

    Remove replacement characters the translation gained on its way back.

    Args:
        translated (str): The model's output.
        source (str): The paragraph that was sent. A source that already
            contained a replacement character is left alone — then the character
            is content, not corruption, and deleting it would be the bug.

    Returns:
        str: The output with spurious replacement characters removed, or
            unchanged when there were none or when the source had them too.
    """
    if _REPLACEMENT not in translated:
        return translated
    if _REPLACEMENT in source:
        return translated

    for index, char in enumerate(translated):
        if char == _REPLACEMENT:
            around = translated[max(0, index - 6):index + 7].replace("\n", " ")
            logger.warning("译文里出现替换字符（U+FFFD），已删除：…%s…", around)
            break

    # Between two spaces the character was standing in for a word, so removing
    # it alone would leave a double space where a reader sees a gap. Between two
    # Han characters — the observed case — there is nothing to collapse.
    # 夹在两个空格之间时，它顶替的是一个词，单删它会留下一个读者看得见的双空格。夹在
    # 两个汉字之间时——也就是实际观察到的情形——没有需要合并的东西。
    return (translated.replace(f" {_REPLACEMENT} ", " ")
                      .replace(_REPLACEMENT, ""))


def strip_reasoning(text: str) -> str:
    """
    中文：剥离推理模型输出里的 <think> 思考块，只保留正文。

    Remove a reasoning model's thinking block from its response.

    Needed because MiniMax-M2.5 emits ``<think>…</think>`` ahead of the answer
    and offers no way to turn it off — ``enable_thinking=False``,
    ``thinking={"type":"disabled"}`` and ``reasoning_effort="none"`` were all
    measured against the live endpoint and none suppressed it. Downstream code
    expects JSON or plain text, so the block is stripped here rather than each
    caller learning to cope with it.

    Args:
        text (str): Raw assistant message content.

    Returns:
        str: Content with any thinking block removed and surrounding whitespace
            trimmed. Text without a block is returned unchanged.
    """
    if not text:
        return ""
    if "</think>" not in text.lower():
        return text.strip()
    # A response truncated mid-answer can carry an opening tag without a close;
    # matching to the last close tag handles both shapes.
    # 输出被截断时可能只有开标签，按最后一个闭标签切分可同时覆盖两种形态。
    cleaned = _THINK_BLOCK.sub("", text)
    if "</think>" in cleaned.lower():
        cleaned = _UNCLOSED_THINK.sub("", cleaned)
    return cleaned.strip()


class TencentTranslator(LocalMTTranslator):
    """
    中文：走腾讯云机器翻译（TMT）的段落翻译器，其余守卫沿用本地翻译器。

    Paragraph translator backed by Tencent Cloud Machine Translation.

    Subclasses the local translator rather than the API one because what has to
    change is only the call, not the pipeline around it. A paragraph still goes
    through glossary protection, the formula-run merge, the placeholder check
    and its retry, and the fallback that keeps the English source rather than
    lose a formula. Two methods are replaced: :meth:`build_messages`, which has
    no chat to build, and :meth:`_request`, which posts to TMT instead of to
    llama-server.

    **The retry is weaker here than it looks.** When a placeholder goes missing
    the inherited guard retries at temperature zero with an emphatic
    instruction. TMT has neither a temperature nor an instruction to give, so
    the retry is the same request and will answer the same way. What survives
    is the fallback: the paragraph is left in English. That is the correct
    outcome and it is worth counting — see ``placeholder_failure_count``.

    Attributes:
        name (str): Cache key prefix. BabelDOC stores it in a 20-character
            column, so it must stay short.
    """

    name = "tencent-tmt"

    def __init__(
        self,
        lang_in: str,
        lang_out: str,
        secret_id: str,
        secret_key: str,
        region: str = "ap-guangzhou",
        project_id: int = 0,
        term_repos: list[str] | None = None,
        sent_repos: list[str] | None = None,
        ignore_cache: bool = False,
        glossary_rev: int = 1,
        settings=None,
    ):
        """
        中文：构造腾讯云翻译器。

        Build the Tencent translator.

        Args:
            lang_in (str): Source language code.
            lang_out (str): Target language code.
            secret_id (str): Tencent Cloud SecretId.
            secret_key (str): Tencent Cloud SecretKey.
            region (str): Region the requests are billed against.
            project_id (int): Tencent Cloud project id; ``0`` is the default.
            term_repos (list[str] | None): Custom term repository ids, applied
                by TMT itself. This is how a glossary reaches TMT: substituting
                markers into the text instead was measured on arXiv:2608.20061
                to fix eight terminology defects and cost 62% of the Chinese in
                long paragraphs, because a paragraph carrying many markers
                reverts to its English source as soon as TMT drops one.
            sent_repos (list[str] | None): Sentence repository ids.
            ignore_cache (bool): Skip BabelDOC's translation cache.
            glossary_rev (int): Glossary revision, part of the cache key so a
                changed glossary does not read stale translations.
            settings (LocalModelSettings | None): Only the fields the inherited
                guard reads are used; the endpoint fields are ignored because
                nothing here talks to llama-server.
        """
        from local_translator.config import LocalModelSettings

        super().__init__(
            lang_in=lang_in,
            lang_out=lang_out,
            settings=settings or LocalModelSettings(),
            ignore_cache=ignore_cache,
            glossary_rev=glossary_rev,
        )
        self._secret_id = secret_id
        self._secret_key = secret_key
        self._region = region
        self._project_id = project_id
        self._term_repos = list(term_repos or [])
        self._sent_repos = list(sent_repos or [])
        self._client = httpx.Client(timeout=30.0)
        # Atomic because BabelDOC calls the translator from several worker
        # threads at once.
        # 用原子计数：BabelDOC 会从多个工作线程同时调用翻译器。
        self.source_char_count = AtomicInteger()
        self.target_char_count = AtomicInteger()

    def build_messages(self, text: str, *, strict: bool = False) -> list[dict[str, str]]:
        """
        中文：TMT 没有对话可构造，把正文原样带过去。

        Carry the text through; there is no chat to build.

        Args:
            text (str): Paragraph text, placeholders and all.
            strict (bool): Ignored. The inherited guard sets it on a retry to
                ask the model to try harder, and TMT has no such dial.

        Returns:
            list[dict[str, str]]: A single user turn, which is the shape
                :meth:`_request` is handed.
        """
        return [{"role": "user", "content": text}]

    def _request(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        grammar: str | None = None,
    ) -> str:
        """
        中文：把这一段交给 TMT。

        Send one paragraph to TMT.

        Args:
            messages (list[dict[str, str]]): As built above; only the last
                turn's content is sent.
            temperature (float): Ignored — TMT does not sample.
            grammar (str | None): Ignored — TMT cannot be constrained.

        Returns:
            str: The translated paragraph, or an empty string when TMT returned
                nothing, which the inherited guard treats as a failed attempt.
        """
        from local_translator.engine import tencent

        text = messages[-1]["content"] if messages else ""
        if not text.strip():
            return ""
        # Counted here rather than inferred later, because this is the exact
        # string handed to the service and therefore the exact thing it bills
        # for. Retries count again, as they are charged again.
        # 在这里计数而不是事后推算，因为这正是交给服务的那个字符串，也正是它计费的对象。
        # 重试会再计一次，因为它确实会被再收一次费。
        self.source_char_count.inc(len(text))
        answer = tencent.translate_text(
            text,
            secret_id=self._secret_id,
            secret_key=self._secret_key,
            region=self._region,
            source=tencent._language(self.lang_in),
            target=tencent._language(self.lang_out),
            project_id=self._project_id,
            term_repos=self._term_repos,
            sent_repos=self._sent_repos,
            client=self._client,
        ).strip()
        self.target_char_count.inc(len(answer))
        return answer

    def usage_summary(self) -> dict:
        """
        中文：TMT 按字符计费、不返回 token 数，因此报告字符数。

        Report what TMT charges by, which is characters rather than tokens.

        TMT's response carries no usage figure, so these are counted locally as
        each request is sent. That is not the same as reading the figure off
        the bill, and the two can differ — a request that fails after being
        received is charged but may not be counted, and Tencent's own
        definition of a character need not match ``len()``. It is close enough
        to answer the question a reader actually has, which is whether this
        month's work is anywhere near the free allowance.

        Returns:
            dict: ``source_chars`` (characters sent, which is what the service
                bills for), ``target_chars`` (characters returned), and the
                paragraph and guard counters inherited from the local
                translator.
        """
        summary = super().usage_summary()
        summary.update({
            "source_chars": self.source_char_count.value,
            "target_chars": self.target_char_count.value,
            # TMT is not a token service; reporting zeros here keeps the usage
            # page from showing a token figure that has no meaning.
            # TMT 不是按 token 计费的服务；这里报 0，避免用量页显示一个没有意义的 token 数。
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        })
        return summary


class ApiTranslator(TranslationGuards, OpenAITranslator):
    """
    中文：云端 API 翻译器，保留 BabelDOC 原生 LLM 管线并处理推理模型的思考块。

    Cloud API translator.

    Subclasses BabelDOC's ``OpenAITranslator`` so API mode keeps the full
    LLM-only pipeline — batched paragraphs, glossary blocks and document context
    — which a capable hosted model handles well.

    It overrides ``do_llm_translate`` for two reasons specific to reasoning
    models, which many OpenAI-compatible endpoints now serve:

    - The base implementation returns the message content verbatim, so a
      ``<think>…</think>`` prefix reaches BabelDOC's JSON parser and breaks
      term extraction outright.
    - The base implementation caps output at 2048 tokens, which the thinking
      block alone can exhaust before the answer begins. Measured: a three-term
      extraction request produced nothing usable at 300 tokens and correct JSON
      at 4096.
    """

    # Generous enough that deliberation cannot crowd out the answer.
    # 预算足够宽，避免思考过程挤掉正文。
    max_output_tokens = 8192

    # Stop at the template's turn markers when driving raw completions.
    # 走原始补全时，在模板的轮次标记处停止。
    raw_stop = ["[e~[", "]~b]"]

    def __init__(self, *args, skip_reasoning: bool = False, prompt_template: str = "", **kwargs):
        """
        中文：初始化 API 翻译器，可选跳过推理模型的思考阶段。

        Args:
            skip_reasoning (bool): Drive the raw completions endpoint with a
                pre-closed think block instead of the chat endpoint.
            prompt_template (str): Raw prompt format with ``{system}`` and
                ``{user}`` placeholders. Required when ``skip_reasoning`` is on.
            *args: Forwarded to ``OpenAITranslator``.
            **kwargs: Forwarded to ``OpenAITranslator``.
        """
        super().__init__(*args, **kwargs)
        self.skip_reasoning = bool(skip_reasoning and prompt_template)
        self.prompt_template = prompt_template
        if self.skip_reasoning:
            self.add_cache_impact_parameters("skip_reasoning", True)

    def _raw_complete(self, system: str, user: str, max_tokens: int) -> str:
        """
        中文：走 /v1/completions，用预先闭合的 think 块跳过思考阶段。

        Complete a prompt through the raw endpoint with thinking short-circuited.

        MiniMax-M2.5's chat template appends a bare ``<think>`` to every
        generation prompt, which is what forces deliberation and why no request
        parameter can switch it off. Rendering the prompt here and closing that
        block immediately makes the model begin its answer instead.

        Args:
            system (str): System instruction.
            user (str): User message.
            max_tokens (int): Output budget.

        Returns:
            str: The completion text, stripped.
        """
        prompt = self.prompt_template.format(system=system, user=user)
        response = self.client.completions.create(
            model=self.model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=self.options.get("temperature", 0) if self.send_temperature else 0,
            stop=self.raw_stop,
        )
        self.update_token_count(response)
        return (response.choices[0].text or "").strip()

    def do_llm_translate(self, text, rate_limit_params: dict | None = None):
        """
        中文：调用 API 并剥离思考块后返回。

        Issue an LLM request and return its answer without the thinking block.

        Args:
            text (str): Fully composed prompt built by BabelDOC.
            rate_limit_params (dict | None): Supplied by BabelDOC; may request
                JSON mode.

        Returns:
            str | None: The answer with any reasoning block removed, or ``None``
                when BabelDOC probes capability by passing ``None``.
        """
        if text is None:
            return None

        if self.skip_reasoning:
            # BabelDOC composes one self-contained instruction, so it goes in as
            # the user turn with a minimal system line.
            # BabelDOC 已拼好完整指令，直接作为 user 轮，系统提示保持最小。
            return strip_reasoning(
                self._raw_complete(
                    "You are a professional, faithful translation engine.",
                    text,
                    self.max_output_tokens,
                )
            )

        options: dict = {}
        if self.send_temperature:
            options.update(self.options)
        if self.enable_json_mode_if_requested and (rate_limit_params or {}).get(
            "request_json_mode", False
        ):
            options["response_format"] = {"type": "json_object"}

        response = self.client.chat.completions.create(
            model=self.model,
            **options,
            max_tokens=self.max_output_tokens,
            messages=[{"role": "user", "content": text}],
            extra_body=self.extra_body,
        )
        self.update_token_count(response)
        return strip_reasoning(response.choices[0].message.content or "")

    def do_translate(self, text, rate_limit_params: dict | None = None) -> str:
        """
        中文：逐段翻译路径同样剥离思考块。

        Translate one paragraph, stripping any reasoning block.

        Args:
            text (str): Paragraph text.
            rate_limit_params (dict | None): Supplied by BabelDOC.

        Returns:
            str: The translation with any reasoning block removed.
        """
        if self.skip_reasoning:
            messages = self.prompt(text)
            system = next((m["content"] for m in messages if m["role"] == "system"), "")
            user = next((m["content"] for m in messages if m["role"] == "user"), text)
            return repair_replacement(
                strip_reasoning(self._raw_complete(system, user, 2048)), text)
        return repair_replacement(
            strip_reasoning(super().do_translate(text, rate_limit_params)), text)

    def usage_summary(self) -> dict[str, int]:
        """
        中文：以与本地翻译器相同的结构返回 token 用量。

        Snapshot token usage in the same shape as :meth:`LocalMTTranslator.usage_summary`.

        Returns:
            dict[str, int]: Same keys as the local translator; placeholder
                counters are always 0 because the LLM pipeline performs its own
                placeholder repair.
        """
        return {
            "prompt_tokens": self.prompt_token_count.value,
            "completion_tokens": self.completion_token_count.value,
            "total_tokens": self.token_count.value,
            "para_total": self.translate_call_count,
            "para_cache_hit": self.translate_cache_call_count,
            "placeholder_retries": 0,
            "placeholder_failures": 0,
        }
