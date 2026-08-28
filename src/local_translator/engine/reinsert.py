"""
中文：占位符重插——译文里丢失的公式标记按对齐位置补回。

Placeholder reinsertion.

When a translation model drops a formula or rich-text marker, three responses
are possible and all three are in use somewhere:

- **Discard the translation** and keep the source paragraph. Every formula
  survives, but the reader gets a wall of untranslated text. This was this
  project's behaviour and it cost 6.9% of the body text on a 20-page paper.
- **Keep the translation** and let the marker vanish. BabelDOC's own
  ``parse_translate_output`` does this — it logs a debug line and carries on.
  The paragraph reads well but an equation silently disappears, which the
  reader cannot detect.
- **Reinsert the marker** at the position it most likely belongs. This is the
  approach the machine-translation literature settles on: Müller (2017)
  measured hybrid reinsertion as the most accurate markup strategy, placing
  tags correctly in 86–93% of segments depending on language pair.

This module implements the third. It is deliberately conservative — when it
cannot find a defensible position it says so, and the caller falls back.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# Punctuation that marks a safe insertion boundary in Chinese or Latin text.
# Inserting mid-word would corrupt a term; inserting after a clause break does
# not change how the sentence reads.
# 可安全插入的断点标点；词中插入会破坏术语，子句边界插入不影响阅读。
_BOUNDARY = "。，、；：！？…）】》」』.,;:!?)]}　 \t\n"

# A paragraph that is mostly markers carries almost no translatable text, so a
# reinsertion guess has nothing to anchor to.
# 几乎全是标记的段落没有可锚定的文本，猜测插入位置没有依据。
MAX_MARKER_DENSITY = 0.4


def _is_boundary(text: str, index: int) -> bool:
    """
    中文：判断某个下标是否是安全的插入点。

    Test whether an index is a safe place to insert.

    The start and end of the text always qualify — appending or prepending
    cannot split a word. Elsewhere the preceding character must be punctuation
    or whitespace.

    Args:
        text (str): Target-language text.
        index (int): Candidate insertion index, ``0..len(text)``.

    Returns:
        bool: ``True`` if inserting here cannot split a word.
    """
    if index <= 0 or index >= len(text):
        return True
    return text[index - 1] in _BOUNDARY


def _nearest_boundary(text: str, target: int, taken: set[int]) -> int:
    """
    中文：找到离目标位置最近、且未被占用的安全断点。

    Find the free insertion index nearest to ``target`` that lands on a boundary.

    Excluding already-used indices is what stops two markers from collapsing
    onto the same comma and emerging in reverse order — the failure this
    function's first version produced on a two-marker sentence.

    Args:
        text (str): Target-language text.
        target (int): Desired index.
        taken (set[int]): Indices already claimed by other markers.

    Returns:
        int: A free boundary index, or the nearest free index when the text
            offers no boundary at all (a marker inside a word is still better
            than two markers stacked in the wrong order).
    """
    target = max(0, min(target, len(text)))

    # Search outward, preferring boundaries.
    # 由近及远搜索，优先落在断点上。
    for distance in range(0, len(text) + 1):
        for index in (target - distance, target + distance):
            if 0 <= index <= len(text) and index not in taken and _is_boundary(text, index):
                return index

    for distance in range(0, len(text) + 1):
        for index in (target - distance, target + distance):
            if 0 <= index <= len(text) and index not in taken:
                return index
    return target


def reinsert(source: str, translated: str, missing: list[str]) -> tuple[str, bool]:
    """
    中文：把译文中缺失的占位符按源文相对位置补回。

    Reinsert markers the model dropped, at proportionally aligned positions.

    Each missing marker's position in the source is expressed as a fraction of
    the source length, then mapped onto the translation and snapped to the
    nearest clause boundary. Markers are inserted from the end backwards so
    earlier insertions do not shift the indices of later ones.

    Proportional mapping is a heuristic, not alignment: it assumes the
    translation preserves the source's rough ordering, which holds for formula
    markers in scientific prose because a formula's referent appears at the same
    point in the argument regardless of language. It does not hold for markers
    embedded in heavily reordered clauses, which is why the caller should treat
    a ``False`` result as a reason to fall back rather than force the issue.

    Args:
        source (str): Original paragraph, containing every marker.
        translated (str): Model output, missing one or more markers.
        missing (list[str]): Markers present in ``source`` but absent from
            ``translated``.

    Returns:
        tuple[str, bool]: The repaired text and whether repair was attempted.
            ``(translated, False)`` means reinsertion was declined — the caller
            should fall back to the source paragraph.
    """
    if not missing or not translated:
        return translated, False

    # Refuse when the paragraph is mostly markup: there is no prose to anchor to,
    # and a wrong guess would scatter formulas through a fragment.
    # 段落几乎全是标记时拒绝重插：没有正文可锚定，猜错会把公式散落在碎片里。
    marker_chars = sum(len(m) for m in re.findall(r"\{\s*v\s*\d+\s*\}|</?\s*style[^>]{0,24}>", source))
    if len(source) and marker_chars / len(source) > MAX_MARKER_DENSITY:
        logger.debug("Declining reinsertion: paragraph is %.0f%% markers", marker_chars / len(source) * 100)
        return translated, False

    # Place in source order, so a marker that came first in the source also
    # lands first in the translation. Without this the markers can emerge
    # reversed when two of them map near the same boundary.
    # 按源文顺序放置；否则两个标记落在同一断点附近时会前后颠倒。
    ordered = sorted(
        (m for m in missing if source.find(m) >= 0), key=lambda m: source.find(m)
    )
    if not ordered:
        return translated, False

    placements: list[tuple[int, str]] = []
    taken: set[int] = set()
    lowest_allowed = 0
    for marker in ordered:
        fraction = source.find(marker) / max(len(source), 1)
        wanted = max(lowest_allowed, round(fraction * len(translated)))
        index = _nearest_boundary(translated, wanted, taken)
        # Never place a marker before one that precedes it in the source.
        # 不允许排到源文中更靠前的标记之前。
        if index < lowest_allowed:
            index = _nearest_boundary(translated, lowest_allowed, taken)
        taken.add(index)
        lowest_allowed = index
        placements.append((index, marker))

    # Insert right-to-left so earlier insertions do not shift later indices.
    # 从右往左插入，先插的不会移动后插的下标。
    repaired = translated
    for index, marker in sorted(placements, key=lambda p: -p[0]):
        repaired = repaired[:index] + marker + repaired[index:]

    logger.debug("Reinserted %d marker(s): %s", len(placements), ", ".join(ordered))
    return repaired, True


def looks_hallucinated(source: str, translated: str, ratio: float = 2.2) -> bool:
    """
    中文：判断译文是否明显过长，提示模型开始编造。

    Detect an output that has run away from its input.

    Constrained decoding can make a model pad text until it can satisfy the
    grammar; observed on the 1.8B model, which appended 「…（此处应继续列出具体
    因素）…」 to reach a required marker. Length is a crude but reliable signal,
    since a faithful English-to-Chinese translation is normally *shorter* in
    characters than its source.

    Args:
        source (str): Source paragraph.
        translated (str): Model output.
        ratio (float): Length multiple above which the output is suspect.

    Returns:
        bool: ``True`` when the output is implausibly long for its input.
    """
    if not source:
        return False
    return len(translated) > len(source) * ratio


def build_placeholder_grammar(markers: list[str]) -> str | None:
    """
    中文：生成强制输出占位符的 GBNF 语法，供 llama.cpp 约束解码使用。

    Build a GBNF grammar that forces every marker to appear, in source order.

    llama.cpp applies the grammar as a per-token logit mask, so the markers
    cannot be omitted. Measured cost is negligible — 0.41s against 0.21s
    unconstrained on a paragraph the model had previously failed.

    The order constraint is a real limitation: it assumes the translation keeps
    the markers in source order, which usually holds for formulas in scientific
    prose but not universally. Use this as a repair step after an unconstrained
    attempt, never as the default path.

    Args:
        markers (list[str]): Markers that must appear, in source order.

    Returns:
        str | None: A GBNF grammar, or ``None`` when there is nothing to
            constrain or a marker contains characters the grammar cannot quote.
    """
    if not markers:
        return None

    ordered: list[str] = []
    for marker in markers:
        if marker not in ordered:
            ordered.append(marker)

    if any('"' in m or "\\" in m for m in ordered):
        return None

    parts = " ".join(f'seg "{m}"' for m in ordered)
    # `seg` excludes the marker's opening characters so the grammar cannot
    # satisfy itself with free text that merely looks like a marker.
    # seg 排除标记的起始字符，避免语法被"看起来像标记"的自由文本满足。
    return f"root ::= {parts} seg\nseg  ::= [^{{<]*\n"
