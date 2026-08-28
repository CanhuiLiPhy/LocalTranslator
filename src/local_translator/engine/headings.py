"""
中文：论文里那些标准章节标题不必送去翻译——它们的译法是固定的。

Recognise the standard section headings and answer them from a table.

**Why they are worth taking out of the model's hands.** Measured over the
paragraph records of 33 translated papers: across *different* papers the
translation cache hits 2.68% of blocks by count and 0.07% by character, and of
the 4,229 blocks of real prose it hits exactly zero. Every one of those hits is
a heading — ``Abstract`` appears in 23 papers, ``References`` in 17,
``1 Introduction`` in 15. They are the only text a research paper shares with
another research paper.

**And the model does not render them consistently.** The same records show
``References`` coming back as both 「参考文献」 and 「参考资料」,
``1 Introduction`` as both 「1 引言」 and 「1 简介」, ``Author contributions``
as both 「作者贡献」 and 「作者的贡献」. A heading is a landmark: a reader who
scrolls back looking for 「引言」 should not have to consider that this paper
calls it 「简介」.

**Worse, some come back wrong.** With no sentence around it a bare word is
ambiguous, and the model resolves it by frequency rather than by context, which
in a paper is the wrong prior::

    Epochs   -> 「时代」   (should be 轮次)
    Final    -> 「决赛」   (should be 最终)
    property -> 「财产」   (should be 属性)
    Training -> 「培训」   (should be 训练)

Those are table headers read as everyday English.

**What this does.** Matches a block against the known headings — ignoring the
section number, the case, and the punctuation that separates them — and returns
the settled Chinese for it, keeping whatever numbering the source used. Nothing
is sent to the model, so nothing can come back different from last time.

The reader can switch it off; see ``PdfSettings.translate_headings``.
"""

from __future__ import annotations

import re

# The heading text, without number or case, mapped to the rendering to use.
# Chosen from what the model itself produced most often across the 33 papers,
# except where it was simply wrong.
# 去掉编号与大小写之后的标题原文，映射到固定译法。取值来自 33 篇论文里模型自己给出
# 得最多的那一种，除非那一种本身就是错的。
_HEADINGS: dict[str, str] = {
    # Front matter / 前置部分
    "abstract": "摘要",
    "contents": "目录",
    "table of contents": "目录",
    "keywords": "关键词",
    "index terms": "索引词",
    # Body / 正文
    "introduction": "引言",
    "background": "背景",
    "related work": "相关工作",
    "related works": "相关工作",
    "preliminaries": "预备知识",
    "method": "方法",
    "methods": "方法",
    "methodology": "方法",
    "experiments": "实验",
    "experiment": "实验",
    "experimental setup": "实验设置",
    "experimental results": "实验结果",
    "results": "结果",
    "results and discussion": "结果与讨论",
    "analysis": "分析",
    "discussion": "讨论",
    "conclusion and future work": "结论与展望",
    # Back matter / 后置部分
    "references": "参考文献",
    "bibliography": "参考文献",
    "acknowledgement": "致谢",
    "acknowledgements": "致谢",
    "acknowledgment": "致谢",
    "acknowledgments": "致谢",
    "appendix": "附录",
    "appendices": "附录",
    "author contributions": "作者贡献",
    "competing interests": "利益冲突声明",
    "conflict of interest": "利益冲突声明",
    "conflicts of interest": "利益冲突声明",
}

# The numbering that can precede a heading, and the punctuation that can follow
# the number. Deliberately permissive about the separator: a paper may write
# "3. Background", "3 Background", "3.1 Background", "III. Background" or
# "Background", and they are the same heading. The number itself is kept — it
# is a cross-reference target, not language.
# 标题前可能出现的编号，以及编号之后可能跟的标点。分隔符刻意放宽：同一个标题可能写成
# "3. Background"、"3 Background"、"3.1 Background"、"III. Background" 或
# "Background"，它们是同一个东西。编号本身保留——它是交叉引用的目标，不是语言。
_NUMBER = re.compile(
    r"""^\s*
    (?P<number>
        (?: \d+ (?:[.\-]\d+)* )        # 3      3.1     3-1
      | (?: [IVXLC]+ )                 # III
      | (?: [A-Z] )                    # A            (appendix style)
    )
    \s* [.)、．]? \s+
    """,
    re.VERBOSE,
)

# Trailing punctuation a heading may carry.
# 标题末尾可能带的标点。
_TRAILING = re.compile(r"[\s.:：、，,]+$")


def _normalise(text: str) -> tuple[str, str]:
    """
    中文：把标题拆成「编号」和「归一化后的标题文字」。

    Split a candidate heading into its number and its normalised text.

    Args:
        text (str): The block as it appears in the document.

    Returns:
        tuple[str, str]: The leading number exactly as written (empty when there
            is none, and including the separator and space that followed it),
            and the heading text lower-cased with surrounding punctuation and
            whitespace removed.
    """
    stripped = text.strip()
    match = _NUMBER.match(stripped)
    if match:
        number = stripped[:match.end()]
        rest = stripped[match.end():]
    else:
        number, rest = "", stripped
    rest = _TRAILING.sub("", rest)
    return number, " ".join(rest.lower().split())


def rendering(text: str) -> str | None:
    """
    中文：若这一块是标准章节标题，返回它的固定译法；否则返回 None。

    Return the settled rendering for a standard section heading.

    Args:
        text (str): One block of source text, as the segmenter produced it.

    Returns:
        str | None: The heading rendered in Chinese with the source's own
            numbering in front, or ``None`` when this block is not one of the
            known headings — in which case it goes to the model as usual.

    Examples:
        ``"3. Background"`` → ``"3. 背景"``;
        ``"3 Background"`` → ``"3 背景"``;
        ``"BACKGROUND"`` → ``"背景"``;
        ``"II. Related Work"`` → ``"II. 相关工作"``;
        ``"Background of the study"`` → ``None`` — not a bare heading.
    """
    if not text or len(text) > 60:
        return None
    number, key = _normalise(text)
    settled = _HEADINGS.get(key)
    if settled is None:
        return None
    return f"{number}{settled}" if number else settled


def is_heading(text: str) -> bool:
    """
    中文：这一块是不是标准章节标题。

    Whether a block is one of the standard section headings.

    Args:
        text (str): One block of source text.

    Returns:
        bool: ``True`` when :func:`rendering` would answer it.
    """
    return rendering(text) is not None


def known() -> list[str]:
    """
    中文：列出全部已收录的标题（归一化形式），供界面与测试使用。

    List every heading in the table, in normalised form.

    Returns:
        list[str]: The keys, sorted, so the count shown in the settings page and
            the count exercised by the tests come from the same place.
    """
    return sorted(_HEADINGS)

# Journal furniture — the same sentence in every paper from the same publisher.
# 期刊的固定件——同一家出版社的每篇论文里都是同一句话。
_HEADINGS.update({
    "author information": "作者信息",
    "open access": "开放获取",
})

# Blocks that are not language at all and should come back exactly as they went
# in. A table header like ``5-shot`` names an experimental condition; translated
# it becomes 「5 次射击」 or 「5-发」, and the column it labels stops matching the
# text that refers to it.
# 根本不是语言、应当原样返回的块。像 5-shot 这样的表头指的是实验条件；一旦翻译就成了
# 「5 次射击」之类，它标注的那一列便与正文里提到它的地方对不上了。
_VERBATIM = (
    re.compile(r"^\s*\d+[\s\-–]?shot\s*$", re.I),          # 0-shot, 5 shot
    re.compile(r"^\s*top[\s\-–]?\d+\s*$", re.I),            # top-1, top 5
    re.compile(r"^\s*[a-z]{1,4}@\d+\s*$", re.I),            # p@1, mAP@50
    re.compile(r"^\s*\d+[\s\-–]?(?:way|class|shot|gram|bit|B|M|K)\s*$"),  # 7B, 8-bit
)


def verbatim(text: str) -> bool:
    """
    中文：这一块是不是「根本不该翻译」的东西（如 5-shot、top-1、7B）。

    Whether a block should be returned exactly as it came.

    These are not headings with a settled rendering — they are labels that name
    an experimental condition or a model size, and any translation of them is
    wrong. A table column headed ``5-shot`` is referred to as ``5-shot`` in the
    prose, and the two must still match afterwards.

    Args:
        text (str): One block of source text.

    Returns:
        bool: ``True`` when the block should pass through untranslated.
    """
    if not text or len(text) > 24:
        return False
    return any(pattern.match(text) for pattern in _VERBATIM)
