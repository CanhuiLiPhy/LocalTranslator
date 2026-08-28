r"""
中文：把拆出来的片段送去翻译，并逐条检查译文没有破坏 LaTeX。

Translate the pieces and check that each came back as usable LaTeX.

The translator is this project's own — local, cloud or hybrid, with the
glossary and the term protections already attached — so a LaTeX run inherits
everything the PDF pipeline has. What this module adds is the part specific to
source: a model asked to translate ``\\textbf{Results} on \\ref{tab:main}``
must return the commands intact, and when it does not, the paragraph has to be
put back rather than shipped broken.

**Why every translation is checked rather than trusted.** A dropped brace turns
into a compile error hundreds of lines away, and a dropped ``\\ref`` turns into
a paper that references nothing. Both are far more expensive to find in the
output than to catch here, where the original is still at hand.
"""

from __future__ import annotations

import concurrent.futures
import logging
import time

import regex

from local_translator.engine.translators import protect_keeps, restore_keeps
from local_translator.latex.parser import LatexDocument, Piece
from local_translator.latex.prompting import (
    about,
    insist,
    is_heading,
    use_latex_prompt,
)

logger = logging.getLogger(__name__)

# The placeholder the parser leaves behind. Every one present in a piece must
# come back, or the block it stands for is lost.
# 解析器留下的占位符。片段里出现的每一个都必须回来，否则它代表的那一块就没了。
_MARK = regex.compile(r"<LTX\d+>")

# A LaTeX control sequence. Counted before and after so a translation that
# swallowed one is rejected.
# LaTeX 控制序列。译前译后各数一次，吞掉了就拒收。
_COMMAND = regex.compile(r"\\[a-zA-Z@]+\*?")

# Commands whose loss silently changes the paper rather than breaking it. A
# missing \ref leaves "see " dangling; a missing \cite loses an attribution.
# These are checked by name, not just by count.
# 丢了不会报错、却会悄悄改变论文的命令。少一个 \ref，正文就剩「见 」；少一个 \cite，
# 就少了一处出处。这些按名字核对，不只数个数。
_CRITICAL = ("ref", "eqref", "cite", "citep", "citet", "label", "autoref",
             "includegraphics", "url", "href")

# Formatting a translator may reasonably add: Chinese emphasis often wants a
# bold run where English used italics, or none at all. These are real commands
# that any document can use, so adding one cannot break a compile — unlike
# \textemph or \stylesheet, which the model also invented and which exist
# nowhere.
# 译者可以合理添加的格式命令：中文的强调常常需要一处加粗，而英文原文可能用的是斜体或
# 根本没有强调。这些都是任何文档都能用的真实命令，加上它们不会让编译失败——不像模型
# 同样造出来的 \textemph 和 \stylesheet，那两个哪里都不存在。
_MAY_ADD = frozenset({
    "textbf", "textit", "emph", "texttt", "textrm", "textsf", "underline",
    "textsc", "textsl", "quad", "qquad", ",", ";", ":", "!", "%", "&", "_",
    "#", "$", "{", "}", "\\",
})


# Environments that put their contents in math mode without any dollar signs.
# A translation inside one of these may use _ and ^ freely; outside, it may not.
# 不用美元符号就进入数学模式的环境。译文在这些环境内部可以随便用 _ 和 ^，在外面不行。
_MATH_ENVIRONMENTS = frozenset({
    "equation", "equation*", "align", "align*", "alignat", "alignat*",
    "gather", "gather*", "flalign", "flalign*", "multline", "multline*",
    "split", "cases", "array", "eqnarray", "eqnarray*", "math", "displaymath",
    "subequations", "IEEEeqnarray", "IEEEeqnarray*",
})

_BEGIN_END = regex.compile(r"\\(begin|end)\s*\{([^}]*)\}")


def math_mode_spans(text: str) -> list[tuple[int, int]]:
    r"""
    中文：找出文本里处于数学模式的区间。

    Find the stretches of a document that are in math mode.

    Args:
        text (str): LaTeX source.

    Returns:
        list[tuple[int, int]]: Half-open ``(start, end)`` ranges. Covers ``$``,
            ``$$``, ``\(…\)``, ``\[…\]`` and the environments that are
            mathematical without any delimiter.

    Note:
        An escaped ``\$`` is a dollar sign in the text and does not open math
        mode. Missing that is how a price list becomes a formula.
        转义的 \$ 是正文里的美元符号，不进入数学模式。漏掉这一点，一份价目表就会变成公式。
    """
    spans: list[tuple[int, int]] = []
    index, length = 0, len(text)
    dollar_start = None
    while index < length:
        character = text[index]
        if character == "\\" and index + 1 < length:
            following = text[index + 1]
            if following in "()[]":
                closer = {"(": "\\)", "[": "\\]"}.get(following)
                if closer:
                    end = text.find(closer, index + 2)
                    end = length if end < 0 else end + 2
                    spans.append((index, end))
                    index = end
                    continue
            index += 2
            continue
        if character == "$":
            width = 2 if text.startswith("$$", index) else 1
            if dollar_start is None:
                dollar_start = index
            else:
                spans.append((dollar_start, index + width))
                dollar_start = None
            index += width
            continue
        index += 1
    if dollar_start is not None:
        spans.append((dollar_start, length))

    for match in _BEGIN_END.finditer(text):
        if match.group(1) != "begin" or match.group(2) not in _MATH_ENVIRONMENTS:
            continue
        closer = regex.search(rf"\\end\s*\{{{regex.escape(match.group(2))}\}}",
                              text[match.end():])
        end = match.end() + closer.end() if closer else length
        spans.append((match.start(), end))
    return spans


# Commands whose braced argument is a name, a key or a path rather than prose.
# An underscore in there belongs to the identifier — escaping it turns
# \cref{sec:alg_prefTree} into a reference to a label that does not exist, and
# TeX reports it as "Missing \endcsname inserted" on a line that looks fine.
# 花括号里是名字、键或路径而不是散文的命令。那里的下划线属于标识符——转义它会把
# \cref{sec:alg_prefTree} 变成指向一个不存在的标签，而 TeX 会在一行看起来没问题的
# 地方报「Missing \endcsname inserted」。
_IDENTIFIER_COMMANDS = (
    "label", "ref", "cref", "Cref", "crefrange", "eqref", "autoref", "pageref",
    "nameref", "vref", "cite", "citep", "citet", "citeauthor", "citeyear",
    "citealp", "citealt", "citenum", "nocite", "bibliography",
    "bibliographystyle", "includegraphics", "input", "include", "url", "href",
    "path", "verb", "usepackage", "documentclass", "hypersetup", "addbibresource",
)

_IDENTIFIER_PATTERN = regex.compile(
    r"\\(?:" + "|".join(_IDENTIFIER_COMMANDS) + r")\*?\s*(?:\[[^\]]*\])?\s*\{",
)


def identifier_spans(text: str) -> list[tuple[int, int]]:
    """
    中文：找出「参数是标识符而非文字」的那些命令的参数区间。

    Find the argument of every command whose argument is an identifier.

    Args:
        text (str): LaTeX source.

    Returns:
        list[tuple[int, int]]: Half-open ranges covering the braced arguments,
            found by counting braces so a nested group does not end the span
            early.
    """
    spans: list[tuple[int, int]] = []
    for match in _IDENTIFIER_PATTERN.finditer(text):
        depth, index = 1, match.end()
        while index < len(text) and depth:
            character = text[index]
            if character == "\\":
                index += 2
                continue
            if character == "{":
                depth += 1
            elif character == "}":
                depth -= 1
            index += 1
        spans.append((match.end(), index))
    return spans


def _in_math(index: int, spans: list[tuple[int, int]]) -> bool:
    """中文：这个位置在不在数学模式里。 / Whether this position is in math mode."""
    return any(start <= index < end for start, end in spans)


# A token that only makes sense in math mode: something, then ^ or _, then
# either a braced group or a single character. Used to find the ones the model
# left outside math so they can be put back inside it.
# 只有在数学模式里才讲得通的记号：一段东西，然后 ^ 或 _，然后一个花括号组或单个字符。
# 用来找出被模型留在数学模式之外的那些，好把它们放回去。
# A caret is math whichever form follows it — a circumflex in running prose is
# essentially never what was meant. An underscore is only math when a braced
# group follows: bare `L0_Mx` is an identifier, and wrapping it in dollars
# splits it into `$L0_M$x`, which is worse than the escaping it needs instead.
# 尖号无论后面跟什么都是数学——正文里的音调符号基本上不会是本意。下划线只有在后面跟
# 花括号组时才是数学：裸的 L0_Mx 是标识符，包进美元符号会被切成 $L0_M$x，比它本该
# 得到的转义更糟。
# Commands that only exist in math mode. A translation that uses one outside
# math compiles no further: TeX says "\mathrm allowed only in math mode" and
# stops. Not an exhaustive list of every math command — an exhaustive list is
# not maintainable — but the ones a translator reaches for when it rewrites a
# sentence that contained a formula.
# 只存在于数学模式里的命令。译文在数学模式之外用了它们，编译就到此为止：TeX 说
# 「\mathrm allowed only in math mode」然后停下。这不是所有数学命令的完全清单——完全
# 清单维护不了——而是模型在改写一句含公式的话时会伸手去拿的那些。
_MATH_ONLY = (
    "mathrm|mathbf|mathcal|mathbb|mathit|mathsf|mathtt|mathfrak|boldsymbol"
    "|frac|dfrac|tfrac|sqrt|sum|prod|int|oint|lim|inf|sup|partial|nabla"
    "|alpha|beta|gamma|delta|epsilon|varepsilon|zeta|eta|theta|vartheta|iota"
    "|kappa|lambda|mu|nu|xi|pi|rho|sigma|tau|upsilon|phi|varphi|chi|psi|omega"
    "|Gamma|Delta|Theta|Lambda|Xi|Pi|Sigma|Upsilon|Phi|Psi|Omega"
    "|times|cdot|leq|geq|neq|approx|equiv|propto|sim|pm|mp|infty"
    "|rightarrow|leftarrow|Rightarrow|Leftarrow|leftrightarrow|to|mapsto"
    # `\underline` is not one of these: LaTeX defines it for text mode as well,
    # so a table cell may hold `\underline{0.614}` with no mathematics in sight
    # and wrapping it in dollars changes a number into a formula.
    # \underline 不在此列：LaTeX 在文本模式下也定义了它，所以一个表格单元格里可以就是
    # \underline{0.614}、周围没有任何数学，把它包进美元符号会把一个数字变成公式。
    "|hat|tilde|bar|vec|dot|ddot|overline|text"
)

# A math command with its braced arguments — but never a placeholder as one.
# ``{v7}`` has the shape of a braced argument, and it stood for the ``&`` between
# two table cells: the repair wrapped ``\underline{0.614}`` *and the cell
# separator after it* in dollars, and TeX stopped at "Extra }, or forgotten $".
# A placeholder is never an argument; it is a piece of the document standing
# aside.
# 带花括号参数的数学命令——但绝不把占位符当成参数。{v7} 长得就像一个花括号参数，而它
# 代表的是两个表格单元格之间的 &：那次修复把 \underline{0.614} **连同后面的列分隔符**
# 一起包进了美元符号，TeX 停在「Extra }, or forgotten $」。占位符从来不是参数，它是文档
# 里暂时站到一边去的一块东西。
_MATH_COMMAND = regex.compile(
    rf"\\(?:{_MATH_ONLY})\b(?:\s*\{{(?![vk]\d+\}})[^{{}}]*\}})*")

_MATH_TOKEN = regex.compile(
    r"(?:\\[a-zA-Z]+|[A-Za-z0-9\u0370-\u03ff]+)"
    r"(?:\^(?:\{[^{}]*\}|[A-Za-z0-9])|_\{[^{}]*\})+")


def restore_math_mode(text: str) -> tuple[str, int]:
    r"""
    中文：把落在数学模式之外的上下标重新包进 $…$。

    Put a superscript that ended up outside math mode back inside it.

    Args:
        text (str): A translation.

    Returns:
        tuple[str, int]: The repaired text and how many tokens were wrapped.

    Note:
        Measured on the PRL paper: the source reads ``$	au^{-2}$`` and the
        translation came back as ``τ^{-2}`` — the Greek letter spelled out and
        the dollars gone. Escaping the caret would compile but mean something
        else, a circumflex accent; the delimiters are what went missing, so
        the delimiters are what is restored.
        在 PRL 那篇上实测：原文是 $	au^{-2}$，译文回来是 τ^{-2}——希腊字母写成了字面，
        美元符号没了。给尖号加转义能编过，但意思就变了，成了个音调符号；丢的是定界符，
        补的就该是定界符。
    """
    spans = math_mode_spans(text) + identifier_spans(text)
    found = sorted(
        [m for m in _MATH_TOKEN.finditer(text)] + [m for m in _MATH_COMMAND.finditer(text)],
        key=lambda m: (m.start(), -m.end()))

    pieces, index, wrapped = [], 0, 0
    for match in found:
        if match.start() < index:
            continue  # 已被前一个匹配吃掉。 / Already inside a previous match.
        if _in_math(match.start(), spans) or _in_math(match.end() - 1, spans):
            continue
        pieces.append(text[index:match.start()])
        pieces.append("$" + match.group(0) + "$")
        index = match.end()
        wrapped += 1
    pieces.append(text[index:])
    return "".join(pieces), wrapped


# A control word run straight into a Chinese character. TeX ends a control word
# at the first non-letter — but xeCJK makes CJK characters *letters*, so
# `\model与现有的` parses as one control sequence named `model与现有的`, which is
# undefined. In the English source the macro was followed by a space or a
# brace; the translation put a character there instead.
# 控制字后面直接接汉字。TeX 在第一个非字母处结束控制字——但 xeCJK 把汉字设成了**字母**，
# 于是 \model与现有的 会被解析成一个名为 model与现有的 的控制序列，而它没有定义。英文
# 原文里宏后面跟的是空格或花括号，译文在那里换成了一个字。
_RUN_ON_MACRO = regex.compile(r"(\\[a-zA-Z]+)(?=[\u3000-\u303f\u4e00-\u9fff\uff01-\uff65])")


def separate_macros_from_chinese(text: str) -> tuple[str, int]:
    r"""
    中文：在控制字与紧随其后的汉字之间插入 {}，否则宏名会把汉字吃进去。

    Put ``{}`` between a control word and the Chinese character after it.

    Args:
        text (str): A translation.

    Returns:
        tuple[str, int]: The repaired text and how many macros were separated.

    Note:
        Measured on the LLaMA paper: ``\model与现有的大型语言模型`` — the macro
        ``\model`` is defined on line 76 and used on line 363, and TeX still
        calls it undefined, because under xeCJK the name it actually read was
        ``model与现有的大型语言模型``.
        在 LLaMA 那篇上实测：\model与现有的大型语言模型——宏 \model 在第 76 行定义、
        第 363 行使用，TeX 依然说它未定义，因为在 xeCJK 之下它真正读到的名字是
        model与现有的大型语言模型。
    """
    return _RUN_ON_MACRO.subn(r"\1{}", text)


# Two explicit line breaks with nothing between them but braces and space. The
# first one ends a line that has no content, and LaTeX says so: "There's no line
# here to end."
# 两个显式换行之间除了花括号和空白什么都没有。第一个换行要结束的那一行没有内容，LaTeX
# 就是这么说的：「There's no line here to end.」
_DOUBLE_BREAK = regex.compile(r"\\\\(?:\[[^\]]*\])?[ \t]*(\}*)[ \t\r\n]*(?=\\\\)")


def collapse_double_breaks(text: str) -> tuple[str, int]:
    r"""
    中文：去掉紧挨着另一个换行的那个多余换行。

    Drop a line break that has another one right behind it.

    Args:
        text (str): A translation with its protected commands restored.

    Returns:
        tuple[str, int]: The repaired text and how many breaks were dropped.

    Note:
        Measured on the PRL paper. The source reads
        ``\textbf{\large Supplemental Material for\\[0.5em] ``…title…''}\\[1em]``
        — a break in the middle of the heading. Chinese puts 「的补充材料」 last, so
        the model moved the break to the end of the group with it, leaving
        ``…补充材料\\[0.5em]}\\[1em]``: two breaks in a row, the first with an
        empty line to end. The reordering is right for the language; only the
        now-redundant break has to go.
        在 PRL 那篇上实测。原文是 \textbf{\large Supplemental Material for\\[0.5em]
        ``…标题…''}\\[1em]——换行在标题中间。中文把「的补充材料」放在最后，模型就把换行
        跟着挪到了整组末尾，于是成了 …补充材料\\[0.5em]}\\[1em]：两个换行连在一起，第一
        个要结束的是一行空的。语序上模型是对的；该去掉的只是那个已经多余的换行。
    """
    return _DOUBLE_BREAK.subn(r"\1", text)


# A blank line, which is how LaTeX writes \par.
# 空行，也就是 LaTeX 写 \par 的方式。
_BLANK_LINE = regex.compile(r"\n[ \t]*(?:\n[ \t]*)+")


def match_paragraph_breaks(source: str, translated: str) -> tuple[str, int]:
    r"""
    中文：原文没有空行的地方，译文也不许有——空行在命令参数里就是错误。

    Remove paragraph breaks the translation invented.

    Args:
        source (str): The passage that was sent.
        translated (str): What came back.

    Returns:
        tuple[str, int]: The repaired text and how many breaks were removed.
            When the source has a blank line of its own the translation is left
            alone: a body with several paragraphs is entitled to them.

    Note:
        A blank line is ``\par``, and ``\par`` inside a command's argument is an
        error — TeX says "Paragraph ended before \@sect was complete", or
        ``\NR@gettitle``, or whichever piece of machinery was mid-argument, and
        names a line some distance from the blank one. Measured on
        arXiv:2608.22244: the translation of a two-sentence ``\caption`` put a
        blank line between the sentences, and the paper stopped compiling.
        空行就是 \par，而 \par 出现在命令参数里是错误——TeX 会说「Paragraph ended before
        \@sect was complete」，或者 \NR@gettitle，或者当时正读到一半的随便哪个内部宏，并
        且报出的行号离那个空行还有一段距离。在 arXiv:2608.22244 上实测：一条两句话的
        \caption，译文在两句之间加了一个空行，整篇论文就编不过了。
    """
    if _BLANK_LINE.search(source):
        return translated, 0
    return _BLANK_LINE.subn("\n", translated)


def restore_command_case(source: str, translated: str) -> tuple[str, int]:
    r"""
    中文：把译文里被改了大小写的命令名改回原样。

    Put back the original spelling of a command the model recased.

    Args:
        source (str): The passage that was sent.
        translated (str): What came back.

    Returns:
        tuple[str, int]: The repaired text and how many names were restored.

    Note:
        A model that reads ``\IEEEPARstart`` writes ``\IEEEParstart``, because
        that is how the name would be capitalised if it were English. TeX calls
        the result undefined, and the check here calls it an invented command —
        both true, and both a long way from "one letter changed case". The
        repair is exact: a name is only rewritten when the source holds one that
        differs from it in case alone.
        模型读到 \IEEEPARstart，写出来是 \IEEEParstart——因为按英文的写法就该这么大小写。
        TeX 说这个命令未定义，这里的检查说它是凭空造的命令——两句都没错，但离「有个字母
        换了大小写」都很远。这个修复很精确：只有当原文里存在一个仅大小写不同的名字时，
        才会改写。
    """
    known = {name.lower(): name for name in _COMMAND.findall(source)}
    if not known:
        return translated, 0

    def repair(match: regex.Match) -> str:
        """中文：换回原文的写法。 / Spell it the way the source did."""
        found = known.get(match.group(0).lower())
        return found if found and found != match.group(0) else match.group(0)

    repaired, _ = _COMMAND.subn(repair, translated)
    changed = sum(1 for before, after in zip(_COMMAND.findall(translated),
                                             _COMMAND.findall(repaired), strict=True)
                  if before != after)
    return repaired, changed


# Environments where & is a column separator and must not be escaped. Outside
# them a bare & is "Misplaced alignment tab character", which is how the Gemini
# report failed: the translation of a sentence naming a team wrote an ampersand
# the source had escaped.
# & 在其中是列分隔符、不能转义的环境。在这些之外，裸的 & 就是「Misplaced alignment tab
# character」，Gemini 那篇正是这样失败的：一句提到某个团队的译文写了个 & ，而原文里
# 那个是转义过的。
_TABULAR_ENVIRONMENTS = frozenset({
    "tabular", "tabular*", "tabularx", "tabulary", "longtable", "longtable*",
    "array", "align", "align*", "alignat", "alignat*", "flalign", "flalign*",
    "eqnarray", "eqnarray*", "matrix", "pmatrix", "bmatrix", "vmatrix",
    "cases", "split", "supertabular", "threeparttable",
})


def _tabular_spans(text: str) -> list[tuple[int, int]]:
    """中文：找出表格类环境的范围。 / Find the stretches inside tabular environments."""
    spans: list[tuple[int, int]] = []
    for match in _BEGIN_END.finditer(text):
        if match.group(1) != "begin" or match.group(2) not in _TABULAR_ENVIRONMENTS:
            continue
        closer = regex.search(rf"\\end\s*\{{{regex.escape(match.group(2))}\}}",
                              text[match.end():])
        spans.append((match.start(),
                      match.end() + closer.end() if closer else len(text)))
    return spans


def escape_stray_specials(text: str) -> tuple[str, int]:
    r"""
    中文：把正文里没转义的 _、#、& 和 % 补上反斜杠。

    Escape ``_``, ``#``, ``&`` and ``%`` that ended up in text mode unescaped.

    ``%`` is the costly one. It starts a comment, so it does not fail — it
    silently eats the rest of the line. ``超过90%的选择准确率`` swallowed the
    ``\end{itemize}`` that shared its line, and TeX reported it nine hundred
    lines later as ``\begin{itemize} on input line 102 ended by
    \end{document}``; the same character inside a caption ate the closing brace
    and produced ``File ended while scanning use of \caption@xdblarg``. Two of
    five papers failed to compile on this one character.

    Any ``%`` here is a per-cent sign: real comments were stripped when the
    document was parsed, long before the model saw anything.
    ``%`` 是代价最大的一个。它开启注释，所以不会报错——它会安静地吃掉本行剩下的内容。
    ``超过90%的选择准确率`` 吃掉了与它同行的 ``\end{itemize}``，TeX 在九百行之后才报
    ``\begin{itemize} on input line 102 ended by \end{document}``；同一个字符出现在图注
    里则吃掉了右花括号，报 ``File ended while scanning use of \caption@xdblarg``。五篇
    里有两篇栽在这一个字符上。

    这里出现的任何 ``%`` 都是百分号：真正的注释在解析文档时就已剥掉，远在模型看到任何
    东西之前。

    Args:
        text (str): A translation.

    Returns:
        tuple[str, int]: The repaired text and how many characters were fixed.

    Note:
        This is the single most common way a translation fails to compile.
        Asked to translate a sentence naming ``L0\_Mx``, the model returns
        ``L0_Mx`` — correct as prose, and ``Missing $ inserted`` to TeX, which
        reports it hundreds of lines from where the paragraph actually is.
        Repairing beats refusing: the translation is right, only its escaping
        is wrong. Math mode is left alone, and so is the argument of a command
        whose argument is an identifier — escaping the underscore in
        ``\cref{sec:alg_prefTree}`` breaks the reference instead of fixing it.
        这是译文编译不过最常见的一种。让模型翻译一句提到 L0\_Mx 的话，它会返回
        L0_Mx——作为文字没错，对 TeX 却是 Missing $ inserted，而且报错位置离这一段
        有几百行。修复优于拒收：译文是对的，只是转义没做。
    """
    spans = math_mode_spans(text) + identifier_spans(text) + _tabular_spans(text)
    out, fixed = [], 0
    for index, character in enumerate(text):
        if character in "_#&%" and not _in_math(index, spans):
            if index and text[index - 1] == "\\":
                out.append(character)
                continue
            out.append("\\" + character)
            fixed += 1
        else:
            out.append(character)
    return "".join(out), fixed


# Inline commands whose argument is a key, not prose. They sit in the middle of
# a sentence, so they cannot be peeled to the edges like a section heading — and
# left in place the model rewrites them. Measured on the LLaMA paper:
# \citet{hendrycks2020measuring} came back as \handrycks2020measuring, the
# command and its key fused into a macro that exists nowhere. Citations alone
# cost 5% of that paper's body.
#
# \href is excluded: its second argument is prose a reader sees and should be
# translated, so protecting the whole thing would leave link text in English.
# 参数是键而不是散文的行内命令。它们待在句子中间，没法像小节标题那样挪到边上——而留在
# 原地，模型就会改写它们。在 LLaMA 那篇上实测：\citet{hendrycks2020measuring} 回来成了
# \handrycks2020measuring，命令和它的键熔成了一个根本不存在的宏。光引用一项就吃掉了
# 那篇正文的 5%。
#
# 排除 \href：它的第二个参数是读者能看到的散文、应当翻译，整个保护起来会让链接文字
# 留在英文。
_INLINE_KEYED = regex.compile(
    r"\\(?:cite[a-zA-Z]*|ref|cref|Cref|crefrange|eqref|autoref|pageref|nameref"
    r"|vref|label|url|path|includegraphics|input|include|bibliography)"
    r"\*?\s*(?:\[[^\]]*\])?\s*\{[^{}]*\}"
    # An explicit line break, which is a backslash followed by a backslash and
    # nothing else. Left in view it is read as the start of the *next* word's
    # command: the IEEEtran title ``Feedback: \\Scene-specific Small Models``
    # came back as ``\场景特定的小``, a command name in Chinese, and the paper's
    # title was refused and shipped in English.
    # 显式换行，就是两个反斜杠、后面什么都没有。留在模型眼前，它会把它读成**下一个**词
    # 的命令开头：IEEEtran 那篇的标题 Feedback: \\Scene-specific Small Models 回来成了
    # \场景特定的小——一个中文的命令名——于是整篇论文的标题被拒收、留在了英文。
    r"|\\\\\*?(?:\[[^\]]*\])?"
    r"|\\(?:newline|linebreak|par|noindent|centering|bigskip|medskip|smallskip)\b"
    # Conditionals. A paragraph the author disabled opens with ``\if0`` and
    # closes with ``\fi`` many lines later, so the model sees only the ``\if0``
    # — noise at the start of a sentence — and drops it. Measured on the Gemini
    # report: the ``\if0`` went missing, the ``\fi`` stayed, and the disabled
    # section was typeset, which is how the compile came to ask for a figure
    # file the paper does not ship.
    # 条件语句。作者停用的一段以 \if0 开头、在很多行之后才以 \fi 收尾，所以模型只看到
    # 那个 \if0——一句话开头的噪音——就把它丢了。在 Gemini 那篇上实测：\if0 没了、\fi
    # 还在，被停用的那一节就被排版了出来，编译于是去找一个论文根本没附带的图片文件。
    r"|\\if[a-zA-Z@]+|\\if(?![a-zA-Z@])[ \t]*[0-9]"
    r"|\\(?:else|fi|clearpage|newpage|pagebreak|appendix|onecolumn|twocolumn"
    r"|maketitle|tableofcontents|columnbreak|hline|toprule|midrule|bottomrule)\b"
    # Settings and lengths. Their arguments are numbers, colour components and
    # package options — never prose — and a model handed them rewrites them:
    # ``\setcounter{equation}{0}`` came back as ``\arabic{equation}{0}``.
    # 设置与长度。它们的参数是数字、颜色分量和宏包选项——从来不是散文——而模型拿到它们
    # 就会改写：\setcounter{equation}{0} 回来成了 \arabic{equation}{0}。
    r"|\\(?:vspace|hspace)\*?\s*\{[^{}]*\}"
    # Style, size and colour names. They are keys into a package's own table,
    # never words: a paragraph that is nothing but ``\thispagestyle{firstpage}``
    # came back as 「标题样式设置为第一页」 with the command gone, and a title
    # carrying ``\fontsize{15.5pt}{23pt}\color{black}`` lost both along with
    # most of itself.
    # 样式名、字号与颜色名。它们是宏包内部表格的键，从来不是词：一个整段只有
    # \thispagestyle{firstpage} 的段落回来成了「标题样式设置为第一页」，命令没了；而一个
    # 带 \fontsize{15.5pt}{23pt}\color{black} 的标题，把这两样连同它自己的大半一起弄丢了。
    r"|\\(?:thispagestyle|pagestyle|theoremstyle|bibliographystyle)\s*\{[^{}]*\}"
    r"|\\fontsize\s*\{[^{}]*\}\s*\{[^{}]*\}"
    r"|\\(?:color|textcolor|pagecolor)\s*(?:\[[^\]]*\])?\s*\{[^{}]*\}"
    r"|\\(?:captionsetup|sisetup|tcbset|tcbuselibrary|lstdefinestyle"
    r"|numberwithin|counterwithin)\s*(?:\[[^\]]*\])?\s*(?:\{[^{}]*\}){1,2}"
    r"|\\(?:setlength|addtolength|setcounter|stepcounter|refstepcounter"
    r"|definecolor|lstset|pagenumbering|graphicspath|hypersetup|arraystretch)"
    r"\s*(?:\{[^{}]*\}){1,3}"
    # Table structure. A cell separator and a rule are not words, and a model
    # that moves one has moved a column. Protected rather than merely checked,
    # because a placeholder that must come back is a stronger promise than a
    # count that must match: the model reproduces {v7} where the column belongs.
    # 表格结构。列分隔符和横线不是词，模型挪动它就等于挪动了一列。这里是保护而不只是
    # 事后核对，因为「必须还回来的占位符」比「个数必须相等」是更强的约束：模型会把
    # {v7} 放在这一列该在的位置上。
    r"|(?<!\\)&"
    r"|\\(?:cmidrule|cline|specialrule)\s*(?:\([^)]*\))?\s*(?:\[[^\]]*\])?"
    r"\s*\{[^{}]*\}"
    r"|\\(?:rowcolor|cellcolor|columncolor)\s*(?:\[[^\]]*\])?\s*\{[^{}]*\}"
    # Only the leading arguments of \multicolumn and \multirow: the span and
    # the alignment are structure, while the last argument is the heading a
    # reader wants translated.
    # \multicolumn 与 \multirow 只保护前面的参数：跨度和对齐是结构，最后一个参数才是
    # 读者想看到译文的那个表头。
    r"|\\multicolumn\s*\{[^{}]*\}\s*\{[^{}]*\}"
    r"|\\multirow\s*(?:\[[^\]]*\])?\s*\{[^{}]*\}\s*\{[^{}]*\}"
    # A bibliography entry's key, and the separator between its fields. Both
    # are machinery: a mangled key is a citation that resolves to nothing.
    # 参考文献条目的键，以及字段之间的分隔符。两者都是机器用的东西：键被改坏，引用就
    # 指向不存在的地方。
    r"|\\bibitem\s*(?:\[[^\]]*\])?\s*\{[^{}]*\}"
    r"|\\(?:newblock|natexlab|bibinfo|bibnamefont|bibfnamefont|urlprefix)\b"
    # Font size and shape declarations. They take no argument, carry no
    # meaning a translator can render, and are long enough to get mistyped:
    # ``\footnotesize`` came back as ``\footnotesiz`` twice in one paper and as
    # ``\footnotesizebit`` in another, each time refusing a whole table.
    # 字号与字形声明。它们不带参数，没有可翻译的含义，而且长到会被打错：\footnotesize
    # 在一篇论文里两次回来成了 \footnotesiz，在另一篇里成了 \footnotesizebit，每次都
    # 赔掉一整张表。
    # A list item marker. Protected so the number of items is enforced by the
    # placeholder check rather than hoped for: a translation that merged two
    # bullets used to be caught only when it also lost the environment.
    # 列表项标记。保护起来，条目个数就由占位符检查来保证，而不是靠指望：以前译文把两条
    # 并成一条，只有在它同时弄丢了环境时才会被抓住。
    # A link's address, but not its text. \href takes the URL first and the
    # words a reader sees second; protecting the whole command would leave the
    # link text in English, and leaving the whole command in view loses the
    # link — measured on the Gemini report, two lists of policy links came back
    # with the \href gone and only the words left.
    # 链接的地址，但不包括链接文字。\href 的第一个参数是网址，第二个是读者看得见的文字；
    # 整条保护会让链接文字留在英文，整条暴露则会丢掉链接——在 Gemini 那篇上实测，两份
    # 政策链接清单回来时 \href 没了，只剩下文字。
    # Pseudocode line and block structure. Which macros are protected and
    # which are left in view is the whole design of translating an algorithm:
    # a macro that carries no words is held behind a placeholder, so the number
    # of lines and the shape of the nesting are enforced rather than hoped for;
    # a macro whose argument is a condition or a comment stays in view, because
    # 「if the buffer is empty」 is a sentence and a reader wants it in Chinese.
    # The indentation a reader sees comes from these macros, never from the
    # whitespace in the file, so nothing is lost when the model reflows.
    # 伪代码的行结构与块结构。哪些宏保护、哪些留着，就是翻译算法块的全部设计：不带文字
    # 的宏藏在占位符后面，于是行数与嵌套的形状是被强制的而不是指望来的；参数是条件或
    # 注释的宏留在视野里，因为「if the buffer is empty」是一句话，读者要看到它的中文。
    # 读者看到的缩进来自这些宏，从来不来自文件里的空白，所以模型重排也丢不掉什么。
    r"|\\(?:State|Statex|Else|EndIf|EndFor|EndWhile|EndFunction|EndProcedure"
    r"|EndLoop|EndSwitch|EndCase|Loop|Repeat|Break|Continue|Require|Ensure"
    r"|Return|KwTo|KwRet|LinesNumbered|DontPrintSemicolon|SetAlgoLined"
    r"|SetAlgoNoLine|BlankLine|nl|lnl)(?![a-zA-Z])"
    r"|\\(?:SetKwInOut|SetKwFunction|SetKwData|SetAlgoNlRelativeSize"
    r"|SetKwProg|SetKw)\s*(?:\{[^{}]*\}){1,3}"
    # algorithm2e ends each statement with `\;`. It is a line break wearing the
    # costume of a spacing command, and losing one merges two statements.
    # algorithm2e 用 \; 结束每条语句。那是一个披着间距命令外衣的换行，丢掉一个就把两条
    # 语句并成了一条。
    r"|\\;"
    # Any command whose braced argument is shaped like a label key: no spaces,
    # and a colon or an underscore in it. The named list above cannot cover the
    # macros a paper defines for itself — this one writes \Eqref, and its
    # underscores were escaped as though they were prose, which turns
    # \Eqref{eq:released_transport_estimator} into a reference to a label that
    # does not exist. TeX reports it as "Missing \endcsname inserted".
    # 任何花括号参数长得像标签键的命令：里面没有空格，且带冒号或下划线。上面那份点名的
    # 清单管不到论文自己定义的宏——这一篇定义了 \Eqref，它的下划线被当成散文转义了，于是
    # \Eqref{eq:released_transport_estimator} 指向了一个不存在的标签。TeX 报
    # 「Missing \endcsname inserted」。
    r"|\\[a-zA-Z@]+\*?\s*\{[A-Za-z0-9][A-Za-z0-9_:.+-]*[_:][A-Za-z0-9_:.+-]*\}"
    r"|\\href\s*\{[^{}]*\}"
    r"|\\item\s*(?:\[[^\]]*\])?"
    r"|\\(?:tiny|scriptsize|footnotesize|small|normalsize|large|Large|LARGE"
    r"|huge|Huge|bfseries|mdseries|itshape|slshape|scshape|upshape|normalfont"
    r"|rmfamily|sffamily|ttfamily|centering|raggedright|raggedleft"
    r"|em|bf|it|sc|sl|tt|rm|sf)(?![a-zA-Z])")

# What an inline protected command is replaced by while the model works on the
# sentence around it. A different letter from the parser's marks so the two
# cannot be confused when reading a log.
# 模型处理这句话时，被保护的行内命令用什么替代。字母与解析器的记号不同，这样看日志时
# 不会把两者搞混。
# Braces, not angle brackets. A model reads `{v1}` as a placeholder it has seen
# ten thousand times in localisation data and carries it through; `<LTR1>` looks
# like markup it is free to drop, and it dropped enough of them to cost 10% of
# the PRL paper's body.
# 用花括号而不是尖括号。模型把 {v1} 读作它在本地化数据里见过一万次的占位符，会原样带
# 过去；<LTR1> 看起来像它可以随手删掉的标记，而它删掉的量足以吃掉 PRL 那篇正文的 10%。
_INLINE_MARK = "{{v{}}}"
_ANY_MARK = regex.compile(r"<LTX\d+>|\{v\d+\}|\{k\d+\}")

# Inline mathematics, in either notation. Protected for the same reason the
# PDF pipeline protects formulas — that pipeline replaces every formula with a
# ``{v1}`` before the model sees the sentence, and it is why formulas survive
# there.
#
# Measured on the PRL paper: shown ``$\sigma^2 \equiv \langle(\phi-\bar\phi)^2\rangle$``
# the model returned ``$σ^2$≡⟨(φ-φ̄)^2⟩`` — the mathematics rewritten in Unicode,
# half of it outside the dollars. TeX stopped at "Missing $ inserted", and the
# repairs here made it worse rather than better, because there was no longer a
# formula to repair.
# 行内数学，两种写法都算。保护它的理由和 PDF 管线保护公式的理由一样——那条管线在模型
# 看到句子之前就把每个公式换成了 {v1}，公式能在那里存活正是因为这个。
#
# 在 PRL 那篇上实测：给它 $\sigma^2 \equiv \langle(\phi-\bar\phi)^2\rangle$，回来的是
# $σ^2$≡⟨(φ-φ̄)^2⟩——数学被改写成了 Unicode，一半还落在美元符号外面。TeX 停在
# 「Missing $ inserted」，而这里的修复只会越修越糟，因为已经没有公式可修了。
_INLINE_MATH = regex.compile(
    r"(?<!\\)\$\$(?:[^$\\]|\\.)+?\$\$"
    r"|(?<!\\)\$(?:[^$\\]|\\.)+?\$"
    r"|\\\((?:[^\\]|\\[^)])*?\\\)"
    r"|\\\[(?:[^\\]|\\[^\]])*?\\\]")


def macro_pattern(names) -> regex.Pattern | None:
    r"""
    中文：把论文自定义的宏名编成一个匹配「用到它们的地方」的正则。

    Build a pattern matching uses of the paper's own macros.

    Args:
        names (Iterable[str]): Macro names without the backslash, as collected
            from the document's ``\newcommand`` definitions.

    Returns:
        regex.Pattern | None: A pattern matching a use of any of them, or
            ``None`` when the paper defines none.

    Note:
        A paper's own macro is the one command a model has no way to know. The
        LLaMA paper writes its model's name as ``\model`` in several hundred
        places, including its title; handed that, the model returned
        ``\mymodel`` once and left the title in English another time. The
        definition is already lifted out of sight — protecting the *uses* is
        what makes that consistent.
        论文自己定义的宏，是模型唯一没有办法知道的命令。LLaMA 那篇在几百处——包括标题里
        ——用 \model 写它的模型名；给它看，它一次返回了 \mymodel，另一次干脆把标题留在了
        英文。定义本身已经被抽走看不见了——把**用到的地方**也保护起来，才算前后一致。
    """
    escaped = sorted({regex.escape(name) for name in names if name and name != "?"},
                     key=len, reverse=True)
    if not escaped:
        return None
    return regex.compile(r"\\(?:" + "|".join(escaped) + r")(?![a-zA-Z@])")


def protect_inline(text: str, macros: regex.Pattern | None = None, *,
                   commands: bool = False) -> tuple[str, dict[str, str]]:
    r"""
    中文：把行内引用、公式与论文自定义的宏换成记号，别让模型看见命令本身。

    Replace inline citations, formulas and the paper's own macros with marks.

    Args:
        text (str): The text about to be translated.
        macros (regex.Pattern | None): Uses of the paper's own macros, from
            :func:`macro_pattern`.
        commands (bool): Also replace every remaining command *name*, leaving
            its braces and their contents in view. Set for pseudocode, where
            the commands are the language's keywords and read to a model as
            English words: measured on the IEEEtran paper's algorithm2e block,
            ``\For{…}{`` came back as ``对于 {`` and ``\KwIn{…}`` as ``输入：…``
            with the braces gone, which is an unbalanced block, not a
            translation. What it never sees, it cannot translate.
            也把剩下的每个命令**名**换掉，花括号和里面的内容仍留在视野里。用于伪代码——
            那里的命令就是这门语言的关键字，在模型看来是英文单词：在 IEEEtran 那篇的
            algorithm2e 块上实测，\For{…}{ 回来成了「对于 {」，\KwIn{…} 成了「输入：…」，
            花括号没了——那是一个不闭合的块，不是译文。它看不见的东西，它就翻不动。

    Returns:
        tuple[str, dict[str, str]]: The text with marks in place, and what each
            mark stands for.
    """
    mapping: dict[str, str] = {}

    def swap(match: regex.Match) -> str:
        """中文：换成记号并记下原样。 / Swap in a mark and remember the original."""
        mark = _INLINE_MARK.format(len(mapping) + 1)
        mapping[mark] = match.group(0)
        return mark

    # The parser's own marks are normalised to the same form. They are dropped
    # at the same rate as anything else that looks like markup — 11% of the PRL
    # paper's body was refused for losing the marks standing in for its display
    # equations — and the model has no reason to treat two notations
    # differently. One notation, and it is the one models preserve.
    # 解析器自己的记号也归一成同样的形式。它们和别的看起来像标记的东西一样会被丢掉——
    # PRL 那篇有 11% 的正文因为丢了代表行间公式的记号而被拒收——而模型没有理由对两种
    # 写法区别对待。只用一种写法，而且用模型会保留的那一种。
    text = regex.sub(r"<LTX\d+>", swap, text)
    text = _INLINE_MATH.sub(swap, text)
    text = _INLINE_KEYED.sub(swap, text)
    if macros is not None:
        text = macros.sub(swap, text)
    if commands:
        text = _COMMAND.sub(swap, text)
    return text, mapping


def restore_inline(text: str, mapping: dict[str, str]) -> str:
    """
    中文：把记号换回原来的命令。

    Put the protected commands back.

    Args:
        text (str): A translation containing marks.
        mapping (dict[str, str]): What each mark stands for.

    Returns:
        str: The translation with the commands restored.
    """
    for mark, original in mapping.items():
        text = text.replace(mark, original)
    return text


_EDGE_MARKS_HEAD = regex.compile(r"^(?:\s*\{v\d+\})+\s*")
_EDGE_MARKS_TAIL = regex.compile(r"\s*(?:\{v\d+\}\s*)+$")


def _peel_guarded(text: str) -> tuple[str, str, str]:
    r"""
    中文：把首尾的占位符从待翻译文本里摘出来，别让模型看见。

    Hold placeholders at the two ends out of the model's sight.

    Args:
        text (str): The protected text about to be sent.

    Returns:
        tuple[str, str, str]: What comes before, what to translate, what comes
            after. The three join back to ``text`` exactly.

    Note:
        The same reasoning as the parser's :func:`_peel_marks`, one layer
        further in. The LLaMA paper's title is ``\model: Open and Efficient
        Foundation Language Models``; with the paper's own macro protected it
        becomes ``{v1}: Open and Efficient…``, and a mark at the very start
        reads as noise — the model dropped it and the title was refused. A mark
        in the middle of a sentence is a different thing entirely: it stands
        for a formula or a citation that belongs where it is, so it stays.
        与解析器里 _peel_marks 的道理相同，只是往里又深了一层。LLaMA 那篇的标题是
        \model: Open and Efficient Foundation Language Models；把论文自己的宏保护起来
        之后，它变成 {v1}: Open and Efficient…，而开头的记号看着就是噪音——模型把它丢了，
        标题因此被拒收。句子中间的记号完全是另一回事：它代表一个就该待在那儿的公式或
        引用，所以留在原地。
    """
    head = _EDGE_MARKS_HEAD.match(text)
    lead = head.group(0) if head else ""
    rest = text[len(lead):]
    end = _EDGE_MARKS_TAIL.search(rest)
    tail = end.group(0) if end else ""
    middle = rest[:len(rest) - len(tail)] if tail else rest
    return lead, middle, tail


def _critical_counts(text: str) -> dict[str, int]:
    """
    中文：数出文本里各个关键命令出现了几次。

    Count how many times each critical command appears.

    Args:
        text (str): LaTeX source.

    Returns:
        dict[str, int]: Command name to count, omitting those absent.
    """
    found: dict[str, int] = {}
    for name in _CRITICAL:
        count = len(regex.findall(rf"\\{name}\b", text))
        if count:
            found[name] = count
    return found


# A word long enough that keeping it verbatim means something. Acronyms and
# symbols survive any translation; four-letter-plus words do not.
# 长到「原样保留」有意义的词。缩写和符号在任何译文里都会留下，四个字母以上的词不会。
_WORD = regex.compile(r"[A-Za-z]{4,}")

# Above this share of the source's long words surviving verbatim, the answer is
# the source rather than a translation of it.
# 原文里的长词有这么大比例原样出现在译文里，那答案就是原文本身而不是它的译文。
_ECHO_SHARE = 0.9

# Below this many long words the share means nothing: a three-word caption can
# keep all three and still be a translation.
# 长词少于这个数时，比例不说明任何问题：三个词的图注三个都留着也可能是译文。
_ECHO_FLOOR = 8

# How much longer than its source a translation may be before it is not one.
# Generous: a heading of two words can legitimately triple, and 「的」-heavy
# Chinese runs long on short inputs.
# 译文比原文长到什么程度就不再是译文。放得宽：两个词的标题合理地翻三倍，短输入上多「的」
# 的中文也会偏长。
_LENGTH_RATIO = 1.2
_LENGTH_SLACK = 20

# Chinese characters, counted rather than the whole answer. The short-heading
# instruction invites the model to keep the English term in parentheses after
# the Chinese — 「大规模多任务语言理解（Massive Multitask Language Understanding,
# MMLU）」 for a four-character heading — and a rule that measured the whole
# string called that a runaway. What runs away is the *Chinese*: an invented
# paragraph is sixty characters of it where a heading needs four.
# 数汉字，而不是数整个答案。短标题的指令鼓励模型在中文后面用括号保留英文原词——四个
# 字符的标题会得到「大规模多任务语言理解（Massive Multitask Language Understanding,
# MMLU）」——而按整串长度判断的规则会把这当成失控。真正失控的是**中文**：凭空写出的一段
# 有六十个汉字，而一个标题只需要四个。
_CJK = regex.compile(r"[\u3400-\u9fff]")

# A loose ceiling on the whole answer as well, for the case where the model
# answers at length in the source language and the echo test does not fire
# because it rewrote rather than copied.
# 对整个答案也留一道宽松的上限，用于模型用原文语言长篇作答、而它是改写不是照抄、因此
# 原样返回的判断没有触发的情况。
_TOTAL_RATIO = 3.0
_TOTAL_SLACK = 120

# What counts as content: letters, digits and Chinese. Braces, backslashes and
# punctuation are not what a translation is measured in.
# 什么算内容：字母、数字和汉字。花括号、反斜杠和标点不是衡量译文的单位。
_CONTENT = regex.compile(r"[0-9A-Za-z㐀-鿿]")

# The least a translation may keep of its source, measured in content
# characters. Chinese is shorter than English — 0.3 to 0.6 is the usual band —
# but it is not an order of magnitude shorter.
#
# Measured on arXiv:2608.12209: a title reading
# ``\fontsize{15.5pt}{23pt}\sffamily\bfseries\color{black} Generation as
# Auxiliary Supervision: …`` came back as ``\sffamily\bfseries, \\, \\`` —
# the commands dropped, the title itself gone, two commas left. Every other
# check passed it: nothing was invented, no brace was unbalanced, no
# placeholder was lost, and it is certainly not too long.
# 译文至少要保住原文多少内容，按内容字符数算。中文比英文短——通常在 0.3 到 0.6 之间——
# 但短不了一个数量级。
#
# 在 arXiv:2608.12209 上实测：标题
# \fontsize{15.5pt}{23pt}\sffamily\bfseries\color{black} Generation as Auxiliary
# Supervision: … 回来成了 \sffamily\bfseries, \\, \\——命令丢了，标题本身没了，剩两个
# 逗号。其余每一条检查都放行了：没凭空造命令，花括号配平，占位符没丢，长度当然也不算长。
_MINIMUM_SHARE = 0.22

# Below this much content the share means nothing: a two-word heading can lose
# half of itself to a good translation.
# 内容少于这么多时比例不说明问题：两个词的标题被译好之后本来就可能只剩一半。
_MINIMUM_FLOOR = 40


def _spaces(text: str) -> str:
    """中文：把空白归一，好逐字比较。 / Normalise whitespace for an exact comparison."""
    return regex.sub(r"\s+", " ", text).strip()


def echoed(source: str, translated: str, *, exact: bool = False) -> bool:
    r"""
    中文：判断模型是不是把原文原样退了回来。

    Report whether the model returned the source instead of translating it.

    Args:
        source (str): The passage that was sent, marks and all.
        translated (str): What came back.

    Returns:
        bool: ``True`` when almost every long word of the source is still
            present verbatim, which no translation into another language
            leaves behind.

    Note:
        Why this is checked at all: BabelDOC's prompt tells the model to return
        the original when translation is unnecessary, and a paragraph of LaTeX
        reads to it as the "codes" that clause names. When the echo breaks
        nothing — no dropped brace, no renamed command — every other check here
        passes and the English ships as though it had been translated. That is
        what a reader sees as 「整段漏译」.
        为什么要查这个：BabelDOC 的提示词让模型在「不需要翻译」时原样返回，而一段 LaTeX
        在它看来正是那句话点名的「代码」。当这种原样返回没弄坏任何东西——没掉花括号、
        没改命令名——这里其他每一条检查都会放行，英文就被当成译文交付了。读者看到的
        「整段漏译」就是这么来的。
        Commands and placeholders are discounted first: they are meant to
        survive, and counting them would make every short passage look echoed.
        先扣掉命令和占位符：它们本来就该原样存活，把它们算进去会让每一段短文本都像是
        原样返回。
    """
    if _spaces(source) == _spaces(translated):
        # Word for word the same answer as the question. A heading of three
        # words is under the threshold below and would otherwise pass silently:
        # "Closed-book Question Answering" came back unchanged and shipped as
        # though it had been translated.
        # 一字不差地把问题当答案交回来。三个词的标题达不到下面那个门槛，不查这一条就会
        # 悄悄放行：Closed-book Question Answering 原样回来，被当成译文交付了。
        return True

    if exact:
        # A table's cells are mostly names and numbers that any correct
        # translation keeps: 「GPT-3」, 「Gopher」, 「0-shot」. Measured on the LLaMA
        # paper with tables switched on, the word-survival test below called
        # six correctly translated tables an echo and refused them. Where the
        # content is meant to survive, only an answer identical to the question
        # is evidence of one.
        # 表格的单元格大多是任何正确译文都会保留的名字和数字：GPT-3、Gopher、0-shot。在
        # LLaMA 那篇上打开表格实测，下面那条按词存活率的判断把六张翻对了的表当成原样
        # 返回、拒收掉了。内容本来就该存活的地方，只有「答案与问题一模一样」才算证据。
        return False

    stripped = regex.sub(r"\\[a-zA-Z@]+\*?|\{v\d+\}|<LTX\d+>", " ", source)
    words = _WORD.findall(stripped)
    if len(words) < _ECHO_FLOOR:
        return False
    kept = sum(1 for word in words if word in translated)
    return kept >= _ECHO_SHARE * len(words)


# A number as it appears in a table cell or a citation: digits, optionally with
# a decimal point, a thousands separator or a sign.
# 数字在表格单元格或引用里的样子：数位，可能带小数点、千位分隔符或正负号。
# Digits with any separators *inside* them, and nothing trailing: a full stop
# after a page number belongs to the sentence, not to the number, and counting
# it made 「pp. 7439.」 and 「pp. 7439，」 look like different figures.
# 数位以及夹在**中间**的分隔符，末尾不带任何东西：页码后面的句点属于句子而不属于数字，
# 把它算进去会让 pp. 7439. 与 pp. 7439， 看起来像是两个不同的数。
_NUMBER = regex.compile(r"[-+]?\d+(?:[.,]\d+)*")


def numbers_kept(source: str, translated: str) -> str:
    """
    中文：核对表格与文献里的数字一个不少、一个不变。

    Check that every number came back unchanged.

    Args:
        source (str): The passage that was sent.
        translated (str): What came back.

    Returns:
        str: Empty when the numbers match, otherwise what changed.

    Note:
        Applied to tables and bibliographies, not to prose. A table cell *is*
        its number, and a model that renders 58.4 as 58,4 or rounds it has
        published a different result; a sentence, by contrast, may legitimately
        turn "thirty" into 三十 or drop a figure into a different clause.
        用于表格与参考文献，不用于散文。表格单元格本身就是那个数字，模型把 58.4 写成
        58,4 或者四舍五入，等于发表了另一个结果；而一句话里把 thirty 写成三十、或把
        某个数字挪到别的从句去，都是合理的。
    """
    before = sorted(_NUMBER.findall(source))
    after = sorted(_NUMBER.findall(translated))
    if before == after:
        return ""
    lost = [value for value in before if value not in after]
    if lost:
        return f"数字对不上，少了 {'、'.join(lost[:3])}"
    return f"数字对不上（{len(before)} 个变成 {len(after)} 个）"


def commands_kept(source: str, translated: str) -> str:
    r"""
    中文：核对源里的每个命令都还在，一个都没少。

    Check that no command was dropped.

    Args:
        source (str): The passage that was sent.
        translated (str): What came back.

    Returns:
        str: Empty when every command survived, otherwise which one did not.

    Note:
        Applied to pseudocode, not to prose. In an algorithm every visible
        command is structure — ``\If`` without its ``\EndIf`` is a block that
        never closes — whereas a translation of a sentence may reasonably drop
        an ``\emph`` that Chinese does not want.
        用于伪代码，不用于散文。算法块里每一个看得见的命令都是结构——\If 少了 \EndIf，
        这个块就永远不闭合——而一句话的译文合理地丢掉一个中文不需要的 \emph 是可以的。
    """
    before: dict[str, int] = {}
    for name in _COMMAND.findall(source):
        before[name] = before.get(name, 0) + 1
    after: dict[str, int] = {}
    for name in _COMMAND.findall(translated):
        after[name] = after.get(name, 0) + 1
    lost = sorted(name for name, count in before.items()
                  if after.get(name, 0) < count)
    if lost:
        return f"丢了命令 {'、'.join(lost[:3])}"
    return ""


def check(source: str, translated: str, *, numbers: bool = False,
          commands: bool = False, heading: bool = False) -> str:
    """
    中文：检查译文能不能用；不能用就说明理由。

    Decide whether a translation is usable.

    Args:
        source (str): The LaTeX that was sent.
        translated (str): What came back.
        numbers (bool): Also require every number to come back unchanged. Set
            for tables and bibliographies, where a number is the content.
        commands (bool): Also require every command to come back. Set for
            pseudocode, where a dropped command is a block that never closes.
        heading (bool): Whether this block is a heading rather than a passage.
            A heading is exempt from the minimum length: 「复杂度分析」 is a
            complete and correct rendering of "Complexity Analysis of the
            Proposed Method", and a rule that called it too short refused three
            correct translations on the bench.
            这一块是标题而不是段落。标题不受最短长度约束：「复杂度分析」是 "Complexity
            Analysis of the Proposed Method" 完整而正确的译法，而把它判为过短的规则在
            bench 上拒收了三条正确的译文。

    Returns:
        str: Empty when the translation is fine, otherwise why it is not.

    Note:
        Brace balance is checked rather than brace count: a translation may
        legitimately add or drop a pair, but it may never leave one open, and
        an unclosed brace is what turns into an error hundreds of lines later.
        检查花括号是否配平而不是数个数：译文可以合理地增删一对，但绝不能留下不闭合的
        那一个，而不闭合的花括号正是几百行之后才报错的那种东西。
    """
    if not translated.strip():
        return "译文为空"

    if echoed(source, translated, exact=numbers):
        return "模型把原文退了回来，没有翻译"

    # A translation into Chinese is shorter than its English source in
    # characters, not longer — 0.6 to 0.9 times is the usual range. Well past
    # that, the model stopped translating and started writing.
    #
    # Measured on the LLaMA paper: asked for ``Code generation``, the two words
    # of a \subsection argument, it returned a five-paragraph essay about an
    # end-to-end neural architecture that appears nowhere in the paper. The
    # essay went inside ``\subsection{`` and TeX stopped with "Paragraph ended
    # before \@sect was complete" — a message about braces, four hundred lines
    # from the invention that caused it.
    # 中文译文按字符数比英文原文短，而不是长——通常是 0.6 到 0.9 倍。远远超出这个范围，
    # 就说明模型不是在翻译，而是在写作。
    #
    # 在 LLaMA 那篇上实测：让它翻译 \subsection 参数里的两个词 Code generation，它回来
    # 的是五段关于端到端神经网络架构的文章，而论文里根本没有这些内容。这段文章进了
    # \subsection{ 里面，TeX 报「Paragraph ended before \@sect was complete」停下——一条
    # 讲花括号的报错，离真正的病因有四百行。
    if not heading:
        content = len(_CONTENT.findall(_ANY_MARK.sub(" ", source)))
        kept = len(_CONTENT.findall(_ANY_MARK.sub(" ", translated)))
        if content >= _MINIMUM_FLOOR and kept < _MINIMUM_SHARE * content:
            return f"译文过短（{content} 字的内容只剩 {kept} 字）"

    chinese = len(_CJK.findall(translated))
    if (chinese > _LENGTH_SLACK + _LENGTH_RATIO * len(source)
            or len(translated) > _TOTAL_SLACK + _TOTAL_RATIO * len(source)):
        return f"译文长度异常（{len(source)} 字变成 {len(translated)} 字）"

    wanted = _ANY_MARK.findall(source)
    got = _ANY_MARK.findall(translated)
    lost = [mark for mark in set(wanted) if got.count(mark) < wanted.count(mark)]
    if lost:
        return f"丢了占位符 {'、'.join(sorted(lost)[:4])}"

    depth = 0
    escaped = False
    for character in translated:
        if escaped:
            escaped = False
            continue
        if character == "\\":
            escaped = True
        elif character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth < 0:
                return "花括号不配平（多了一个右括号）"
    if depth:
        return f"花括号不配平（少了 {depth} 个右括号）"

    if numbers:
        trouble = numbers_kept(source, translated)
        if trouble:
            return trouble

    if commands:
        trouble = commands_kept(source, translated)
        if trouble:
            return trouble

    dollars = len([index for index, character in enumerate(translated)
                   if character == "$" and (not index or translated[index - 1] != "\\")])
    if dollars % 2:
        return "美元符号不成对，数学模式没有闭合"

    before, after = _critical_counts(source), _critical_counts(translated)
    for name, count in before.items():
        if after.get(name, 0) < count:
            return f"丢了 {count - after.get(name, 0)} 处 \\{name}"

    # A translation must not invent control sequences. Measured on the PRL
    # paper: \setcounter{equation}{0} came back as \arabicfamily{equation}{0},
    # a command that exists nowhere, and TeX reports it as "Undefined control
    # sequence" — true but unhelpful, four hundred lines from the paragraph.
    # The invariant is simple and strong: translating prose never needs a
    # command the source did not already use.
    # 译文不得凭空造出控制序列。在 PRL 那篇上实测：\setcounter{equation}{0} 回来变成了
    # \arabicfamily{equation}{0}，一个根本不存在的命令，TeX 报「Undefined control
    # sequence」——没说错，但没用，而且离那一段有四百行。这条不变量简单而有力：翻译散文
    # 从来不需要原文没用过的命令。
    # A command name is ASCII letters and nothing else, so a backslash followed
    # by a Chinese character is a command name the model translated. Measured on
    # the LLaMA paper: \footnote came back as \脚注, and every check here missed
    # it — the pattern that finds control sequences only matches ASCII, so the
    # invented-command test never saw it and the count was unchanged. TeX
    # reported "Undefined control sequence" and stopped.
    # 命令名只能由 ASCII 字母构成，所以反斜杠后面跟着汉字，就是模型把命令名给翻译了。
    # 在 LLaMA 那篇上实测：\footnote 回来成了 \脚注，而这里每一条检查都没抓住它——找
    # 控制序列的那个正则只匹配 ASCII，凭空造命令的检查根本没看见它，个数也没变。TeX
    # 报「Undefined control sequence」然后停下。
    translated_names = regex.findall(r"\\([^\x00-\x7f]+)", translated)
    if translated_names:
        return f"命令名被翻译了：\\{translated_names[0][:6]}"

    known = set(_COMMAND.findall(source))
    invented = sorted({name for name in _COMMAND.findall(translated)
                       if name not in known and name.lstrip("\\") not in _MAY_ADD})
    if invented:
        return f"凭空造了命令 {'、'.join(invented[:3])}"

    # Environments are compared name by name against the source rather than
    # merely counted. Counting caught neither of the two failures it was
    # written for: \begin{} with an empty name ("Environment  undefined"), and
    # an \begin{itemize} whose \end went missing, which TeX reports four
    # thousand lines later as "ended by \end{document}".
    #
    # Compared against the source, not balanced within the piece: `center` and
    # `proof` are deliberately left in the body, so a paragraph may hold one
    # half of one and be perfectly correct.
    # 环境要逐名与原文对照，而不只是数个数。数个数对它本该抓的两种失败一个也没抓住：
    # 名字为空的 \begin{}（报「Environment  undefined」），以及丢了 \end 的
    # \begin{itemize}，TeX 在四千行之后才报「ended by \end{document}」。
    #
    # 是与原文对照，而不是要求在这一块内部自洽：center 和 proof 是有意留在正文里的，
    # 所以一个段落里可能只有其中半边，而且完全正确。
    def environments(text: str) -> dict[tuple[str, str], int]:
        """中文：数出每个 begin/end 与它的名字。 / Count each begin/end by name."""
        found: dict[tuple[str, str], int] = {}
        for match in _BEGIN_END.finditer(text):
            key = (match.group(1), match.group(2).strip())
            found[key] = found.get(key, 0) + 1
        return found

    before, after = environments(source), environments(translated)
    if before != after:
        empty = [key for key in after if not key[1]]
        if empty:
            return "出现了没有名字的 environment"
        changed = sorted({key[1] for key in set(before) ^ set(after)}
                         | {key[1] for key in before if before.get(key) != after.get(key)})
        return f"environment 对不上：{'、'.join(changed[:3])}"
    return ""


# How many times to ask again for a block whose translation was refused. The
# model is not deterministic, and a brace it dropped once it usually keeps on
# the next attempt — measured on the ICLR paper, two of three refusals were
# recovered by asking again. Past two extra attempts the failure is in the
# block, not the draw.
# 一块译文被拒收后再问几次。模型不是确定性的，这一次掉的花括号下一次通常就在了——在
# ICLR 那篇上实测，三处拒收里有两处再问一次就好了。超过两次，问题就在这块本身而不在
# 随机性上了。
# Regions where a number is the content rather than a detail of the prose.
# 数字本身就是内容、而不是行文细节的那些区域。
_EXACT_NUMBERS = frozenset({"table", "table_footnote", "reference",
                            "algorithm"})

# Regions where a whitelisted term must survive in English no matter what, and
# is therefore held behind a marker rather than merely asked for. The same rule
# the PDF path applies: there is no prose here for a marker to disturb, and a
# translated 「baseline」 in a table header is exactly what a whitelist exists to
# prevent.
# 白名单术语必须原样存活、因而用标记扣住而不只是口头要求的区域。与 PDF 那条路的判断
# 相同：这里没有会被标记打乱的行文，而表头里出现「基线」正是白名单要防的事。
# Regions where an answer identical to the question is a correct answer. A
# table of model names and numbers has nothing in it to translate, and so does
# a bibliography entry, and so does an author line. Measured on the LLaMA paper
# with tables switched on: nine tables came back unchanged, correctly, and were
# reported as nine failures. The output is the same either way — what changes is
# whether the run tells the truth about itself.
# 答案与问题一模一样也算正确答案的那些区域。一张全是模型名和数字的表里没有可翻译的东
# 西，一条参考文献条目、一行作者信息同样如此。在 LLaMA 那篇上打开表格实测：九张表原样
# 返回，这是对的，却被报成九次失败。产物两种情况下都一样——变的只是这次运行有没有如实
# 说出自己的情况。
_ECHO_ACCEPTABLE = frozenset({
    "table", "table_footnote", "reference", "author", "affiliation",
    "algorithm",
})

_STRICT_TERMS = frozenset({
    "table", "table_footnote", "table_caption", "figure_caption",
})


def terminology_line(pairs, keeps) -> str:
    """
    中文：把术语对与白名单拼成一行约束，交给提示词。

    Compose the terminology constraint line for one block.

    Args:
        pairs (list[tuple[str, str]]): Source term and the rendering to use.
        keeps (list[str]): Terms that must stay in the source language, written
            as identity mappings — the only formulation a translation model
            reliably obeys.

    Returns:
        str: One line, or empty when the glossary matched nothing here.
    """
    mappings = [f"{source}={target}" for source, target in pairs]
    mappings += [f"{term}={term}" for term in keeps]
    if not mappings:
        return ""
    return "Terminology (use exactly these renderings): " + "; ".join(mappings) + "."

# How long to wait after a transport error before asking again, multiplied by
# the attempt number.
# 传输错误之后等多久再问，按第几次重试成倍数。
_BACKOFF = 2.0

_RETRIES = 2


# Below this a block is not worth splitting: the failure is in the sentence,
# not in the size, and halving it twice more only spends calls.
# 小于这个长度的块不值得再切：问题出在句子本身而不在长度上，再对半切两次只是白花调用。
_SPLIT_FLOOR = 700


def _split_core(core: str) -> tuple[str, str] | None:
    r"""
    中文：把一大块文字在靠中间的自然边界切成两半。

    Split a long block in two at a natural boundary near its middle.

    Args:
        core (str): The text to split.

    Returns:
        tuple[str, str] | None: The two halves, or ``None`` when the block is
            too short to be worth splitting or has no boundary to split at.

    Note:
        Why splitting is worth doing at all: PRL's body arrives as 52 blocks,
        one of which is 5,006 characters — 11% of the paper in a single call.
        One dropped brace there costs a ninth of the translation, and asking
        again gets the same answer, because the difficulty is in the block. Two
        halves each get their own verdict, and a failure costs half as much.
        为什么值得切：PRL 的正文是 52 块，其中一块 5006 字——一次调用里装着全文的 11%。
        那里掉一个花括号就赔掉九分之一的译文，再问一次还是同样的答案，因为难点就在这
        块本身。切成两半，各自独立判定，失败的代价也就减半。
    """
    if len(core) < _SPLIT_FLOOR:
        return None
    middle = len(core) // 2
    for pattern in (r"\n[ \t]*\n", r"(?<=[.!?])\s+(?=[A-Z])", r"(?<=[。！？])",
                    # A pseudocode block has no blank line and often no full
                    # stop; what it has is one statement per line.
                    # 伪代码块里没有空行，往往也没有句号；它有的是一行一条语句。
                    r"\\;[ \t]*\n", r"\n(?=[ \t]*\\State\b)"):
        cuts = [m.end() for m in regex.finditer(pattern, core)]
        if not cuts:
            continue
        cut = min(cuts, key=lambda position: abs(position - middle))
        if _SPLIT_FLOOR // 2 < cut < len(core) - _SPLIT_FLOOR // 2:
            return core[:cut], core[cut:]
    return None


# Longest block sent in one call. Not a guess about what a model can hold — it
# is about what a refusal costs. The Gemini report's model-card appendix arrives
# as a single 22,938-character paragraph because ``multicols`` is deliberately
# left in the text and the block has no blank line in it; one dropped
# ``\begin`` there refused 15% of the paper at once. Cut to this length, the
# same failure costs one chunk.
# 一次调用最多送多长。这不是在猜模型能装下多少——这是在算一次拒收的代价。Gemini 那篇的
# 模型卡附录是一整段 22938 字，因为 multicols 有意留在正文里、而这一块中间没有空行；
# 那里掉一个 \begin 就一次性拒收了全文的 15%。切到这个长度，同样的失败只赔掉一块。
_MAX_CORE = 2000


def _chunks(core: str) -> list[str]:
    """
    中文：把过长的一块按自然边界切成若干段，每段不超过上限。

    Cut a block into pieces no longer than :data:`_MAX_CORE`.

    Args:
        core (str): The text to cut.

    Returns:
        list[str]: The pieces in order, joining back to exactly ``core``. A
            block with no boundary to cut at is returned whole rather than cut
            mid-sentence.
    """
    if len(core) <= _MAX_CORE:
        return [core]
    halves = _split_core(core)
    if halves is None:
        return [core]
    return _chunks(halves[0]) + _chunks(halves[1])


def _translate_one(piece: Piece, translator, ignore_cache: bool,
                   macros: regex.Pattern | None = None, glossary=None,
                   hard_protect: bool = False) -> tuple[Piece, str]:
    """
    中文：翻译一块；过长的先切开，每段独立判定，一段失败只赔掉那一段。

    Translate one piece, cutting it up first when it is too long.

    Args:
        piece (Piece): The block to translate.
        translator: This project's translator, already configured.
        ignore_cache (bool): Passed through to the translator.

    Returns:
        tuple[Piece, str]: The piece and why its translation was refused. A
            piece cut into chunks keeps the source of any chunk that was
            refused and the translation of every chunk that was not, so a
            failure costs one chunk rather than a section.
    """
    parts = _chunks(piece.core)
    if len(parts) == 1:
        return _attempt(piece, translator, ignore_cache, macros, glossary,
                        hard_protect)

    pieces = [Piece(mark=piece.mark, kind=piece.kind, name=piece.name,
                    source=part, label=piece.label, core=part, translate=True)
              for part in parts]
    complaints = []
    for chunk in pieces:
        _, trouble = _attempt(chunk, translator, True, macros, glossary,
                              hard_protect)
        if trouble:
            complaints.append(trouble)
    if not any(chunk.translated for chunk in pieces):
        return piece, complaints[0] if complaints else "译文为空"
    piece.translated = "".join(chunk.translated or chunk.core for chunk in pieces)
    if complaints:
        logger.warning("切成 %d 段后有 %d 段被拒收（%s/%s）", len(pieces),
                       len(complaints), piece.kind, piece.name)
        return piece, complaints[0]
    logger.info("切成 %d 段后采用（%s/%s，%d 字）", len(pieces), piece.kind,
                piece.name, len(piece.core))
    return piece, ""


def _attempt(piece: Piece, translator, ignore_cache: bool,
             macros: regex.Pattern | None = None, glossary=None,
             hard_protect: bool = False) -> tuple[Piece, str]:
    """
    中文：翻译一块，返回它和问题说明（没问题则为空）。

    Translate one piece.

    Args:
        piece (Piece): The block to translate.
        translator: This project's translator, already configured.
        ignore_cache (bool): Passed through to the translator.

    Returns:
        tuple[Piece, str]: The piece and why its translation was refused, empty
            when it was accepted. A refused piece keeps its source, so the
            paper compiles with that paragraph still in English rather than
            not at all.
    """
    complaint = ""
    for attempt in range(_RETRIES + 1):
        whole, protected = protect_inline(piece.core, macros,
                                          commands=piece.label == "algorithm")
        lead, guarded, tail = _peel_guarded(whole)
        held: dict[str, str] = {}
        if glossary is not None:
            pairs, keeps = glossary(guarded)
            strict = piece.label in _STRICT_TERMS or hard_protect
            if strict and keeps:
                guarded, held = protect_keeps(guarded, keeps)
                keeps = []
            about(piece.label, terminology_line(pairs, keeps))
        else:
            about(piece.label, "")
        bare = regex.sub(r"\\[a-zA-Z@]+\*?", " ", _ANY_MARK.sub(" ", guarded))
        if not regex.search(r"[A-Za-z]{2}", bare):
            # Nothing but protected structure and commands. Spending a call
            # here gets the same string back at best, and at worst the model
            # rewrites the colour definition it was handed. Commands are
            # discounted along with the placeholders: a footnote that reads
            # ``\notsotiny\url{…}`` is a link and a size, and there is no
            # sentence in it to translate.
            # 除了被保护的结构和命令什么都不剩。在这里花一次调用，最好的结果是原样拿回
            # 来，最坏的结果是模型改写了给它的颜色定义。命令与占位符一同扣除：一条写作
            # \notsotiny\url{…} 的脚注是一个链接加一个字号，里面没有句子可翻。
            piece.translated = piece.core
            return piece, ""

        try:
            # A retry must not be served the cached answer that was already
            # refused, so the cache is bypassed from the second attempt on.
            # 重试不能拿到那条已经被拒收的缓存结果，因此从第二次起绕开缓存。
            result = translator.translate(guarded,
                                          ignore_cache=ignore_cache or attempt > 0)
        except Exception as problem:  # noqa: BLE001
            complaint = f"翻译出错：{type(problem).__name__}"
            # A transport error is not an answer to argue with, it is a moment
            # to wait through. Retrying instantly spends all three attempts
            # inside the same blip: measured on one run, six blocks were lost
            # to APIConnectionError in the space of a second, and the same
            # blocks translated fine a minute later.
            # 传输错误不是一个可以理论的答案，而是一段该等过去的时间。立刻重试会把三次
            # 机会全花在同一次抖动里：某次运行实测，六块译文在一秒钟之内全都栽在
            # APIConnectionError 上，而同样的块过一分钟就翻好了。
            time.sleep(_BACKOFF * (attempt + 1))
            continue
        result = (result or "").strip()
        # Order matters: a superscript is put back into math mode first, so the
        # escaper then sees it as math and leaves its underscores alone.
        # 顺序要紧：先把上标放回数学模式，转义那一步才会把它当数学、不去动里面的下划线。
        # Before the repairs, because everything below reads the text: a
        # replacement character the model wedged into a word would otherwise be
        # carried through them and compiled into the PDF.
        # 放在各项修复之前，因为下面每一步都要读这段文字：模型塞进词里的替换字符，否则
        # 会被一路带过去、编译进 PDF。
        from local_translator.engine.translators import repair_replacement

        result = repair_replacement(result, guarded)
        result, wrapped = restore_math_mode(result)
        result, fixed = escape_stray_specials(result)
        result, separated = separate_macros_from_chinese(result)
        result, recased = restore_command_case(whole, result)
        result, rejoined = match_paragraph_breaks(guarded, result)
        fixed += separated + recased + rejoined
        if fixed or wrapped:
            logger.debug("修了 %d 处转义、%d 处数学模式（%s/%s）",
                         fixed, wrapped, piece.kind, piece.name)
        complaint = check(guarded, result,
                          numbers=piece.label in _EXACT_NUMBERS,
                          commands=piece.label == "algorithm",
                          heading=is_heading(guarded))
        if complaint.startswith("丢了占位符"):
            # Name the tokens on the next attempt. A model that dropped one
            # from a caption carrying nine formulas usually keeps them all once
            # it is told which they are — the same reminder the PDF pipeline's
            # local translator issues after a placeholder goes missing.
            # 下一次把这些记号点名列出来。一条带九个公式的图注掉了一个占位符，模型在被
            # 告知有哪些之后通常就全都留着了——这与 PDF 管线里本地翻译器在占位符丢失后
            # 发出的提醒是同一条。
            insist(_ANY_MARK.findall(guarded))
        if not complaint:
            # The separation runs again on the restored text, and this is the
            # run that matters: the macros it has to separate are the ones that
            # were behind a placeholder while the model worked, so before
            # restoration there was nothing there to separate. Measured on the
            # LLaMA paper: ``{v1}与其他基础模型`` restored to ``\model与其他基础模型``
            # and TeX called \model undefined, four hundred lines in.
            # 还原之后再做一次分隔，而这一次才是起作用的：需要分隔的宏，在模型工作时正
            # 藏在占位符后面，还原之前那里根本没有东西可分。在 LLaMA 那篇上实测：
            # {v1}与其他基础模型 还原成 \model与其他基础模型，TeX 在四百行处说 \model
            # 未定义。
            restored = restore_inline(restore_keeps(lead + result + tail, held),
                                      protected)
            restored, _ = separate_macros_from_chinese(restored)
            piece.translated, _ = collapse_double_breaks(restored)
            if attempt:
                logger.info("第 %d 次重试后采用（%s/%s）", attempt, piece.kind, piece.name)
            return piece, ""

    if complaint.startswith("模型把原文退了回来") and piece.label in _ECHO_ACCEPTABLE:
        # Asked twice more and answered the same way twice more. In these
        # regions that is the model saying there is nothing here to translate,
        # and it is usually right.
        # 又问了两次，两次答案相同。在这些区域，那是模型在说这里没有可翻译的东西，而它
        # 通常是对的。
        piece.translated = piece.core
        logger.debug("原样保留（%s/%s，%s）", piece.kind, piece.name, piece.label)
        return piece, ""

    # Everything above failed on the block as a whole. Halve it and judge each
    # part on its own: a block that is a tenth of the paper should not be lost
    # entire because one sentence in it came back wrong.
    # 以上都是把整块当一个整体失败的。对半切开、各自判定：一块占全文十分之一的文字，
    # 不该因为其中一句翻坏了就整块丢掉。
    halves = _split_core(piece.core)
    if halves is not None:
        parts = []
        for half in halves:
            piecelet = Piece(mark=piece.mark, kind=piece.kind, name=piece.name,
                             source=half, label=piece.label, core=half,
                             translate=True)
            _, trouble = _attempt(piecelet, translator, True, macros, glossary,
                                  hard_protect)
            if trouble or not piecelet.translated:
                return piece, complaint
            parts.append(piecelet.translated)
        piece.translated = "".join(parts)
        logger.info("拆成两半后采用（%s/%s，%d 字）", piece.kind, piece.name,
                    len(piece.core))
        return piece, ""
    return piece, complaint


def translate_document(document: LatexDocument, translator, *, workers: int = 8,
                       ignore_cache: bool = False, glossary=None,
                       hard_protect: bool = False, on_progress=None) -> dict:
    """
    中文：翻译整篇文档里所有需要翻译的块。

    Translate every piece of a document that asks for it.

    Args:
        document (LatexDocument): A parsed document.
        translator: This project's translator.
        workers (int): How many pieces to translate at once.
        ignore_cache (bool): Passed through to the translator.
        glossary (Callable | None): The project's glossary matcher, mapping a
            block of text to the terms it contains. Applied here rather than
            inside the translator because the LaTeX path calls the plain
            ``translate`` entry point, which the cloud backend's own glossary
            handling never reaches.
        hard_protect (bool): Hold whitelisted terms behind markers in running
            prose as well, not only in tables and captions.
        on_progress (Callable | None): Called with ``(done, total)``.

    Returns:
        dict: ``total``, ``translated``, ``partial`` and ``refused``, plus
            ``reasons`` counting why pieces were refused. ``partial`` counts
            blocks that were cut up and came back with some chunks translated
            and some not. The counts are the point: a run
            that refused half its paragraphs produced a paper in English and
            should say so rather than look like a success.
    """
    pending = [piece for piece in document.pieces if piece.translate]
    total = len(pending)
    if not total:
        return {"total": 0, "translated": 0, "refused": 0, "reasons": {}}

    # The PDF pipeline's prompt tells the model to return the original when it
    # judges translation unnecessary, and LaTeX is exactly what it judges that
    # way. Swapped before the first call rather than at construction, so a
    # translator built for anything else is unaffected.
    # PDF 管线的提示词让模型在自认为不需要翻译时原样返回，而 LaTeX 正是它会这么判断的
    # 东西。在第一次调用之前换掉，而不是在构造时换，这样为别的用途造的翻译器不受影响。
    use_latex_prompt(translator)

    # The paper's own macros, collected from the definitions the parser lifted.
    # 论文自己定义的宏，取自解析器抽走的那些定义。
    macros = macro_pattern(piece.name for piece in document.pieces
                           if piece.kind == "macro")

    reasons: dict[str, int] = {}
    partial = 0
    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = [pool.submit(_translate_one, piece, translator, ignore_cache,
                               macros, glossary, hard_protect)
                   for piece in pending]
        for future in concurrent.futures.as_completed(futures):
            piece, complaint = future.result()
            done += 1
            if complaint and piece.translated:
                # Part of the block came back and part did not. Counted apart
                # from a refusal, which keeps `translated + refused == total`.
                # 这一块回来了一部分、另一部分没有。与拒收分开计数，好让
                # 「采用 + 拒收 = 总数」这条恒等式仍然成立。
                partial += 1
            if complaint:
                key = regex.sub(r"\s*[0-9]+\s*", " N ", complaint)
                reasons[key] = reasons.get(key, 0) + 1
                logger.warning("拒收一块译文（%s/%s）：%s", piece.kind, piece.name,
                               complaint)
            if on_progress:
                on_progress(done, total)

    translated = sum(1 for piece in pending if piece.translated)
    logger.info("翻译完成：%d/%d 块被采用，%d 块保留原文",
                translated, total, total - translated)
    return {"total": total, "translated": translated, "partial": partial,
            "refused": total - translated, "reasons": reasons}
