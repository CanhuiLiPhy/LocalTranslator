r"""
中文：匹配 LaTeX 结构的正则积木，改编自 LaTeXTrans。

Regular expressions for matching LaTeX structure.

Adapted from LaTeXTrans (NiuTrans, MIT licence), whose ``src/formats/latex``
worked out which constructs have to be recognised and in what order. The
expressions here are rewritten against this project's needs but the shape of
the problem — and the recursive brace pattern that solves it — is theirs.

**Why the ``regex`` module and not ``re``.** LaTeX braces nest, and a
``\\caption{a \\textbf{b} c}`` cannot be matched by any expression that treats
``}`` as the end. ``regex`` supports recursive subpatterns — ``(?3)`` re-enters
group 3 — which matches a balanced group in one pass. The standard library has
no equivalent, and hand-rolling a brace counter for every construct would be a
worse copy of the same thing.
"""

from __future__ import annotations

import regex

# Optional argument in square brackets, as in ``\caption[short]{long}``. Kept
# non-greedy and free of nested brackets: an optional argument that itself
# contains brackets is rare enough that matching it would cost more in false
# positives than it saves.
# 方括号里的可选参数，如 \caption[short]{long}。写成非贪婪且不含嵌套方括号：可选参数
# 内部再带方括号的情况少见，为它放宽匹配带来的误判多于收益。
OPTIONS = r"\[[^\[\]]*?\]"

# Horizontal space only. A LaTeX command may be separated from its argument by
# spaces and tabs but not by a blank line, which ends the paragraph.
# 只匹配水平空白。LaTeX 命令与其参数之间可以有空格和制表符，但不能有空行——空行意味着
# 段落结束。
SPACES = r"[ \t]*"


def braced(group: int) -> str:
    """
    中文：匹配一对配平的花括号，允许任意层嵌套。

    Match one balanced brace group, however deeply nested.

    Args:
        group (int): The group number this pattern will occupy in the final
            expression. The recursion refers to itself by number, so the caller
            has to know where it sits.

    Returns:
        str: A pattern matching ``{…}`` with the contents captured. Written
            with possessive quantifiers so a failure to close backs out rather
            than retrying every split of the input.
    """
    return rf"\{{((?:[^{{}}]++|(?{group}))*+)\}}"


def command(names: str, arguments: int | None = None) -> regex.Pattern:
    """
    中文：匹配某个 LaTeX 命令连同它的参数。

    Match a LaTeX command together with its arguments.

    Args:
        names (str): Alternation of command names without the backslash, as in
            ``caption|caption\\*``.
        arguments (int | None): How many braced arguments the command takes.
            ``None`` means one, preceded by an optional bracketed argument.

    Returns:
        regex.Pattern: Compiled, with ``DOTALL`` so an argument may span lines.
            Group 1 is the command name; the braced arguments follow.
    """
    pattern = rf"\\({names})"
    if arguments is None:
        pattern += rf"{SPACES}({OPTIONS})?"
        arguments, first = 1, 3
    else:
        first = 2
    for index in range(arguments):
        pattern += rf"{SPACES}({braced(index * 2 + first)})"
    if arguments == 0:
        pattern += r"(?=[^a-zA-Z])"
    return regex.compile(pattern, regex.DOTALL)


def environment(names: str) -> regex.Pattern:
    """
    中文：匹配 \\begin{名字}…\\end{名字}。

    Match ``\\begin{name}…\\end{name}``.

    Args:
        names (str): Alternation of environment names, or ``.*?`` for any.

    Returns:
        regex.Pattern: Group 1 is the environment's name, group 3 its body.

    Note:
        ``document``, ``center``, ``proof`` and ``multicols`` are excluded.
        They wrap prose rather than replacing it — lifting a ``center`` out of
        the text would take the paragraph inside it out of the translation, and
        lifting ``document`` would take the whole paper.
        排除 document、center、proof、multicols。它们包裹正文而不是替代正文——把
        center 整个抽走会连同里面的段落一起退出翻译，抽走 document 则是整篇论文。
    """
    return regex.compile(
        rf"\\begin{SPACES}\{{(?!document\b|center\b|proof\b|multicols\b)({names})\}}"
        rf"{SPACES}({OPTIONS})?(.*?)\\end{SPACES}\{{\1\}}",
        regex.DOTALL,
    )


def innermost_environment(names: str) -> regex.Pattern:
    """
    中文：只匹配「里面没有再嵌套环境」的那一层 \\begin{名字}…\\end{名字}。

    Match an environment whose body contains no environment of its own.

    Args:
        names (str): Alternation of environment names, or ``.*?`` for any.

    Returns:
        regex.Pattern: Same groups as :func:`environment`.

    Note:
        Why the innermost one first. A ``table`` float holds a ``tabular``, and
        the cells a reader wants translated are in the ``tabular``. Matching the
        float first takes the whole thing out in one piece and the ``tabular``
        inside it is never seen again — which is why every table in the output
        was still in English. Lifting the innermost environment first leaves a
        placeholder behind, so the float becomes innermost in its turn and is
        lifted after what it contains. Assembly resolves in reverse order, so
        "contained is lifted earlier" is exactly the order it needs.
        为什么先取最里层。一个 table 浮动体里装着 tabular，而读者想要翻译的单元格在
        tabular 里。先匹配浮动体就把整块一次性抽走，里面的 tabular 再也见不到了——这正是
        产物里每张表都还是英文的原因。先抽最里层的环境，原地留下占位符，浮动体于是轮到
        自己成为最里层、在它所包含的东西之后被抽走。装配是倒序解析的，「被包含者先抽」
        恰好就是它需要的顺序。
    """
    return regex.compile(
        rf"\\begin{SPACES}\{{(?!document\b|center\b|proof\b|multicols\b)({names})\}}"
        rf"{SPACES}({OPTIONS})?((?:(?!\\begin{SPACES}\{{).)*?)\\end{SPACES}\{{\1\}}",
        regex.DOTALL,
    )


def new_command() -> regex.Pattern:
    """
    中文：匹配用户自定义的宏与环境定义。

    Match a user's macro or environment definition.

    Returns:
        regex.Pattern: Group 1 or 2 is the name, group 3 the argument count.
            Definitions are lifted out whole and put back untouched: their
            bodies are code, and a translated ``\\newcommand`` breaks every
            place that uses it.
    """
    return regex.compile(
        rf"\\(?:newcommand\*?|def|renewcommand|newenvironment|renewenvironment)"
        rf"{SPACES}(?:\{{\\([a-zA-Z]+)\}}|\\([a-zA-Z]+)){SPACES}(?:\[(\d)\])?"
        rf"{SPACES}({braced(4)})",
        regex.DOTALL,
    )


BEGIN_DOCUMENT = regex.compile(r"\\begin\s*\{\s*document\s*\}")
END_DOCUMENT = regex.compile(r"\\end\s*\{\s*document\s*\}")
DOCUMENT_CLASS = regex.compile(rf"\\documentclass{SPACES}({OPTIONS})?{SPACES}\{{[^}}]*\}}")


def strip_comments(tex: str) -> str:
    """
    中文：去掉 % 行注释与 comment 环境，保留转义的 \\%。

    Remove ``%`` comments and ``comment`` environments.

    Args:
        tex (str): LaTeX source.

    Returns:
        str: The same source with comments gone. An escaped ``\\%`` is left
            alone — it is a per-cent sign in the text, not a comment.
    """
    tex = regex.sub(r"\\begin\s*\{comment\}.*?\\end\s*\{comment\}", "", tex,
                    flags=regex.DOTALL)
    kept = []
    for line in tex.splitlines():
        if regex.match(r"^\s*%", line):
            continue
        kept.append(regex.sub(r"(?<!\\)%.*", "", line).rstrip())
    return "\n".join(kept)
