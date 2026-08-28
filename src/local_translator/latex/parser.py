r"""
中文：把一棵 LaTeX 源码树拆成「要翻译的文字」和「原样保留的结构」，译完再装回去。

Take a LaTeX source tree apart into text to translate and structure to keep.

Adapted from LaTeXTrans (NiuTrans, MIT licence). What is borrowed is the order
of operations, which is where the difficulty lies and which that project had
already worked out:

    comments → \\input → \\newcommand → \\caption → environments → body

Every step but the last lifts something out and leaves a placeholder. The order
is not arbitrary. Captions come out **before** environments so that a figure
whose caption is already a placeholder can be lifted whole and never
translated, while its caption travels separately and is. Reverse those two and
a figure's caption is either lost inside a protected block or the whole figure
is sent to the model.

**What is not borrowed.** LaTeXTrans carries its own LLM client, prompts,
agents and progress bars. This project already has a translator with three
backends, a glossary and selective translation, and routing LaTeX segments
through it is the point of merging rather than vendoring.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import regex

from local_translator.latex import patterns
from local_translator.latex.selection import Selection

logger = logging.getLogger(__name__)

# Which region of a paper each environment belongs to, named with the labels the
# settings page uses. One vocabulary for both paths: a reader who switched
# 「表格」 off did so about their papers, not about a file format.
# 每个环境属于论文的哪个区域，用设置页那套标签命名。两条路共用一套词汇：读者关掉
# 「表格」，关的是自己论文里的表格，而不是某一种文件格式里的表格。
_ENVIRONMENT_LABELS: dict[str, str] = {
    # Display mathematics. Never sent whatever the setting says — it is
    # mathematics, and it comes through TeX unchanged, which is the whole
    # reason to translate from source.
    # 行间公式。无论设置如何都不送——那是数学，经 TeX 原样通过，而这正是从源码翻译的
    # 全部理由。
    **{name: "isolate_formula" for name in (
        "equation", "equation*", "align", "align*", "alignat", "alignat*",
        "gather", "gather*", "flalign", "flalign*", "multline", "multline*",
        "split", "split*", "cases", "cases*", "subequations", "eqnarray",
        "eqnarray*", "math", "displaymath", "IEEEeqnarray", "IEEEeqnarray*",
        # The inner halves of the same package. These only ever appear inside
        # one of the above, which is why they were missed: while environments
        # were lifted outermost first, an ``equation`` was taken whole and the
        # ``aligned`` inside it was never seen on its own. Lifting innermost
        # first — which is what makes a table's cells reachable — brought them
        # into view, and an unnamed environment defaults to prose.
        # 同一批宏包的「内层」环境。它们只会出现在上面那些环境里面，所以此前一直没露过面：
        # 环境由外向内抽取时，equation 是整块拿走的，里面的 aligned 根本不会单独出现。改成
        # 由内向外之后——正是这一改让表格的单元格够得着——它们露了出来，而没有名字的环境
        # 默认按散文处理。
        "aligned", "alignedat", "gathered", "dcases", "dcases*", "rcases",
        "drcases", "matrix", "matrix*", "pmatrix", "pmatrix*", "bmatrix",
        "bmatrix*", "Bmatrix", "Bmatrix*", "vmatrix", "vmatrix*", "Vmatrix",
        "Vmatrix*", "smallmatrix", "psmallmatrix", "bsmallmatrix",
        "subarray", "numcases", "subnumcases", "empheq", "multlined",
        "IEEEeqnarraybox", "IEEEeqnarraybox*", "array", "array*",
    )},
    **{name: "figure" for name in (
        "figure", "figure*", "wrapfigure", "SCfigure", "subfigure",
        "subfigure*", "tikzpicture",
    )},
    **{name: "table" for name in (
        "table", "table*", "tabular", "tabular*", "tabularx", "tabulary",
        "longtable", "longtable*", "sidewaystable", "sidewaystable*",
        "threeparttable", "wraptable",
    )},
    "tablenotes": "table_footnote",
    "thebibliography": "reference",
    # Pseudocode. Whether it is worth translating depends on which package
    # wrote it, and both answers live in the same label:
    #
    # ``algpseudocode`` builds each line from a macro — ``\State``, ``\If{…}``,
    # ``\Comment{…}`` — so the indentation a reader sees comes from the control
    # structure and not from the whitespace in the file. The formulas are
    # already behind ``$…$``. What is left is comments and the English half of
    # each statement, which is exactly what a Chinese reader wants translated.
    #
    # ``algorithm2e`` writes denser lines: ``\lFor{…}{…}`` with the body inline,
    # ``\;`` ending each statement, and often more mathematics than words. It
    # still works, but there is less to gain and more to lose, which is why the
    # switch is off unless a reader turns it on.
    # 伪代码。值不值得翻译，取决于是哪个宏包写的，而两种答案共用同一个标签：
    #
    # algpseudocode 的每一行都由宏构成——\State、\If{…}、\Comment{…}——所以读者看到的
    # 缩进来自控制结构，而不是文件里的空白。公式本来就在 $…$ 里。剩下的是注释和每条语句
    # 里英文的那一半，而这正是中文读者想看到译文的部分。
    #
    # algorithm2e 写得更密：\lFor{…}{…} 把循环体写在同一行，\; 结束每条语句，而且往往
    # 公式比文字多。它同样能处理，只是可得的少、可失的多——这就是这个开关默认不开的理由。
    **{name: "algorithm" for name in (
        "algorithm", "algorithm*", "algorithmic", "algorithmic*", "algorithm2e",
        "algo", "algorithmx", "pseudocode", "procedure",
    )},
}

# Environments whose body is code or data. Never prose, whatever the settings
# say, so they carry no label at all.
# 内容是代码或数据的环境。无论设置如何都不是散文，因此干脆不给标签。
_CODE_ENVIRONMENTS = frozenset({
    "verbatim", "verbatim*", "lstlisting", "lstlisting*", "minted", "minted*",
    "tcolorbox", "CJK", "scope", "filecontents", "filecontents*",
})

# How many braced arguments an environment takes before its body starts. A
# ``tabular``'s column specification is an argument, not content: handed
# ``{cc}a & b`` the model rewrites the specification and the table stops being a
# table.
# 环境在正文开始之前带几个花括号参数。tabular 的列格式是参数而不是内容：把
# {cc}a & b 交给模型，它会改写列格式，表格就不再是表格了。
_ENVIRONMENT_ARGUMENTS: dict[str, int] = {
    "tabular": 1, "longtable": 1, "longtable*": 1, "array": 1,
    "thebibliography": 1, "multicols": 1, "minipage": 1,
    "tabular*": 2, "tabularx": 2, "tabulary": 2,
}

# Commands whose braced argument is prose, and which region that prose belongs
# to. Applied in this order, innermost construct first: a ``\thanks`` inside an
# ``\author`` has to be lifted before the block that contains it, because
# assembly resolves placeholders in reverse.
# 花括号里是散文的命令，以及那段散文属于哪个区域。按这个顺序处理，越靠内的越先：
# \author 里面的 \thanks 必须先于包着它的块被抽走，因为装配是倒序解析占位符的。
_COMMAND_PASSES: tuple[tuple[str, str], ...] = (
    (r"footnote|footnotetext|thanks", "footnote"),
    (r"author|authors|icmlauthor|IEEEauthorblockN", "author"),
    (r"affiliation|affil|institute|institution|address|email"
     r"|icmlaffiliation|icmlcorrespondingauthor|IEEEauthorblockA", "affiliation"),
    (r"title|subtitle|icmltitle|icmltitlerunning", "document_title"),
    (r"keywords|IEEEkeywords", "plain text"),
    (r"caption|caption\*|subcaption|subcaption\*", "caption"),
    (r"section|section\*|subsection|subsection\*|subsubsection|subsubsection\*"
     r"|paragraph|paragraph\*|subparagraph|chapter|chapter\*", "title"),
)

# Commands whose argument is a file, a key or a package name. Never prose, and
# lifted before anything else can claim them.
# 参数是文件名、键或宏包名的命令。从来不是散文，先于其它一切抽走。
_PROTECTED_COMMANDS = (
    r"bibliography|bibliographystyle|includegraphics|input"
    r"|include|usepackage|documentclass|label|newcommand|def"
)

# Labels whose content is actually sent to a model. ``figure`` gates a
# different action entirely — rewriting the image file — and display
# mathematics is never sent at all.
# 内容真的会送到模型手里的那些标签。figure 管的完全是另一件事——重写图片文件——而行间
# 公式根本不送。
_SENDABLE: frozenset[str] = frozenset({
    "title", "plain text", "figure_caption", "table", "table_caption",
    "table_footnote", "document_title", "author", "affiliation", "footnote",
    "reference", "algorithm",
})

_BEGIN_OR_END = regex.compile(r"\\(begin|end)\s*\{([A-Za-z@*]+)\}")

# A sub-figure caption, which several classes put in the *optional* argument
# rather than a braced one: ``\subfloat[Scheme with the original repository.]``.
# The ordinary command pass reads braced arguments, so these went straight
# past it and every sub-caption in an IEEEtran paper stayed in English while
# the figure's own caption was translated.
# 子图的图注，有几个文档类把它放在**可选**参数里而不是花括号里：
# \subfloat[Scheme with the original repository.]。普通的命令那一轮读的是花括号参数，
# 于是这些从旁边溜了过去——IEEEtran 论文里每一条子图注都留在英文，而整张图的图注却翻
# 好了。
_BRACKET_CAPTION = regex.compile(
    r"\\(subfloat|subcaptionbox|subbottom|subfigure|subtable)\s*"
    r"\[((?:[^\[\]]|\[[^\]]*\])*)\]")

# Environments that carry prose and are declared *before* ``\begin{document}``.
# The abstract is the one that matters: several classes — googledeepmind among
# them — have the author write ``\begin{abstract}…\end{abstract}`` in the
# preamble, where ``\maketitle`` later prints it. A parser that only looks for
# environments in the body never sees it, and the paper ships with an English
# abstract on page one while everything after it is Chinese.
#
# A whitelist rather than "every environment in the preamble": a preamble also
# holds ``filecontents`` and package-definition blocks, whose bodies are code.
# 带散文、而且声明在 \begin{document} **之前**的环境。要紧的是摘要：有几个文档类——
# googledeepmind 就是一个——让作者把 \begin{abstract}…\end{abstract} 写在前言里，
# 之后由 \maketitle 印出来。只在正文里找环境的解析器根本看不见它，于是论文第一页的
# 摘要还是英文，后面却全是中文。
#
# 用白名单而不是「前言里所有环境」：前言里还有 filecontents 之类，正文是代码。
_PREAMBLE_PROSE = r"abstract\*?|keywords|IEEEkeywords|abstracttext"

# The drop-cap that opens the first paragraph of a section in several journal
# classes. Its two arguments are one word cut in half — ``{A}{ccording}`` — and
# handing that to a model asks it to reassemble the word before translating it,
# which it does by dropping the command and translating the sentence. That is
# not wrong, but it is not a decision a model should be making: the paragraph
# silently loses the paper's opening capital.
# 几个期刊文档类里用来开头的首字下沉命令。它的两个参数是一个词被劈成两半——
# {A}{ccording}——把这个交给模型，等于让它先把词拼回去再翻译，而它的做法是丢掉命令、
# 翻译整句。这不算错，但不该由模型来定：那一段就这么悄悄丢了论文的开篇大写字母。
_DROPCAP = regex.compile(
    r"^\s*\\(IEEEPARstart|lettrine)\s*(?:\[[^\]]*\])?\s*\{([^{}]*)\}\s*\{([^{}]*)\}")

# A character that can stand as a dropped capital: a Chinese character or a
# letter. A translation opening with a backslash or a placeholder gets no drop
# cap, because what would be enlarged is the first character of a command.
# 能当首字下沉的字符：汉字或字母。以反斜杠或占位符开头的译文不加首字下沉，否则被放大
# 的会是一个命令的第一个字符。
_CAPITALISABLE = regex.compile(r"[一-鿿A-Za-z]")

# What a lifted block is replaced by. Short, so it costs few tokens; distinctive,
# so it cannot collide with prose; and plain ASCII, because a model asked to
# preserve an exotic character often does not.
# 抽走的块用什么替代。要短，省 token；要独特，不能与正文碰撞；要纯 ASCII，因为让模型
# 原样保留生僻字符，它常常做不到。
_MARK = "<LTX{}>"


@dataclass
class Piece:
    r"""
    中文：从源码里抽出来的一块，可能要翻译，也可能原样放回。

    One block lifted out of the source.

    Attributes:
        mark (str): The placeholder left in its place.
        kind (str): ``env``, ``prose``, ``macro`` or ``input``.
        name (str): The environment or command name, for diagnosis.
        source (str): The LaTeX exactly as it was.
        prefix (str): Scaffolding before the text, put back untouched.
        core (str): The text sent to the model, and the only part sent.
        suffix (str): Scaffolding after the text, put back untouched.
        label (str): Which region of the paper this is, named the way the
            settings page names it — ``table``, ``figure_caption``, ``author``.
            Empty for a block that is never prose, such as a macro definition.
            This is what decides whether the block is translated: the parser
            says what a block *is*, and the reader's settings say what to do
            with it.
        translate (bool): Whether the block's text should reach the model.
        translated (str): What came back for ``core``, empty until it does.
        dropcap (str): Name of the drop-cap command this paragraph opened with.
            The translation's first character is enlarged in its place. Empty
            for the paragraphs that had none, which is nearly all of them.

    Note:
        **Only ``core`` reaches the model**, and that is the difference between
        a translation that comes back usable and one that does not. Measured on
        the IEEEtran paper before the split existed, 13% of the body was
        refused, almost all of it for two reasons that are the same reason.
        Handed ``<LTX160>\n<LTX92>\nWe evaluate the proposed…`` the model
        returns the Chinese and drops the leading marks — they are noise to it.
        Handed ``\begin{abstract}…\end{abstract}`` it translates the
        environment's *name*, returning ``\begin{}摘要}``. Neither is the model
        behaving badly; both follow from showing it something it was told not
        to change. What it never sees, it cannot break.
        **只有 core 会到模型手里**，而这正是译文能不能用的分界。在 IEEEtran 那篇上，
        没有这个切分时 13% 的正文被拒收，几乎全部出自两个说到底是同一个的原因。给它
        ``<LTX160>\n<LTX92>\nWe evaluate the proposed…``，它返回中文而丢掉开头那些
        记号——很合理，那看起来就是噪音。给它 ``\begin{abstract}…\end{abstract}``，
        它把环境**名字**也翻了，返回 ``\begin{}摘要}``。这都不是模型在乱来；两者都源于
        把不许它改的东西摆到了它面前。它看不见的东西，它就弄不坏。
    """

    mark: str
    kind: str
    name: str
    source: str
    label: str = ""
    prefix: str = ""
    core: str = ""
    suffix: str = ""
    translate: bool = False
    translated: str = ""
    dropcap: str = ""

    def rendered(self) -> str:
        """中文：装回去时用哪一份。 / Which version goes back into the document."""
        if not (self.translate and self.translated):
            return self.source
        body = self.translated
        if self.dropcap and _CAPITALISABLE.match(body[:1]):
            # Enlarged, not dropped. ``\IEEEPARstart`` lowers its letter by two
            # lines and reserves the space with a \parshape written for a Latin
            # capital; handed 「根」 it lowers the character but reserves nothing,
            # so the character lands at the start of the *second* line and the
            # paragraph reads 「据…未来的 / 根 无线通信系统」. Rendered and compared
            # side by side: enlarging the first character keeps the visual cue
            # the class was after and stays readable.
            # 是放大，不是下沉。\IEEEPARstart 把字母下移两行，并用一段为拉丁大写字母写的
            # \parshape 预留位置；给它「根」，它把字下移了却什么也没预留，于是这个字落到
            # 了**第二行**开头，整段读起来成了「据……未来的 / 根 无线通信系统」。三种写法
            # 并排编译对照过：放大首字既保留了文档类想要的视觉提示，又不影响阅读。
            body = f"{{\\Large {body[0]}}}{body[1:]}"
        return self.prefix + body + self.suffix


@dataclass
class LatexDocument:
    """
    中文：一篇拆开的 LaTeX 论文。

    A LaTeX paper taken apart.

    Attributes:
        root (Path): The directory the sources were extracted into.
        main (Path): The main ``.tex`` file, the one carrying
            ``\\documentclass``.
        preamble (str): Everything before ``\\begin{document}``.
        body (str): The document body with every lifted block replaced by its
            placeholder.
        pieces (list[Piece]): The lifted blocks, in the order they were taken.
        selection (Selection): Which regions the reader asked to translate.
    """

    root: Path
    main: Path
    preamble: str = ""
    body: str = ""
    tail: str = ""
    pieces: list[Piece] = field(default_factory=list)
    selection: Selection = field(default_factory=Selection)
    _count: int = 0

    def _next_mark(self) -> str:
        """中文：取下一个占位符。 / Take the next placeholder."""
        self._count += 1
        return _MARK.format(self._count)

    def _lift(self, text: str, source: str, kind: str, name: str,
              label: str = "", span: tuple[int, int] | None = None) -> str:
        """
        中文：把一块抽出来，原地留下占位符；译不译由标签和读者的设置决定。

        Lift one block out and leave a placeholder in its place.

        Args:
            text (str): The document as it stands.
            source (str): The block to lift, verbatim.
            kind (str): What sort of block it is.
            name (str): Its environment or command name.
            label (str): Which region of the paper this is. Empty means the
                block is never prose and is never translated.
            span (tuple[int, int] | None): Offsets into ``source`` bounding the
                part that may be sent. The rest is scaffolding.

        Returns:
            str: The document with the block replaced by its placeholder.
        """
        mark = self._next_mark()
        start, end = span if span else (0, len(source))
        prefix, core, suffix = source[:start], source[start:end], source[end:]
        translate = label in _SENDABLE and self.selection.wants(label)
        if translate:
            prefix, core, suffix = _peel_marks(prefix, core, suffix)
            if not regex.search(r"[A-Za-z]{2}", core):
                # Nothing left to translate once the scaffolding is off. Sending
                # it anyway spends a call to get the same string back, and the
                # model sometimes returns something else.
                # 脚手架去掉之后没剩下可翻的东西了。照送不误只会花一次调用换回同一个
                # 字符串，而且模型有时还会给出别的东西。
                translate = False
        self.pieces.append(Piece(mark=mark, kind=kind, name=name, source=source,
                                 label=label, prefix=prefix, core=core,
                                 suffix=suffix, translate=translate))
        return text.replace(source, mark, 1)


def assemble(document: LatexDocument) -> str:
    """
    中文：把译文装回骨架，还原成一份完整的 .tex。

    Put the translations back and rebuild the document.

    Args:
        document (LatexDocument): A parsed document whose pieces may carry
            translations.

    Returns:
        str: Complete LaTeX source. Placeholders are resolved innermost-first,
            because a lifted block can contain the placeholder of a block
            lifted before it — a figure holds its caption's mark — and
            resolving outermost-first would leave that mark in the output.

    Raises:
        ValueError: If any placeholder is still present after every piece has
            been substituted, which means a block was lost rather than put
            back. Failing here is the point: a silently dropped placeholder is
            a paragraph missing from the paper.
    """
    text = document.preamble + document.body + document.tail

    # Later pieces were lifted from earlier ones, so resolving in reverse order
    # expands the outer block first and then what was inside it.
    # 后抽出的块来自先抽出的块内部，因此倒序解析：先展开外层，再展开里面的东西。
    for piece in reversed(document.pieces):
        text = text.replace(piece.mark, piece.rendered())

    leftover = regex.findall(r"<LTX\d+>", text)
    if leftover:
        missing = ", ".join(sorted(set(leftover))[:6])
        raise ValueError(f"装配后仍有 {len(leftover)} 个占位符未还原：{missing}")
    return text


def _peel_marks(prefix: str, core: str, suffix: str) -> tuple[str, str, str]:
    r"""
    中文：把 core 头尾单独成行的占位符移到 prefix/suffix，别让模型看见。

    Move placeholders at the edges of the text out of the model's sight.

    A paragraph often opens with the marks of the section heading and the label
    that preceded it. They carry no meaning for a translator, and a model handed
    ``<LTX160>\n<LTX92>\nWe evaluate…`` returns the Chinese without them —
    reasonably, since they look like noise. The block is then refused for
    losing them, and a correct translation is thrown away.

    Only edge marks are peeled. One in the middle of a sentence stands for a
    formula or a citation that belongs where it is, and the model has to place
    it.

    Args:
        prefix (str): Scaffolding before the text.
        core (str): The text.
        suffix (str): Scaffolding after the text.

    Returns:
        tuple[str, str, str]: The three parts with edge marks moved outwards.
    """
    leading = regex.match(r"^(?:\s*<LTX\d+>)+\s*", core)
    if leading:
        prefix += leading.group(0)
        core = core[leading.end():]
    trailing = regex.search(r"\s*(?:<LTX\d+>\s*)+$", core)
    if trailing:
        suffix = trailing.group(0) + suffix
        core = core[:trailing.start()]
    return prefix, core, suffix


def _read(path: Path) -> str:
    """中文：按 UTF-8 读，坏字节不致命。 / Read as UTF-8, tolerating bad bytes."""
    return path.read_text(encoding="utf-8", errors="replace")


def _merge_inputs(text: str, root: Path, depth: int = 0) -> str:
    """
    中文：把 \\input 与 \\include 的内容就地展开，最多递归若干层。

    Inline every ``\\input`` and ``\\include``, recursively.

    Args:
        text (str): The document so far.
        root (Path): Directory that relative paths resolve against.
        depth (int): Current recursion depth.

    Returns:
        str: The document with included files spliced in. A file that cannot be
            found is left as the original command, which is what a reader would
            want: the compile then reports the missing file by name rather than
            silently dropping a chapter.

    Note:
        Bounded at eight levels. A paper that includes itself would otherwise
        recurse until memory runs out, and no real paper nests that deep.
        限制八层。互相 include 的论文否则会一直递归到内存耗尽，而真实论文不会嵌这么深。
    """
    if depth > 8:
        logger.warning("Stopped inlining \\input at depth %d", depth)
        return text

    pattern = patterns.command(r"input|include")
    result = ""
    position = 0
    while True:
        match = pattern.search(text, position)
        if match is None:
            result += text[position:]
            break
        # Group 3 is the braced group *including* its braces; group 4 is what is
        # inside. Reading group 3 asks the filesystem for "{chapter.tex}", which
        # never exists, so every \input was silently left un-inlined — and a
        # paper that keeps each section in its own file lost its entire body.
        # 第 3 组是**带花括号**的整体，第 4 组才是里面的内容。取第 3 组等于拿
        # 「{chapter.tex}」去问文件系统，永远不存在，于是每个 \input 都被悄悄跳过——
        # 而把每一节单独存一个文件的论文，正文就整个丢了。
        name = (match.group(4) or "").strip()
        if not name:
            result += match.group(0)
            position = match.end()
            continue
        candidate = root / name
        if not candidate.exists():
            candidate = root / f"{name}.tex"
        result += text[position:match.start()]
        if candidate.is_file():
            inner = patterns.strip_comments(_read(candidate))
            result += _merge_inputs(inner, root, depth + 1)
        else:
            logger.warning("\\input names a file that is not here: %s", name)
            result += match.group(0)
        position = match.end()
    return result


def parse(root: Path, main: Path, *, selection: Selection | None = None,
          bibliography: str = "") -> LatexDocument:
    """
    中文：读入一篇论文并拆成可翻译片段与受保护块。

    Read a paper and take it apart.

    Args:
        root (Path): Directory holding the extracted sources.
        main (Path): The main ``.tex`` file.
        selection (Selection | None): Which regions to translate. The default
            is the one the ``pdfmath`` preset describes.
        bibliography (str): Contents of the paper's ``.bbl``, spliced in where
            ``\\bibliography{…}`` stands. Passed in rather than read here so the
            entries take part in the ordinary passes: a reader who asked for
            translated references gets each ``\\bibitem`` as its own block, and
            one who did not gets the environment lifted whole, exactly as
            before.

    Returns:
        LatexDocument: With ``preamble``, ``body`` and ``pieces`` filled in.
            The body is what a translator sees: prose with placeholders where
            the structure was.
    """
    document = LatexDocument(root=root, main=main,
                             selection=selection or Selection())
    text = _merge_inputs(patterns.strip_comments(_read(main)), root)
    if bibliography:
        text = _splice_bibliography(text, bibliography)

    opening = patterns.BEGIN_DOCUMENT.search(text)
    closing = patterns.END_DOCUMENT.search(text)
    if opening is None:
        # No \begin{document} means this is not a main file. Treating the whole
        # thing as body is the reading that loses nothing.
        # 没有 \begin{document} 说明这不是主文件。整篇当正文处理，这样不会丢东西。
        document.preamble, rest, document.tail = "", text, ""
    else:
        document.preamble = text[:opening.end()]
        end = closing.start() if closing else len(text)
        rest = text[opening.end():end]
        document.tail = text[end:]

    # The preamble is not inert. A paper's title and its abstract are declared
    # there, before \begin{document}, and \maketitle only prints what was
    # already set — so a parser that starts at the body translates the whole
    # paper and leaves its title in English. The preamble gets the same macro
    # and command treatment as the body; only paragraphs are skipped, because
    # it has none.
    # 前言不是死的。论文的标题和摘要就声明在那里、在 \begin{document} 之前，而
    # \maketitle 只是把已经设好的东西印出来——所以从正文开始解析的话，整篇都译了，
    # 标题却还是英文。前言与正文一样处理宏和命令；只跳过段落，因为它没有段落。
    document.preamble = _lift_commands(document, document.preamble)

    # Macros first: their bodies are code, and a definition that reaches the
    # model comes back broken for every place that uses it.
    # 先处理宏：宏体是代码，定义一旦进了模型，所有用到它的地方都会跟着坏掉。
    macro = patterns.new_command()
    while (match := macro.search(rest)) is not None:
        rest = document._lift(rest, match.group(0), "macro",
                              match.group(1) or match.group(2) or "?")

    # File references and package names are lifted before anything else can
    # claim them, and never translated.
    # 文件引用与宏包名先于其他一切抽走，且从不翻译。
    protected = patterns.command(_PROTECTED_COMMANDS)
    while (match := protected.search(rest)) is not None:
        rest = document._lift(rest, match.group(0), "keep", match.group(1))

    rest = _lift_labelled_commands(document, rest)
    rest = _lift_environments(document, rest)

    document.body = _lift_paragraphs(document, rest)
    logger.info("拆出 %d 块（%d 块需要翻译），正文模板 %d 字；%s",
                len(document.pieces),
                sum(1 for p in document.pieces if p.translate), len(document.body),
                document.selection.describe())
    return document


_BIBLIOGRAPHY_CALL = regex.compile(r"\\bibliography\s*\{[^}]*\}")


def _splice_bibliography(text: str, entries: str) -> str:
    r"""
    中文：把 .bbl 的内容放到 \bibliography{…} 的位置上。

    Put the compiled bibliography where its command stands.

    Args:
        text (str): The document, with ``\input`` already inlined.
        entries (str): Contents of the ``.bbl``.

    Returns:
        str: The document with the entries spliced in, or unchanged when there
            is no ``\bibliography`` command to replace — the paper is then
            already carrying its own ``thebibliography``.

    Note:
        This runs before parsing rather than after assembly, which is where it
        used to. The reason is the reference switch: spliced in first, the
        entries are an ordinary ``thebibliography`` environment that the parser
        labels and the settings govern. Spliced in afterwards they arrive when
        every decision has already been made.
        这一步现在放在解析之前，而不是像以前那样放在装配之后。原因是参考文献开关：先
        拼进来，这些条目就是一个普通的 thebibliography 环境，解析器会给它贴标签、设置
        说了算。拼在后面的话，它们到场时每个决定都已经做完了。
    """
    if not _BIBLIOGRAPHY_CALL.search(text):
        return text
    # A .bbl opens with the style file's own comment banner. Comments were
    # stripped from the main file before this point, so these arrive after the
    # only pass that would have removed them — and the banner then reads as a
    # paragraph and gets sent to a model to translate.
    # .bbl 开头带着样式文件自己的注释横幅。主文件的注释在此之前就已经去掉了，这些却是在
    # 唯一会去掉它们的那一步之后才到场——于是横幅被当成一个段落，送去给模型翻译。
    entries = patterns.strip_comments(entries)
    logger.info("把 %d 字参考文献嵌入文档", len(entries))
    return _BIBLIOGRAPHY_CALL.sub(lambda _: entries, text, count=1)


def _caption_label(text: str, position: int) -> str:
    r"""
    中文：判断这条 \caption 属于图还是表。

    Decide whether a caption belongs to a figure or to a table.

    Args:
        text (str): The document as it stands.
        position (int): Where the caption starts.

    Returns:
        str: ``"table_caption"`` or ``"figure_caption"``.

    Note:
        Captions are lifted before environments, so the floats are still intact
        here and the innermost unclosed one is the caption's owner. A caption
        outside any float is called a figure caption, which is what it almost
        always is.
        图注在环境之前抽取，所以此时浮动体还是完整的，最里层尚未闭合的那个就是这条图注
        的归属。不在任何浮动体里的图注归为图注，因为它几乎总是。
    """
    open_names: list[str] = []
    for match in _BEGIN_OR_END.finditer(text, 0, position):
        name = match.group(2)
        if match.group(1) == "begin":
            open_names.append(name)
        else:
            for index in range(len(open_names) - 1, -1, -1):
                if open_names[index] == name:
                    del open_names[index]
                    break
    for name in reversed(open_names):
        label = _ENVIRONMENT_LABELS.get(name)
        if label == "table":
            return "table_caption"
        if label == "figure":
            return "figure_caption"
    return "figure_caption"


def _lift_labelled_commands(document: LatexDocument, text: str) -> str:
    r"""
    中文：按 _COMMAND_PASSES 的顺序，逐类抽取参数是散文的命令。

    Lift each command whose argument is prose, one region at a time.

    Args:
        document (LatexDocument): The document being taken apart.
        text (str): The stretch to process.

    Returns:
        str: The stretch with each of them replaced by a placeholder.
    """
    while (match := _BRACKET_CAPTION.search(text)) is not None:
        whole = match.group(0)
        start = whole.rindex("[") + 1
        text = document._lift(text, whole, "prose", match.group(1),
                              label=_caption_label(text, match.start()),
                              span=(start, len(whole) - 1))

    for names, label in _COMMAND_PASSES:
        pattern = patterns.command(names)
        while (match := pattern.search(text)) is not None:
            chosen = _caption_label(text, match.start()) if label == "caption" else label
            text = document._lift(text, match.group(0), "prose", match.group(1),
                                  label=chosen, span=_argument_span(match))
    return text


_OPEN_DISPLAY = regex.compile(r"(?<!\\)\\\[")
_CLOSE_DISPLAY = regex.compile(r"(?<!\\)\\\]")


def _inside_mathematics(text: str, position: int) -> bool:
    r"""
    中文：这个位置是不是待在某个数学环境（或 \[…\]）里面。

    Report whether a position sits inside display mathematics.

    Args:
        text (str): The document as it stands.
        position (int): Where the construct starts.

    Returns:
        bool: ``True`` when an unclosed mathematical environment — or an
            unclosed ``\[`` — encloses it.

    Note:
        The list of mathematical environments above can never be complete: a
        paper may load a package that defines its own, and that environment
        will be full of mathematics and empty of prose. What is reliable is
        where it sits. Anything inside ``equation`` is mathematics whatever it
        is called, and this is the check that says so — measured on
        arXiv:2608.22244, whose five ``aligned`` blocks were translated as
        prose and came back with escaped underscores and inserted dollar signs,
        stopping the compile at "Missing \right. inserted".
        上面那份数学环境清单不可能完备：论文可以加载一个自己定义环境的宏包，而那个环境里
        面全是数学、没有散文。可靠的是它待在哪里。equation 里面的任何东西，不管叫什么名
        字，都是数学——这条检查说的就是这件事。在 arXiv:2608.22244 上实测：它的五个
        aligned 块被当成散文翻译，回来时下划线被转义、还凭空多了美元符号，编译停在
        「Missing \right. inserted」。
    """
    depth = 0
    open_names: list[str] = []
    for match in _BEGIN_OR_END.finditer(text, 0, position):
        name = match.group(2)
        if match.group(1) == "begin":
            open_names.append(name)
        else:
            for index in range(len(open_names) - 1, -1, -1):
                if open_names[index] == name:
                    del open_names[index]
                    break
    if any(_ENVIRONMENT_LABELS.get(name) == "isolate_formula"
           for name in open_names):
        return True

    before = text[:position]
    depth = len(_OPEN_DISPLAY.findall(before)) - len(_CLOSE_DISPLAY.findall(before))
    return depth > 0


def _lift_environments(document: LatexDocument, text: str) -> str:
    r"""
    中文：由内向外抽取环境，最里层的先走。

    Lift environments innermost first.

    Args:
        document (LatexDocument): The document being taken apart.
        text (str): The stretch to process.

    Returns:
        str: The stretch with every environment replaced by a placeholder.

    Note:
        Innermost first is what makes a table's cells reachable: the cells are
        in the ``tabular``, and lifting the ``table`` float first would take the
        ``tabular`` away inside it. See
        :func:`patterns.innermost_environment`.
        由内向外，表格的单元格才够得着：单元格在 tabular 里，先抽 table 浮动体就会把
        tabular 一并带走。见 patterns.innermost_environment。
    """
    pattern = patterns.innermost_environment(r"[A-Za-z@*]+")
    while (match := pattern.search(text)) is not None:
        name = match.group(1)
        label = "" if name in _CODE_ENVIRONMENTS else _ENVIRONMENT_LABELS.get(
            name, "plain text")
        if label != "isolate_formula" and _inside_mathematics(text, match.start()):
            label = "isolate_formula"
        sendable = label in _SENDABLE and document.selection.wants(label)
        text = document._lift(text, match.group(0), "env", name, label=label,
                              span=_body_span(match) if sendable else None)
    return text


def _argument_span(match) -> tuple[int, int]:
    r"""
    中文：在 \caption{…} 这类命令里，定出「花括号内部」的范围。

    Locate the braced argument inside a matched command.

    Args:
        match: A match from :func:`patterns.command`, whose group 3 is the
            braced group including its braces.

    Returns:
        tuple[int, int]: Offsets into ``match.group(0)`` bounding the argument's
            contents. Sending ``\section{Introduction}`` whole invites the model
            to rewrite the command; sending ``Introduction`` cannot.
    """
    whole = match.group(0)
    braced = match.group(3)
    if not braced:
        return (0, len(whole))
    start = whole.rfind(braced)
    return (start + 1, start + len(braced) - 1)


def _body_span(match) -> tuple[int, int]:
    r"""
    中文：在 \begin{env}…\end{env} 里，定出「环境正文」的范围。

    Locate an environment's body inside the matched block.

    Args:
        match: A match from :func:`patterns.environment`, whose group 3 is the
            body.

    Returns:
        tuple[int, int]: Offsets into ``match.group(0)`` bounding the body,
            with any environment arguments left outside it. Handed the whole
            block, the model translated the environment's *name* —
            ``\begin{abstract}`` came back as ``\begin{}摘要}``, and 1777
            characters of the IEEEtran abstract were refused for it.
    """
    whole = match.group(0)
    body = match.group(3)
    if body is None:
        return (0, len(whole))
    start = whole.find(body, whole.find("}") + 1)
    end = start + len(body)

    # Some environments take arguments, and those arguments sit at the front of
    # what the pattern calls the body. A tabular's column specification is the
    # clearest case: handed ``{cc}a & b`` the model rewrites the specification
    # and what comes back is no longer a table.
    # 有些环境带参数，而这些参数就排在正则所谓的「正文」最前面。tabular 的列格式最能
    # 说明问题：把 {cc}a & b 交给模型，它会改写列格式，回来的东西就不再是表格了。
    for _ in range(_ENVIRONMENT_ARGUMENTS.get(match.group(1), 0)):
        argument = regex.match(r"\s*(?:\[[^\]]*\])?\s*\{[^{}]*\}", whole[start:end])
        if argument is None:
            break
        start += argument.end()
    return (start, end)


def _lift_commands(document: LatexDocument, text: str) -> str:
    """
    中文：对一段源码做宏、受保护命令、散文命令与摘要环境四轮抽取。

    Lift macros, protected commands, prose commands and the abstract out of one
    stretch.

    Args:
        document (LatexDocument): The document being taken apart.
        text (str): The stretch to process, typically the preamble.

    Returns:
        str: The stretch with each of them replaced by a placeholder.
    """
    macro = patterns.new_command()
    while (match := macro.search(text)) is not None:
        text = document._lift(text, match.group(0), "macro",
                              match.group(1) or match.group(2) or "?")

    protected = patterns.command(_PROTECTED_COMMANDS)
    while (match := protected.search(text)) is not None:
        text = document._lift(text, match.group(0), "keep", match.group(1))

    text = _lift_labelled_commands(document, text)

    every = patterns.environment(_PREAMBLE_PROSE)
    while (match := every.search(text)) is not None:
        text = document._lift(text, match.group(0), "env", match.group(1),
                              label="plain text", span=_body_span(match))
    return text


def _is_prose(text: str) -> bool:
    """
    中文：这一段里有没有值得翻译的文字。

    Report whether a run of body text carries anything worth translating.

    Args:
        text (str): One paragraph of the body, placeholders and all.

    Returns:
        bool: ``True`` when at least three letters remain once placeholders and
            commands are discounted. A paragraph that is only ``<LTX7>`` and a
            ``\vspace`` is structure, and sending it to a model can only
            damage it.
    """
    stripped = regex.sub(r"<LTX\d+>", " ", text)
    stripped = regex.sub(r"\\[a-zA-Z@]+\*?", " ", stripped)
    return len(regex.findall(r"[A-Za-z]", stripped)) >= 3


# A block of this many lines, nearly all of them ended by an explicit line
# break, is a list of entries rather than prose.
# 一整块里有这么多行、而且几乎每一行都以显式换行收尾，那它是一份条目清单而不是散文。
_LIST_LINES = 20
_LIST_SHARE = 0.8


def _is_entry_list(text: str) -> bool:
    r"""
    中文：这一块是不是「一行一条」的名单，而不是散文。

    Report whether a block is a list of entries rather than a paragraph.

    Args:
        text (str): One block of the body.

    Returns:
        bool: ``True`` when it runs to many lines and nearly every one ends
            with ``\\``.

    Note:
        Written for the Gemini report's contributor list: 1,300 names, one per
        line, arriving as a single 22,938-character block because it sits in a
        ``multicols`` and has no blank line in it. Translating it would be
        wrong even if it worked — a name written 「邓阳江」 no longer matches the
        person — and refusing it cost 15% of the paper's blocks in one go.
        为 Gemini 那篇的贡献者名单而写：1300 个名字，一行一个，因为待在 multicols 里、
        中间又没有空行，整个作为一块 22938 字送过来。就算翻得出来也是错的——名字写成
        「邓阳江」就再也对不上那个人——而拒收它一次就赔掉了全文 15% 的块。
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < _LIST_LINES:
        return False
    ended = sum(1 for line in lines if line.endswith("\\\\"))
    return ended >= _LIST_SHARE * len(lines)


# A run of this many comma-separated fragments, nearly all of them one to four
# capitalised words, is a list of names.
# 这么多个逗号分隔的片段，而且几乎每一个都是一到四个首字母大写的词，那就是一份名单。
_NAMES_PARTS = 30
_NAMES_SHARE = 0.8


def _is_name_list(text: str) -> bool:
    """
    中文：这一块是不是一长串人名（致谢里的那种）。

    Report whether a block is a long run of personal names.

    Args:
        text (str): One block of the body.

    Returns:
        bool: ``True`` for an acknowledgements roll — many comma-separated
            fragments, nearly all of them capitalised words.

    Note:
        The Gemini report's acknowledgements are one sentence followed by 2,900
        characters of names on a single line, so the one-name-per-line test
        above does not see them. Names are not translated in this project — a
        person written 「邓阳江」 no longer matches the person — so the block is
        left alone rather than sent and refused.
        Gemini 那篇的致谢是一句话后面跟着 2900 字的人名，全在一行上，所以上面那条「一行
        一个」的判断看不见它。本项目不翻译人名——写成「邓阳江」就再也对不上那个人——所以
        整块留着不动，而不是送出去再被拒收。
    """
    parts = [part.strip() for part in text.split(",")]
    if len(parts) < _NAMES_PARTS:
        return False
    namelike = 0
    for part in parts:
        words = part.split()
        if not 1 <= len(words) <= 4:
            continue
        if all(word[:1].isupper() for word in words if word[:1].isalpha()):
            namelike += 1
    return namelike >= _NAMES_SHARE * len(parts)


def _lift_paragraphs(document: LatexDocument, body: str) -> str:
    """
    中文：把正文按空行切段，每段抽成一块，正文本身只剩占位符骨架。

    Lift each paragraph of the body, leaving a skeleton of placeholders.

    A blank line is LaTeX's own paragraph break, so it is the division the
    document already believes in. Splitting anywhere else — by sentence, by
    length — would put a break where the author did not.

    Args:
        document (LatexDocument): The document being taken apart.
        body (str): The body with structure already lifted out.

    Returns:
        str: The body reduced to placeholders and the whitespace between them.
    """
    out = []
    for block in regex.split(r"(\n[ \t]*\n)", body):
        if not block.strip() or regex.fullmatch(r"\n[ \t]*\n", block):
            out.append(block)
            continue
        if not _is_prose(block) or _is_entry_list(block) or _is_name_list(block):
            out.append(block)
            continue
        mark = document._next_mark()
        prefix, core, suffix = _peel_marks("", block, "")
        dropcap = ""
        opening = _DROPCAP.match(core)
        if opening:
            # The word is put back together and the command set aside, so the
            # model reads "According" and not "{A}{ccording}".
            # 把词拼回去、命令搁在一边，模型读到的是 According 而不是 {A}{ccording}。
            dropcap = opening.group(1)
            core = opening.group(2) + opening.group(3) + core[opening.end():]
        translate = document.selection.wants("plain text")
        document.pieces.append(Piece(mark=mark, kind="text", name="paragraph",
                                     source=block, label="plain text",
                                     prefix=prefix, core=core, suffix=suffix,
                                     translate=translate, dropcap=dropcap))
        out.append(mark)
    return "".join(out)
