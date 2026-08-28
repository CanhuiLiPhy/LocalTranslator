r"""
中文：让一篇按 pdfLaTeX 写的论文能用 XeTeX 编译，并且排得出中文。

Make a paper written for pdfLaTeX compile under XeTeX, in Chinese.

Two requirements pull in opposite directions. Chinese output needs XeTeX or
LuaTeX — pdfTeX can only reach CJK through ``CJKutf8`` and a handful of fonts.
But arXiv papers are written for pdfLaTeX, and measured on the seven papers in
this project's bench, four of the seven fail to compile under XeTeX at all.

The four failures are two causes, and neither is a missing package:

**``\\pdfoutput=1``.** arXiv's own convention for telling its build system to
use pdfLaTeX. Under XeTeX it makes ``hyperref`` load its pdfTeX driver,
``hpdftex.def``, which then dies on an undefined control sequence. The line
means nothing to us, so it goes.

**microtype's tracking is pdfTeX-only.** Worse, it is usually the *class* file
that loads microtype — ``deepseek.cls``, ``sn-jnl.cls`` — so no edit to the
preamble can reach it. A no-op ``microtype.sty`` placed beside the document
shadows the real package, because TeX searches the working directory first.
microtype only adjusts spacing, so removing it changes no content.

With both applied, five of the seven compile.

**Then the Chinese.** ``CJKutf8`` under XeTeX drops every Chinese character and
still writes a PDF, reporting only ``could not represent character`` in a
warning — a silent failure at the last step of the pipeline. ``xeCJK`` with an
explicit font is what works, and it is added at the end of the preamble, the
position most compatible with classes that set up fonts of their own.

Kept deliberately small: line breaking and a font, nothing else. Adding
``\punctstyle`` for Chinese punctuation compression looked free and broke two
of the five papers with ``xeCJK.sty: Undefined control sequence`` — the command
is not in every build of the package. A preamble addition that fails takes the
whole document with it, so each line here has to earn its place.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import regex

from local_translator.latex import patterns

logger = logging.getLogger(__name__)

# A no-op stand-in for microtype. Every command it defines is one a document or
# another package may call, so each has to exist and do nothing.
# microtype 的空壳。它定义的每个命令，文档或其他宏包都可能调用，因此每个都得存在且
# 什么也不做。
_MICROTYPE_STUB = r"""\NeedsTeXFormat{LaTeX2e}
\ProvidesPackage{microtype}[2026/01/01 no-op stub: tracking is pdfTeX-only]
\DeclareOption*{}
\ProcessOptions\relax
\providecommand{\textls}[2][]{#2}
\providecommand{\lsstyle}{}
\providecommand{\textmicrotypecontext}[2]{#2}
\newcommand{\SetTracking}[3][]{}
\newcommand{\microtypesetup}[1]{}
\newcommand{\microtypecontext}[1]{}
\newcommand{\DeclareMicrotypeSet}[3][]{}
\newcommand{\UseMicrotypeSet}[2][]{}
\newcommand{\DeclareMicrotypeSetDefault}[2][]{}
\newcommand{\SetProtrusion}[3][]{}
\newcommand{\SetExpansion}[3][]{}
\newcommand{\SetExtraKerning}[3][]{}
\newcommand{\SetExtraSpacing}[3][]{}
\newcommand{\DisableLigatures}[2][]{}
\newcommand{\DeclareCharacterInheritance}[3][]{}
\newcommand{\LoadMicrotypeFile}[1]{}
% xeCJK checks whether microtype is loaded and, finding it, calls into its
% internals. A stub that answers the first question but not the second makes
% xeCJK die on \MT@addto@setup — so the internals other packages reach for
% have to exist here too.
% xeCJK 会检查 microtype 是否加载，发现「已加载」就去调它的内部命令。空壳只答对第一问
% 而没有第二问的接口，xeCJK 就会死在 \MT@addto@setup 上——因此别的宏包会伸手来拿的
% 内部命令，这里也得有。
\newcommand{\MT@addto@setup}[1]{}
\newcommand{\MT@restore@setup}[1]{}
\let\MT@setup@\@empty
\newcommand{\MT@is@composite}{}
\newcommand{\MT@map@case}[1]{}
"""

# Packages that only make sense under pdfTeX and break or do nothing under
# XeTeX. inputenc is the notable one: XeTeX is Unicode natively, and a document
# declaring latin1 makes it misread the very characters we are adding.
# 只在 pdfTeX 下有意义、在 XeTeX 下会坏掉或无效的宏包。inputenc 最典型：XeTeX 原生就是
# Unicode，而声明 latin1 会让它误读我们刚加进去的那些字符。
_POINTLESS = ("inputenc", "fontenc", "CJKutf8", "CJK")

# CJK fonts to try, best first. All ship with macOS; the list is tried against
# what is actually installed rather than assumed.
# 依次尝试的中文字体，越靠前越优先。都是 macOS 自带；实际按已安装的来挑，不做假设。
_FONT_CANDIDATES = ("Songti SC", "STSong", "PingFang SC", "Heiti SC",
                    "Noto Serif CJK SC", "Source Han Serif SC", "SimSun")


# Where macOS keeps its CJK faces, and the file each family lives in. Naming a
# font by *file* rather than by family is what makes it resolve reliably: a
# document that loads fontspec early — fontawesome5 does — leaves the family
# lookup unable to find "Songti SC", while the same font by path is found.
# macOS 的中日韩字体在哪、每个字族对应哪个文件。按**文件**而不是按字族指定，才能稳定
# 解析：提前加载 fontspec 的文档（fontawesome5 就会）会让字族查找找不到「Songti SC」，
# 而同一个字体按路径就能找到。
# family, folder, file, regular face index, bold face index.
#
# The index matters and is not guessable. A .ttc is a collection, and naming
# the file alone gets face 0 — which in Songti.ttc is `STSongti-SC-Black`, the
# heaviest weight there is, so a whole paper comes out looking bold. Probed
# face by face: 0 Black, 1 SC-Bold, 2 TC-Bold, 3 SC-Light, 4 STSong, 5 TC-Light.
# 字族、目录、文件、正体字面序号、粗体字面序号。
#
# 序号很重要，而且猜不出来。.ttc 是一个集合，只给文件名会拿到第 0 个字面——在
# Songti.ttc 里那是 STSongti-SC-Black，最重的一档，于是整篇论文看起来都是粗的。逐个
# 探测的结果：0 Black、1 SC-Bold、2 TC-Bold、3 SC-Light、4 STSong、5 TC-Light。
_FONT_FILES: tuple[tuple[str, str, str, int, int], ...] = (
    ("Songti SC", "/System/Library/Fonts/Supplemental/", "Songti.ttc", 4, 1),
    ("PingFang SC", "/System/Library/Fonts/", "PingFang.ttc", 2, 5),
    ("Heiti SC", "/System/Library/Fonts/", "STHeiti Medium.ttc", 0, 0),
    ("STHeiti", "/System/Library/Fonts/", "STHeiti Light.ttc", 0, 0),
)


@dataclass(frozen=True)
class Font:
    """
    中文：一个可用的中文字体，按文件定位。

    A usable CJK font, located by file.

    Attributes:
        family (str): Its family name, for messages.
        directory (str): The folder holding it, with a trailing slash.
        stem (str): The file name without its extension.
        extension (str): The extension, with the dot.
    """

    family: str
    directory: str
    stem: str
    extension: str
    index: int = 0
    bold_index: int = 0

    def declaration(self) -> str:
        """中文：写进 .tex 的那一行。 / The line that goes into the document."""
        return (f"\\setCJKmainfont[Path={self.directory},"
                f"Extension={self.extension},FontIndex={self.index},"
                f"BoldFeatures={{FontIndex={self.bold_index}}}]{{{self.stem}}}")


def find_cjk_font(preferred: str | None = None) -> Font | None:
    """
    中文：找一个本机确实存在的中文字体文件。

    Find a CJK font file this machine actually has.

    Args:
        preferred (str | None): A family named in the settings, tried first.

    Returns:
        Font | None: The font, or ``None`` when the machine has no CJK face at
            all — in which case the caller must say so rather than compile a
            document whose Chinese would silently vanish.
    """
    order = list(_FONT_FILES)
    if preferred:
        order.sort(key=lambda entry: entry[0] != preferred)
    for family, directory, filename, index, bold in order:
        path = Path(directory) / filename
        if path.is_file():
            return Font(family=family, directory=directory, stem=path.stem,
                        extension=path.suffix, index=index, bold_index=bold)
    return None


def available_cjk_font(preferred: str | None = None) -> str:
    """
    中文：挑一个本机装了的中文字体族名（仅用于显示）。

    Choose the family name of an installed CJK font, for display.

    Args:
        preferred (str | None): A font named in the settings, tried first.

    Returns:
        str: A font family name. Falls back to the first candidate when the
            system font list cannot be read, because naming a font that is
            missing produces a clear compile error, while naming none at all
            produces a PDF with the Chinese silently absent.
    """
    wanted = ([preferred] if preferred else []) + list(_FONT_CANDIDATES)
    listing = ""
    finder = shutil.which("fc-list")
    if finder:
        try:
            listing = subprocess.run([finder, ":", "family"], capture_output=True,
                                     text=True, timeout=20).stdout
        except Exception:  # noqa: BLE001
            listing = ""
    if not listing:
        # macOS without fontconfig: look at the font folders directly.
        # 没有 fontconfig 的 macOS：直接看字体目录。
        names = []
        for folder in (Path("/System/Library/Fonts"), Path("/Library/Fonts"),
                       Path.home() / "Library/Fonts"):
            if folder.is_dir():
                # rglob, not glob: macOS keeps Songti and the other CJK faces in
                # a Supplemental subfolder, and looking only at the top level
                # finds none of them.
                # 用 rglob 而不是 glob：macOS 把宋体等中日韩字体放在 Supplemental
                # 子目录里，只看顶层一个也找不到。
                names += [item.stem for item in folder.rglob("*")
                          if item.suffix.lower() in (".ttc", ".ttf", ".otf")]
        listing = "\n".join(names)

    # A font file is named for the face, not the family: macOS ships
    # `Songti.ttc` and the family is `Songti SC`. So a family counts as present
    # when some installed name is a prefix of it, compared without spaces.
    # 字体文件按字面名命名，不是按字族：macOS 装的是 Songti.ttc，字族叫 Songti SC。
    # 因此判定标准是「某个已装的名字是该字族的前缀」，比较时忽略空格。
    installed = {line.strip().replace(" ", "").lower()
                 for line in listing.splitlines() if line.strip()}
    for font in wanted:
        flat = font.replace(" ", "").lower()
        if any(flat.startswith(name) or name.startswith(flat)
               for name in installed if len(name) >= 4):
            return font
    logger.warning("没有找到已安装的中文字体，回退到 %s", wanted[0])
    return wanted[0]


def _strip_pdfoutput(text: str) -> tuple[str, int]:
    """
    中文：去掉 \\pdfoutput=1。

    Remove arXiv's ``\\pdfoutput`` directive.

    Args:
        text (str): LaTeX source.

    Returns:
        tuple[str, int]: The source and how many lines were removed.
    """
    text, count = regex.subn(r"^[ \t]*\\pdfoutput[ \t]*=[ \t]*\d+[ \t]*$",
                             "% \\\\pdfoutput removed: this is compiled with XeTeX",
                             text, flags=regex.M)
    return text, count


def _comment_out_packages(text: str) -> tuple[str, list[str]]:
    """
    中文：注释掉在 XeTeX 下无意义或有害的宏包。

    Comment out packages that XeTeX does not need and may choke on.

    Args:
        text (str): LaTeX source.

    Returns:
        tuple[str, list[str]]: The source and the names removed.
    """
    removed: list[str] = []

    def replace(match: regex.Match) -> str:
        """中文：把整条 \\usepackage 注释掉。 / Comment the whole \\usepackage out."""
        inside = match.group(2) or ""
        hit = [name for name in _POINTLESS
               if regex.search(rf"\b{regex.escape(name)}\b", inside)]
        if not hit:
            return match.group(0)
        removed.extend(hit)
        return "% " + match.group(0).replace("\n", "\n% ")

    pattern = regex.compile(
        rf"\\usepackage{patterns.SPACES}(?:{patterns.OPTIONS})?"
        rf"{patterns.SPACES}\{{([^}}]*)\}}".replace("([^}]*)", "([^}]*)"),
        regex.DOTALL)
    # Rebuild with an explicit group for the braced names.
    # 重新构造，给花括号里的名字一个明确的捕获组。
    pattern = regex.compile(
        rf"(\\usepackage{patterns.SPACES}(?:{patterns.OPTIONS})?{patterns.SPACES})"
        rf"\{{([^}}]*)\}}", regex.DOTALL)
    return pattern.sub(replace, text), removed


# Stand-ins for the CJK package's environments, needed when that package is
# removed but the document still uses it. The Gemini report sets its Chinese and
# Japanese examples in `CJK*`, and removing the package without these turns a
# working document into "Environment CJK* undefined". Under xeCJK the wrapper
# has nothing left to do, so it does nothing.
# CJK 宏包环境的替身：宏包被移除、而文档仍在使用时需要它们。Gemini 那篇的中日文示例就
# 排在 CJK* 里，移除宏包却不补这一段，能编的文档会变成「Environment CJK* undefined」。
# 在 xeCJK 之下这个包装已无事可做，所以它什么也不做。
_CJK_SHIM = (
    "\\providecommand{\\CJKfamily}[1]{}\n"
    "\\providecommand{\\CJKtilde}{}\n"
    "\\providecommand{\\CJKnospace}{}\n"
    "\\newenvironment{CJK}[2]{}{}\n"
    "\\newenvironment{CJK*}[2]{}{}\n"
)


_PACKAGE_WITH_OPTIONS = regex.compile(
    r"(?<!\\)\\usepackage\s*\[([^\]]*)\]\s*\{([^{}]*)\}")


def hoist_package_options(text: str) -> tuple[str, int]:
    r"""
    中文：把每个 \usepackage[选项]{宏包} 的选项提前登记到 \documentclass 之前。

    Register every package's options before the class is loaded.

    Args:
        text (str): The assembled source, preamble and all.

    Returns:
        tuple[str, int]: The source with the declarations prepended, and how
            many were added.

    Note:
        This is the standard cure for "Option clash for package …", and the
        clash is not rare: a class loads ``xcolor`` with no options, the paper
        then asks for ``\usepackage[svgnames]{xcolor}``, and LaTeX stops.
        Measured on arXiv:2608.12209, whose own sources fail this way before
        anything is translated — its ``style.cls`` requires ``xcolor`` and its
        preamble loads it three more times with three different option sets.
        ``\PassOptionsToPackage`` records the options for whoever loads the
        package first, so all of them are in force and the later
        ``\usepackage`` finds nothing new to complain about.
        这是「Option clash for package …」的标准解法，而这种冲突并不少见：文档类不带选项
        加载 xcolor，论文接着写 \usepackage[svgnames]{xcolor}，LaTeX 就停下。在
        arXiv:2608.12209 上实测——它的源码在翻译之前就是这么失败的：style.cls 里
        \RequirePackage{xcolor}，前言里又用三组不同的选项加载了三次。
        \PassOptionsToPackage 把选项记给「第一个加载该宏包的人」，于是所有选项都生效，后面
        那个 \usepackage 也就找不到新东西可抱怨了。

        Applied to every package with options, not only to those that clash.
        The declaration is inert when there is no clash — it is how a document
        is supposed to ask for options it cannot control the loading order of —
        and detecting a clash in advance would mean knowing what the class and
        every package load, which is what running LaTeX is for.
        对所有带选项的宏包都做，而不只是会冲突的那些。没有冲突时这条声明是惰性的——文档
        本来就该用它来请求那些自己控制不了加载顺序的选项——而提前判断会不会冲突，等于要
        知道文档类和每个宏包各自加载了什么，那正是跑一遍 LaTeX 要做的事。
    """
    opening = patterns.DOCUMENT_CLASS.search(text)
    if opening is None:
        return text, 0

    declarations: list[str] = []
    for match in _PACKAGE_WITH_OPTIONS.finditer(text, opening.end()):
        options = match.group(1).strip()
        if not options:
            continue
        for name in match.group(2).split(","):
            name = name.strip()
            if not name:
                continue
            line = f"\\PassOptionsToPackage{{{options}}}{{{name}}}"
            if line not in declarations:
                declarations.append(line)
    if not declarations:
        return text, 0

    block = ("% ---- added by 本地翻译: options registered before the class ----\n"
             + "\n".join(declarations) + "\n% ---- end ----\n")
    return block + text, len(declarations)


def prepare(text: str, font: str, build_dir: Path) -> str:
    """
    中文：对一篇论文的源码做全部前言手术，并在编译目录放好 microtype 空壳。

    Apply every preamble repair and write the microtype stub.

    Args:
        text (str): The assembled LaTeX source, translated or not.
        font (str): CJK font family to set.
        build_dir (Path): Directory the document will be compiled in; the stub
            is written here so it shadows the real package.

    Returns:
        str: Source ready for XeTeX.
    """
    (build_dir / "microtype.sty").write_text(_MICROTYPE_STUB, encoding="utf-8")

    text, stripped = _strip_pdfoutput(text)
    text, removed = _comment_out_packages(text)
    text, hoisted = hoist_package_options(text)
    if hoisted:
        logger.info("把 %d 组宏包选项提前登记，避免 Option clash", hoisted)
    removed_cjk = any(name.startswith("CJK") for name in removed)
    if stripped:
        logger.info("去掉 %d 处 \\pdfoutput", stripped)
    if removed:
        logger.info("注释掉 XeTeX 下无用的宏包：%s", "、".join(sorted(set(removed))))

    chosen = find_cjk_font(font)
    if chosen is None:
        raise RuntimeError(
            "这台机器上没有找到任何中文字体文件，无法排出中文。"
            "请在设置里指定一个中文字体。")
    setup = (
        "\n% ---- added by 本地翻译: Chinese typesetting under XeTeX ----\n"
        "\\usepackage{xeCJK}\n"
        + chosen.declaration() + "\n"
        "\\XeTeXlinebreaklocale \"zh\"\n"
        "\\XeTeXlinebreakskip = 0pt plus 1pt\n"
        + (_CJK_SHIM if removed_cjk else "")
        + "% ---- end ----\n"
    )
    opening = patterns.BEGIN_DOCUMENT.search(text)
    if opening is None:
        logger.warning("找不到 \\begin{document}，中文字体设置无处安放")
        return text
    return text[:opening.start()] + setup + text[opening.start():]
