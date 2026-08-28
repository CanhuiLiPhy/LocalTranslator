r"""
中文：LaTeX 源码翻译的完整流程：取源码 → 解析 → 翻译 → 装配 → 编译。

The whole of source-based translation, end to end.

Where the PDF pipeline reads a printed page and puts Chinese back into its
boxes, this one reads what the page was printed *from*. The difference shows in
what cannot go wrong: a formula is a formula because it is written as one, a
figure keeps its own coordinate system, and a paragraph reflows because TeX
reflows it. None of the defects this project spent weeks on — clipped glyphs,
stranded subscripts, boxes a tenth of a point too narrow — exists here.

What replaces them is a different risk: the source has to compile. That is what
:mod:`local_translator.latex.preamble` and the checks in
:mod:`local_translator.latex.translate` are for.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from local_translator.latex import (
    dual,
    engine,
    figures,
    parser,
    preamble,
    source,
    translate,
)
from local_translator.latex.selection import Selection

logger = logging.getLogger(__name__)

# Name of the translated document written beside the original sources. Kept
# distinct so a reader can compare the two, and so a second run does not
# compile against the first one's output. Defined in :mod:`source` because
# :func:`source.find_main` has to skip it.
# 与原始源码并排写出的译文文件名。取一个不同的名字，读者可以两相对照，重跑也不会拿上
# 一次的产物去编译。定义在 source 里，因为 source.find_main 必须跳过它。
TRANSLATED = source.TRANSLATED

# The English half of the bilingual PDF, and the bilingual PDF itself. Named
# beside the translation so a reader who keeps the sources can see all three.
# 双语版的英文那一半，以及双语版本身。与译文并排命名，保留源码的读者能看到三份。
ORIGINAL = "original-en.tex"
BILINGUAL = "translated-dual.pdf"


def read_bibliography(main: Path) -> str:
    r"""
    中文：读出论文预编译好的 .bbl，供解析时嵌入 \bibliography{…} 的位置。

    Read the paper's pre-built bibliography.

    arXiv submissions carry a compiled ``.bbl`` rather than the ``.bib`` it was
    built from, and LaTeX finds that file by *job* name. Writing the
    translation as ``translated-zh.tex`` sends it looking for
    ``translated-zh.bbl``. Copying the file across under the new name is the
    obvious repair, and it is not enough: the compile then reads the file and
    still prints an empty References section, because resolving citations that
    way depends on what BibTeX did or did not do on an earlier pass.

    Splicing the entries in has none of that. The ``thebibliography``
    environment is simply *there*, in the document, on the first pass.
    Twenty-three citations in the ICLR paper rendered as ``?`` with no error —
    only a warning in the log — until they were.

    Args:
        main (Path): The original main ``.tex``, whose stem names the ``.bbl``.

    Returns:
        str: The entries, or empty when there is no ``.bbl`` — in which case
            the document's own ``thebibliography`` or a real ``.bib`` is
            already doing the work.

    Note:
        The splice happens during parsing rather than after assembly, so the
        entries are an ordinary environment the parser can label ``reference``.
        Whether they are then translated is the reader's choice, and the
        default — leave them — is what a reader wants: a reference they cannot
        match against the literature is worse than one they can.
        嵌入发生在解析期间而不是装配之后，这样这些条目就是一个普通环境，解析器可以给它
        贴上 reference 标签。译不译由读者决定，而默认的「不译」正是读者想要的：一条对不
        上文献的引用，比一条能对上的更糟。
    """
    bbl = main.with_suffix(".bbl")
    if not bbl.is_file():
        return ""
    return bbl.read_text(encoding="utf-8", errors="replace")


@dataclass
class Outcome:
    """
    中文：一次源码翻译的结果。

    What one source translation produced.

    Attributes:
        pdf (Path | None): The translation-only PDF, when there is one.
        dual (Path | None): The bilingual PDF, when one was asked for and both
            halves compiled.
        tex (Path | None): The translated source, which exists even when the
            compile failed — it is what a reader needs in order to fix it.
        stats (dict): Counts from the translation step.
        seconds (float): Compile time.
        error (str): Why there is no PDF, empty on success.
        log (str): The compiler's output.
    """

    pdf: Path | None = None
    dual: Path | None = None
    tex: Path | None = None
    stats: dict = field(default_factory=dict)
    seconds: float = 0.0
    error: str = ""
    log: str = ""

    @property
    def ok(self) -> bool:
        """中文：出 PDF 了没有。 / Whether a PDF came out."""
        return self.pdf is not None and self.pdf.is_file()


def translate_tree(root: Path, main: Path, translator, *, workers: int = 8,
                   ignore_cache: bool = False, font: str | None = None,
                   tectonic: Path | None = None, system_engine: Path | None = None,
                   selection: Selection | None = None, glossary=None,
                   hard_protect: bool = False, translate_image=None,
                   output_mono: bool = True, output_dual: bool = True,
                   alternating_dual: bool = False, dual_translate_first: bool = False,
                   on_progress=None) -> Outcome:
    """
    中文：翻译一棵已经解开的源码树并编译。

    Translate an unpacked source tree and compile it.

    Args:
        root (Path): Directory holding the sources. Written into, so the caller
            should pass a working copy rather than anything precious.
        main (Path): The main ``.tex``.
        translator: This project's translator.
        workers (int): How many pieces to translate at once.
        ignore_cache (bool): Passed through to the translator.
        font (str | None): Preferred CJK font family.
        tectonic (Path | None): The bundled engine.
        system_engine (Path | None): A system ``xelatex``, preferred when given
            because its macro coverage is better than what downloads on demand.
        selection (Selection | None): Which regions to translate — the same
            switches the PDF path reads. Defaults to the ``pdfmath`` preset.
        glossary (Callable | None): The project's glossary matcher.
        hard_protect (bool): Hold whitelisted terms behind markers in prose too.
        translate_image (Callable | None): Takes image data and returns a PNG
            with the text translated. Called only when the selection asks for
            figures; passing ``None`` leaves every figure alone whatever the
            selection says, which is what happens when OCR is not installed.
        output_mono (bool): Keep the translation-only PDF.
        output_dual (bool): Also build the bilingual PDF, which costs a second
            compile of the same document with the translations left out.
        alternating_dual (bool): Interleave the pages instead of placing them
            side by side.
        dual_translate_first (bool): Put the translation on the left, or first.
        on_progress (Callable | None): Called with ``(done, total)``.

    Returns:
        Outcome: With the PDF when the document compiled.

    Raises:
        RuntimeError: If no compiler is available at all, which is a
            configuration problem rather than a document problem and should not
            be reported as a failed translation.
    """
    chosen = system_engine or tectonic or engine.find_system_engine() \
        or engine.find_tectonic()
    if chosen is None:
        raise RuntimeError(
            "找不到 LaTeX 编译器。请安装含 TeX 功能的版本，或在设置里指定 TeX 路径。")
    is_system = chosen == system_engine or chosen.name in ("xelatex", "lualatex")

    selection = selection or Selection()
    document = parser.parse(root, main, selection=selection,
                            bibliography=read_bibliography(main))

    if selection.translate_images and translate_image is not None:
        figures.translate_figures(document, translate_image)
    elif selection.translate_images:
        logger.info("选择了翻译图片，但本机没有可用的图片翻译（OCR 模型未安装），插图保持原样")

    stats = translate.translate_document(
        document, translator, workers=workers, ignore_cache=ignore_cache,
        glossary=glossary, hard_protect=hard_protect, on_progress=on_progress)

    text = preamble.prepare(parser.assemble(document), font, root)
    output = root / TRANSLATED
    output.write_text(text, encoding="utf-8")

    result = engine.compile_document(output, chosen, system=is_system)
    outcome = Outcome(pdf=result.pdf, tex=output, stats=stats,
                      seconds=result.seconds, error=result.error, log=result.log)
    if not output_mono and not output_dual:
        # Both switched off. A run that produces nothing is not a preference a
        # reader can have meant, so the translation is kept.
        # 两个都关了。一趟什么都不产出的运行不可能是读者的本意，所以留下译文。
        output_mono = True
    if not outcome.ok or not output_dual:
        return outcome

    outcome.dual = _build_dual(document, root, font, chosen, is_system,
                               result.pdf, alternating=alternating_dual,
                               translation_first=dual_translate_first)
    if not output_mono and outcome.dual is not None:
        # The reader asked for the bilingual PDF only. The translation-only one
        # was still built — it is half of the bilingual PDF — so it is removed
        # rather than never made.
        # 读者只要双语版。单语版仍然是造出来了的——它是双语版的一半——所以是造完再删，
        # 而不是没造。
        result.pdf.unlink(missing_ok=True)
        outcome.pdf = None
    return outcome


def _build_dual(document, root: Path, font: str | None, engine_file: Path,
                is_system: bool, translated_pdf: Path, *, alternating: bool,
                translation_first: bool):
    """
    中文：编一份「同样的文档但不放译文」，再与译文版拼成双语对照。

    Compile the same document without its translations and pair the two.

    Args:
        document (LatexDocument): The parsed document, translations and all.
        root (Path): The source tree.
        font (str | None): CJK font family, applied to both halves so the two
            compile through the same preamble.
        engine_file (Path): The chosen engine.
        is_system (bool): Whether it is a system engine.
        translated_pdf (Path): The translation-only PDF, already compiled.
        alternating (bool): Interleave pages instead of placing them abreast.
        translation_first (bool): Put the translation on the left, or first.

    Returns:
        Path | None: The bilingual PDF, or ``None`` when the English half did
            not compile — in which case the translation is still there and the
            run is still a success, one output short.

    Note:
        The English half is assembled from the same pieces with ``translated``
        cleared, not from the original ``.tex``. The preamble this project
        writes is what makes the document compile under XeTeX with a CJK font;
        an original compiled without it is a different document, and on four of
        seven arXiv papers it is a document that does not compile at all.
        英文那一半是把同一批片段的 translated 清空之后重新装配出来的，而不是拿原始 .tex
        去编。本项目改写的前言正是这份文档能在 XeTeX 下带中文字体编过的原因；不带这份
        前言编出来的原文是另一份文档，而七篇 arXiv 论文里有四篇那样根本编不过。
    """
    kept = [piece.translated for piece in document.pieces]
    for piece in document.pieces:
        piece.translated = ""
    try:
        english = preamble.prepare(parser.assemble(document), font, root)
    finally:
        for piece, translation in zip(document.pieces, kept, strict=True):
            piece.translated = translation

    source_tex = root / ORIGINAL
    source_tex.write_text(english, encoding="utf-8")
    result = engine.compile_document(source_tex, engine_file, system=is_system)
    if result.pdf is None:
        logger.warning("原文半边没编过，只出单语版：%s", result.error[:160])
        return None

    target = root / BILINGUAL
    left, right = ((translated_pdf, result.pdf) if translation_first
                   else (result.pdf, translated_pdf))
    if alternating:
        return dual.alternating(left, right, target)
    return dual.side_by_side(left, right, target)


async def translate_arxiv(arxiv_id: str, workspace: Path, translator, **options) -> Outcome:
    """
    中文：下载一篇 arXiv 论文的源码并翻译编译。

    Fetch a paper's source from arXiv and translate it.

    Args:
        arxiv_id (str): Identifier without a version suffix.
        workspace (Path): Where to download and build.
        translator: This project's translator.
        **options: Passed to :func:`translate_tree`.

    Returns:
        Outcome: With the PDF when the document compiled.

    Raises:
        source.NoSource: When arXiv has no LaTeX source for this paper, which
            is a reason to fall back to translating the PDF rather than an
            error to show a reader.
    """
    workspace.mkdir(parents=True, exist_ok=True)
    archive = await source.download(arxiv_id, workspace)
    root, main = source.prepare_workspace(archive, workspace / "source")
    return translate_tree(root, main, translator, **options)


def translate_folder(folder: Path, workspace: Path, translator, **options) -> Outcome:
    """
    中文：翻译用户自己提供的一个 LaTeX 源码文件夹。

    Translate a LaTeX project the reader supplied.

    Args:
        folder (Path): The reader's source folder, copied rather than modified.
        workspace (Path): Where to build.
        translator: This project's translator.
        **options: Passed to :func:`translate_tree`.

    Returns:
        Outcome: With the PDF when the document compiled.

    Raises:
        source.NoSource: When the folder holds no main ``.tex``.
    """
    build = workspace / "source"
    if build.exists():
        shutil.rmtree(build)
    # Copied, never translated in place: a reader who points at their own
    # project should get it back untouched whatever happens here.
    # 复制而不是就地翻译：读者指向的是自己的项目，无论这里发生什么，它都该原样还在。
    shutil.copytree(folder, build)
    main = source.find_main(build)
    if main is None:
        raise source.NoSource("这个文件夹里找不到带 \\documentclass 的主 .tex 文件")
    return translate_tree(build, main, translator, **options)
