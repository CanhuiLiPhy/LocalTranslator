r"""
中文：把插图里的文字也译过来——识别、回填、换掉 \includegraphics 指向的文件。

Translate the text inside a paper's figures.

A figure in a LaTeX source is a file, and its labels are pixels or vector
strokes inside that file rather than text in the document. Nothing in the
parser can reach them. What can reach them is the image translation this
project already has: recognition runs locally on PP-OCRv6, each recognised line
is translated, and the translation is drawn back over the box the original
occupied. The result is written beside the original and the ``\includegraphics``
is pointed at it.

**Why a vector figure is rasterised.** Most arXiv figures are PDFs, and their
labels are real text objects — in principle they could be rewritten in place,
the way the PDF pipeline rewrites a page. In practice that is the whole of the
PDF pipeline, re-aimed at a figure: fonts to embed, boxes to fit, clips to
respect. Rendering the page to a bitmap and running the same OCR path over it
gives one code path for every figure format, at the cost of the figure's vector
sharpness. That cost is why this is off unless a reader asks for it.

**中文：** LaTeX 里的插图是一个文件，图上的标签是文件内部的像素或矢量笔画，而不是文档
里的文字，解析器够不着。够得着的是本项目已有的图片翻译：本地 PP-OCRv6 识别，逐行翻译，
再把译文画回原文所在的框里。产物写在原图旁边，\includegraphics 改指向它。

矢量图为什么要栅格化：多数 arXiv 插图是 PDF，标签是真正的文字对象，原则上可以像 PDF
管线重写页面那样就地改写——但那实际上等于把整条 PDF 管线再对着一张插图做一遍。渲染成
位图、走同一条 OCR 路，可以让所有图片格式共用一条代码路径，代价是矢量清晰度。正因为
有这个代价，这项功能默认不开。
"""

from __future__ import annotations

import logging
from pathlib import Path

import regex

logger = logging.getLogger(__name__)

# The \includegraphics call, with its optional arguments and its path.
# \includegraphics 调用，连同可选参数与路径。
_INCLUDE = regex.compile(
    r"(\\includegraphics\s*)((?:\[[^\]]*\])?)(\s*\{)([^{}]*)(\})")

# Extensions LaTeX will try when the path carries none, in the order graphicx
# itself prefers them.
# 路径不带扩展名时 LaTeX 会依次尝试的扩展名，顺序与 graphicx 自己的偏好一致。
_EXTENSIONS = (".pdf", ".png", ".jpg", ".jpeg", ".eps", ".ps", ".gif", ".tif", ".tiff")

# Formats the OCR path reads directly. Anything else is rendered to a bitmap
# first.
# OCR 直接能读的格式。其余的先渲染成位图。
_RASTER = frozenset({".png", ".jpg", ".jpeg", ".gif", ".tif", ".tiff", ".bmp"})

# Resolution for rendering a vector figure before recognition, in dots per
# inch. High enough that an eight-point axis label survives recognition, low
# enough that a full-page figure stays a few megabytes.
# 识别之前把矢量图渲染成位图的分辨率（每英寸点数）。够高，八号的坐标轴标签才认得出来；
# 够低，一张整页插图才不会变成几十兆。
_DPI = 300

# What the translated copy is called, beside the original.
# 译文版图片在原图旁边叫什么。
_SUFFIX = ".zh.png"


def _resolve(root: Path, reference: str) -> Path | None:
    """
    中文：把 \\includegraphics 里的路径对应到磁盘上的文件。

    Find the file a graphics reference names.

    Args:
        root (Path): The source tree's root.
        reference (str): The path as written, which may carry no extension.

    Returns:
        Path | None: The file, or ``None`` when nothing matches — a paper that
            references a figure it does not ship is not this module's problem
            to solve, and saying so in the log is more use than guessing.
    """
    candidate = (root / reference.strip()).resolve()
    if candidate.is_file():
        return candidate
    for extension in _EXTENSIONS:
        with_extension = candidate.with_suffix(extension)
        if with_extension.is_file():
            return with_extension
    return None


def _rasterise(path: Path, dpi: int) -> bytes | None:
    """
    中文：把矢量图渲染成位图，好让 OCR 读得到。

    Render a vector figure to a bitmap.

    Args:
        path (Path): A PDF, EPS or PS file.
        dpi (int): Rendering resolution.

    Returns:
        bytes | None: PNG data, or ``None`` when the file cannot be opened.
    """
    try:
        import pymupdf
    except ImportError:  # pragma: no cover - pymupdf is a hard dependency
        return None
    try:
        with pymupdf.open(path) as document:
            if not document.page_count:
                return None
            return document[0].get_pixmap(dpi=dpi).tobytes("png")
    except Exception:  # noqa: BLE001
        logger.warning("渲染插图失败：%s", path.name, exc_info=True)
        return None


def translate_figures(document, translate_image, *, dpi: int = _DPI) -> dict:
    r"""
    中文：把文档里每一张插图翻译一遍，并把 \includegraphics 改指向译文版。

    Translate every figure the document includes.

    Args:
        document (LatexDocument): A parsed document. Its ``\includegraphics``
            calls are lifted as pieces of kind ``keep``, and this rewrites
            their source in place, so assembly picks the change up with no
            further work.
        translate_image (Callable[[bytes], bytes | None]): Takes image data and
            returns a PNG with the text translated, or ``None`` when there was
            nothing to translate. Supplied by the caller so this module needs
            neither the OCR engine nor a translator of its own.
        dpi (int): Resolution for rendering vector figures.

    Returns:
        dict: ``total`` figures seen, ``translated`` rewritten, ``skipped``
            left alone, and ``missing`` referenced but not present.

    Note:
        Each distinct file is translated once however often it is included, and
        a file already carrying a translated copy is not translated again — a
        second run over the same working directory costs nothing.
        每个文件无论被 include 几次都只翻一次；已经有译文版的文件不再翻第二遍——对同一
        个工作目录重跑一次不花任何代价。
    """
    root = document.root
    done: dict[Path, Path | None] = {}
    counts = {"total": 0, "translated": 0, "skipped": 0, "missing": 0}

    for piece in document.pieces:
        if piece.kind != "keep" or piece.name != "includegraphics":
            continue
        rewritten, changed = _rewrite(piece.source, root, done, translate_image,
                                      dpi, counts)
        if changed:
            piece.source = rewritten
    logger.info("插图翻译：%d 张里译了 %d 张，跳过 %d 张，缺失 %d 张",
                counts["total"], counts["translated"], counts["skipped"],
                counts["missing"])
    return counts


def _rewrite(source: str, root: Path, done: dict, translate_image, dpi: int,
             counts: dict) -> tuple[str, bool]:
    r"""
    中文：处理一处 \includegraphics，返回改写后的写法与是否改动过。

    Handle one ``\includegraphics`` call.

    Args:
        source (str): The call as written.
        root (Path): The source tree's root.
        done (dict): Files already handled, mapping the original to its
            translated copy or to ``None``.
        translate_image (Callable): As in :func:`translate_figures`.
        dpi (int): Rendering resolution for vector figures.
        counts (dict): Running totals, updated in place.

    Returns:
        tuple[str, bool]: The call, and whether it changed.
    """
    match = _INCLUDE.search(source)
    if match is None:
        return source, False
    counts["total"] += 1

    path = _resolve(root, match.group(4))
    if path is None:
        counts["missing"] += 1
        logger.debug("插图不在源码里：%s", match.group(4))
        return source, False

    if path not in done:
        done[path] = _translated_copy(path, translate_image, dpi)
        if done[path] is None:
            counts["skipped"] += 1
        else:
            counts["translated"] += 1
    replacement = done[path]
    if replacement is None:
        return source, False

    # Written relative to the root, and with forward slashes: that is what
    # graphicx expects, on every platform.
    # 写成相对根目录的路径，并且用正斜杠：graphicx 在任何平台上都要这个。
    relative = replacement.relative_to(root).as_posix()
    return source[:match.start()] + match.group(1) + match.group(2) + \
        match.group(3) + relative + match.group(5) + source[match.end():], True


def _translated_copy(path: Path, translate_image, dpi: int) -> Path | None:
    """
    中文：为一张图产出译文版，返回它的路径；没东西可译就返回 None。

    Produce the translated copy of one figure.

    Args:
        path (Path): The figure file.
        translate_image (Callable): As in :func:`translate_figures`.
        dpi (int): Rendering resolution for vector figures.

    Returns:
        Path | None: The translated copy, or ``None`` when the figure carries
            no recognisable text, when recognition or translation failed, or
            when the file cannot be read. Every one of those is a reason to
            keep the original figure, which is why they share an answer.
    """
    target = path.with_name(path.stem + _SUFFIX)
    if target.is_file():
        return target

    if path.suffix.lower() in _RASTER:
        try:
            data = path.read_bytes()
        except OSError:
            return None
    else:
        data = _rasterise(path, dpi)
        if data is None:
            return None

    try:
        png = translate_image(data)
    except Exception:  # noqa: BLE001
        logger.warning("插图翻译失败：%s", path.name, exc_info=True)
        return None
    if not png:
        return None

    target.write_bytes(png)
    logger.debug("插图译文写入 %s", target.name)
    return target
