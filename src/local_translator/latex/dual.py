r"""
中文：把英文原版与中文译版拼成一份双语对照 PDF。

Build the bilingual PDF out of the two compiled documents.

The PDF pipeline gets its dual for free: it rewrites a page in place, so the
translated page 7 *is* page 7 and pairing the two is arithmetic. Here TeX lays
the translation out from scratch, and Chinese sets shorter than English, so the
two documents drift apart — a fourteen-page paper becomes thirteen and the
pairing slips by one somewhere in the middle.

The answer is not to pretend otherwise. The pages are paired by index, the
shorter document is padded with blank pages so the pairing never collapses into
a shift, and a reader who reaches the end of one side sees blank paper rather
than the wrong page. Drift within a paper is usually a fraction of a page,
because the same source produced both.

**Where the English half comes from.** Not from the original ``.tex``: that is
what the paper's own preamble compiles to, and this project rewrites the
preamble for XeTeX and CJK before compiling anything. Measured across seven
arXiv papers, three of the originals compile unmodified. The English half is
therefore the *same assembled document with the translations left out* — same
preamble, same patches, same engine, so it compiles whenever the translation
does, and the two halves differ in exactly one thing: the language.

**中文：** PDF 那条管线的双语版是白得的——它在原地重写页面，译文第 7 页**就是**第 7 页，
配对是道算术题。这里是 TeX 从头排版，中文又比英文排得短，两份文档会渐渐错开。

办法不是假装没这回事。按页序配对，短的一份补空白页，配对就不会塌成整体错位；读到一侧结尾
的人看到的是白纸，而不是错的那一页。同一份源码产出的两边，篇幅差通常不到一页。

英文那半不是拿原始 .tex 编的：那是论文自己的前言编出来的东西，而本项目在编译之前会为 XeTeX
与中文重写前言——七篇 arXiv 论文实测，原样能编过的只有三篇。英文半边用的是**同一份装配好的
文档、只是不放译文**：同样的前言、同样的补丁、同样的引擎，译文能编过它就能编过，两边只差一
件事——语言。
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def _blank_like(document, page_number: int):
    """
    中文：取一页作为空白页的尺寸参照。

    Pick the page whose size a blank filler should copy.

    Args:
        document (pymupdf.Document): The document being padded.
        page_number (int): Index of the page that has no counterpart.

    Returns:
        pymupdf.Rect: The rectangle to use, falling back to A4 when the
            document has no pages at all to copy from.
    """
    import pymupdf

    if document.page_count:
        return document[min(page_number, document.page_count - 1)].rect
    return pymupdf.paper_rect("a4")


def side_by_side(left_path, right_path, output, *, left_first: bool = True):
    """
    中文：把两份 PDF 逐页并排拼成一份，页宽翻倍。

    Place the two documents side by side, one spread per page.

    Args:
        left_path (Path): The document for the left half.
        right_path (Path): The document for the right half.
        output (Path): Where to write the result.
        left_first (bool): Kept for symmetry with the PDF pipeline's own
            option; the caller decides which document is which and passes them
            in that order, so this only records the intent.

    Returns:
        Path | None: ``output`` when a file was written, ``None`` when either
            input could not be opened — in which case the caller still has both
            halves and should say so rather than report a dual that is not
            there.
    """
    import pymupdf

    try:
        left = pymupdf.open(left_path)
        right = pymupdf.open(right_path)
    except Exception:  # noqa: BLE001
        logger.warning("拼双语版时打不开其中一份 PDF", exc_info=True)
        return None

    spread = pymupdf.open()
    try:
        for index in range(max(left.page_count, right.page_count)):
            left_rect = (left[index].rect if index < left.page_count
                         else _blank_like(left, index))
            right_rect = (right[index].rect if index < right.page_count
                          else _blank_like(right, index))
            width = left_rect.width + right_rect.width
            height = max(left_rect.height, right_rect.height)
            page = spread.new_page(width=width, height=height)

            if index < left.page_count:
                page.show_pdf_page(
                    pymupdf.Rect(0, 0, left_rect.width, left_rect.height),
                    left, index)
            if index < right.page_count:
                page.show_pdf_page(
                    pymupdf.Rect(left_rect.width, 0, width, right_rect.height),
                    right, index)
        spread.save(output, garbage=3, deflate=True)
    finally:
        spread.close()
        left.close()
        right.close()
    logger.info("双语并排：%s", output.name)
    return output


def alternating(first_path, second_path, output):
    """
    中文：把两份 PDF 交替编排——第一份的第 1 页、第二份的第 1 页、第一份的第 2 页……

    Interleave the two documents, one page from each in turn.

    Args:
        first_path (Path): The document whose page comes first in each pair.
        second_path (Path): The other one.
        output (Path): Where to write the result.

    Returns:
        Path | None: ``output``, or ``None`` when either input could not be
            opened.
    """
    import pymupdf

    try:
        first = pymupdf.open(first_path)
        second = pymupdf.open(second_path)
    except Exception:  # noqa: BLE001
        logger.warning("拼双语版时打不开其中一份 PDF", exc_info=True)
        return None

    merged = pymupdf.open()
    try:
        for index in range(max(first.page_count, second.page_count)):
            for document in (first, second):
                if index < document.page_count:
                    merged.insert_pdf(document, from_page=index, to_page=index)
                else:
                    rect = _blank_like(document, index)
                    merged.new_page(width=rect.width, height=rect.height)
        merged.save(output, garbage=3, deflate=True)
    finally:
        merged.close()
        first.close()
        second.close()
    logger.info("双语交替：%s", output.name)
    return output
