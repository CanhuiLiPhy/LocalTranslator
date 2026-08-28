"""
中文：先便宜地判断一份 PDF 有没有可能是扫描件，不可能就跳过昂贵的扫描检测。

Decide cheaply whether a PDF could be a scan at all.

**What the expensive check does.** ``DetectScannedFile`` renders every page
twice — once as it is, once with the text removed — converts both to grey and
compares them with structural similarity. A page whose appearance does not
change when the text is taken away is an image with an invisible OCR layer
under it, which is what the OCR workaround exists for.

That is the right test and an expensive one: two rasters per page plus several
float intermediates inside ``structural_similarity``, through OpenCV and
scikit-image. Measured on a 19-page paper it costs **0.24 GB** of resident
memory, about a seventh of the whole parsing pipeline, on a document that could
not possibly be a scan.

**The cheap precondition.** A scanned page is a picture of a page. If no page
carries images covering a meaningful share of its area, no page can be a scan,
and the expensive comparison can only reach the answer we already have. Image
placements are listed in the page's resources, so this reads them without
rendering anything.

Nothing here decides that a document *is* a scan — only that it cannot be, in
which case the detection is skipped. Anything the cheap test is unsure about is
handed to the real one.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Share of a page's area that images must cover before the page is worth
# examining. A scan covers essentially all of it; a paper's figures, even a
# full-width one, stay well under half.
# 页面上图像需覆盖的面积比例，达到该比例才值得进一步检查。扫描件几乎覆盖整页；
# 论文插图即便通栏也远低于一半。
_COVERAGE = 0.5

# Share of the page the *largest single* image must also cover. Summed coverage
# alone is not enough: page 8 of arXiv:2608.19072 is a grid of heat maps whose
# images together cover 90% of the paper, and whose largest is 16% — a mosaic,
# not a picture of a page. Requiring one image to carry a quarter of the page
# separates the two, and still admits a scan delivered in up to four bands.
# 还要求*最大的单张*图像覆盖的比例。仅看合计覆盖不够：arXiv:2608.19072 第 8 页是一
# 组热力图，所有图像合计覆盖 90%，最大的一张只有 16%——那是拼贴，不是一整页的照片。
# 要求有一张图占到四分之一页即可区分两者，同时仍能接纳最多分成四条的扫描件。
_LARGEST = 0.25


def could_be_scanned(path: str) -> bool:
    """
    中文：判断这份 PDF 是否有可能是扫描件（有页面被图像大面积覆盖）。

    Report whether any page of a PDF is mostly covered by images.

    Args:
        path (str): Path to the PDF to inspect.

    Returns:
        bool: ``True`` when at least one page carries image placements covering
            :data:`_COVERAGE` of its area with a single image covering
            :data:`_LARGEST`, or when the file cannot be read — an unreadable
            answer must not silently disable a safety check.
    """
    try:
        import pymupdf

        with pymupdf.open(path) as doc:
            for page in doc:
                area = abs(page.rect.get_area())
                if area <= 0:
                    return True
                covered = 0.0
                largest = 0.0
                for info in page.get_image_info():
                    box = pymupdf.Rect(info["bbox"]) & page.rect
                    if not box.is_empty:
                        placed = abs(box.get_area())
                        covered += placed
                        largest = max(largest, placed)
                if covered >= area * _COVERAGE and largest >= area * _LARGEST:
                    return True
    except Exception:  # noqa: BLE001
        logger.debug("Could not inspect %s for scanned pages", path, exc_info=True)
        return True
    return False
