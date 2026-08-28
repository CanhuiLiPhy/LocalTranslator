"""
中文：输出 PDF 的体积压缩——把分辨率过剩的插图降到目标 DPI，其余原样保留。

Shrink the finished PDF by resampling images that carry far more resolution
than they can display.

**Where the weight is.** Measured on arXiv:2510.04704, a 27-page paper: the
finished document held 14.3 MB, of which 12.4 MB was images and 0.6 MB fonts.
Those images are not large on the page — one is 1600x1600 pixels shown in a
43-point box, which is 2673 DPI. The median across the document's 72 images is
2667 DPI and the maximum 4468. The same figures at the same resolution are
present in the *source* paper, so this is inherited from how the authors
exported them rather than anything the translation pipeline does.

**What this changes and what it does not.** Only the sampling rate: an image is
redrawn at the target DPI and re-encoded losslessly as PNG. Nothing is quantised
and no lossy codec is involved, so the only difference is resolution the display
size could never have shown. Measured against the uncompressed output, rendering
both at 150 DPI: mean per-channel difference 0.06/255, with 0.21% of pixels
differing by more than 8. The file goes from 14.3 MB to 3.7 MB.

**JPEG was tried and rejected.** At quality 92 the same document came out at
3.55 MB against PNG's 3.56 MB — a 0.3% saving for a lossy codec on line art,
where ringing around text is exactly the artefact to avoid. Where the saving is
this close, lossless wins on principle.

**Soft masks are the trap.** 130 of this document's 144 image objects carry an
``SMask`` holding their transparency. Replacing the base image without
recombining the mask first produces a file that looks plausible in a size
report and is visibly destroyed on the page — figures come back as opaque black
rectangles. The first implementation did exactly that, and the numeric measure
did not catch it: a mean difference of 15/255 reads like the ordinary cost of
resampling until the page is rendered and looked at. :func:`_resample` combines
the mask before scaling for that reason.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# How far above the target an image must sit before it is worth redrawing.
# Resampling costs time and always perturbs antialiasing slightly, so an image
# already close to the target is left alone.
# 图像要高出目标多少才值得重绘。重采样有时间成本，且总会轻微改变抗锯齿，因此已经接近目标
# 的图像不予处理。
_MARGIN = 1.15

# Save options for the compressed result. `clean` and `use_objstms` together
# take the same document from 14.88 MB to 14.29 MB before any image is touched
# — object streams and a rebuilt cross-reference table, at a cost of 0.2 s.
# 压缩结果的保存选项。在动任何图像之前，clean 与 use_objstms 二者就能把同一份文档从
# 14.88 MB 降到 14.29 MB——对象流加重建的交叉引用表，代价 0.2 秒。
_SAVE = {
    "garbage": 4,
    "deflate": True,
    "deflate_images": True,
    "deflate_fonts": True,
    "clean": True,
    "use_objstms": 1,
}


def _resample(document, page, xref: int, smask: int, scale: float) -> bool:
    """
    中文：把一张图按比例重绘，先合并透明遮罩再缩放。

    Redraw one image at a fraction of its current size.

    Args:
        document (pymupdf.Document): Document holding the image.
        page (pymupdf.Page): Page the image is drawn on; the replacement is
            issued through the page because that is where pymupdf exposes it.
        xref (int): Cross-reference number of the base image.
        smask (int): Cross-reference number of its soft mask, or ``0``. When
            present it must be recombined before scaling — see the module note.
        scale (float): Factor to multiply both dimensions by, below 1.

    Returns:
        bool: ``True`` when the image was replaced.
    """
    import pymupdf

    pixels = pymupdf.Pixmap(document, xref)
    if smask:
        pixels = pymupdf.Pixmap(pixels, pymupdf.Pixmap(document, smask))
    width = max(1, int(pixels.width * scale))
    height = max(1, int(pixels.height * scale))
    if width >= pixels.width or height >= pixels.height:
        return False
    smaller = pymupdf.Pixmap(pixels, width, height, None)
    page.replace_image(xref, stream=smaller.tobytes("png"))
    return True


def compress(path: Path, target_dpi: int) -> dict[str, int | float]:
    """
    中文：就地压缩一个 PDF，把超出目标分辨率的图降下来。

    Compress a PDF in place, resampling over-resolution images.

    Writes to a temporary file beside the original and moves it into place, so
    a failure part-way leaves the original intact rather than truncated.

    Args:
        path (Path): PDF to compress, rewritten in place.
        target_dpi (int): Effective resolution to aim for, in dots per inch.
            300 keeps print quality; 200 suits screen reading. A value of 0 or
            less disables the pass entirely.

    Returns:
        dict: ``before`` and ``after`` byte counts, ``images`` resampled, and
            ``saved`` as a fraction of the original size. All zero when the
            pass was skipped or the file could not be read.
    """
    empty = {"before": 0, "after": 0, "images": 0, "saved": 0.0}
    if target_dpi <= 0 or not path.is_file():
        return empty

    import pymupdf

    before = path.stat().st_size
    document = pymupdf.open(path)
    changed = 0
    try:
        seen: set[int] = set()
        for number in range(document.page_count):
            page = document[number]
            for info in page.get_images(full=True):
                xref, smask = info[0], info[1]
                if xref in seen:
                    continue
                seen.add(xref)
                rectangles = page.get_image_rects(xref)
                if not rectangles or rectangles[0].width < 1:
                    continue
                try:
                    image = document.extract_image(xref)
                except Exception:  # noqa: BLE001
                    continue
                # Effective resolution is pixels across divided by the width it
                # is actually drawn at, which is the only number that matters —
                # an image's own pixel count says nothing without it.
                # 有效分辨率是横向像素数除以它实际被画出来的宽度，这是唯一有意义的数字
                # ——脱离显示尺寸，一张图自身的像素数说明不了任何问题。
                dpi = image["width"] / (rectangles[0].width / 72)
                if dpi <= target_dpi * _MARGIN:
                    continue
                try:
                    if _resample(document, page, xref, smask, target_dpi / dpi):
                        changed += 1
                except Exception:  # noqa: BLE001
                    # One awkward image must not cost the whole pass; the
                    # original stays in place and the rest still shrink.
                    # 单张图出问题不该拖垮整趟；那一张保持原样，其余照常压缩。
                    logger.debug("Could not resample image %d", xref, exc_info=True)

        temporary = path.with_suffix(".compress.pdf")
        document.save(temporary, **_SAVE)
    finally:
        if not document.is_closed:
            document.close()

    after = temporary.stat().st_size
    # Keep whichever is smaller. Object streams and a rebuilt xref usually win,
    # but a document with no over-resolution images and little redundancy can
    # come out marginally larger, and there is no reason to accept that.
    # 保留较小的那一份。对象流与重建的 xref 通常更小，但一份没有分辨率过剩的图、冗余也少
    # 的文档可能反而略大，没有理由接受这种结果。
    if after < before:
        temporary.replace(path)
    else:
        temporary.unlink(missing_ok=True)
        after = before
    return {
        "before": before,
        "after": after,
        "images": changed,
        "saved": 1.0 - after / before if before else 0.0,
    }
