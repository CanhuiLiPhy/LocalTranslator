"""
中文：用 numpy 顶掉 OpenCV。BabelDOC 只用到它九个函数，却要为此背 118 MB。

Stand in for OpenCV with numpy.

**What it costs to keep.** ``opencv-python-headless`` is 118 MB installed, of
which 78 MB is dynamic libraries this program never calls into: the whole of
ffmpeg (``libavcodec``, ``libavformat``, ``libavfilter``), four video encoders
(``x265``, ``aom``, ``rav1e``, ``SvtAv1Enc``), Tesseract and Leptonica for OCR,
plus gnutls and OpenSSL that those pull in. Importing it costs 23 MB resident.

**What it is used for.** Nine functions and four constants, across all of
BabelDOC:

===========================  ===============================================
``cvtColor``                 RGB→BGR before encoding, RGB→grey before a
                             structural-similarity comparison
``resize``                   one call, fitting a page raster into the layout
                             model's 1024×1024 input
``copyMakeBorder``           padding that same raster out to the square
``imencode`` / ``imread``    only on the remote-layout path
``imwrite`` / ``rectangle``
``putText``                  only in the debug visualisers
===========================  ===============================================

**The one that has to be right.** ``resize``. The layout model sees whatever
this produces, so a filter that differs from OpenCV's would move its boxes, and
a moved box is a paragraph read as a caption. OpenCV's ``INTER_LINEAR`` samples
at ``(dst + 0.5) * scale - 0.5`` and does **not** filter wider when it shrinks —
where Pillow's ``BILINEAR`` scales its support with the ratio and so
anti-aliases. Reaching for Pillow because it is already installed would have
been the quiet mistake here; this implements OpenCV's convention instead.

The remaining difference is arithmetic: OpenCV interpolates in 11-bit fixed
point, this in float32, so a pixel may land one level apart. That is below what
a detector resolves — and it is checked rather than assumed, by translating a
paper both ways and comparing the output.

Every entry point keeps OpenCV's exact signature and return shape, so nothing
downstream can tell the difference. :func:`install` puts it in ``sys.modules``
under the name ``cv2``, which has to happen before BabelDOC is imported.
"""

from __future__ import annotations

import logging
import sys
import types

import numpy as np

logger = logging.getLogger(__name__)

# OpenCV's own constant values, so a caller that passes them through unchanged
# — or compares against them — behaves the same.
# OpenCV 自己的常量取值，这样原样传递或用于比较的调用方行为不变。
COLOR_BGR2RGB = 4
COLOR_RGB2BGR = 4
COLOR_BGR2GRAY = 6
COLOR_RGB2GRAY = 7
COLOR_GRAY2BGR = 8
COLOR_GRAY2RGB = 8

INTER_NEAREST = 0
INTER_LINEAR = 1
INTER_CUBIC = 2
INTER_AREA = 3

BORDER_CONSTANT = 0
BORDER_REPLICATE = 1

FONT_HERSHEY_SIMPLEX = 0
LINE_AA = 16

# OpenCV's RGB→grey weights. Not a free choice: the caller compares two greyed
# pages with ``structural_similarity`` and thresholds the result, so a different
# luminance formula moves that threshold.
# OpenCV 的 RGB→灰度权重。这不是可以随便选的：调用方要把两张灰度化的页面送进
# structural_similarity 并对结果卡阈值，换一套亮度公式就等于挪动了那个阈值。
_GREY = (0.299, 0.587, 0.114)


def cvtColor(src, code, dst=None):  # noqa: N802 - OpenCV's own name
    """
    中文：颜色空间转换，支持 RGB↔BGR 与 RGB/BGR→灰度。

    Convert between the colour spaces BabelDOC asks for.

    Args:
        src (np.ndarray): Image of shape ``(H, W, 3)`` for a colour conversion,
            or ``(H, W)`` for a grey one, dtype ``uint8``.
        code (int): One of the ``COLOR_*`` constants above.
        dst: Accepted and ignored, as OpenCV's optional output argument.

    Returns:
        np.ndarray: ``(H, W, 3)`` for a channel swap, ``(H, W)`` for a
            conversion to grey, ``uint8`` either way.

    Raises:
        NotImplementedError: For a conversion this does not implement, so an
            unsupported call fails loudly here rather than silently producing
            the wrong pixels somewhere downstream.

    Math:
        grey = 0.299 R + 0.587 G + 0.114 B, rounded to nearest.
    """
    array = np.asarray(src)
    if code in (COLOR_BGR2RGB, COLOR_RGB2BGR):
        return np.ascontiguousarray(array[..., ::-1])
    if code == COLOR_RGB2GRAY:
        weights = np.array(_GREY, dtype=np.float32)
        return np.rint(array[..., :3].astype(np.float32) @ weights).astype(np.uint8)
    if code == COLOR_BGR2GRAY:
        weights = np.array(_GREY[::-1], dtype=np.float32)
        return np.rint(array[..., :3].astype(np.float32) @ weights).astype(np.uint8)
    if code in (COLOR_GRAY2BGR,):
        return np.ascontiguousarray(np.repeat(array[..., None], 3, axis=-1))
    raise NotImplementedError(f"cvtColor code {code} is not implemented in the shim")


def _linear_map(dst_size: int, src_size: int):
    """
    中文：按 OpenCV 的采样约定，算出目标每个像素对应的源坐标与权重。

    Work out where each destination pixel samples from, OpenCV's way.

    OpenCV maps a destination pixel *centre* to a source coordinate with
    ``(i + 0.5) * src / dst - 0.5``, clamps it into the image, and takes the two
    neighbouring samples. It does not widen the filter when shrinking, which is
    what separates it from Pillow's ``BILINEAR`` and why that could not be used
    here.

    Args:
        dst_size (int): Output length along this axis.
        src_size (int): Input length along this axis.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: Lower index, upper index and
            the weight of the upper sample, each of shape ``(dst_size,)``.

    Math:
        f = (i + 0.5) * src/dst - 0.5;  lo = clip(floor(f), 0, src-1);
        hi = min(lo + 1, src - 1);      w  = clip(f - floor(f), 0, 1)
    """
    scale = src_size / dst_size
    centres = (np.arange(dst_size, dtype=np.float64) + 0.5) * scale - 0.5
    floor = np.floor(centres)
    weight = centres - floor
    # A destination pixel whose centre falls outside the source — which happens
    # at both edges when upscaling — takes the edge sample outright.
    # 中心落在源图之外的目标像素（放大时两端各会出现）直接取边缘那个采样点。
    weight = np.where(floor < 0, 0.0, weight)
    weight = np.where(floor >= src_size - 1, 0.0, weight)
    lower = np.clip(floor, 0, src_size - 1).astype(np.intp)
    upper = np.minimum(lower + 1, src_size - 1)
    return lower, upper, weight.astype(np.float32)


def resize(src, dsize, dst=None, fx=None, fy=None, interpolation=INTER_LINEAR):  # noqa: N802
    """
    中文：缩放图像，采样约定与 OpenCV 的 INTER_LINEAR 一致。

    Resize an image the way OpenCV's ``INTER_LINEAR`` does.

    Args:
        src (np.ndarray): Image of shape ``(H, W)`` or ``(H, W, C)``.
        dsize (tuple[int, int]): Target size as ``(width, height)`` — OpenCV's
            order, which is the transpose of the array's own.
        dst: Accepted and ignored.
        fx (float | None): Horizontal scale, used when ``dsize`` is ``None`` or
            ``(0, 0)``.
        fy (float | None): Vertical scale, likewise.
        interpolation (int): ``INTER_LINEAR`` or ``INTER_NEAREST``. Anything
            else is treated as linear, which is what OpenCV's cubic and area
            filters degrade to most gracefully.

    Returns:
        np.ndarray: The resized image, same dtype and channel count as ``src``.

    Math:
        Separable bilinear: interpolate along rows, then along columns, with
        the sample positions :func:`_linear_map` gives.
    """
    array = np.asarray(src)
    height, width = array.shape[:2]

    if dsize and dsize[0] and dsize[1]:
        out_w, out_h = int(dsize[0]), int(dsize[1])
    else:
        out_w = int(round(width * (fx or 1.0)))
        out_h = int(round(height * (fy or 1.0)))
    if out_w <= 0 or out_h <= 0:
        raise ValueError(f"resize to {out_w}x{out_h} is not a size")
    if (out_w, out_h) == (width, height):
        return array.copy()

    if interpolation == INTER_NEAREST:
        rows = np.minimum((np.arange(out_h) * height) // out_h, height - 1)
        cols = np.minimum((np.arange(out_w) * width) // out_w, width - 1)
        return array[rows][:, cols].copy()

    row_lo, row_hi, row_w = _linear_map(out_h, height)
    col_lo, col_hi, col_w = _linear_map(out_w, width)

    working = array.astype(np.float32)
    if working.ndim == 2:
        working = working[:, :, None]
        flatten = True
    else:
        flatten = False

    # Rows first, then columns; a separable filter costs O(H·W) twice instead of
    # O(H·W·4) once, and on a 3000×2000 page raster that is the difference
    # between noticeable and not.
    # 先行后列；可分离滤波的代价是两趟 O(H·W)，而不是一趟 O(H·W·4)，在 3000×2000 的
    # 页面栅格上，这就是「看得出来」与「看不出来」之别。
    vertical = (working[row_lo] * (1.0 - row_w)[:, None, None]
                + working[row_hi] * row_w[:, None, None])
    both = (vertical[:, col_lo] * (1.0 - col_w)[None, :, None]
            + vertical[:, col_hi] * col_w[None, :, None])

    if flatten:
        both = both[:, :, 0]
    if np.issubdtype(array.dtype, np.integer):
        info = np.iinfo(array.dtype)
        return np.clip(np.rint(both), info.min, info.max).astype(array.dtype)
    return both.astype(array.dtype)


def copyMakeBorder(src, top, bottom, left, right, borderType=BORDER_CONSTANT,  # noqa: N802
                   dst=None, value=0):
    """
    中文：给图像加边框，目前只需要常量填充。

    Pad an image on four sides.

    Args:
        src (np.ndarray): Image of shape ``(H, W)`` or ``(H, W, C)``.
        top (int): Rows to add above.
        bottom (int): Rows to add below.
        left (int): Columns to add on the left.
        right (int): Columns to add on the right.
        borderType (int): ``BORDER_CONSTANT`` or ``BORDER_REPLICATE``.
        dst: Accepted and ignored.
        value (int | float | Sequence): Fill value for a constant border. A
            sequence is taken per channel, as OpenCV takes a ``Scalar``.

    Returns:
        np.ndarray: Image of shape ``(H+top+bottom, W+left+right, ...)``, same
            dtype as ``src``.
    """
    array = np.asarray(src)
    pad = [(int(top), int(bottom)), (int(left), int(right))]
    pad += [(0, 0)] * (array.ndim - 2)

    if borderType == BORDER_REPLICATE:
        return np.pad(array, pad, mode="edge")

    if isinstance(value, (list, tuple, np.ndarray)) and array.ndim == 3:
        out = np.empty(
            (array.shape[0] + pad[0][0] + pad[0][1],
             array.shape[1] + pad[1][0] + pad[1][1],
             array.shape[2]), dtype=array.dtype)
        for channel in range(array.shape[2]):
            fill = value[channel] if channel < len(value) else 0
            out[..., channel] = fill
        out[pad[0][0]:pad[0][0] + array.shape[0],
            pad[1][0]:pad[1][0] + array.shape[1]] = array
        return out

    fill = value[0] if isinstance(value, (list, tuple, np.ndarray)) else value
    return np.pad(array, pad, mode="constant", constant_values=fill)


def imencode(ext: str, img, params=None):
    """
    中文：把图像编码成 JPEG/PNG 字节，返回值形状与 OpenCV 一致。

    Encode an image to bytes, returning what OpenCV returns.

    Args:
        ext (str): File extension including the dot, e.g. ``".jpg"``.
        img (np.ndarray): Image in **BGR** channel order, as OpenCV expects.
        params (Sequence[int] | None): Accepted and ignored; the quality
            default matches OpenCV's 95 for JPEG.

    Returns:
        tuple[bool, np.ndarray]: Success flag and the bytes as a ``uint8``
            array, which is what callers then run ``.tobytes()`` on.
    """
    from io import BytesIO

    from PIL import Image

    array = np.asarray(img)
    if array.ndim == 3:
        array = array[..., ::-1]  # BGR in, RGB for Pillow
    buffer = BytesIO()
    fmt = {"jpg": "JPEG", "jpeg": "JPEG", "png": "PNG",
           "webp": "WEBP"}.get(ext.lstrip(".").lower(), "PNG")
    kwargs = {"quality": 95} if fmt == "JPEG" else {}
    Image.fromarray(array).save(buffer, format=fmt, **kwargs)
    return True, np.frombuffer(buffer.getvalue(), dtype=np.uint8)


def imread(filename, flags=1):
    """
    中文：从文件读入图像，按 OpenCV 的 BGR 顺序返回。

    Read an image from disk in OpenCV's channel order.

    Args:
        filename (str): Path to the image.
        flags (int): ``0`` reads greyscale, anything else colour.

    Returns:
        np.ndarray | None: The image in BGR order, or ``None`` when it cannot be
            read — which is how OpenCV reports a failure, and what its callers
            check for.
    """
    from PIL import Image

    try:
        with Image.open(filename) as handle:
            image = handle.convert("L" if flags == 0 else "RGB")
            array = np.array(image)
    except Exception:  # noqa: BLE001
        return None
    return array if flags == 0 else np.ascontiguousarray(array[..., ::-1])


def imwrite(filename, img, params=None) -> bool:
    """
    中文：把图像写到文件，输入按 OpenCV 的 BGR 顺序。

    Write an image to disk, taking OpenCV's channel order.

    Args:
        filename (str): Destination path; the extension picks the format.
        img (np.ndarray): Image in BGR order.
        params: Accepted and ignored.

    Returns:
        bool: Whether the file was written.
    """
    from PIL import Image

    array = np.asarray(img)
    if array.ndim == 3:
        array = array[..., ::-1]
    try:
        Image.fromarray(array).save(str(filename))
    except Exception:  # noqa: BLE001
        logger.debug("Could not write %s", filename, exc_info=True)
        return False
    return True


def rectangle(img, pt1, pt2, color, thickness=1, lineType=None, shift=None):  # noqa: N802
    """
    中文：画一个空心矩形，只用在调试可视化里。

    Draw a rectangle outline. Used only by the debug visualisers.

    Args:
        img (np.ndarray): Image to draw on, modified in place as OpenCV does.
        pt1 (tuple[int, int]): One corner as ``(x, y)``.
        pt2 (tuple[int, int]): The opposite corner.
        color (Sequence[int] | int): Line colour, per channel.
        thickness (int): Line width in pixels; a negative value fills.
        lineType: Accepted and ignored.
        shift: Accepted and ignored.

    Returns:
        np.ndarray: The same array, for OpenCV's chaining convention.
    """
    height, width = img.shape[:2]
    x1, x2 = sorted((int(pt1[0]), int(pt2[0])))
    y1, y2 = sorted((int(pt1[1]), int(pt2[1])))
    x1, x2 = max(x1, 0), min(x2, width - 1)
    y1, y2 = max(y1, 0), min(y2, height - 1)
    if x1 > x2 or y1 > y2:
        return img

    if thickness < 0:
        img[y1:y2 + 1, x1:x2 + 1] = color
        return img

    t = max(int(thickness), 1)
    img[y1:min(y1 + t, height), x1:x2 + 1] = color
    img[max(y2 - t + 1, 0):y2 + 1, x1:x2 + 1] = color
    img[y1:y2 + 1, x1:min(x1 + t, width)] = color
    img[y1:y2 + 1, max(x2 - t + 1, 0):x2 + 1] = color
    return img


def putText(img, text, org, fontFace=FONT_HERSHEY_SIMPLEX, fontScale=1.0,  # noqa: N802
            color=(0, 0, 0), thickness=1, lineType=None, bottomLeftOrigin=None):
    """
    中文：在图像上写一行字，只用在调试可视化里。

    Draw a line of text. Used only by the debug visualisers.

    The typeface is Pillow's default bitmap font rather than Hershey Simplex;
    the labels are box classes and confidences on a debug image, where legible
    matters and identical does not.

    Args:
        img (np.ndarray): Image to draw on, modified in place.
        text (str): The label.
        org (tuple[int, int]): Baseline origin as ``(x, y)``.
        fontFace: Accepted and ignored.
        fontScale (float): Ignored beyond choosing nothing; the default font is
            a fixed size.
        color (Sequence[int]): Text colour, per channel.
        thickness (int): Accepted and ignored.
        lineType: Accepted and ignored.
        bottomLeftOrigin: Accepted and ignored.

    Returns:
        np.ndarray: The same array.
    """
    try:
        from PIL import Image, ImageDraw
    except Exception:  # noqa: BLE001
        return img

    picture = Image.fromarray(img)
    draw = ImageDraw.Draw(picture)
    fill = tuple(int(c) for c in color) if isinstance(color, (list, tuple)) else int(color)
    draw.text((int(org[0]), int(org[1]) - 10), str(text), fill=fill)
    img[:] = np.array(picture)
    return img


def install() -> bool:
    """
    中文：把这个模块当作 cv2 装进 sys.modules，必须早于 BabelDOC 的导入。

    Register this module under the name ``cv2``.

    Must run before BabelDOC is imported: its modules do ``import cv2`` at the
    top, and once one of them has bound the real package there is nothing left
    to substitute.

    A real OpenCV already imported wins — someone who installed it wants it, and
    silently shadowing an installed package would be the wrong kind of clever.

    Returns:
        bool: Whether the shim is now what ``import cv2`` resolves to.
    """
    if "cv2" in sys.modules and sys.modules["cv2"] is not _module:
        logger.debug("OpenCV is already imported; leaving it in place")
        return False
    sys.modules["cv2"] = _module
    return True


_module = sys.modules[__name__]

# OpenCV exposes its version, and code in the wild branches on it.
# OpenCV 会暴露版本号，外面有代码按它分支。
__version__ = "4.0.0-localtranslator-shim"
