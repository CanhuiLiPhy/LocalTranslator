"""
中文：把译文原位画回图片——每行译文覆盖到原文所在的方框里，做成对照图。

Draw the translation back onto the image, in place.

Line-by-line translations carry coordinates from OCR, so each one can be
painted over the box the original text occupied. The result is the same picture
with Chinese where the source language was — the arrangement a reader wants for
a figure or a screenshot, where a line pulled out into a list loses the spatial
meaning the layout carried.

**Fitting text to a box.** The translated line rarely has the width of the
original. The font size is chosen to fill the box height, then reduced until the
line also fits the box width, down to a floor below which it would be unreadable;
past the floor the text is left at the floor size and allowed to run, because a
too-small-but-present translation is more useful than none. The original text is
covered with a rectangle whose colour is sampled from the box's border, so the
patch blends into a coloured panel rather than always being white.

**Font.** PyMuPDF ships a Droid Sans Fallback face under the name ``china-s``
that covers Simplified Chinese, so no font file has to be located on the host
and the renderer works the same on any machine.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Simplified-Chinese font bundled with PyMuPDF; no host font file needed.
# PyMuPDF 自带的简体中文字体，无需在宿主机上寻找字体文件。
_FONT = "china-s"

# The overlay text starts at the size that fills this fraction of the box height
# and shrinks from there to fit the width.
# 覆盖文字从填满该比例框高的字号起步，再按宽度缩小。
_HEIGHT_FILL = 0.82

# Never shrink below this many points; past it the text is left to overflow
# rather than become unreadable.
# 字号不缩到该值以下；再小就任其溢出，也不让它小到读不清。
_MIN_FONTSIZE = 6.0

# Padding inside the box before the text, in points.
# 框内文字的内边距（点）。
_PAD = 1.0


def _sample_background(pixels, box, width, height):
    """
    中文：从方框边缘采样背景色，用来覆盖原文而不突兀。

    Sample a background colour from the box's edge.

    The mean of the pixels just outside the text usually is the panel or paper
    the text sits on, which makes the cover patch disappear into a coloured
    background instead of stamping a white block onto it.

    Args:
        pixels (numpy.ndarray): Image as ``(H, W, 3)`` uint8, RGB.
        box (tuple[float, float, float, float]): ``(x0, y0, x1, y1)``.
        width (int): Image width.
        height (int): Image height.

    Returns:
        tuple[float, float, float]: RGB in ``[0, 1]``; white when sampling fails.
    """
    import numpy as np

    x0, y0, x1, y1 = box
    x0i, y0i = max(0, int(x0)), max(0, int(y0))
    x1i, y1i = min(width, int(x1)), min(height, int(y1))
    if x1i <= x0i or y1i <= y0i:
        return (1.0, 1.0, 1.0)

    strips = []
    top = pixels[max(0, y0i - 2):y0i, x0i:x1i]
    bottom = pixels[y1i:min(height, y1i + 2), x0i:x1i]
    for strip in (top, bottom):
        if strip.size:
            strips.append(strip.reshape(-1, strip.shape[-1]))
    if not strips:
        patch = pixels[y0i:y1i, x0i:x1i].reshape(-1, pixels.shape[-1])
    else:
        patch = np.concatenate(strips, axis=0)
    if not patch.size:
        return (1.0, 1.0, 1.0)
    # The median rejects the dark text pixels that a mean would smear in.
    # 用中位数排除深色文字像素，均值会把它们混进来。
    rgb = np.median(patch[:, :3], axis=0) / 255.0
    return (float(rgb[0]), float(rgb[1]), float(rgb[2]))


def _text_colour(background):
    """
    中文：按背景明暗选黑或白字，保证对比度。

    Choose black or white text for contrast against the background.

    Args:
        background (tuple[float, float, float]): Background RGB in ``[0, 1]``.

    Returns:
        tuple[float, float, float]: Text RGB in ``[0, 1]``.
    """
    r, g, b = background
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return (0.1, 0.1, 0.1) if luminance > 0.5 else (0.96, 0.96, 0.96)


def render_overlay(image: bytes, lines) -> bytes | None:
    """
    中文：生成一张「译文原位覆盖」的图片。

    Produce a copy of the image with each line's translation drawn in place.

    Args:
        image (bytes): Original image, any format PyMuPDF can open.
        lines (list[dict]): Recognised lines, each ``{text, translation, box}``
            where ``box`` is four ``(x, y)`` corner points in image pixels.
            Lines without a ``translation`` are skipped and left as the original.

    Returns:
        bytes | None: PNG bytes of the rendered image, or ``None`` when the
            image cannot be opened.
    """
    import numpy as np
    import pymupdf

    try:
        # A Pixmap decodes to the image's native pixels, which is the coordinate
        # space OCR reported its boxes in. Page.get_pixmap() would instead
        # rasterise at 72 DPI and rescale a 96-DPI PNG to 3/4 size, leaving every
        # box misaligned — the bug that let the original text show through.
        # Pixmap 解码到图片的原生像素，正是 OCR 报告框坐标所用的空间。改用
        # Page.get_pixmap() 会按 72 DPI 光栅化、把 96 DPI 的 PNG 缩到 3/4，导致每个框
        # 错位——这正是原文透出来的根源。
        base_pix = pymupdf.Pixmap(image)
    except Exception:  # noqa: BLE001
        logger.warning("Could not open image for overlay", exc_info=True)
        return None

    width, height = base_pix.width, base_pix.height
    if width == 0 or height == 0:
        return None

    channels = base_pix.n
    buffer = np.frombuffer(base_pix.samples, dtype=np.uint8).reshape(height, width, channels)
    pixels = buffer[:, :, :3] if channels >= 3 else np.repeat(buffer, 3, axis=2)

    # Build the output as a PDF page the size of the native image, so a box at
    # image pixel (x, y) lands at page point (x, y). Rendering back to a pixmap
    # at the end flattens everything to PNG.
    # 把输出建成与原生图片同尺寸的 PDF 页，使图片像素 (x, y) 处的框落在页面点 (x, y)。
    # 最后重新渲染成 pixmap 展平为 PNG。
    out = pymupdf.open()
    out_page = out.new_page(width=width, height=height)
    rect = pymupdf.Rect(0, 0, width, height)
    out_page.insert_image(rect, stream=image)

    font = pymupdf.Font(_FONT)
    drawn = 0

    for line in lines:
        translation = (line.get("translation") or "").strip()
        if not translation:
            continue
        if translation == (line.get("text") or "").strip():
            # The translation is the original: a model name, an axis unit, a
            # term the whitelist holds in English. Covering the original with a
            # copy of itself can only make it worse — the patch is redrawn at
            # whatever size fits the box, which is how 「Model」 came out as
            # 「M o d e l」 across a figure. Leaving the pixels alone keeps them
            # exactly as the author drew them.
            # 译文就是原文：模型名、坐标轴单位、被白名单留在英文的术语。用它自己盖住
            # 自己只会更糟——补丁会按框的大小重新排一遍，图上的 Model 就是这样变成
            # 「M o d e l」的。不动那些像素，它们就还是作者画出来的样子。
            continue
        box = line.get("box")
        if not box or len(box) < 4:
            continue

        xs = [float(point[0]) for point in box]
        ys = [float(point[1]) for point in box]
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
        box_w, box_h = x1 - x0, y1 - y0
        if box_w < 4 or box_h < 4:
            continue

        background = _sample_background(pixels, (x0, y0, x1, y1), width, height)
        # Pad the cover slightly beyond the tight text box so the ascenders and
        # descenders of the original line do not peek out above and below the
        # patch. Kept to a small fraction of the box height so it cannot reach
        # into a neighbouring line.
        # 覆盖块比紧贴文字的框略微外扩，避免原文行的上伸/下伸笔画从补丁上下露出。
        # 外扩仅取框高的一小部分，不会侵入相邻行。
        pad_y = min(box_h * 0.18, 4.0)
        out_page.draw_rect(
            pymupdf.Rect(x0 - 1, y0 - pad_y, x1 + 1, y1 + pad_y),
            color=background, fill=background, width=0,
        )

        # Choose the largest size that fits the box width, bounded by height.
        # 选取在框宽内能放下的最大字号，上界由框高决定。
        size = max(_MIN_FONTSIZE, box_h * _HEIGHT_FILL)
        text_w = font.text_length(translation, fontsize=size)
        available = box_w - 2 * _PAD
        if text_w > available:
            size = max(_MIN_FONTSIZE, size * available / text_w)

        colour = _text_colour(background)
        # Baseline: sit the text on the box, nudged up so descenders stay inside.
        # 基线：让文字坐在框内，略微上移使下伸部分不出框。
        baseline_y = y1 - max(1.0, box_h * 0.18)
        try:
            out_page.insert_text(
                pymupdf.Point(x0 + _PAD, baseline_y),
                translation,
                fontname=_FONT,
                fontfile=None,
                fontsize=size,
                color=colour,
            )
            drawn += 1
        except Exception:  # noqa: BLE001
            logger.debug("Could not draw overlay line %r", translation[:20], exc_info=True)

    result_pix = out_page.get_pixmap(matrix=pymupdf.Matrix(1.5, 1.5))
    png = result_pix.tobytes("png")
    out.close()
    base_pix = None
    logger.debug("Rendered overlay with %d translated line(s)", drawn)
    return png
