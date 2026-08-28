"""
中文：按 Form XObject 的 BBox 裁掉界外内容——PDF 规范要求如此，而中间表示没做。

Apply a Form XObject's BBox as the clip the PDF specification says it is.

**What the specification says.** A form XObject's ``BBox`` entry is not a hint
about where its content happens to sit: it is a clipping boundary, and anything
the form draws outside it is not painted. Every reader honours this.

**What the intermediate representation did instead.** It projected the form's
contents into the page whether or not they fell inside the BBox. Content a
reader never sees therefore arrived in the IL as ordinary page content, at
coordinates where nothing is drawn.

**What that costs.** Measured on page 4 of arXiv:2608.17760, whose figure is a
form XObject with ``BBox [684.599 165.535 1064.22 411.863]``. Its stream
positions six runs of text; three sit inside the BBox and three do not:

    inside   Tm 751.12 332.60  →  page x=382.9   drawn
             Tm 819.38 172.10  →  page x=413.1   drawn
             Tm 971.07 217.28  →  page x=480.3   drawn
    outside  Tm 237.38 322.08  →  page x=155.2   not drawn, but in the IL
             Tm 310.50 140.78  →  page x=187.6   not drawn, but in the IL
             Tm 448.88 153.72  →  page x=249.0   not drawn, but in the IL

The three outside runs carry the same labels as the three inside — Target,
Closest learnware, General solution — so the figure's caption text appeared
twice in the IL: once in the right column where it is drawn, and once in the
*left* column where it is not. The invisible copy was gathered into a paragraph
spanning x=144 to x=354, came to rest under a left-column paragraph, and the
overlap pass lifted that paragraph's box from 60pt to 20pt to clear it. The
text was then typeset at 0.65 against 9.96pt everywhere else on the page.

**Why the test is on the page-space box.** The IL stores each element already
transformed, and ``on_xobj_begin`` records the same BBox transformed by the same
matrix. Comparing the two needs no arithmetic of our own — and arithmetic of our
own is exactly what should be avoided here, since the transform chain that
produced these coordinates is the thing under suspicion.
"""

from __future__ import annotations

import json
import logging
import os

logger = logging.getLogger(__name__)

# Set LOCALTRANS_XOBJCLIP_PROBE to a file path to have every clipped element's
# page-space box appended there as JSON lines. What this module removes is by
# definition content no reader draws, which makes the claim unfalsifiable from
# the output alone: the record is what lets a regression re-render each dropped
# box against the source page and confirm nothing visible went with it.
# 把 LOCALTRANS_XOBJCLIP_PROBE 设成一个文件路径，即可把每个被裁元素的页面坐标框以
# JSON 行追加进去。本模块删掉的按定义就是阅读器不绘制的内容，光看产物无法证伪；有了这份
# 记录，回归测试才能拿每个被裁的框回到原文页面上重渲染，确认没有连可见内容一起裁掉。
_PROBE = os.getenv("LOCALTRANS_XOBJCLIP_PROBE")

# How far outside its BBox an element may sit and still be kept, in points.
# Not zero: a glyph's box is its ink extent, and a letter resting on the
# boundary can round a fraction of a point past it. Content the specification
# actually clips is out by tens of points, not by one.
# 元素允许超出 BBox 多少仍予保留（点）。不取零：字形的框是墨迹范围，压在边界上的字母
# 可能因舍入越界零点几个点。真正被规范裁掉的内容超出的是几十个点，不是一个点。
_SLACK = 1.0


def _outside(box, bounds) -> bool:
    """
    中文：判断一个元素是否整体落在某个 XObject 的框之外。

    Report whether an element lies wholly outside a bounding box.

    Args:
        box (Box): The element's box, in page coordinates.
        bounds (tuple[float, float, float, float]): The XObject's page-space
            box as ``(x, y, x2, y2)``, already normalised.

    Returns:
        bool: ``True`` when the two do not overlap at all. An element that
            straddles the boundary is kept: the specification clips it, but
            clipping half a glyph is not something this can express, and
            dropping it outright would lose text a reader can see.
    """
    x, y, x2, y2 = bounds
    return (box.x2 < x - _SLACK or box.x > x2 + _SLACK
            or box.y2 < y - _SLACK or box.y > y2 + _SLACK)


def _bounds(xobject) -> tuple[float, float, float, float] | None:
    """
    中文：取出 XObject 的页面坐标框，并把可能翻转的坐标归位。

    Return an XObject's page-space box, normalised.

    The box is built by transforming two opposite corners, so a matrix that
    flips an axis — and the figure that prompted this module uses one — leaves
    ``y`` above ``y2``. Comparing against it unnormalised excludes everything.

    Args:
        xobject (PdfXobject): The XObject to read.

    Returns:
        tuple | None: ``(x, y, x2, y2)`` with ``x <= x2`` and ``y <= y2``, or
            ``None`` when the XObject carries no usable box.
    """
    box = getattr(xobject, "box", None)
    if box is None:
        return None
    values = (box.x, box.y, box.x2, box.y2)
    if any(v is None for v in values):
        return None
    return (min(box.x, box.x2), min(box.y, box.y2),
            max(box.x, box.x2), max(box.y, box.y2))


def _prune(page) -> tuple[int, int]:
    """
    中文：删掉页面上落在其所属 XObject 框外的字符与曲线。

    Drop the page's characters and curves that fall outside their own XObject.

    Args:
        page (Page): The page, at the point where the IL for it is complete.

    Returns:
        tuple[int, int]: How many characters and curves were dropped.
    """
    record = [] if _PROBE else None
    bounds = {}
    for xobject in getattr(page, "pdf_xobject", None) or []:
        limits = _bounds(xobject)
        if limits is not None:
            bounds[xobject.xobj_id] = limits
    if not bounds:
        return 0, 0

    def keep(element) -> bool:
        """中文：该元素是否应保留。 / Whether to keep this element."""
        limits = bounds.get(getattr(element, "xobj_id", 0) or 0)
        if limits is None:
            return True
        box = getattr(element, "box", None)
        visual = getattr(element, "visual_bbox", None)
        if visual is not None and getattr(visual, "box", None) is not None:
            box = visual.box
        if box is None:
            return True
        if not _outside(box, limits):
            return True
        if record is not None:
            record.append({
                "page": getattr(page, "page_number", None),
                "kind": type(element).__name__,
                "char": getattr(element, "char_unicode", None),
                "xobj": getattr(element, "xobj_id", None),
                "box": [box.x, box.y, box.x2, box.y2],
                "bbox": list(limits),
            })
        return False

    characters = getattr(page, "pdf_character", None)
    curves = getattr(page, "pdf_curve", None)
    dropped_chars = dropped_curves = 0

    if characters is not None:
        kept = [c for c in characters if keep(c)]
        dropped_chars = len(characters) - len(kept)
        if dropped_chars:
            page.pdf_character = kept
    if curves is not None:
        kept = [c for c in curves if keep(c)]
        dropped_curves = len(curves) - len(kept)
        if dropped_curves:
            page.pdf_curve = kept
    if record:
        with open(_PROBE, "a", encoding="utf-8") as handle:
            for entry in record:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return dropped_chars, dropped_curves


def install() -> None:
    """
    中文：在一页的中间表示建好之后、切分段落之前执行裁剪。

    Clip once a page's IL is complete and before paragraphs are formed.

    Wraps ``ActiveILCreater.on_page_end``: at that point every element the page
    holds has been projected and nothing has yet read them. Idempotent, so a
    second call is harmless.

    Raises:
        AttributeError: If BabelDOC no longer exposes the method, so an upgrade
            that moves it fails loudly rather than silently doing nothing.
    """
    from babeldoc.format.pdf.document_il.frontend.il_creater_active import (
        ActiveILCreater,
    )

    if getattr(ActiveILCreater.on_page_end, "_xobjclip_patched", False):
        return

    original = ActiveILCreater.on_page_end

    def on_page_end(self, *args, **kwargs):
        """中文：先裁剪，再交给原逻辑收尾。 / Clip, then let the original finish."""
        try:
            page = getattr(self, "current_page", None)
            if page is not None:
                chars, curves = _prune(page)
                if chars or curves:
                    logger.debug(
                        "Clipped %d character(s) and %d curve(s) drawn outside "
                        "their XObject's BBox on page %s",
                        chars, curves, getattr(page, "page_number", "?"))
        except Exception:  # noqa: BLE001
            # A page that cannot be clipped keeps everything, which is how every
            # page behaved before this existed.
            # 裁不了的页面保留全部内容，也就是本模块存在之前所有页面的行为。
            logger.warning("Could not apply XObject clipping on a page", exc_info=True)
        return original(self, *args, **kwargs)

    on_page_end._xobjclip_patched = True
    ActiveILCreater.on_page_end = on_page_end
