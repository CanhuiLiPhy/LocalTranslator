"""
中文：把图内文字从页面段落里摘出去——图片原样搬回时，它们的文字不该再被当作正文处理。

Take a figure's own text out of the page's paragraphs.

**Why it must go.** When figures are preserved verbatim, the whole figure —
artwork and labels together — is stamped back from the source. Anything the
pipeline typesets in that area is covered by the stamp, so a paragraph made of
figure labels is not merely useless: it is translated at cost, it is laid out
where nothing will show it, and its box takes part in every layout decision made
about its neighbours.

**What that costs in practice.** On page 4 of arXiv:2608.17760 the labels of a
right-column figure — Target, Closest learnware, General solution — were
gathered into one paragraph whose bounding box ran from x=144 to x=354, across
the gutter and into the *left* column. It came to rest directly beneath a
left-column paragraph, was counted as something below it, and denied that
paragraph the room it needed. The paragraph was typeset at 0.65 against 9.96pt
everywhere else on the page.

**Why the test is on characters, not boxes.** That aggregate box overlaps the
figure's own layout box by two points out of two hundred, so a box-containment
test says it is not in the figure at all. Its characters tell the truth: every
one of them sits inside the figure. Individual labels that were *not* gathered
up — 「Target」 at x=383, 「learnwares」 at x=475 — are plainly inside by either
test. Reading the characters therefore catches both shapes of the problem.

**When figures are translated instead.** Then this does nothing, and the labels
travel the ordinary path: the figure is not stamped back, so what the pipeline
typesets is what the reader sees. Handling those labels well is a separate
matter from keeping them out of the way when nobody asked for them.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Layout classes whose interior text belongs to the drawing rather than to the
# page. Only figures for now: a table's cells are content a reader may want
# translated, and tables are handled by their own label.
# 其内部文字属于图形而非页面的版面类别。目前只有图片：表格的单元格是读者可能想读的内容，
# 而表格由它自己的标签处理。
_FIGURE_CLASSES = frozenset({"figure"})

# How much of a paragraph must sit inside a figure before the paragraph is
# treated as belonging to it. Not 1.0: a label that straddles the figure's
# border by a character or two is still the figure's.
# 一个段落有多大比例落在图内，才算属于这张图。不取 1.0：有个别字符压在图的边界上，
# 那也仍然是图的文字。
_INSIDE_SHARE = 0.8


def _characters(paragraph):
    """
    中文：遍历段落里的全部字符——此时它们在 composition 里，还没有分行。

    Yield every character in a paragraph.

    At the stage this runs, a paragraph holds its characters in
    ``pdf_paragraph_composition`` and has no ``pdf_line`` yet: lines are formed
    later, during typesetting. Reading only the lines therefore finds nothing,
    which is what an earlier version of this module did — it installed, ran on
    every page, and matched no paragraph anywhere.

    Args:
        paragraph (PdfParagraph): Paragraph to walk.

    Yields:
        The character objects, in reading order.
    """
    for composition in getattr(paragraph, "pdf_paragraph_composition", None) or []:
        for name in ("pdf_character", "pdf_line", "pdf_same_style_characters",
                     "pdf_same_style_unicode_characters", "pdf_formula"):
            part = getattr(composition, name, None)
            if part is None:
                continue
            if name == "pdf_character":
                yield part
                continue
            yield from getattr(part, "pdf_character", None) or []


def _centre(character):
    """
    中文：返回字符的中心点，取不到则返回 None。

    Return a character's centre, or ``None`` when it has no usable box.

    Args:
        character: A ``PdfCharacter``.

    Returns:
        tuple[float, float] | None: ``(x, y)`` at the character's centre.
    """
    visual = getattr(character, "visual_bbox", None)
    box = getattr(visual, "box", None) if visual is not None else None
    if box is None:
        return None
    return ((box.x + box.x2) / 2, (box.y + box.y2) / 2)


def _belongs_to_a_figure(paragraph, figures) -> bool:
    """
    中文：判断一个段落是否属于某张图——按字符位置，不按聚合框。

    Decide whether a paragraph is a figure's own text.

    Args:
        paragraph (PdfParagraph): Paragraph under consideration.
        figures (list): Boxes of the page's figure regions.

    Returns:
        bool: ``True`` when at least :data:`_INSIDE_SHARE` of its characters
            fall inside one of them. A paragraph with no characters is never
            claimed — there is nothing to judge it by.
    """
    total = inside = 0
    for character in _characters(paragraph):
        centre = _centre(character)
        if centre is None:
            continue
        total += 1
        x, y = centre
        if any(box.x <= x <= box.x2 and box.y <= y <= box.y2 for box in figures):
            inside += 1
    return total > 0 and inside / total >= _INSIDE_SHARE


def install() -> None:
    """
    中文：在段落切分之后、任何人读到段落之前，把图内文字摘掉。

    Drop figure text once paragraphs exist and before anything reads them.

    Hooks ``ParagraphFinder.process_page``, the stage that produces the
    paragraphs and the first at which both they and ``page.page_layout`` are in
    hand. Idempotent, so a second call is harmless.

    Raises:
        AttributeError: If BabelDOC no longer exposes the method, so an upgrade
            that moves it fails loudly rather than silently doing nothing.
    """
    from babeldoc.format.pdf.document_il.midend.paragraph_finder import ParagraphFinder

    if getattr(ParagraphFinder.process_page, "_figuretext_patched", False):
        return

    original = ParagraphFinder.process_page

    def process_page(self, page, *args, **kwargs):
        """中文：先跑原逻辑，再摘掉图内文字。 / Run the original pass, then prune."""
        result = original(self, page, *args, **kwargs)
        try:
            figures = [layout.box for layout in (page.page_layout or [])
                       if layout.box is not None
                       and layout.class_name in _FIGURE_CLASSES]
            if figures and page.pdf_paragraph:
                keep = [p for p in page.pdf_paragraph
                        if not _belongs_to_a_figure(p, figures)]
                dropped = len(page.pdf_paragraph) - len(keep)
                if dropped:
                    page.pdf_paragraph = keep
                    logger.debug("Dropped %d figure-text paragraph(s) on page %s",
                                 dropped, getattr(page, "page_number", "?"))
        except Exception:  # noqa: BLE001
            # A page whose layout cannot be read keeps all its paragraphs, which
            # is what happened before this existed.
            # 版面读不出来的页面保留全部段落，也就是本模块存在之前的行为。
            logger.warning("Could not prune figure text on a page", exc_info=True)
        return result

    process_page._figuretext_patched = True
    ParagraphFinder.process_page = process_page
