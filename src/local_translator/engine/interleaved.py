"""
中文：一段正文被穿插其间的水印、行号切成两半时，把它接回去。

Rejoin a paragraph that was cut in two by something drawn in the middle of it.

**The symptom.** On page 1 of the paper this was reported on, a citation came
out as ``…目标知识 [。`` at the end of one line and ``10].`` at the start of the
next, with the full stop stranded inside the bracket. The line had ended more
than a hundred points short of the margin, so nothing about the width explained
the break.

**Why it happens.** ``ParagraphFinder`` walks ``page.pdf_character`` in **draw**
order and opens a new paragraph whenever a character's layout box changes::

    #9   layout=1   plain text  '…fail to acquire the target knowledge ['
    #10  layout=12  abandon     'arXiv:2608.14071v1  [cs.AI]  14 Aug 2026'
    #11  layout=1   plain text  '10]. Therefore, maintaining …'

The arXiv stamp down the left margin is a separate layout, and the producer
happened to draw it between the ``[`` and the ``10]`` of a citation. Draw order
is not reading order, so the body paragraph was cut at that point — mid-word, as
it happens, mid-citation. Both halves are then translated on their own and laid
out on their own, which is what puts a bracket at one line's end and its digits
at the next line's start.

BabelDOC knows about one instance of this: ``merge_alternating_line_number_para
graphs`` rejoins ``a l+ c`` when the interruption is line numbers. Its test for
the middle is that the text is ASCII digits and spaces, so a stamp, a margin
note or a rotated header does not qualify.

**The fix.** Restore the invariant the grouping is built on — one layout box
holds one paragraph at this stage — by merging back any two paragraphs that
share a layout box and are separated only by paragraphs from *other* boxes. The
interrupting paragraph keeps its own place in the list; only the split halves
are rejoined. A box that legitimately holds several paragraphs is unaffected,
because such a box was never split here in the first place: the later passes
(short lines, bullets, numbered lists) do that, and they run afterwards.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def _mergeable(first, second) -> bool:
    """
    中文：判断两段是否本属同一段——同一版面框、同一 XObject、同一书写方向。

    Decide whether two paragraphs are halves of one.

    Args:
        first (PdfParagraph): The earlier half.
        second (PdfParagraph): The later half.

    Returns:
        bool: ``True`` when both carry the same non-null ``layout_id`` and agree
            on ``xobj_id`` and ``vertical``. The layout id is what makes them
            the same block; the other two guard against joining text that is
            drawn in a different coordinate space or runs the other way, where
            draw order says nothing about reading order.
    """
    if first.layout_id is None or first.layout_id != second.layout_id:
        return False
    if first.xobj_id != second.xobj_id:
        return False
    return bool(first.vertical) == bool(second.vertical)


def rejoin(finder, paragraphs) -> int:
    """
    中文：把被其它版面框打断的同一段落接回去，返回合并次数。

    Merge paragraphs that were split by an interleaved layout.

    Walks the list once. For each paragraph, scans forward past paragraphs
    belonging to other layouts; the first one that belongs to the *same* layout
    is appended to it and removed from the list. Scanning continues from the
    same paragraph so a block interrupted several times is fully rejoined.

    Args:
        finder (ParagraphFinder): The stage, used for ``update_paragraph_data``
            so the merged paragraph's box, xobj and indent are recomputed the
            way BabelDOC computes them everywhere else.
        paragraphs (list[PdfParagraph]): Paragraphs for one page, in draw order.
            Modified in place.

    Returns:
        int: How many paragraphs were merged away.
    """
    merged = 0
    index = 0
    while index < len(paragraphs):
        first = paragraphs[index]
        if first.layout_id is None:
            index += 1
            continue

        ahead = index + 1
        while ahead < len(paragraphs):
            candidate = paragraphs[ahead]
            if candidate.layout_id == first.layout_id:
                if not _mergeable(first, candidate):
                    break
                first.pdf_paragraph_composition.extend(
                    candidate.pdf_paragraph_composition
                )
                finder.update_paragraph_data(first)
                del paragraphs[ahead]
                merged += 1
                # The same block may be interrupted more than once, so carry on
                # from where this half was removed rather than starting over.
                # 同一块可能被打断多次，从删除处继续往后找，不必重头再来。
                continue
            ahead += 1

        index += 1
    return merged


def install() -> None:
    """
    中文：在 BabelDOC 按版面框分完段之后，补一次接合。

    Rejoin split paragraphs right after BabelDOC groups characters into them.

    Wraps ``ParagraphFinder._group_characters_into_paragraphs``, which is the
    step that does the splitting and the last point at which the halves are
    still plain character compositions — before lines are formed, before
    formulas are parsed, and before every later pass starts making decisions
    from a paragraph's text.

    Idempotent, so a second call is harmless.

    Raises:
        AttributeError: If BabelDOC no longer exposes the method, so an upgrade
            that moves it fails at startup rather than silently doing nothing.
    """
    from babeldoc.format.pdf.document_il.midend.paragraph_finder import ParagraphFinder

    original = ParagraphFinder._group_characters_into_paragraphs
    if getattr(original, "_interleaved_patched", False):
        return

    def _group_characters_into_paragraphs(self, page, layout_index, layout_map):
        """中文：先按原逻辑分段，再把被打断的段接回去。 / Group as before, then rejoin what was interrupted."""
        paragraphs = original(self, page, layout_index, layout_map)
        try:
            merged = rejoin(self, paragraphs)
        except Exception:  # noqa: BLE001
            # A failure here must not cost the page: without the repair the
            # paragraphs are split, which is how they were before this existed.
            # 这里失败不应赔上整页：不修就是切开的状态，也就是本模块出现之前的样子。
            logger.warning("Rejoining interrupted paragraphs failed on a page",
                           exc_info=True)
            return paragraphs
        if merged:
            logger.debug("Rejoined %d interrupted paragraph(s) on page %s",
                         merged, page.page_number)
        return paragraphs

    _group_characters_into_paragraphs._interleaved_patched = True
    ParagraphFinder._group_characters_into_paragraphs = _group_characters_into_paragraphs
