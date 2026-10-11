"""
中文：从 PDF 自带的命名锚点提取结构信息——文献条目位置、列表项边界等。

Structural anchors recovered from the PDF's own named destinations.

LaTeX's ``hyperref`` writes a named destination for everything it cross-
references: ``cite.Foo2024`` at each bibliography entry, ``Item.3`` at each
enumerated list item, ``figure.2``, ``section.4`` and so on. These are the
document's own statement about its structure, placed by the tool that typeset
it, and they are exact — a coordinate, not an estimate.

That makes them a better source than the heuristics they replace. Deciding
whether a paragraph is a bibliography entry by counting citation-like patterns
in its text works until an entry happens to carry only one such pattern, at
which point a reference is quietly translated and the author names are mangled.
An anchor is not a guess about the paragraph; it is the typesetter saying "a
reference begins here".

Not every PDF has them — a scanned document has none, and some producers omit
them — so anchors are used to *add* certainty where they exist rather than to
replace the text heuristics entirely.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# Vertical slack in points when matching an anchor against a paragraph box.
# The anchor sits on the first line's baseline, which can fall a little outside
# the box the layout model drew around the paragraph.
# 把锚点与段落框匹配时的垂直容差（点）。锚点位于首行基线上，可能略微落在版面
# 模型给出的段落框之外。
_SLACK = 6.0


@dataclass
class DocumentAnchors:
    """
    中文：一篇文档的命名锚点集合，按页组织。

    Named destinations for one document, grouped by page.

    Attributes:
        citations (dict[int, list[tuple[float, float]]]): Page index to the
            ``(x, y)`` coordinates of bibliography entry anchors, sorted by
            descending y as they appear down the page. The x is kept because a
            two-column page needs it — see :meth:`is_citation`.
        list_items (dict[int, list[float]]): Page index to the y coordinates of
            enumerated list item anchors.
        available (bool): Whether the document carried any named destinations.
    """

    citations: dict[int, list[tuple[float, float]]] = field(default_factory=dict)
    list_items: dict[int, list[float]] = field(default_factory=dict)
    available: bool = False

    @classmethod
    def load(cls, path: Path | str) -> DocumentAnchors:
        """
        中文：从 PDF 读取命名锚点。

        Read named destinations from a PDF.

        Failure is not an error: a document without anchors simply yields an
        empty set, and every caller treats that as "no extra information".

        Args:
            path (Path | str): Source PDF.

        Returns:
            DocumentAnchors: Anchors found, empty when the file has none or
                could not be read.
        """
        try:
            import pymupdf

            doc = pymupdf.open(str(path))
        except Exception:  # noqa: BLE001
            logger.debug("Could not open %s for anchors", path, exc_info=True)
            return cls()

        citations: dict[int, list[float]] = defaultdict(list)
        items: dict[int, list[float]] = defaultdict(list)
        try:
            names = doc.resolve_names()
        except Exception:  # noqa: BLE001
            logger.debug("Document exposes no resolvable names", exc_info=True)
            doc.close()
            return cls()
        finally:
            pass

        for name, target in names.items():
            page = target.get("page")
            where = target.get("to")
            if page is None or not where:
                continue
            x, y = float(where[0]), float(where[1])
            head = name.split(".", 1)[0].lower()
            if head == "cite":
                citations[page].append((x, y))
            elif head == "item":
                items[page].append(y)

        doc.close()
        found = bool(citations or items)
        if found:
            logger.info("Structural anchors: %d citation(s), %d list item(s)",
                        sum(len(v) for v in citations.values()),
                        sum(len(v) for v in items.values()))
        return cls(
            citations={k: sorted(v, key=lambda p: -p[1]) for k, v in citations.items()},
            list_items={k: sorted(v, reverse=True) for k, v in items.items()},
            available=found,
        )

    def is_citation(self, page: int, y0: float, y1: float,
                    x0: float | None = None, x1: float | None = None) -> bool:
        """
        中文：判断某页上的一个区域是否位于文献区内；双栏时只与同栏的条目比较。

        Report whether a paragraph's box lies within the bibliography.

        Containing an anchor is sufficient but not necessary. An anchor marks
        where an entry *begins*, and a long entry is often split into several
        paragraphs, only the first of which holds it — measured on a real paper,
        four continuation lines were translated for exactly this reason while
        their first lines were correctly preserved.

        The bibliography is contiguous *within a column*: once it starts, it
        runs to the foot of that column. So anything below the topmost anchor
        **of the same column** is part of it, while text above it — a
        conclusion section sharing the page — is not, and keeps being
        translated.

        Comparing against the whole page instead is what broke that promise on
        a two-column paper. On page 11 of arXiv:2608.09438 the bibliography
        fills the right column from the very top, putting its highest anchor at
        y 739.5, while the conclusion sits in the *left* column reaching only
        y 718. Measured against the page the conclusion looked like it was
        below the first entry, so the whole section — and the acknowledgements
        under it — was preserved as English while the layout model had
        correctly called both plain text.

        Args:
            page (int): Zero-based page index.
            y0 (float): Lower edge of the paragraph box, in PDF coordinates.
            y1 (float): Upper edge of the paragraph box.
            x0 (float | None): Left edge of the box. Given together with
                ``x1``, only anchors whose x falls inside that span are
                considered, which is what confines the test to one column.
                Omitting both restores the whole-page comparison.
            x1 (float | None): Right edge of the box.

        Returns:
            bool: ``True`` when the box lies in the bibliography.
        """
        anchors = self.citations.get(page)
        if not anchors:
            return False
        if x0 is not None and x1 is not None:
            left, right = min(x0, x1) - _SLACK, max(x0, x1) + _SLACK
            anchors = [a for a in anchors if left <= a[0] <= right]
            if not anchors:
                return False
        # PDF coordinates grow upward, so "below the first entry" means a
        # smaller y than the highest anchor in this column.
        # PDF 坐标向上增长，因此「在首条目之下」意味着 y 小于本栏最高的锚点。
        return max(y0, y1) <= anchors[0][1] + _SLACK

    def items_within(self, page: int, y0: float, y1: float) -> list[float]:
        """
        中文：返回落在某垂直区间内的列表项锚点。

        List the enumerated-item anchors inside a vertical span.

        A paragraph containing more than one is a run of list items that the
        layout model merged into a single block, and the anchors say exactly
        where the boundaries between them lie.

        Args:
            page (int): Zero-based page index.
            y0 (float): Lower edge of the paragraph box.
            y1 (float): Upper edge of the paragraph box.

        Returns:
            list[float]: Anchor y coordinates, top of the page first.
        """
        low, high = min(y0, y1) - _SLACK, max(y0, y1) + _SLACK
        return [y for y in self.list_items.get(page, ()) if low <= y <= high]

    def summary(self) -> dict[str, int]:
        """
        中文：返回锚点统计。

        Report how many anchors were found.

        Returns:
            dict[str, int]: Counts by kind.
        """
        return {
            "citation_anchors": sum(len(v) for v in self.citations.values()),
            "list_item_anchors": sum(len(v) for v in self.list_items.values()),
            "citation_pages": len(self.citations),
        }
