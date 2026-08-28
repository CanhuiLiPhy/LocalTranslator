"""
中文：识别版面模型没有类别的几种文字——文章标题、作者、单位、脚注，供选择性翻译使用。

Recognise the text kinds the layout model has no class for.

DocLayout-YOLO's docstructbench head emits ten classes, and four kinds of text a
reader has an opinion about are not among them:

* the **document title** — the model calls it ``title``, the same class it gives
  every section heading, so the two cannot be switched apart;
* the **author list** and the **affiliations** under it — both come back as
  ``plain text``;
* **footnotes** — also ``plain text``, or ``abandon`` when they sit low enough
  to look like a running foot.

Author names and their institutions are the clearest case: translating them
produces 「邓阳江」 for a name the reader needs in order to search for the paper,
and turns "The Hong Kong University of Science and Technology" into a string
that matches nothing. They are left in the original by default. Footnotes are
the opposite — they are ordinary prose and are translated by default — but a
reader who wants them left alone can now say so.

**How the front matter is bounded.** Everything on the first page above the
abstract is front matter. The abstract is found by its own heading, and failing
that by the first numbered section heading; failing both, only the top of the
page is considered, which is deliberately timid because a false positive here
means a paragraph of the paper silently staying in English.

Within the front matter the title is the largest type on the page — that is what
makes it the title — and what follows it is authors and affiliations, told apart
by the institution words and reference marks that affiliations carry.

**Why a side table rather than a relabel.** ``paragraph.layout_label`` is
BabelDOC's own field and it reads ``title`` to pick the heading it feeds the
translator as context. Overwriting it to say ``document_title`` would take that
context away, so the classification is kept beside the paragraph, keyed by the
id BabelDOC already assigns it.
"""

from __future__ import annotations

import logging
import os
import re

logger = logging.getLogger(__name__)

# The kinds this module recognises, in the order a reader meets them.
# 本模块能识别的类型，按读者遇到的顺序排列。
KINDS: tuple[str, ...] = ("document_title", "author", "affiliation", "footnote")

# Where the abstract begins, in the languages the front matter is written in.
# 摘要的起始处，按前置部分可能使用的语言列出。
_ABSTRACT = re.compile(r"^\s*(abstract|a\s*b\s*s\s*t\s*r\s*a\s*c\s*t|摘\s*要)\b", re.I)

# A numbered section heading, used as the fallback boundary when a paper has no
# abstract heading of its own.
# 带编号的章节标题；论文没有独立的摘要标题时用它作为边界。
_SECTION = re.compile(r"^\s*\d+(\.\d+)*\.?\s+\S")

# Words that mark an affiliation rather than a person.
# 标识单位而非人名的词。
_INSTITUTION = re.compile(
    r"\b(univ|universit|institut|laborator|\blabs?\b|college|school|academy|"
    r"academia|department|dept\b|faculty|hospital|centre|center|research|"
    r"corporation|\bcorp\b|\binc\b|\bltd\b|gmbh|technolog|polytechnic|"
    r"foundation|\bgroup\b|\bteam\b|\bai\b)",
    re.I,
)

# An affiliation line often opens with the reference mark that ties it to a
# name, and often carries a contact address.
# 单位行常以与人名对应的标记开头，也常带联系邮箱。
_REFERENCE_MARK = re.compile(r"^\s*[\d*†‡§¶∗⋆]")
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")

# 中文单位名。
# Institution words in Chinese.
_INSTITUTION_CJK = re.compile(r"(大学|学院|研究院|研究所|实验室|学校|医院|公司|集团)")

# A title has to be at least this much larger than the body to be the title.
# Below it the front matter is set in one size and there is nothing to single
# out, so nothing is claimed.
# 标题字号至少要比正文大这一比例才算标题。低于此值说明前置部分只有一种字号，
# 无从分辨，此时不做任何判定。
_TITLE_RATIO = 1.15

# Paragraphs within this many points of the largest size belong to the title
# too, which is what makes a two-line title one thing.
# 与最大字号相差在该点数内的段落同属标题，两行标题因此仍是一个整体。
_TITLE_SLACK = 0.6

# Share of the page height, measured from the foot, inside which a small
# paragraph reads as a footnote.
# 从页脚往上的高度比例，小字号段落落在其中即视为脚注。
_FOOTNOTE_BAND = 0.22

# A footnote is set smaller than the body. Papers use 8pt against 10pt; the
# threshold sits above that ratio so a body paragraph that merely shrank during
# typesetting is not swept in.
# 脚注字号小于正文。论文常见 8pt 对 10pt；阈值取在该比例之上，以免把排版时缩小
# 的正文段落也算进去。
_FOOTNOTE_RATIO = 0.92

# Front matter is looked for only this far down the first page when the abstract
# cannot be found at all.
# 完全找不到摘要时，前置部分只在首页顶部这一比例内寻找。
_FRONT_BAND = 0.30

# The longest a paragraph may be and still be called an author list or an
# affiliation. Both are lists; an abstract is prose, and the two do not overlap
# in length. Measured across the bench: the longest author list is the Muon
# paper's twenty-five names at about 50 words, while the shortest abstract runs
# past 100.
#
# This exists because the boundary above is only as good as the abstract
# heading it looks for. A paper that opens straight into its abstract with no
# heading — arXiv:2312.11805 does — falls back to the top-30% band, which
# swallows the abstract; it was then classified as an affiliation and left in
# English while every other paragraph was translated.
# 一个段落最多多长还能被称作作者列表或单位。两者都是「列表」，而摘要是散文，二者的长度
# 区间并不重叠。基准集里实测：最长的作者列表是 Muon 那篇的二十五位作者，约 50 词；而最短
# 的摘要超过 100 词。
#
# 之所以需要这条，是因为上面那个边界的可靠性取决于它要找的摘要标题。一篇不写标题、直接
# 开始摘要的论文——arXiv:2312.11805 就是——会回退到顶部 30% 的带宽，把摘要一并吞进去；
# 它随后被判成单位而保留英文，而同页其余段落都译了。
_MAX_FRONT_WORDS = 100

# Layout classes that are never front matter, however high on the page they
# sit. A figure and its caption can appear above the abstract — the second test
# paper opens with a full-width plate — and neither is an author.
# 无论位置多靠上都不属于前置部分的版面类别。插图及其图注可以出现在摘要之上——第二篇
# 测试论文开篇就是一整幅通栏图——它们都不是作者。
# A labelled metadata field: a capitalised label, optionally a word or two more,
# then a colon. Journals print "Received: 16 August 2023" and "Accepted: 25
# October 2023" beside the author list, and both are dates, not names. Without
# this they fall to the "anything else here is an author" default and are left
# in English — and since only the one that reaches into the front-matter band is
# caught, a page ends up with "Received: 16 August 2023" above
# "接受日期：2023年10月25日". A person's name never carries a colon.
# 带标签的元数据字段：一个首字母大写的标签词，后面可再跟一两个词，然后是冒号。期刊会把
# "Received: 16 August 2023"、"Accepted: 25 October 2023" 排在作者名单旁边，它们是日期
# 而不是名字。没有这一条，它们会落进「前置部分里其余的都算作者」的兜底分支而保留英文；
# 又因为只有伸进前置区间的那一行会被捕获，页面上就会出现 "Received: 16 August 2023"
# 压着「接受日期：2023年10月25日」。人名里从不带冒号。
_METADATA_FIELD = re.compile(r"^[A-Z][A-Za-z]*(?:\s+[a-z]+){0,2}:\s")

_NOT_FRONT_MATTER = frozenset({
    "figure", "table", "abandon",
    "figure_caption", "table_caption", "table_footnote", "formula_caption",
    "isolate_formula",
})

# Classification by paragraph id, filled as pages are parsed.
# 按段落 id 记录的分类结果，随页面解析逐步填入。
_CLASSIFIED: dict[str, str] = {}

# Set LOCALTRANS_FRONTMATTER_PROBE=1 to log what each front-matter paragraph was
# classified as. The classification is invisible in the output — a paragraph
# that should have been left alone simply comes back translated — so there is
# no way to tell a wrong answer from a missing one without this.
# 设 LOCALTRANS_FRONTMATTER_PROBE=1 可记录每个前置段落被判成了什么。分类结果在产物里
# 不可见——本该保留的段落只是被译掉了——没有这个探针就无法区分「判错了」和「没判」。
_PROBE = bool(os.getenv("LOCALTRANS_FRONTMATTER_PROBE"))


def kind_of(paragraph) -> str | None:
    """
    中文：查询某段落被识别成了哪一类。

    Report which kind a paragraph was recognised as.

    Args:
        paragraph (PdfParagraph): Paragraph to look up.

    Returns:
        str | None: One of :data:`KINDS`, or ``None`` when the paragraph is
            ordinary content.
    """
    return _CLASSIFIED.get(getattr(paragraph, "debug_id", None) or "")


def reset() -> None:
    """
    中文：清空分类结果，供新文档使用。

    Forget every classification, so a new document starts clean.
    """
    _CLASSIFIED.clear()


def _text(paragraph) -> str:
    """
    中文：取段落文字。

    Read a paragraph's text.

    Args:
        paragraph (PdfParagraph): Paragraph to read.

    Returns:
        str: Its Unicode text, empty when it has none.
    """
    return (paragraph.unicode or "").strip()


def _size(paragraph) -> float:
    """
    中文：取段落基准字号。

    Read a paragraph's base font size.

    Args:
        paragraph (PdfParagraph): Paragraph to measure.

    Returns:
        float: Font size in points, ``0.0`` when unknown.
    """
    style = getattr(paragraph, "pdf_style", None)
    return float(getattr(style, "font_size", 0.0) or 0.0)


def _body_size(paragraphs) -> float:
    """
    中文：估算该页正文字号——按字数加权取最常见的字号。

    Estimate a page's body font size.

    The most common size weighted by how much text is set in it, which is the
    body by definition; headings and footnotes are too short to win.

    Args:
        paragraphs (list): Paragraphs of one page.

    Returns:
        float: The dominant font size, ``0.0`` when the page holds no text.
    """
    weights: dict[float, int] = {}
    for paragraph in paragraphs:
        size = round(_size(paragraph), 1)
        if size <= 0:
            continue
        weights[size] = weights.get(size, 0) + len(_text(paragraph))
    if not weights:
        return 0.0
    return max(weights.items(), key=lambda item: item[1])[0]


def _looks_like_affiliation(text: str) -> bool:
    """
    中文：判断一行前置文字是单位而不是人名。

    Decide whether a line of front matter names an institution.

    Args:
        text (str): The paragraph's text.

    Returns:
        bool: ``True`` when it reads as an affiliation.
    """
    return bool(
        _INSTITUTION.search(text)
        or _INSTITUTION_CJK.search(text)
        or _EMAIL.search(text)
        or _REFERENCE_MARK.match(text)
    )


def _front_boundary(page) -> float | None:
    """
    中文：找出前置部分的下边界——摘要（或首个编号章节）的顶部。

    Locate where the front matter ends.

    Args:
        page (Page): The first page of the document.

    Returns:
        float | None: The y of the boundary in the layout's own coordinates,
            where larger is higher up the page. ``None`` when neither an
            abstract nor a numbered heading is present.
    """
    best = None
    for paragraph in page.pdf_paragraph:
        box = getattr(paragraph, "box", None)
        if box is None:
            continue
        text = _text(paragraph)
        if not text:
            continue
        if _ABSTRACT.match(text) or (
            paragraph.layout_label == "title" and _SECTION.match(text)
        ):
            best = box.y2 if best is None else max(best, box.y2)
    return best


def _classify_front_matter(page, boundary: float) -> None:
    """
    中文：把前置部分的段落分成标题、作者与单位。

    Split the front matter into title, authors and affiliations.

    Args:
        page (Page): The first page.
        boundary (float): Everything above this y is front matter.
    """
    front = []
    for paragraph in page.pdf_paragraph:
        box = getattr(paragraph, "box", None)
        if box is None or not _text(paragraph):
            continue
        # Where a paragraph *begins*, not where it ends. Nature Communications
        # prints no abstract heading, so the boundary falls back to the top
        # 30% of the page — 553.6 on an 791-point page — and the author list
        # runs from 562.5 down to 513.4, starting above the line and finishing
        # below it. Requiring the whole box to clear the boundary dropped it,
        # and every name was translated.
        # 看段落从哪里开始，而不是到哪里结束。《自然·通讯》不写摘要标题，边界因此退回
        # 页面顶部 30%——791 点的页面上是 553.6——而作者名单从 562.5 延伸到 513.4，
        # 起点在线上、终点在线下。要求整个框都越过边界就会漏掉它，于是每个名字都被翻译了。
        if box.y2 <= boundary:
            continue
        if paragraph.layout_label in _NOT_FRONT_MATTER:
            continue
        front.append(paragraph)
    if not front:
        return

    largest = max(_size(paragraph) for paragraph in front)
    body = _body_size(page.pdf_paragraph)
    if _PROBE:
        logger.info("[frontmatter] 边界=%.1f 前置段落=%d 最大字号=%.1f 正文字号=%.1f",
                    boundary, len(front), largest, body)
    if largest <= 0 or body <= 0:
        # Without type sizes there is no evidence to divide the front matter on,
        # and calling all of it the title would leave the authors' own names
        # riding on a guess.
        # 没有字号就没有划分前置部分的依据，把它整个当成标题会让作者姓名的处置
        # 建立在猜测之上。
        return
    titled = largest >= body * _TITLE_RATIO

    for paragraph in front:
        key = getattr(paragraph, "debug_id", None)
        if not key:
            if _PROBE:
                logger.info("[frontmatter] 无 debug_id，跳过: %r", _text(paragraph)[:48])
            continue
        if titled and _size(paragraph) >= largest - _TITLE_SLACK:
            _CLASSIFIED[key] = "document_title"
        elif len(_text(paragraph).split()) > _MAX_FRONT_WORDS:
            # Prose this long is the abstract or the opening of the body, not a
            # list of names. Leaving it unclassified is what lets it be
            # translated like the rest of the page.
            # 这么长的散文是摘要或正文开头，不是名单。不给它分类，它才会像同页其余段落
            # 一样被翻译。
            if _PROBE:
                logger.info("[frontmatter] %-15s 字号%5.1f  %r", "过长→不算前置",
                            _size(paragraph), _text(paragraph)[:48])
            continue
        elif _METADATA_FIELD.match(_text(paragraph)):
            # A labelled field, not a name. Leaving it unclassified is what lets
            # it be translated like the rest of the page.
            # 带标签的字段，不是名字。不给它分类，它才会像同页其余部分一样被翻译。
            if _PROBE:
                logger.info("[frontmatter] %-15s 字号%5.1f  %r", "元数据字段→翻译",
                            _size(paragraph), _text(paragraph)[:48])
            continue
        elif _looks_like_affiliation(_text(paragraph)):
            _CLASSIFIED[key] = "affiliation"
        else:
            _CLASSIFIED[key] = "author"
        if _PROBE:
            logger.info("[frontmatter] %-15s key=%r 字号%5.1f  %r",
                        _CLASSIFIED[key], key, _size(paragraph), _text(paragraph)[:48])


def _classify_footnotes(page) -> None:
    """
    中文：把页脚附近的小字号段落标为脚注。

    Mark the small paragraphs at the foot of a page as footnotes.

    Args:
        page (Page): Any page of the document.
    """
    box = page.cropbox.box if page.cropbox and page.cropbox.box else None
    if box is None:
        return
    height = box.y2 - box.y
    if height <= 0:
        return
    band = box.y + height * _FOOTNOTE_BAND
    body = _body_size(page.pdf_paragraph)
    if body <= 0:
        return

    for paragraph in page.pdf_paragraph:
        key = getattr(paragraph, "debug_id", None)
        area = getattr(paragraph, "box", None)
        if not key or area is None or key in _CLASSIFIED:
            continue
        if not _text(paragraph) or paragraph.layout_label in ("figure", "table"):
            continue
        if area.y2 > band:
            continue
        size = _size(paragraph)
        if size <= 0 or size > body * _FOOTNOTE_RATIO:
            continue
        _CLASSIFIED[key] = "footnote"


def classify_page(page) -> None:
    """
    中文：对一页做分类，结果存入模块内的表中。

    Classify one page, recording the result for :func:`kind_of`.

    Args:
        page (Page): A page of BabelDOC's intermediate layout, after paragraphs
            have been formed.
    """
    if not getattr(page, "pdf_paragraph", None):
        return

    if page.page_number == 0:
        boundary = _front_boundary(page)
        if boundary is None:
            box = page.cropbox.box if page.cropbox and page.cropbox.box else None
            if box is not None:
                boundary = box.y2 - (box.y2 - box.y) * _FRONT_BAND
        if boundary is not None:
            _classify_front_matter(page, boundary)

    _classify_footnotes(page)


def install() -> None:
    """
    中文：在样式计算之后接上分类步骤——那时段落才有字号可读。

    Classify each page once its paragraphs have styles.

    Wraps ``StylesAndFormulas.process_page`` rather than the paragraph finder:
    ``paragraph.pdf_style`` is filled in by ``process_page_styles``, at the end
    of that stage, and until then every paragraph reports a size of zero. An
    earlier version hooked the paragraph finder and, with every size equal,
    called the whole of the front matter the title.

    Idempotent, so a second call is harmless.

    Raises:
        AttributeError: If BabelDOC no longer exposes the method, so an upgrade
            that moves it fails loudly at startup instead of silently doing
            nothing.
    """
    from babeldoc.format.pdf.document_il.midend.styles_and_formulas import (
        StylesAndFormulas,
    )

    if getattr(StylesAndFormulas.process_page, "_frontmatter_patched", False):
        return

    original = StylesAndFormulas.process_page

    def process_page(self, page):
        """中文：先跑原逻辑，再分类。 / Run the original pass, then classify."""
        result = original(self, page)
        try:
            classify_page(page)
        except Exception:  # noqa: BLE001
            # Failing here must not cost the page: without a classification
            # every paragraph is simply treated as ordinary content.
            # 这里失败不应赔上整页：没有分类结果，所有段落按普通正文处理即可。
            logger.warning("Front-matter classification failed on a page", exc_info=True)
        return result

    process_page._frontmatter_patched = True
    StylesAndFormulas.process_page = process_page
