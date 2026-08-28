"""
中文：术语覆盖率评估——决定一篇文档是否还需要动用 API 提取术语。

Glossary coverage assessment.

This is the gate that makes the accumulation loop converge. Before translating,
the document's technical vocabulary is sampled and checked against the existing
glossary. When coverage is already high the API is not consulted at all, so as
the glossary grows the system asks the cloud less and less — which is the whole
point of accumulating terms rather than re-extracting them per document.

Candidate extraction is deliberately a heuristic rather than a model call: if
deciding whether to call the API itself required calling the API, the gate would
save nothing.
"""

from __future__ import annotations

import logging
import re
from collections import Counter

logger = logging.getLogger(__name__)

# Acronyms: 2–6 capitals, optionally pluralised (LLM, LLMs, DFT, XRD).
# 缩写：2–6 个大写字母，可带复数。
_ACRONYM = re.compile(r"\b([A-Z]{2,6})s?\b")

# Title-case compounds: "Batch Normalization", "Density Functional Theory".
# 首字母大写的复合术语。
_TITLE_PHRASE = re.compile(r"\b([A-Z][a-z]{2,}(?:[ -][A-Z][a-z]{2,}){1,3})\b")

# Lower-case technical compounds that follow an article, which in scientific
# prose reliably introduces a term ("the attention mechanism", "a residual
# connection"). Matching bare lower-case bigrams everywhere would swamp the
# sample with ordinary English.
# 冠词后的小写复合词在科技文献中通常引出术语；不加限制会被普通英语淹没。
_ARTICLE_PHRASE = re.compile(
    r"\b(?:the|a|an)\s+([a-z]{3,}(?:[ -][a-z]{3,}){1,2})\b", re.IGNORECASE
)

# Capitalised words that are ordinary English rather than terminology. Without
# this, sentence-initial words dominate the candidate set and depress coverage
# for reasons that have nothing to do with the glossary.
# 句首的普通英文词会污染候选集，压低覆盖率且与术语库无关。
_STOPWORDS: frozenset[str] = frozenset(
    """
    the this that these those there here where when which while what who whom
    and but for nor yet with without within from into onto upon about above
    below under over between among during before after since until
    however therefore moreover furthermore nevertheless nonetheless thus hence
    figure table section appendix equation chapter part page note
    we our they their its his her it is are was were be been being
    can could may might must shall should will would do does did
    in on at by to of as if or an a no not all any both each few more most
    other some such only own same so than too very just also then once
    first second third next last previous following given
    """.split()
)

# Acronyms that are units or common abbreviations rather than terminology.
# 单位与常见缩写，不是术语。
_ACRONYM_STOPWORDS: frozenset[str] = frozenset(
    {"AND", "THE", "FOR", "NOT", "ALL", "ARE", "WAS", "CAN", "MAY", "USA", "USD", "PDF", "URL"}
)


def extract_candidates(text: str, max_candidates: int = 400) -> list[str]:
    """
    中文：从文档文本中启发式抽取候选技术术语。

    Extract candidate technical terms from document text.

    Three patterns cover most of how terminology appears in scientific writing:
    acronyms, title-cased compounds, and lower-case compounds introduced by an
    article. Candidates are ranked by frequency, because a term the paper uses
    repeatedly matters far more to translation consistency than one mentioned
    once in passing.

    Args:
        text (str): Document text, typically the first pages plus the abstract.
        max_candidates (int): Cap on returned candidates, most frequent first.

    Returns:
        list[str]: Candidate terms, deduplicated case-insensitively and ordered
            by descending frequency.
    """
    counts: Counter[str] = Counter()

    for match in _ACRONYM.finditer(text):
        term = match.group(1)
        if term not in _ACRONYM_STOPWORDS:
            counts[term] += 1

    for match in _TITLE_PHRASE.finditer(text):
        phrase = match.group(1)
        words = phrase.split()
        # Drop phrases that are mostly ordinary words capitalised by position.
        # 丢弃大部分是普通词、只因位置而大写的短语。
        if sum(1 for w in words if w.lower() in _STOPWORDS) > len(words) // 2:
            continue
        counts[phrase] += 1

    for match in _ARTICLE_PHRASE.finditer(text):
        phrase = match.group(1).lower()
        words = phrase.split()
        if any(w in _STOPWORDS for w in words):
            continue
        counts[phrase] += 1

    seen: set[str] = set()
    ordered: list[str] = []
    for term, _ in counts.most_common():
        key = term.lower()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(term)
        if len(ordered) >= max_candidates:
            break

    return ordered


def read_document_text(pdf_path, max_pages: int = 6) -> str:
    """
    中文：读取 PDF 前若干页的文本，用于评估术语覆盖率。

    Read text from the first pages of a PDF.

    Only the opening pages are read: the abstract and introduction introduce
    essentially all of a paper's terminology, and parsing fifty pages to decide
    whether to make one API call would cost more than the call.

    Args:
        pdf_path (Path | str): Path to the PDF.
        max_pages (int): Number of leading pages to read.

    Returns:
        str: Concatenated page text, or an empty string if the PDF cannot be read.
    """
    try:
        import pymupdf

        with pymupdf.open(pdf_path) as doc:
            pages = min(max_pages, doc.page_count)
            return "\n".join(doc[i].get_text() for i in range(pages))
    except Exception:
        logger.warning("Could not read text from %s for coverage check", pdf_path, exc_info=True)
        return ""


def assess(pdf_path, matcher) -> dict[str, object]:
    """
    中文：评估一篇文档的术语覆盖率。

    Measure how much of a document's terminology the glossary already covers.

    Args:
        pdf_path (Path | str): Path to the source PDF.
        matcher (GlossaryMatcher | None): Compiled glossary matcher. ``None`` or
            an empty matcher yields zero coverage, which correctly routes the
            first documents through API extraction.

    Returns:
        dict[str, object]: Keys ``coverage`` (float 0–1), ``total`` (candidate
            count), ``covered`` (how many matched) and ``missing`` (up to 40
            uncovered candidates, most frequent first).
    """
    text = read_document_text(pdf_path)
    candidates = extract_candidates(text)

    if not candidates:
        # No candidates means the heuristic found nothing to judge by; treating
        # that as full coverage avoids an API call that has no terms to extract.
        # 抽不出候选说明无从判断，视为已覆盖，避免一次无的放矢的 API 调用。
        return {"coverage": 1.0, "total": 0, "covered": 0, "missing": []}

    if matcher is None or len(matcher) == 0:
        return {
            "coverage": 0.0,
            "total": len(candidates),
            "covered": 0,
            "missing": candidates[:40],
        }

    missing: list[str] = []
    covered = 0
    for term in candidates:
        pairs, keeps = matcher(term)
        if pairs or keeps:
            covered += 1
        else:
            missing.append(term)

    coverage = covered / len(candidates)
    logger.info(
        "Glossary coverage %.0f%% (%d/%d candidate terms)",
        coverage * 100,
        covered,
        len(candidates),
    )
    return {
        "coverage": coverage,
        "total": len(candidates),
        "covered": covered,
        "missing": missing[:40],
    }
