"""
中文：arXiv 论文获取——解析链接/编号、取元数据、下载 PDF。

arXiv paper acquisition.

Accepts anything a user is likely to paste — a bare id, an ``abs`` page URL, a
``pdf`` URL, an old-style category id — resolves the paper's real title and
authors through the arXiv API, and downloads the PDF into the library under a
readable folder name.

Using the API rather than just downloading the PDF matters for the glossary and
the library UI: the embedded PDF title is frequently missing or wrong, whereas
the API gives the canonical title and author list.
"""

from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

ARXIV_API = "https://export.arxiv.org/api/query"

# Modern identifiers look like 2510.04704 or 2510.04704v4; historical ones like
# math.GT/0309136. Both forms may be wrapped in an abs/pdf URL.
# 新式编号形如 2510.04704(v4)，旧式形如 math.GT/0309136，两者都可能包在 URL 里。
_NEW_STYLE = re.compile(r"(?<!\d)(\d{4}\.\d{4,5})(v\d+)?(?!\d)")
_OLD_STYLE = re.compile(r"([a-z-]+(?:\.[A-Z]{2})?/\d{7})(v\d+)?")

_ATOM = "{http://www.w3.org/2005/Atom}"


@dataclass
class ArxivPaper:
    """
    中文：一篇已下载的 arXiv 论文。

    A downloaded arXiv paper.

    Attributes:
        arxiv_id (str): Identifier including version when known, e.g. ``2510.04704v4``.
        title (str): Canonical title from the arXiv API.
        authors (str): Comma-separated author names.
        abs_url (str): Canonical abstract page URL.
        pdf_path (Path): Location of the downloaded PDF on disk.
    """

    arxiv_id: str
    title: str
    authors: str
    abs_url: str
    pdf_path: Path


def parse_arxiv_id(reference: str) -> str:
    """
    中文：从链接或文本中提取 arXiv 编号。

    Extract an arXiv identifier from a user-supplied reference.

    Args:
        reference (str): A bare id, an ``arxiv.org/abs/…`` or ``arxiv.org/pdf/…``
            URL (with or without a trailing ``.pdf``), or text containing one.

    Returns:
        str: The identifier, preserving the version suffix when present.

    Raises:
        ValueError: If no arXiv identifier can be found in the input.
    """
    text = reference.strip()
    if text.lower().endswith(".pdf"):
        text = text[:-4]

    match = _NEW_STYLE.search(text)
    if match:
        return match.group(1) + (match.group(2) or "")

    match = _OLD_STYLE.search(text)
    if match:
        return match.group(1) + (match.group(2) or "")

    raise ValueError(
        f"Could not find an arXiv id in {reference!r}. "
        "Expected something like 2510.04704 or https://arxiv.org/abs/2510.04704"
    )


async def fetch_metadata(arxiv_id: str, timeout: float = 30.0) -> dict[str, str]:
    """
    中文：调用 arXiv API 获取论文标题与作者。

    Retrieve a paper's title and authors from the arXiv API.

    Metadata failures are non-fatal to the caller: the returned dict falls back
    to the identifier as the title, so a temporarily unreachable API only costs
    a nicer folder name, not the download itself.

    Args:
        arxiv_id (str): Identifier, with or without version suffix.
        timeout (float): HTTP timeout in seconds.

    Returns:
        dict[str, str]: Keys ``title``, ``authors`` and ``abs_url``.
    """
    fallback = {
        "title": arxiv_id,
        "authors": "",
        "abs_url": f"https://arxiv.org/abs/{arxiv_id}",
    }

    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=timeout) as client:
            resp = await client.get(ARXIV_API, params={"id_list": arxiv_id, "max_results": 1})
            resp.raise_for_status()
            root = ET.fromstring(resp.text)
    except Exception:
        logger.warning("arXiv metadata lookup failed for %s", arxiv_id, exc_info=True)
        return fallback

    entry = root.find(f"{_ATOM}entry")
    if entry is None:
        return fallback

    title_el = entry.find(f"{_ATOM}title")
    title = " ".join(title_el.text.split()) if title_el is not None and title_el.text else arxiv_id

    authors = [
        name.text.strip()
        for author in entry.findall(f"{_ATOM}author")
        if (name := author.find(f"{_ATOM}name")) is not None and name.text
    ]

    id_el = entry.find(f"{_ATOM}id")
    abs_url = id_el.text.strip() if id_el is not None and id_el.text else fallback["abs_url"]

    return {"title": title, "authors": ", ".join(authors), "abs_url": abs_url}


async def download_pdf(arxiv_id: str, dest: Path, timeout: float = 120.0) -> Path:
    """
    中文：下载 arXiv PDF 到指定路径。

    Download a paper's PDF.

    Args:
        arxiv_id (str): Identifier, with or without version suffix.
        dest (Path): Destination file path. Parent directories are created.
        timeout (float): HTTP timeout in seconds.

    Returns:
        Path: ``dest``, now containing the PDF.

    Raises:
        httpx.HTTPStatusError: If arXiv returns an error status.
        ValueError: If the response is not a PDF, which happens when arXiv
            serves its "paper is being generated" HTML placeholder.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://arxiv.org/pdf/{arxiv_id}"

    async with httpx.AsyncClient(follow_redirects=True, timeout=timeout) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        content = resp.content

    if not content.startswith(b"%PDF"):
        raise ValueError(
            f"arXiv did not return a PDF for {arxiv_id}; the paper may still be "
            "processing. Try again in a few minutes."
        )

    dest.write_bytes(content)
    logger.info("Downloaded arXiv:%s to %s (%.1f MB)", arxiv_id, dest, len(content) / 1024**2)
    return dest


async def fetch_arxiv(reference: str, library_dir: Path) -> ArxivPaper:
    """
    中文：一步完成——解析编号、取元数据、下载 PDF 到文库。

    Resolve a reference and download the paper into the library.

    The paper is placed in ``<library>/<arxiv_id>/``, so the source PDF and its
    translations end up side by side in a folder named after something stable
    and recognisable rather than a content hash.

    An already-downloaded PDF is reused rather than re-fetched, which makes
    re-submitting the same paper cheap and keeps arXiv from being hit repeatedly.

    Args:
        reference (str): arXiv id or URL.
        library_dir (Path): Root of the managed library.

    Returns:
        ArxivPaper: Metadata plus the on-disk PDF location.

    Raises:
        ValueError: If the reference cannot be parsed, or arXiv does not serve
            a PDF for it.
    """
    arxiv_id = parse_arxiv_id(reference)
    folder = library_dir / arxiv_id.replace("/", "_")
    pdf_path = folder / f"{arxiv_id.replace('/', '_')}.pdf"

    meta = await fetch_metadata(arxiv_id)

    if pdf_path.exists() and pdf_path.stat().st_size > 0:
        logger.info("Reusing already-downloaded arXiv:%s at %s", arxiv_id, pdf_path)
    else:
        await download_pdf(arxiv_id, pdf_path)

    return ArxivPaper(
        arxiv_id=arxiv_id,
        title=meta["title"],
        authors=meta["authors"],
        abs_url=meta["abs_url"],
        pdf_path=pdf_path,
    )
