"""
中文：pdf.js 资源的下载与安装（本地自带，不走 CDN）。

Vendoring of the pdf.js viewer.

The reader renders PDFs with Mozilla's pdf.js rather than the browser's built-in
viewer, because the two-pane mode needs to *read and set* the current page and
zoom level in order to keep the panes in step — something a native viewer iframe
does not expose.

pdf.js is downloaded once into the data directory and served
from there. It is deliberately not bundled into the repository (it is several
megabytes of third-party build output) and never loaded from a CDN, so the
application keeps working with no network at all — which is the whole point of a
local translator.
"""

from __future__ import annotations

import io
import logging
import zipfile
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

PDFJS_REPO = "mozilla/pdf.js"
PDFJS_DIRNAME = "pdfjs"


def pdfjs_dir() -> Path:
    """
    中文：返回 pdf.js 的安装目录。

    Locate the directory pdf.js is installed into.

    Returns:
        Path: ``<data dir>/static/pdfjs``. May not exist yet.
    """
    # Under the data directory, never inside the package. A packaged application
    # lives in /Applications, which the reader may not be able to write to and
    # which a signed bundle must not have written to in any case — a modified
    # bundle fails its own signature check and macOS refuses to open it.
    # Installing here also means one download survives every future version.
    # 放在数据目录下，绝不放进包内。打包后的程序装在「应用程序」里，读者未必有写权限；
    # 而且签名过的应用包一旦被改写就通不过自身的签名校验，macOS 会拒绝打开。装在这里
    # 还意味着下载一次即可供以后所有版本使用。
    from local_translator.config import get_paths

    return get_paths().data_dir / "static" / PDFJS_DIRNAME


def is_installed() -> bool:
    """
    中文：判断 pdf.js 是否已安装。

    Check whether a usable pdf.js build is present.

    Returns:
        bool: ``True`` if the viewer entry point exists.
    """
    return (pdfjs_dir() / "web" / "viewer.html").is_file()


def resolve_asset(tag: str | None = None) -> tuple[str, str, int]:
    """
    中文：查询 pdf.js 发布包地址。

    Look up the pdf.js distribution archive on GitHub.

    The ``legacy`` build is preferred: it targets older JavaScript features and
    therefore works across a wider range of browser versions, at a negligible
    size difference for a locally served asset.

    Args:
        tag (str | None): Release tag such as ``"v6.2.108"``. Defaults to latest.

    Returns:
        tuple[str, str, int]: ``(release_tag, download_url, size_in_bytes)``.

    Raises:
        RuntimeError: If the release exposes no distribution archive.
    """
    api = (
        f"https://api.github.com/repos/{PDFJS_REPO}/releases/tags/{tag}"
        if tag
        else f"https://api.github.com/repos/{PDFJS_REPO}/releases/latest"
    )
    with httpx.Client(follow_redirects=True, timeout=30.0) as client:
        resp = client.get(api)
        resp.raise_for_status()
        data = resp.json()

    assets = data.get("assets", [])
    for preference in ("legacy-dist.zip", "dist.zip"):
        for asset in assets:
            if asset["name"].endswith(preference):
                return data["tag_name"], asset["browser_download_url"], asset.get("size", 0)

    raise RuntimeError(f"pdf.js release {data.get('tag_name')} has no dist archive")


def install(tag: str | None = None, force: bool = False) -> Path:
    """
    中文：下载并解压 pdf.js 到静态资源目录。

    Download and unpack the pdf.js viewer.

    Args:
        tag (str | None): Release tag to install. Defaults to latest.
        force (bool): Reinstall even when already present.

    Returns:
        Path: The installation directory.

    Raises:
        RuntimeError: If the archive does not contain ``web/viewer.html``.
    """
    target = pdfjs_dir()
    if is_installed() and not force:
        logger.debug("pdf.js already installed at %s", target)
        return target

    release_tag, url, size = resolve_asset(tag)
    logger.info("Downloading pdf.js %s (%.1f MB)", release_tag, size / 1e6)

    with httpx.Client(follow_redirects=True, timeout=120.0) as client:
        resp = client.get(url)
        resp.raise_for_status()
        payload = resp.content

    if target.exists():
        import shutil

        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        archive.extractall(target)

    if not is_installed():
        raise RuntimeError(f"pdf.js archive from {url} did not contain web/viewer.html")

    logger.info("Installed pdf.js %s to %s", release_tag, target)
    return target


def viewer_url(pdf_url: str, page: int = 1, sidebar: bool = False) -> str:
    """
    中文：拼装 pdf.js viewer 的访问地址。

    Build a URL that opens a PDF in the vendored viewer.

    Args:
        pdf_url (str): URL the viewer should load, already percent-encoded.
        page (int): Page number to open at, 1-based.
        sidebar (bool): Open the outline sidebar. Off by default because pdf.js
            otherwise restores whatever the reader last left open, and in the
            two-pane mode that means two sidebars eating most of two half-width
            panes. The sidebar button still works.

    Returns:
        str: A viewer URL such as
            ``/static/pdfjs/web/viewer.html?file=%2Fapi%2Fpdf%3F...#page=3``.
    """
    from urllib.parse import quote

    mode = "" if sidebar else "&pagemode=none"
    return (f"/static/{PDFJS_DIRNAME}/web/viewer.html?file={quote(pdf_url, safe='')}"
            f"#page={page}{mode}")
