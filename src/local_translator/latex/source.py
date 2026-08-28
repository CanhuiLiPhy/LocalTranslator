"""
中文：取回并解开一篇 arXiv 论文的 LaTeX 源码。

Fetch and unpack a paper's LaTeX source from arXiv.

arXiv serves the submitted source at ``/e-print/{id}``, usually a gzipped tar
of the whole project — the ``.tex`` files, the figures, the class and any
bibliography. Occasionally it is a bare gzipped ``.tex``, and rarely a PDF the
author uploaded instead of source, which has no source to translate.
"""

from __future__ import annotations

import gzip
import logging
import shutil
import tarfile
import zipfile
from pathlib import Path

import httpx

# Name of the translated document written beside the original sources. It lives
# here rather than in the pipeline because :func:`find_main` has to know it: a
# previous run's output is a ``.tex`` with a document class and more text than
# the original, so it wins the "which file is the paper" contest outright.
# 与原始源码并排写出的译文文件名。定义在这里而不是在 pipeline 里，因为 find_main 必须
# 知道它：上一次运行的产物是一个带文档类、正文还比原文长的 .tex，不排除就会直接赢得
# 「哪个文件才是论文」的比较。
TRANSLATED = "translated-zh.tex"

logger = logging.getLogger(__name__)

EPRINT = "https://arxiv.org/e-print/{}"

# arXiv asks automated clients to identify themselves. A request without this
# is served more slowly and may be refused outright.
# arXiv 要求自动化客户端表明身份。不带这个的请求会被降速，甚至直接拒绝。
_AGENT = "local-translator/2.0 (LaTeX source translation)"

# Largest source archive to accept. The biggest in this project's bench is the
# Gemini report at 26 MB; well past this is a mistake rather than a paper.
# 接受的源码包体积上限。本项目 bench 里最大的是 Gemini 那篇 26 MB；远超此值不是论文，
# 是出了岔子。
_MAX_BYTES = 200 * 1024 * 1024


class NoSource(RuntimeError):
    """中文：这篇没有可用的 LaTeX 源码。 / This paper has no usable LaTeX source."""


def _safe_members(archive, destination: Path):
    """
    中文：过滤掉会写到目标目录之外的条目。

    Drop archive entries that would write outside the destination.

    Args:
        archive: An open tar or zip archive.
        destination (Path): Where the archive is being extracted.

    Yields:
        Members that resolve inside ``destination``. An archive containing
        ``../../etc/passwd`` is not a paper, and extracting it would write
        wherever the entry pointed.
    """
    root = destination.resolve()
    members = archive.getmembers() if hasattr(archive, "getmembers") else archive.infolist()
    for member in members:
        name = getattr(member, "name", None) or getattr(member, "filename", "")
        target = (root / name).resolve()
        if target == root or root in target.parents:
            yield member
        else:
            logger.warning("源码包里有指向目录之外的条目，已跳过：%s", name)


async def download(arxiv_id: str, destination: Path, timeout: float = 180.0) -> Path:
    """
    中文：下载 e-print 源码包。

    Download the e-print archive.

    Args:
        arxiv_id (str): Identifier without a version suffix.
        destination (Path): Directory to write the archive into.
        timeout (float): Seconds to allow.

    Returns:
        Path: The downloaded archive.

    Raises:
        NoSource: When arXiv has nothing to serve, or serves something too
            large to be a paper.
    """
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / f"{arxiv_id}.archive"
    url = EPRINT.format(arxiv_id)
    async with httpx.AsyncClient(follow_redirects=True, timeout=timeout,
                                 headers={"User-Agent": _AGENT}) as client:
        response = await client.get(url)
        if response.status_code == 404:
            raise NoSource(f"arXiv 上没有 {arxiv_id} 的源码（可能只提供了 PDF）")
        response.raise_for_status()
        payload = response.content
    if len(payload) > _MAX_BYTES:
        raise NoSource(f"源码包 {len(payload) / 1048576:.0f} MB，超过上限")
    if len(payload) < 512:
        raise NoSource("源码包太小，不像是一篇论文")
    archive.write_bytes(payload)
    logger.info("下载 %s 源码：%.1f MB", arxiv_id, len(payload) / 1048576)
    return archive


def unpack(archive: Path, destination: Path) -> Path:
    """
    中文：解开源码包，自动识别 tar、zip 或单个 gzip 文件。

    Unpack an e-print archive, whatever shape it is in.

    Args:
        archive (Path): The downloaded file.
        destination (Path): Directory to extract into.

    Returns:
        Path: The directory holding the sources.

    Raises:
        NoSource: When the archive is a PDF, or nothing that can be read.
    """
    destination.mkdir(parents=True, exist_ok=True)
    head = archive.read_bytes()[:5]
    if head.startswith(b"%PDF"):
        raise NoSource("arXiv 只提供了 PDF，这篇没有 LaTeX 源码可翻译")

    if tarfile.is_tarfile(archive):
        with tarfile.open(archive) as tar:
            tar.extractall(destination, members=list(_safe_members(tar, destination)))
        return destination
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as zipped:
            zipped.extractall(destination,
                              members=list(_safe_members(zipped, destination)))
        return destination

    # A bare gzipped file, which arXiv uses for a single-file submission.
    # 单个 gzip 文件，arXiv 用它来提供只有一个文件的投稿。
    try:
        with gzip.open(archive) as source:
            content = source.read()
    except OSError as problem:
        raise NoSource(f"源码包读不开：{problem}") from problem
    if content[:5] == b"%PDF":
        raise NoSource("arXiv 只提供了 PDF，这篇没有 LaTeX 源码可翻译")
    (destination / "main.tex").write_bytes(content)
    return destination


def find_main(root: Path) -> Path | None:
    """
    中文：在源码目录里找出主 .tex 文件。

    Find the main ``.tex`` file in an unpacked source tree.

    Args:
        root (Path): The directory holding the sources.

    Returns:
        Path | None: The main file, or ``None`` when nothing declares a
            document class.

    Note:
        A paper may carry several files with ``\\documentclass`` — a template
        left behind, a supplement, a rebuttal. The one with
        ``\\begin{document}`` *and* the most text is the paper; the others are
        leftovers. Preferring a file called ``main`` or ``ms`` breaks on the
        many papers that call theirs something else.
        一篇论文里可能有好几个带 \\documentclass 的文件——留下的模板、补充材料、答辩稿。
        既有 \\begin{document}、正文又最多的那个才是论文，其余是残留。按文件名叫不叫
        main 来挑，会在大量并非如此命名的论文上出错。
    """
    best, best_size = None, -1
    for candidate in sorted(root.rglob("*.tex")):
        if candidate.name == TRANSLATED:
            # A previous run's output. It has a document class and more text
            # than the original — it is the original plus Chinese — so without
            # this it wins every time and the second run translates the first
            # run's translation.
            # 上一次运行的产物。它有文档类，正文还比原文长——它就是原文加上中文——所以
            # 不排除的话它每次都会胜出，第二次运行翻译的就是第一次的译文。
            continue
        try:
            text = candidate.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if "\\documentclass" not in text or "\\begin{document}" not in text:
            continue
        if len(text) > best_size:
            best, best_size = candidate, len(text)
    return best


def prepare_workspace(archive: Path, workspace: Path) -> tuple[Path, Path]:
    """
    中文：解包到工作目录并找到主文件。

    Unpack into a workspace and locate the main file.

    Args:
        archive (Path): The downloaded archive.
        workspace (Path): Directory to build in. Cleared first, so a retry does
            not compile against a previous run's leftovers.
        
    Returns:
        tuple[Path, Path]: The source directory and the main ``.tex``.

    Raises:
        NoSource: When no main file can be identified.
    """
    if workspace.exists():
        shutil.rmtree(workspace)
    unpack(archive, workspace)
    main = find_main(workspace)
    if main is None:
        raise NoSource("源码里找不到带 \\documentclass 和 \\begin{document} 的主文件")
    logger.info("主文件：%s", main.relative_to(workspace))
    return workspace, main
