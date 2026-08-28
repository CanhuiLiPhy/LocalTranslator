"""
中文：文档库管理——文档标识、输出目录解析、不覆盖同名文件。

Document library management.

Decides three things for every PDF entering the system:

1. **Identity** — a stable ``doc_id`` derived from file content, so the same
   paper imported twice (say, downloaded from arXiv and later dragged in from
   Downloads) is recognised as one document with one history.
2. **Where translations go** — next to the source file for PDFs already on disk,
   or into a per-document library folder for uploads and arXiv downloads.
3. **Not clobbering anything** — an existing output file is never overwritten;
   a versioned name is chosen instead.
"""

from __future__ import annotations

import hashlib
import logging
import re
import unicodedata
from pathlib import Path

logger = logging.getLogger(__name__)

SOURCE_LOCAL = "local"
SOURCE_UPLOAD = "upload"
SOURCE_ARXIV = "arxiv"

# Read the file in 1 MiB blocks when hashing / 哈希时按 1MiB 分块读取
_HASH_BLOCK = 1 << 20


def compute_doc_id(path: Path) -> str:
    """
    中文：根据文件内容计算稳定的文档标识。

    Derive a stable document id from file content.

    Content addressing (rather than hashing the path) means re-importing the
    same PDF from a different directory reuses the existing document and its
    translation history instead of creating a duplicate entry.

    Args:
        path (Path): Path to the source PDF.

    Returns:
        str: First 16 hex characters of the SHA-256 of the file contents — short
            enough for readable directory names, wide enough that an accidental
            collision across a personal library is not a practical concern.

    Raises:
        OSError: If the file cannot be read.
    """
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(_HASH_BLOCK):
            digest.update(block)
    return digest.hexdigest()[:16]


def slugify(text: str, max_length: int = 60) -> str:
    """
    中文：把标题转成适合做目录名的短横线小写串。

    Convert a title into a filesystem-safe directory name.

    Args:
        text (str): Arbitrary title text, possibly containing CJK characters,
            punctuation and whitespace.
        max_length (int): Maximum length of the result.

    Returns:
        str: Lowercase, hyphen-separated slug containing only word characters
            and hyphens. Returns ``"document"`` if nothing usable remains.
    """
    normalised = unicodedata.normalize("NFKC", text).strip().lower()
    slug = re.sub(r"[^\w一-鿿]+", "-", normalised, flags=re.UNICODE)
    slug = re.sub(r"-{2,}", "-", slug).strip("-")
    return slug[:max_length] or "document"


def unique_path(path: Path) -> Path:
    """
    中文：若目标已存在则改名（加 .v2/.v3 后缀），绝不覆盖。

    Return a path that does not yet exist, versioning the name if necessary.

    Overwriting is never the right default here: a translated PDF may be the
    result of an hour of local inference, and silently replacing it because a
    document was re-run with different settings would destroy work.

    Args:
        path (Path): Desired output path, e.g. ``paper.zh.dual.pdf``.

    Returns:
        Path: ``path`` itself when free, otherwise the first available
            ``paper.zh.dual.v2.pdf``, ``…v3.pdf`` and so on.
    """
    if not path.exists():
        return path

    stem, suffix = path.stem, path.suffix
    parent = path.parent
    version = 2
    while True:
        candidate = parent / f"{stem}.v{version}{suffix}"
        if not candidate.exists():
            return candidate
        version += 1


def resolve_output_dir(
    source_path: Path,
    source_kind: str,
    library_dir: Path,
    *,
    title: str | None = None,
    doc_id: str | None = None,
) -> Path:
    """
    中文：决定译文写到哪个目录。

    Decide where a document's translations should be written.

    Local files keep their translations beside the original, which is what makes
    the output immediately useful in the user's own folder structure. Uploads and
    arXiv downloads have no meaningful "original folder", so they get a dedicated
    directory in the library holding the source PDF and its translations together.

    Args:
        source_path (Path): Path to the source PDF.
        source_kind (str): One of ``"local"``, ``"upload"``, ``"arxiv"``.
        library_dir (Path): Root of the managed library.
        title (str | None): Document title, used to name library folders.
        doc_id (str | None): Document id, used as a fallback folder name.

    Returns:
        Path: Directory that will receive the translated PDFs. Created if missing.
    """
    if source_kind == SOURCE_LOCAL:
        target = source_path.parent
    else:
        name = slugify(title) if title else (doc_id or source_path.stem)
        target = library_dir / name

    target.mkdir(parents=True, exist_ok=True)
    return target


def expected_outputs(source_path: Path, output_dir: Path, lang_out: str) -> dict[str, Path]:
    """
    中文：推算 BabelDOC 将生成的产物路径。

    Predict the output filenames BabelDOC will produce.

    BabelDOC names its results ``<stem>.<lang_out>.dual.pdf`` and
    ``<stem>.<lang_out>.mono.pdf`` in the configured output directory. Knowing
    these in advance lets the UI show whether a document has already been
    translated without consulting the database.

    Args:
        source_path (Path): Source PDF path.
        output_dir (Path): Directory outputs are written to.
        lang_out (str): Target language code, e.g. ``"zh"``.

    Returns:
        dict[str, Path]: Keys ``dual`` and ``mono`` mapped to their paths. The
            files may or may not exist yet.
    """
    stem = source_path.stem
    return {
        "dual": output_dir / f"{stem}.{lang_out}.dual.pdf",
        "mono": output_dir / f"{stem}.{lang_out}.mono.pdf",
    }


def already_translated(source_path: Path, output_dir: Path, lang_out: str) -> bool:
    """
    中文：判断该文档是否已有译文产物（用于批量翻译时跳过）。

    Check whether translations already exist for a document.

    Used by folder batch import to skip work that has already been done.

    Args:
        source_path (Path): Source PDF path.
        output_dir (Path): Directory outputs would be written to.
        lang_out (str): Target language code.

    Returns:
        bool: ``True`` if either the dual or mono output is present.
    """
    outputs = expected_outputs(source_path, output_dir, lang_out)
    return any(p.exists() for p in outputs.values())


def read_pdf_metadata(path: Path) -> dict[str, object]:
    """
    中文：读取 PDF 的页数、标题等基本元信息。

    Extract lightweight metadata from a PDF.

    Failures are non-fatal: a PDF that PyMuPDF cannot open here may still be
    translatable, and a missing page count only degrades the UI slightly.

    Args:
        path (Path): Path to the PDF.

    Returns:
        dict[str, object]: Keys ``page_count`` (int | None), ``title``
            (str | None) and ``file_size`` (int). Values are ``None`` when
            unavailable.
    """
    info: dict[str, object] = {
        "page_count": None,
        "title": None,
        "file_size": path.stat().st_size if path.exists() else 0,
    }
    try:
        import pymupdf

        with pymupdf.open(path) as doc:
            info["page_count"] = doc.page_count
            raw_title = (doc.metadata or {}).get("title") or ""
            title = raw_title.strip()
            # Many PDFs carry a useless embedded title such as "Microsoft Word -
            # untitled" or the LaTeX job name; prefer the filename in that case.
            # 很多 PDF 内嵌标题没有意义，这类情况回退用文件名。
            if title and len(title) > 4 and not title.lower().startswith("microsoft word"):
                info["title"] = title
    except Exception:
        logger.debug("Could not read PDF metadata from %s", path, exc_info=True)
    return info


# The suffix a copy's title carries. Full-width brackets because the titles they
# attach to are read in a Chinese interface, and because a Latin "(1)" is far
# more likely to already occur at the end of a real paper title.
# 副本标题的后缀。用全角括号：这些标题是在中文界面里读的，而且真实论文标题末尾出现
# 半角 "(1)" 的可能性要大得多。
_COPY_SUFFIX = re.compile(r"（(\d+)）\s*$")


def base_title(title: str) -> str:
    """
    中文：去掉标题末尾的副本序号，得到原始标题。

    Strip a copy suffix from a title.

    Args:
        title (str): Title, possibly ending in 「（2）」.

    Returns:
        str: The title without its copy suffix, whitespace trimmed.
    """
    return _COPY_SUFFIX.sub("", title or "").strip()


def copy_title(title: str, index: int) -> str:
    """
    中文：给标题加上副本序号。

    Add a copy number to a title.

    Args:
        title (str): Base title, without a copy suffix.
        index (int): Copy number, 1 for the first copy.

    Returns:
        str: e.g. ``"Attention Is All You Need（1）"``.
    """
    return f"{base_title(title)}（{index}）"


def next_copy_index(existing: list) -> int:
    """
    中文：根据库里已有的同名条目算出下一个副本序号。

    Choose the next free copy number for a title.

    Args:
        existing (list): Document rows already carrying this title, as returned
            by :meth:`Database.documents_titled`.

    Returns:
        int: The lowest number not yet used, starting at 1.
    """
    used = set()
    for row in existing:
        match = _COPY_SUFFIX.search(row["title"] or "")
        if match:
            used.add(int(match.group(1)))
    index = 1
    while index in used:
        index += 1
    return index


def register_document(
    db,
    source_path: Path,
    source_kind: str,
    library_dir: Path,
    *,
    title: str | None = None,
    source_ref: str | None = None,
    authors: str | None = None,
    copy_index: int | None = None,
) -> tuple[str, Path]:
    """
    中文：把一个 PDF 登记进文档库，返回 (doc_id, 输出目录)。

    Register a PDF in the library, creating or refreshing its database row.

    Args:
        db (Database): Database handle.
        source_path (Path): Path to the source PDF. Must exist.
        source_kind (str): ``"local"``, ``"upload"`` or ``"arxiv"``.
        library_dir (Path): Root of the managed library.
        title (str | None): Known title; falls back to PDF metadata, then to the
            filename stem.
        source_ref (str | None): Original reference such as an arXiv URL.
        authors (str | None): Comma-separated authors, when known.
        copy_index (int | None): When set, register this as a numbered copy of a
            document already in the library rather than reusing its entry. The
            id is salted with the number and the title gains a 「（N）」 suffix,
            which is what keeps the two runs apart in the library, in their
            output folders and in the reader.

    Returns:
        tuple[str, Path]: ``(doc_id, output_dir)``.

    Raises:
        FileNotFoundError: If ``source_path`` does not exist.
    """
    if not source_path.exists():
        raise FileNotFoundError(source_path)

    doc_id = compute_doc_id(source_path)
    meta = read_pdf_metadata(source_path)
    effective_title = title or meta.get("title") or source_path.stem

    # A copy needs its own identity. The id is content-addressed, so the same
    # PDF submitted twice would otherwise land on the same row and the second
    # run would appear to do nothing — which is exactly the reported defect.
    # 副本必须有自己的身份。id 是按内容算的，同一个 PDF 提交两次否则会落到同一行上，
    # 第二次运行看起来就像什么都没发生——这正是用户报告的问题。
    if copy_index:
        doc_id = f"{doc_id}-{copy_index}"
        effective_title = copy_title(str(effective_title), copy_index)

    output_dir = resolve_output_dir(
        source_path,
        source_kind,
        library_dir,
        title=str(effective_title),
        doc_id=doc_id,
    )

    db.upsert_document(
        doc_id,
        source_path=source_path,
        output_dir=output_dir,
        source_kind=source_kind,
        title=str(effective_title),
        source_ref=source_ref,
        page_count=meta.get("page_count"),
        file_size=meta.get("file_size"),
        authors=authors,
    )
    return doc_id, output_dir
