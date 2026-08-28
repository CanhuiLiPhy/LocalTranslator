"""
中文：随源码发布的术语库快照——读它、写它，以及它为什么取代了联网抓取。

The glossary that ships with the source, rather than being fetched at first run.

**What it replaces.** Seeding used to contact two hosts: a SPARQL endpoint at
query.wikidata.org, once per subject class with a four-second pause between
classes, and a Markdown file on raw.githubusercontent.com. On one measured run
that took 25 minutes 30 seconds, several classes were rate-limited into a
two-minute backoff, two gave up after four attempts, and the packaged
application began the whole thing without anyone pressing anything. What came
back was also not stable: Wikidata moves underneath you, so two installations a
month apart did not get the same glossary, and nobody could say what either one
had.

**What it is now.** One tab-separated file beside this module, read in
milliseconds, identical on every machine and in every edition, and reviewable in
a diff like any other source file. ``git checkout <rev> -- glossary.tsv``
restores exactly the corpus a past release shipped, which the network path could
never do.

**Why a text file and not a database.** Because the point is that a person
curates it. A binary would make ``git diff`` useless, and the whole reason this
file exists is that its author edits it and wants to see what changed. Rows are
written in a fixed order for the same reason: re-exporting an unchanged corpus
must produce a byte-identical file, so a diff shows the edits and nothing else.

**Retired terms travel too.** A row with ``enabled = 0`` is a decision, not an
absence — it says a person looked at this term and said no. The store's
existence check matches on the source and target alone and never consults
``enabled``, so shipping the retired rows is what stops a later import from
quietly reinstating them.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

from local_translator.glossary.store import POLICY_KEEP, POLICY_TRANSLATE, TermRecord

logger = logging.getLogger(__name__)

# The snapshot, beside the module that reads it. PyInstaller does not collect a
# data file just because it sits in a package, so ``packaging/localtrans.spec``
# names this one explicitly — see the ``datas`` entry there. A previous data
# file in this directory, ``common-english.txt``, was never named and therefore
# never shipped: in the packaged application the rule that consults it silently
# did nothing.
# 快照文件，与读它的模块同目录。PyInstaller 不会仅因为一个数据文件位于包内就收集它，
# 因此 packaging/localtrans.spec 里显式点名了它——见那里的 datas 项。本目录下先前那个
# 数据文件 common-english.txt 就从未被点名、因而从未进包：在打包版里，依赖它的那条规则
# 一直在静默空转。
SNAPSHOT = Path(__file__).resolve().parent / "glossary.tsv"

# The provenance written on every row of an export. The original source is kept
# in the file's own column, so this is not a loss — it is the answer to "where
# did this installation's glossary come from", which is now always the same.
# 导出时写在每一行上的来源。原始来源仍保留在文件自己的列里，因此这不是信息损失——它回答
# 的是「这台机器上的术语库从哪来」，而这个答案如今始终一致。
BUNDLED = "bundled"

_COLUMNS = ("src", "tgt", "policy", "domain", "source", "confidence", "enabled")

# Characters that would break the format if a term contained one. None does in
# the corpus this was written against; the check exists so that the day one
# does, the export refuses rather than writing a file the loader misreads.
# 一旦术语里出现就会破坏格式的字符。在编写本模块所依据的语料里一个也没有；这项检查的意义
# 在于：将来真出现时，导出会拒绝执行，而不是写出一个加载器会读错的文件。
_FORBIDDEN = ("\t", "\n", "\r")


def _clean(value: object) -> str:
    """
    中文：把一个字段转成可写进 TSV 的字符串。

    Render one field as a string safe for the file.

    Args:
        value (object): Column value from the database, possibly ``None``.

    Returns:
        str: The value as text, with ``None`` becoming an empty field.
    """
    return "" if value is None else str(value).strip()


def export(rows, target: Path = SNAPSHOT) -> dict[str, int]:
    """
    中文：把术语库写成随源码发布的快照文件。

    Write the glossary out as the snapshot that ships with the source.

    Rows are sorted by their normalised source term and then by target, so that
    exporting an unchanged corpus twice produces the same bytes and a diff shows
    only real edits. The header carries a digest of the data lines alone — not
    of the whole file — so that re-exporting identical content does not change
    the digest even if a comment above it did.

    Args:
        rows (Iterable[Mapping]): Database rows carrying at least the keys in
            :data:`_COLUMNS`. Extra keys are ignored; ``hit_count``,
            ``created_at`` and the row id are deliberately not written, being
            facts about one machine rather than about the glossary.
        target (Path): Where to write. Defaults to the shipped location.

    Returns:
        dict[str, int]: ``{"rows": n, "enabled": n, "bytes": n}``.

    Raises:
        ValueError: If any term contains a tab or a newline, which the format
            cannot represent and the loader could not read back.
    """
    prepared: list[tuple[str, ...]] = []
    for row in rows:
        values = tuple(_clean(row[name] if name in row.keys() else None)
                       for name in _COLUMNS)
        for name, value in zip(_COLUMNS, values, strict=True):
            if any(bad in value for bad in _FORBIDDEN):
                raise ValueError(
                    f"术语字段 {name} 含制表符或换行，无法写入快照：{value!r}"
                )
        prepared.append(values)

    prepared.sort(key=lambda v: (v[0].lower(), v[1]))
    body = "\n".join("\t".join(v) for v in prepared)
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()[:16]
    enabled = sum(1 for v in prepared if v[-1] == "1")

    text = (
        "# 本地翻译 术语库快照 / Local Translator glossary snapshot\n"
        "# 由 `localtrans glossary snapshot` 生成，请勿手工编辑——改术语库请用界面，再重新导出。\n"
        "# Generated by `localtrans glossary snapshot`; edit the glossary in the\n"
        "# interface and re-export rather than editing this file by hand.\n"
        f"# digest: {digest}\n"
        f"# rows: {len(prepared)} ({enabled} enabled)\n"
        f"# columns: {chr(9).join(_COLUMNS)}\n"
        f"{body}\n"
    )
    target.write_text(text, encoding="utf-8")
    return {"rows": len(prepared), "enabled": enabled,
            "bytes": len(text.encode("utf-8"))}


def load(path: Path = SNAPSHOT) -> list[TermRecord]:
    """
    中文：读入随源码发布的术语库快照。

    Read the shipped snapshot.

    A malformed line is skipped with a warning rather than raising: a glossary
    with one bad row is worth more than no glossary at all, and the alternative
    is an application that will not start because of a typo in a data file.

    Args:
        path (Path): File to read. Defaults to the shipped location.

    Returns:
        list[TermRecord]: Every row, retired ones included. Empty when the file
            is absent, which is what a source checkout looks like before the
            first export.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        logger.warning("No bundled glossary at %s; nothing to seed from", path)
        return []

    records: list[TermRecord] = []
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip() or line.startswith("#"):
            continue
        fields = line.split("\t")
        if len(fields) != len(_COLUMNS):
            logger.warning("Glossary snapshot line %d has %d fields, expected %d; skipped",
                           number, len(fields), len(_COLUMNS))
            continue
        src, tgt, policy, domain, source, confidence, enabled = fields
        if not src.strip() or not tgt.strip():
            continue
        try:
            trust = float(confidence) if confidence else 1.0
        except ValueError:
            trust = 1.0
        records.append(TermRecord(
            src=src,
            tgt=tgt,
            policy=POLICY_KEEP if policy == POLICY_KEEP else POLICY_TRANSLATE,
            domain=domain or None,
            source=source or BUNDLED,
            confidence=trust,
            enabled=enabled != "0",
        ))
    return records


def digest(path: Path = SNAPSHOT) -> str:
    """
    中文：返回快照数据行的摘要，用于判断本机是否已经导入过这一版。

    Return a digest of the snapshot's data lines.

    Computed over the rows alone, so that regenerating a file whose terms did
    not change yields the same digest even though the comment header did.

    Args:
        path (Path): File to read.

    Returns:
        str: Sixteen hex characters, or an empty string when the file is absent.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return ""
    body = "\n".join(line for line in text.splitlines()
                     if line.strip() and not line.startswith("#"))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()[:16]
