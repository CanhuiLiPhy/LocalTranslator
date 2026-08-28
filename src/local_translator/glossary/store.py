"""
中文：累积式术语库的读写层。

Cumulative glossary storage.

The glossary is the mechanism that makes terminology consistent across documents
*and* progressively removes the need for the cloud API. Every term the system
learns is written here, so each subsequent paper starts with better coverage
than the last. Once coverage of a document's technical vocabulary passes a
threshold, API-based term extraction is skipped entirely and the run is fully
local and free.

Two term policies share one table:

``translate``
    A preferred rendering: ``ablation study → 消融实验``.

``keep``
    A whitelist entry that must stay in the source language, stored with
    ``tgt == src``. Representing it this way rather than as a separate concept
    means it flows through both translation backends unchanged: the local prompt
    lists it under "keep in the source language", and BabelDOC's own glossary
    block — whose instruction is "always use the glossary's Target Term for any
    occurrence of its Source Term" — reproduces the source term verbatim.
"""

from __future__ import annotations

import csv
import io
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from local_translator.db import Database, utcnow

logger = logging.getLogger(__name__)

POLICY_TRANSLATE = "translate"
POLICY_KEEP = "keep"

# Domains the seed importer classifies terms into. Ordered as a reader would
# scan them — the physical sciences, then the formal ones, then the applied —
# because this tuple drives the order of the filter and of the summary tiles.
# 种子导入时的学科分类。顺序按阅读习惯排列：先自然科学，再形式科学，后应用学科；
# 该元组同时决定筛选下拉与统计卡片的顺序。
DOMAINS = (
    "physics", "chemistry", "materials", "biology", "medicine",
    "math", "cs", "ai", "engineering", "general",
)

_WHITESPACE = re.compile(r"\s+")


def normalise_source(term: str) -> str:
    """
    中文：规范化术语用于去重（小写、压缩空白）。

    Normalise a source term for deduplication.

    Args:
        term (str): Raw source term.

    Returns:
        str: Lower-cased term with runs of whitespace collapsed to one space
            and surrounding whitespace removed.
    """
    return _WHITESPACE.sub(" ", term.lower()).strip()


@dataclass
class TermRecord:
    """
    中文：一条术语记录。

    A single glossary entry.

    Attributes:
        src (str): Source-language term as it should be displayed.
        tgt (str): Target-language rendering; equals ``src`` when ``policy`` is
            ``"keep"``.
        policy (str): ``"translate"`` or ``"keep"``.
        domain (str | None): Subject area, one of :data:`DOMAINS`.
        source (str | None): Provenance — ``wikidata``, ``jiqizhixin``,
            ``builtin``, ``auto-api`` or ``manual``.
        confidence (float): Trust score, used to prefer better sources when the
            same term arrives from several places.
        enabled (bool): Whether the matcher may use this term. ``False`` records
            a decision to retire it: the row stays so that a later import cannot
            silently bring it back, because the existence check below matches on
            ``src_norm`` and ``tgt`` alone and does not look at this column.
    """

    src: str
    tgt: str
    policy: str = POLICY_TRANSLATE
    domain: str | None = None
    source: str | None = None
    confidence: float = 1.0
    enabled: bool = True


class GlossaryStore:
    """
    中文：术语库的增删改查与批量导入导出。

    Read/write access to the cumulative glossary.

    Attributes:
        db (Database): Underlying database handle.
    """

    def __init__(self, db: Database):
        self.db = db

    # ------------------------------------------------------------------ #
    # Writing / 写入
    # ------------------------------------------------------------------ #

    def upsert_many(self, terms: list[TermRecord], *, overwrite: bool = False) -> dict[str, int]:
        """
        中文：批量写入术语，已存在的默认保留原值。

        Insert many terms in one transaction.

        By default an existing entry is left alone, which is what makes the
        accumulation safe: a bulk re-import, or a low-confidence automatic
        extraction, can never silently overwrite a rendering the user corrected
        by hand. Pass ``overwrite=True`` only for deliberate refreshes.

        Args:
            terms (list[TermRecord]): Entries to write.
            overwrite (bool): Replace target, domain and confidence of existing
                entries instead of keeping them.

        Returns:
            dict[str, int]: Counts under keys ``inserted``, ``updated`` and
                ``skipped``.
        """
        inserted = updated = skipped = 0
        now = utcnow()

        with self.db.write() as conn:
            for term in terms:
                src = term.src.strip()
                tgt = (term.tgt or "").strip()
                if term.policy == POLICY_KEEP:
                    tgt = src
                if not src or not tgt:
                    skipped += 1
                    continue

                norm = normalise_source(src)
                existing = conn.execute(
                    "SELECT id FROM glossary_terms WHERE src_norm = ? AND tgt = ?",
                    (norm, tgt),
                ).fetchone()

                if existing is not None:
                    if overwrite:
                        conn.execute(
                            """
                            UPDATE glossary_terms
                            SET policy = ?, domain = ?, source = ?, confidence = ?,
                                enabled = ?, updated_at = ?
                            WHERE id = ?
                            """,
                            (term.policy, term.domain, term.source, term.confidence,
                             1 if term.enabled else 0, now, existing["id"]),
                        )
                        updated += 1
                    else:
                        skipped += 1
                    continue

                conn.execute(
                    """
                    INSERT INTO glossary_terms
                        (src, src_norm, tgt, policy, domain, source, confidence,
                         hit_count, enabled, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?)
                    """,
                    (src, norm, tgt, term.policy, term.domain, term.source,
                     term.confidence, 1 if term.enabled else 0, now, now),
                )
                inserted += 1

        logger.info(
            "Glossary upsert: %d inserted, %d updated, %d skipped", inserted, updated, skipped
        )
        return {"inserted": inserted, "updated": updated, "skipped": skipped}

    def delete_by_source(self, source: str) -> int:
        """
        中文：删除某个来源导入的全部术语，用于重新导入时先清掉旧数据。

        Remove every term that came from one import source.

        Re-seeding cannot simply overwrite: a class that is no longer queried
        leaves its old rows behind, and a pair that was dropped for being wrong
        stays in the glossary forever. Clearing the source first makes a re-seed
        mean what it says.

        Hand-edited and learned terms are untouched, because they carry a
        different source value.

        Args:
            source (str): Import source, e.g. ``wikidata``.

        Returns:
            int: Number of rows removed.
        """
        with self.db.write() as conn:
            cursor = conn.execute("DELETE FROM glossary_terms WHERE source = ?", (source,))
            return cursor.rowcount or 0

    def add(self, term: TermRecord) -> None:
        """
        中文：添加或更新单条术语（手工编辑时使用，会覆盖已有值）。

        Add or replace one term, as used by manual editing in the UI.

        Args:
            term (TermRecord): The entry to write.
        """
        self.upsert_many([term], overwrite=True)

    def update(self, term_id: int, **fields: Any) -> None:
        """
        中文：更新指定术语的字段。

        Update columns of one glossary row.

        Args:
            term_id (int): Row identifier.
            **fields: Column/value pairs. When ``src`` is changed, ``src_norm``
                is recomputed so lookups stay correct.
        """
        if not fields:
            return
        if "src" in fields:
            fields["src_norm"] = normalise_source(str(fields["src"]))
        fields["updated_at"] = utcnow()

        assignments = ", ".join(f"{k} = ?" for k in fields)
        with self.db.write() as conn:
            conn.execute(
                f"UPDATE glossary_terms SET {assignments} WHERE id = ?",
                (*fields.values(), term_id),
            )

    def delete(self, term_id: int) -> None:
        """
        中文：删除一条术语。

        Args:
            term_id (int): Row identifier.
        """
        with self.db.write() as conn:
            conn.execute("DELETE FROM glossary_terms WHERE id = ?", (term_id,))

    def set_policy(self, term_id: int, policy: str) -> None:
        """
        中文：切换术语策略；改为 keep 时把译文设为原文本身。

        Switch a term between "translate this" and "keep it in English".

        Args:
            term_id (int): Row identifier.
            policy (str): ``"translate"`` or ``"keep"``.
        """
        row = self.db.query_one("SELECT src FROM glossary_terms WHERE id = ?", (term_id,))
        if row is None:
            return
        fields: dict[str, Any] = {"policy": policy}
        if policy == POLICY_KEEP:
            fields["tgt"] = row["src"]
        self.update(term_id, **fields)

    def set_enabled(self, term_id: int, enabled: bool) -> None:
        """
        中文：停用或恢复一条术语；停用后不再注入翻译，但记录保留。

        Retire a term from the live glossary, or bring it back.

        A retired row is kept rather than deleted for two reasons: the user can
        review and reverse a cleaning decision, and its presence stops a later
        ``glossary seed`` re-import from silently re-adding the same entry,
        because ``upsert_many`` matches on ``(src_norm, tgt)`` regardless of
        this flag.

        Args:
            term_id (int): Row identifier.
            enabled (bool): ``True`` to use the term when translating,
                ``False`` to retire it.

        Returns:
            None.
        """
        self.update(term_id, enabled=1 if enabled else 0)

    def force_keep(self, term: str, *, domain: str | None = None) -> str:
        """
        中文：把一个词强制设为「保持英文」，并清掉它已有的其它译法。

        Make a term keep its English form, displacing any rendering it has.

        Adding a keep entry cannot be a plain insert. The glossary is keyed on
        ``(src_norm, tgt)``, so a term that already has a Chinese rendering —
        「Agent → 智能体」 — would gain a *second* row rather than change, and the
        matcher would then offer the translator both. So every row for the term
        is found first: one becomes the keep entry, the rest are retired.

        Args:
            term (str): The term, spelled as it should appear in the output.
            domain (str | None): Subject to file it under. ``None`` leaves an
                existing row's subject alone and files a new row under nothing.

        Returns:
            str: ``"added"``, ``"converted"`` when an existing rendering was
                displaced, or ``"unchanged"`` when it was already a keep entry.
        """
        norm = normalise_source(term)
        rows = self.db.query(
            "SELECT id, src, tgt, policy, enabled, domain FROM glossary_terms "
            "WHERE src_norm = ? ORDER BY id", (norm,))

        if not rows:
            self.upsert_many([TermRecord(
                src=term, tgt=term, policy=POLICY_KEEP,
                domain=domain, source="manual", confidence=1.0)])
            return "added"

        # Prefer a row that is already the keep entry, so the term's own
        # spelling and history survive.
        # 优先选已经是 keep 的那一行，好保住这个词自己的写法与历史。
        chosen = next((r for r in rows if (r["tgt"] or "") == (r["src"] or "")), rows[0])
        was_keep = (chosen["policy"] == POLICY_KEEP and bool(chosen["enabled"])
                    and (chosen["tgt"] or "") == (chosen["src"] or ""))

        # The provenance becomes "manual": this is a decision someone made by
        # hand, and it must survive any later bulk operation on the source it
        # happened to arrive from. Leaving it as "auto-api" once cost four of
        # these entries, retired by a sweep over that source.
        # 来源改为 manual：这是人手做的决定，必须能挺过日后针对它原本来源的任何批量
        # 操作。保留 auto-api 曾让其中四条被一次按来源的清理顺手停用。
        fields: dict[str, Any] = {"src": term, "tgt": term, "policy": POLICY_KEEP,
                                  "enabled": 1, "source": "manual",
                                  "confidence": 1.0}
        if domain:
            fields["domain"] = domain
        try:
            self.update(chosen["id"], **fields)
        except Exception:  # noqa: BLE001
            # Another row already holds this exact pair; retire this one instead
            # of leaving the write half-done.
            # 已有另一行占着同样的组合；停用这一行，而不是让写入停在半路。
            logger.debug("Could not convert term %s; retiring it", term, exc_info=True)
            self.update(chosen["id"], enabled=0)
            return "unchanged"

        for row in rows:
            if row["id"] != chosen["id"] and row["enabled"]:
                self.update(row["id"], enabled=0)

        return "unchanged" if was_keep else "converted"

    # ------------------------------------------------------------------ #
    # Reading / 读取
    # ------------------------------------------------------------------ #

    def active_terms(self) -> list[tuple[str, str, str]]:
        """
        中文：读取所有启用中的术语，供匹配器使用。

        Load all enabled terms for the paragraph matcher.

        Whitelist entries are returned first. The matcher deduplicates surface
        forms case-insensitively, so when a term exists under both policies —
        the built-in whitelist protects ``token`` while an imported source
        renders ``Token`` as 词元 — whichever arrives first wins. Ordering by
        policy makes the whitelist win deterministically instead of by
        accident of equal string length.

        Returns:
            list[tuple[str, str, str]]: ``(source, target, policy)`` triples,
                whitelist entries first, then longest terms first.
        """
        rows = self.db.query(
            """
            SELECT src, tgt, policy FROM glossary_terms
            WHERE enabled = 1
            ORDER BY (policy = 'keep') DESC, LENGTH(src) DESC
            """
        )
        return [(r["src"], r["tgt"], r["policy"]) for r in rows]

    def search(
        self,
        query: str = "",
        *,
        domain: str | None = None,
        policy: str | None = None,
        enabled: bool | None = True,
        limit: int = 200,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """
        中文：搜索术语，可按学科、策略与启用状态过滤；默认只看启用中的。

        Search the glossary.

        The ``enabled`` default matters: cleaning retires an entry by clearing
        its flag rather than deleting the row, so every reader that does not
        ask otherwise must see the live glossary — the same set the matcher
        injects — and not the retired entries alongside it.

        Args:
            query (str): Substring matched against source and target text.
            domain (str | None): Restrict to one subject area.
            policy (str | None): Restrict to ``"translate"`` or ``"keep"``.
            enabled (bool | None): ``True`` for live entries (the default),
                ``False`` for retired ones, ``None`` for both.
            limit (int): Maximum rows returned.
            offset (int): Rows to skip, for pagination.

        Returns:
            list[dict[str, Any]]: Matching rows, most-used first.
        """
        clauses: list[str] = []
        params: list[Any] = []

        if enabled is not None:
            clauses.append("enabled = ?")
            params.append(1 if enabled else 0)
        if query:
            clauses.append("(src LIKE ? OR tgt LIKE ?)")
            params.extend([f"%{query}%", f"%{query}%"])
        if domain:
            clauses.append("domain = ?")
            params.append(domain)
        if policy:
            clauses.append("policy = ?")
            params.append(policy)

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.extend([limit, offset])

        rows = self.db.query(
            f"""
            SELECT * FROM glossary_terms {where}
            ORDER BY hit_count DESC, LENGTH(src) DESC
            LIMIT ? OFFSET ?
            """,
            tuple(params),
        )
        return [dict(r) for r in rows]

    def stats(self) -> dict[str, Any]:
        """
        中文：返回术语库统计——总数、按策略、按来源、按学科。

        Summarise the glossary's size and composition.

        Every count covers the live glossary only. Retired entries are reported
        separately under ``disabled`` so that the headline figure is the number
        of terms translation actually uses.

        Returns:
            dict[str, Any]: Keys ``total`` (live entries), ``disabled``
                (retired entries), and ``by_policy`` / ``by_source`` /
                ``by_domain``, each mapping a label to a count of live entries.
        """
        total = self.db.query_one(
            "SELECT COUNT(*) AS n FROM glossary_terms WHERE enabled = 1")["n"]
        disabled = self.db.query_one(
            "SELECT COUNT(*) AS n FROM glossary_terms WHERE enabled = 0")["n"]

        def group(column: str) -> dict[str, int]:
            rows = self.db.query(
                f"SELECT {column} AS k, COUNT(*) AS n FROM glossary_terms "
                f"WHERE enabled = 1 GROUP BY {column}"
            )
            return {(r["k"] or "unknown"): r["n"] for r in rows}

        return {
            "total": total,
            "disabled": disabled,
            "by_policy": group("policy"),
            "by_source": group("source"),
            "by_domain": group("domain"),
        }

    def record_hits(self, sources: list[str]) -> None:
        """
        中文：累加术语命中次数，用于排序与清理低价值条目。

        Increment usage counters for terms that matched a document.

        Hit counts drive the ordering in the management UI and make it possible
        to spot noisy imported terms that never fire.

        Args:
            sources (list[str]): Source terms that matched.
        """
        if not sources:
            return
        now = utcnow()
        with self.db.write() as conn:
            for src in sources:
                conn.execute(
                    """
                    UPDATE glossary_terms
                    SET hit_count = hit_count + 1, updated_at = ?
                    WHERE src_norm = ?
                    """,
                    (now, normalise_source(src)),
                )

    # ------------------------------------------------------------------ #
    # Import / export / 导入导出
    # ------------------------------------------------------------------ #

    def export_csv(self, path: Path, *, only_enabled: bool = True) -> int:
        """
        中文：导出为 BabelDOC 兼容的术语表 CSV。

        Export the glossary as a BabelDOC-compatible CSV.

        The ``source,target`` column names are what ``babeldoc.glossary.Glossary.from_csv``
        requires. Whitelist entries export as ``LLM,LLM``, which BabelDOC's own
        glossary instruction turns into "leave this term as-is" — so the API
        backend honours the whitelist without needing any extra mechanism.

        Args:
            path (Path): Destination CSV path.
            only_enabled (bool): Export only enabled terms.

        Returns:
            int: Number of rows written.
        """
        where = "WHERE enabled = 1" if only_enabled else ""
        rows = self.db.query(f"SELECT src, tgt FROM glossary_terms {where} ORDER BY src")

        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["source", "target"])
            for row in rows:
                writer.writerow([row["src"], row["tgt"]])

        logger.info("Exported %d glossary terms to %s", len(rows), path)
        return len(rows)

    def import_csv(
        self,
        path: Path,
        *,
        domain: str | None = None,
        source: str = "manual",
        overwrite: bool = False,
    ) -> dict[str, int]:
        """
        中文：从 CSV 导入术语。

        Import terms from a CSV file.

        Args:
            path (Path): CSV with ``source`` and ``target`` columns, and an
                optional ``policy`` column.
            domain (str | None): Domain to tag imported terms with.
            source (str): Provenance label recorded on each row.
            overwrite (bool): Replace existing entries.

        Returns:
            dict[str, int]: Counts under ``inserted``, ``updated`` and ``skipped``.

        Raises:
            ValueError: If the file lacks the required columns.
        """
        content = path.read_bytes().decode("utf-8-sig", errors="replace")
        reader = csv.DictReader(io.StringIO(content))

        if not reader.fieldnames or not {"source", "target"} <= set(reader.fieldnames):
            raise ValueError(f"{path} must contain 'source' and 'target' columns")

        terms = [
            TermRecord(
                src=(row.get("source") or "").strip(),
                tgt=(row.get("target") or "").strip(),
                policy=(row.get("policy") or POLICY_TRANSLATE).strip(),
                domain=domain,
                source=source,
            )
            for row in reader
        ]
        return self.upsert_many(terms, overwrite=overwrite)
