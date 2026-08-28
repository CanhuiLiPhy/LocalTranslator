"""
中文：SQLite 持久化层——文档、任务、术语库。

SQLite persistence layer.

This module holds the entire durable state of the application: which documents
exist, what happened to every translation job, and the cumulative glossary.
Keeping job state here rather than in the web session is what lets the browser
be refreshed, closed and reopened without disturbing a running translation.

Concurrency model:
    Only the server process writes. Worker subprocesses report progress back
    over a queue and the parent performs the write, which sidesteps
    multi-process write contention entirely. WAL mode is still enabled so
    read-only queries (the UI polling job state) never block the writer.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1

# Job lifecycle states / 任务生命周期状态
JOB_QUEUED = "queued"
JOB_RUNNING = "running"
JOB_COMPLETED = "completed"
JOB_FAILED = "failed"
JOB_CANCELLED = "cancelled"
JOB_INTERRUPTED = "interrupted"

ACTIVE_JOB_STATES = (JOB_QUEUED, JOB_RUNNING)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id            TEXT PRIMARY KEY,
    title         TEXT,
    source_kind   TEXT NOT NULL,
    source_ref    TEXT,
    source_path   TEXT NOT NULL,
    output_dir    TEXT NOT NULL,
    page_count    INTEGER,
    file_size     INTEGER,
    authors       TEXT,
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_documents_created ON documents(created_at DESC);

CREATE TABLE IF NOT EXISTS jobs (
    id                   TEXT PRIMARY KEY,
    doc_id               TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    status               TEXT NOT NULL,
    progress             REAL NOT NULL DEFAULT 0,
    stage                TEXT,
    stage_current        INTEGER DEFAULT 0,
    stage_total          INTEGER DEFAULT 0,
    error                TEXT,
    backend              TEXT,
    config_json          TEXT,
    pid                  INTEGER,

    dual_path            TEXT,
    mono_path            TEXT,
    glossary_path        TEXT,
    tracking_path        TEXT,

    -- Tencent's translation API bills by source character and returns no token
    -- figure at all, so a run through it would otherwise record nothing. These
    -- hold what that service actually charges for.
    -- 腾讯云的翻译接口按原文字符计费，且完全不返回 token 数，因此走它的任务本来什么都
    -- 记不下来。这两列存的正是该服务实际计费的对象。
    source_chars         INTEGER NOT NULL DEFAULT 0,
    target_chars         INTEGER NOT NULL DEFAULT 0,

    prompt_tokens        INTEGER NOT NULL DEFAULT 0,
    completion_tokens    INTEGER NOT NULL DEFAULT 0,
    total_tokens         INTEGER NOT NULL DEFAULT 0,
    para_total           INTEGER NOT NULL DEFAULT 0,
    para_cache_hit       INTEGER NOT NULL DEFAULT 0,
    placeholder_retries  INTEGER NOT NULL DEFAULT 0,
    placeholder_failures INTEGER NOT NULL DEFAULT 0,
    est_cost             REAL,
    duration_ms          INTEGER,

    created_at           TEXT NOT NULL,
    started_at           TEXT,
    finished_at          TEXT
);

CREATE INDEX IF NOT EXISTS idx_jobs_status  ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_doc     ON jobs(doc_id);
CREATE INDEX IF NOT EXISTS idx_jobs_created ON jobs(created_at DESC);

CREATE TABLE IF NOT EXISTS glossary_terms (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    src         TEXT NOT NULL,
    src_norm    TEXT NOT NULL,
    tgt         TEXT NOT NULL,
    policy      TEXT NOT NULL DEFAULT 'translate',
    domain      TEXT,
    source      TEXT,
    confidence  REAL DEFAULT 1.0,
    hit_count   INTEGER NOT NULL DEFAULT 0,
    enabled     INTEGER NOT NULL DEFAULT 1,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL,
    UNIQUE(src_norm, tgt)
);

CREATE INDEX IF NOT EXISTS idx_glossary_norm   ON glossary_terms(src_norm);
CREATE INDEX IF NOT EXISTS idx_glossary_policy ON glossary_terms(policy, enabled);
CREATE INDEX IF NOT EXISTS idx_glossary_domain ON glossary_terms(domain);
"""


def utcnow() -> str:
    """
    中文：返回 ISO-8601 格式的当前 UTC 时间字符串。

    Produce a timestamp for storage.

    ISO-8601 strings sort lexicographically in the same order as chronologically,
    so ``ORDER BY created_at`` works without any date parsing.

    Returns:
        str: e.g. ``"2026-08-17T06:12:44.123456+00:00"``.
    """
    return datetime.now(UTC).isoformat()


class Database:
    """
    中文：SQLite 数据库封装，负责建表、迁移与常用查询。

    Thin wrapper around a SQLite database file.

    Attributes:
        path (Path): Location of the database file.
    """

    def __init__(self, path: Path):
        """
        中文：打开（必要时创建）数据库并应用 schema。

        Args:
            path (Path): Database file path. Parent directories are created.
        """
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._write_lock = threading.Lock()
        self._init_schema()

    # ------------------------------------------------------------------ #
    # Connection handling / 连接管理
    # ------------------------------------------------------------------ #

    def _connect(self) -> sqlite3.Connection:
        """
        中文：返回当前线程的连接（线程局部，避免跨线程共享）。

        Return this thread's connection, opening it on first use.

        SQLite connections are not safe to share across threads, and NiceGUI
        serves requests from a thread pool, so each thread gets its own.

        Returns:
            sqlite3.Connection: Connection with ``Row`` factory and WAL enabled.
        """
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.path, timeout=30.0, isolation_level=None)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("PRAGMA busy_timeout=30000")
            self._local.conn = conn
        return conn

    @contextmanager
    def write(self) -> Iterator[sqlite3.Connection]:
        """
        中文：写事务上下文管理器，串行化写入并自动提交/回滚。

        Serialised write transaction.

        Yields:
            sqlite3.Connection: Connection inside an open transaction. The
                transaction is committed on clean exit and rolled back if the
                body raises.
        """
        conn = self._connect()
        with self._write_lock:
            conn.execute("BEGIN IMMEDIATE")
            try:
                yield conn
            except Exception:
                conn.execute("ROLLBACK")
                raise
            else:
                conn.execute("COMMIT")

    def query(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        """
        中文：执行只读查询并返回全部结果行。

        Run a read-only query.

        Args:
            sql (str): SQL statement.
            params (tuple): Positional bind parameters.

        Returns:
            list[sqlite3.Row]: All result rows, possibly empty.
        """
        return self._connect().execute(sql, params).fetchall()

    def query_one(self, sql: str, params: tuple = ()) -> sqlite3.Row | None:
        """
        中文：执行只读查询并返回首行。

        Run a read-only query returning at most one row.

        Args:
            sql (str): SQL statement.
            params (tuple): Positional bind parameters.

        Returns:
            sqlite3.Row | None: First row, or ``None`` if the result is empty.
        """
        return self._connect().execute(sql, params).fetchone()

    def close(self) -> None:
        """中文：关闭当前线程的连接。 / Close this thread's connection if open."""
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None

    # ------------------------------------------------------------------ #
    # Schema / 建表与迁移
    # ------------------------------------------------------------------ #

    def _init_schema(self) -> None:
        """
        中文：创建表结构并记录 schema 版本。

        Create tables if absent and stamp the schema version.

        Uses SQLite's ``user_version`` pragma as the migration marker. Future
        releases add migration steps here keyed on the stored version.
        """
        conn = self._connect()
        current = conn.execute("PRAGMA user_version").fetchone()[0]
        conn.executescript(_SCHEMA)

        # CREATE TABLE IF NOT EXISTS does nothing to a table that already
        # exists, so a column added to _SCHEMA never reaches a database made by
        # an earlier release. Each one is added here instead, guarded by what
        # the table actually has rather than by a version number — the check is
        # cheap, and it stays correct if a database skips a release.
        # CREATE TABLE IF NOT EXISTS 对已存在的表毫无作用，因此加进 _SCHEMA 的新列永远
        # 到不了旧版本建立的数据库。这里逐列补上，判断依据是表里实际有什么，而不是版本号
        # ——这项检查很便宜，而且在数据库跳过某个版本时依然正确。
        have = {row[1] for row in conn.execute("PRAGMA table_info(jobs)")}
        for column in ("source_chars", "target_chars"):
            if column not in have:
                conn.execute(f"ALTER TABLE jobs ADD COLUMN {column} INTEGER NOT NULL DEFAULT 0")
                logger.info("Added jobs.%s to the existing database", column)
        if current != SCHEMA_VERSION:
            conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
            logger.info("Database schema at version %d (%s)", SCHEMA_VERSION, self.path)

    # ------------------------------------------------------------------ #
    # Documents / 文档
    # ------------------------------------------------------------------ #

    def upsert_document(
        self,
        doc_id: str,
        *,
        source_path: Path | str,
        output_dir: Path | str,
        source_kind: str,
        title: str | None = None,
        source_ref: str | None = None,
        page_count: int | None = None,
        file_size: int | None = None,
        authors: str | None = None,
    ) -> None:
        """
        中文：插入或更新一条文档记录。

        Insert a document, or refresh its metadata if it already exists.

        Args:
            doc_id (str): Stable identifier (content hash or arXiv id).
            source_path (Path | str): Absolute path to the source PDF.
            output_dir (Path | str): Directory the translated files go to.
            source_kind (str): ``"local"``, ``"upload"`` or ``"arxiv"``.
            title (str | None): Human-readable title; falls back to the filename.
            source_ref (str | None): Original reference such as an arXiv URL.
            page_count (int | None): Number of pages in the source PDF.
            file_size (int | None): Source file size in bytes.
            authors (str | None): Comma-separated author list, when known.
        """
        now = utcnow()
        with self.write() as conn:
            conn.execute(
                """
                INSERT INTO documents
                    (id, title, source_kind, source_ref, source_path, output_dir,
                     page_count, file_size, authors, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    title       = COALESCE(excluded.title, documents.title),
                    source_path = excluded.source_path,
                    output_dir  = excluded.output_dir,
                    page_count  = COALESCE(excluded.page_count, documents.page_count),
                    file_size   = COALESCE(excluded.file_size, documents.file_size),
                    authors     = COALESCE(excluded.authors, documents.authors),
                    updated_at  = excluded.updated_at
                """,
                (
                    doc_id,
                    title,
                    source_kind,
                    source_ref,
                    str(source_path),
                    str(output_dir),
                    page_count,
                    file_size,
                    authors,
                    now,
                    now,
                ),
            )

    def get_document(self, doc_id: str) -> sqlite3.Row | None:
        """
        中文：按 id 查询文档。

        Args:
            doc_id (str): Document identifier.

        Returns:
            sqlite3.Row | None: The document row, or ``None`` if unknown.
        """
        return self.query_one("SELECT * FROM documents WHERE id = ?", (doc_id,))

    def delete_documents(self, doc_ids) -> int:
        """
        中文：从库里删掉这些文档及其任务记录，返回真正删掉的条数。

        Remove documents and everything recorded about them.

        The ``jobs`` table declares ``ON DELETE CASCADE`` on its document
        reference, but SQLite only honours that with ``PRAGMA foreign_keys``
        on — which is per connection, not per database, and easy to lose. The
        jobs are therefore deleted explicitly, so the behaviour does not depend
        on a pragma being set somewhere else.

        Args:
            doc_ids (Iterable[str]): Document identifiers.

        Returns:
            int: How many document rows were removed. A id that is not there
                counts zero rather than raising — deleting something already
                gone is not an error to the caller.
        """
        ids = [str(i) for i in doc_ids]
        if not ids:
            return 0
        marks = ",".join("?" * len(ids))
        with self.write() as conn:
            conn.execute(f"DELETE FROM jobs WHERE doc_id IN ({marks})", ids)
            cursor = conn.execute(
                f"DELETE FROM documents WHERE id IN ({marks})", ids)
        return cursor.rowcount

    def documents_titled(self, title: str) -> list[sqlite3.Row]:
        """
        中文：按标题查同名文档，连它的副本一起返回。

        Find documents whose title matches, copies included.

        Matching is case-insensitive and ignores surrounding whitespace, because
        a title read from PDF metadata and the same title read from arXiv differ
        in exactly those ways. The copy suffix 「（1）」 is matched too, so that
        submitting a paper a third time can number the copy correctly.

        Args:
            title (str): Title to look for, without a copy suffix.

        Returns:
            list[sqlite3.Row]: Matching document rows, oldest first.
        """
        pattern = title.strip()
        return self.query(
            """
            SELECT * FROM documents
            WHERE LOWER(TRIM(title)) = LOWER(?)
               OR LOWER(TRIM(title)) LIKE LOWER(?)
            ORDER BY created_at
            """,
            (pattern, f"{pattern}（%）"),
        )

    def list_documents(self, limit: int = 200, offset: int = 0) -> list[sqlite3.Row]:
        """
        中文：按更新时间倒序列出文档，并附带最近一次任务的状态。

        List documents most-recently-updated first, joined with their latest job.

        Args:
            limit (int): Maximum rows to return.
            offset (int): Rows to skip, for pagination.

        Returns:
            list[sqlite3.Row]: Document rows augmented with ``last_status``,
                ``last_job_id``, ``dual_path`` and ``mono_path`` from the most
                recent job for that document.
        """
        return self.query(
            """
            SELECT d.*,
                   j.id       AS last_job_id,
                   j.status   AS last_status,
                   j.progress AS last_progress,
                   j.dual_path,
                   j.mono_path,
                   j.total_tokens,
                   j.duration_ms
            FROM documents d
            LEFT JOIN jobs j ON j.id = (
                SELECT id FROM jobs
                WHERE doc_id = d.id
                ORDER BY created_at DESC
                LIMIT 1
            )
            ORDER BY d.updated_at DESC
            LIMIT ? OFFSET ?
            """,
            (limit, offset),
        )

    # ------------------------------------------------------------------ #
    # Jobs / 任务
    # ------------------------------------------------------------------ #

    def create_job(
        self,
        job_id: str,
        doc_id: str,
        *,
        backend: str,
        config: dict[str, Any] | None = None,
    ) -> None:
        """
        中文：创建一个排队中的翻译任务。

        Insert a new job in the ``queued`` state.

        Args:
            job_id (str): Unique job identifier.
            doc_id (str): Document this job translates.
            backend (str): ``"local"`` or ``"api"``.
            config (dict | None): Snapshot of the settings used, stored as JSON
                so a job's parameters remain inspectable after settings change.
        """
        with self.write() as conn:
            conn.execute(
                """
                INSERT INTO jobs (id, doc_id, status, backend, config_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    doc_id,
                    JOB_QUEUED,
                    backend,
                    json.dumps(config or {}, ensure_ascii=False),
                    utcnow(),
                ),
            )

    def update_job(self, job_id: str, **fields: Any) -> None:
        """
        中文：更新任务的任意字段。

        Update arbitrary columns of a job row.

        Args:
            job_id (str): Job identifier.
            **fields: Column/value pairs to write. Unknown column names raise a
                :class:`sqlite3.OperationalError`, which surfaces typos early
                rather than silently dropping updates.
        """
        if not fields:
            return
        assignments = ", ".join(f"{k} = ?" for k in fields)
        with self.write() as conn:
            conn.execute(
                f"UPDATE jobs SET {assignments} WHERE id = ?",
                (*fields.values(), job_id),
            )

    def get_job(self, job_id: str) -> sqlite3.Row | None:
        """
        中文：按 id 查询任务，并联带文档信息。

        Args:
            job_id (str): Job identifier.

        Returns:
            sqlite3.Row | None: Job row joined with the document's title and
                source path, or ``None`` if unknown.
        """
        return self.query_one(
            """
            SELECT j.*, d.title, d.source_path, d.source_kind, d.page_count
            FROM jobs j JOIN documents d ON d.id = j.doc_id
            WHERE j.id = ?
            """,
            (job_id,),
        )

    def list_jobs(
        self,
        statuses: tuple[str, ...] | None = None,
        limit: int = 100,
    ) -> list[sqlite3.Row]:
        """
        中文：列出任务，可按状态过滤。

        List jobs, newest first, optionally filtered by status.

        Args:
            statuses (tuple[str, ...] | None): Restrict to these statuses.
                ``None`` returns every job.
            limit (int): Maximum rows to return.

        Returns:
            list[sqlite3.Row]: Job rows joined with document title and path.
        """
        base = """
            SELECT j.*, d.title, d.source_path, d.source_kind, d.page_count
            FROM jobs j JOIN documents d ON d.id = j.doc_id
        """
        if statuses:
            placeholders = ", ".join("?" for _ in statuses)
            sql = f"{base} WHERE j.status IN ({placeholders}) ORDER BY j.created_at DESC LIMIT ?"
            return self.query(sql, (*statuses, limit))
        return self.query(f"{base} ORDER BY j.created_at DESC LIMIT ?", (limit,))

    def reap_stale_jobs(self) -> int:
        """
        中文：把上次退出时遗留的运行中任务标记为中断。

        Mark jobs left in an active state by a previous run as ``interrupted``.

        Called once at startup. A job cannot survive the server process that
        owned its worker, so anything still marked queued or running is stale.
        Because BabelDOC caches translations per paragraph, retrying such a job
        mostly replays from cache and finishes quickly.

        Returns:
            int: Number of jobs transitioned to ``interrupted``.
        """
        placeholders = ", ".join("?" for _ in ACTIVE_JOB_STATES)
        with self.write() as conn:
            cur = conn.execute(
                f"""
                UPDATE jobs
                SET status = ?, error = ?, finished_at = ?
                WHERE status IN ({placeholders})
                """,
                (
                    JOB_INTERRUPTED,
                    "Server restarted while this job was running",
                    utcnow(),
                    *ACTIVE_JOB_STATES,
                ),
            )
            count = cur.rowcount or 0
        if count:
            logger.info("Marked %d stale job(s) as interrupted", count)
        return count

    def usage_totals(self) -> dict[str, Any]:
        """
        中文：汇总全部已完成任务的 token 用量与费用。

        Aggregate token usage across all completed jobs, split by backend.

        Used by the settings page to show whether the growing glossary is
        actually reducing API usage over time.

        Returns:
            dict[str, Any]: Mapping of backend name to a dict with keys
                ``jobs``, ``prompt_tokens``, ``completion_tokens``,
                ``total_tokens`` and ``est_cost``.
        """
        rows = self.query(
            """
            SELECT backend,
                   COUNT(*)                    AS jobs,
                   SUM(source_chars)           AS source_chars,
                   SUM(target_chars)           AS target_chars,
                   SUM(prompt_tokens)          AS prompt_tokens,
                   SUM(completion_tokens)      AS completion_tokens,
                   SUM(total_tokens)           AS total_tokens,
                   SUM(COALESCE(est_cost, 0))  AS est_cost
            FROM jobs
            WHERE status = ?
            GROUP BY backend
            """,
            (JOB_COMPLETED,),
        )
        return {
            (r["backend"] or "unknown"): {
                "jobs": r["jobs"],
                "source_chars": r["source_chars"] or 0,
                "target_chars": r["target_chars"] or 0,
                "prompt_tokens": r["prompt_tokens"] or 0,
                "completion_tokens": r["completion_tokens"] or 0,
                "total_tokens": r["total_tokens"] or 0,
                "est_cost": r["est_cost"] or 0.0,
            }
            for r in rows
        }


_db: Database | None = None


def get_db(path: Path | None = None) -> Database:
    """
    中文：返回进程内共享的 Database 单例。

    Return the process-wide database handle, creating it on first use.

    Args:
        path (Path | None): Database location. Defaults to the configured
            application data directory. Ignored once the singleton exists.

    Returns:
        Database: Shared database instance.
    """
    global _db
    if _db is None:
        if path is None:
            from local_translator.config import get_paths

            path = get_paths().db_path
        _db = Database(path)
    return _db
