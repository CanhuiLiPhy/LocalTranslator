"""
中文：词典存储——ECDICT 数据的导入、词形还原与查询。

Dictionary storage, import and lookup.

Kept in its own SQLite file rather than the application database. It is bulk
reference data: written once, read constantly, and worth roughly 200 MB. Mixing
it into ``state.db`` would put it in the same file as job history and the
glossary, where it would dominate every backup and could not be discarded
without taking the rest with it.

Lookup resolves inflections. A reader who selects "running" in a PDF wants the
entry for "run", and a dictionary that answers "not found" for every inflected
form is useless on real text. Two sources of morphology are used together: the
lemma list, which maps forms to their base, and each entry's own ``exchange``
field, which records the inflections a word produces.
"""

from __future__ import annotations

import csv
import logging
import re
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS entries (
  word        TEXT PRIMARY KEY,
  phonetic    TEXT,
  definition  TEXT,
  translation TEXT,
  pos         TEXT,
  collins     INTEGER,
  oxford      INTEGER,
  tag         TEXT,
  bnc         INTEGER,
  frq         INTEGER,
  exchange    TEXT
);

CREATE TABLE IF NOT EXISTS lemma (
  form TEXT PRIMARY KEY,
  base TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS meta (
  key   TEXT PRIMARY KEY,
  value TEXT
);

CREATE TABLE IF NOT EXISTS zh_index (
  term TEXT NOT NULL,
  word TEXT NOT NULL,
  prio INTEGER NOT NULL,
  PRIMARY KEY (term, word)
) WITHOUT ROWID;
"""

# Leading part-of-speech and register markers on a Chinese gloss line:
# "n. 注意力", "vt. 使...", "[网络] 兜帽", "1. 罩".
# 中文释义行开头的词性与语域标记。
_GLOSS_PREFIX = re.compile(r"^(?:\s*(?:\[[^\]]{1,12}\]|[a-zA-Z]{1,6}\.|\d{1,2}[\.、]))+\s*")

# Separators between senses inside one line.
# 一行内各义项之间的分隔符。
_GLOSS_SPLIT = re.compile(r"[;；,，、]")

# Parenthetical qualifiers, dropped so "罩（布质）" indexes as "罩".
# 括注一律去掉，使「罩（布质）」以「罩」入索引。
_GLOSS_NOTE = re.compile(r"[（(\[【][^）)\]】]{0,20}[）)\]】]")

_CJK = re.compile(r"[\u4e00-\u9fff]")

# Glosses longer than this are explanations rather than words, and indexing them
# only adds rows that no one will ever type.
# 超过该长度的释义是解释而非词，收进索引只会徒增没人会输入的行。
_MAX_GLOSS = 14

# Inflection codes in ECDICT's `exchange` field, e.g. "p:ran/d:run/i:running".
# Only the ones that map a form back to a base are useful here.
# ECDICT `exchange` 字段的词形代码，例如 "p:ran/d:run/i:running"。
# 这里只用得上能把变形映射回原形的那些。
_EXCHANGE = re.compile(r"(?:^|/)0:([^/]+)")

# Suffix rules applied only after the curated data has failed. The lemma list
# was compiled from a general corpus and does not cover recent technical
# vocabulary — "embeddings" and "tokenizers" are both absent — which is exactly
# the vocabulary this reader meets. Each entry is (suffix, replacements).
# 仅在词表数据查不到时才使用的后缀规则。词形表基于通用语料编制，未覆盖较新的技术
# 词汇——"embeddings" 与 "tokenizers" 都不在其中——而这恰是本工具读者会遇到的词。
# 每项为（后缀，候选还原）。
_SUFFIXES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("ies", ("y",)),
    ("ves", ("f", "fe")),
    ("ses", ("s", "se")),
    ("xes", ("x",)),
    ("zes", ("z", "ze")),
    ("ches", ("ch",)),
    ("shes", ("sh",)),
    ("es", ("", "e")),
    ("s", ("",)),
    ("ing", ("", "e")),
    ("ed", ("", "e")),
    ("er", ("", "e")),
    ("est", ("", "e")),
    ("ly", ("",)),
)


@dataclass
class Entry:
    """
    中文：一条词典释义。

    One dictionary entry.

    Attributes:
        word (str): Headword as stored.
        phonetic (str): IPA pronunciation, empty when unknown.
        translation (str): Chinese definition, newline-separated by sense.
        definition (str): English definition, newline-separated by sense.
        pos (str): Part-of-speech distribution, e.g. ``n:52/v:48``.
        collins (int): Collins star rating, 0–5; higher is more common.
        oxford (int): 1 when the word is in the Oxford 3000 core list.
        tag (str): Exam vocabulary tags such as ``cet4`` or ``ielts``.
        frq (int): Contemporary corpus frequency rank; 0 when unranked.
        exchange (str): Inflections this word produces.
        matched_via (str): How the query reached this entry — ``exact``,
            ``lemma`` or ``exchange`` — so the UI can say "showing *run* for
            *running*" instead of silently answering a different question.
    """

    word: str
    phonetic: str = ""
    translation: str = ""
    definition: str = ""
    pos: str = ""
    collins: int = 0
    oxford: int = 0
    tag: str = ""
    frq: int = 0
    exchange: str = ""
    matched_via: str = "exact"


@dataclass
class ReverseHit:
    """
    中文：一条汉英查询结果。

    One Chinese-to-English match.

    Attributes:
        term (str): The Chinese sense that matched.
        word (str): English headword.
        phonetic (str): IPA, without slashes; may be empty.
        translation (str): The headword's full Chinese sense block.
        frq (int): Contemporary-corpus frequency rank; 0 when unranked.
        collins (int): Collins star rating, 0–5.
        oxford (int): 1 when the word is in the Oxford core list.
        prio (int): Position of the matching sense within the entry, 0 first.
        match (str): ``exact``, ``prefix`` or ``contains``.
    """

    term: str
    word: str
    phonetic: str
    translation: str
    frq: int
    collins: int
    oxford: int
    prio: int
    match: str


def _glosses(translation: str | None) -> list[str]:
    """
    中文：把一条词条的中文释义块拆成可索引的义项列表。

    Split an entry's Chinese sense block into indexable terms.

    Args:
        translation (str | None): Raw ``translation`` field, in which ECDICT
            writes line breaks as the two characters backslash-n.

    Returns:
        list[str]: Distinct Chinese senses in the order they appear, each at
            most ``_MAX_GLOSS`` characters and containing at least one Chinese
            character. Latin-only glosses are dropped: they are cross-references
            such as "= between", useless as a Chinese search key.
    """
    if not translation:
        return []

    text = translation.replace("\\n", "\n")
    out: list[str] = []
    seen: set[str] = set()
    for line in text.split("\n"):
        line = _GLOSS_NOTE.sub("", _GLOSS_PREFIX.sub("", line))
        for piece in _GLOSS_SPLIT.split(line):
            term = piece.strip().strip("。.·-—…“”\"' \t")
            if not term or len(term) > _MAX_GLOSS or term in seen:
                continue
            if not _CJK.search(term):
                continue
            seen.add(term)
            out.append(term)
    return out


class DictionaryStore:
    """
    中文：ECDICT 词库的本地存储与查询。

    Local store for the ECDICT dataset.

    Attributes:
        path (Path): SQLite file holding the dictionary.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        """
        中文：返回当前线程的连接。

        Return this thread's connection, opening it on first use.

        Returns:
            sqlite3.Connection: Connection with ``Row`` factory.
        """
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.path, timeout=30.0, isolation_level=None)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA busy_timeout=30000")
            self._local.conn = conn
        return conn

    def _init_schema(self) -> None:
        """中文：建表。 / Create tables if absent."""
        self._connect().executescript(SCHEMA)

    # ------------------------------------------------------------------ #
    # State / 状态
    # ------------------------------------------------------------------ #

    def count(self) -> int:
        """
        中文：返回词条总数。

        Count stored entries.

        Returns:
            int: Number of headwords, zero when the dictionary is not installed.
        """
        try:
            row = self._connect().execute("SELECT COUNT(*) AS n FROM entries").fetchone()
        except sqlite3.Error:
            return 0
        return int(row["n"]) if row else 0

    def installed(self) -> bool:
        """
        中文：词库是否已导入。

        Report whether the dictionary is usable.

        Returns:
            bool: ``True`` once entries have been imported.
        """
        return self.count() > 0

    # ------------------------------------------------------------------ #
    # Import / 导入
    # ------------------------------------------------------------------ #

    def import_csv(self, path: Path, progress=None, batch: int = 20000) -> int:
        """
        中文：导入 ECDICT CSV，按批提交。

        Import the ECDICT CSV.

        Rows are inserted in batches inside a single transaction. Committing per
        row would take hours for 770,000 entries; committing once at the end
        would hold the whole dataset in the journal.

        Args:
            path (Path): ``ecdict.csv``.
            progress (Callable[[int], None] | None): Called with the running row
                count, for reporting.
            batch (int): Rows per executemany call.

        Returns:
            int: Number of entries imported.
        """
        conn = self._connect()
        total = 0
        rows: list[tuple] = []

        # The dataset has entries with embedded newlines and very long fields.
        # CSV 中存在含换行的字段，且个别字段很长。
        csv.field_size_limit(10_000_000)

        def flush() -> None:
            if not rows:
                return
            conn.executemany(
                "INSERT OR REPLACE INTO entries "
                "(word, phonetic, definition, translation, pos, collins, oxford, "
                " tag, bnc, frq, exchange) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                rows,
            )
            rows.clear()

        def number(value: str) -> int:
            try:
                return int(value)
            except (TypeError, ValueError):
                return 0

        conn.execute("BEGIN")
        try:
            with path.open("r", encoding="utf-8", newline="") as handle:
                for record in csv.DictReader(handle):
                    word = (record.get("word") or "").strip()
                    if not word:
                        continue
                    rows.append((
                        word,
                        (record.get("phonetic") or "").strip(),
                        (record.get("definition") or "").strip(),
                        (record.get("translation") or "").strip(),
                        (record.get("pos") or "").strip(),
                        number(record.get("collins")),
                        number(record.get("oxford")),
                        (record.get("tag") or "").strip(),
                        number(record.get("bnc")),
                        number(record.get("frq")),
                        (record.get("exchange") or "").strip(),
                    ))
                    total += 1
                    if len(rows) >= batch:
                        flush()
                        if progress:
                            progress(total)
            flush()
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise

        # Built after the bulk load: maintaining an index during insertion costs
        # far more than building it once at the end.
        # 批量导入完成后再建索引：边插边维护索引的代价远高于最后一次性建立。
        conn.execute("CREATE INDEX IF NOT EXISTS idx_entries_lower "
                     "ON entries(word COLLATE NOCASE)")
        conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES ('entries', ?)",
                     (str(total),))
        if progress:
            progress(total)
        logger.info("Imported %d dictionary entries", total)
        return total

    def import_lemma(self, path: Path) -> int:
        """
        中文：导入词形还原表（form → base）。

        Import the lemma list mapping inflected forms to their base word.

        Args:
            path (Path): ``lemma.en.txt``.

        Returns:
            int: Number of form-to-base mappings stored.
        """
        conn = self._connect()
        pairs: list[tuple[str, str]] = []

        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line or line.startswith(";") or "->" not in line:
                    continue
                head, forms = line.split("->", 1)
                # The head carries a frequency after a slash: "be/4109826".
                # 词头后带频次，以斜杠分隔："be/4109826"。
                base = head.split("/", 1)[0].strip().lower()
                if not base:
                    continue
                for form in forms.split(","):
                    form = form.strip().lower()
                    if form and form != base:
                        pairs.append((form, base))

        conn.execute("BEGIN")
        try:
            conn.executemany("INSERT OR REPLACE INTO lemma(form, base) VALUES (?,?)", pairs)
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
        logger.info("Imported %d lemma mappings", len(pairs))
        return len(pairs)

    # ------------------------------------------------------------------ #
    # Lookup / 查询
    # ------------------------------------------------------------------ #

    def _row(self, word: str) -> sqlite3.Row | None:
        """
        中文：按词形取一行，大小写不敏感。

        Fetch one row by headword, case-insensitively.

        Args:
            word (str): Headword to find.

        Returns:
            sqlite3.Row | None: The row, or ``None``.
        """
        return self._connect().execute(
            "SELECT * FROM entries WHERE word = ? COLLATE NOCASE LIMIT 1", (word,)
        ).fetchone()

    def lookup(self, query: str) -> Entry | None:
        """
        中文：查词；找不到时尝试词形还原。

        Look a word up, falling back to its base form.

        Three attempts are made in order of confidence: the word as typed, the
        lemma list's base form, and the base recorded in an inflected entry's
        own ``exchange`` field. The last catches forms the lemma list omits,
        since the two datasets were compiled independently.

        Args:
            query (str): Word to look up.

        Returns:
            Entry | None: The entry, with ``matched_via`` recording which
                attempt succeeded, or ``None`` when nothing matched.
        """
        word = (query or "").strip()
        if not word:
            return None

        row = self._row(word)
        if row and (row["translation"] or row["definition"]):
            return self._entry(row, "exact")

        lower = word.lower()
        base = self._connect().execute(
            "SELECT base FROM lemma WHERE form = ? LIMIT 1", (lower,)
        ).fetchone()
        if base:
            found = self._row(base["base"])
            if found:
                return self._entry(found, "lemma")

        # An inflected entry often exists but carries no translation of its own;
        # its `exchange` field points at the lemma.
        # 变形词往往有词条但没有自己的释义，其 `exchange` 字段指向原形。
        if row and row["exchange"]:
            match = _EXCHANGE.search(row["exchange"])
            if match:
                found = self._row(match.group(1))
                if found:
                    return self._entry(found, "exchange")

        if row:
            return self._entry(row, "exact")

        stripped = self._by_suffix(lower)
        if stripped:
            return stripped

        return None

    def _by_suffix(self, word: str) -> Entry | None:
        """
        中文：按后缀规则还原后再查，作为最后手段。

        Try suffix stripping as a last resort.

        Only reached once the lemma list and the exchange field have both
        failed, so a guessed stem can never override curated morphology. A
        candidate is accepted only if it is itself a headword carrying a
        definition, which rejects most bad guesses: "bus" does not survive
        being read as a plural because "bu" is not a word.

        Args:
            word (str): Lower-cased query.

        Returns:
            Entry | None: The entry for a recovered stem, or ``None``.
        """
        for suffix, replacements in _SUFFIXES:
            if not word.endswith(suffix) or len(word) - len(suffix) < 3:
                continue
            stem = word[: -len(suffix)]
            candidates = [stem + r for r in replacements]
            # A doubled final consonant marks the inflection: running -> run.
            # 末尾辅音重复标记变形：running -> run。
            if len(stem) >= 4 and stem[-1] == stem[-2] and stem[-1].isalpha():
                candidates.append(stem[:-1])
            for candidate in candidates:
                found = self._row(candidate)
                if found and (found["translation"] or found["definition"]):
                    return self._entry(found, "suffix")
        return None

    @staticmethod
    def _unescape(value: str | None) -> str:
        r"""
        中文：把 ECDICT 中字面的 \n 还原为真正的换行。

        Turn ECDICT's literal backslash-n sequences into real newlines.

        The dataset separates senses with a backslash followed by the letter n
        rather than a newline, since its CSV keeps every entry on one line.
        Rendered as-is those two characters appear inside the definition.

        Args:
            value (str | None): Raw field text.

        Returns:
            str: Text with sense breaks as newlines.
        """
        return (value or "").replace("\\n", "\n").replace("\\r", "")

    @classmethod
    def _entry(cls, row: sqlite3.Row, via: str) -> Entry:
        """
        中文：把数据库行转成 Entry。

        Convert a row into an :class:`Entry`.

        Args:
            row (sqlite3.Row): Database row.
            via (str): How the lookup reached it.

        Returns:
            Entry: The populated entry.
        """
        return Entry(
            word=row["word"],
            phonetic=row["phonetic"] or "",
            translation=cls._unescape(row["translation"]),
            definition=cls._unescape(row["definition"]),
            pos=row["pos"] or "",
            collins=row["collins"] or 0,
            oxford=row["oxford"] or 0,
            tag=row["tag"] or "",
            frq=row["frq"] or 0,
            exchange=row["exchange"] or "",
            matched_via=via,
        )

    # ------------------------------------------------------------------ #
    # Chinese → English / 汉英
    # ------------------------------------------------------------------ #

    def reverse_count(self) -> int:
        """
        中文：返回汉英索引的条目数，未建立时为 0。

        Returns:
            int: Number of rows in the Chinese-to-English index.
        """
        try:
            row = self._connect().execute("SELECT COUNT(*) FROM zh_index").fetchone()
        except sqlite3.Error:
            return 0
        return int(row[0]) if row else 0

    def reverse_ready(self) -> bool:
        """
        中文：汉英索引是否已建立。

        Returns:
            bool: ``True`` once the reverse index holds entries.
        """
        return self.reverse_count() > 0

    def build_reverse_index(self, progress=None, batch: int = 50000) -> int:
        """
        中文：从已导入的词条反向建立汉英索引。

        Build the Chinese-to-English index from the entries already imported.

        ECDICT is an English-headword dictionary: the Chinese only exists inside
        each entry's ``translation`` field, as a block of sense lines. Searching
        that field directly is not an option — a ``LIKE '%注意力%'`` scan reads
        every one of 770,000 rows — and SQLite's trigram full-text tokenizer
        cannot help either, since it needs at least three characters and a large
        share of Chinese words are two.

        So the senses are split out once and stored as their own rows. Each line
        loses its part-of-speech marker and any parenthetical qualifier, then
        splits on the separators ECDICT uses between senses; whatever survives
        that and still contains a Chinese character is indexed against the
        headword. The position of the sense within the entry is kept, so a word
        whose *first* sense is the query outranks one that lists it eighth.

        Runs entirely off the local database, which matters for anyone who has
        already installed the dictionary: adding reverse lookup must not mean
        downloading the 66 MB source again.

        Args:
            progress (Callable[[int], None] | None): Called with the number of
                entries processed so far.
            batch (int): Rows buffered before each insert.

        Returns:
            int: Number of index rows written.
        """
        conn = self._connect()
        conn.execute("DELETE FROM zh_index")

        rows: list[tuple[str, str, int]] = []
        total = 0
        seen = 0

        def flush() -> None:
            nonlocal total
            if not rows:
                return
            conn.executemany(
                "INSERT OR IGNORE INTO zh_index(term, word, prio) VALUES (?, ?, ?)", rows
            )
            total += len(rows)
            rows.clear()

        cursor = conn.execute(
            "SELECT word, translation FROM entries "
            "WHERE translation IS NOT NULL AND translation != ''"
        )
        conn.execute("BEGIN")
        try:
            for word, translation in cursor:
                seen += 1
                for prio, term in enumerate(_glosses(translation)):
                    rows.append((term, word, prio))
                if len(rows) >= batch:
                    flush()
                    if progress:
                        progress(seen)
            flush()
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise

        conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES ('zh_index', ?)",
                     (str(total),))
        if progress:
            progress(seen)
        logger.info("Built reverse index with %d term rows from %d entries", total, seen)
        return total

    def lookup_zh(self, query: str, limit: int = 24) -> list[ReverseHit]:
        """
        中文：按中文查英文，先精确义项、再前缀、最后包含。

        Look up English words by a Chinese term.

        Matching runs in three passes, widening only when the previous one has
        not filled the quota: an exact sense match, then senses starting with
        the query, then senses containing it. The order is what keeps 注意力 from
        being buried under 注意力不集中 and 注意力缺陷障碍, which a single
        substring search would rank alongside it.

        Args:
            query (str): Chinese word or phrase.
            limit (int): Maximum results.

        Returns:
            list[ReverseHit]: Matches, best first, one per English headword.
        """
        term = (query or "").strip()
        if not term:
            return []

        conn = self._connect()
        # Commonness first, sense position only as a tiebreak. Ranking by
        # position instead buries the word the reader wants: "attention" lists
        # 注意 before 注意力, so it scores second on position and loses to
        # "attentiveness", whose only sense is 注意力 — an answer that is
        # technically a match and practically never the one being looked for.
        # 先看词的常用度，义项位置只作次级依据。反过来按位置排会把想查的词埋掉：
        # attention 的义项是「注意」在前、「注意力」在后，按位置只能排第二，输给
        # 唯一义项就是「注意力」的 attentiveness——匹配没错，但基本不是要找的词。
        order = ("ORDER BY CASE WHEN e.frq > 0 THEN e.frq ELSE 999999 END, "
                 "e.collins DESC, e.oxford DESC, z.prio, "
                 "LENGTH(z.term), LENGTH(e.word)")
        select = ("SELECT z.term, z.prio, e.word, e.phonetic, e.translation, "
                  "e.frq, e.collins, e.oxford FROM zh_index z "
                  "JOIN entries e ON e.word = z.word WHERE ")

        passes = (
            ("z.term = ?", (term,), "exact"),
            ("z.term LIKE ? AND z.term != ?", (term + "%", term), "prefix"),
            ("z.term LIKE ? AND z.term NOT LIKE ?", ("%" + term + "%", term + "%"),
             "contains"),
        )

        out: list[ReverseHit] = []
        found: set[str] = set()
        for where, params, kind in passes:
            if len(out) >= limit:
                break
            sql = f"{select}{where} {order} LIMIT ?"
            for row in conn.execute(sql, (*params, limit * 3)):
                if row["word"] in found:
                    continue
                found.add(row["word"])
                out.append(ReverseHit(
                    term=row["term"],
                    word=row["word"],
                    phonetic=self._unescape(row["phonetic"]),
                    translation=self._unescape(row["translation"]),
                    frq=int(row["frq"] or 0),
                    collins=int(row["collins"] or 0),
                    oxford=int(row["oxford"] or 0),
                    prio=int(row["prio"] or 0),
                    match=kind,
                ))
                if len(out) >= limit:
                    break
        return out

    def suggest(self, prefix: str, limit: int = 12) -> list[str]:
        """
        中文：按前缀返回候选词，常用词优先。

        Suggest headwords starting with a prefix.

        Ordered by corpus frequency so that "com" offers *come* before
        *comburent*. Unranked words sort last rather than first, which is what a
        plain ``ORDER BY frq`` would do given that unranked is stored as zero.

        Args:
            prefix (str): Leading characters.
            limit (int): Maximum suggestions.

        Returns:
            list[str]: Matching headwords.
        """
        prefix = (prefix or "").strip()
        if not prefix:
            return []
        rows = self._connect().execute(
            "SELECT word FROM entries WHERE word LIKE ? COLLATE NOCASE "
            "ORDER BY CASE WHEN frq > 0 THEN frq ELSE 999999 END, LENGTH(word) LIMIT ?",
            (prefix.replace("%", "") + "%", limit),
        ).fetchall()
        return [row["word"] for row in rows]


_STORE: DictionaryStore | None = None
_LOCK = threading.Lock()


def get_dictionary(path: Path | None = None) -> DictionaryStore:
    """
    中文：返回进程内共享的词典实例。

    Return the process-wide dictionary store.

    Args:
        path (Path | None): Override for the SQLite file; defaults to
            ``dictionary.db`` beside the application database.

    Returns:
        DictionaryStore: Shared instance.
    """
    global _STORE
    with _LOCK:
        if _STORE is None:
            if path is None:
                from local_translator.config import get_paths

                path = get_paths().data_dir / "dictionary.db"
            _STORE = DictionaryStore(path)
        return _STORE
