"""
中文：清理翻译过程中产生的中间文件。

Remove what a translation leaves behind.

**How much there is.** BabelDOC works in a directory of its own, and it works by
writing the document out again at each stage: the input copied in, the fonts
subsetted, the table of contents rebuilt, the mono and dual versions before
they are published. On this machine, sixty documents had left 4.1 GB behind,
the largest single one 705 MB — for a paper whose two output PDFs come to 12 MB.
None of it is read again once the job ends.

The LaTeX path leaves a second pile: the unpacked source tree, the images it
rasterised, and the ``.aux``/``.bbl``/``.log`` files of two full compiles.

**What is kept.** ``translate_tracking.json`` — BabelDOC's paragraph-by-paragraph
record of what went in and what came out. It is small, the job row points at it,
and it is the only artefact here that answers a question the outputs cannot
("what did it do with this sentence?").

**When it runs.** At the end of every job, in the worker process, after the
outputs have been moved to their destination and before the process exits. Doing
it there rather than in the parent is what makes it safe: BabelDOC is finished
by then and nothing still holds a handle. A reader who would rather keep the
intermediates can say so in the settings, and then this does nothing.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

# Files worth keeping out of a working directory. Everything else in there is
# an intermediate form of a document that has already been published.
# 工作目录里值得留下的文件。其余都是某份已经发布的文档的中间形态。
KEEP = ("translate_tracking.json",)


def size_of(path: str | Path) -> int:
    """
    中文：统计一个目录占用的字节数。

    Measure how much a directory holds.

    Args:
        path (str | Path): Directory to measure. A path that does not exist
            measures zero rather than raising, because "nothing there" and
            "nothing to clean" are the same answer to the caller.

    Returns:
        int: Total size in bytes of every regular file beneath it. Symbolic
            links are counted as the links they are, not as their targets, so
            nothing outside the directory is attributed to it.
    """
    root = Path(path)
    if not root.exists():
        return 0
    total = 0
    for item in root.rglob("*"):
        try:
            if item.is_file() and not item.is_symlink():
                total += item.stat().st_size
        except OSError:
            # A file that vanished between the walk and the stat: a running job
            # writing into the same tree. Not an error, just one file's worth of
            # imprecision in a number that is shown rounded to megabytes anyway.
            # 在遍历与 stat 之间消失的文件：同一棵树里还有任务在写。这不算错误，只是让
            # 一个本来就按兆字节取整显示的数字少算了一个文件。
            continue
    return total


def sweep(working_dir: str | Path, *, keep: tuple[str, ...] = KEEP) -> int:
    """
    中文：清空一个任务的工作目录，只留下值得保留的少数文件，返回释放的字节数。

    Empty one job's working directory, keeping only what is worth keeping.

    Args:
        working_dir (str | Path): The directory BabelDOC was given. A path that
            does not exist is not an error.
        keep (tuple[str, ...]): File names to leave in place, matched on the
            name alone at any depth.

    Returns:
        int: Bytes freed. Zero when there was nothing to remove, which is also
            what a directory that could not be read returns — this runs on the
            way out of a job that has already succeeded, and failing to tidy up
            is not a reason to report that job as failed.
    """
    root = Path(working_dir)
    if not root.is_dir():
        return 0

    freed = 0
    try:
        for item in sorted(root.rglob("*"), key=lambda p: len(p.parts), reverse=True):
            try:
                if item.is_symlink() or item.is_file():
                    if item.name in keep:
                        continue
                    freed += item.stat().st_size if not item.is_symlink() else 0
                    item.unlink(missing_ok=True)
                elif item.is_dir():
                    # Deepest first, so a directory is only reached once its
                    # contents are gone; one that still holds a kept file
                    # refuses to go and should.
                    # 从最深处开始，因此轮到某个目录时其内容已清空；仍存有保留文件的目录
                    # 会拒绝删除，这正是应有的行为。
                    item.rmdir()
            except OSError:
                continue
    except OSError:
        logger.debug("Could not walk %s while cleaning up", root, exc_info=True)
        return freed

    if freed:
        logger.info("Cleaned up %.0f MB of intermediates in %s", freed / 1048576, root.name)
    return freed


def sweep_latex(source_dir: str | Path) -> int:
    """
    中文：删除一篇论文解包出来的 LaTeX 源码树，返回释放的字节数。

    Remove a paper's unpacked LaTeX source tree.

    Unlike a BabelDOC working directory, nothing here is worth keeping: the
    archive can be fetched again from arXiv, and the compiled PDFs have already
    been published to the output directory.

    Args:
        source_dir (str | Path): The tree that was unpacked for one paper —
            the workspace directory, not the ``source`` subdirectory inside it,
            so the downloaded archive goes too.

    Returns:
        int: Bytes freed, or zero when there was nothing there.
    """
    root = Path(source_dir)
    if not root.is_dir():
        return 0
    freed = size_of(root)
    try:
        shutil.rmtree(root)
    except OSError:
        logger.debug("Could not remove the LaTeX workspace at %s", root, exc_info=True)
        return 0
    logger.info("Removed %.0f MB of LaTeX sources in %s", freed / 1048576, root.name)
    return freed


def workspace_of(latex_source: str | Path) -> Path | None:
    """
    中文：从源码根目录反推出它所属的 _latex 工作区目录。

    Find the workspace a LaTeX source root was unpacked into.

    A job carries the *root* of the source tree — the directory holding the main
    ``.tex`` file, which may sit several levels inside the archive. What has to
    be removed is the workspace above it, so that the downloaded archive goes
    with the files unpacked from it.

    Args:
        latex_source (str | Path): The source root a job was given.

    Returns:
        Path | None: The workspace directory, or ``None`` when the path does not
            lie under a ``_latex`` directory — which means it is a tree the
            reader supplied themselves, and removing it is not this program's
            business.
    """
    try:
        path = Path(latex_source).resolve()
    except OSError:
        return None
    for parent in (path, *path.parents):
        if parent.parent.name == "_latex":
            return parent
    return None


def sweep_all(meta_dir: str | Path, latex_dir: str | Path | None = None,
              *, keep: tuple[str, ...] = KEEP,
              skip: frozenset[str] | set[str] = frozenset()) -> tuple[int, int]:
    """
    中文：清理全部历史中间产物，返回释放的字节数与处理过的目录数。

    Clean up every job's leftovers at once.

    This is the manual sweep behind the settings button, for the backlog that
    accumulated before automatic cleanup existed, or after a reader turns it
    back on.

    Args:
        meta_dir (str | Path): The directory holding one working directory per
            document.
        latex_dir (str | Path | None): The directory holding one unpacked source
            tree per paper, if the LaTeX path has been used.
        keep (tuple[str, ...]): File names to leave in place.
        skip (frozenset[str] | set[str]): Directory names to leave alone,
            matched against the immediate child name — the document id for a
            working directory, the arXiv id for a source tree. This is how a job
            that is running right now keeps its files: deleting a document out
            from under a translation in progress would fail that translation.

    Returns:
        tuple[int, int]: Bytes freed, and how many document directories were
            touched.
    """
    freed = 0
    touched = 0

    root = Path(meta_dir)
    if root.is_dir():
        for item in sorted(root.iterdir()):
            if not item.is_dir() or item.name in skip:
                continue
            gone = sweep(item, keep=keep)
            if gone:
                touched += 1
            freed += gone

    if latex_dir:
        tree = Path(latex_dir)
        if tree.is_dir():
            for item in sorted(tree.iterdir()):
                if not item.is_dir() or item.name in skip:
                    continue
                gone = sweep_latex(item)
                if gone:
                    touched += 1
                freed += gone

    logger.info("Manual sweep freed %.0f MB across %d directories",
                freed / 1048576, touched)
    return freed, touched


# ---------------------------------------------------------------------- #
# Stores that grow with use rather than per job / 随使用而增长的存储
# ---------------------------------------------------------------------- #

# How much of llama-server's own output to keep. It is the log of a subprocess
# this program only supervises, appended to on every start and never rotated;
# 17 MB had accumulated on the machine this was written on. What it is for is
# answering "did the model actually receive a request", and the last few
# megabytes answer that as well as all of it.
# 保留多少 llama-server 自己的输出。那是本程序只做监督的一个子进程的日志，每次启动都
# 追加、从不轮转；写这段话的机器上已经堆了 17 MB。它的用途是回答「模型到底有没有收到
# 请求」，而最后几兆和全部一样能回答。
LOG_LIMIT = 8 * 1024 * 1024

# How much of BabelDOC's paragraph cache to keep.
#
# Closing the application means the translating is done, so most of this cache
# has served its purpose. What is worth carrying over is the handful of papers
# still being worked on — the ones where a setting gets changed and the paper is
# run again, which is exactly the case the cache makes nearly free.
#
# Sized from measurement rather than guessed: across the 68 documents whose
# paragraph records are on this machine, one document produces a median of 689
# cache entries and a 90th-percentile of 1,170, at 2,209 bytes an entry once the
# table's index is counted — so a median paper is 1.5 MB and a long one 2.5 MB.
# Five median papers is 7.5 MB, five long ones 12.5 MB; 10 MB sits between them,
# which is about four long papers or six ordinary ones.
#
# Not cleared outright, only capped: emptying it on exit would send the next run
# back to the API for paragraphs it had already paid for.
# BabelDOC 段落缓存保留多少。
#
# 关掉程序意味着这一轮翻译结束了，缓存里绝大部分已经完成使命。值得留下的是手头还在改的
# 那几篇——改一处设置再跑一遍同一篇，正是缓存让它几乎不花钱的场景。
#
# 这个数是实测来的，不是拍的：本机留有逐段记录的 68 篇文档里，一篇产生的缓存条数中位为
# 689、90 分位为 1,170，而算上表的索引后每条 2,209 字节——于是中位一篇 1.5 MB，长一点的
# 一篇 2.5 MB。五篇中位是 7.5 MB，五篇长文是 12.5 MB，取中间的 10 MB——约合四篇长文或
# 六篇普通文章。
#
# 只封顶不清空：退出时倒空它，等于让下一趟回头去为已经付过钱的段落再付一次。
#: Cap for a released build, where the cache has almost nothing to do. Measured
#: across 33 papers: between *different* papers the cache answers 2.68% of
#: blocks by count, 0.07% by character, and exactly zero of the 4,229 blocks of
#: real prose — every hit is a section heading, median nine characters long.
#: Its one real use is re-translating the same paper after changing a setting,
#: where it answers 78.9%. A reader translates a paper once; a developer runs it
#: again all day. Hence the two numbers.
#: 发布版的上限——那里缓存几乎无事可做。33 篇实测：**不同**论文之间，缓存按段落数命中
#: 2.68%、按字符数 0.07%，而 4,229 段实质正文里命中 0 段——每一次命中都是章节标题，中位
#: 长度九个字符。它唯一真正的用处是「改完设置再译同一篇」，那里命中 78.9%。读者一篇只译
#: 一次，开发者一天要重跑很多次。所以是两个数。
RELEASE_CACHE_LIMIT = 0

#: Cap on a development checkout, where re-running the same paper is the whole
#: working day.
#: 开发检出上的上限——那里「重跑同一篇」就是一整天的工作方式。
DEVELOPMENT_CACHE_LIMIT = 50 * 1024 * 1024

CACHE_LIMIT = DEVELOPMENT_CACHE_LIMIT


def cap_log(path: str | Path, limit: int = LOG_LIMIT) -> int:
    """
    中文：日志超过上限就留下末尾一段，返回释放的字节数。

    Trim a log file that has no rotation of its own.

    The tail is kept rather than the head: a log is read to find out what just
    happened.

    Args:
        path (str | Path): The log file.
        limit (int): Size above which it is trimmed, in bytes. The file is left
            holding roughly half of this.

    Returns:
        int: Bytes freed, or zero when the file was already small enough.
    """
    log = Path(path)
    try:
        if not log.is_file() or log.stat().st_size <= limit:
            return 0
        before = log.stat().st_size
        keep = limit // 2
        with log.open("rb") as handle:
            handle.seek(-keep, 2)
            handle.readline()          # drop the partial line at the cut
            tail = handle.read()
        log.write_bytes(b"=== earlier entries trimmed ===\n" + tail)
        freed = before - log.stat().st_size
    except OSError:
        logger.debug("Could not trim %s", log, exc_info=True)
        return 0
    logger.info("Trimmed %s by %.0f MB", log.name, freed / 1048576)
    return freed


def cap_translation_cache(path: str | Path, limit: int = CACHE_LIMIT) -> int:
    """
    中文：把 BabelDOC 的段落缓存压到上限以内，从最旧的删起，返回释放的字节数。

    Cap BabelDOC's paragraph cache, evicting the oldest entries.

    The table carries no timestamp — its columns are the engine, its parameters,
    the source text and the translation — so ``rowid`` order stands in for
    recency. It is a fair stand-in: the primary key is an autoincrementing
    integer and the unique constraint is ``ON CONFLICT REPLACE``, so an entry
    that is written again is written with a new, higher id.

    Args:
        path (str | Path): The cache database.
        limit (int): Size to bring it under, in bytes.

    Returns:
        int: Bytes freed, or zero when it was already under the limit or could
            not be opened — another process may hold it, and failing to tidy up
            is not worth reporting as an error.
    """
    import sqlite3

    database = Path(path)
    try:
        if not database.is_file() or database.stat().st_size <= limit:
            return 0
        before = database.stat().st_size
        connection = sqlite3.connect(database, timeout=5.0)
        try:
            total = connection.execute(
                "SELECT COUNT(*) FROM _translationcache").fetchone()[0]
            if not total:
                return 0
            # Proportional, then repeated: the first estimate assumes every row
            # is the same size and they are not, so one pass lands near the
            # limit rather than under it. Three passes is enough for any ratio
            # that occurs here, and the loop stops as soon as it is under.
            # 先按比例估，再重复：第一次估算假设每行一样大，而它们并不一样，所以一趟
            # 只能落到限值附近而不是限值以内。三趟足以覆盖这里会出现的任何比例，一旦
            # 进入限内就停。
            for _ in range(3):
                current = database.stat().st_size
                if current <= limit:
                    break
                total = connection.execute(
                    "SELECT COUNT(*) FROM _translationcache").fetchone()[0]
                if not total:
                    break
                # No safety margin in the estimate: VACUUM reclaims index pages
                # as well as row pages, so a proportional guess already lands
                # under. Shaving it further would throw away entries the limit
                # was chosen to keep.
                # 估算里不留余量：VACUUM 连索引页一起回收，按比例估本来就会落到限值以
                # 内；再往下砍等于扔掉这个限值本来打算留住的条目。
                keep = max(1, int(total * limit / current))
                connection.execute(
                    "DELETE FROM _translationcache WHERE id NOT IN "
                    "(SELECT id FROM _translationcache ORDER BY id DESC LIMIT ?)",
                    (keep,))
                connection.commit()
                connection.execute("VACUUM")
                connection.commit()
        finally:
            connection.close()
        freed = before - database.stat().st_size
    except Exception:  # noqa: BLE001
        logger.debug("Could not cap the translation cache at %s", database, exc_info=True)
        return 0
    logger.info("Capped the paragraph cache, freeing %.0f MB", freed / 1048576)
    return freed


def clear_translation_cache(path: str | Path) -> int:
    """
    中文：清空 BabelDOC 的段落缓存，返回释放的字节数。

    Empty BabelDOC's paragraph cache.

    For the reader who asks for it explicitly — after changing model or prompt,
    when every cached translation was produced by something they no longer want
    to reproduce. Not called on exit: see :data:`CACHE_LIMIT`.

    Args:
        path (str | Path): The cache database.

    Returns:
        int: Bytes freed.
    """
    import sqlite3

    database = Path(path)
    try:
        if not database.is_file():
            return 0
        before = database.stat().st_size
        connection = sqlite3.connect(database, timeout=5.0)
        try:
            connection.execute("DELETE FROM _translationcache")
            connection.commit()
            connection.execute("VACUUM")
            connection.commit()
        finally:
            connection.close()
        freed = before - database.stat().st_size
    except Exception:  # noqa: BLE001
        logger.debug("Could not clear the translation cache", exc_info=True)
        return 0
    logger.info("Cleared the paragraph cache, freeing %.0f MB", freed / 1048576)
    return freed


def babeldoc_cache_path() -> Path:
    """
    中文：定位 BabelDOC 的段落缓存文件，且不为此导入 BabelDOC。

    Locate BabelDOC's paragraph cache, without importing BabelDOC to do it.

    The obvious implementation asks BabelDOC where its cache lives, so that a
    change upstream is followed rather than guessed at. That turns out to be a
    trap: importing ``babeldoc.assets.assets`` pulls in the whole of BabelDOC —
    164 MB resident, several seconds in a frozen build — and the process that
    calls this is the *server*, which is deliberately built never to import it.
    Asking the question cost more than the answer is worth, and it hung the
    settings page for over ninety seconds.

    So the path is spelled out, and BabelDOC is consulted only when something
    else has already imported it and the answer is therefore free.
    显而易见的写法是去问 BabelDOC 它的缓存在哪，好让上游改了位置时能自动跟上。那是个陷阱：
    import babeldoc.assets.assets 会把整个 BabelDOC 拉起来——常驻 164 MB，在打包版里要好几
    秒——而调用这里的正是**服务进程**，它是特意做成从不导入 BabelDOC 的。问这个问题的代价
    超过了答案的价值，并且把设置页挂了九十秒以上。

    因此路径写死，只有在别处已经导入过 BabelDOC、答案不要钱时才去问它。

    Returns:
        Path: The database's path, whether or not it exists yet.
    """
    import sys

    module = sys.modules.get("babeldoc.assets.assets")
    if module is not None:
        try:
            probe = Path(module.get_cache_file_path("probe", "fonts"))
            return probe.parent.parent / "cache.v1.db"
        except Exception:  # noqa: BLE001
            logger.debug("Could not read BabelDOC's cache location", exc_info=True)
    return Path.home() / ".cache" / "babeldoc" / "cache.v1.db"


def on_exit(paths, *, keep_intermediates: bool = False,
            cache_mb: int | None = None) -> int:
    """
    中文：退出时收尾——封顶日志与段落缓存，扫掉遗留的中间产物。

    Tidy up the stores that grow with use, on the way out.

    Three of them, treated differently because they are different things:
    leftover working directories are waste and go; llama-server's log has no
    rotation of its own and is trimmed to its tail; the paragraph cache is
    valuable and is capped, not cleared.

    Args:
        paths (AppPaths): Resolved application directories.
        keep_intermediates (bool): When set, leftover working directories are
            left alone, matching the setting that governs them per job.
        cache_mb (int | None): Cap for the paragraph cache in megabytes; ``0``
            empties it, ``None`` falls back to :data:`CACHE_LIMIT`.

    Returns:
        int: Total bytes freed.
    """
    freed = 0
    try:
        if not keep_intermediates:
            gone, _ = sweep_all(paths.meta_dir, paths.library_dir / "_latex")
            freed += gone
        freed += cap_log(paths.logs_dir / "llama-server.log")
        if cache_mb is None:
            cache_mb = CACHE_LIMIT // (1024 * 1024)
        if cache_mb <= 0:
            freed += clear_translation_cache(babeldoc_cache_path())
        else:
            freed += cap_translation_cache(babeldoc_cache_path(),
                                           cache_mb * 1024 * 1024)
    except Exception:  # noqa: BLE001
        logger.debug("Exit cleanup did not finish", exc_info=True)
    if freed:
        logger.info("Exit cleanup freed %.0f MB", freed / 1048576)
    return freed
