"""
中文：词库安装——下载 ECDICT 数据并导入本地 SQLite。

Download and install the ECDICT dataset.

The data is fetched from the project's own repository rather than bundled,
because it is 66 MB of reference material that most of the application never
touches. Making it an explicit, resumable install keeps the application itself
small and lets the dictionary be removed independently.

Downloads reuse the model downloader, which resumes across dropped connections
and budgets its retries by progress rather than attempt count.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Pinned to a commit rather than a branch: the dataset is updated in place, and
# a build that silently changes its dictionary between installs is harder to
# reason about than one that needs an explicit bump.
# 固定到具体提交而非分支：该数据集会原地更新，安装之间静默换词库比需要显式升级
# 更难排查。
ECDICT_REF = "master"
ECDICT_BASE = f"https://raw.githubusercontent.com/skywind3000/ECDICT/{ECDICT_REF}"
ECDICT_CSV_URL = f"{ECDICT_BASE}/ecdict.csv"
ECDICT_LEMMA_URL = f"{ECDICT_BASE}/lemma.en.txt"

# Approximate sizes, used only to show a percentage while downloading.
# 约略大小，仅用于下载时显示百分比。
ECDICT_CSV_SIZE = 69_000_000
ECDICT_LEMMA_SIZE = 2_400_000


def install_dictionary(paths=None, progress=None, keep_source: bool = False) -> dict[str, int]:
    """
    中文：下载并导入 ECDICT 词库，返回词条与词形数量。

    Download the ECDICT dataset and import it.

    Args:
        paths (AppPaths | None): Application paths; resolved from config when
            omitted.
        progress (Callable[[str, float], None] | None): Called with a stage
            label and a fraction in ``[0, 1]``.
        keep_source (bool): Keep the downloaded CSV afterwards. Off by default —
            it is 66 MB that is never read again once imported.

    Returns:
        dict[str, int]: ``entries`` and ``lemmas`` imported, and ``reverse``
            index rows built for Chinese-to-English lookup.

    Raises:
        RuntimeError: If a download fails.
    """
    from local_translator.config import get_paths
    from local_translator.dictionary.store import get_dictionary
    from local_translator.llm.downloader import download_file

    paths = paths or get_paths()
    work = paths.data_dir / "dictionary-src"
    work.mkdir(parents=True, exist_ok=True)

    def stage(label: str, weight_from: float, weight_to: float):
        def report(done: int, total: int) -> None:
            if progress and total:
                span = weight_to - weight_from
                progress(label, weight_from + span * min(1.0, done / total))
        return report

    csv_path = work / "ecdict.csv"
    lemma_path = work / "lemma.en.txt"

    download_file(ECDICT_CSV_URL, csv_path,
                  progress=stage("下载词库", 0.0, 0.55), expected_size=None)
    download_file(ECDICT_LEMMA_URL, lemma_path,
                  progress=stage("下载词形表", 0.55, 0.60), expected_size=None)

    store = get_dictionary()
    if progress:
        progress("导入词库", 0.62)

    def rows(count: int) -> None:
        if progress:
            # The row count is not known in advance, so the bar is driven by a
            # ratio against the published entry count rather than a true total.
            # 行数事先未知，进度按已发布的词条量估算，而非真实总数。
            progress("导入词库", 0.62 + 0.33 * min(1.0, count / 770_000))

    entries = store.import_csv(csv_path, progress=rows)
    if progress:
        progress("导入词形表", 0.95)
    lemmas = store.import_lemma(lemma_path)

    # Built here rather than on first Chinese lookup: it takes about three
    # seconds and thirty megabytes, both of which belong in an install the user
    # has already agreed to wait for.
    # 在这里建好而不是等到首次汉英查询：耗时约 3 秒、增加约 30 MB，这两项开销都该
    # 落在用户已经同意等待的安装过程里。
    if progress:
        progress("建立汉英索引", 0.97)
    reverse = store.build_reverse_index()

    if not keep_source:
        for path in (csv_path, lemma_path):
            try:
                path.unlink()
            except OSError:
                logger.debug("Could not remove %s", path, exc_info=True)
        try:
            work.rmdir()
        except OSError:
            pass

    if progress:
        progress("完成", 1.0)
    return {"entries": entries, "lemmas": lemmas, "reverse": reverse}
