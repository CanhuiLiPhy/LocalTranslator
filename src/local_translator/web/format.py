"""
中文：界面展示用的格式化工具。

Display formatting helpers shared by the web pages.

Kept separate from the page code so the same wording appears in the job list,
the library table and the CLI without being retyped.
"""

from __future__ import annotations

from pathlib import Path

# Job status to (Chinese label, Quasar colour) / 任务状态到（中文标签, 颜色）
STATUS_LABELS: dict[str, tuple[str, str]] = {
    "queued": ("排队中", "grey"),
    "running": ("翻译中", "blue"),
    "completed": ("已完成", "green"),
    "failed": ("失败", "red"),
    "cancelled": ("已取消", "orange"),
    "interrupted": ("已中断", "amber"),
}


# Statuses a reader would call "did not work": the paper needs redoing.
# 读者眼里「没成」的状态：这篇要重来。
UNSUCCESSFUL: frozenset[str] = frozenset({"failed", "cancelled", "interrupted"})


def local_time(iso: str | None) -> str:
    """
    中文：把数据库里的 ISO 时间串转成本地时区的「年-月-日 时:分」。

    Render a stored ISO timestamp in the reader's local time zone.

    The database stores UTC with an explicit offset
    (``2026-08-28T10:15:02.405191+00:00``); a reader searching for the paper
    they translated "this morning" thinks in wall-clock time, so the string
    shown — and matched by the history search — is local.

    Args:
        iso (str | None): ISO-8601 timestamp, or ``None``.

    Returns:
        str: ``"2026-08-28 18:15"`` in local time, ``"—"`` for ``None``, and
            the raw string when it does not parse.
    """
    if not iso:
        return "—"
    from datetime import datetime

    try:
        moment = datetime.fromisoformat(iso)
    except ValueError:
        return iso
    return moment.astimezone().strftime("%Y-%m-%d %H:%M")


def status_label(status: str) -> tuple[str, str]:
    """
    中文：把任务状态码转成中文标签与颜色。

    Map a job status to its display label and colour.

    Args:
        status (str): Status code stored in the database.

    Returns:
        tuple[str, str]: ``(chinese_label, quasar_colour)``. Unknown statuses
            fall back to the raw code in grey.
    """
    return STATUS_LABELS.get(status, (status, "grey"))


def human_bytes(size: float | None) -> str:
    """
    中文：格式化字节数。

    Format a byte count.

    Args:
        size (float | None): Byte count, or ``None``.

    Returns:
        str: e.g. ``"6.2 GB"``, or ``"—"`` when the input is ``None``.
    """
    if not size:
        return "—"
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if abs(value) < 1024.0:
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} TB"


def human_tokens(count: int | None) -> str:
    """
    中文：格式化 token 数量。

    Format a token count compactly.

    Args:
        count (int | None): Number of tokens.

    Returns:
        str: e.g. ``"224.6k"`` or ``"812"``; ``"0"`` when absent.
    """
    if not count:
        return "0"
    if count >= 1_000_000:
        return f"{count / 1_000_000:.1f}M"
    if count >= 1000:
        return f"{count / 1000:.1f}k"
    return str(count)


def human_duration(milliseconds: int | None) -> str:
    """
    中文：格式化耗时。

    Format a duration.

    Args:
        milliseconds (int | None): Elapsed time in milliseconds.

    Returns:
        str: e.g. ``"8分42秒"`` or ``"12.4秒"``; ``"—"`` when absent.
    """
    if not milliseconds:
        return "—"
    seconds = milliseconds / 1000
    if seconds < 60:
        return f"{seconds:.1f}秒"
    minutes, rest = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}分{rest}秒"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}小时{minutes}分"


def usage_line(job: dict) -> str:
    """
    中文：生成翻译完成后显示的用量小字。

    Build the one-line token usage summary shown under a finished job.

    Presents cache hits alongside token counts because the two together explain
    the cost: re-translating a document mostly hits cache, and seeing that
    directly is what makes the caching behaviour trustworthy.

    Args:
        job (dict): Job row containing token and paragraph counters.

    Returns:
        str: A single line such as ``"输入 128.4k / 输出 96.2k / 合计 224.6k
            tokens · 段落缓存命中 312/1450 · 耗时 8分42秒 · 本地模型（无费用）"``.
    """
    parts = [
        f"输入 {human_tokens(job.get('prompt_tokens'))}"
        f" / 输出 {human_tokens(job.get('completion_tokens'))}"
        f" / 合计 {human_tokens(job.get('total_tokens'))} tokens"
    ]

    total = job.get("para_total") or 0
    if total:
        parts.append(f"段落缓存命中 {job.get('para_cache_hit') or 0}/{total}")

    if job.get("duration_ms"):
        parts.append(f"耗时 {human_duration(job.get('duration_ms'))}")

    backend = job.get("backend")
    if backend in ("api", "hybrid"):
        # Hybrid spends on the API for the share it routed there, so labelling
        # it "no cost" like a local run would understate what it actually used.
        # 混合模式对分流到 API 的部分是有花费的，标成「无费用」会低报实际用量。
        label = "API 模型" if backend == "api" else "混合模式"
        cost = job.get("est_cost")
        parts.append(f"预估费用 ¥{cost:.4f}" if cost else label)
    else:
        parts.append("本地模型（无费用）")

    failures = job.get("placeholder_failures") or 0
    if failures:
        parts.append(f"⚠ {failures} 段占位符异常，已保留原文")

    return " · ".join(parts)


def short_path(path: str | Path | None, keep: int = 2) -> str:
    """
    中文：把长路径缩短为末尾若干段，便于在表格里显示。

    Shorten a path for display in a narrow column.

    Args:
        path (str | Path | None): Path to shorten.
        keep (int): Number of trailing components to keep.

    Returns:
        str: e.g. ``"…/papers/2510.04704v4.pdf"``, or ``"—"`` when absent.
    """
    if not path:
        return "—"
    parts = Path(path).parts
    if len(parts) <= keep:
        return str(path)
    return "…/" + "/".join(parts[-keep:])


def display_path(path: Path | str) -> str:
    """
    中文：把路径显示成 ~/… 的形式，避免界面上出现用户名。

    Render a path for display, with the home directory shown as ``~``.

    An absolute path inside the home directory carries the account name, which
    then appears in every screenshot of the page and in anything a reader
    copies out of it. The tilde form is also what a reader recognises and what
    they can paste back into a shell.

    A path outside the home directory is returned unchanged: shortening is only
    possible where there is a prefix to shorten, and inventing one would be
    misleading.

    Args:
        path (Path | str): Path to render.

    Returns:
        str: ``~/relative/part`` when the path lies under the home directory,
            otherwise the path unchanged.
    """
    text = Path(path)
    try:
        inside = text.relative_to(Path.home())
    except ValueError:
        return str(text)
    # relative_to() returns "." for the home directory itself, which would
    # render as the odd-looking "~/.".
    # 对家目录本身，relative_to() 返回 "."，直接拼接会得到别扭的 "~/."。
    return "~" if str(inside) == "." else f"~/{inside}"
