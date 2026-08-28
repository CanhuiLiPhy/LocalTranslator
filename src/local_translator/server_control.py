"""
中文：服务进程的启停控制——PID 文件、状态查询与优雅停止。

Start-up bookkeeping and shutdown for the server process.

The application is meant to be used without a terminal, which makes stopping it
as much a feature as starting it. A launcher that can only start leaves a
process holding a port and several gigabytes of model weights with no obvious
way to release them.

A PID file is used rather than matching on process names. Name matching finds
the wrong process as readily as the right one — an editor with the project open,
another checkout, a shell whose history mentions the command — and the failure
mode is killing something the user did not ask to kill.
"""

from __future__ import annotations

import logging
import os
import signal
import time
from pathlib import Path

logger = logging.getLogger(__name__)


def write_pid(path: Path) -> None:
    """
    中文：写入当前进程 PID。

    Record this process's PID.

    Args:
        path (Path): PID file location.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(os.getpid()), encoding="utf-8")


def clear_pid(path: Path) -> None:
    """
    中文：删除 PID 文件（仅当它属于当前进程）。

    Remove the PID file, but only if it is ours.

    Checking ownership matters on an unclean exit: a second instance may have
    started and rewritten the file, and deleting it then would leave that
    instance unstoppable.

    Args:
        path (Path): PID file location.
    """
    try:
        if path.is_file() and path.read_text(encoding="utf-8").strip() == str(os.getpid()):
            path.unlink()
    except OSError:
        logger.debug("Could not clear %s", path, exc_info=True)


def _alive(pid: int) -> bool:
    """
    中文：判断进程是否存在。

    Report whether a process exists.

    Args:
        pid (int): Process id.

    Returns:
        bool: ``True`` when the process is running.
    """
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # It exists and belongs to someone else, which for our purposes means
        # it is not ours to stop.
        # 它存在但属于其他用户，就本工具而言即「不该由我们停止」。
        return False
    return True


def running_pid(path: Path) -> int | None:
    """
    中文：返回正在运行的服务 PID，没有则返回 None。

    Read the PID of the running server.

    A stale file — one naming a process that has since exited — is treated as
    "not running" and removed, so a crash does not leave the app permanently
    believing it is already up.

    Args:
        path (Path): PID file location.

    Returns:
        int | None: The PID, or ``None`` when nothing is running.
    """
    try:
        pid = int(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None

    if _alive(pid):
        return pid
    try:
        path.unlink()
    except OSError:
        pass
    return None


def stop_server(path: Path, timeout: float = 20.0) -> tuple[bool, str]:
    """
    中文：优雅停止服务进程，超时后强制结束。

    Stop the running server, escalating if it does not exit.

    ``SIGTERM`` first, because the shutdown handler is what cancels running
    translations, stops the inference server and closes the database cleanly.
    ``SIGKILL`` only after the timeout, since killing outright would leave
    llama-server orphaned and still holding its weights in memory.

    Args:
        path (Path): PID file location.
        timeout (float): Seconds to wait for a clean exit.

    Returns:
        tuple[bool, str]: Whether anything was stopped, and a message for the
            user.
    """
    pid = running_pid(path)
    if pid is None:
        return False, "服务没有在运行"

    try:
        os.kill(pid, signal.SIGTERM)
    except OSError as exc:
        return False, f"无法停止进程 {pid}：{exc}"

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _alive(pid):
            return True, f"已停止服务（PID {pid}）"
        time.sleep(0.3)

    try:
        os.kill(pid, signal.SIGKILL)
    except OSError:
        pass
    return True, f"服务未在 {timeout:.0f} 秒内退出，已强制结束（PID {pid}）"
