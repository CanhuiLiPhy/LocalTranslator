"""
中文：桌面应用入口——一个窗口，关掉窗口即整个程序退出。

Run the translator as a desktop application.

The web server, the browser page and the inference server are three processes
that a reader should never have to think about. This module makes them one
application: double-clicking opens a window, closing the window ends everything
it started, and nothing is left listening afterwards.

**What is different from ``localtrans serve --native``.** The command is meant
to be run from a terminal by someone who will also stop it there, so it prints
to the console, uses the configured port whatever else is on it, and leaves a
second instance to fail on a bind error. An application cannot do any of those
things:

* **Output goes to a file**, because a packaged bundle has no console to print
  to and a crash the reader cannot see is a crash they cannot report.
* **The port is checked first.** If our own server is already up, this instance
  opens a window onto that server instead of starting a second one; if the port
  belongs to something else, a free one is taken. Either way what opens is a
  window — an application that answered a double-click with a browser tab would
  not be an application.
* **Teardown is guaranteed.** The graceful path is NiceGUI's shutdown hook, and
  behind it sits a handler that stops the inference server on the way out
  however the process ends — a window closed while a translation runs would
  otherwise leave several gigabytes of weights resident with nothing to serve.
"""

from __future__ import annotations

import atexit
import logging
import os
import signal
import socket
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# Window size at first open, in points. Wide enough for the reader's two-page
# spread, which is the widest thing the interface shows.
# 首次打开时的窗口尺寸（点）。宽度以阅读器的双页对照为准，那是界面里最宽的内容。
_WINDOW = (1440, 960)

# Ports tried after the configured one, when that is taken by something that is
# not our own server.
# 配置端口被其它程序占用时，依次尝试的备用端口。
_FALLBACK_PORTS = range(8081, 8091)

# Kept so the log does not grow without bound across sessions.
# 保留的日志字节数上限，避免多次运行后无限增长。
_LOG_BYTES = 2 * 1024 * 1024


def _log_to_file(logs_dir: Path) -> None:
    """
    中文：把日志写进数据目录，打包后的应用没有控制台。

    Send logging to a file inside the data directory.

    A packaged application has no console, so a traceback printed to stderr goes
    nowhere. Writing it down is what makes a fault reportable.

    Args:
        logs_dir (Path): Directory to write ``app.log`` into. Created if absent.

    Returns:
        None.
    """
    from logging.handlers import RotatingFileHandler

    logs_dir.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(
        logs_dir / "app.log", maxBytes=_LOG_BYTES, backupCount=2, encoding="utf-8",
    )
    handler.setFormatter(logging.Formatter(
        "%(asctime)s  %(levelname)-7s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(logging.INFO)
    for noisy in ("httpx", "httpcore", "urllib3", "uvicorn.access", "watchfiles"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _in_use(port: int, host: str = "127.0.0.1") -> bool:
    """
    中文：判断某个端口是否已被占用。

    Report whether a port is already accepting connections.

    Args:
        port (int): Port to test.
        host (str): Address to test it on.

    Returns:
        bool: ``True`` when something is listening.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.4)
        return probe.connect_ex((host, port)) == 0


def _ours(port: int, host: str = "127.0.0.1") -> bool:
    """
    中文：判断占用端口的是不是本程序自己的服务。

    Report whether the server on a port is this application's.

    Asks the API for its health endpoint rather than assuming: a port in the
    8080 range is a popular one, and opening a window onto a stranger's web
    application would be worse than failing.

    Args:
        port (int): Port to ask.
        host (str): Address to ask on.

    Returns:
        bool: ``True`` when the answer looks like ours.
    """
    import json
    import urllib.request

    try:
        with urllib.request.urlopen(f"http://{host}:{port}/api/status", timeout=1.0) as reply:
            body = json.loads(reply.read().decode("utf-8"))
    except Exception:  # noqa: BLE001
        return False
    return isinstance(body, dict) and {"model", "backend"} <= body.keys()


def _free_port(preferred: int, host: str = "127.0.0.1") -> int | None:
    """
    中文：挑一个可用端口；若本程序已在运行则返回 None。

    Choose a port to serve on.

    Args:
        preferred (int): The configured port, tried first.
        host (str): Address to bind on.

    Returns:
        int | None: A free port, or ``None`` when this application is already
            running — in which case the caller should show that window rather
            than start a second server.
    """
    if not _in_use(preferred, host):
        return preferred
    if _ours(preferred, host):
        return None
    for candidate in _FALLBACK_PORTS:
        if candidate != preferred and not _in_use(candidate, host):
            logger.info("Port %d is taken; serving on %d instead", preferred, candidate)
            return candidate
    return preferred


def _stop_everything_on_the_way_out(service) -> None:
    """
    中文：无论进程怎么结束，都确保推理服务被停掉。

    Make sure the inference server is stopped however this process ends.

    NiceGUI's shutdown hook covers the ordinary case. This covers the rest: a
    window closed mid-translation, a signal, an unhandled exception. Without it
    the weights stay resident — up to 12 GB of Metal buffers for the 7B — with
    nothing left to serve them to.

    Args:
        service (TranslatorService): The running service.

    Returns:
        None.
    """
    stopped = False

    def stop(*_args) -> None:
        """中文：停掉推理服务，只执行一次。 / Stop the inference server, once."""
        nonlocal stopped
        if stopped:
            return
        stopped = True
        try:
            service.supervisor.stop(include_adopted=True)
        except Exception:  # noqa: BLE001
            logger.warning("Could not stop the inference server on exit", exc_info=True)
        try:
            from local_translator.server_control import clear_pid

            clear_pid(service.paths.pid_path)
        except Exception:  # noqa: BLE001
            pass

    atexit.register(stop)
    for received in (signal.SIGTERM, signal.SIGINT):
        try:
            previous = signal.getsignal(received)

            def handler(signum, frame, _previous=previous):
                """中文：先收尾再交回原处理器。 / Clean up, then defer."""
                stop()
                if callable(_previous):
                    _previous(signum, frame)
                else:
                    sys.exit(0)

            signal.signal(received, handler)
        except ValueError:
            # Not on the main thread; the atexit hook still applies.
            # 不在主线程上；atexit 钩子依然有效。
            pass


def _window_onto(port: int) -> None:
    """
    中文：给一个已经在跑的服务开一个窗口，不再启动第二个服务。

    Open a window onto a server that is already running.

    This is what happens when the application is started while its own server is
    up — a second copy launched from the Dock, or a server left behind by
    ``localtrans serve``. Starting another server would fail on the port, and
    opening a browser tab would make an application that is supposed to be a
    window behave like a web page. Neither is what the reader asked for by
    double-clicking the icon.

    Args:
        port (int): Port the running server is on.

    Returns:
        None. Returns when the window is closed.
    """
    import webview

    logger.info("A server is already on port %d; opening a window onto it", port)
    webview.create_window("本地翻译", f"http://127.0.0.1:{port}",
                          width=_WINDOW[0], height=_WINDOW[1])
    webview.start()


def run(port: int | None = None) -> int:
    """
    中文：以桌面应用方式启动，关闭窗口即完全退出。

    Start the application in a desktop window.

    Args:
        port (int | None): Port to serve on. ``None`` uses the configured port,
            falling back to a free one when that is taken.

    Returns:
        int: Process exit status — ``0`` on a clean exit, ``1`` when the
            desktop window cannot be opened at all.
    """
    from local_translator.config import get_paths, get_settings

    paths = get_paths().ensure()
    _log_to_file(paths.logs_dir)
    logger.info("Starting the desktop application")

    settings = get_settings()
    chosen = port or settings.server.port
    resolved = _free_port(chosen)

    from local_translator.web import app as web_app

    if not web_app._native_available():
        logger.error("pywebview is not installed, so no window can be opened")
        sys.stderr.write(
            "缺少 pywebview，无法打开窗口。请重新安装本程序。\n"
            "pywebview is missing, so the window cannot be opened.\n"
        )
        return 1

    if resolved is None:
        _window_onto(chosen)
        return 0

    from local_translator.service import get_service

    _stop_everything_on_the_way_out(get_service())

    # NiceGUI reads this to size the window it hands to pywebview.
    # NiceGUI 读取该值来决定交给 pywebview 的窗口尺寸。
    os.environ.setdefault("LOCALTRANS_WINDOW", "x".join(str(n) for n in _WINDOW))
    web_app.run(host="127.0.0.1", port=resolved, show=False, native=True)
    return 0


def main() -> int:
    """
    中文：打包后可执行文件的入口。

    Entry point for the packaged executable.

    The first thing it does is hand control to ``multiprocessing`` in case this
    process is a worker rather than the application. A frozen bundle has no
    importable ``__main__`` module, so a child process is started by running the
    executable again with arguments describing the work; without
    ``freeze_support`` those arguments are ignored and the child opens a second
    copy of the application, which opens a third. Observed as the interface
    starting and stopping in a loop.

    Returns:
        int: Process exit status.
    """
    import multiprocessing

    multiprocessing.freeze_support()
    try:
        return run()
    except Exception:  # noqa: BLE001
        logger.exception("The application stopped with an error")
        return 1


if __name__ == "__main__":
    sys.exit(main())
