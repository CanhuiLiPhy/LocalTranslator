"""
中文：调用 Tectonic 把 .tex 编译成 PDF。

Compile LaTeX to PDF with Tectonic.

**Why Tectonic and not a system TeX.** A full TeX Live is six gigabytes and
most readers do not have one — measured on the machine this was built on,
there was no ``pdflatex``, ``xelatex``, ``latexmk`` or ``kpsewhich`` anywhere.
An edition that depended on a system installation would install dead. Tectonic
is a single 54 MB binary under the MIT licence that fetches the macro packages
a document actually asks for, so it works on a machine with no TeX at all.

**Why XeTeX.** Tectonic is XeTeX underneath, which is what makes Chinese
output possible: see :mod:`local_translator.latex.preamble` for what that costs
on the input side and how it is paid.

A system TeX is still used when the reader points at one, because its macro
coverage is better than anything that downloads on demand.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# How long one compile may take. A ninety-page paper with a hundred figures
# took 33 seconds here; a run past this is not slow but stuck, usually on a
# macro waiting for input that will never come.
# 一次编译最多允许多久。九十页、上百张图的论文在这里用了 33 秒；超过这个不是慢，是卡住
# 了，通常是某个宏在等一个永远不会来的输入。
_TIMEOUT = 600

# Names a system TeX might use, best first. xelatex is required rather than
# preferred: the Chinese setup this project writes is xeCJK, which pdflatex
# cannot load.
# 系统 TeX 可能用的名字，越靠前越优先。xelatex 是必需而非偏好：本项目写入的中文方案是
# xeCJK，pdflatex 加载不了。
_SYSTEM_ENGINES = ("xelatex", "lualatex")

_SEARCH_PATHS = (
    "/Library/TeX/texbin",
    "/usr/local/texlive/bin/universal-darwin",
    "/opt/homebrew/bin",
    "/usr/local/bin",
    "/usr/bin",
)


@dataclass
class Result:
    """
    中文：一次编译的结果。

    What one compile produced.

    Attributes:
        pdf (Path | None): The output, when there is one.
        seconds (float): Wall time.
        log (str): Combined output, kept whole so a failure can be read.
        error (str): The first line that looks like an error, for a message a
            reader can act on. Empty when the compile succeeded.
    """

    pdf: Path | None
    seconds: float
    log: str
    error: str = ""

    @property
    def ok(self) -> bool:
        """中文：出 PDF 了没有。 / Whether a PDF came out."""
        return self.pdf is not None and self.pdf.is_file()


def find_tectonic(bin_dir: Path | None = None) -> Path | None:
    """
    中文：找到 tectonic 可执行文件。

    Locate the Tectonic binary.

    Args:
        bin_dir (Path | None): The application's own binary directory, which is
            where a bundled copy lives beside llama-server.

    Returns:
        Path | None: The binary, or ``None`` when there is none to find.
    """
    places = []
    if bin_dir:
        places.append(bin_dir / "tectonic")
    # Inside a frozen bundle the engine ships as a data file, so it is beside
    # the unpacked resources rather than in the reader's data directory.
    # 在冻结的应用包里，引擎是以数据文件的形式随包发布的，所以它在解包出来的资源旁边，
    # 而不在读者的数据目录里。
    frozen = getattr(sys, "_MEIPASS", None)
    if frozen:
        places += [Path(frozen) / "tectonic", Path(frozen) / "bin" / "tectonic"]
    if getattr(sys, "frozen", False):
        here = Path(sys.executable).resolve().parent
        places += [here / "tectonic", here.parent / "Resources" / "tectonic"]
    for candidate in places:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    found = shutil.which("tectonic")
    return Path(found) if found else None


def find_system_engine() -> Path | None:
    """
    中文：找一个系统自带的 XeTeX/LuaTeX。

    Locate a system XeTeX or LuaTeX, if the machine has one.

    Returns:
        Path | None: The engine, or ``None``. Searched in the places macOS TeX
            distributions install to as well as on ``PATH``, because
            ``/Library/TeX/texbin`` is often absent from a GUI application's
            environment even when MacTeX is installed.
    """
    for name in _SYSTEM_ENGINES:
        found = shutil.which(name)
        if found:
            return Path(found)
        for folder in _SEARCH_PATHS:
            candidate = Path(folder) / name
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return candidate
    return None


def resolve(configured: str | None = None, bin_dir: Path | None = None):
    """
    中文：定下这次用哪个引擎，找不到就返回 None。

    Decide which engine to use, if any.

    Args:
        configured (str | None): A path the reader set in the settings.
        bin_dir (Path | None): The application's binary directory.

    Returns:
        tuple[Path, bool] | None: The engine and whether it is a system one, or
            ``None`` when this installation cannot compile LaTeX at all.

    Note:
        Whether the feature exists is answered by looking for the binary, not
        by a flag written at install time. A marker can be wrong — copied from
        another machine, left behind by an upgrade — and a feature that claims
        to be present and then fails is worse than one that says it is absent.
        功能在不在，靠找可执行文件来回答，而不是靠安装时写下的标记。标记可能是错的——
        从别的机器拷来的、升级后残留的——而一个自称存在、真用起来却失败的功能，比一个
        老老实实说自己不在的更糟。
    """
    if configured:
        candidate = Path(configured).expanduser()
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate, candidate.name in _SYSTEM_ENGINES
        logger.warning("设置里的 TeX 路径不可用：%s", configured)

    system = find_system_engine()
    if system is not None:
        return system, True
    bundled = find_tectonic(bin_dir)
    if bundled is not None:
        return bundled, False
    return None


def available(configured: str | None = None, bin_dir: Path | None = None) -> bool:
    """中文：这套安装能不能编译 LaTeX。 / Whether this installation can compile LaTeX."""
    return resolve(configured, bin_dir) is not None


def describe(configured: str | None = None, bin_dir: Path | None = None) -> str:
    """
    中文：用一句话说明当前用的是哪个引擎。

    Describe the engine in one line, for the settings page.

    Args:
        configured (str | None): A path the reader set.
        bin_dir (Path | None): The application's binary directory.

    Returns:
        str: What a reader needs to know: which engine, or that there is none.
    """
    found = resolve(configured, bin_dir)
    if found is None:
        return "未找到 TeX 引擎，LaTeX 源码翻译不可用"
    engine_path, is_system = found
    kind = "系统 TeX" if is_system else "随包 Tectonic"
    return f"{kind}：{engine_path}"


def _first_error(output: str) -> str:
    """
    中文：从编译日志里挑出第一条像样的报错。

    Pick the first line of a log that names what went wrong.

    Args:
        output (str): Combined stdout and stderr.

    Returns:
        str: One line, or the log's tail when nothing matches — an empty
            message tells a reader nothing, and the tail at least shows where
            it stopped.
    """
    for line in output.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("error:") or stripped.startswith("! "):
            return stripped[:300]
    tail = [line for line in output.splitlines() if line.strip()]
    return tail[-1][:300] if tail else ""


def compile_document(main: Path, engine: Path, *, system: bool = False,
                     passes: int = 2) -> Result:
    """
    中文：编译一份文档，返回结果。

    Compile one document.

    Args:
        main (Path): The ``.tex`` to compile. Its parent is the working
            directory, so the microtype stub and the paper's own class files
            are found.
        engine (Path): Tectonic, or a system ``xelatex``.
        system (bool): Whether ``engine`` is a system engine, which takes
            different arguments and has to be run more than once to settle
            cross-references.
        passes (int): How many times to run a system engine. Ignored for
            Tectonic, which decides for itself.

    Returns:
        Result: With the PDF when one was produced.
    """
    import time

    folder = main.parent
    pdf = main.with_suffix(".pdf")
    pdf.unlink(missing_ok=True)

    if system:
        command = [str(engine), "-interaction=nonstopmode",
                   "-halt-on-error", main.name]
        rounds = max(1, passes)
    else:
        command = [str(engine), "-X", "compile", "--keep-logs", main.name]
        rounds = 1

    started = time.time()
    output = ""
    for _ in range(rounds):
        try:
            done = subprocess.run(command, cwd=folder, capture_output=True,
                                  text=True, timeout=_TIMEOUT)
        except subprocess.TimeoutExpired:
            return Result(None, time.time() - started, output,
                          f"编译超过 {_TIMEOUT} 秒，已放弃")
        except Exception as problem:  # noqa: BLE001
            return Result(None, time.time() - started, output, str(problem)[:300])
        output += (done.stdout or "") + (done.stderr or "")
        if not system and done.returncode != 0:
            break

    seconds = time.time() - started
    if pdf.is_file():
        logger.info("编译成功：%s（%.1f 秒）", pdf.name, seconds)
        return Result(pdf, seconds, output)
    return Result(None, seconds, output, _first_error(output))
