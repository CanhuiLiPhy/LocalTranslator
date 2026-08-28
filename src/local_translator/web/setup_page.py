"""
中文：首次运行向导——在窗口里把推理程序、模型权重与阅读器组件装好。

First-run setup, done inside the window.

The command line has ``localtrans setup`` and ``localtrans setup-reader``, which
is fine for someone who cloned the repository. Someone who downloaded an
application has no terminal in the picture at all, and an application that opens
to an error telling them to run a command has failed at the only step that
matters.

**What this installs.** Three things, each checked before it is offered:

* **the inference program** — a prebuilt ``llama-server``, about 10 MB;
* **the model weights** — the size depends on the edition, 1.5 GB for the local
  lightweight one and about 7 GB for the full one;
* **the reader** — pdf.js, without which the bilingual view has nothing to
  render with;
* **the glossary seed** — the initial renderings of technical terms;
* **the offline dictionary** — for looking a word up while reading.

The last two used to be command-line steps, which is no use to someone who
downloaded an application: the bundle carries no ``localtrans`` executable, so
the glossary page's own advice — run ``localtrans glossary seed`` — named a
program the reader does not have.

Fonts are not listed: BabelDOC fetches its own on the first translation, and a
progress bar for something the reader did not ask for is noise.

**Downloads resume.** Each is written to a ``.part`` file next to its
destination, so a window closed halfway through costs the time already spent and
nothing else. Pressing install again continues from where it stopped.

The page is registered at ``/setup`` and is where :func:`needs_setup` sends a
reader who has not finished it.
"""

from __future__ import annotations

import asyncio
import logging

from nicegui import ui

from local_translator import edition
from local_translator.config import get_paths

logger = logging.getLogger(__name__)

# Refresh rate for a download's progress bar, in seconds. Fast enough to look
# live, slow enough that a 7 GB download does not spend its time redrawing.
# 下载进度条的刷新间隔（秒）。快到看起来是实时的，又不至于让 7GB 的下载把时间花在重绘上。
_TICK = 0.25


def _missing() -> dict[str, bool]:
    """
    中文：检查三样组件各自是否还缺。

    Report which components are still missing.

    Returns:
        dict[str, bool]: ``runtime``, ``model`` and ``reader``, each ``True``
            when that component still needs installing. ``model`` is always
            ``False`` on the lite edition, which runs against a cloud API and
            has no weights to fetch.
    """
    from local_translator.llm.downloader import find_installed_models, llama_server_path
    from local_translator.web import pdfjs

    paths = get_paths()
    local = edition.local_model() is not None
    return {
        "runtime": local and llama_server_path(paths) is None,
        "model": local and not find_installed_models(paths),
        "reader": not pdfjs.is_installed(),
        "dictionary": not _dictionary_is_installed(),
    }


def _glossary_is_empty() -> bool:
    """
    中文：判断术语库是否还是空的。

    Report whether the glossary has no terms yet.

    A packaged application has no terminal, so telling the reader to run
    ``localtrans glossary seed`` — which is what the empty glossary page used to
    say — is telling them to run a program they do not have. The seed belongs in
    this wizard instead.

    Returns:
        bool: ``True`` when the glossary holds nothing.
    """
    try:
        from local_translator.db import get_db

        rows = get_db().query("SELECT COUNT(*) AS n FROM glossary_terms WHERE enabled = 1")
        return not rows or not rows[0]["n"]
    except Exception:  # noqa: BLE001
        logger.debug("Could not count the glossary", exc_info=True)
        return False


def _dictionary_is_installed() -> bool:
    """
    中文：判断离线词典是否已安装。

    Report whether the offline dictionary is in place.

    Returns:
        bool: ``True`` when it is, and ``True`` on any error, so a broken check
            never blocks the wizard on a component that is merely optional.
    """
    try:
        from local_translator.dictionary import get_dictionary

        return bool(get_dictionary().installed())
    except Exception:  # noqa: BLE001
        logger.debug("Could not check the dictionary", exc_info=True)
        return True


def needs_setup() -> bool:
    """
    中文：判断是否还需要走首次运行向导。

    Report whether the first-run wizard still has work to do.

    Returns:
        bool: ``True`` when any component is missing.
    """
    try:
        return any(_missing().values())
    except Exception:  # noqa: BLE001
        logger.debug("Could not check the installation", exc_info=True)
        return False


def _human(size: float) -> str:
    """
    中文：把字节数写成人能读的形式。

    Format a byte count for a person.

    Args:
        size (float): Number of bytes.

    Returns:
        str: The size in the largest unit that keeps it above one.
    """
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit in ("B", "KB") else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


class _Step:
    """
    中文：向导里的一步——一行说明、一个进度条、一个状态。

    One step of the wizard: a description, a progress bar and a state.

    Attributes:
        done (bool): Whether this step has completed.
        bytes_done (int): Bytes received so far.
        bytes_total (int): Bytes expected, or ``0`` while unknown.
    """

    def __init__(self, title: str, detail: str, install):
        self.title = title
        self.detail = detail
        self.install = install
        self.done = False
        self.failed = ""
        self.bytes_done = 0
        self.bytes_total = 0
        self.label = None
        self.bar = None
        self.stage = ""

    def note(self, done: int, total: int) -> None:
        """
        中文：记录下载进度，供界面读取。

        Record download progress for the interface to read.

        Args:
            done (int): Bytes received.
            total (int): Bytes expected, ``0`` when the server did not say.

        Returns:
            None.
        """
        self.bytes_done, self.bytes_total = done, total

    def render(self) -> None:
        """中文：画出这一步。 / Draw this step."""
        with ui.card().classes("w-full"), ui.column().classes("w-full gap-1"):
            ui.label(self.title).classes("text-base font-medium")
            ui.label(self.detail).classes("text-xs text-gray-600")
            self.bar = ui.linear_progress(value=0, show_value=False).classes("w-full")
            self.label = ui.label("").classes("text-xs text-gray-600")

    def refresh(self) -> None:
        """中文：把最新进度写进界面。 / Push the latest progress into the interface."""
        if self.bar is None or self.label is None:
            return
        if self.failed:
            self.label.set_text(f"失败：{self.failed}")
            return
        if self.done:
            self.bar.set_value(1.0)
            self.label.set_text("已完成")
            return
        if self.stage:
            # A step whose work is not a download reports what it is doing
            # rather than a byte count, because bytes would mean nothing here.
            # 工作内容不是下载的步骤，报告它在做什么而不是字节数——字节数在这里没有意义。
            self.bar.set_value(self.bytes_done / max(self.bytes_total, 1))
            self.label.set_text(self.stage)
        elif self.bytes_total:
            self.bar.set_value(self.bytes_done / self.bytes_total)
            self.label.set_text(
                f"{_human(self.bytes_done)} / {_human(self.bytes_total)}")
        elif self.bytes_done:
            self.label.set_text(_human(self.bytes_done))


def _steps() -> list[_Step]:
    """
    中文：按当前缺什么，列出需要执行的步骤。

    List the steps this installation still needs.

    Returns:
        list[_Step]: One step per missing component, in install order — the
            runtime before the weights it will serve, and the reader last
            because a translation can be produced without it.
    """
    from local_translator.llm.downloader import install_llama_cpp, install_model
    from local_translator.web import pdfjs

    paths = get_paths().ensure()
    missing = _missing()
    spec = edition.local_model() or {}
    steps: list[_Step] = []

    if missing["runtime"]:
        step = _Step("推理程序", "llama.cpp 的 llama-server，下载约 11 MB", None)
        step.install = lambda s=step: install_llama_cpp(paths, progress=s.note)
        steps.append(step)

    if missing["model"]:
        quant = spec.get("quant", "Q6_K")
        note = spec.get("size_note", "")
        step = _Step("翻译模型", f"Hunyuan-MT2 {quant} 权重{'，' + note if note else ''}", None)
        step.install = lambda s=step, q=quant: install_model(
            paths, q, repo=spec.get("repo") or "tencent/Hy-MT2-7B-GGUF", progress=s.note)
        steps.append(step)

    if missing["reader"]:
        step = _Step("双语阅读器", "pdf.js 组件，下载约 7 MB", None)
        step.install = lambda s=step: pdfjs.install()
        steps.append(step)

    if missing["dictionary"]:
        step = _Step("离线词典", "英汉与汉英词条，供阅读时查词，下载约 70 MB", None)
        step.install = lambda s=step: _install_dictionary(s)
        steps.append(step)

    return steps


def _seed_glossary() -> None:
    """
    中文：导入术语库种子。

    Populate the glossary from its seed sources.

    Returns:
        None.
    """
    from local_translator.db import get_db
    from local_translator.glossary.seed import seed
    from local_translator.glossary.store import GlossaryStore

    report = seed(GlossaryStore(get_db()))
    logger.info("Seeded the glossary: %s", report)


def _install_dictionary(step) -> None:
    """
    中文：下载并导入离线词典。

    Download and import the offline dictionary.

    Args:
        step (_Step): The wizard step, whose ``note`` receives the progress.

    Returns:
        None.
    """
    from local_translator.dictionary.install import install_dictionary

    def note(stage: str, fraction: float) -> None:
        """中文：把「阶段 + 比例」换成进度条要的「已完成 / 总数」。 / Adapt the progress shape."""
        step.stage = stage
        step.note(int(fraction * 1000), 1000)

    install_dictionary(get_paths(), progress=note)


@ui.page("/setup")
def setup_page() -> None:
    """
    中文：首次运行向导页面。

    Render the first-run wizard.

    Returns:
        None.
    """
    ui.page_title("本地翻译 · 首次设置")

    with ui.column().classes("w-full max-w-3xl mx-auto p-8 gap-4"):
        ui.label("首次设置").classes("text-2xl font-semibold")
        ui.label(
            f"当前版本：{edition.display_name()}。"
            "下面的组件只需下载一次，之后每次打开都直接可用。"
        ).classes("text-sm text-gray-700")

        steps = _steps()
        if not steps:
            ui.label("一切就绪。").classes("text-base")
            ui.button("开始使用", on_click=lambda: ui.navigate.to("/")).props("unelevated")
            return

        for step in steps:
            step.render()

        status = ui.label("").classes("text-sm")
        start = ui.button("开始下载").props("unelevated")

        async def install_all() -> None:
            """中文：依次执行每一步，全部完成后进入主界面。 / Run every step, then continue."""
            start.disable()
            for step in steps:
                status.set_text(f"正在下载：{step.title}")
                try:
                    await asyncio.to_thread(step.install)
                    step.done = True
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Setup step %s failed", step.title, exc_info=True)
                    step.failed = str(exc)
                    status.set_text(f"「{step.title}」下载失败，可以重试。")
                    start.set_text("重试")
                    start.enable()
                    return
            status.set_text("全部完成。")
            ui.navigate.to("/")

        start.on_click(install_all)
        ui.timer(_TICK, lambda: [step.refresh() for step in steps])
