"""
中文：空闲时把本地模型从内存/显存里卸掉，有任务时再自动装回来。

Unload the local model while nothing is being translated.

A desktop application is open far longer than it is working. llama-server holds
the weights and its Metal buffers for as long as it runs — about 1.4 GB for the
1.8B and roughly ten times that for the 7B at full concurrency — so an app left
open all afternoon keeps that memory away from everything else on the machine,
and keeps the GPU's allocations alive, for no benefit at all.

The watchdog stops the server once no translation has run for a configured
number of seconds. Nothing else changes: :meth:`TranslatorService.ensure_model_ready`
already starts the server before a local job, so the next translation simply
pays the load time again — measured at about **9 seconds for the 1.8B**, against
one to two minutes for the paper that follows it.

It deliberately does not unload while work is queued, only while there is none,
so a batch of ten documents is not made to reload the model between each.

Off by default. The full edition keeps its server resident, which is the right
choice on a machine that has the memory to spare; the mini edition turns this on
because it exists for machines that do not.
"""

from __future__ import annotations

import logging
import threading
import time

logger = logging.getLogger(__name__)

# How often to look. The check is two cheap in-process reads, so this only
# bounds how late an unload can be, not how much it costs: an unload lands
# somewhere between the threshold and the threshold plus one tick — at the
# default that is 10 to 20 seconds after the last job.
# 检查间隔。每次检查只是两次进程内读取，因此它只决定卸载最晚发生在何时，不构成开销：
# 实际卸载落在「阈值」到「阈值 + 一个间隔」之间——按默认值即最后一个任务结束后
# 10 到 20 秒。
_TICK = 10.0

# The smallest threshold the watchdog will honour, zero excepted. A threshold at
# or below the tick means one poll landing on an idle moment unloads the model,
# and the poll can land in the gap between the server reporting ready and the
# first paragraph reaching it. Anything positive but smaller is raised to this.
# 看门狗认可的最小阈值（0 除外，0 表示关闭）。阈值一旦低于或等于轮询间隔，只要有一次轮询
# 落在空闲瞬间就会卸载模型，而这次轮询完全可能落在「服务报告就绪」与「第一个段落到达」
# 之间的空隙里。任何正数但更小的取值都会被抬到这个下限。
_FLOOR = 60.0


def _clamped(seconds: int) -> int:
    """
    中文：把小于下限的正阈值抬到下限；0 表示关闭看门狗，原样保留。

    Clamp a positive idle threshold up to :data:`_FLOOR`.

    Args:
        seconds (int): Threshold as configured, in seconds. ``0`` or negative
            disables the watchdog and is returned unchanged.

    Returns:
        int: ``seconds`` when it is zero, negative, or already at least
            :data:`_FLOOR`; otherwise :data:`_FLOOR` as an integer.
    """
    if seconds <= 0 or seconds >= _FLOOR:
        return seconds
    logger.info("Idle threshold %ds is below the %ds floor; using the floor",
                seconds, int(_FLOOR))
    return int(_FLOOR)


class IdleUnloader:
    """
    中文：空闲计时器——超过阈值且无任务在跑/排队时停掉推理服务。

    Stops the inference server after a period with no translation.

    Attributes:
        seconds (int): Idle seconds before unloading. 0 disables the watchdog.
    """

    def __init__(self, service, seconds: int):
        """
        中文：绑定到一个 TranslatorService。

        Bind the watchdog to a service.

        Args:
            service (TranslatorService): Service owning the supervisor and the
                job manager. Held rather than copied so a settings change that
                replaces either is picked up.
            seconds (int): Idle seconds before unloading; 0 disables.
        """
        self._service = service
        self.seconds = seconds
        self._last_busy = time.monotonic()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    @property
    def seconds(self) -> int:
        """中文：空闲阈值（秒）。 / Idle threshold in seconds."""
        return self._seconds

    @seconds.setter
    def seconds(self, value: int) -> None:
        """
        中文：设置空闲阈值，低于下限的正数会被抬到下限。

        Set the idle threshold, clamping an unsafe value up to :data:`_FLOOR`.

        The settings page binds to this through the service, so clamping here
        rather than at the call sites is what makes an unsafe value
        unreachable however it arrives.

        Args:
            value (int): Threshold in seconds; ``0`` disables the watchdog.
        """
        self._seconds = _clamped(value)

    def start(self) -> None:
        """
        中文：启动后台看门狗线程（seconds<=0 则什么也不做）。

        Start the watchdog thread, unless it is disabled or already running.
        """
        if self.seconds <= 0 or self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="idle-unloader",
                                        daemon=True)
        self._thread.start()
        logger.info("Idle unload active: the local model is released after %ds idle",
                    self.seconds)

    def stop(self) -> None:
        """
        中文：停止看门狗线程。

        Stop the watchdog thread and wait briefly for it to finish.
        """
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=_TICK + 1.0)

    def note_activity(self) -> None:
        """
        中文：记录一次活动，把空闲计时清零。

        Reset the idle timer, called whenever work starts.
        """
        self._last_busy = time.monotonic()

    def _busy(self) -> bool:
        """
        中文：判断模型当前是否有人在用——本进程的队列，以及推理服务自己的槽位。

        Report whether anything is using the model.

        Two questions, because either alone is wrong. The job manager knows
        about this process's queue, which is what stops a batch reloading the
        model between documents. The inference server knows about every client,
        which is what stops this window unloading a model a command-line run is
        using.

        Returns:
            bool: ``True`` while a job is running or queued here, or while the
                server reports a slot decoding, or while either question could
                not be answered.
        """
        try:
            snapshot = self._service.manager.snapshot()
        except Exception:  # noqa: BLE001
            # A manager that cannot be read is not evidence of idleness.
            # 读不到管理器状态不等于空闲。
            return True
        if snapshot.get("running") or snapshot.get("queued"):
            return True

        # The manager only knows about work this process submitted. A CLI
        # translation running beside an open window drives the same server and
        # is invisible here, so ask the server itself; `None` means it could not
        # be asked, which is not the same as idle.
        # 管理器只知道本进程提交的任务。与窗口并行运行的命令行翻译驱动的是同一个
        # 服务，在这里完全看不见，因此再问服务本身；None 表示问不到，那不等于空闲。
        serving = self._service.supervisor.busy()
        return serving is not False

    def _run(self) -> None:
        """中文：看门狗主循环。 / The watchdog loop."""
        while not self._stop.wait(_TICK):
            if self.seconds <= 0:
                continue
            if self._busy():
                self._last_busy = time.monotonic()
                continue
            if time.monotonic() - self._last_busy < self.seconds:
                continue

            supervisor = self._service.supervisor
            if not supervisor.is_running():
                # Already down; keep the timer from firing repeatedly.
                # 已经停了，避免反复触发。
                self._last_busy = time.monotonic()
                continue
            try:
                # include_adopted, because the server this window is talking to
                # was very often started by a command-line run that has since
                # exited. Without it the one case the watchdog exists for — a
                # model left resident with nobody to release it — is the one
                # case it never handles. It only ever stops a process this
                # application recorded starting; see LlamaServerSupervisor.stop.
                # 必须带 include_adopted：当前窗口连着的服务，往往是某次命令行翻译
                # 留下的。不带这个参数，看门狗唯一存在的意义——没人负责释放的常驻
                # 模型——恰恰是它永远处理不了的情形。它只会停掉本应用记录过自己
                # 启动的进程，见 LlamaServerSupervisor.stop。
                supervisor.stop(include_adopted=True)
                logger.info("Local model unloaded after %ds idle; it will reload "
                            "automatically on the next translation", self.seconds)
            except Exception:  # noqa: BLE001
                logger.warning("Could not unload the local model", exc_info=True)
            finally:
                self._last_busy = time.monotonic()
