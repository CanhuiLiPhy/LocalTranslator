"""
中文：托管 llama-server 子进程的生命周期（启动、健康检查、关闭）。

Lifecycle management for the local ``llama-server`` process.

The application owns the inference server rather than expecting the user to
start it by hand: :class:`LlamaServerSupervisor` launches it on demand, waits
until the model is actually loaded, and shuts it down cleanly on exit.

If a healthy server is already listening on the configured port — because the
user started one themselves, or a previous app instance left it running — the
supervisor adopts it instead of spawning a competing process, and will not kill
a process it did not start.
"""

from __future__ import annotations

import logging
import socket
import subprocess
import threading
import time
from pathlib import Path

import httpx

from local_translator.config import AppPaths, LocalModelSettings
from local_translator.llm.downloader import llama_server_path

logger = logging.getLogger(__name__)


class LlamaServerError(RuntimeError):
    """中文：llama-server 启动或健康检查失败。 / Raised when llama-server cannot be started or stays unhealthy."""


class LlamaServerSupervisor:
    """
    中文：管理本地 llama-server 进程。

    Supervise a local ``llama-server`` instance exposing an OpenAI-compatible
    API for the Hy-MT2 translation model.

    Attributes:
        paths (AppPaths): Path registry, used to locate binaries, weights and logs.
        settings (LocalModelSettings): Server and sampling configuration.
    """

    def __init__(self, paths: AppPaths, settings: LocalModelSettings):
        self.paths = paths
        self.settings = settings
        self._process: subprocess.Popen | None = None
        self._log_handle = None
        self._adopted = False
        # Startup is serialised. Translation, image recognition and the status
        # page can all ask for the server at once, and two threads that both
        # find it down will otherwise both spawn one — the second then dies on
        # a bind error and reports it as a startup failure to whichever feature
        # asked last.
        # 启动过程串行化。翻译、图片识别与状态页可能同时请求服务，两个线程若都发现
        # 它没起来就会各启一个，第二个随即因端口占用而退出，并把这个错误当成启动
        # 失败报给最后发起请求的那个功能。
        self._lock = threading.RLock()
        self._write_pid()

    # ------------------------------------------------------------------ #
    # Introspection / 状态查询
    # ------------------------------------------------------------------ #

    @property
    def base_url(self) -> str:
        """
        中文：OpenAI 兼容接口的 base URL。

        Returns:
            str: e.g. ``http://127.0.0.1:18080/v1``.
        """
        return self.settings.base_url

    @property
    def owns_process(self) -> bool:
        """
        中文：当前进程是否由本 supervisor 启动（决定是否可以关闭它）。

        Returns:
            bool: ``True`` when this supervisor spawned the running server and
                is therefore allowed to terminate it.
        """
        return self._process is not None and not self._adopted

    def model_path(self) -> Path:
        """
        中文：返回当前速度模式对应的 GGUF 权重路径。

        Resolve the weight file for the active speed mode.

        Returns:
            Path: The configured translation weights.
        """
        return self.paths.find_model(self.settings.model_file)

    def draft_model_path(self) -> Path | None:
        """
        中文：返回可用的投机解码 draft 模型路径，未配置或文件缺失时返回 None。

        Resolve the draft model used for speculative decoding.

        Returns:
            Path | None: Path to the draft weights, or ``None`` when speculative
                decoding is switched off or the file has not been downloaded.
        """
        if not self.settings.speculative:
            return None
        name = self.settings.draft_model_file
        if not name:
            return None
        path = self.paths.find_model(name)
        return path if path.exists() else None

    def busy(self, timeout: float = 2.0) -> bool | None:
        """
        中文：问推理服务自己是否正在处理请求（谁在用都算）。

        Ask the inference server whether any slot is decoding.

        The job manager only knows about work this process submitted. A CLI
        translation running beside an open window, or a second window, drives
        the same server through the same port and is invisible to it — and an
        idle watchdog that trusts only its own manager would unload a model
        somebody else is in the middle of using. llama.cpp answers this directly.

        Args:
            timeout (float): Seconds to wait for the reply.

        Returns:
            bool | None: ``True`` when a slot is processing, ``False`` when all
                are idle, and ``None`` when the question could not be answered —
                which callers must not read as "idle".
        """
        try:
            import httpx

            reply = httpx.get(f"http://{self.settings.host}:{self.settings.port}/slots",
                              timeout=timeout)
            if reply.status_code != 200:
                return None
            slots = reply.json()
            if not isinstance(slots, list):
                return None
            return any(slot.get("is_processing") for slot in slots)
        except Exception:  # noqa: BLE001
            return None

    def health(self, timeout: float = 2.0) -> bool:
        """
        中文：检查服务是否已就绪（模型加载完成）。

        Probe the server's ``/health`` endpoint.

        llama-server answers 200 only once weights are loaded and it is ready to
        accept completions; while loading it returns 503. Both connection
        failures and non-200 responses are reported as unhealthy.

        Args:
            timeout (float): HTTP timeout in seconds.

        Returns:
            bool: ``True`` if the server is up and ready.
        """
        url = f"http://{self.settings.host}:{self.settings.port}/health"
        try:
            # trust_env=False keeps the probe off any configured HTTP proxy.
            # A loopback request routed through a proxy fails, and a health
            # probe that answers "down" for a server that is up is worse than
            # no probe at all: it makes the supervisor start a second one.
            # trust_env=False 让探测绕开代理设置。经代理发出的回环请求必然失败，而
            # 把已在运行的服务报成「未启动」比不探测更糟——它会让 supervisor 再起一个。
            resp = httpx.get(url, timeout=timeout, trust_env=False)
            return resp.status_code == 200
        except httpx.HTTPError as exc:
            logger.debug("Health probe of %s failed: %s", url, exc)
            return False

    def port_taken(self) -> bool:
        """
        中文：判断配置端口上是否已有进程在监听。

        Report whether anything is already listening on the configured port.

        Distinct from :meth:`health`, and the distinction is the whole point: a
        server that is loading its weights holds the port while answering 503,
        so it is occupied but not yet healthy. Spawning during that window is
        what produces "couldn't bind HTTP server socket".

        Returns:
            bool: ``True`` when a TCP connection to the port succeeds.
        """
        try:
            with socket.create_connection((self.settings.host, self.settings.port),
                                          timeout=1.0):
                return True
        except OSError:
            return False

    def is_running(self) -> bool:
        """
        中文：判断被管理的子进程是否仍存活。

        Returns:
            bool: ``True`` if a spawned process exists and has not exited.
        """
        return self._process is not None and self._process.poll() is None

    # ------------------------------------------------------------------ #
    # Lifecycle / 生命周期
    # ------------------------------------------------------------------ #

    def _build_command(self, binary: Path, model: Path) -> list[str]:
        """
        中文：拼装 llama-server 启动命令行。

        Assemble the llama-server command line.

        Note that llama.cpp's ``-c`` is the *total* KV context shared by all
        parallel slots, so each of the ``-np`` slots effectively gets
        ``n_ctx / n_parallel`` tokens. Sampling parameters are deliberately not
        passed here: they are sent per-request by the translator, so changing
        them in the UI does not require restarting the server.

        Args:
            binary (Path): Path to the ``llama-server`` executable.
            model (Path): Path to the GGUF weights.

        Returns:
            list[str]: Argument vector suitable for :class:`subprocess.Popen`.
        """
        s = self.settings
        cmd = [
            str(binary),
            "-m", str(model),
            "--host", s.host,
            "--port", str(s.port),
            "-c", str(s.n_ctx),
            "-np", str(s.n_parallel),
            "-ngl", str(s.n_gpu_layers),
            # Use the chat template embedded in the GGUF so Hy-MT2 receives the
            # prompt format it was trained on.
            # 使用 GGUF 内置聊天模板，保证提示词格式与训练一致。
            "--jinja",
            # The log goes to a file, so ANSI colour codes would only add noise.
            # 日志写入文件，彩色转义序列只会变成乱码。
            "--log-colors", "off",
        ]

        # Memory settings. Each is only added when configured, so an
        # installation that never asked for them gets exactly the command line
        # it got before.
        # 内存相关参数。仅在配置了才追加，没开过的安装拿到的命令行与从前完全一致。
        if s.flash_attention:
            cmd += ["-fa", "on"]
        if s.kv_cache_type:
            # The V cache can only be quantised when flash attention is on;
            # quantising K alone is still worth having, so only V is gated.
            # V 缓存必须在 flash attention 打开时才能量化；单独量化 K 仍有收益，
            # 因此只对 V 做门控。
            cmd += ["-ctk", s.kv_cache_type]
            if s.flash_attention:
                cmd += ["-ctv", s.kv_cache_type]
            else:
                logger.info("Quantising only the K cache; -ctv needs flash attention")
        if s.batch_size:
            cmd += ["-b", str(s.batch_size)]
        if s.ubatch_size:
            cmd += ["-ub", str(s.ubatch_size)]
        # llama-server's prompt cache defaults to a ceiling of 8 GiB and fills
        # it at roughly 5 MB per prompt it has not seen before. Every paragraph
        # is a prompt it has not seen before, so the store grows all run and is
        # never read back. Passed always, including the zero, because the
        # default is the thing being corrected.
        # llama-server 的提示词缓存默认上限 8 GiB，每遇到一个没见过的提示词就占用约
        # 5 MB。而每一段都是它没见过的提示词，于是这个仓库整场翻译一直在涨、却从来
        # 没被读回来过。此项一律下发，包括 0——要纠正的正是那个默认值。
        cmd += ["--cache-ram", str(max(0, s.cache_ram_mib))]

        # Speculative decoding, only when the draft weights are actually present.
        # A missing draft file must not stop the server from starting.
        # 仅在 draft 权重确实存在时启用投机解码；缺文件不应导致服务起不来。
        draft = self.draft_model_path()
        if draft is not None:
            cmd += ["-md", str(draft)]
            logger.info("Speculative decoding enabled with draft model %s", draft.name)

        return cmd

    def start(self, wait_timeout: float = 300.0) -> None:
        """
        中文：启动 llama-server 并等待其就绪；若端口上已有健康服务则直接复用。

        Start the inference server and block until it reports healthy.

        If a healthy server is already listening on the configured port it is
        adopted rather than duplicated, and this supervisor will not stop it.

        Args:
            wait_timeout (float): Seconds to wait for the model to finish
                loading. A 7B Q6_K model typically needs a few seconds from
                warm page cache and up to a minute on first load.

        Raises:
            LlamaServerError: If the binary or weights are missing, the process
                exits during startup, or it never becomes healthy in time.
        """
        with self._lock:
            self._start_locked(wait_timeout)

    def _start_locked(self, wait_timeout: float) -> None:
        """
        中文：``start`` 的实际实现，调用时须已持有启动锁。

        Body of :meth:`start`, called with the startup lock held.

        Args:
            wait_timeout (float): Seconds to wait for the model to load.

        Raises:
            LlamaServerError: As documented on :meth:`start`.
        """
        if self.health():
            if not self.is_running():
                logger.info("Adopting existing llama-server on port %d", self.settings.port)
                self._adopted = True
            return

        # Something holds the port without being ready yet — almost always a
        # server still mapping several gigabytes of weights. Wait for it rather
        # than launching a competitor that is guaranteed to die on bind.
        # 端口被占用但尚未就绪——几乎总是某个服务还在映射几 GB 权重。等它，而不是
        # 另起一个必然因绑定失败而退出的进程。
        if self.port_taken():
            logger.info("Port %d is already in use; waiting for it to become healthy",
                        self.settings.port)
            deadline = time.monotonic() + wait_timeout
            while time.monotonic() < deadline:
                if self.health():
                    if not self.is_running():
                        logger.info("Adopting existing llama-server on port %d",
                                    self.settings.port)
                        self._adopted = True
                    return
                if not self.port_taken():
                    break
                time.sleep(0.5)
            else:
                raise LlamaServerError(
                    f"Port {self.settings.port} is held by another process that never "
                    f"became a healthy llama-server. Stop whatever is listening on it, "
                    f"or set a different port in the settings."
                )

        binary = llama_server_path(self.paths)
        if binary is None:
            raise LlamaServerError(
                "llama-server is not installed. Run `localtrans setup` to download it."
            )

        model = self.model_path()
        if not model.exists():
            raise LlamaServerError(
                f"Model weights not found at {model}. "
                "Run `localtrans setup --quant Q6_K` to download them."
            )

        cmd = self._build_command(binary, model)
        log_path = self.paths.logs_dir / "llama-server.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        # This handle is opened for append and never rotated, so without a check
        # here the file only ever grows — 17 MB had accumulated on the machine
        # this was written on. Trimmed at start rather than only at exit,
        # because one long session can add plenty by itself.
        # 这个句柄以追加方式打开且从不轮转，没有这道检查，文件就只会一直变大——写这段话
        # 的机器上已经堆了 17 MB。放在启动时裁而不是只在退出时裁，因为单单一次长会话
        # 自己就能加不少。
        try:
            from local_translator.engine import intermediates

            intermediates.cap_log(log_path)
        except Exception:  # noqa: BLE001
            logger.debug("Could not trim the llama-server log", exc_info=True)

        logger.info("Starting llama-server: %s", " ".join(cmd))
        self._log_handle = log_path.open("a", encoding="utf-8")
        self._log_handle.write(f"\n=== start {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        self._log_handle.flush()

        # Recorded so that a later run of the application — which finds this
        # server already healthy and adopts it — can still release its memory
        # when the user quits. Without the record, adopting means never being
        # able to stop it, and several gigabytes of weights stay resident.
        # 记录下来，使本应用的后续运行——它会发现该服务健康并直接接管——在用户退出
        # 时仍能释放其内存。不记录的话，接管就意味着永远无法停止，几 GB 的权重会
        # 一直驻留。
        self._process = subprocess.Popen(
            cmd,
            stdout=self._log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        self._adopted = False

        deadline = time.monotonic() + wait_timeout
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                tail = self._read_log_tail(log_path)
                raise LlamaServerError(
                    f"llama-server exited with code {self._process.returncode} during startup.\n"
                    f"Last log lines:\n{tail}"
                )
            if self.health():
                logger.info("llama-server ready at %s", self.base_url)
                return
            time.sleep(0.5)

        self._stop_locked()
        raise LlamaServerError(
            f"llama-server did not become healthy within {wait_timeout:.0f}s. "
            f"See {log_path} for details."
        )

    def ensure_running(self, wait_timeout: float = 300.0) -> None:
        """
        中文：确保服务可用；已就绪则立即返回，否则按配置决定是否自动启动。

        Guarantee that a healthy server is reachable before translation begins.

        Args:
            wait_timeout (float): Seconds to wait for startup.

        Raises:
            LlamaServerError: If the server is down and ``auto_start`` is
                disabled, or if starting it fails.
        """
        if self.health():
            return
        with self._lock:
            # Re-probe under the lock: while this thread waited for it, another
            # may have started the server already.
            # 取得锁后重新探测：等锁期间可能已有别的线程把服务起好了。
            if self.health():
                return
            if not self.settings.auto_start:
                raise LlamaServerError(
                    f"No llama-server reachable at {self.base_url} and auto-start is disabled."
                )
            self._start_locked(wait_timeout)

    def restart_for_model_change(self, wait_timeout: float = 300.0) -> None:
        """
        中文：切换模型后重启服务（llama-server 一次只能加载一个模型）。

        Restart the server so a model change takes effect.

        ``llama-server`` serves exactly one model per process, so switching
        between the standard and fast models means replacing the process. An
        adopted server is left alone — it belongs to whoever started it.

        Args:
            wait_timeout (float): Seconds to wait for the new model to load.

        Raises:
            LlamaServerError: If the replacement server fails to come up.
        """
        with self._lock:
            if self._adopted:
                logger.info("Not restarting an adopted llama-server for a model change")
                return
            self._stop_locked()
            self._start_locked(wait_timeout)

    def stop(self, timeout: float = 15.0, include_adopted: bool = False) -> None:
        """
        中文：优雅关闭推理服务；``include_adopted`` 为真时也关闭本应用此前留下的实例。

        Terminate the inference server, escalating to SIGKILL if it does not exit.

        A server this process started is always stopped. One it merely adopted is
        left alone by default, since stopping something we did not start would
        surprise whoever did. That default is wrong for an explicit quit, though:
        the adopted server is almost always this same application's, left over
        from a previous run, and leaving it resident means the user quits the app
        and several gigabytes of weights stay in memory with no obvious way to
        release them. ``include_adopted`` covers that case, and is safe because
        it only ever stops a process this application recorded having started.

        Args:
            timeout (float): Seconds to wait for graceful shutdown before killing.
            include_adopted (bool): Also stop a server adopted from an earlier
                run of this application.
        """
        with self._lock:
            self._stop_locked(timeout, include_adopted)

    def _stop_locked(self, timeout: float = 15.0, include_adopted: bool = False) -> None:
        """
        中文：``stop`` 的实际实现，调用时须已持有启动锁。

        Body of :meth:`stop`, called with the startup lock held.

        Args:
            timeout (float): Seconds to wait for graceful shutdown before killing.
            include_adopted (bool): Also stop a server adopted from an earlier run.
        """
        if self._adopted:
            if include_adopted:
                self._stop_recorded(timeout)
            else:
                logger.debug("Not stopping adopted llama-server")
            return
        proc = self._process
        if proc is None:
            return

        if proc.poll() is None:
            logger.info("Stopping llama-server (pid %d)", proc.pid)
            proc.terminate()
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                logger.warning("llama-server did not exit in %.0fs, killing", timeout)
                proc.kill()
                proc.wait(timeout=5)

        self._process = None
        self._clear_pid()
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None

    def _pid_path(self) -> Path:
        """中文：llama-server 的 PID 文件路径。 / Path of the inference server's PID file."""
        return self.paths.data_dir / "llama-server.pid"

    def _write_pid(self) -> None:
        """中文：记录本次启动的 llama-server PID。 / Record the spawned server's PID."""
        if self._process is None:
            return
        try:
            self._pid_path().parent.mkdir(parents=True, exist_ok=True)
            self._pid_path().write_text(str(self._process.pid), encoding="utf-8")
        except OSError:
            logger.debug("Could not record llama-server pid", exc_info=True)

    def _clear_pid(self) -> None:
        """中文：删除 PID 文件。 / Remove the PID file."""
        try:
            self._pid_path().unlink(missing_ok=True)
        except OSError:
            logger.debug("Could not clear llama-server pid", exc_info=True)

    def _stop_recorded(self, timeout: float) -> None:
        """
        中文：按 PID 文件停止此前运行留下的推理服务。

        Stop the inference server recorded by an earlier run.

        Args:
            timeout (float): Seconds to wait for a clean exit before killing.
        """
        import os
        import signal
        import time

        try:
            pid = int(self._pid_path().read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            logger.debug("No recorded llama-server pid to stop")
            return

        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            self._clear_pid()
            return

        logger.info("Stopping adopted llama-server (pid %d)", pid)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                os.kill(pid, 0)
            except OSError:
                self._clear_pid()
                return
            time.sleep(0.3)

        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass
        self._clear_pid()

    @staticmethod
    def _read_log_tail(path: Path, lines: int = 20) -> str:
        """
        中文：读取日志文件末尾若干行，用于错误提示。

        Read the last lines of a log file for inclusion in an error message.

        Args:
            path (Path): Log file path.
            lines (int): Number of trailing lines to return.

        Returns:
            str: The trailing lines, or a placeholder if unreadable.
        """
        try:
            content = path.read_text(encoding="utf-8", errors="replace").splitlines()
            return "\n".join(content[-lines:])
        except OSError:
            return "(log unavailable)"

    def __enter__(self) -> LlamaServerSupervisor:
        """中文：进入上下文时确保服务就绪。 / Ensure the server is running on context entry."""
        self.ensure_running()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """中文：退出上下文时关闭服务。 / Stop the managed server on context exit."""
        self.stop()
