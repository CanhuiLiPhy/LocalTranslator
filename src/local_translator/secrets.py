"""
中文：API 密钥读写——只经系统钥匙串，绝不落盘。

API credential storage backed by the system keychain.

API keys are never written to ``settings.json``. The settings file records only
the *account name*; the secret itself lives in the macOS Keychain (or the
platform equivalent) and is read into memory when a job needs it.

The environment variable escape hatch exists for headless or CI use, where no
keychain is available.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from local_translator.config import KEYRING_SERVICE

logger = logging.getLogger(__name__)

# Shared key directory, so one credential serves the shell, other tools and
# this application rather than being pasted separately into each.
# 共享密钥目录，让 shell、其他工具与本应用共用同一份凭证，避免到处重复粘贴。
KEY_DIR = Path.home() / ".keys"


# How long a keychain call may take before it is treated as having no answer, in
# seconds.
#
# The keychain is not a local lookup. macOS decides whether the calling binary
# is allowed to read an item, and where it is not sure it puts a dialog on the
# screen and blocks the caller until someone answers. An application whose
# signature changed — which is every rebuild of an unsigned bundle — is exactly
# that case. Called from the event loop, the block is fatal: the server binds
# its port, logs that it is ready, and then answers nothing. Observed on a
# packaged build hanging for over three minutes with no window ever opening, and
# not killable by SIGTERM because the signal handler cannot run either.
#
# So every keychain call is made on a thread this module abandons if it does not
# answer. Two seconds is far more than an allowed read needs and far less than a
# reader would wait staring at a bouncing icon.
# 一次钥匙串调用超过这个秒数就当作没有答案。
#
# 钥匙串不是本地查表。macOS 要判断调用方是否有权读取该条目，拿不准时会弹窗并阻塞调用方
# 直到有人回答。签名变过的应用——未签名包每次重新构建都算——正是这种情况。在事件循环上
# 调用，这个阻塞是致命的：服务绑好端口、写下「已就绪」，然后什么也不回答。实测某次打包
# 构建挂了三分多钟、窗口始终没打开，而且 SIGTERM 杀不掉，因为信号处理函数同样跑不起来。
#
# 因此每一次钥匙串调用都放在一个线程上，超时就放弃它。两秒远多于一次被允许的读取所需，
# 又远少于读者对着跳动的图标能忍受的时间。
_KEYCHAIN_TIMEOUT = 2.0

# The same bound for a write, which gets far longer.
#
# A read happens at startup, when the reader is not looking at a dialog and has
# no idea why the program is waiting; two seconds is already too long. A write
# happens because they just pressed 「保存密钥」, so the permission dialog is
# expected and they are about to click it. Failing at two seconds would refuse a
# key the reader was in the middle of authorising.
# 写入用的同一组界限，宽松得多。
#
# 读取发生在启动时，读者面前没有对话框，也不知道程序在等什么，两秒已经嫌久。写入是因为
# 他刚按下「保存密钥」，权限对话框是意料之中的，他正要去点。两秒就判失败，等于拒绝了一把
# 读者正在授权的密钥。
_KEYCHAIN_WRITE_TIMEOUT = 20.0


def _without_blocking(call, *args):
    """
    中文：在守护线程里执行钥匙串调用，超时就放弃，绝不阻塞调用方，也绝不拖住退出。

    Run a keychain call on a daemon thread and give up if it does not answer.

    A blocked keychain call cannot be cancelled. ``SecItemCopyMatching`` sits in
    ``mach_msg`` waiting for a SecurityAgent dialog the reader has not answered,
    and neither a signal nor a timeout reaches it: the thread stays where it is
    until the dialog is dismissed or the process dies.

    The thread therefore has to be one that nobody will ever wait on, and
    ``ThreadPoolExecutor`` is not that. ``shutdown(wait=False)`` does return at
    once, but leaving the ``with`` block calls ``shutdown(wait=True)`` on the way
    out, which joins the worker that was just abandoned — and since Python 3.9
    that worker is not a daemon. The symptom was the whole application freezing
    on the first keychain read: the window alive, the server accepting
    connections and answering none of them, and SIGTERM ineffective because the
    interpreter joins non-daemon threads at exit too.

    A bare daemon thread is joined by nobody, so the timeout is real and the
    process still exits.

    Args:
        call (Callable): The keychain function to run.
        *args: Its arguments.

    Returns:
        Any: What the call returned, or ``None`` when it raised or did not
            answer within :data:`_KEYCHAIN_TIMEOUT`. An abandoned thread is left
            to finish on its own; being a daemon, it cannot hold up exit.
    """
    answered, value = _attempt(call, *args, timeout=_KEYCHAIN_TIMEOUT)
    if not answered:
        logger.warning(
            "The system keychain did not answer within %.0fs — it is probably "
            "waiting for you to allow access. Continuing without a stored key; "
            "set %s in the environment to skip the keychain entirely.",
            _KEYCHAIN_TIMEOUT, "LOCALTRANS_API_KEY_<account>",
        )
    return value


def _attempt(call, *args, timeout: float):
    """
    中文：在守护线程里跑一次钥匙串调用，返回（是否答复, 返回值）。

    Run one keychain call on a daemon thread.

    Args:
        call (Callable): The keychain function to run.
        *args: Its arguments.
        timeout (float): Seconds to wait before abandoning the thread.

    Returns:
        tuple[bool, Any]: ``(answered, value)``. ``answered`` is ``False`` when
            the call was still running at the deadline, in which case ``value``
            is ``None`` and the thread is left to finish on its own. A call that
            raised counts as answered with a value of ``None``, because the
            keychain gave a definite reply.
    """
    import threading

    answer: list = []

    def run() -> None:
        """中文：在后台线程里跑钥匙串调用。 / Run the call off the caller's thread."""
        try:
            answer.append(call(*args))
        except Exception:  # noqa: BLE001
            logger.warning("The keychain call failed", exc_info=True)
            answer.append(None)

    thread = threading.Thread(target=run, name="keychain", daemon=True)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        return False, None
    return True, answer[0] if answer else None


def _keyring():
    """
    中文：返回可用的 keyring 模块；打包后要显式指定后端，否则它找不到。

    Return the ``keyring`` module with a working backend selected.

    keyring picks its backend by scanning installed distributions for entry
    points. A frozen application has no distributions to scan, so the scan finds
    nothing and keyring installs its ``fail`` backend — which raises on read and
    on write. The symptom is not an error the reader can act on: the settings
    page reports the key as saved, the translation goes out without one, and the
    endpoint answers 401.

    Naming the platform backend directly is what a frozen build needs, and is
    harmless when run from source because the same backend is what discovery
    would have chosen anyway.

    Returns:
        module | None: ``keyring``, ready to use, or ``None`` when it cannot be
            imported or no backend on this platform can store a secret.
    """
    try:
        import keyring
        from keyring.backends import fail
    except Exception:  # noqa: BLE001
        logger.warning("keyring is not available", exc_info=True)
        return None

    try:
        if not isinstance(keyring.get_keyring(), fail.Keyring):
            return keyring
    except Exception:  # noqa: BLE001
        pass

    for module_name, class_name in (
        ("keyring.backends.macOS", "Keyring"),
        ("keyring.backends.Windows", "WinVaultKeyring"),
        ("keyring.backends.SecretService", "Keyring"),
    ):
        try:
            module = __import__(module_name, fromlist=[class_name])
            backend = getattr(module, class_name)()
            backend.priority  # noqa: B018 — raises where the platform cannot use it
        except Exception:  # noqa: BLE001
            continue
        keyring.set_keyring(backend)
        logger.info("Selected the %s keyring backend explicitly", class_name)
        return keyring

    logger.warning("No usable keyring backend on this platform")
    return None


def _env_var_name(account: str) -> str:
    """
    中文：返回某账户对应的环境变量名。

    Derive the environment variable that can override a stored key.

    Args:
        account (str): Keychain account name, e.g. ``"deepseek"``.

    Returns:
        str: Upper-cased variable name, e.g. ``"LOCALTRANS_API_KEY_DEEPSEEK"``.
    """
    safe = "".join(ch if ch.isalnum() else "_" for ch in account).upper()
    return f"LOCALTRANS_API_KEY_{safe}"


def read_key_values(account: str) -> dict[str, str]:
    """
    中文：读出 ~/.keys/<账户>.env 里的全部 KEY=value，供需要多个值的凭证使用。

    Read every ``KEY=value`` pair from a key file.

    :func:`get_api_key` answers the common case, where a provider is identified
    by one bearer token. Some are not: Tencent Cloud signs each request with a
    SecretId *and* a SecretKey, and neither is useful alone.

    Values are returned, never logged. Callers must keep them out of messages
    and out of anything written to disk.

    Args:
        account (str): Account name, which is the file's stem.

    Returns:
        dict[str, str]: Upper-cased names mapped to their values, empty when the
            file is absent or unreadable. Surrounding quotes are stripped, as
            in the single-value reader.
    """
    path = KEY_DIR / f"{account}.env"
    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("#") or "=" not in line:
                continue
            name, _, value = line.partition("=")
            values[name.strip().upper()] = value.strip().strip("\"'")
    except OSError:
        logger.debug("Could not read the key file for %r", account, exc_info=True)
        return {}
    return values


def _read_key_file(account: str) -> str | None:
    """
    中文：从 ~/.keys/<账户>.env 读取 API_KEY。

    Read a key from the shared key directory.

    Parses a minimal ``KEY=value`` format rather than pulling in a dotenv
    dependency: the file is ours, and anything more permissive would invite
    quoting mistakes in a file whose contents must never be echoed to debug.

    Args:
        account (str): Account name, matching the file stem.

    Returns:
        str | None: The key, or ``None`` if the file or the entry is absent.
    """
    path = KEY_DIR / f"{account}.env"
    if not path.is_file():
        return None
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("#") or "=" not in line:
                continue
            name, _, value = line.partition("=")
            if name.strip().upper() == "API_KEY":
                value = value.strip().strip("\"'")
                return value or None
    except OSError:
        # Deliberately not logging the path's contents on failure.
        # 失败时刻意不记录文件内容。
        logger.warning("Could not read the key file for account %r", account)
    return None


def get_api_key(account: str) -> str | None:
    """
    中文：读取指定账户的 API 密钥——环境变量 → ~/.keys 文件 → 系统钥匙串。

    Retrieve an API key.

    Resolution order is environment variable, then the shared ``~/.keys``
    directory, then the system keychain. The environment comes first so a
    temporary override never requires mutating stored credentials; the keychain
    comes last because it is the most secure store and therefore the one a
    deliberate lookup should fall back to rather than be shadowed by.

    Args:
        account (str): Account name, used for all three lookups.

    Returns:
        str | None: The key, or ``None`` if not configured anywhere.
    """
    env = os.getenv(_env_var_name(account))
    if env:
        return env.strip()

    from_file = _read_key_file(account)
    if from_file:
        return from_file

    ring = _keyring()
    if ring is None:
        return None
    return _without_blocking(ring.get_password, KEYRING_SERVICE, account)


def key_source(account: str) -> str | None:
    """
    中文：报告密钥来自哪一层，用于界面显示（不返回密钥本身）。

    Report which layer supplied a key, without exposing its value.

    Args:
        account (str): Account name.

    Returns:
        str | None: ``"env"``, ``"file"``, ``"keychain"``, or ``None`` when no
            key is configured.
    """
    if os.getenv(_env_var_name(account)):
        return "env"
    if _read_key_file(account):
        return "file"
    ring = _keyring()
    try:
        if ring is not None and _without_blocking(ring.get_password,
                                                  KEYRING_SERVICE, account):
            return "keychain"
    except Exception:
        return None
    return None


def set_api_key(account: str, key: str) -> None:
    """
    中文：把 API 密钥写入系统钥匙串。

    Store an API key in the system keychain.

    Args:
        account (str): Keychain account name.
        key (str): The secret to store.

    Raises:
        RuntimeError: If no keychain backend is available.
    """
    ring = _keyring()
    if ring is None:
        raise RuntimeError(
            "系统钥匙串不可用，无法保存密钥。可改用环境变量 "
            f"{_env_var_name(account)} 或 ~/.keys/{account}.env。"
        )
    answered, _ = _attempt(ring.set_password, KEYRING_SERVICE, account, key,
                           timeout=_KEYCHAIN_WRITE_TIMEOUT)
    if not answered:
        raise RuntimeError(
            f"系统钥匙串在 {_KEYCHAIN_WRITE_TIMEOUT:.0f} 秒内没有响应，密钥未保存。"
            "多半是弹出了「允许访问钥匙串」的对话框在等你点确认——点完再存一次即可。"
            f"也可以改用环境变量 {_env_var_name(account)} 或 ~/.keys/{account}.env 绕开钥匙串。"
        )
    logger.info("Stored API key for account %r in the keychain", account)


def delete_api_key(account: str) -> None:
    """
    中文：从钥匙串删除指定账户的密钥。

    Remove a stored API key.

    Args:
        account (str): Keychain account name. Deleting a key that does not exist
            is not an error.
    """
    ring = _keyring()
    if ring is None:
        return
    _attempt(ring.delete_password, KEYRING_SERVICE, account,
             timeout=_KEYCHAIN_WRITE_TIMEOUT)


def has_api_key(account: str) -> bool:
    """
    中文：判断是否已配置该账户的密钥（不返回密钥本身）。

    Check whether a key is configured, without exposing its value.

    Args:
        account (str): Keychain account name.

    Returns:
        bool: ``True`` if a non-empty key is available.
    """
    return bool(get_api_key(account))
