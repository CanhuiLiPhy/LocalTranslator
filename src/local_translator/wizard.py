"""
中文：首次运行的交互式配置向导——问清楚版本、语言、云端 API 与本地模型，然后落盘。

Interactive first-run configuration.

Everything this asks about could be typed into ``settings.json`` by hand, and
for a long time that was the only way. The reason for a wizard is that the two
pieces a new installation actually needs — an OpenAI-compatible endpoint and its
key — are exactly the two the application deliberately does *not* ship: the
endpoint has no sensible default, and the key must never touch a file in the
repository. Asking for both once, storing the key in the system keychain and
proving the pair works before writing anything is the shortest honest path from
a fresh clone to a translated page.

The wizard is re-runnable. It reads the current settings, offers them as the
default for every question, and leaves anything the reader accepts untouched, so
running it again to change one endpoint does not reset the rest.
"""

from __future__ import annotations

import logging

import typer

from local_translator import edition
from local_translator.config import LANGUAGES, get_paths, get_settings, save_settings
from local_translator.secrets import get_api_key, key_source, set_api_key

logger = logging.getLogger(__name__)

# Endpoints people are most likely to be pointing at, offered as a menu so the
# common cases need no typing and the shape of the answer is unambiguous. None
# of these is a default: the reader picks one or types their own.
# 常见的几个端点，做成菜单让常见情况免于手输，也让「该填什么形状的东西」一目了然。
# 这里没有任何默认值：要么选一个，要么自己填。
KNOWN_ENDPOINTS: tuple[tuple[str, str, str], ...] = (
    ("DeepSeek", "https://api.deepseek.com/v1", "deepseek-chat"),
    ("OpenAI", "https://api.openai.com/v1", "gpt-4o-mini"),
    ("Moonshot Kimi", "https://api.moonshot.cn/v1", "moonshot-v1-8k"),
    ("智谱 GLM", "https://open.bigmodel.cn/api/paas/v4", "glm-4-flash"),
    ("阿里云百炼", "https://dashscope.aliyuncs.com/compatible-mode/v1", "qwen-plus"),
    ("Ollama（本机）", "http://127.0.0.1:11434/v1", "qwen2.5:7b"),
)

_PROBE_PROMPT = "Reply with the single word OK."


def _account_for(display: str) -> str:
    """
    中文：由显示名推导一个合法的钥匙串账户名。

    Derive a keychain account name from a provider's display name.

    Args:
        display (str): Human-readable provider name, possibly with spaces or
            non-ASCII characters.

    Returns:
        str: A lower-case ASCII slug, falling back to ``"cloud"`` when the name
            has no usable characters — a Chinese provider name would otherwise
            reduce to the empty string.
    """
    slug = "".join(ch if ch.isascii() and ch.isalnum() else "-" for ch in display)
    slug = "-".join(part for part in slug.lower().split("-") if part)
    return slug or "cloud"


def _rule(title: str) -> None:
    """
    中文：打印一节标题。

    Print a section heading.

    Args:
        title (str): Heading text.
    """
    typer.echo("")
    typer.secho(f"── {title} " + "─" * max(0, 56 - len(title)), fg=typer.colors.BRIGHT_BLACK)


def _ok(message: str) -> None:
    """中文：打印一条成功信息。 / Print a success line."""
    typer.secho(f"  ✓ {message}", fg=typer.colors.GREEN)


def _warn(message: str) -> None:
    """中文：打印一条警告信息。 / Print a warning line."""
    typer.secho(f"  ! {message}", fg=typer.colors.YELLOW)


def _fail(message: str) -> None:
    """中文：打印一条失败信息。 / Print a failure line."""
    typer.secho(f"  ✗ {message}", fg=typer.colors.RED)


def _choose_endpoint() -> tuple[str, str, str] | None:
    """
    中文：让用户从常见端点里选一个，或选择手动填写。

    Offer the known endpoints as a numbered menu.

    Returns:
        tuple[str, str, str] | None: ``(display name, base URL, model)`` for the
            chosen provider, or ``None`` when the reader wants to type their own.
    """
    typer.echo("  常见的 OpenAI 兼容端点：")
    for index, (name, url, model) in enumerate(KNOWN_ENDPOINTS, start=1):
        typer.echo(f"    {index}. {name:<14} {url}  ({model})")
    typer.echo(f"    {len(KNOWN_ENDPOINTS) + 1}. 其它（手动填写）")

    choice = typer.prompt("  选择", default=str(len(KNOWN_ENDPOINTS) + 1))
    try:
        index = int(choice)
    except ValueError:
        return None
    if 1 <= index <= len(KNOWN_ENDPOINTS):
        return KNOWN_ENDPOINTS[index - 1]
    return None


def _probe_api(base_url: str, model: str, key: str, timeout: float = 30.0) -> tuple[bool, str]:
    """
    中文：用一次最小请求验证端点、模型与密钥是否匹配。

    Send one minimal chat request to prove the endpoint works.

    A wizard that writes settings without checking them merely moves the failure
    to the first translation, where it surfaces as a stack trace inside a worker
    subprocess instead of a sentence the reader can act on.

    Args:
        base_url (str): OpenAI-compatible base URL, ending in ``/v1``.
        model (str): Model name to request.
        key (str): API key. Never logged.
        timeout (float): Seconds to wait for the response.

    Returns:
        tuple[bool, str]: Whether the call succeeded, and a message to show —
            the model's reply on success, or the error on failure.
    """
    try:
        from openai import OpenAI

        client = OpenAI(api_key=key, base_url=base_url, timeout=timeout, max_retries=0)
        reply = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": _PROBE_PROMPT}],
            max_tokens=16,
            temperature=0.0,
        )
        text = (reply.choices[0].message.content or "").strip()
        return True, text.replace("\n", " ")[:80] or "(empty reply)"
    except Exception as exc:  # noqa: BLE001
        # The exception text can carry the URL but never the key: the OpenAI
        # client redacts credentials in its own repr.
        # 异常文本可能带 URL，但不会带密钥：OpenAI 客户端自己会遮蔽凭证。
        return False, f"{type(exc).__name__}: {exc}"


def _configure_api(settings) -> bool:
    """
    中文：配置云端 API——端点、模型、密钥，并当场验证。

    Ask for the cloud endpoint and verify it.

    Args:
        settings (Settings): Live settings object, mutated in place.

    Returns:
        bool: ``True`` when a working endpoint was configured.
    """
    api = settings.api_model
    _rule("云端 API")

    if api.base_url and api.model_name:
        typer.echo(f"  当前：{api.display_name}  {api.base_url}  ({api.model_name})")
        if not typer.confirm("  重新配置？", default=False):
            return bool(get_api_key(api.keyring_account))

    picked = _choose_endpoint()
    if picked:
        display, base_url, model = picked
        # The menu fixes the URL; only the model is worth confirming, since one
        # endpoint serves several and the cheap one is rarely the default.
        # 菜单已经确定了 URL；只有模型值得再确认一次——同一个端点有好几个模型，
        # 便宜的那个通常不是默认。
        model = typer.prompt("  模型名", default=model).strip()
    else:
        display = typer.prompt("  名称（仅用于界面显示）", default=api.display_name or "Cloud model")
        base_url = typer.prompt("  Base URL（OpenAI 兼容，通常以 /v1 结尾）",
                                default=api.base_url or "").strip().rstrip("/")
        model = typer.prompt("  模型名", default=api.model_name or "").strip()

    account = typer.prompt("  钥匙串账户名（用于保存密钥）",
                           default=api.keyring_account or _account_for(display))

    if not base_url or not model:
        _warn("端点或模型名为空，跳过云端配置")
        return False

    source = key_source(account)
    key = get_api_key(account)
    if key and not typer.confirm(f"  已在 {source} 找到该账户的密钥，换一个？", default=False):
        pass
    else:
        # hide_input keeps the key off the screen; it is passed straight to the
        # keychain and never written to settings.json or to a log.
        # hide_input 让密钥不回显；它直接进钥匙串，不写进 settings.json，也不进日志。
        key = typer.prompt("  API key（输入不回显）", hide_input=True).strip()
        if not key:
            _warn("未输入密钥，跳过云端配置")
            return False

    typer.echo("  正在验证…")
    ok, message = _probe_api(base_url, model, key)
    if not ok:
        _fail(f"验证失败：{message}")
        if not typer.confirm("  仍然保存这份配置？", default=False):
            return False
    else:
        _ok(f"端点可用，模型回复：{message}")

    if source != "env" and source != "file":
        try:
            set_api_key(account, key)
            _ok(f"密钥已存入系统钥匙串（账户 {account}），不会写入任何文件")
        except RuntimeError as exc:
            _fail(str(exc))
            return False

    api.enabled = True
    api.display_name = display
    api.base_url = base_url
    api.model_name = model
    api.keyring_account = account
    return ok


def _configure_languages(settings) -> None:
    """
    中文：配置源语言与目标语言。

    Ask for the language pair.

    Args:
        settings (Settings): Live settings object, mutated in place.
    """
    _rule("语言")
    translation = settings.translation
    typer.echo(f"  可用语种共 {len(LANGUAGES)} 个，常用：en、zh、ja、ko、fr、de、es、ru")
    lang_in = typer.prompt("  源语言", default=translation.lang_in).strip()
    lang_out = typer.prompt("  目标语言", default=translation.lang_out).strip()
    for code, name in (("源语言", lang_in), ("目标语言", lang_out)):
        if name not in LANGUAGES:
            _warn(f"{code} {name!r} 不在已知列表里，仍会照原样使用")
    translation.lang_in = lang_in
    translation.lang_out = lang_out
    _ok(f"{lang_in} → {lang_out}")


def _configure_backend(settings, api_ready: bool) -> None:
    """
    中文：选择默认翻译后端（轻量版只有云端一种）。

    Choose the default translation backend.

    Args:
        settings (Settings): Live settings object, mutated in place.
        api_ready (bool): Whether a working cloud endpoint was configured.
    """
    available = edition.backends()
    if len(available) == 1:
        settings.translation.backend = available[0]
        return

    _rule("默认后端")
    size = (edition.local_model() or {}).get("size_note", "一份")
    names = {
        "local": f"本地模型 — 离线，不产生费用，需要先下载{size}权重",
        "api": "云端 API — 质量稳定，按 token 计费",
        "hybrid": "混合 — 本地与云端并行，最快，云端用量约为纯 API 的三分之一",
    }
    for index, key in enumerate(available, start=1):
        typer.echo(f"    {index}. {names[key]}")

    current = settings.translation.backend
    default = str(available.index(current) + 1) if current in available else "1"
    choice = typer.prompt("  选择", default=default)
    try:
        backend = available[int(choice) - 1]
    except (ValueError, IndexError):
        backend = available[0]

    if backend in ("api", "hybrid") and not api_ready:
        _warn(f"「{backend}」需要可用的云端 API，但刚才没有配置成功；先设为 local")
        backend = "local"
    settings.translation.backend = backend
    _ok(f"默认后端：{backend}")


def _offer_local_model(settings) -> None:
    """
    中文：完整版下询问是否现在下载本地模型权重。

    Offer to download the local model weights, on the full edition only.

    Args:
        settings (Settings): Live settings object, read for the chosen quant.
    """
    if edition.is_lite():
        return

    from local_translator.llm.downloader import find_installed_models, llama_server_path

    paths = get_paths()
    _rule("本地模型")
    if llama_server_path(paths) and find_installed_models(paths):
        _ok("llama.cpp 与模型权重都已安装")
        return

    spec = edition.local_model() or {}
    typer.echo(f"  本地模型需要 llama.cpp 运行时与 {spec.get('size_note', '一份')} 的 "
               f"Hy-MT2 权重（{spec.get('repo', '')}）。")
    if settings.translation.backend == "api":
        typer.echo("  当前默认后端是云端 API，不装也能正常使用。")
    if not typer.confirm("  现在下载？", default=settings.translation.backend != "api"):
        typer.echo("  跳过。之后运行 `localtrans setup` 即可。")
        return

    # Calling the CLI command object would pass typer's Option descriptors as
    # values; the installer functions underneath it take plain arguments.
    # 直接调 CLI 命令对象会把 typer 的 Option 描述符当成值传进去；下面这些
    # 安装函数接受的是普通参数。
    from local_translator.cli import _progress_bar
    from local_translator.llm.downloader import install_llama_cpp, install_model

    try:
        callback, close = _progress_bar("llama.cpp")
        try:
            install_llama_cpp(paths, progress=callback)
        finally:
            close()

        callback, close = _progress_bar("Hy-MT2")
        try:
            weights = install_model(paths, spec["quant"], repo=spec["repo"],
                                    progress=callback)
        finally:
            close()
        settings.local_model.model_file = weights.name
        _ok(f"本地模型已安装：{weights.name}")
    except Exception as exc:  # noqa: BLE001
        _fail(f"下载失败：{exc}。之后可重新运行 `localtrans setup`，下载是断点续传的。")


def _offer_reader() -> None:
    """
    中文：询问是否下载 pdf.js（双语阅读器所需，约 10 MB）。

    Offer to download the pdf.js viewer the bilingual reader needs.
    """
    from local_translator.web import pdfjs

    _rule("阅读器")
    if pdfjs.is_installed():
        _ok("pdf.js 已安装")
        return
    typer.echo("  双语阅读器用本地自带的 pdf.js 渲染，不依赖 CDN，约 10 MB。")
    if not typer.confirm("  现在下载？", default=True):
        typer.echo("  跳过。之后运行 `localtrans setup-reader` 即可。")
        return
    try:
        target = pdfjs.install()
        _ok(f"pdf.js 已安装到 {target}")
    except Exception as exc:  # noqa: BLE001
        _fail(f"下载失败：{exc}。之后可重新运行 `localtrans setup-reader`。")


def run(non_interactive: bool = False) -> int:
    """
    中文：跑完整个配置向导，返回退出码。

    Run the whole wizard.

    Args:
        non_interactive (bool): Only report what is and is not configured, and
            change nothing. Used by the installer to decide whether the reader
            still has to be asked anything.

    Returns:
        int: ``0`` when the installation is ready to translate something,
            ``1`` when it still needs configuring.
    """
    paths = get_paths().ensure()
    settings = get_settings()

    typer.secho(f"Local Translator · {edition.display_name()}", bold=True)
    typer.echo(f"配置文件：{paths.settings_path}")

    if non_interactive:
        api = settings.api_model
        ready = bool(api.base_url and api.model_name and get_api_key(api.keyring_account))
        if ready:
            _ok(f"云端 API 已配置（{api.display_name}）")
        else:
            _warn("云端 API 尚未配置")
        return 0 if ready or not edition.is_lite() else 1

    api_ready = _configure_api(settings)
    _configure_languages(settings)
    _configure_backend(settings, api_ready)
    _offer_local_model(settings)
    _offer_reader()

    save_settings(settings)
    _rule("完成")
    _ok(f"配置已保存到 {paths.settings_path}")
    typer.echo("  启动：localtrans serve      （或双击「本地翻译」应用）")
    typer.echo("  自检：localtrans doctor")
    return 0
