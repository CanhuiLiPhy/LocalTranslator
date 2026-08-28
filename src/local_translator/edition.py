"""
中文：发行版本——完整版（本地 + 云端 + 混合）与轻量版（仅云端 API）。

Which edition of the application is running.

The project ships in three shapes, which differ in what has to be downloaded.
The **full** edition drives Hy-MT2 7B through llama.cpp alongside a cloud
endpoint and can run the two at once; it wants a GPU and about six gigabytes of
weights. The **mini** edition is the same machinery on the 1.8B sibling — about
one gigabyte, and it runs on a Mac that has no room for the 7B. The **lite**
edition talks only to an OpenAI-compatible API and downloads nothing at all.

The difference is not a build flag baked into the code. Both editions contain
the same modules — separating them at the source level would mean maintaining
two trees — so the edition is a small piece of runtime state that decides which
backends are offered and which commands are allowed. That way a reader who
started on the lite edition can switch to the full one by downloading the
weights, without reinstalling anything.

Resolution order:
    1. ``$LOCALTRANS_EDITION`` — an explicit override, useful for testing both
       shapes from one checkout.
    2. ``edition.txt`` beside this module — written when the package is built or
       exported, and what makes the lite download behave as the lite edition.
    3. ``full``, because a source checkout is the complete project.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

FULL = "full"
MINI = "mini"
LITE = "lite"
EDITIONS: tuple[str, ...] = (FULL, MINI, LITE)

# What each backend is called where a reader sees it. Kept here beside the list
# of backends themselves, because separate copies drift: adding 「tencent」 to
# BACKENDS while four interface modules kept their own dictionaries made every
# page that renders a backend name raise KeyError, and the glossary page — which
# only wanted a header — answered HTTP 500.
#
# Three lengths, because the three sites need different ones: a word for a
# toggle, a phrase for a heading, a sentence for a tooltip.
# 每个后端在界面上的名字。与后端列表本身放在一起，因为分散的副本会漂移：往 BACKENDS 里
# 加了「tencent」，而四个界面模块各自留着自己的字典，于是每个要显示后端名字的页面都抛
# KeyError——术语库页只是想画个页眉，结果返回了 HTTP 500。
#
# 给三种长度，因为三处用法需要的不一样：切换器要一个词，标题要一个短语，提示要一句话。
BACKEND_SHORT: dict[str, str] = {
    "local": "本地", "api": "云端", "hybrid": "混合",
    # Retired as a backend, kept because completed jobs recorded it.
    # 已不再是后端，保留是因为历史任务里记录过这个值。
    "tencent": "腾讯云",
}
BACKEND_NAMES: dict[str, str] = {
    "local": "本地模型", "api": "云端 API", "hybrid": "混合", "tencent": "腾讯云翻译",
    "unknown": "未知",
}
BACKEND_NOTES: dict[str, str] = {
    "local": "本地模型（离线，不产生费用）",
    "api": "云端 API",
    "hybrid": "混合模式（本地与云端并行）",
    "tencent": "腾讯云机器翻译（按字符计费）",
}

# The cloud services the 「云端」 backend can be pointed at. One is used at a
# time: the setting names which, and nothing sends a paragraph to both.
# 「云端」后端可以指向的云服务。同一时间只用一个：设置项指明是哪一个，不存在把同一段
# 落同时发给两家的情况。
CLOUD_PROVIDER_NAMES: dict[str, str] = {
    "openai": "OpenAI 兼容接口",
    "tencent": "腾讯云翻译",
}

CLOUD_PROVIDER_NOTES: dict[str, str] = {
    "openai": "",
    "tencent": "腾讯云机器翻译（TMT），用 SecretId 与 SecretKey 签名认证。按原文字符计费，每月前 500 万字符免费（约合 100 篇论文），超出后 58 元／百万字符。",
}


# Human-readable names, used wherever the edition is shown to a reader.
# 面向用户显示的版本名。
EDITION_NAMES: dict[str, str] = {
    FULL: "完整版（本地 7B + 云端 API）",
    MINI: "本地轻量版（本地 1.8B + 云端 API）",
    LITE: "轻量版（仅云端 API）",
}

# Backends each edition can offer. The lite edition has no local model, so
# neither `local` nor `hybrid` — which drives the local model too — can work.
# 各版本可提供的后端。轻量版没有本地模型，因此 local 与同样会调用本地模型的 hybrid
# 都不可用。
_BACKENDS: dict[str, tuple[str, ...]] = {
    FULL: ("local", "api", "hybrid"),
    MINI: ("local", "api", "hybrid"),
    LITE: ("api",),
}

# Which weights each edition's `setup` fetches, and what the local server loads.
# The 1.8B is the same family and the same prompt format, so nothing downstream
# changes — only the file, and whether there is a separate draft model to
# speculate with. There is not: in the mini edition the 1.8B *is* the model, and
# a model cannot draft for itself.
# 各版本 setup 下载的权重，以及本地推理服务加载哪一个。1.8B 与 7B 同族、同 prompt
# 格式，因此下游没有任何东西需要改，变的只有文件本身，以及有没有一个单独的 draft
# 模型可供投机解码——本地轻量版没有：1.8B 本身就是主模型，模型不能给自己当草稿。
_LOCAL_MODEL: dict[str, dict] = {
    FULL: {
        "repo": "tencent/Hy-MT2-7B-GGUF",
        "quant": "Q6_K",
        "draft_repo": "tencent/Hy-MT2-1.8B-GGUF",
        "draft_quant": "Q4_K_M",
        "size_note": "约 6.2 GB",
    },
    MINI: {
        "repo": "tencent/Hy-MT2-1.8B-GGUF",
        "quant": "Q4_K_M",
        "draft_repo": None,
        "draft_quant": None,
        "size_note": "约 1.1 GB",
    },
}

_MARKER = Path(__file__).with_name("edition.txt")


def current() -> str:
    """
    中文：返回当前发行版本，"full" 或 "lite"。

    Resolve the running edition.

    An unrecognised value is treated as the full edition rather than raising:
    the consequence of guessing wrong is a backend switch offering an option
    that then fails with a clear message, whereas raising here would stop the
    application from starting at all.

    Returns:
        str: Either :data:`FULL` or :data:`LITE`.
    """
    override = (os.getenv("LOCALTRANS_EDITION") or "").strip().lower()
    if override in EDITIONS:
        return override
    if override:
        logger.warning("Ignoring unknown LOCALTRANS_EDITION=%r", override)

    try:
        marker = _MARKER.read_text(encoding="utf-8").strip().lower()
    except OSError:
        return FULL
    return marker if marker in EDITIONS else FULL


def is_lite() -> bool:
    """
    中文：当前是否为轻量版。

    Report whether the lite edition is running.

    Returns:
        bool: ``True`` when only the cloud API is available.
    """
    return current() == LITE


def display_name() -> str:
    """
    中文：返回当前版本的中文名称。

    Name the running edition for display.

    Returns:
        str: A short Chinese label, e.g. 「轻量版（仅云端 API）」.
    """
    return EDITION_NAMES[current()]


def backends() -> tuple[str, ...]:
    """
    中文：返回当前版本支持的翻译后端。

    List the translation backends this edition can use.

    Returns:
        tuple[str, ...]: Backend keys in the order they should be offered,
            a subset of ``("local", "api", "hybrid")``.
    """
    return _BACKENDS[current()]


def supports(backend: str) -> bool:
    """
    中文：判断某个后端在当前版本是否可用。

    Check whether a backend is available in this edition.

    Args:
        backend (str): One of ``"local"``, ``"api"``, ``"hybrid"``.

    Returns:
        bool: ``True`` when the backend can be selected.
    """
    return backend in backends()


def default_backend() -> str:
    """
    中文：返回当前版本的默认后端。

    The backend to start from when nothing has been configured.

    Returns:
        str: ``"api"`` on the lite edition, ``"local"`` on the other two — an
            edition that has weights defaults to using them, because doing so
            costs nothing and needs no credentials.
    """
    return "api" if is_lite() else "local"


def local_model() -> dict | None:
    """
    中文：返回当前版本要下载/加载的本地模型规格。

    Describe the local weights this edition uses.

    Returns:
        dict | None: Keys ``repo``, ``quant``, ``draft_repo``, ``draft_quant``
            and ``size_note``; ``None`` on the lite edition, which has no local
            model at all.
    """
    return _LOCAL_MODEL.get(current())


def supports_speculative() -> bool:
    """
    中文：当前版本是否可用投机解码（需要一个更小的同族草稿模型）。

    Report whether speculative decoding is possible in this edition.

    Returns:
        bool: ``True`` only when the edition downloads a separate, smaller draft
            model. The mini edition's 1.8B is already the smallest of the family
            and cannot draft for itself.
    """
    spec = local_model()
    return bool(spec and spec.get("draft_repo"))


def require_local(feature: str) -> None:
    """
    中文：轻量版下调用本地模型相关功能时，给出明确的升级提示。

    Refuse a local-model feature on the lite edition, with a way forward.

    Args:
        feature (str): What was being attempted, named as the user would name
            it, e.g. ``"下载本地模型"``.

    Raises:
        RuntimeError: Always, when running the lite edition.
    """
    if not is_lite():
        return
    raise RuntimeError(
        f"{EDITION_NAMES[LITE]}不包含本地模型，无法{feature}。"
        "改用云端 API，或按 README 的说明换成本地轻量版 / 完整版"
        "（把 src/local_translator/edition.txt 改成 mini 或 full，"
        "再运行 localtrans setup）。"
    )
