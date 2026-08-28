"""
中文：下载并安装 llama.cpp 运行时与 Hy-MT2 GGUF 权重。

One-shot installers for the two external artefacts the application needs:

1. A prebuilt ``llama.cpp`` macOS/arm64 release, which provides the
   ``llama-server`` binary exposing an OpenAI-compatible HTTP API.
2. A quantised Hy-MT2 translation model in GGUF format from the official
   ``tencent/Hy-MT2-7B-GGUF`` repository on Hugging Face.

Both downloads are resumable and verified by size, because the model weights are
several gigabytes and a half-finished file must never be mistaken for a good one.
"""

from __future__ import annotations

import logging
import os
import pathlib
import platform
import shutil
import stat
import sys
import tarfile
import tempfile
import time
import zipfile
from collections.abc import Callable
from pathlib import Path

import httpx

from local_translator.config import AppPaths

logger = logging.getLogger(__name__)

LLAMA_CPP_REPO = "ggml-org/llama.cpp"
HF_MODEL_REPO = "tencent/Hy-MT2-7B-GGUF"

# Quantisations the UI offers, ordered from smallest to highest fidelity.
# UI 中可选的量化档位，按体积从小到大排列。
SUPPORTED_QUANTS: tuple[str, ...] = ("Q4_K_M", "Q6_K", "Q8_0")

# Binaries and shared libraries we keep from the llama.cpp archive. Everything
# else (test tools, examples) is discarded to keep the install small.
# 只保留必要的可执行文件与动态库，其余丢弃。
_WANTED_BINARIES = ("llama-server", "llama-cli")

# Suffix the host puts on an executable.
# 本平台可执行文件的后缀。
_EXE = ".exe" if sys.platform.startswith("win") else ""

ProgressCallback = Callable[[int, int], None]
"""Callback receiving ``(bytes_done, bytes_total)``; total is 0 when unknown."""


def _human(n: float) -> str:
    """
    中文：把字节数格式化成人类可读字符串。

    Format a byte count for log and CLI output.

    Args:
        n (float): Byte count.

    Returns:
        str: e.g. ``"6.2 GB"``.
    """
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024.0:
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} PB"


def _stream_to_part(
    client: httpx.Client,
    url: str,
    part: Path,
    *,
    expected_size: int | None,
    progress: ProgressCallback | None,
) -> None:
    """
    中文：从当前分片长度处续传一次，把数据追加写入 .part 文件。

    Perform one transfer attempt, appending to an existing partial file.

    Args:
        client (httpx.Client): Client used for the request.
        url (str): Source URL.
        part (Path): Partial file to append to; its current length is the resume
            offset.
        expected_size (int | None): Known final size, used for the progress
            total when the server does not report ``content-length``.
        progress (ProgressCallback | None): Progress callback.

    Raises:
        httpx.HTTPError: On any transport or status failure.
    """
    resume_from = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={resume_from}-"} if resume_from else {}

    with client.stream("GET", url, headers=headers) as resp:
        # A server that ignores Range replies 200 and restarts from byte zero,
        # so the existing partial data must be discarded to avoid corruption.
        # 服务端忽略 Range 时会返回 200 并从头开始传，此时必须丢弃已有分片，否则文件会被写坏。
        if resume_from and resp.status_code == 200:
            logger.info("Server ignored Range header; restarting from zero")
            part.unlink(missing_ok=True)
            resume_from = 0

        resp.raise_for_status()

        content_length = int(resp.headers.get("content-length", 0) or 0)
        total = (resume_from + content_length) if content_length else (expected_size or 0)
        done = resume_from

        with part.open("ab" if resume_from else "wb") as f:
            for chunk in resp.iter_bytes(chunk_size=1 << 20):
                f.write(chunk)
                done += len(chunk)
                if progress:
                    progress(done, total)


def download_file(
    url: str,
    dest: Path,
    *,
    progress: ProgressCallback | None = None,
    expected_size: int | None = None,
    timeout: float = 60.0,
    max_stalled_attempts: int = 8,
) -> Path:
    """
    中文：断点续传下载，网络中断自动重连，完成后原子重命名。

    Download ``url`` to ``dest``, resuming automatically across network failures.

    Bytes are streamed into a sibling ``<dest>.part`` file, and each attempt
    resumes from that file's current length via an HTTP ``Range`` request. This
    matters in practice: a multi-gigabyte model download is very likely to have
    the connection dropped at least once, and restarting from zero each time
    would make it effectively impossible to finish.

    Retries are budgeted by *progress*, not by attempt count — any attempt that
    transfers at least one byte resets the failure budget, so a flaky link
    eventually completes while a genuinely dead endpoint still fails fast.

    Only once the transfer completes (and matches ``expected_size``, when known)
    is the partial file renamed onto ``dest``, so an interrupted run can never
    leave a truncated file that later looks valid.

    Args:
        url (str): Source URL. Redirects are followed.
        dest (Path): Final destination path. Parent directories are created.
        progress (ProgressCallback | None): Called periodically with
            ``(bytes_done, bytes_total)``. ``bytes_total`` is 0 if unknown.
        expected_size (int | None): Expected final size in bytes. When given and
            the finished file does not match, the download is treated as failed.
        timeout (float): Connect/read timeout in seconds.
        max_stalled_attempts (int): Consecutive attempts allowed to fail without
            transferring any bytes before giving up.

    Returns:
        Path: ``dest``, now containing the complete file.

    Raises:
        httpx.HTTPError: If the download stalls for ``max_stalled_attempts``
            consecutive attempts.
        OSError: If the completed file size does not match ``expected_size``.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")

    if part.exists():
        logger.info("Resuming download of %s from %s", dest.name, _human(part.stat().st_size))

    stalled = 0
    attempt = 0
    with httpx.Client(follow_redirects=True, timeout=timeout) as client:
        while True:
            attempt += 1
            before = part.stat().st_size if part.exists() else 0
            try:
                _stream_to_part(
                    client, url, part, expected_size=expected_size, progress=progress
                )
                break
            except httpx.HTTPStatusError:
                # A bad status will not fix itself by retrying.
                # 状态码错误重试也不会好转，直接抛出。
                raise
            except (httpx.HTTPError, OSError) as exc:
                after = part.stat().st_size if part.exists() else 0
                if after > before:
                    stalled = 0
                    logger.warning(
                        "Transfer interrupted at %s (attempt %d): %s — resuming",
                        _human(after),
                        attempt,
                        exc,
                    )
                else:
                    stalled += 1
                    logger.warning(
                        "Transfer made no progress (attempt %d, %d/%d stalled): %s",
                        attempt,
                        stalled,
                        max_stalled_attempts,
                        exc,
                    )
                    if stalled >= max_stalled_attempts:
                        raise
                time.sleep(min(2 ** min(stalled, 5), 30))

    final_size = part.stat().st_size
    if expected_size is not None and final_size != expected_size:
        raise OSError(
            f"Download size mismatch for {dest.name}: "
            f"got {final_size} bytes, expected {expected_size}"
        )

    part.replace(dest)
    logger.info("Downloaded %s (%s)", dest, _human(final_size))
    return dest


# --------------------------------------------------------------------------- #
# llama.cpp runtime
# --------------------------------------------------------------------------- #


def _llama_cpp_asset_name() -> str:
    """
    中文：返回当前平台对应的 llama.cpp 发布包名片段。

    Determine which llama.cpp release asset matches the host platform.

    Returns:
        str: Substring identifying the asset, e.g. ``"bin-macos-arm64"``.

    Raises:
        RuntimeError: On platforms with no prebuilt asset, where the user must
            build llama.cpp themselves.
    """
    system = platform.system()
    machine = platform.machine()
    if system == "Darwin":
        return "bin-macos-arm64" if machine == "arm64" else "bin-macos-x64"
    if system == "Linux" and machine in ("x86_64", "AMD64"):
        return "bin-ubuntu-x64"
    if system == "Linux" and machine in ("arm64", "aarch64"):
        return "bin-ubuntu-arm64"
    if system == "Windows" and machine in ("AMD64", "x86_64"):
        # Vulkan rather than CUDA: it runs on AMD, Intel and NVIDIA alike and
        # falls back to the CPU where no device is usable, whereas a CUDA build
        # refuses to start without an NVIDIA driver. Untested — see the note in
        # `scripts/build-app.ps1`.
        # 选 Vulkan 而非 CUDA：它在 AMD、Intel 与 NVIDIA 上都能跑，没有可用设备时退回
        # CPU；而 CUDA 版本没有 NVIDIA 驱动就直接起不来。未经实测——见
        # scripts/build-app.ps1 中的说明。
        return "bin-win-vulkan-x64"
    raise RuntimeError(
        f"No prebuilt llama.cpp binary for {system}/{machine}; "
        "please build llama.cpp manually and point LOCALTRANS_LLAMA_BIN at it."
    )


# The asset a pointer-only release carries instead of binaries; its contents are
# the tag of the nightly that has them.
# 只含指针的 release 里代替二进制的那个资产，内容是真正带二进制的 nightly 的标签。
_NIGHTLY_POINTER = "nightly-tag.txt"


def _release(client, tag: str | None) -> dict:
    """
    中文：取一个 release 的元数据；tag 为 None 时取最新的。

    Fetch one release's metadata.

    Args:
        client (httpx.Client): Open client.
        tag (str | None): Release tag, or ``None`` for the latest.

    Returns:
        dict: The release as the API returns it.
    """
    path = f"releases/tags/{tag}" if tag else "releases/latest"
    resp = client.get(f"https://api.github.com/repos/{LLAMA_CPP_REPO}/{path}")
    resp.raise_for_status()
    return resp.json()


def _asset(release: dict, wanted: str) -> dict | None:
    """
    中文：在一个 release 里找名字含指定片段的资产。

    Find the asset whose name contains a fragment.

    Args:
        release (dict): Release metadata.
        wanted (str): Substring identifying the asset.

    Returns:
        dict | None: The asset, or ``None`` when the release has no such file.
    """
    for asset in release.get("assets", []):
        if wanted in asset["name"]:
            return asset
    return None


def resolve_llama_cpp_asset(tag: str | None = None) -> tuple[str, str, int]:
    """
    中文：查询 GitHub Release，返回当前平台的 llama.cpp 下载地址。

    Query the GitHub releases API for a llama.cpp build matching this platform.

    Args:
        tag (str | None): Specific release tag such as ``"b10456"``. Defaults to
            the latest release.

    Returns:
        tuple[str, str, int]: ``(release_tag, download_url, size_in_bytes)``.

    Raises:
        RuntimeError: If the release contains no asset for this platform.
    """
    wanted = _llama_cpp_asset_name()
    with httpx.Client(follow_redirects=True, timeout=30.0) as client:
        data = _release(client, tag)
        found = _asset(data, wanted)
        if found:
            return data["tag_name"], found["browser_download_url"], found.get("size", 0)

        # The release carries no binaries, only a pointer to the nightly that
        # does. This is what `releases/latest` returns as of 2026-08: a tag
        # `v0.2.0` whose single asset is `nightly-tag.txt` holding `b10566`.
        # Following it is the supported way to reach the current build; without
        # this, first-run setup fails for every new installation.
        # 该 release 里没有二进制，只有一个指向真正 nightly 的指针。2026-08 起
        # releases/latest 返回的正是这个：标签 v0.2.0，唯一的资产是内容为 b10566 的
        # nightly-tag.txt。跟随它是取到当前构建的正当方式；没有这一步，每一次全新安装
        # 的首次设置都会失败。
        pointer = _asset(data, _NIGHTLY_POINTER)
        if tag is None and pointer:
            nightly = client.get(pointer["browser_download_url"]).text.strip()
            if nightly:
                logger.info("llama.cpp %s points at nightly %s", data["tag_name"], nightly)
                data = _release(client, nightly)
                found = _asset(data, wanted)
                if found:
                    return (data["tag_name"], found["browser_download_url"],
                            found.get("size", 0))

    raise RuntimeError(f"llama.cpp release {data.get('tag_name')} has no asset matching {wanted!r}")


def _extract_zip_safely(archive, destination: Path) -> None:
    """
    中文：解压 zip，并拒绝任何越界路径。

    Extract a zip, refusing any member that would land outside the destination.

    ``ZipFile.extractall`` does sanitise member names, but only by stripping
    leading separators and drive letters; this checks the resolved path so that
    a crafted archive cannot write outside the temporary directory.

    Args:
        archive (zipfile.ZipFile): The open archive.
        destination (Path): Directory to extract into.

    Returns:
        None.

    Raises:
        RuntimeError: If a member resolves outside ``destination``.
    """
    root = destination.resolve()
    for member in archive.namelist():
        target = (root / member).resolve()
        if root not in target.parents and target != root:
            raise RuntimeError(f"Refusing to extract {member!r} outside {root}")
    archive.extractall(destination)


def install_llama_cpp(
    paths: AppPaths,
    *,
    tag: str | None = None,
    progress: ProgressCallback | None = None,
    force: bool = False,
) -> Path:
    """
    中文：下载并解压 llama.cpp，把 llama-server 及其动态库放进 bin 目录。

    Download a prebuilt llama.cpp release and install its binaries.

    The archive is extracted to a temporary directory, then ``llama-server``,
    ``llama-cli`` and every shared library found alongside them are flattened
    into ``paths.bin_dir``. Keeping the ``.dylib``/``.so`` files next to the
    executables is what allows the ``@rpath``-relative links inside the official
    builds to resolve without any environment variable.

    Args:
        paths (AppPaths): Path registry providing the destination ``bin_dir``.
        tag (str | None): Release tag to install. Defaults to the latest.
        progress (ProgressCallback | None): Download progress callback.
        force (bool): Reinstall even if ``llama-server`` already exists.

    Returns:
        Path: Absolute path to the installed ``llama-server`` executable.

    Raises:
        RuntimeError: If the archive contains no ``llama-server`` binary.
    """
    target = paths.llama_server_bin()
    if target.exists() and not force:
        logger.info("llama-server already installed at %s", target)
        return target

    release_tag, url, size = resolve_llama_cpp_asset(tag)
    logger.info("Installing llama.cpp %s (%s)", release_tag, _human(size))

    paths.bin_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="llamacpp-") as tmpdir:
        tmp = Path(tmpdir)
        # The official releases are gzipped tar on Unix and zip on Windows.
        # 官方发布包在 Unix 上是 gzip tar，在 Windows 上是 zip。
        is_zip = url.endswith(".zip")
        archive = tmp / ("llama.zip" if is_zip else "llama.tar.gz")
        download_file(url, archive, progress=progress, expected_size=size or None)

        extract_root = tmp / "extracted"
        extract_root.mkdir()
        if is_zip:
            with zipfile.ZipFile(archive) as zf:
                _extract_zip_safely(zf, extract_root)
        else:
            with tarfile.open(archive, "r:gz") as tf:
                # filter="data" strips setuid bits and rejects paths escaping
                # the destination; it is the safe default from Python 3.12 on.
                # filter="data" 会剥离特权位并拒绝越界路径，是 3.12+ 的安全默认值。
                tf.extractall(extract_root, filter="data")

        server_src = next(
            (p for p in extract_root.rglob(f"llama-server{_EXE}") if p.is_file()),
            None,
        )
        if server_src is None:
            raise RuntimeError(f"llama-server not found inside {url}")

        # Everything the binaries need lives in the same directory in the
        # official archives, so copy that directory's binaries and libraries.
        # 官方包里可执行文件与依赖库同目录，整目录挑选拷贝即可。
        src_dir = server_src.parent
        installed: list[str] = []
        for item in src_dir.iterdir():
            if not item.is_file():
                continue
            is_lib = item.suffix in (".dylib", ".so", ".dll") or ".so." in item.name
            is_bin = item.stem in _WANTED_BINARIES if _EXE else item.name in _WANTED_BINARIES
            if not (is_lib or is_bin):
                continue
            dst = paths.bin_dir / item.name
            shutil.copy2(item, dst)
            if is_bin:
                dst.chmod(dst.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
            installed.append(item.name)

        logger.info("Installed %d files into %s", len(installed), paths.bin_dir)

    if not target.exists():
        raise RuntimeError(f"Installation finished but {target} is missing")
    return target


# --------------------------------------------------------------------------- #
# Model weights
# --------------------------------------------------------------------------- #


def resolve_model_file(quant: str, repo: str = HF_MODEL_REPO) -> tuple[str, int]:
    """
    中文：在 HF 仓库中按量化档位查找 GGUF 文件名（大小写不敏感）。

    Look up the GGUF filename for a quantisation level in a Hugging Face repo.

    Filenames in the official repository are inconsistently cased — for example
    ``HY-MT2-7B-Q6_K.gguf`` but ``Hy-MT2-7B-Q4_K_M.gguf`` — so matching is done
    case-insensitively against the quantisation suffix rather than by
    constructing the name from a template.

    Args:
        quant (str): Quantisation label such as ``"Q6_K"``. Case-insensitive.
        repo (str): Hugging Face model repository id.

    Returns:
        tuple[str, int]: ``(filename, size_in_bytes)``. Size is 0 when the API
            does not report it.

    Raises:
        ValueError: If no ``.gguf`` file matches the requested quantisation.
    """
    with httpx.Client(follow_redirects=True, timeout=30.0) as client:
        resp = client.get(f"https://huggingface.co/api/models/{repo}", params={"blobs": "true"})
        resp.raise_for_status()
        data = resp.json()

    suffix = f"-{quant.lower()}.gguf"
    for sib in data.get("siblings", []):
        name = sib["rfilename"]
        if name.lower().endswith(suffix):
            return name, int(sib.get("size") or 0)

    available = sorted(
        s["rfilename"] for s in data.get("siblings", []) if s["rfilename"].endswith(".gguf")
    )
    raise ValueError(f"No GGUF for quantisation {quant!r} in {repo}. Available: {available}")


def install_model(
    paths: AppPaths,
    quant: str = "Q6_K",
    *,
    repo: str = HF_MODEL_REPO,
    progress: ProgressCallback | None = None,
    force: bool = False,
) -> Path:
    """
    中文：下载指定量化档位的 Hy-MT2 GGUF 权重到共享模型目录（默认 ~/LLM/hunyuan-mt）。

    Download the Hy-MT2 GGUF weights for a given quantisation level.

    Args:
        paths (AppPaths): Path registry providing the destination ``models_dir``.
        quant (str): Quantisation label, one of :data:`SUPPORTED_QUANTS`.
            ``Q6_K`` (~6.2 GB) is the recommended balance of quality and size.
        repo (str): Hugging Face model repository id.
        progress (ProgressCallback | None): Download progress callback.
        force (bool): Re-download even if the file already exists at the right size.

    Returns:
        Path: Absolute path to the downloaded ``.gguf`` file.
    """
    filename, size = resolve_model_file(quant, repo)
    dest = paths.models_dir / filename

    if dest.exists() and not force:
        if size and dest.stat().st_size != size:
            logger.warning(
                "%s exists but is %s (expected %s); re-downloading",
                dest.name,
                _human(dest.stat().st_size),
                _human(size),
            )
        else:
            logger.info("Model already present at %s", dest)
            return dest

    url = f"https://huggingface.co/{repo}/resolve/main/{filename}"
    logger.info("Downloading %s (%s) from %s", filename, _human(size), repo)
    return download_file(url, dest, progress=progress, expected_size=size or None, timeout=120.0)


def find_installed_models(paths: AppPaths) -> list[Path]:
    """
    中文：列出已下载的 GGUF 权重文件。

    List GGUF weight files already present in the models directory.

    Args:
        paths (AppPaths): Path registry.

    Returns:
        list[Path]: Sorted list of ``.gguf`` files, possibly empty.
    """
    # Both roots are listed: a machine part-way through the move has weights in
    # each, and reporting only the new one would present a working install as
    # empty.
    # 两处都要列出：迁移途中两边都可能有权重，只报新目录会把可用的安装显示为空。
    if not paths.models_dir.exists() and not paths.legacy_models_dir.exists():
        return []
    found: dict[str, pathlib.Path] = {}
    for root in (paths.legacy_models_dir, paths.models_dir):
        if not root.exists():
            continue
        for item in root.glob("*.gguf"):
            if item.is_file():
                # The new location wins on name collision, since it is iterated
                # second and is where a fresh download lands.
                # 同名时以新目录为准：它排在后面，也是新下载的落点。
                found[item.name] = item
    return sorted(found.values())


def llama_server_path(paths: AppPaths) -> Path | None:
    """
    中文：返回可用的 llama-server 路径，未安装则返回 None。

    Locate a usable ``llama-server`` executable.

    Honours the ``LOCALTRANS_LLAMA_BIN`` environment variable so a
    self-compiled build (for example one with different acceleration options)
    can be used instead of the downloaded release.

    Args:
        paths (AppPaths): Path registry.

    Returns:
        Path | None: Path to the executable, or ``None`` if not installed.
    """
    override = os.getenv("LOCALTRANS_LLAMA_BIN")
    if override:
        p = Path(override).expanduser()
        return p if p.exists() else None
    target = paths.llama_server_bin()
    return target if target.exists() else None
