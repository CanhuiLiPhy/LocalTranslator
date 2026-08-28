"""
中文：全局路径解析与可持久化配置。

Application-wide path resolution and persisted user settings for Local Translator.

All mutable state lives outside the source tree so the project directory stays
clean and the app survives reinstalls:

- Application data (database, model weights, binaries, per-document metadata)
  goes to ``~/Library/Application Support/LocalTranslator`` on macOS.
- User-visible documents (PDFs fetched from arXiv or uploaded through the web
  UI, together with their translations) go to ``~/Documents/LocalTranslator``.

Secrets (API keys) are never written to ``settings.json``; they are stored in
the system keychain via :mod:`keyring` and referenced here only by account name.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)

APP_NAME = "LocalTranslator"

# Keychain service name used for all API key entries / 所有 API 密钥使用的钥匙串服务名
KEYRING_SERVICE = "local-translator"


# --------------------------------------------------------------------------- #
# Paths / 路径
# --------------------------------------------------------------------------- #


# Where the pointer file lives: always the platform's own data directory, never
# a directory the pointer itself could move. A file that says where things are
# cannot be kept where it says things are.
# 指针文件放在哪里：永远是平台自己的数据目录，绝不放在这个指针本身可以挪走的目录里。
# 一个「说明东西在哪」的文件，不能存在它自己所指的地方。
_POINTER_NAME = "paths.json"


def _platform_data_dir() -> Path:
    """
    中文：平台默认的数据目录，不看任何覆盖设置。

    The platform's own data directory, before any override is considered.

    Returns:
        Path: ``~/Library/Application Support/LocalTranslator`` on macOS,
            ``%LOCALAPPDATA%\\LocalTranslator`` on Windows, and the XDG data
            directory elsewhere.
    """
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / APP_NAME
    if sys.platform.startswith("win"):
        # The place Windows expects per-user application data that is not worth
        # roaming — a database and a few gigabytes of weights are exactly that.
        # Windows 为每用户、不值得漫游的应用数据准备的位置——一个数据库加上几个 GB 的
        # 权重正属于此类。
        local = os.getenv("LOCALAPPDATA")
        base = Path(local) if local else Path.home() / "AppData" / "Local"
        return base / APP_NAME
    xdg = os.getenv("XDG_DATA_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".local" / "share"
    return base / APP_NAME


def read_directory_choices() -> dict[str, str]:
    """
    中文：读出用户自己设过的目录位置。

    Read the directories the reader chose, if any.

    Returns:
        dict[str, str]: Keys among ``library_dir`` and ``llm_dir``. Missing keys
            mean "use the default"; a corrupt file means the same, because a
            path setting that cannot be read is not a reason to refuse to start.
    """
    pointer = _platform_data_dir() / _POINTER_NAME
    try:
        loaded = json.loads(pointer.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}
    return {key: str(value) for key, value in loaded.items()
            if key in ("library_dir", "llm_dir") and value}


def write_directory_choices(choices: dict[str, str]) -> Path:
    """
    中文：把用户选的目录位置写进指针文件。

    Record the directories the reader chose.

    Args:
        choices (dict[str, str]): Keys among ``library_dir`` and ``llm_dir``. An
            empty or absent value clears that choice and restores the default.

    Returns:
        Path: The pointer file that was written.

    Note:
        Nothing is moved. The database stores absolute paths, so translations
        already made keep opening from where they are; only new work goes to
        the new place. Moving several gigabytes on a reader's behalf, while
        they watch a settings page, is not something to do quietly.
        不搬动任何东西。数据库里存的是绝对路径，已经译好的东西还在原处、照样打得开；
        只有新产生的东西去新地方。替读者搬几个 G，而他正看着一个设置页——这种事不该
        悄悄做。
    """
    pointer = _platform_data_dir() / _POINTER_NAME
    pointer.parent.mkdir(parents=True, exist_ok=True)
    kept = {key: value.strip() for key, value in choices.items()
            if key in ("library_dir", "llm_dir") and value and value.strip()}
    pointer.write_text(json.dumps(kept, ensure_ascii=False, indent=2),
                       encoding="utf-8")
    logger.info("目录设置已写入 %s：%s", pointer, kept or "全部恢复默认")
    return pointer


def _default_llm_dir() -> Path:
    """
    中文：返回大模型统一存放目录，可用 LOCALTRANS_LLM_DIR 覆盖。

    Resolve the shared root for large model weights.

    Weights are kept outside the application data directory on purpose. They are
    large, slow to fetch and not specific to this application — the same
    machine may serve several tools from one copy, and a future OCR or embedding
    model belongs beside the translation weights rather than inside one app's
    private folder. Keeping them separate also means the application's own data
    (database, settings, logs) stays small enough to back up casually.

    Resolution order:
        1. ``$LOCALTRANS_LLM_DIR`` if set.
        2. ``~/LLM``.

    Returns:
        Path: Absolute path to the model root. Not created here; call
            :meth:`AppPaths.ensure`.
    """
    chosen = read_directory_choices().get("llm_dir")
    override = os.getenv("LOCALTRANS_LLM_DIR") or chosen
    if override:
        return Path(override).expanduser().resolve()
    return (Path.home() / "LLM").resolve()


def _default_data_dir() -> Path:
    """
    中文：返回应用数据根目录，可用 LOCALTRANS_DATA_DIR 环境变量覆盖。

    Resolve the root directory for application-managed data.

    Resolution order:
        1. ``$LOCALTRANS_DATA_DIR`` if set (absolute or user-expandable path).
        2. ``~/Library/Application Support/LocalTranslator`` on macOS.
        3. ``%LOCALAPPDATA%\\LocalTranslator`` on Windows.
        4. ``$XDG_DATA_HOME/LocalTranslator`` or ``~/.local/share/LocalTranslator``
           elsewhere.

    Returns:
        Path: Absolute path to the data root. The directory is not created here;
            call :meth:`AppPaths.ensure` for that.
    """
    override = os.getenv("LOCALTRANS_DATA_DIR")
    if override:
        return Path(override).expanduser().resolve()
    return _platform_data_dir()


def _default_library_dir() -> Path:
    """
    中文：返回用户可见的文档库目录，可用 LOCALTRANS_LIBRARY_DIR 覆盖。

    Resolve the user-facing library directory that holds downloaded/uploaded
    source PDFs alongside their translated outputs.

    Returns:
        Path: Absolute path to the library root, defaulting to
            ``~/Documents/LocalTranslator/library``.
    """
    chosen = read_directory_choices().get("library_dir")
    override = os.getenv("LOCALTRANS_LIBRARY_DIR") or chosen
    if override:
        return Path(override).expanduser().resolve()
    return Path.home() / "Documents" / APP_NAME / "library"


class AppPaths:
    """
    中文：集中管理应用使用的所有目录与文件路径。

    Central registry of every directory and well-known file the application
    uses. Instantiating this class is cheap and side-effect free; call
    :meth:`ensure` to materialise the directory tree on disk.

    Attributes:
        data_dir (Path): Application data root.
        db_path (Path): SQLite database file (WAL mode).
        settings_path (Path): JSON file holding :class:`Settings`.
        pid_path (Path): PID of the running server, written at startup and
            removed on shutdown, so the app can be stopped without a
            terminal and without guessing at a process name.
        bin_dir (Path): Extracted llama.cpp binaries.
        llm_dir (Path): Shared root for model weights, ``~/LLM`` by default and
            deliberately outside ``data_dir`` — see :func:`_default_llm_dir`.
        models_dir (Path): Translation weights, one family per subdirectory so
            that adding an OCR or embedding model later does not mix filenames
            from different projects in one flat folder.
        legacy_models_dir (Path): Where weights lived before they moved under
            ``llm_dir``. Read-only fallback, so an installation that has not
            been migrated still finds its model instead of silently offering to
            download six gigabytes again.
        meta_dir (Path): Per-document working directories and tracking JSON.
        logs_dir (Path): Rotating application and llama-server logs.
        library_dir (Path): User-facing PDF library.
    """

    def __init__(self, data_dir: Path | None = None, library_dir: Path | None = None,
                 llm_dir: Path | None = None):
        self.data_dir = (data_dir or _default_data_dir()).resolve()
        self.library_dir = (library_dir or _default_library_dir()).resolve()
        self.llm_dir = (llm_dir or _default_llm_dir()).resolve()

        self.db_path = self.data_dir / "state.db"
        self.settings_path = self.data_dir / "settings.json"
        self.pid_path = self.data_dir / "server.pid"
        self.bin_dir = self.data_dir / "bin"
        self.models_dir = self.llm_dir / "hunyuan-mt"
        self.ocr_models_dir = self.llm_dir / "paddleocr"
        self.legacy_models_dir = self.data_dir / "models"
        self.meta_dir = self.data_dir / "meta"
        self.logs_dir = self.data_dir / "logs"

    def ensure(self) -> AppPaths:
        """
        中文：创建所有需要的目录（幂等）。

        Create every managed directory if missing. Safe to call repeatedly.

        Returns:
            AppPaths: ``self``, to allow fluent use.
        """
        for d in (
            self.data_dir,
            self.bin_dir,
            self.llm_dir,
            self.models_dir,
            self.meta_dir,
            self.logs_dir,
            self.library_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)
        return self

    def find_model(self, filename: str) -> Path:
        """
        中文：解析权重文件路径，新位置没有时回退到旧位置。

        Resolve a weight file, falling back to the pre-migration location.

        Returns the new path when the file is absent from both, so callers that
        report "not installed" and offer to download still behave correctly.

        Args:
            filename (str): Weight filename, e.g. ``HY-MT2-7B-Q6_K.gguf``.

        Returns:
            Path: Location of the file, preferring ``models_dir``.
        """
        current = self.models_dir / filename
        if current.exists():
            return current
        legacy = self.legacy_models_dir / filename
        if legacy.exists():
            return legacy
        return current

    def meta_dir_for(self, doc_id: str) -> Path:
        """
        中文：返回某个文档的中间产物目录（tracking JSON、BabelDOC working dir）。

        Return the per-document metadata directory, creating it if needed.

        Keeping BabelDOC's working directory here (rather than next to the
        source PDF) is what stops intermediate artefacts from polluting the
        user's own folders.

        Args:
            doc_id (str): Stable document identifier (hex digest or arXiv id).

        Returns:
            Path: Absolute path to ``<meta_dir>/<doc_id>``.
        """
        p = self.meta_dir / doc_id
        p.mkdir(parents=True, exist_ok=True)
        return p

    def llama_server_bin(self) -> Path:
        """
        中文：返回 llama-server 可执行文件路径（可能尚未下载）。

        Return the expected location of the ``llama-server`` executable.

        Returns:
            Path: ``<bin_dir>/llama-server``, with ``.exe`` on Windows.
                Existence is not guaranteed; the caller should check and run
                ``localtrans setup`` if missing.
        """
        name = "llama-server.exe" if sys.platform.startswith("win") else "llama-server"
        return self.bin_dir / name


# --------------------------------------------------------------------------- #
# Settings / 配置
# --------------------------------------------------------------------------- #

# Layout labels produced by BabelDOC's document layout model. The taxonomy comes
# from babeldoc/format/pdf/document_il/utils/layout_helper.py.
# BabelDOC 版面模型输出的标签集合，取自 layout_helper.py。
# The layout model in use (DocLayout-YOLO docstructbench, the ONNX weights this
# project ships) emits exactly these ten classes and no others. The list used to
# hold twenty-seven, carried over from other layout backends, and the seventeen
# extra ones could never match a paragraph — which is why the settings page
# offered a long column of switches that did nothing at all.
# 当前使用的版面模型（DocLayout-YOLO docstructbench，本项目附带的 ONNX 权重）只输出
# 以下十个类别。此列表原有二十七项，是从其它版面后端沿用来的，多出的十七项永远匹配不到
# 任何段落——设置页那一长列毫无作用的开关正源于此。
ALL_LAYOUT_LABELS: tuple[str, ...] = (
    "title",
    "plain text",
    "abandon",
    "figure",
    "figure_caption",
    "table",
    "table_caption",
    "table_footnote",
    "isolate_formula",
    "formula_caption",
)

# Text the layout model has no class for, recognised by reading the page instead
# (see :mod:`local_translator.engine.frontmatter`). The model calls the paper's
# own title ``title`` — the same class it gives every section heading — and puts
# authors, affiliations and footnotes in ``plain text``, so none of the four can
# be switched on its own through the model's labels.
# 版面模型没有类别、改由阅读页面识别的文字（见 engine/frontmatter.py）。模型把论文
# 标题也叫 title——与各级小标题同类——而作者、单位、脚注都归入 plain text，因此这四
# 项都无法通过模型标签单独开关。
TEXT_KINDS: tuple[str, ...] = (
    "document_title",
    "author",
    "affiliation",
    "footnote",
    # References are recognised by reading them — `looks_like_bibliography` and
    # the citation anchors — because the layout model mislabels them: on one
    # paper only 1 of 4 reference paragraphs carried a `reference` label. The
    # switch was previously hardcoded on; naming it here makes it a choice, and
    # makes the same choice govern the LaTeX path, where a bibliography is not
    # guessed at but written down as `thebibliography`.
    # 参考文献靠阅读内容识别——looks_like_bibliography 与引用锚点——因为版面模型认不准：
    # 实测一篇论文的 4 段参考文献里只有 1 段带 reference 标签。这个开关以前是写死打开的；
    # 在这里给它一个名字，它就成了一项选择，而且同一项选择也管得住 LaTeX 那条路——在那里
    # 参考文献不用猜，它就明明白白写作 thebibliography。
    "reference",
    # Pseudocode is recognised by reading it (`is_pseudocode`) in the PDF path
    # and is written down as an `algorithm` environment in the LaTeX one. Like
    # the bibliography switch, this was hardcoded on; naming it makes it a
    # choice, and the same choice governs both paths.
    # 伪代码在 PDF 那条路上靠阅读内容识别（is_pseudocode），在 LaTeX 这边就明明白白写作
    # algorithm 环境。与参考文献开关一样，它原本是写死打开的；给它一个名字，它就成了一项
    # 选择，而且同一项选择管住两条路。
    "algorithm",
)

# Content the pipeline recognises by reading it rather than by layout class, and
# puts back from the source verbatim. The layout model has no class for an
# algorithm float or a contents page, and its `table` box covers only some of
# the algorithm floats in a single paper, so these are named separately and are
# switchable in their own right.
# 由内容识别（而非版面类别）并原样搬回的部分。版面模型没有「算法块」或「目录页」这样的
# 类别，且同一篇论文里它只把一部分算法块框成 table，因此这些单列出来、各自可控。
PRESERVE_KINDS: tuple[str, ...] = (
    "figure",
    "table",
    "formula",
    "algorithm",
    "contents",
)

# Preset skip-sets for selective translation, mirroring PDFMathTranslate's
# behaviour of leaving footnotes / author info / tables untranslated.
# 选择性翻译预设，对齐 PDFMathTranslate「不翻脚注、姓名、表格」的行为。
# The layout model (DocLayout-YOLO docstructbench) emits exactly ten labels:
# title, plain text, abandon, figure, figure_caption, table, table_caption,
# table_footnote, isolate_formula, formula_caption. Only those can ever match a
# paragraph. The others below (header, footer, footnote, reference, content, …)
# are kept as intent and because the RPC layout backends use a different label
# set, but on the ONNX model they never fire — the behaviour they describe is
# delivered by content and anchor detection instead: references by
# `looks_like_bibliography` and the citation anchors, display equations by
# `is_display_formula`, pseudocode by `is_pseudocode`. The three real skips that
# do the visible work are `abandon` (headers/footers/page furniture), `table`
# (table bodies) and `isolate_formula` (display equations), plus `table_footnote`
# and `formula_caption`.
# 版面模型（DocLayout-YOLO docstructbench）只产出十个标签：title、plain text、
# abandon、figure、figure_caption、table、table_caption、table_footnote、
# isolate_formula、formula_caption。只有这些能匹配到段落。下面其余的（header、
# footer、footnote、reference、content…）保留作意图说明、也因 RPC 版面后端用的是另一
# 套标签，但在 ONNX 模型上永不生效——它们描述的行为改由内容与锚点识别实现：参考文献
# 靠 looks_like_bibliography 与引用锚点，行间公式靠 is_display_formula，伪代码靠
# is_pseudocode。真正起作用的三个标签是 abandon（页眉页脚等）、table（表格主体）与
# isolate_formula（行间公式），外加 table_footnote 与 formula_caption。
LAYOUT_PRESETS: dict[str, tuple[str, ...]] = {
    # Only labels the model actually emits. References, pseudocode, contents
    # pages and display equations are handled by content detection and by
    # verbatim region stamping, not by these labels.
    # 仅包含模型确实会输出的标签。参考文献、伪代码、目录页与行间公式由内容识别与区块
    # 原样搬回处理，不走这些标签。
    "pdfmath": (
        "abandon",
        "table",
        "table_footnote",
        "isolate_formula",
        "formula_caption",
        # A translated author list is worse than an untranslated one: a name
        # rendered as 「邓阳江」 no longer matches the paper it belongs to, and an
        # institution translated out of English matches nothing a reader can
        # look up. Footnotes are ordinary prose and stay translated.
        # 译过的作者名比不译更糟：写成「邓阳江」就再也对不上这篇论文，单位名译出
        # 英文后也查不到任何东西。脚注是普通正文，仍然翻译。
        "author",
        "affiliation",
        "reference",
        "algorithm",
        # PDFMathTranslate leaves figures alone, and so does the preset named
        # after it. It matters more than it used to: in the PDF path this label
        # governs text drawn inside a figure, and in the LaTeX path it governs
        # whether the figure's own image file is rewritten with its text
        # translated — a heavier operation, and one that rasterises a vector
        # figure to do it. Neither should happen unless a reader asks.
        # PDFMathTranslate 不动插图，以它命名的预设也不动。这一点比以前更要紧：在 PDF
        # 那条路上，这个标签管的是画在图里的文字；在 LaTeX 那条路上，它管的是要不要把
        # 图片文件本身重写成译文版——那是更重的操作，而且为此要把矢量图栅格化。两者都
        # 不该在读者没开口的情况下发生。
        "figure",
    ),
    "conservative": (
        "abandon",
        "author",
        "affiliation",
        "reference",
        "algorithm",
    ),
    "everything": (),
}

# Everything selective translation can be asked to leave alone: the classes the
# layout model emits, plus the kinds recognised by reading the page.
# 选择性翻译可以跳过的全部类型：版面模型输出的类别，加上通过阅读页面识别的类型。
SKIPPABLE_LABELS: tuple[str, ...] = ALL_LAYOUT_LABELS + TEXT_KINDS


# Languages the local Hy-MT2 model is trained for: 33 languages plus five
# Chinese ethnic-minority languages and dialects, per Tencent's own listing.
# The cloud model handles at least these, so one list serves both backends and
# the selector does not change meaning when the backend does.
#
# Right-to-left scripts are marked because the constraint is typesetting rather
# than translation: BabelDOC positions text left to right when rebuilding a
# page, so Arabic, Hebrew, Persian and Urdu translate correctly but may lay out
# incorrectly in the resulting PDF. They are offered rather than hidden, since
# they work for the plain-text translator where no typesetting is involved.
#
# 本地 Hy-MT2 训练覆盖的语种：33 种语言外加 5 种中国少数民族语言与方言，取自腾讯
# 官方清单。云端模型至少覆盖这些，因此一份清单同时服务两个后端，切换后端时选择器
# 的含义不变。
#
# 从右向左书写的文种单独标注，因为限制在排版而非翻译：BabelDOC 重建页面时按从左
# 到右定位文本，所以阿拉伯语、希伯来语、波斯语、乌尔都语能正确翻译，但产出的 PDF
# 版式可能不正确。这些语种仍然提供而非隐藏，因为纯文本翻译不涉及排版。
LANGUAGES: dict[str, str] = {
    "zh": "中文",
    "zh-Hant": "繁體中文",
    "en": "English",
    "ja": "日本語",
    "ko": "한국어",
    "fr": "Français",
    "de": "Deutsch",
    "es": "Español",
    "pt": "Português",
    "it": "Italiano",
    "ru": "Русский",
    "uk": "Українська",
    "pl": "Polski",
    "cs": "Čeština",
    "nl": "Nederlands",
    "tr": "Türkçe",
    "vi": "Tiếng Việt",
    "th": "ไทย",
    "id": "Bahasa Indonesia",
    "ms": "Bahasa Melayu",
    "tl": "Filipino",
    "km": "ភាសាខ្មែរ",
    "my": "မြန်မာ",
    "hi": "हिन्दी",
    "bn": "বাংলা",
    "ta": "தமிழ்",
    "te": "తెలుగు",
    "mr": "मराठी",
    "gu": "ગુજરાતી",
    "ar": "العربية（从右向左）",
    "he": "עברית（从右向左）",
    "fa": "فارسی（从右向左）",
    "ur": "اردو（从右向左）",
    "yue": "粤语",
    "bo": "བོད་སྐད།（藏语）",
    "ug": "ئۇيغۇرچە（维吾尔语）",
    "mn": "Монгол（蒙古语）",
    "kk": "Қазақша（哈萨克语）",
}

# Scripts whose PDF layout BabelDOC cannot be trusted with, kept separate so a
# caller can warn without re-deriving the list from the display names.
# BabelDOC 无法可靠排版的文种，单独列出，便于调用方直接提示而不必从显示名反推。
RTL_LANGUAGES: frozenset[str] = frozenset({"ar", "he", "fa", "ur"})


class LocalModelSettings(BaseModel):
    """
    中文：本地 llama-server 与 Hy-MT2 模型的运行参数。

    Runtime configuration for the locally hosted Hy-MT2 translation model
    served through llama.cpp's OpenAI-compatible ``llama-server``.

    Attributes:
        model_file (str): GGUF filename inside ``AppPaths.models_dir``
            (``~/LLM/hunyuan-mt`` by default).
        speculative (bool): Whether to run with a draft model. The draft
            proposes tokens and the full model verifies them, which is
            mathematically lossless — the output distribution is unchanged, so
            this is purely a speed setting. Measured on a 20-page paper it is
            worth about 4% (158s against 165s); the gain is modest because high
            concurrency already saturates the GPU, and speculation trades spare
            compute for fewer sequential steps. Worth turning off when the GPU
            is also serving something else.
        draft_model_file (str | None): Smaller model from the same family used
            as the draft. ``None`` disables speculative decoding regardless of
            the ``speculative`` flag.
        host (str): Bind address for llama-server. Loopback only by default.
        port (int): TCP port for llama-server.
        n_ctx (int): Total KV context in tokens. llama.cpp splits this across
            slots, so each of the ``n_parallel`` slots gets ``n_ctx / n_parallel``
            tokens — 32768/8 = 4096 per paragraph, comfortably above what a
            single PDF paragraph plus its glossary block needs.
        n_parallel (int): Number of concurrent decoding slots (``-np``).
        n_gpu_layers (int): Layers offloaded to Metal; 99 means "all".
        temperature (float): Sampling temperature. Tencent recommends 0.7 for
            free-form translation, but placeholders such as ``{v1}`` survive
            far more reliably at low temperature, so we default to 0.2.
        top_p (float): Nucleus sampling cutoff (official recommendation 0.6).
        top_k (int): Top-k cutoff (official recommendation 20).
        repeat_penalty (float): Repetition penalty (official recommendation 1.05).
        max_tokens (int): Hard cap on generated tokens per paragraph.
        request_timeout (float): Per-request HTTP timeout in seconds.
        auto_start (bool): Whether the app should launch llama-server itself.
        flash_attention (bool): Pass ``-fa on``. Cheaper attention arithmetic and
            a smaller KV cache; on by default only where memory is tight.
        kv_cache_type (str): Quantisation for the K and V caches (``-ctk``/
            ``-ctv``), e.g. ``"q8_0"``. Empty keeps llama.cpp's f16 default.
            Requires flash attention for the V side.
        batch_size (int): ``-b``. 0 keeps llama.cpp's default of 2048.
        ubatch_size (int): ``-ub``. 0 keeps llama.cpp's default of 512. The
            compute buffers are sized from this, so lowering it is the single
            most effective way to shrink a long prompt's memory spike.
        cache_ram_mib (int): ``--cache-ram``. llama-server keeps the KV state of
            prompts it has seen so a later request sharing a prefix can skip
            re-processing it, and the default ceiling for that store is **8192
            MiB**. Translation never collects on it: every paragraph is a
            different prompt, and the only shared prefix is a system message
            short enough to reprocess in milliseconds. Measured over 150
            paragraph-sized requests, the store grew **827 MB** and saved
            **0.4 seconds** — 0.7% — so it is off by default. Raise it if you
            run many parallel slots, where idle slots are swapped through it.
        idle_unload_seconds (int): Stop llama-server after this many seconds
            with no translation running, releasing the weights and the Metal
            buffers. 0 disables it and keeps the server resident.

            Ten seconds by default, because reloading turns out to be
            nearly free and holding the weights is not: measured on this
            project's own models, a cold start is about 2s for the 7B and 0.6s
            for the 1.8B — llama.cpp maps the file rather than reading it —
            against 12.9 GB of Metal buffers the 7B otherwise holds for as long
            as the application stays open. Nothing is ever unloaded while a job
            is running or queued, so the only cost is those two seconds on a
            translation started after a pause.
    """

    model_file: str = "HY-MT2-7B-Q6_K.gguf"
    speculative: bool = True
    draft_model_file: str | None = "Hy-MT2-1.8B-Q4_K_M.gguf"
    flash_attention: bool = False
    kv_cache_type: str = ""
    batch_size: int = 0
    ubatch_size: int = 0
    cache_ram_mib: int = 0
    # Five minutes, not the ten seconds this used to be. Ten equalled the
    # watchdog's own tick interval, so a single poll finding the model idle was
    # enough to unload it — and the gap between "server ready" and "job queued"
    # is not empty: it holds whatever the caller does next, such as downloading
    # a paper. Five minutes still releases the memory of an abandoned session
    # while being far longer than any submission path.
    # 五分钟，而不是先前的十秒。十秒恰好等于看门狗自己的轮询间隔，因此只要一次轮询发现
    # 空闲就足以卸载模型——而「服务就绪」到「任务入队」之间并非空档，其间要做调用方接下来
    # 的事，例如下载论文。五分钟既能释放被遗忘会话占用的内存，又远长于任何提交路径。
    idle_unload_seconds: int = 300

    @model_validator(mode="after")
    def _match_the_edition(self) -> LocalModelSettings:
        """
        中文：本地轻量版没有 7B、也没有可当草稿的更小模型，这里把两者都纠正过来。

        Point the local model at weights this edition actually downloads.

        The mini edition installs the 1.8B as its main model. A settings file
        written by the full edition names the 7B, which is simply not on disk,
        and enables speculative decoding against a draft that is now the main
        model — llama-server would be asked to load the same file twice.

        Only the defaults are corrected: a reader who deliberately downloaded
        the 7B and named it keeps it, because the check is against the
        edition's own default rather than against what happens to be installed.

        Returns:
            LocalModelSettings: ``self``, so the validator can be chained.
        """
        from local_translator import edition

        if edition.current() != edition.MINI:
            return self
        if self.model_file == LocalModelSettings.model_fields["model_file"].default:
            self.model_file = "Hy-MT2-1.8B-Q4_K_M.gguf"
        if not edition.supports_speculative():
            self.speculative = False
            self.draft_model_file = None

        # Memory settings the mini edition wants and the full edition should
        # not have changed under it. Each is only filled in where the reader
        # left the shipped default, so an explicit choice survives.
        # 本地轻量版需要、而完整版不该被动到的内存设置。仅在读者保留出厂默认时才填，
        # 显式改过的值不覆盖。
        fields = LocalModelSettings.model_fields
        if self.flash_attention == fields["flash_attention"].default:
            self.flash_attention = True
        if self.kv_cache_type == fields["kv_cache_type"].default:
            self.kv_cache_type = "q8_0"
        if self.batch_size == fields["batch_size"].default:
            self.batch_size = 512
        if self.ubatch_size == fields["ubatch_size"].default:
            self.ubatch_size = 128
        if self.n_parallel == fields["n_parallel"].default:
            # Four slots, because the fifth buys nothing on this model.
            # Measured on a 23-page paper: one slot took 316s, four took 196s
            # and eight took 195s. Four is where the curve flattens, and it
            # holds less GPU memory than eight does.
            # 四个槽位，因为这个模型上第五个买不到任何东西。23 页论文实测：一个槽位
            # 316 秒，四个 196 秒，八个 195 秒。四个正是曲线走平的位置，且占用的显存
            # 少于八个。
            self.n_parallel = 4
        if self.n_ctx == fields["n_ctx"].default:
            # A quarter of the full edition's context, which on this model is
            # free. Measured on a 23-page paper with eight slots: 32768 tokens
            # cost 3,733 MiB of GPU memory and 8192 cost 2,232, and the two runs
            # took 202s and 196s — the same, within noise — with identical
            # output (Chinese character ratio 0.374 both times, no placeholder
            # left behind). A paragraph plus its glossary block does not come
            # close to filling even a quarter of the smaller figure, so the
            # larger context was buying nothing but memory pressure.
            #
            # The full edition keeps 32768: the same cut there costs 27% of the
            # wall time (194s → 246s) to save 2 GB, which is a trade rather than
            # a free win, and is offered as a setting instead.
            # 完整版上下文的四分之一，在这个模型上是白给的。23 页论文、八个槽位实测：
            # 32768 token 占用 3,733 MiB 显存，8192 占用 2,232，两次耗时 202 秒与
            # 196 秒——在噪声范围内相同——输出也一致（中文字符占比均为 0.374，无残留
            # 占位符）。一个段落加上它的术语块，连较小那个数字的四分之一都填不满，更大的
            # 上下文除了内存压力什么也没换来。
            #
            # 完整版仍用 32768：同样的削减在那里要用 27% 的耗时（194 秒 → 246 秒）
            # 去换 2 GB，那是权衡而非白赚，因此作为设置项提供而不改默认。
            self.n_ctx = 8192
        return self
    host: str = "127.0.0.1"
    port: int = 18080
    n_ctx: int = 32768
    n_parallel: int = 8
    n_gpu_layers: int = 99
    temperature: float = 0.2
    top_p: float = 0.6
    top_k: int = 20
    repeat_penalty: float = 1.05
    max_tokens: int = 2048
    request_timeout: float = 300.0
    auto_start: bool = True

    @property
    def base_url(self) -> str:
        """
        中文：返回 OpenAI 兼容接口的 base URL。

        Returns:
            str: e.g. ``http://127.0.0.1:18080/v1``.
        """
        return f"http://{self.host}:{self.port}/v1"


class ApiModelSettings(BaseModel):
    """
    中文：可选的云端 API 模型配置；密钥存在钥匙串，此处只记账户名。

    Configuration for an optional cloud API model, used for high-quality
    translation runs and for glossary term extraction (which the local
    translation-only model cannot do).

    Attributes:
        enabled (bool): Whether this model may be used at all.
        display_name (str): Human-readable label shown in the UI.
        model_name (str): Model identifier sent to the provider.
        base_url (str): OpenAI-compatible endpoint, e.g.
            ``https://api.example.com/v1``. Empty until you point it at a
            provider; the cloud backend stays unavailable until it is set.
        keyring_account (str): Account name under ``KEYRING_SERVICE`` holding
            the API key. The key itself is never persisted to disk.
        skip_reasoning (bool): Bypass a reasoning model's thinking phase by
            driving the raw completions endpoint with an already-closed think
            block. Only meaningful for models whose chat template hardcodes an
            opening ``<think>`` tag, where no API parameter can disable the
            deliberation. Measured against one such model on a single
            paragraph: 3.43s and 213 output tokens with thinking, 0.97s and 29
            without, with no loss of translation quality. Needs
            ``chat_prompt_template`` to be filled in for the model in use.
        chat_prompt_template (str): Raw prompt format used when
            ``skip_reasoning`` is on. Recover it from the server's own
            ``/tokenize`` rendering for the model you are calling.
            Placeholders: ``{system}``, ``{user}``.
        price_in_per_1m (float | None): Optional input price per 1M tokens, in
            the provider's currency, used only for cost estimation display.
        price_out_per_1m (float | None): Optional output price per 1M tokens.
    """

    enabled: bool = False
    display_name: str = "Cloud model"
    model_name: str = ""
    base_url: str = ""
    keyring_account: str = "cloud"
    skip_reasoning: bool = False
    chat_prompt_template: str = ""
    price_in_per_1m: float | None = None
    price_out_per_1m: float | None = None


def default_cache_mb() -> int:
    """
    中文：段落缓存的默认上限——发布版为 0（关闭），开发检出为 50 MB。

    Default cap for the paragraph cache, in megabytes.

    The two differ because the cache serves two different people. Measured over
    33 papers: between *different* papers it answers 2.68% of blocks by count,
    0.07% by character, and none at all of the 4,229 blocks of real prose —
    every hit is a section heading. Re-translating the *same* paper after
    changing a setting, it answers 78.9%.

    A reader translates a paper once, so a released build carries a cache that
    can only ever return section headings, which are now answered from a table
    anyway. A development checkout re-runs the same paper all day, so it keeps
    one.

    Returns:
        int: 0 in a frozen application, 50 in a source checkout. A reader who
            wants one can set it; the default is what each build is for.
    """
    import sys

    return 0 if getattr(sys, "frozen", False) else 50


class TranslationSettings(BaseModel):
    """
    中文：翻译行为参数。

    Language pair and throughput settings shared by both backends.

    Attributes:
        lang_in (str): Source language code passed to BabelDOC.
        lang_out (str): Target language code passed to BabelDOC.
        backend (str): ``"local"``, ``"api"`` or ``"hybrid"``. Hybrid runs both
            at once, which is faster than either alone because they occupy
            different hardware.
        hybrid_local_slots (int): Concurrent requests the local model may have
            in flight in hybrid mode. This is the load control and, indirectly,
            the split: a fixed-rate resource given a fixed time budget does a
            fixed amount of work. Local throughput saturates near 16 concurrent
            requests and regresses beyond that as the KV cache thrashes (515
            chars/s at 16, 423 at 24), so there is nothing to gain by raising
            it far, and 8 leaves the machine usable.
        hybrid_local_share_cap (float): Hard ceiling on the local character
            share, as a safety net. The slot discipline normally settles well
            below it, so it does not usually bind.
        hybrid_api_marker_threshold (int): Marker count at or above which a
            paragraph always goes to the API. Local fallback rate climbs from
            0.4% at zero markers to 12.5% at six or more.
        hybrid_api_length_threshold (int): Character count at or above which a
            paragraph always goes to the API.
        hybrid_api_strategy (str): ``"direct"`` sends one request per
            API-routed paragraph; ``"batched"`` groups them. Direct is the
            default because the API is concurrency-elastic — throughput climbs
            with requests in flight while latency stays flat — so grouping
            paragraphs whose callers are blocked *reduces* concurrency instead
            of raising throughput. Batched remains available for runs where
            fewer repeated prompts matter more than wall time.
        hybrid_max_batch (int): Paragraphs per batched API request, used only
            by the ``batched`` strategy.
        hybrid_pool_workers (int): BabelDOC worker threads in hybrid mode. Must
            exceed ``hybrid_local_slots`` by enough to leave threads for the
            API side, since a thread routed to either backend is occupied until
            that backend answers. Measured on a 19-page paper, translation
            stage only: 24 workers/8 slots took 26.1s, 32/8 took 17.7s, 48/12
            took 15.4s, 64/16 took 13.5s. The gains come from API concurrency,
            which is still climbing at 64; 48 is chosen as the point past which
            the endpoint is being leaned on harder for a second or two.
        pool_max_workers (int | None): BabelDOC paragraph worker threads.
            Oversubscribing the server's slots measurably helps: on a 20-page
            paper, 8 slots with 16 workers took 165s against 186s for 4 and 4,
            because queued requests keep every slot busy across BabelDOC's
            per-page synchronisation points.
        qps (float): Rate limit. Meaningless for local inference, so it is set
            very high by default and only matters in API mode.
        min_text_length (int): Paragraphs shorter than this are left untouched.
        custom_system_prompt (str): Extra instruction appended to the prompt.
        ignore_cache (bool): Bypass BabelDOC's paragraph-level translation cache.
        translation_cache_mb (int): Cap on BabelDOC's paragraph cache, in
            megabytes, applied when the application exits. ``0`` empties it —
            which is the default in a released build, where the cache can only
            ever answer section headings and those are answered from a table
            now. See :func:`default_cache_mb`.
        keep_intermediates (bool): Keep everything a job wrote into its working
            directory instead of clearing it when the job ends. Off by default,
            and the difference is not small: BabelDOC writes the document out
            again at every stage, and sixty papers left 4.1 GB behind on the
            machine this was measured on — the largest single one 705 MB, for a
            paper whose published output is 12 MB. Nothing reads any of it after
            the job, so the only reason to keep it is to look at it.
    """

    lang_in: str = "en"
    lang_out: str = "zh"
    # validate_default is needed: pydantic skips validation of defaults, so a
    # fresh install of the lite edition would start on `local` — the one value
    # that edition cannot honour — until something wrote the file once.
    # 必须开 validate_default：pydantic 不校验默认值，否则轻量版全新安装会从 local
    # 起步——恰恰是该版本唯一无法满足的值——直到有人写过一次配置文件为止。
    backend: str = Field(default="local", validate_default=True)

    # Which cloud service the "云端" backend actually talks to. Kept apart from
    # `backend` because the two answer different questions: `backend` is where
    # a paragraph goes (local, cloud, or split between them), and this is which
    # cloud. Folding them together — as an earlier "tencent" backend value did
    # — made the two mutually exclusive in the wrong dimension: it meant hybrid
    # could never use Tencent, and it put one cloud service beside "local" in
    # the toggle as though it were a different kind of thing.
    # 「云端」后端实际连的是哪家云服务。与 backend 分开，是因为两者回答的是不同问题：
    # backend 决定段落去哪（本地、云端，或两者分流），而这一项决定是哪家云。把它们揉成
    # 一个——先前那个 "tencent" 后端值就是如此——会在错误的维度上造成互斥：混合模式将
    # 永远用不了腾讯云，而且它会和「本地」并列出现在切换器里，仿佛是同一类东西。
    cloud_provider: str = Field(default="openai", validate_default=True)

    @field_validator("cloud_provider")
    @classmethod
    def _known_provider(cls, value: str) -> str:
        """
        中文：认不出来的服务商一律折回 openai。

        Coerce an unrecognised cloud provider back to the default.

        Args:
            value (str): Provider named in the settings file.

        Returns:
            str: ``"openai"`` or ``"tencent"``.
        """
        return value if value in ("openai", "tencent") else "openai"

    @model_validator(mode="before")
    @classmethod
    def _migrate_tencent_backend(cls, data):
        """
        中文：把旧的 backend="tencent" 迁移成 backend="api" + cloud_provider="tencent"。

        Migrate the retired ``tencent`` backend value.

        Settings files written before the cloud services were merged name
        ``tencent`` as a backend. Read as-is that value now falls through the
        edition check to ``local``, which would silently move a reader who had
        chosen a paid cloud service onto the local model — or, in the lite
        edition, onto nothing at all.

        Args:
            data: Raw mapping from the settings file, or anything else pydantic
                hands us, which is passed through untouched.

        Returns:
            The mapping with the old value rewritten.
        """
        if isinstance(data, dict) and data.get("backend") == "tencent":
            data = {**data, "backend": "api", "cloud_provider": "tencent"}
        return data

    @field_validator("backend")
    @classmethod
    def _available_in_this_edition(cls, value: str) -> str:
        """
        中文：轻量版没有本地模型，把 local/hybrid 折回 api。

        Coerce a backend the running edition cannot provide.

        A settings file written by the full edition names ``local``; opened by
        the lite edition, which ships no weights, that value would send every
        paragraph to an inference server that is not there. The switch is not
        shown in the lite UI either, so nothing would tell the reader why.

        Args:
            value (str): Backend named in the settings file.

        Returns:
            str: ``value`` when this edition supports it, otherwise the
                edition's default backend.
        """
        from local_translator import edition

        if edition.supports(value):
            return value
        replacement = edition.default_backend()
        logger.info("Backend %r is not available in the %s edition; using %r",
                    value, edition.current(), replacement)
        return replacement

    hybrid_local_slots: int = 12
    hybrid_local_share_cap: float = 0.6
    hybrid_api_marker_threshold: int = 3
    hybrid_api_length_threshold: int = 900
    hybrid_api_strategy: str = "direct"
    hybrid_max_batch: int = 10
    hybrid_pool_workers: int = 48
    pool_max_workers: int | None = 16
    qps: float = 1000.0
    min_text_length: int = 5
    custom_system_prompt: str = ""
    ignore_cache: bool = False
    translation_cache_mb: int = Field(default_factory=default_cache_mb)
    keep_intermediates: bool = False


class PdfSettings(BaseModel):
    """
    中文：PDF 输出版式设置。默认输出并排拼版双语 PDF。

    Output formatting options forwarded to BabelDOC's ``TranslationConfig``.

    The important default here is ``use_alternating_pages_dual = False``, which
    makes BabelDOC emit a side-by-side dual PDF (page width doubled, page count
    equal to the original) instead of the interleaved original/translation page
    sequence. See ``pdf_creater.create_side_by_side_dual_pdf``.

    Attributes:
        output_dual (bool): Emit the bilingual PDF.
        output_mono (bool): Emit the translation-only PDF (also feeds the
            two-pane reader mode).
        use_alternating_pages_dual (bool): ``False`` = side-by-side layout,
            ``True`` = alternating pages.
        dual_translate_first (bool): Put the translation on the left / first.
        watermark_mode (str): ``watermarked`` | ``no_watermark`` | ``both``.
        skip_clean (bool): Skip PyMuPDF's structural cleaning pass when writing
            the output. Measured on a 20-page paper, writing the PDFs dominates
            the non-translation half of the pipeline, and the cleaning pass is
            most of that cost. Skipping it yields a larger file that every
            reader still opens correctly.
        restore_images (bool): Re-place bitmaps that page regeneration drops.
        restore_links (bool): Reinstate the source document's hyperlinks,
            which BabelDOC strips along with every other annotation.
        preserve_regions (bool): Copy pseudocode, tables and display
            equations back from the source verbatim, the way figures are
            restored. Leaving them untranslated is not enough: the page is
            rebuilt either way, which re-flows their lines and drops the
            vector rules inside them.
        translate_headings (bool): Send the standard section headings to the
            model instead of answering them from a table. Off by default, and
            the default is the measured one: across 33 papers those headings are
            the only text different papers share, and the model renders them
            inconsistently (「参考文献」 and 「参考资料」 for the same word) and
            sometimes wrongly (``Epochs`` as 「时代」, ``Final`` as 「决赛」).
            See :mod:`engine.headings`.
        translate_contents (bool): Rewrite each row of a contents page into the
            target language, in place. Builds on the verbatim stamp rather than
            replacing it: the page is copied from the source first, then each
            row's title is redacted and redrawn at its own baseline with the
            leader refilled, so nothing reflows and the page numbers do not
            move. Off by default — an untranslated contents page is entirely
            correct, which is more than a reflowed one would be.
        preserve_kinds (list[str]): Which of :data:`PRESERVE_KINDS` to put back
            verbatim. Lets a reader who wants translated table cells switch
            tables off without giving up the treatment for equations.
            Off is useful for telling apart images BabelDOC lost from images the
            repair placed badly.
        skip_form_render (bool): Skip re-emitting the page's Form XObjects and
            inline images. Off, and it must stay off. The name misleads: a
            BabelDOC ``pdfForm`` is not an interactive AcroForm widget but a
            ``/Fm0 Do`` invocation — the container a paper's figures and vector
            graphics are drawn through, plus every inline image, plus the forms
            belonging to formulas. Skipping it drops all of that from the
            output, which is what made figure translation look broken: the
            labels were translated and written into the figure's own stream, and
            then the stream was never invoked. See
            ``pdf_creater.FormRenderUnit``.
        enhance_compatibility (bool): Turn on BabelDOC's compatibility bundle.
        auto_enable_ocr_workaround (bool): Detect scanned pages and OCR them.
        primary_font_family (str | None): ``serif`` | ``sans-serif`` | ``script``.
            ``None`` follows the source: a serif original gets Source Han Serif,
            a sans-serif one Source Han Sans. The three values override that for
            the whole document.
        script_font (str | None): Path to a font file to use where the source
            text is *italic*. Chinese has no true italic, so a calligraphic or
            Fangsong face is substituted; ``None`` uses the bundled Huiwen
            Fangsong, replacing BabelDOC's own LXGW WenKai GB. Point this at a
            file to use a third; see :mod:`local_translator.engine.fonts`.
        script_font_serif (bool): Whether that file is a serif face. A Fangsong
            or Song face is, a Kai or Hei face is not, and the font matcher
            believes what it is told.
        rebalance_page_breaks (bool): Move the shorter half of a sentence a page
            break cut in two onto the other page before translating, so neither
            half reaches the model as a fragment. See
            :mod:`local_translator.engine.crosspage`. On by default: measured on
            a 19-page paper it repairs two sentences that would otherwise each
            be translated as half a thought, and moves a clause onto the
            adjacent page to do it — a visible change, but a smaller one than
            「高」 standing alone where 「高质量领域」 belongs.
    """

    output_dual: bool = True
    output_mono: bool = True
    use_alternating_pages_dual: bool = False
    dual_translate_first: bool = False
    watermark_mode: str = "no_watermark"
    skip_clean: bool = False
    skip_form_render: bool = False
    restore_images: bool = True
    restore_links: bool = True

    # Target resolution for images in the finished PDF; 0 leaves them alone.
    # Off by default: resampling rewrites every figure in the document, and a
    # default that silently modifies output is the wrong default even when the
    # measured difference is small. Reader chooses; 300 keeps print quality.
    # 成品 PDF 中图像的目标分辨率；0 表示不处理。默认关闭：重采样会重写文档里的每一张插图，
    # 而一个会静默改动产物的默认值即便实测差异很小也仍是错的默认值。由读者选择；300 保持
    # 印刷质量。
    compress_dpi: int = 0
    preserve_regions: bool = True
    preserve_kinds: list[str] = Field(default_factory=lambda: list(PRESERVE_KINDS))
    translate_headings: bool = False
    translate_contents: bool = False
    enhance_compatibility: bool = False
    auto_enable_ocr_workaround: bool = True
    primary_font_family: str | None = None
    script_font: str | None = None
    script_font_serif: bool = True
    rebalance_page_breaks: bool = True


class LayoutFilterSettings(BaseModel):
    """
    中文：选择性翻译设置——哪些版面区域跳过不译。

    Selective-translation settings. Paragraphs whose ``layout_label`` appears in
    ``skip_labels`` keep their original text instead of being translated.

    Attributes:
        enabled (bool): Master switch for selective translation.
        preset (str): Key into :data:`LAYOUT_PRESETS`, or ``"custom"``.
        skip_labels (list[str]): Effective skip set. Kept in sync with ``preset``
            unless the preset is ``"custom"``.
    """

    enabled: bool = True
    preset: str = "pdfmath"
    skip_labels: list[str] = Field(default_factory=lambda: list(LAYOUT_PRESETS["pdfmath"]))

    def effective_skip_labels(self) -> set[str]:
        """
        中文：返回实际生效的跳过标签集合。

        Resolve the active skip set, honouring the preset unless it is custom.

        Returns:
            set[str]: Layout labels to leave untranslated. Empty when disabled.
        """
        if not self.enabled:
            return set()
        if self.preset in LAYOUT_PRESETS and self.preset != "custom":
            return set(LAYOUT_PRESETS[self.preset])
        return set(self.skip_labels)


class GlossarySettings(BaseModel):
    """
    中文：术语库设置，包括累积策略与白名单保护强度。

    Settings for the cumulative glossary that keeps terminology consistent
    across documents and progressively reduces reliance on the API model.

    Attributes:
        enabled (bool): Inject matched glossary terms into translation prompts.
        max_terms_per_prompt (int): Cap on injected terms per paragraph, to keep
            the prompt short enough for a 7B translation model.
        auto_extract (bool): Allow API-based term extraction when coverage is
            low. Off, and not currently offered in the interface. Extraction
            runs after every document and wrote straight into the glossary, so
            it kept refilling the glossary with what the last cleaning had just
            removed — five test runs put 2,087 entries back, none of them
            filed under a subject. It stays off until intake is good enough to
            be worth the API call.
        coverage_threshold (float): If the existing glossary already covers this
            fraction of detected technical terms, skip API extraction entirely.
        hard_protect_whitelist (bool): Extend marker protection of
            ``policy='keep'`` terms to running prose. Off: a marker is a token
            where a word was, so it can disturb word order, and in prose a rare
            leak reads better than a sentence built around a placeholder.
            Figures and tables are protected regardless of this setting, because
            there is no prose there to disturb and a translated 「baseline」 in an
            axis label is exactly what the whitelist exists to prevent. The
            prompt-level identity mapping used elsewhere is obeyed most of the
            time but not always: measured on one page of arXiv:2608.19072 the
            model kept 「Agent」, 「token」 and 「pass@」 every time and still
            rendered 「baseline」 as 「基线」 five times.
        glossary_rev (int): Bumped on manual glossary edits; mixed into the
            translation cache key so hand-corrected terms take effect on retranslation.
    """

    enabled: bool = True
    max_terms_per_prompt: int = 24
    auto_extract: bool = False
    coverage_threshold: float = 0.8
    hard_protect_whitelist: bool = False
    glossary_rev: int = 1


class ServerSettings(BaseModel):
    """
    中文：Web 服务监听设置。

    Web server binding for the NiceGUI + FastAPI process.

    Attributes:
        host (str): Bind address; loopback by default so the service is not
            exposed to the local network.
        port (int): TCP port for the web UI.
        max_concurrent_jobs (int): How many documents translate at once.
            Two, not one: translation is GPU-bound while parsing, typesetting
            and PDF writing are CPU-bound, so overlapping two documents hides
            one document's pipeline behind another's translation. Raising it
            further only queues work against a saturated GPU.
    """

    host: str = "127.0.0.1"
    port: int = 8080
    max_concurrent_jobs: int = 2


class LatexSettings(BaseModel):
    """
    中文：从 LaTeX 源码翻译论文的设置。

    Settings for translating a paper from its LaTeX source.

    Translating the source rather than the printed page removes a whole class
    of defect — nothing is clipped, rescaled or overlapped, because TeX lays
    the page out from scratch. It needs two things the PDF path does not: the
    paper's source, which only arXiv reliably provides, and a TeX engine.

    Attributes:
        prefer_for_arxiv (bool): Translate an arXiv paper from its source when
            the source is available. Falls back to the PDF when it is not, so
            turning this on never costs a translation.
        engine_path (str): A TeX engine to use instead of the bundled one.
            Empty means use whatever the installation provides. A system
            xelatex has better macro coverage than one that downloads packages
            on demand, so a reader who has TeX Live should say so here.
        cjk_font (str): Font family for the Chinese text. Empty means pick one
            the machine actually has.
        workers (int): How many blocks to translate at once.
        keep_sources (bool): Keep the translated ``.tex`` and its working
            directory after compiling. What a reader needs in order to fix a
            paper that did not compile, and useless clutter otherwise.
    """

    prefer_for_arxiv: bool = True
    engine_path: str = ""
    cjk_font: str = ""
    workers: int = 8
    keep_sources: bool = True


class Settings(BaseModel):
    """
    中文：应用全部配置的聚合根，序列化为 settings.json。

    Root settings object, persisted as JSON. Loaded once at startup and saved
    whenever the user changes something in the settings page.

    Attributes:
        local_model (LocalModelSettings): Local llama-server configuration.
        api_model (ApiModelSettings): Optional cloud model configuration.
        translation (TranslationSettings): Language pair and throughput.
        pdf (PdfSettings): Output layout options.
        layout_filter (LayoutFilterSettings): Selective translation.
        glossary (GlossarySettings): Cumulative glossary behaviour.
        server (ServerSettings): Web server binding.
        latex (LatexSettings): Translating from LaTeX source rather than PDF.
        low_memory (bool): Trade throughput for footprint wherever the two are
            in tension. At present it selects how the document layout model is
            loaded — see :mod:`local_translator.engine.layout_model` — which is
            worth about 0.7 GB on every translation. Defaults on for the mini
            edition and off for the others, so an installation that has the
            memory keeps the faster path.
    """

    low_memory: bool = False

    @model_validator(mode="after")
    def _low_memory_follows_the_edition(self) -> Settings:
        """
        中文：本地轻量版按实测结果配好低内存模式与线程数。

        Give the mini edition the settings measured for it.

        Low-memory mode, and a worker pool sized to its slot count. Both are
        only applied where the reader left the shipped default, so an explicit
        choice survives.

        Returns:
            Settings: ``self``.
        """
        from local_translator import edition

        if edition.current() != edition.MINI:
            return self
        if self.low_memory == Settings.model_fields["low_memory"].default:
            self.low_memory = True
        # Worker threads follow the slot count: oversubscribing two to one keeps
        # every slot busy across BabelDOC's per-page synchronisation points,
        # which is where the full edition's 16-for-8 ratio comes from.
        # 工作线程数跟随槽位数：二比一的超额订阅能让每个槽位在 BabelDOC 的逐页同步点
        # 之间都有活干，完整版 8 槽 16 线程正是这个比例。
        workers = TranslationSettings.model_fields["pool_max_workers"].default
        if self.translation.pool_max_workers == workers:
            self.translation.pool_max_workers = self.local_model.n_parallel * 2
        return self

    local_model: LocalModelSettings = Field(default_factory=LocalModelSettings)
    api_model: ApiModelSettings = Field(default_factory=ApiModelSettings)
    translation: TranslationSettings = Field(default_factory=TranslationSettings)
    pdf: PdfSettings = Field(default_factory=PdfSettings)
    layout_filter: LayoutFilterSettings = Field(default_factory=LayoutFilterSettings)
    glossary: GlossarySettings = Field(default_factory=GlossarySettings)
    server: ServerSettings = Field(default_factory=ServerSettings)
    latex: LatexSettings = Field(default_factory=LatexSettings)

    @classmethod
    def load(cls, path: Path) -> Settings:
        """
        中文：从 JSON 文件加载配置，文件缺失或损坏时回退到默认值。

        Load settings from disk, tolerating a missing or corrupt file.

        Unknown keys are ignored and missing keys fall back to their defaults,
        so upgrading the app never hard-fails on an older settings file.

        Args:
            path (Path): Path to ``settings.json``.

        Returns:
            Settings: Parsed settings, or a default instance on any failure.
        """
        if not path.exists():
            return cls()
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            _repair(raw)
            return cls.model_validate(raw)
        except Exception:
            logger.warning("Failed to read settings from %s, using defaults", path, exc_info=True)
            return cls()

    def save(self, path: Path) -> None:
        """
        中文：原子写入配置文件，避免中途崩溃损坏文件。

        Persist settings to disk atomically (write to a temporary file in the
        same directory, then rename) so an interrupted write cannot corrupt the
        existing configuration.

        Args:
            path (Path): Destination ``settings.json`` path. Parent directories
                are created if needed.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(
            json.dumps(self.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp.replace(path)


# --------------------------------------------------------------------------- #
# Module-level accessors / 模块级访问器
# --------------------------------------------------------------------------- #

_paths: AppPaths | None = None
_settings: Settings | None = None


def get_paths() -> AppPaths:
    """
    中文：返回进程内共享的 AppPaths 单例（首次调用会创建目录）。

    Return the process-wide :class:`AppPaths` singleton, creating the directory
    tree on first access.

    Returns:
        AppPaths: Shared path registry.
    """
    global _paths
    if _paths is None:
        _paths = AppPaths().ensure()
    return _paths


# Settings whose stored value was wrong rather than chosen, mapped to what they
# must become. A default change alone does not reach an existing installation:
# the old value is already written into settings.json and wins on load.
# 曾经存错（而非用户选择）的配置项，及其必须被改成的值。只改默认值救不了已装好的
# 环境——旧值已经写进 settings.json，加载时会覆盖默认值。
_FORCED: dict[tuple[str, str], object] = {
    # Shipped as True on the belief that a "form" was an interactive AcroForm
    # widget. It is a Form XObject invocation — the container figures, vector
    # graphics and inline images are drawn through — so skipping it dropped
    # every figure's own content from the output.
    # 当初设为 True，是误以为 form 指交互式表单控件。它其实是 Form XObject 的调用，
    # 也就是图、矢量图与内联图像的绘制容器，跳过它等于把每张图自己的内容整个丢掉。
    ("pdf", "skip_form_render"): False,
    # Turned off while the feature is withdrawn from the interface; a stored
    # True from before would otherwise keep extraction running.
    # 该功能已从界面撤下，此处一并关闭；否则此前存下的 True 会让提取继续运行。
    ("glossary", "auto_extract"): False,
}


def _repair(raw: dict) -> None:
    """
    中文：加载时修正曾经存错的配置项。

    Overwrite stored settings that were wrong rather than chosen.

    Args:
        raw (dict): Parsed ``settings.json``, edited in place.

    Returns:
        None.
    """
    for (section, key), value in _FORCED.items():
        block = raw.get(section)
        if isinstance(block, dict) and block.get(key) != value:
            logger.info("Repairing setting %s.%s: %r -> %r",
                        section, key, block.get(key), value)
            block[key] = value


def get_settings(reload: bool = False) -> Settings:
    """
    中文：返回进程内共享的 Settings 单例。

    Return the process-wide :class:`Settings` singleton, loading it from disk on
    first access.

    Args:
        reload (bool): Force a re-read from disk, discarding in-memory changes.

    Returns:
        Settings: Shared settings object.
    """
    global _settings
    if _settings is None or reload:
        _settings = Settings.load(get_paths().settings_path)
    return _settings


def save_settings(settings: Settings | None = None) -> None:
    """
    中文：保存配置到磁盘并刷新单例。

    Persist the given settings (or the current singleton) and make it the
    active in-memory instance.

    Args:
        settings (Settings | None): Settings to save. Defaults to the singleton.
    """
    global _settings
    s = settings or get_settings()
    s.save(get_paths().settings_path)
    _settings = s
