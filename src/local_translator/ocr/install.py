"""
中文：OCR 模型安装——下载 PP-OCRv6 的官方 ONNX 导出。

Download the PP-OCRv6 ONNX models.

The small tier is the default: 31 MB for both stages against 165 MB for medium,
and on printed text — screenshots, figure labels, photographed pages — the gap
in accuracy is small next to the gap in size. The tiny tier exists for machines
where even that is too much, and medium for photographs of poor quality.

Downloads reuse the model downloader, which resumes across dropped connections
and budgets its retries by progress rather than attempt count.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

HF_BASE = "https://huggingface.co/PaddlePaddle"

# Each tier is a detection and a recognition model, published separately.
# 每个档位由检测与识别两个模型组成，分别发布。
TIERS: dict[str, dict[str, object]] = {
    "tiny": {"det": "PP-OCRv6_tiny_det_onnx", "rec": "PP-OCRv6_tiny_rec_onnx",
             "size_mb": 6, "note": "最小，适合算力紧张的机器"},
    "small": {"det": "PP-OCRv6_small_det_onnx", "rec": "PP-OCRv6_small_rec_onnx",
              "size_mb": 31, "note": "默认，印刷体足够准"},
    "medium": {"det": "PP-OCRv6_medium_det_onnx", "rec": "PP-OCRv6_medium_rec_onnx",
               "size_mb": 165, "note": "最准，适合拍摄质量较差的图"},
}

# The recognition config carries the 18,708-character dictionary, so it is not
# optional despite being small.
# 识别配置里带着 18708 字的字符表，虽小但不可省略。
FILES = ("inference.onnx", "inference.yml")


def install_ocr(tier: str = "small", paths=None, progress=None) -> Path:
    """
    中文：下载指定档位的 PP-OCRv6 模型，返回安装目录。

    Download one PP-OCRv6 tier.

    Args:
        tier (str): ``tiny``, ``small`` or ``medium``.
        paths (AppPaths | None): Application paths; resolved from config when
            omitted.
        progress (Callable[[str, float], None] | None): Called with a stage
            label and a fraction in ``[0, 1]``.

    Returns:
        Path: Directory containing ``det/`` and ``rec/``.

    Raises:
        ValueError: If the tier is unknown.
    """
    from local_translator.config import get_paths
    from local_translator.llm.downloader import download_file

    if tier not in TIERS:
        raise ValueError(f"未知的模型档位：{tier!r}")

    paths = paths or get_paths()
    target = paths.ocr_models_dir / f"PP-OCRv6_{tier}"
    spec = TIERS[tier]

    # Detection is roughly a third of the download, recognition the rest.
    # 检测约占三分之一，识别占其余部分。
    spans = {"det": (0.0, 0.35), "rec": (0.35, 1.0)}

    for stage in ("det", "rec"):
        folder = target / stage
        folder.mkdir(parents=True, exist_ok=True)
        low, high = spans[stage]
        label = "下载检测模型" if stage == "det" else "下载识别模型"

        for index, name in enumerate(FILES):
            url = f"{HF_BASE}/{spec[stage]}/resolve/main/{name}"
            # The weights dominate; the config is a rounding error on the bar.
            # 权重占绝大部分，配置文件在进度条上可忽略。
            share = 0.97 if index == 0 else 0.03
            base = low + (high - low) * (0.0 if index == 0 else 0.97)

            # `label` is bound as a default alongside base/span: it is rebound
            # on the next stage, and a closure over it would report the
            # recognition stage's name while the detection stage is still
            # downloading.
            # label 与 base/span 一样绑成默认参数：它在下一阶段会被重新赋值，
            # 闭包捕获会导致检测模型还在下载时就报出识别模型的名字。
            def report(done: int, total: int, label=label, base=base,
                       span=(high - low) * share) -> None:
                if progress and total:
                    progress(label, base + span * min(1.0, done / total))

            download_file(url, folder / name, progress=report)

    if progress:
        progress("完成", 1.0)
    logger.info("Installed PP-OCRv6 %s into %s", tier, target)
    return target
