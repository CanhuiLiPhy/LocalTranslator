"""
中文：PP-OCRv6 推理——文本检测（DB）与文本识别（CTC）。

PP-OCRv6 inference: DB text detection followed by CTC recognition.

Two models run in sequence. Detection produces a per-pixel probability map from
which text boxes are recovered; recognition reads each cropped box into a string.
Both are the official ONNX exports, driven through ``onnxruntime``.

**On the unclip step.** Detection shrinks text regions during training, so the
predicted boxes must be expanded again before cropping. PaddleOCR does this with
``pyclipper``'s polygon offset. That dependency is avoided here: the boxes are
already minimum-area *rectangles*, and offsetting a rectangle outward by a fixed
distance is the same as growing its width and height by twice that distance —
which numpy does directly. The result is identical for rectangles and differs
only for genuinely non-convex polygons, which text lines are not. Avoiding the
dependency matters because the whole point of this OCR backend is that it adds
no new packages to the project.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Detection input is padded to a multiple of this, as the network downsamples by
# 32 and a ragged edge would misalign the output map against the image.
# 检测输入需补齐到该值的整数倍：网络下采样 32 倍，边缘不齐会使输出图与原图错位。
_STRIDE = 32

# Longest side the detection input is scaled to. Larger finds smaller text at
# quadratic cost; 960 reads body text in a screenshot without being slow.
# 检测输入的最长边。更大能找到更小的字但代价是平方增长；960 足以读出截图正文。
_DET_LIMIT = 960

# ImageNet statistics, as the detection model was normalised with them.
# ImageNet 统计量，检测模型按此归一化。
_DET_MEAN = (0.485, 0.456, 0.406)
_DET_STD = (0.229, 0.224, 0.225)

# Recognition reads a fixed-height strip; width follows the aspect ratio.
# 识别读取固定高度的条带，宽度按长宽比缩放。
_REC_HEIGHT = 48
_REC_MAX_WIDTH = 3200


@dataclass
class TextLine:
    """
    中文：一行识别结果。

    One recognised line of text.

    Attributes:
        text (str): Recognised characters.
        confidence (float): Mean per-character probability, in ``[0, 1]``.
        box (list[tuple[float, float]]): Four corner points in image
            coordinates, clockwise from the top left.
    """

    text: str
    confidence: float
    box: list[tuple[float, float]]

    @property
    def top(self) -> float:
        """中文：框的上边界。 / Topmost y of the box."""
        return min(point[1] for point in self.box)

    @property
    def left(self) -> float:
        """中文：框的左边界。 / Leftmost x of the box."""
        return min(point[0] for point in self.box)

    @property
    def height(self) -> float:
        """中文：框的高度。 / Box height."""
        ys = [point[1] for point in self.box]
        return max(ys) - min(ys)


@dataclass
class OcrResult:
    """
    中文：一张图片的识别结果。

    Recognition result for one image.

    Attributes:
        lines (list[TextLine]): Recognised lines in reading order.
        width (int): Source image width in pixels.
        height (int): Source image height in pixels.
        seconds (float): Wall-clock time spent recognising.
    """

    lines: list[TextLine]
    width: int
    height: int
    seconds: float = 0.0

    def as_text(self, paragraph_gap: float = 1.6) -> str:
        """
        中文：把行合并成带段落分隔的纯文本。

        Join the lines into plain text with paragraph breaks.

        Lines separated by more than ``paragraph_gap`` times the local line
        height are treated as different paragraphs. Feeding a translator one
        line at a time would cut sentences at the line wrap and cost the model
        the context it needs; joining everything into one block would merge a
        heading into the paragraph beneath it.

        Args:
            paragraph_gap (float): Vertical gap, in line heights, above which a
                paragraph break is inserted.

        Returns:
            str: Text with single newlines inside a paragraph and blank lines
                between paragraphs.
        """
        if not self.lines:
            return ""

        chunks: list[list[str]] = [[self.lines[0].text]]
        for previous, line in zip(self.lines, self.lines[1:], strict=False):
            reference = max(previous.height, line.height, 1.0)
            if (line.top - previous.top) > paragraph_gap * reference:
                chunks.append([line.text])
            else:
                chunks[-1].append(line.text)
        return "\n\n".join(" ".join(chunk) for chunk in chunks)


class PPOcr:
    """
    中文：PP-OCRv6 检测 + 识别引擎。

    PP-OCRv6 detection and recognition engine.

    Attributes:
        root (Path): Directory holding ``det/`` and ``rec/`` model folders.
    """

    def __init__(self, root: Path):
        self.root = Path(root)
        self._lock = threading.Lock()
        self._det = None
        self._rec = None
        self._charset: list[str] = []

    # ------------------------------------------------------------------ #
    # Loading / 加载
    # ------------------------------------------------------------------ #

    @staticmethod
    def _providers() -> list[str]:
        """
        中文：优先使用 CoreML，回退到 CPU。

        Prefer CoreML where available, falling back to CPU.

        Returns:
            list[str]: onnxruntime execution providers, best first.
        """
        import onnxruntime as ort

        available = set(ort.get_available_providers())
        order = ["CoreMLExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
        return [p for p in order if p in available] or ["CPUExecutionProvider"]

    def available(self) -> bool:
        """
        中文：模型文件是否齐全。

        Report whether the model files are present.

        Returns:
            bool: ``True`` when both stages can be loaded.
        """
        return (self.root / "det" / "inference.onnx").is_file() and (
            self.root / "rec" / "inference.onnx"
        ).is_file()

    def _load(self) -> None:
        """
        中文：按需加载两个模型与字符表（线程安全，只加载一次）。

        Load both models and the character set on first use.

        Raises:
            FileNotFoundError: If the model files are missing.
        """
        if self._det is not None:
            return
        with self._lock:
            if self._det is not None:
                return
            if not self.available():
                raise FileNotFoundError(f"OCR models not installed under {self.root}")

            import onnxruntime as ort
            import yaml

            providers = self._providers()
            options = ort.SessionOptions()
            options.log_severity_level = 3

            self._det = ort.InferenceSession(
                str(self.root / "det" / "inference.onnx"),
                sess_options=options, providers=providers,
            )
            self._rec = ort.InferenceSession(
                str(self.root / "rec" / "inference.onnx"),
                sess_options=options, providers=providers,
            )

            config = yaml.safe_load((self.root / "rec" / "inference.yml").read_text(encoding="utf-8"))
            characters = list(config["PostProcess"]["character_dict"])
            # CTC reserves index 0 for the blank label, and the trailing space is
            # appended by the pipeline rather than stored in the dictionary.
            # CTC 保留索引 0 作为 blank，末尾的空格由管线追加而非存在字典中。
            self._charset = ["<blank>"] + characters + [" "]

            expected = self._rec.get_outputs()[0].shape[-1]
            if isinstance(expected, int) and expected != len(self._charset):
                logger.warning(
                    "Recognition head has %d classes but the character set has %d; "
                    "decoding may be misaligned", expected, len(self._charset),
                )
            logger.info("PP-OCRv6 loaded (%s), %d classes",
                        providers[0], len(self._charset))

    # ------------------------------------------------------------------ #
    # Detection / 检测
    # ------------------------------------------------------------------ #

    def _detect(self, image) -> list:
        """
        中文：检测文本框，返回原图坐标下的四点框。

        Detect text boxes.

        Args:
            image (numpy.ndarray): BGR image.

        Returns:
            list[numpy.ndarray]: Four-point boxes in source image coordinates.
        """
        import cv2
        import numpy as np

        height, width = image.shape[:2]
        scale = min(_DET_LIMIT / max(height, width), 1.0)
        # Round to the network stride in both directions, never to zero.
        # 两个方向都对齐到网络步长，且不得为 0。
        new_w = max(_STRIDE, int(round(width * scale / _STRIDE)) * _STRIDE)
        new_h = max(_STRIDE, int(round(height * scale / _STRIDE)) * _STRIDE)
        resized = cv2.resize(image, (new_w, new_h))

        blob = resized[:, :, ::-1].astype(np.float32) / 255.0
        blob = (blob - np.array(_DET_MEAN, np.float32)) / np.array(_DET_STD, np.float32)
        blob = blob.transpose(2, 0, 1)[None]

        prob = self._det.run(None, {self._det.get_inputs()[0].name: blob})[0][0, 0]

        mask = (prob > 0.2).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        ratio_w, ratio_h = width / new_w, height / new_h
        boxes = []
        for contour in contours[:3000]:
            if cv2.contourArea(contour) < 4:
                continue
            rect = cv2.minAreaRect(contour)
            (cx, cy), (bw, bh), angle = rect
            if min(bw, bh) < 3:
                continue

            score = self._box_score(prob, contour)
            if score < 0.45:
                continue

            # Unclip: expand outward by area * ratio / perimeter, which for a
            # rectangle is exactly a uniform growth of both sides.
            # 外扩：距离为 面积 × 比例 ÷ 周长；对矩形而言即两边各均匀增长。
            area, perimeter = bw * bh, 2.0 * (bw + bh)
            if perimeter <= 0:
                continue
            distance = area * 1.4 / perimeter
            box = cv2.boxPoints(((cx, cy), (bw + 2 * distance, bh + 2 * distance), angle))

            box[:, 0] = np.clip(box[:, 0] * ratio_w, 0, width - 1)
            box[:, 1] = np.clip(box[:, 1] * ratio_h, 0, height - 1)
            boxes.append(self._order(box))

        # Reading order: top to bottom, then left to right within a line. The
        # row tolerance is half a box height, so boxes that sit side by side are
        # not reordered by a few pixels of vertical jitter.
        # 阅读顺序：先上下、同一行内再左右。行容差取半个框高，避免并排的框因几像素
        # 的垂直抖动被打乱顺序。
        boxes.sort(key=lambda b: (round(b[:, 1].min() / max(1.0, (b[:, 1].max() - b[:, 1].min())) * 2),
                                  b[:, 0].min()))
        return boxes

    @staticmethod
    def _box_score(prob, contour) -> float:
        """
        中文：以轮廓内的平均概率作为该框的置信度。

        Score a detection by the mean probability inside its contour.

        Args:
            prob (numpy.ndarray): Probability map.
            contour (numpy.ndarray): Contour points.

        Returns:
            float: Mean probability, or 0 when the contour is degenerate.
        """
        import cv2
        import numpy as np

        h, w = prob.shape
        x0 = int(np.clip(contour[:, 0, 0].min(), 0, w - 1))
        x1 = int(np.clip(contour[:, 0, 0].max(), 0, w - 1))
        y0 = int(np.clip(contour[:, 0, 1].min(), 0, h - 1))
        y1 = int(np.clip(contour[:, 0, 1].max(), 0, h - 1))
        if x1 <= x0 or y1 <= y0:
            return 0.0

        mask = np.zeros((y1 - y0 + 1, x1 - x0 + 1), dtype=np.uint8)
        shifted = contour.copy()
        shifted[:, 0, 0] -= x0
        shifted[:, 0, 1] -= y0
        cv2.fillPoly(mask, [shifted.astype(np.int32)], 1)
        region = prob[y0:y1 + 1, x0:x1 + 1]
        return float(region[mask == 1].mean()) if mask.any() else 0.0

    @staticmethod
    def _order(box):
        """
        中文：把四点排成顺时针（左上起）。

        Order four points clockwise from the top left.

        Args:
            box (numpy.ndarray): Four points in arbitrary order.

        Returns:
            numpy.ndarray: Ordered points.
        """
        import numpy as np

        ordered = np.zeros((4, 2), dtype=np.float32)
        total = box.sum(axis=1)
        ordered[0] = box[np.argmin(total)]
        ordered[2] = box[np.argmax(total)]
        diff = np.diff(box, axis=1).ravel()
        ordered[1] = box[np.argmin(diff)]
        ordered[3] = box[np.argmax(diff)]
        return ordered

    # ------------------------------------------------------------------ #
    # Recognition / 识别
    # ------------------------------------------------------------------ #

    @staticmethod
    def _crop(image, box):
        """
        中文：按四点做透视变换，裁出水平的文字条。

        Rectify a quadrilateral into a horizontal strip.

        A perspective transform is used rather than a bounding-box crop so that
        slightly rotated lines — a photographed page, a skewed screenshot — are
        straightened before recognition rather than read at an angle.

        Args:
            image (numpy.ndarray): BGR source image.
            box (numpy.ndarray): Four ordered corner points.

        Returns:
            numpy.ndarray: Rectified crop.
        """
        import cv2
        import numpy as np

        width = int(max(np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[2] - box[3])))
        height = int(max(np.linalg.norm(box[0] - box[3]), np.linalg.norm(box[1] - box[2])))
        width, height = max(width, 1), max(height, 1)

        target = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
        matrix = cv2.getPerspectiveTransform(box.astype(np.float32), target)
        crop = cv2.warpPerspective(image, matrix, (width, height),
                                   borderMode=cv2.BORDER_REPLICATE)

        # A crop much taller than it is wide is a vertical line of text; rotating
        # it makes it readable by a model trained on horizontal strips.
        # 高远大于宽的裁剪是竖排文字；旋转后才能被横排训练的模型读出。
        if height > width * 1.5:
            crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
        return crop

    def _recognise(self, crops) -> list[tuple[str, float]]:
        """
        中文：批量识别裁剪出的文字条。

        Recognise a batch of cropped strips.

        Crops are sorted by aspect ratio before batching so that strips of
        similar width share a batch. Padding a short strip out to the width of
        the longest in its batch is wasted computation, and grouping by shape
        keeps that waste small.

        Args:
            crops (list[numpy.ndarray]): Rectified strips.

        Returns:
            list[tuple[str, float]]: ``(text, confidence)`` per input, in input
                order.
        """
        import cv2
        import numpy as np

        if not crops:
            return []

        order = sorted(range(len(crops)),
                       key=lambda i: crops[i].shape[1] / max(crops[i].shape[0], 1))
        results: list[tuple[str, float]] = [("", 0.0)] * len(crops)
        name = self._rec.get_inputs()[0].name

        for start in range(0, len(order), 8):
            chunk = order[start:start + 8]
            strips = []
            for index in chunk:
                crop = crops[index]
                ratio = crop.shape[1] / max(crop.shape[0], 1)
                width = min(_REC_MAX_WIDTH, max(16, int(_REC_HEIGHT * ratio)))
                strips.append(cv2.resize(crop, (width, _REC_HEIGHT)))

            batch_width = max(strip.shape[1] for strip in strips)
            batch = np.zeros((len(strips), 3, _REC_HEIGHT, batch_width), dtype=np.float32)
            for row, strip in enumerate(strips):
                blob = strip[:, :, ::-1].astype(np.float32) / 255.0
                blob = (blob - 0.5) / 0.5
                batch[row, :, :, : strip.shape[1]] = blob.transpose(2, 0, 1)

            logits = self._rec.run(None, {name: batch})[0]
            for row, index in enumerate(chunk):
                results[index] = self._ctc_decode(logits[row])

        return results

    def _ctc_decode(self, logits) -> tuple[str, float]:
        """
        中文：CTC 贪心解码，去重并跳过 blank。

        Greedy CTC decode: collapse repeats, drop blanks.

        Args:
            logits (numpy.ndarray): ``(time, classes)`` scores for one strip.

        Returns:
            tuple[str, float]: Decoded text and mean confidence of the kept
                characters.
        """
        import numpy as np

        indices = logits.argmax(axis=1)
        scores = logits.max(axis=1)

        characters: list[str] = []
        kept: list[float] = []
        previous = -1
        for position, index in enumerate(indices):
            if index != previous and index != 0:
                if index < len(self._charset):
                    characters.append(self._charset[index])
                    kept.append(float(scores[position]))
            previous = int(index)

        text = "".join(characters).strip()
        return text, float(np.mean(kept)) if kept else 0.0

    # ------------------------------------------------------------------ #
    # Entry point / 入口
    # ------------------------------------------------------------------ #

    def read(self, image_path: Path | str | bytes, min_confidence: float = 0.5) -> OcrResult:
        """
        中文：识别一张图片，返回按阅读顺序排列的文本行。

        Recognise an image.

        Args:
            image_path (Path | str | bytes): Image file path, or raw image bytes.
            min_confidence (float): Discard lines below this mean confidence.
                Low-scoring lines are usually texture read as characters, and a
                plausible-looking wrong line is worse than a missing one.

        Returns:
            OcrResult: Recognised lines with their boxes.

        Raises:
            ValueError: If the image cannot be decoded.
        """
        import time

        import cv2
        import numpy as np

        self._load()
        started = time.monotonic()

        if isinstance(image_path, bytes):
            buffer = np.frombuffer(image_path, dtype=np.uint8)
            image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        else:
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("无法读取图片")

        boxes = self._detect(image)
        crops = [self._crop(image, box) for box in boxes]
        recognised = self._recognise(crops)

        lines = [
            TextLine(text=text, confidence=confidence,
                     box=[(float(x), float(y)) for x, y in box])
            for box, (text, confidence) in zip(boxes, recognised, strict=True)
            if text and confidence >= min_confidence
        ]

        return OcrResult(lines=lines, width=image.shape[1], height=image.shape[0],
                         seconds=round(time.monotonic() - started, 2))


_ENGINE: PPOcr | None = None
_LOCK = threading.Lock()


def get_ocr(root: Path | None = None) -> PPOcr:
    """
    中文：返回进程内共享的 OCR 引擎。

    Return the process-wide OCR engine.

    Args:
        root (Path | None): Model directory; defaults to the configured OCR
            model path.

    Returns:
        PPOcr: Shared engine. Models load on first use, not here.
    """
    global _ENGINE
    with _LOCK:
        if _ENGINE is None:
            if root is None:
                from local_translator.config import get_paths

                root = get_paths().ocr_models_dir / "PP-OCRv6_small"
            _ENGINE = PPOcr(root)
        return _ENGINE
