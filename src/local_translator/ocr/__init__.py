"""
中文：本地 OCR——基于 PP-OCRv6 的 ONNX 模型，识别图片中的文字。

Local OCR built on PP-OCRv6.

A text-only chat endpoint cannot read an image — it answers an image request
with ``is not a multimodal model`` — and many OpenAI-compatible providers serve
exactly that. Recognition therefore happens locally and only the recognised text
is sent for translation, which also keeps the image itself on this machine.

PP-OCRv6 fits this project unusually well. Its ONNX export runs on the
``onnxruntime`` that BabelDOC already depends on, and the post-processing needs
only ``numpy``, ``opencv`` and ``pyyaml``, which are likewise already present.
The small tier is 31 MB for both stages, covers 50 languages in one model, and
runs anywhere — including the memory-constrained server, which a macOS-only OCR
framework could not.
"""

from local_translator.ocr.engine import OcrResult, PPOcr, TextLine, get_ocr

__all__ = ["OcrResult", "PPOcr", "TextLine", "get_ocr"]
