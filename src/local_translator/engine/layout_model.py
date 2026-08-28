"""
中文：低内存地加载版面模型——省下约 0.7 GB，代价是每页多几百毫秒。

Load the document layout model without paying for CoreML.

**Where the memory goes.** BabelDOC's ``OnnxModel.__init__`` does four expensive
things on macOS, measured on the 72 MB DocLayout-YOLO weights:

1. ``onnx.load(path)`` materialises the whole graph as a Python protobuf;
2. the graph is rewritten to a static ``[1, 3, 1024, 1024]`` input and run
   through shape inference, which builds a second copy;
3. ``model.SerializeToString()`` produces a third copy, as bytes;
4. the session is created on ``CoreMLExecutionProvider``, which compiles and
   holds its own copy of 657 of the 681 nodes.

The result is **+1.03 GB resident for a 72 MB file**, and it is paid on every
translation because the model is loaded in the worker process.

**What this does instead.** One ``InferenceSession`` on the CPU provider,
constructed from the file path so onnxruntime memory-maps the weights rather
than being handed a Python bytes object, with the arena allocator off. Nothing
is loaded twice.

    加载后 RSS      推理峰值      每页
    CoreML（现状）      1.15 GB       1.28 GB       67 ms
    CPU（本模块）       0.26 GB       0.60 GB      246 ms

Two hundred milliseconds a page is a few seconds on a paper, against two thirds
of a gigabyte on every run — which is the right trade only when memory is the
binding constraint, so this is opt-in via ``Settings.low_memory`` and on by
default in the mini edition alone.

The class metadata BabelDOC reads out of the protobuf (``stride``, ``names``) is
taken from ``get_modelmeta().custom_metadata_map`` instead, so the protobuf is
never loaded at all.

**And it is loaded twice.** ``DocLayoutModel.load_onnx`` has no cache, so each
caller builds its own session; a single translation logs the load twice. The
model holds no per-document state, so :func:`_cache_the_model` memoises it and
one copy serves the run.
"""

from __future__ import annotations

import ast
import logging
import threading

logger = logging.getLogger(__name__)

# Threads for layout inference. 0 lets onnxruntime match the machine, which on
# the measurements above is what recovers most of the speed: 809 ms/page at one
# thread against 246 ms at auto, for the same 0.6 GB.
# 版面推理的线程数。0 表示交给 onnxruntime 按机器决定——实测这正是把速度找回来的
# 关键：单线程 809ms/页，auto 246ms/页，而内存同为 0.6 GB。
_THREADS = 0


def install(enabled: bool = True) -> None:
    """
    中文：把 BabelDOC 的 ONNX 版面模型换成低内存加载方式。

    Replace BabelDOC's layout-model constructor with a low-memory one.

    Args:
        enabled (bool): When false, does nothing at all — the caller can pass
            the setting straight through rather than branching.

    Raises:
        AttributeError: If BabelDOC no longer exposes ``OnnxModel``, so an
            upgrade that renames it fails at startup rather than silently
            reverting to the memory-hungry path.
    """
    if not enabled:
        return

    import onnxruntime
    from babeldoc.docvision.base_doclayout import DocLayoutModel
    from babeldoc.docvision.doclayout import OnnxModel

    _cache_the_model(DocLayoutModel)

    if getattr(OnnxModel.__init__, "_low_memory_patched", False):
        return

    def __init__(self, model_path: str) -> None:
        """中文：CPU 执行器 + 直接从文件加载。 / Build the session on the CPU provider."""
        self.model_path = model_path

        options = onnxruntime.SessionOptions()
        # The arena pre-reserves in large blocks, which on a constrained machine
        # is memory held and not used.
        # arena 分配器按大块预留，在内存紧张的机器上就是占着不用。
        options.enable_cpu_mem_arena = False
        options.enable_mem_pattern = False
        options.intra_op_num_threads = _THREADS
        options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.model = onnxruntime.InferenceSession(
            model_path, options, providers=["CPUExecutionProvider"],
        )

        metadata = self.model.get_modelmeta().custom_metadata_map
        self._stride = ast.literal_eval(metadata["stride"])
        self._names = ast.literal_eval(metadata["names"])
        self.lock = threading.Lock()
        logger.info("Layout model loaded on the CPU provider (low-memory mode)")

    __init__._low_memory_patched = True
    OnnxModel.__init__ = __init__


def _cache_the_model(DocLayoutModel) -> None:
    """
    中文：让 load_onnx() 只真正加载一次——它原本每次调用都重建一个完整 session。

    Memoise ``DocLayoutModel.load_onnx``.

    BabelDOC's ``load_onnx`` is a plain static method with no caching, so every
    caller builds a fresh ``InferenceSession``. A single translation logs the
    load twice, which means two complete copies of the model are alive at once.
    Memoising is safe because the model holds no per-document state — its only
    mutable member is a lock, which BabelDOC takes around each inference, so one
    shared instance serialises exactly as two separate ones would.

    Args:
        DocLayoutModel (type): BabelDOC's base class carrying ``load_onnx``.
    """
    if getattr(DocLayoutModel.load_onnx, "_cached", False):
        return

    original = DocLayoutModel.load_onnx
    holder: list = []

    def load_onnx():
        """中文：返回缓存的版面模型实例。 / Return the one layout model instance."""
        if not holder:
            holder.append(original())
        return holder[0]

    load_onnx._cached = True
    DocLayoutModel.load_onnx = staticmethod(load_onnx)
