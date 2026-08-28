"""
中文：在流水线换阶段时回收一次内存——上一阶段的垃圾不该让下一阶段去顶。

Collect garbage between pipeline stages, so one stage's rubbish does not raise
the next stage's high-water mark.

**Where the memory actually is.** Measured on a 19-page paper, the worker peaks
at 1.63 GB while ``tracemalloc`` reports only 0.19 GB of live Python objects.
``vmmap`` explains the gap: at the peak the process holds 610 MB of
``MALLOC_LARGE``, 316 MB of ``MALLOC_SMALL`` and — the part that matters here —
185 MB of ``MALLOC_LARGE (empty)`` plus 73 MB of ``MALLOC_SMALL (empty)``.
"Empty" is the allocator's word for pages it has freed and not returned.

Almost none of that is Python data. It is the native buffers behind Python
wrappers: MuPDF pixmaps, the NumPy images the layout model is fed, ONNX Runtime
arenas. A wrapper that is part of a reference cycle is not freed by reference
counting, so its buffer stays allocated until a cycle collection runs — and
CPython runs one on its own schedule, which is driven by object *counts* and
knows nothing about the megabytes hanging off them.

**What collecting buys.** Running one collection as each stage begins does not
lower the reading at that moment; it lowers the next stage's, because the
allocator now has free blocks to reuse instead of growing the heap:

===============================  ==========  ==========
stage                            unchanged   collecting
===============================  ==========  ==========
after the native parse             0.91 GB     0.90 GB
after the layout parser            1.29 GB     1.22 GB
after styles and formulas          1.47 GB     1.39 GB
**before translation begins**    **1.63 GB** **1.43 GB**
===============================  ==========  ==========

0.20 GB off the peak, which is the number that decides whether a machine can run
this at all — and the peak is reached while the model server is also resident.

``malloc_zone_pressure_relief`` was tried alongside and returns nothing once the
collection has run: the collection's own frees already hand the pages back.
"""

from __future__ import annotations

import gc
import logging
import time

logger = logging.getLogger(__name__)


def install(enabled: bool = True) -> None:
    """
    中文：在每个流水线阶段开始时回收一次循环垃圾。

    Run a cycle collection at the start of every pipeline stage.

    Hooks ``ProgressMonitor.stage_start``, which BabelDOC calls once per stage
    and nowhere else, so this follows the pipeline without naming any of its
    steps — a stage added or renamed upstream is covered without a change here.

    Args:
        enabled (bool): When false, does nothing, so a caller can pass a setting
            straight through.

    Raises:
        AttributeError: If BabelDOC no longer exposes ``stage_start``, so an
            upgrade that moves it fails at startup rather than silently giving
            up the saving.
    """
    if not enabled:
        return

    from babeldoc.progress_monitor import ProgressMonitor

    if getattr(ProgressMonitor.stage_start, "_reclaim_patched", False):
        return

    original = ProgressMonitor.stage_start
    spent = [0.0]

    def stage_start(self, stage_name: str, total: int):
        """中文：进入阶段前先回收上一阶段的垃圾。 / Clear the previous stage's garbage first."""
        began = time.monotonic()
        collected = gc.collect()
        spent[0] += time.monotonic() - began
        if collected:
            logger.debug("Collected %d cyclic object(s) entering %s (%.2fs total so far)",
                         collected, stage_name, spent[0])
        return original(self, stage_name, total)

    stage_start._reclaim_patched = True
    ProgressMonitor.stage_start = stage_start
