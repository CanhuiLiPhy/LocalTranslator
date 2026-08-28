"""
中文：混合模式的分流决策——按在途负载而非累计比例决定每段去向。

Routing policy for hybrid translation.

The two backends have opposite scaling behaviour, and the routing rule follows
from that. Measured on one machine against an OpenAI-compatible cloud
endpoint; the absolute numbers depend on the GPU and the provider, the shape
does not:

    concurrency   local (Hy-MT2 7B)   API
    1             316 chars/s         116 chars/s
    4             321 chars/s         325 chars/s
    16            307 chars/s         749 chars/s

The local model is a **fixed-rate resource**: one GPU, so adding concurrent
requests moves no more text per second. The API is **concurrency-elastic**:
throughput grows almost linearly and latency stays flat, so the only way to use
it well is to keep many requests in flight.

An earlier version of this router split the document by a fixed character share
decided in advance. That cannot be right for two resources whose relative speed
is not known until run time — the API's rate depends on how many threads happen
to be free, which depends on the split itself. The rule here is instead
**work-conserving**: a paragraph goes local whenever the local model has a free
slot, and to the API otherwise. Neither side idles while work remains, and the
realised split settles at whatever ratio the two measured rates imply.

For this document that equilibrium is about 30% local, which is also where the
arithmetic puts the optimum:

    t = local_rate·t + api_rate·t   ⟹   t = C / (local_rate + api_rate)
    49402 / (320 + 749) ≈ 46s,   local share = 320/(320+749) ≈ 0.30

Nothing had to be tuned to land there; the slot discipline finds it, and finds
a different point on other hardware or when the API is busy.

Quality still overrides balance. The local model's placeholder fidelity falls
off sharply with paragraph length — measured across 341 paragraphs:

    markers  paragraphs  mean length  local fallback rate
    0        264         104          0.4%
    1–2      52          383          1.9%
    3–5      17          704          5.9%
    6+       8           1929         12.5%

so marker-dense and very long paragraphs go to the API regardless of load.
"""

from __future__ import annotations

import logging
import re
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum

logger = logging.getLogger(__name__)

_MARKERS = re.compile(r"\{\s*v\s*\d+\s*\}|<\s*style[^>]{0,24}>")


class Route(StrEnum):
    """
    中文：段落的去向。

    Destination for one paragraph.
    """

    LOCAL = "local"
    API = "api"


@dataclass
class RouterPolicy:
    """
    中文：分流策略参数。

    Tunable routing thresholds.

    Attributes:
        local_slots (int): Concurrent requests the local model may have in
            flight. This is the load control: it bounds how much of the machine
            the model occupies at once, and because local throughput is flat in
            concurrency, capping it costs no speed. It also sets how much of the
            document ends up local, since a fixed-rate resource with a fixed
            time budget does a fixed amount of work.
        local_share_cap (float): Hard ceiling on the local character share, as a
            safety net. The slot discipline normally settles well below this, so
            it does not usually bind; it exists so a pathological document
            cannot push the machine harder than intended.
        api_marker_threshold (int): Marker count at or above which a paragraph
            always goes to the API, whatever the current load.
        api_length_threshold (int): Character count at or above which a
            paragraph always goes to the API.
    """

    local_slots: int = 8
    local_share_cap: float = 0.6
    api_marker_threshold: int = 3
    api_length_threshold: int = 900


class Router:
    """
    中文：按在途负载决定去向，并跟踪实际分流比例。

    Decide each paragraph's destination and track the realised split.

    Called from BabelDOC's worker threads, so all state is guarded. A ``LOCAL``
    decision atomically reserves a local slot, which the caller releases through
    :meth:`local_slot`; deciding and reserving separately would let several
    threads see the same free slot and overshoot the load limit.

    Attributes:
        policy (RouterPolicy): Thresholds in force.
    """

    def __init__(self, policy: RouterPolicy | None = None):
        self.policy = policy or RouterPolicy()
        self._lock = threading.Lock()
        self._chars = {Route.LOCAL: 0, Route.API: 0}
        self._paras = {Route.LOCAL: 0, Route.API: 0}
        self._forced_api = 0
        self._overflow_api = 0
        self._local_inflight = 0
        self._peak_local_inflight = 0

    def decide(self, text: str) -> Route:
        """
        中文：为一个段落选择去向；选中本地时同时占用一个槽位。

        Choose a destination, reserving a local slot when the answer is local.

        Args:
            text (str): Paragraph text, markers included.

        Returns:
            Route: Where this paragraph should be translated. A ``LOCAL`` result
                holds a slot that :meth:`local_slot` must release.
        """
        markers = len(_MARKERS.findall(text))
        weight = len(text)

        forced = (
            markers >= self.policy.api_marker_threshold
            or weight >= self.policy.api_length_threshold
        )

        with self._lock:
            if forced:
                route = Route.API
                self._forced_api += 1
            else:
                total = self._chars[Route.LOCAL] + self._chars[Route.API]
                share = self._chars[Route.LOCAL] / total if total else 0.0
                # A cap of 1.0 means "no cap", and must be spelled out: a plain
                # `share < cap` test rejects a fully-local history, because that
                # history has a share of exactly 1.0. Left implicit it silently
                # halves the local share instead of disabling the limit.
                # 上限为 1.0 表示「不限制」，必须显式处理：单纯的 `share < cap` 会
                # 拒绝全本地的历史，因为其份额恰好是 1.0。不写明就会变成把本地份额
                # 砍半，而不是关闭该限制。
                under_cap = (
                    self.policy.local_share_cap >= 1.0
                    or share < self.policy.local_share_cap
                )
                if self._local_inflight < self.policy.local_slots and under_cap:
                    route = Route.LOCAL
                    self._local_inflight += 1
                    self._peak_local_inflight = max(
                        self._peak_local_inflight, self._local_inflight
                    )
                else:
                    # Local is saturated; sending this to the API keeps it
                    # moving instead of queueing behind a busy GPU.
                    # 本地已满载，转交 API 让它继续前进，而不是在 GPU 后排队。
                    route = Route.API
                    self._overflow_api += 1

            self._chars[route] += weight
            self._paras[route] += 1

        return route

    @contextmanager
    def local_slot(self):
        """
        中文：本地调用的作用域，退出时释放槽位。

        Scope a local call, releasing its reserved slot on exit.

        Release happens even when the call raises, so a failing backend cannot
        permanently consume the local budget and starve the local side.

        Yields:
            None
        """
        try:
            yield
        finally:
            with self._lock:
                self._local_inflight -= 1

    def stats(self) -> dict[str, int | float]:
        """
        中文：返回实际分流统计，用于核对策略是否按预期生效。

        Report the realised split.

        Returns:
            dict[str, int | float]: Paragraph and character counts per route,
                the achieved local character share, how many paragraphs the
                quality rules forced to the API, how many overflowed there
                because local was saturated, and the peak local concurrency.
        """
        with self._lock:
            total = self._chars[Route.LOCAL] + self._chars[Route.API]
            return {
                "local_paragraphs": self._paras[Route.LOCAL],
                "api_paragraphs": self._paras[Route.API],
                "local_chars": self._chars[Route.LOCAL],
                "api_chars": self._chars[Route.API],
                "local_share_actual": (self._chars[Route.LOCAL] / total) if total else 0.0,
                "forced_to_api": self._forced_api,
                "overflow_to_api": self._overflow_api,
                "peak_local_inflight": self._peak_local_inflight,
            }
