"""
中文：混合翻译模式——本地模型与云端 API 并行分担，按在途负载分流。

Hybrid translation mode.

A self-contained alternative to the ``local`` and ``api`` backends. Nothing here
is imported by those two, and they are unchanged by its presence: selecting
hybrid swaps in a different translator, it does not alter how the others behave.

Two earlier attempts were both slower than either backend alone, and the reason
turned out to be the same one twice: a wrong assumption about what limits the
API's speed. Instrumenting the dispatch — recording when every request was
enqueued, started and finished — settled it. Measured on this machine:

    concurrency   local (Hy-MT2 7B)   API
    1             255 chars/s         116 chars/s
    8             443 chars/s          —
    16            515 chars/s         749 chars/s
    24            423 chars/s          —

The two resources scale in opposite ways. The local model is one GPU: it
saturates near sixteen concurrent requests and *regresses* beyond that as the
KV cache thrashes. The API is concurrency-elastic: throughput climbs with
requests in flight while latency stays flat, so the only way to use it well is
to keep many requests outstanding.

Both earlier designs optimised request *count* — collecting paragraphs into
batches to reduce round trips. Against an elastic endpoint that is backwards,
and the instrumentation showed exactly how badly: seven batched requests
averaging 10.45 seconds each, running at a measured concurrency of 1.00,
because the collector had a single background thread. The API side spent 73
seconds strictly serial while the local model idled.

So the current design inverts the priority. Concurrency comes first:

    BabelDOC ──per paragraph──▶ Router ──┬──▶ local, bounded by free slots
                                          └──▶ API, one request per paragraph

Routing is **work-conserving**: a paragraph goes local whenever the local model
has a free slot, and to the API otherwise, so neither side idles while work
remains. The realised split is not configured — it settles at whatever ratio the
two measured rates imply, which is the optimum and which moves on its own when
the hardware or the endpoint changes.
"""

from local_translator.engine.hybrid.router import Route, Router, RouterPolicy
from local_translator.engine.hybrid.translator import HybridTranslator

__all__ = ["HybridTranslator", "Route", "Router", "RouterPolicy"]
