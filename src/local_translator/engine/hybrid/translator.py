"""
中文：混合翻译器——把路由与 API 分发组装成 BabelDOC 可用的翻译器。

The hybrid translator itself.

Glues :class:`~local_translator.engine.hybrid.router.Router` to an API dispatch
strategy behind the interface BabelDOC expects. It owns two ordinary backends
and adds nothing to either — the ``local`` and ``api`` modes construct their own
instances and are unaffected by anything here.

Two API strategies exist because the choice is not obvious from first
principles and was settled by measurement:

``direct``
    Each API-routed paragraph becomes its own request, issued from the calling
    worker thread. Concurrency equals the number of worker threads on the API
    side, which is what a concurrency-elastic endpoint wants.

``batched``
    Paragraphs are collected into character-bounded batches by a pool of
    dispatch threads. Fewer requests and fewer repeated prompts, but with
    blocking callers it has a structural weakness: the number of paragraphs in
    flight is capped by the number of blocked worker threads, so grouping them
    into batches *reduces* the number of concurrent requests rather than
    increasing throughput.

``direct`` is the default because the measurements favour it; ``batched``
remains selectable for cost-sensitive runs, where fewer prompt repetitions
matter more than wall time.
"""

from __future__ import annotations

import logging

from babeldoc.translator.translator import BaseTranslator

from local_translator.engine.translators import TranslationGuards

from local_translator.engine.hybrid.router import Route, Router, RouterPolicy
from local_translator.engine.translators import language_name

logger = logging.getLogger(__name__)


class HybridTranslator(TranslationGuards, BaseTranslator):
    """
    中文：按在途负载把段落分给本地模型与云端 API。

    Translate a document across two backends at once.

    Attributes:
        name (str): Cache key prefix. Distinct from either backend's, because a
            paragraph's translation depends on which model produced it and a
            shared key would serve one model's output for the other's request.
        local (BaseTranslator): Local model backend.
        api (BaseTranslator): Cloud API backend.
        router (Router): Per-paragraph routing policy.
    """

    name = "hybrid"

    def __init__(
        self,
        local: BaseTranslator,
        api: BaseTranslator,
        lang_in: str,
        lang_out: str,
        policy: RouterPolicy | None = None,
        api_strategy: str = "direct",
        batch_threads: int = 12,
        max_batch_chars: int = 1200,
        max_batch: int = 10,
        ignore_cache: bool = False,
    ):
        """
        中文：构造混合翻译器。

        Args:
            local (BaseTranslator): Local model translator.
            api (BaseTranslator): Cloud API translator.
            lang_in (str): Source language code.
            lang_out (str): Target language code.
            policy (RouterPolicy | None): Routing thresholds.
            api_strategy (str): ``direct`` or ``batched``.
            batch_threads (int): Dispatch threads, ``batched`` only.
            max_batch_chars (int): Character bound per request, ``batched`` only.
            max_batch (int): Paragraph bound per request, ``batched`` only.
            ignore_cache (bool): Bypass BabelDOC's paragraph cache.
        """
        super().__init__(lang_in, lang_out, ignore_cache)
        self.local = local
        self.api = api
        self.router = Router(policy)
        self.target_language = language_name(lang_out)
        self.api_strategy = api_strategy
        self._dispatcher = None

        if api_strategy == "batched":
            from local_translator.engine.hybrid.dispatcher import ApiDispatcher

            self._dispatcher = ApiDispatcher(
                translate_batch=self._translate_batch,
                translate_single=self._translate_single,
                threads=batch_threads,
                max_chars=max_batch_chars,
                max_batch=max_batch,
            )

        self.add_cache_impact_parameters("engine", "hybrid")

    def do_llm_translate(self, text, rate_limit_params: dict | None = None):
        """
        中文：不实现，以取得逐段接口；分流必须逐段进行。

        Intentionally unimplemented.

        Answering yes here would hand BabelDOC's batched pipeline every
        paragraph, leaving nothing to route.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError("Hybrid routes per paragraph; it needs that interface")

    def _translate_single(self, text: str) -> str:
        """
        中文：单条走 API，直接调用底层以避开重复限流。

        Translate one paragraph through the API backend.

        Calls ``do_translate`` rather than ``translate`` deliberately. BabelDOC's
        ``translate`` passes through a process-wide leaky-bucket limiter, and
        this translator's own ``translate`` has already paid that toll for this
        paragraph. Going through it twice halves the achievable rate for no
        benefit — measured as 120s of a 134s translation stage, against 15s once
        the double toll was removed. Caching is likewise already handled at this
        translator's own level, under its own cache key.

        Args:
            text (str): Paragraph text.

        Returns:
            str: The translation.
        """
        return self.api.do_translate(text)

    def _translate_batch(self, texts: list[str]) -> list[str] | None:
        """
        中文：把一批段落打包成一次 API 请求。

        Translate several paragraphs in one API request.

        Args:
            texts (list[str]): Paragraphs to translate.

        Returns:
            list[str] | None: One translation per input, or ``None`` when the
                reply could not be split reliably.
        """
        from local_translator.engine.hybrid.dispatcher import (
            build_batch_prompt,
            split_batch_response,
        )
        from local_translator.engine.translators import strip_reasoning

        prompt = build_batch_prompt(texts, self.target_language)
        raw = self.api.do_llm_translate(prompt, rate_limit_params={})
        return split_batch_response(strip_reasoning(raw or ""), len(texts))

    def _to_api(self, text: str) -> str:
        """
        中文：按所选策略把段落送往 API。

        Send a paragraph to the API through the configured strategy.

        Args:
            text (str): Paragraph text.

        Returns:
            str: The translation.
        """
        if self._dispatcher is not None:
            return self._dispatcher.submit(text)
        return self.api.do_translate(text)

    def do_translate(self, text: str, rate_limit_params: dict | None = None) -> str:
        """
        中文：按路由把段落交给本地或 API；任一侧失败则切到另一侧。

        Translate one paragraph through its routed backend.

        A ``LOCAL`` decision carries a reserved slot, so the local call is
        wrapped in the router's slot scope and the reservation is released even
        if the backend raises. A backend that fails falls through to the other:
        running two backends is only an availability gain if one can cover for
        the other, and a dropped connection should not cost a paragraph.

        Args:
            text (str): Paragraph text.
            rate_limit_params (dict | None): Passed through from BabelDOC.

        Returns:
            str: The translated paragraph.
        """
        route = self.router.decide(text)

        if route is Route.LOCAL:
            try:
                with self.router.local_slot():
                    return self.local.do_translate(text, rate_limit_params)
            except Exception:
                logger.warning("Local backend failed; retrying on the API", exc_info=True)
                return self._to_api(text)

        try:
            return self._to_api(text)
        except Exception:
            logger.warning("API backend failed; retrying locally", exc_info=True)
            return self.local.do_translate(text, rate_limit_params)

    # Marker formats come from the local backend, since BabelDOC asks the
    # translator how to spell them and both sides must agree.
    # 占位符格式取自本地后端；BabelDOC 向翻译器索取写法，两侧必须一致。

    def get_formular_placeholder(self, placeholder_id: int | str):
        """中文：公式占位符格式。 / Formula marker format."""
        return self.local.get_formular_placeholder(placeholder_id)

    def get_rich_text_left_placeholder(self, placeholder_id: int | str):
        """中文：富文本左标记。 / Opening rich-text marker."""
        return self.local.get_rich_text_left_placeholder(placeholder_id)

    def get_rich_text_right_placeholder(self, placeholder_id: int | str):
        """中文：富文本右标记。 / Closing rich-text marker."""
        return self.local.get_rich_text_right_placeholder(placeholder_id)

    def usage_summary(self) -> dict[str, int | float]:
        """
        中文：汇总两侧用量、分流比例与分发统计。

        Combine both backends' usage with routing and dispatch statistics.

        Returns:
            dict[str, int | float]: Summed token and paragraph counters, plus
                the realised split.
        """
        merged: dict[str, int | float] = {}
        for backend in (self.local, self.api):
            if not hasattr(backend, "usage_summary"):
                continue
            for key, value in backend.usage_summary().items():
                merged[key] = merged.get(key, 0) + value

        merged.update(self.router.stats())
        if self._dispatcher is not None:
            merged.update(self._dispatcher.stats())
        return merged

    def close(self) -> None:
        """中文：停止分发线程。 / Stop the dispatch threads."""
        if self._dispatcher is not None:
            self._dispatcher.stop()
