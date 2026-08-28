"""
中文：按版面标签选择性翻译——跳过脚注、参考文献、作者信息、表格等。

Selective translation by document layout region.

PDFMathTranslate leaves footnotes, author blocks and table contents in their
original language, which keeps a translated paper much closer to the original
and avoids mangling citation strings and author names. BabelDOC translates
everything by default, but its layout model already tags every paragraph with a
``layout_label`` (``footnote``, ``reference``, ``author_info_hybrid``,
``table_text``, …), so the same behaviour can be obtained by filtering on that
tag just before translation.

Implementation approach:
    :class:`LayoutFilteringILTranslator` subclasses BabelDOC's classic
    ``ILTranslator`` and short-circuits :meth:`translate_paragraph` for
    paragraphs whose label is in the skip set. Because
    ``babeldoc.format.pdf.high_level`` instantiates ``ILTranslator`` by name,
    :func:`install` rebinds that module attribute to the subclass. No BabelDOC
    source file is modified, and :func:`install` asserts the symbol exists so a
    future BabelDOC release that renames it fails loudly at startup instead of
    silently reverting to translating everything.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from collections import Counter

from local_translator.engine import frontmatter, translators

logger = logging.getLogger(__name__)

_FM_PROBE = bool(os.getenv("LOCALTRANS_FRONTMATTER_PROBE"))

# Bibliography entries are frequently mislabelled by the layout model — measured
# on one paper, only 1 of 4 reference paragraphs carried a `reference` label, so
# the other three were translated into Chinese. Translating a citation is worse
# than useless: it mangles author names and journal titles a reader needs
# verbatim in order to look the work up.
# 版面模型经常认错参考文献——实测一篇论文 4 段里只有 1 段带 reference 标签，
# 另外 3 段被译成了中文。译过的引文有害无益：作者名与刊名正是要照原样查证的。
_BIBLIOGRAPHY_SIGNALS = (
    re.compile(r"^\s*[\[\(]\d{1,3}[\]\)]"),          # [12] or (12) leading a line
    re.compile(r"\barXiv\s*:\s*\d{4}\.\d{4,5}", re.I),
    re.compile(r"\bdoi\s*:\s*10\.\d{4,}", re.I),
    re.compile(r"\bIn\s+Proceedings\b", re.I),
    re.compile(r"\bpp\.\s*\d+\s*[–-]\s*\d+"),
    re.compile(r"\b\d{1,3}\s*[:(]\s*\d+\s*[–-]\s*\d+\s*,?\s*(19|20)\d{2}"),
)

# Two independent signals are required. A single one fires on ordinary prose —
# "(3)" numbers an equation, and a page range can appear in a figure caption.
# 需要两个独立信号：单个信号会误伤正文——"(3)" 可能是公式编号，页码范围也可能出现在图注里。
_MIN_BIBLIOGRAPHY_SIGNALS = 2


def looks_like_bibliography(text: str) -> bool:
    """
    中文：判断一段文字是否是参考文献条目。

    Detect a bibliography entry from its text.

    Used alongside the layout label rather than instead of it, because the two
    fail in different ways: the label misses entries the model mislabelled,
    while the text heuristic misses entries written in an unusual citation style.

    Args:
        text (str): Paragraph text.

    Returns:
        bool: ``True`` when at least two citation signals are present.
    """
    if not text or len(text) < 40:
        return False
    hits = sum(1 for pattern in _BIBLIOGRAPHY_SIGNALS if pattern.search(text))
    return hits >= _MIN_BIBLIOGRAPHY_SIGNALS


class LayoutFilteringILTranslator:
    """
    中文：占位符类型标注；真实实现由 build_filtering_translator 动态生成。

    Placeholder for typing only; the real subclass is produced at import time by
    :func:`_build_class` so that BabelDOC is imported lazily.
    """


# Longest styled run worth marking up. The first version used 48, on the theory
# that anything longer was a sentence in another face rather than a heading.
# That was wrong about how papers are written: a bulleted findings list
# routinely opens each item with a **bold lead-in sentence** of a hundred
# characters or more, and every one of them came out unbolded because of this
# number. Two hundred still excludes a paragraph typeset wholesale in another
# face, which is the case worth excluding.
# 值得加标记的样式段最大长度。第一版取 48，理由是更长的就是「另一种字体排的句子」
# 而非标题。这个判断误解了论文的写法：结论列表里每个条目开头常常是一整句上百字符的
# **加粗导语**，而它们全都因为这个数字而丢了加粗。取 200 仍然能排除整段用另一种
# 字体排版的情况——那才是真正该排除的。
_MAX_RUN = 200

# How many styled runs a lead-in may be broken into. One is the plain case; a
# lead-in interrupted by an inline formula — 「At a fixed tokens-per-parameter
# ratio TPP, the optimal…」, where TPP is set in the maths font — arrives as two.
# Beyond three the paragraph is not a lead-in at all but styled throughout, and
# a formula-dense one of those produced seventeen tags that the model echoed
# back verbatim.
# 一段导语最多可以被打断成几段。一段是常见情形；被行内公式打断的导语——
# 「At a fixed tokens-per-parameter ratio TPP, the optimal…」，其中 TPP 用数学字体
# 排——会变成两段。超过三段就不再是导语，而是通篇异体；公式密集的那种曾产生十七个
# 标记，被模型原样复读了回来。
_MAX_RUNS = 4

# How many markers of any kind — style tags and formula placeholders together —
# a request may carry and still be asked to reproduce them.
#
# This replaces an earlier pair of rules that required the styling to *begin* at
# the paragraph's head and to end within its first three quarters. Those rules
# came from one real failure: a formula-dense paragraph carrying seventeen style
# tags interleaved with thirteen placeholders was echoed back verbatim by the
# model, leaving the paragraph in English. But "starts at the head" is a
# description of a run-in heading, not of what a model can copy, and it threw
# away every emphasis a paper puts mid-paragraph — measured against the same
# document translated by BabelDOC's own defaults, 2.98% of the Chinese came back
# bold where 5.47% should have.
#
# Counting markers addresses the actual failure. The paragraph that broke
# carried forty-seven. Twelve was measured against twenty-four on a 51-page
# paper: 4.59% of the Chinese came back bold against 5.05%, for a request the
# model is much more certain to reproduce. Twelve is the setting, because half
# a percentage point of emphasis is not worth handing a translation model a
# form it has to work at.
# 一次请求最多可以带多少个标记——样式标记与公式占位符合并计算——仍要求模型原样复现。
#
# 这一条取代了先前两条规则：样式必须从段首开始、且须在前四分之三内结束。那两条源于
# 一次真实故障：一个公式密集的段落带了十七个样式标记、夹杂十三个占位符，模型把请求
# 原样复读了回来，整段留在英文。但「从段首开始」描述的是行内标题的形态，不是模型能
# 复制什么，它把论文写在段落中间的强调全丢了——同一篇文档与 BabelDOC 默认设置对照，
# 译文里 5.47% 该加粗的地方，我们只做到 2.98%。
#
# 数标记个数才对准了真正的故障：出事那一段带了四十七个标记。12 与 24 在一篇 51 页
# 论文上对比过：加粗占比 4.59% 对 5.05%，而 12 给模型的是一个它更有把握复现的形态。
# 取 12——半个百分点的强调，不值得让翻译模型去做一件需要费力的事。
_MAX_MARKERS = 12


def _styling_is_reproducible(translate_input) -> bool:
    """
    中文：判断这次请求的富文本标记是否简单到模型能原样复现——能就下发，不能就放弃样式。

    Decide whether a request's markers are simple enough to ask the model to
    reproduce.

    Three conditions, each of which a real failure taught. There may be no more
    than :data:`_MAX_RUNS` styled runs and no more than :data:`_MAX_MARKERS`
    markers in total, because a request carrying thirty of them was echoed back
    verbatim and left its paragraph in English. And no single run may exceed
    :data:`_MAX_RUN`, which excludes a paragraph typeset wholesale in another
    face — styling that says nothing, since every word of it is emphasised.

    Where in the paragraph the styling sits is deliberately not a condition. An
    earlier version required it to begin at the head, which describes a run-in
    heading rather than anything about what a model can copy, and it discarded
    every emphasis a paper places mid-sentence.

    Args:
        translate_input (TranslateInput | None): What BabelDOC would send the
            translator with rich text enabled.

    Returns:
        bool: ``True`` when the styling is worth asking the model to preserve.
    """
    if not translate_input:
        return False
    text = translate_input.unicode or ""
    placeholders = translate_input.placeholders or []
    runs = [p for p in placeholders if getattr(p, "right_placeholder", None)]
    if not runs or len(runs) > _MAX_RUNS or not text:
        return False

    # Both halves of every styled run, plus every formula placeholder, are
    # tokens the model has to carry across unchanged.
    # 每个样式段的首尾两半，加上每一个公式占位符，都是模型必须原样搬运的记号。
    if len(runs) * 2 + (len(placeholders) - len(runs)) > _MAX_MARKERS:
        return False

    cursor = 0
    for run in runs:
        opening, closing = str(run.left_placeholder), str(run.right_placeholder)
        start = text.find(opening, cursor)
        if start < 0:
            return False
        end = text.find(closing, start + len(opening))
        if end < 0:
            return False
        if (end - (start + len(opening))) > _MAX_RUN:
            return False
        cursor = end + len(closing)
    return True


def _build_llm_only_class():
    """
    中文：构造 LLM-only 管线的过滤子类。

    Construct the filtering subclass for BabelDOC's LLM-only pipeline.

    **Why a second class.** BabelDOC has two translators and picks between them
    by asking whether the engine implements ``do_llm_translate``: an
    OpenAI-compatible cloud model does, so cloud runs go through
    ``ILTranslatorLLMOnly`` — a separate class, not a subclass of
    ``ILTranslator``. Patching only the latter left selective translation
    installed, logged as active, and completely inert on the cloud path. Author
    lists came back as 「程浩·李」, citation anchors were rewritten, and the run
    reported success.

    **Where it hooks.** ``_should_translate_paragraph`` is the single gate every
    paragraph passes through in that pipeline — ``_filter_paragraphs`` is its
    only caller and every stage selects work through that. Returning ``False``
    leaves the paragraph holding its original text, which is what BabelDOC then
    typesets.

    Returns:
        type: A subclass of ``ILTranslatorLLMOnly``.
    """
    from babeldoc.format.pdf.document_il.midend.il_translator_llm_only import (
        ILTranslatorLLMOnly,
    )

    class _LayoutFilteringLLMOnly(ILTranslatorLLMOnly):
        """
        中文：在 LLM-only 管线上应用同一套跳过规则。

        Apply the same skip rules on the LLM-only pipeline.
        """

        def __init__(self, *args, **kwargs):
            """中文：记录本类被真正用上。 / Record that BabelDOC built this."""
            super().__init__(*args, **kwargs)
            _BUILT.add("ILTranslatorLLMOnly")

        def _should_translate_paragraph(self, paragraph, translated_ids=None,
                                        require_body_text: bool = False) -> bool:
            """
            中文：先按原有条件筛，再问共用的跳过判据。

            Filter as the base class does, then consult the shared decision.

            Args:
                paragraph (PdfParagraph): Paragraph to consider.
                translated_ids (set[int] | None): Already-translated ids.
                require_body_text (bool): Whether to require body text.

            Returns:
                bool: ``True`` to translate.
            """
            if not super()._should_translate_paragraph(
                paragraph, translated_ids, require_body_text
            ):
                return False
            strict = needs_strict_terms(paragraph)
            translators.set_strict_terms(strict)
            # This pipeline selects paragraphs a page at a time without handing
            # the page to the gate, so the citation-anchor rule — the only one
            # that needs it — sits out here rather than being fed a guess.
            # 这条管线按页选段但不把页对象交给闸门，因此唯一需要它的引文锚点判据在此
            # 让位，而不是喂给它一个猜测值。
            reason = skip_decision(self, paragraph, None, strict)
            if reason:
                with self._skip_lock:
                    self.skipped[reason] += 1
                return False
            return True

        def _clean_json_output(self, llm_output: str) -> str:
            """
            中文：清洗模型返回的 JSON，并拆掉「把原文当译文」的那条退路。

            Clean the model's JSON, and disarm the echo-as-translation fallback.

            BabelDOC reads each record as
            ``item.get("output", item.get("input"))``: when a model answers one
            entry with ``"input"`` instead of ``"output"`` — echoing the source
            — that echo is taken *as the translation*. The batch is the right
            length so the length check passes, and the 「same as input」 guard
            compares byte for byte, which an echo that re-wrapped its lines does
            not trip. The paragraph is then typeset in the output font, in the
            source language, with nothing logged.

            Dropping the stray ``"input"`` leaves the record with no usable
            value, so BabelDOC declines it and leaves the paragraph alone —
            which :meth:`_retry_untranslated` then notices and re-sends on its
            own. The two halves are needed together: this one stops the echo
            from passing as a translation, and that one recovers the paragraph.

            Args:
                llm_output (str): Raw text from the model.

            Returns:
                str: JSON for BabelDOC to parse, with echo records defused.
            """
            cleaned = super()._clean_json_output(llm_output)
            try:
                parsed = json.loads(cleaned)
            except (ValueError, TypeError):
                # Not parseable here is not ours to fix; BabelDOC reports it.
                # 这里解析不了不归我们管，由 BabelDOC 去报告。
                return cleaned

            records = parsed if isinstance(parsed, list) else [parsed]
            echoes = 0
            for item in records:
                if not isinstance(item, dict):
                    continue
                if "output" not in item and "input" in item:
                    item.pop("input")
                    echoes += 1
            if not echoes:
                return cleaned
            logger.info("模型有 %d 条回吐了原文而非译文，已拒收并将重试", echoes)
            return json.dumps(parsed if isinstance(parsed, list) else records[0],
                              ensure_ascii=False)

        def translate_paragraph(self, batch_paragraph, pbar=None, *args, **kwargs):
            """
            中文：送去翻译之前把该跳过的段落从批次里剔掉。

            Drop the paragraphs that should be left alone before the batch is sent.

            **Why here and not only in the gate.** ``_should_translate_paragraph``
            looked like the single filter in this pipeline, but ``process_page``
            — the path that handles the ordinary, non-cross-page case, which is
            almost every paragraph — repeats those checks inline instead of
            calling it. Overriding the gate alone therefore filtered the two
            rare paths and left the common one untouched: the author list still
            came back translated, and re-running produced a *different*
            translation of the same names, which is what showed the paragraphs
            were reaching the model rather than a cache.

            ``translate_paragraph`` is the one place all four call sites meet.

            Args:
                batch_paragraph (BatchParagraph): Batch to translate. Its
                    ``paragraphs``, ``pages`` and ``trackers`` are parallel
                    lists indexed together, so all three are trimmed in step.
                pbar (tqdm | None): Stage progress bar, advanced once per
                    dropped paragraph so the stage still reaches its total.
                *args: Passed through unchanged.
                **kwargs: Passed through unchanged.

            Returns:
                None
            """
            paragraphs, pages, trackers = [], [], []
            dropped = 0
            for index, paragraph in enumerate(batch_paragraph.paragraphs):
                page = (batch_paragraph.pages[index]
                        if index < len(batch_paragraph.pages) else None)
                strict = needs_strict_terms(paragraph)
                translators.set_strict_terms(strict)
                reason = skip_decision(self, paragraph, page, strict)
                if _FM_PROBE:
                    logger.info("[llm-only] kind=%r label=%r 判定=%r: %r",
                                frontmatter.kind_of(paragraph),
                                (paragraph.layout_label or "").strip(), reason,
                                (paragraph.unicode or "")[:40])
                if reason:
                    with self._skip_lock:
                        self.skipped[reason] += 1
                    dropped += 1
                    continue
                paragraphs.append(paragraph)
                pages.append(page)
                trackers.append(batch_paragraph.trackers[index])

            if dropped:
                batch_paragraph.paragraphs = paragraphs
                batch_paragraph.pages = pages
                batch_paragraph.trackers = trackers
                if pbar is not None:
                    # The stage total counts every paragraph, dropped ones
                    # included; without this the bar stops short of 100%.
                    # 阶段总数把被剔除的段落也算在内，不补这一步进度条就到不了 100%。
                    pbar.advance(dropped)

            if not paragraphs:
                return None

            before = [p.unicode or "" for p in paragraphs]
            result = super().translate_paragraph(batch_paragraph, pbar, *args, **kwargs)
            self._retry_untranslated(batch_paragraph, before, *args, **kwargs)
            return result

        def _retry_untranslated(self, batch_paragraph, before, *args, **kwargs):
            """
            中文：把整段回吐原文的段落挑出来，逐段重试一次。

            Retry, one at a time, the paragraphs that came back untranslated.

            **The defect this exists for.** The batch prompt asks the model for
            ``[{"id": n, "output": "..."}]``. Models sometimes answer one entry
            with ``"input"`` instead of ``"output"``, echoing the source. BabelDOC
            reads that as ``item.get("output", item.get("input"))`` — it takes the
            echoed source *as the translation*. The batch is the right length, so
            its length check passes; and its 「same as input」 guard compares byte
            for byte, which an echo that re-wrapped its lines does not trip. The
            paragraph is then typeset in the output font, in the source language,
            with nothing logged. Measured on one IEEE-format paper: eight body
            paragraphs across four pages.

            **Why retry rather than report.** A single paragraph asked on its own
            is a different, much simpler request than a batch of thirty, and it
            succeeds where the batch entry failed. Only the failures are re-sent,
            so the cost is proportional to the defect rather than to the document.

            Args:
                batch_paragraph (BatchParagraph): The batch just translated; its
                    paragraphs now hold whatever came back.
                before (list[str]): Each paragraph's text before translation.
                *args: Forwarded to the retry call.
                **kwargs: Forwarded to the retry call.
            """
            # `before` was taken from the trimmed list, so the two are the same
            # length — but a super() call that rebuilt the batch would break
            # that silently, and an index error here would lose the whole page.
            # before 取自剔除后的列表，二者等长——但若 super() 重建了批次，这个前提会
            # 悄悄失效，而这里的越界会赔上整页。
            if len(before) != len(batch_paragraph.paragraphs):
                logger.debug("Batch changed length during translation; skipping retry")
                return
            stale = [
                index for index, paragraph in enumerate(batch_paragraph.paragraphs)
                if _looks_untranslated(before[index], paragraph.unicode or "")
            ]
            if not stale:
                return

            logger.info("%d paragraph(s) came back untranslated; retrying singly",
                        len(stale))
            for index in stale:
                single = _one_paragraph_batch(batch_paragraph, index)
                if single is None:
                    continue
                was = batch_paragraph.paragraphs[index].unicode or ""
                try:
                    # No pbar: the batch already advanced it for this paragraph.
                    # 不传进度条：这一段的进度在上一趟已经推进过了。
                    super().translate_paragraph(single, None, *args, **kwargs)
                except Exception:  # noqa: BLE001
                    logger.debug("Retry failed for a paragraph", exc_info=True)
                    continue
                now = batch_paragraph.paragraphs[index].unicode or ""
                if _looks_untranslated(before[index], now):
                    logger.warning("重试后仍未译出：%r", (was or "")[:70])
                else:
                    logger.info("重试成功：%r → %r", (was or "")[:40], now[:40])

    _LayoutFilteringLLMOnly._skip_lock = threading.Lock()
    return _LayoutFilteringLLMOnly


# Which filtering classes BabelDOC actually instantiated this run. Asserting
# that the module attribute was replaced proves only that the assignment
# happened; it cannot prove BabelDOC read it. This does — a class that was never
# constructed never filtered anything, and that is the failure this exists to
# catch. Cleared by :func:`install`.
# 本次运行中 BabelDOC 真正实例化了哪些过滤类。断言模块属性被替换只能证明赋值发生过，
# 证明不了 BabelDOC 读了它。这个可以——从未被构造的类就从未过滤过任何东西，而这正是它要
# 捕捉的故障。由 install() 清空。
_BUILT: set[str] = set()


def was_active() -> bool:
    """
    中文：本次运行是否真的有过滤类被 BabelDOC 用上。

    Report whether BabelDOC instantiated one of the filtering classes.

    Returns:
        bool: ``True`` when selective translation actually took part in this
            run, as opposed to merely having been installed.
    """
    return bool(_BUILT)


# A paragraph that still reads as its source after translation. Counted in
# words rather than characters because a short label legitimately survives
# unchanged — 「CSI」 is 「CSI」 in both languages — while a sentence does not.
# 翻译之后仍然是原文的段落。以词计而非以字符计，因为短标签原样保留是合理的——「CSI」
# 在两种语言里都是「CSI」——而整句不是。
_UNTRANSLATED_WORDS = 6

_LATIN_WORD = re.compile(r"[A-Za-z]{3,}")
_CJK = re.compile(r"[\u4e00-\u9fff]")


def _looks_untranslated(before: str, after: str) -> bool:
    """
    中文：判断一段译文是不是原样回吐的原文。

    Report whether a paragraph came back in its source language.

    Args:
        before (str): The paragraph's text before translation.
        after (str): Its text afterwards.

    Returns:
        bool: ``True`` when the source was a run of prose and the result carries
            no target-language characters at all.
    """
    if not before or not after:
        return False
    if _CJK.search(after):
        return False
    return len(_LATIN_WORD.findall(before)) >= _UNTRANSLATED_WORDS


def _one_paragraph_batch(batch, index: int):
    """
    中文：从批次里取出一段，包成只含它自己的批次。

    Build a one-paragraph batch sharing the original's tracker and page.

    The batch class is BabelDOC's and takes a page tracker it calls
    ``new_paragraph()`` on, so it cannot be constructed from the pieces
    directly; a shallow copy with trimmed lists is both simpler and keeps the
    tracker that already belongs to this paragraph.

    Args:
        batch (BatchParagraph): Batch to take from.
        index (int): Which paragraph.

    Returns:
        BatchParagraph | None: The single-paragraph batch, or ``None`` if the
            batch does not have the parallel lists this depends on.
    """
    import copy

    try:
        single = copy.copy(batch)
        single.paragraphs = [batch.paragraphs[index]]
        single.pages = [batch.pages[index]]
        single.trackers = [batch.trackers[index]]
    except (AttributeError, IndexError):
        return None
    return single


def anchored_citation(anchors, paragraph, page) -> bool:
    """
    中文：用文档自带的引用锚点判断该段是否为文献条目。

    Decide from the document's own anchors whether this is a reference.

    The text heuristic misses entries that happen to carry only one
    citation-like pattern — measured on a real paper, entry [28] leaked through
    and had its author names translated because its bracketed number was split
    into a separate element, leaving only the arXiv id to match on. An anchor is
    not a guess: hyperref placed it there because a bibliography entry begins at
    that coordinate.

    Module-level rather than a method, because BabelDOC's two translation
    pipelines are unrelated classes and both need it. Leaving it on one of them
    is what made the shared decision raise AttributeError on the other.

    Args:
        anchors: Anchor index for the document, or ``None``.
        paragraph (PdfParagraph): Paragraph under consideration.
        page (Page): Owning page, providing the page number. ``None`` when the
            caller has no page to offer, in which case the test declines.

    Returns:
        bool: ``True`` when a citation anchor falls in the paragraph.
    """
    if anchors is None or not anchors.available or page is None:
        return False
    box = getattr(paragraph, "box", None)
    if box is None:
        return False
    try:
        return anchors.is_citation(page.page_number, box.y, box.y2)
    except Exception:  # noqa: BLE001
        return False


def skip_decision(owner, paragraph, page, strict: bool) -> str | None:
    """
    中文：判断一个段落是否应保留原文，返回跳过的理由或 None。

    Decide whether a paragraph should be left untranslated.

    Lifted out of the translator subclass because BabelDOC has two translation
    pipelines and both must apply the same rules. Duplicating the chain in each
    was what let the cloud path silently translate author lists that the local
    path preserved; one copy cannot drift from itself.

    Args:
        owner: The translator instance, carrying ``skip_labels``,
            ``skip_content``, ``skip_bibliography`` and ``anchors``.
        paragraph (PdfParagraph): Paragraph under consideration.
        page (Page): Owning page, used by the citation-anchor rule. May be
            ``None`` where the pipeline does not supply one, in which case that
            rule is skipped rather than guessed at.
        strict (bool): Whether strict term handling applies to this paragraph.

    Returns:
        str | None: The reason to skip, used both as the counter key and as the
            explanation shown in the run summary, or ``None`` to translate.
    """
    label = (paragraph.layout_label or "").strip()
    text = paragraph.unicode or ""

    if strict and is_mostly_numeric(text):
        return "numeric label"
    if has_nothing_to_translate(text):
        return "no words"
    if label and label in owner.skip_labels:
        return label
    kind = frontmatter.kind_of(paragraph)
    if kind and kind in owner.skip_labels:
        return kind
    if not owner.skip_content:
        return None
    if anchored_citation(owner.anchors, paragraph, page):
        return "reference (anchor)"
    if owner.skip_bibliography and looks_like_bibliography(text):
        return "reference (detected by text)"
    if is_display_formula(text):
        return "display formula"
    if owner.skip_pseudocode and is_pseudocode(text):
        return "pseudocode"
    return None


def _build_class():
    """
    中文：动态构造 ILTranslator 的过滤子类（延迟导入 BabelDOC）。

    Construct the filtering subclass, importing BabelDOC lazily.

    Deferring the import keeps ``local_translator.cli`` usable (for ``setup``
    and ``doctor``) even before the heavy BabelDOC dependency chain is loaded.

    Returns:
        type: A subclass of ``babeldoc…il_translator.ILTranslator``.
    """
    from babeldoc.format.pdf.document_il.midend.il_translator import (
        ILTranslator,
        PbarContext,
    )

    class _LayoutFilteringILTranslator(ILTranslator):
        """
        中文：跳过指定版面标签段落的 ILTranslator 子类。

        ``ILTranslator`` variant that leaves configured layout regions untranslated.

        Attributes:
            skip_labels (set[str]): Layout labels to skip. Set on the class by
                :func:`install` before BabelDOC constructs the instance, because
                the construction happens inside BabelDOC and takes no extra
                arguments.
            skip_bibliography (bool): Also leave paragraphs that read as citations
                alone, whatever label the layout model gave them.
            skip_pseudocode (bool): Also leave paragraphs that read as pseudocode
                alone. Set from the ``algorithm`` label, so the switch a reader
                sees governs this path and the LaTeX one alike.
            skip_content (bool): Master switch for the skips decided by reading a
                paragraph rather than by its layout label — bibliography
                entries, display equations and pseudocode. Turning selective
                translation off in the settings clears this, so "translate
                everything" really does mean everything.
            skipped (Counter): Per-label count of skipped paragraphs, useful for
                reporting "left 214 reference lines untranslated" in the UI.
        """

        skip_labels: set[str] = set()
        skip_bibliography: bool = True
        skip_pseudocode: bool = True
        skip_content: bool = True
        anchors = None
        skipped: Counter = Counter()
        _skip_lock = threading.Lock()

        def __init__(self, *args, **kwargs):
            """中文：记录本类被真正用上。 / Record that BabelDOC built this."""
            super().__init__(*args, **kwargs)
            _BUILT.add("ILTranslator")

        def _anchored_citation(self, paragraph, page) -> bool:
            """
            中文：见模块级的 anchored_citation。

            Delegate to :func:`anchored_citation`.

            Args:
                paragraph (PdfParagraph): Paragraph under consideration.
                page (Page): Owning page.

            Returns:
                bool: ``True`` when a citation anchor falls in the paragraph.
            """
            return anchored_citation(self.anchors, paragraph, page)

        def pre_translate_paragraph(self, paragraph, tracker, page_font_map,
                                    xobj_font_map):
            """
            中文：只为「段首一小段异体字」保留富文本标记，其余一律走纯文本。

            Keep rich-text markers for a leading run-in heading, and only that.

            BabelDOC turns rich text off whenever the engine does not implement
            ``llm_translate``:

                disable_rich_text_translate = config.disable_rich_text_translate
                if not self.support_llm_translate:
                    disable_rich_text_translate = True

            A paragraph then goes out as one flat string and is re-rendered in
            its base style, so a run-in bold heading loses its weight — 「致谢」
            opened its paragraph as ordinary text.

            Turning rich text on for everything was tried and is worse. BabelDOC
            marks up *every* run whose style differs from the paragraph's base,
            and on a formula-heavy page that is a great many: one paragraph of
            the test paper came back with **17** style tags interleaved with 13
            formula placeholders, and the model — a 7B translation model, not an
            instruction model — echoed the request back unchanged. Another
            paragraph came back with the tags shredded into
            ``</style><style <style id='5'>id='3'>``, which rendered as literal
            angle brackets in the middle of a section heading.

            So the markers are asked for only where they are nearly free: a
            single run, at the head of the paragraph, short enough to be a
            heading rather than a sentence. That is one pair of tags at the very
            front of the string, which is the shape the run-in heading has and
            the shape the model handles. Everything else is sent flat, exactly
            as before, and simply renders in the base style.

            Args:
                paragraph (PdfParagraph): Paragraph being prepared.
                tracker (ParagraphTranslateTracker): BabelDOC's per-paragraph
                    record.
                page_font_map (dict): Fonts of the page.
                xobj_font_map (dict): Fonts of each XObject on the page.

            Returns:
                tuple: ``(text, translate_input)`` as BabelDOC's own method
                    returns, or ``(None, None)`` when the paragraph is skipped.
            """
            if paragraph.vertical:
                return None, None
            tracker.set_pdf_unicode(paragraph.unicode)
            if paragraph.xobj_id in xobj_font_map:
                page_font_map = xobj_font_map[paragraph.xobj_id]

            translate_input = self.get_translate_input(paragraph, page_font_map,
                                                       False)
            if not _styling_is_reproducible(translate_input):
                translate_input = self.get_translate_input(paragraph,
                                                           page_font_map, True)
            if not translate_input:
                return None, None

            tracker.set_input(translate_input.unicode)
            tracker.set_placeholders(translate_input.placeholders)
            tracker.set_original_placeholders(
                getattr(translate_input, "original_placeholder_tokens", None),
            )
            text = translate_input.unicode
            if len(text) < self.translation_config.min_text_length:
                return None, None
            return text, translate_input

        def translate_paragraph(
            self,
            paragraph,
            page,
            pbar=None,
            tracker=None,
            page_font_map=None,
            xobj_font_map=None,
            paragraph_token_count: int = 0,
            title_paragraph=None,
            local_title_paragraph=None,
        ):
            """
            中文：若段落版面标签在跳过集合中则保留原文，否则走原有翻译流程。

            Translate a paragraph unless its layout region is excluded.

            Skipped paragraphs are simply left alone: BabelDOC renders whatever
            text the paragraph currently holds, so not translating it means the
            original text is typeset unchanged.

            The early return still happens inside ``PbarContext``, which advances
            the progress bar on exit. Skipping that would leave the stage total
            unreachable and the progress bar stuck below 100%.

            Args:
                paragraph (PdfParagraph): Paragraph to translate; carries the
                    ``layout_label`` assigned by the layout model.
                page (Page): Owning page, used by the base class for content
                    filter hints.
                pbar (tqdm | None): Stage progress bar.
                tracker (ParagraphTranslateTracker | None): Records the
                    input/output pair for ``translate_tracking.json``.
                page_font_map (dict[str, PdfFont] | None): Fonts on this page.
                xobj_font_map (dict[int, dict[str, PdfFont]] | None): Fonts per
                    XObject.
                paragraph_token_count (int): Estimated token count, used for
                    rate limiting.
                title_paragraph (TitleContextSnapshot | None): Document title context.
                local_title_paragraph (TitleContextSnapshot | None): Nearest
                    preceding heading.

            Returns:
                None: The base class mutates ``paragraph`` in place.
            """
            strict = needs_strict_terms(paragraph)
            translators.set_strict_terms(strict)
            skip_reason = skip_decision(self, paragraph, page, strict)

            if skip_reason:
                self.translation_config.raise_if_cancelled()
                with PbarContext(pbar):
                    with self._skip_lock:
                        self.skipped[skip_reason] += 1
                    return None

            return super().translate_paragraph(
                paragraph,
                page,
                pbar,
                tracker,
                page_font_map,
                xobj_font_map,
                paragraph_token_count,
                title_paragraph,
                local_title_paragraph,
            )

    return _LayoutFilteringILTranslator


# A paragraph this dominated by formula placeholders is a display equation, not
# prose about one. Translating it can only damage the surrounding markers, and
# what little text it holds is notation.
# 占位符占比达到该程度的段落是行间公式而非讲述公式的正文。翻译它只会破坏周围的
# 标记，而其中仅有的文字也是记号。
_FORMULA_MARKERS = re.compile(r"\{\s*v\s*\d+\s*\}")

# A formula placeholder, plus any spacing, sitting at the very front of a
# request — a bullet or a list number, not part of the sentence.
# 位于请求最前端的公式占位符及其后的空白——那是项目符号或列表编号，不属于句子。
_LEADING_PLACEHOLDER = re.compile(r"\s*\{\s*v\s*\d+\s*\}\s*")
_FORMULA_DOMINANCE = 0.6


def is_display_formula(text: str) -> bool:
    """
    中文：判断一段是否为行间公式（几乎全是公式占位符）。

    Detect a display equation from its placeholder density.

    Args:
        text (str): Paragraph text with formula markers in place.

    Returns:
        bool: ``True`` when markers and their surrounding punctuation account
            for most of the paragraph.
    """
    stripped = text.strip()
    if not stripped or len(stripped) > 400:
        return False
    markers = _FORMULA_MARKERS.findall(stripped)
    if not markers:
        return False
    without = _FORMULA_MARKERS.sub("", stripped)
    # What remains of a display equation is an equation number and punctuation.
    # 行间公式去掉占位符后，剩下的是公式编号与标点。
    residue = re.sub(r"[\s\d\.\,\;\:\(\)\[\]\{\}\+\-=<>/\\|]", "", without)
    return len(residue) <= max(2, int(len(stripped) * (1.0 - _FORMULA_DOMINANCE) * 0.15))


# Pseudocode is code, and translating it destroys it: the caption becomes a
# heading, "return" is rendered as the noun 回报, and the numbered steps overlap
# the formula placeholders they were carefully laid out around. The layout model
# has no `algorithm` class — it labels these blocks `plain text` on some pages
# and `table` on others (verified on a real paper) — so the block cannot be left
# alone by label. It is recognised by content instead, the way a display
# equation is.
# 伪代码就是代码，翻译它等于毁掉它：标题变成小节标题，"return" 被译成名词「回报」，
# 而带编号的步骤会与其精心环绕的公式占位符相互重叠。版面模型没有 `algorithm` 类——
# 实测同一篇论文里它有的页标成 `plain text`、有的标成 `table`——因此无法靠标签跳过，
# 只能像行间公式那样按内容识别。
# Mathematical notation, as it stands in the paragraph text this filter sees.
# The formula placeholders `{vN}` are NOT in it yet — they are substituted later,
# when the LLM request is assembled — so a detector keyed on `{vN}` matches
# nothing here. That is exactly why the "Input:" line of every algorithm was
# still being translated after the first fix: its placeholders had not been
# created at the point the decision is made. Both forms are accepted so the test
# holds wherever it is called from.
# 这里看到的段落文本中的数学记号。公式占位符 {vN} 此时尚未生成——它们要等到拼装 LLM
# 请求时才替换——因此以 {vN} 为准的判据在这里什么都匹配不到。这正是第一版修复后每个
# 算法的 "Input:" 行仍被翻译的原因：做判断时占位符还不存在。两种形式都接受，使该判据
# 在任何调用点都成立。
_MATH_NOTATION = re.compile(
    r"\{\s*v\s*\d+\s*\}"
    r"|\(cid:\d+\)"
    r"|[\u2190-\u21ff\u2200-\u22ff\u27f0-\u27ff\u2a00-\u2aff\u2100-\u214f]"
    r"|[×·∥∈∉⊂⊆∇∂√±∓≤≥≠≈≡→←↦⊤⊥∗∘⊗⊕∑∏∫]"
    r"|[⁰¹²³⁴⁵⁶⁷⁸⁹ₐₑₒₓₙ]"
)

_ALGORITHM_CAPTION = re.compile(r"^\s*Algorithm\s+\d+\b")
_ALGORITHM_STEP = re.compile(r"^\s*\d{1,3}:\s")
_ALGORITHM_KEYWORD = re.compile(
    r"^\s*(Input|Output|Require|Ensure|Repeat|Until|While|For|If|Else"
    r"|end\s+for|end\s+while|end\s+if|return)\b",
    re.I,
)
# Function words that betray a running sentence. A pseudocode header line ("Input:
# X with coefficients …") carries almost none; a proof sentence that merely opens
# with "If" or "For" is thick with them. Three is enough to tell the two apart
# without catching a genuine step line, which never strings this many together.
# 暴露句子结构的虚词。伪代码标题行几乎不含虚词；而以 "If"/"For" 开头的证明句子里满是
# 虚词。三个即可区分两者，又不会误伤真正的步骤行——步骤行绝不会连用这么多虚词。
_PROSE_WORDS = re.compile(
    r"\b(we|our|us|is|are|was|were|be|been|being|this|that|these|those|which"
    r"|where|when|by|and|but|to|of|the|from|than|apply|both|other|instead|will"
    r"|would|can|could|should|may|each|its|their|as|it|has|have)\b",
    re.I,
)
_MIN_PROSE_WORDS = 3


def is_pseudocode(text: str) -> bool:
    """
    中文：判断一段文字是否是算法伪代码（标题行、Input 行或带编号的步骤）。

    Detect a line of algorithm pseudocode from its text.

    The paragraph finder hands pseudocode over one line at a time, so this works
    on a single line rather than a whole block. Three shapes are recognised, and
    a prose guard keeps a proof sentence that opens with "If" or "For" out:

    * the caption ``Algorithm 3 Stabilized Gram Newton-Schulz`` — short and
      title-like, told apart from the sentence "Algorithm 3 gives …" by its
      length and lack of a sentence break;
    * a numbered step ``4: {v1} ← {v2}``;
    * a keyword header ``Input: {v1} with {v2}`` carrying formula placeholders.

    Args:
        text (str): Paragraph text, with formula markers already in place.

    Returns:
        bool: ``True`` when the line should be left untranslated.
    """
    stripped = text.strip()
    if not stripped or len(stripped) > 300:
        return False

    if _ALGORITHM_CAPTION.match(stripped):
        # A caption is a short title; "Algorithm 1 gives the standard …" is prose.
        # 标题是简短的题名；"Algorithm 1 gives the standard …" 是正文。
        return len(stripped) < 70 and not re.search(r"[.。]\s", stripped)

    if len(_PROSE_WORDS.findall(stripped)) >= _MIN_PROSE_WORDS:
        return False

    has_math = bool(_MATH_NOTATION.search(stripped))
    if _ALGORITHM_STEP.match(stripped) and (has_math or len(stripped) < 40):
        return True
    if _ALGORITHM_KEYWORD.match(stripped) and has_math:
        return True
    return False


# A run of letters this long is a word worth translating. Two is enough to let
# 「AI」 and 「ML」 through while rejecting the stray letter in 「4a」 or 「1e-3」.
# 达到这个长度的字母串才算值得翻译的词。取二既放行「AI」「ML」，又能挡住「4a」
# 「1e-3」里的孤立字母。
_WORD_RUN = re.compile(r"[A-Za-z\u00c0-\u024f]{2,}")

# CJK text is already translated, or is a term the model should leave alone;
# either way its presence means the fragment is not a bare number.
# 出现汉字说明该片段已是译文、或是本就不该动的术语；无论哪种，它都不是纯数字片段。
_CJK_TEXT = re.compile(r"[\u4e00-\u9fff\u3040-\u30ff]")

# Formula placeholders are removed before the test: a cell that is nothing but
# a formula has no words either, and sending it to the model risks the
# placeholder coming back mangled.
# 测试前先去掉公式占位符：整格只有一个公式同样没有词，送去翻译还可能把占位符弄坏。
_PLACEHOLDER_RUN = re.compile(r"\{\s*v\d+\s*\}")


def has_nothing_to_translate(text: str) -> bool:
    """
    中文：判断一段文字是否根本没有可翻译的内容（纯数字、符号、公式占位符）。

    Decide whether a fragment holds no word a translator could act on.

    Table cells are the case this exists for. 「0.0% → 1.4%」, 「163/202 (80.7%)」
    and 「1,338」 have nothing to translate and everything to lose: the model
    rewrote the first as 「0。 0% → 1.4%」 — a Chinese full stop where the decimal
    point was — and dropped the thousands separator out of the third.

    Args:
        text (str): Paragraph text, formula placeholders included.

    Returns:
        bool: ``True`` when the text contains no run of two or more letters and
            no CJK, once placeholders are removed.
    """
    stripped = _PLACEHOLDER_RUN.sub(" ", text or "").strip()
    if not stripped:
        return True
    if _CJK_TEXT.search(stripped):
        return False
    return _WORD_RUN.search(stripped) is None


# What a fragment is made of, counted in tokens rather than characters: a run of
# digits or a formula placeholder is one number, a run of two or more letters is
# one word. Characters would be the wrong unit — 「2025」 is four of them and one
# number.
# 片段的构成按记号而非字符计数：一串数字或一个公式占位符算一个数，两个以上连续字母算
# 一个词。按字符计是错的单位——「2025」是四个字符、一个数。
_NUMBER_TOKEN = re.compile(r"\d+|\{\s*v\d+\s*\}")
_WORD_TOKEN = re.compile(r"[A-Za-z\u00c0-\u024f]{2,}")


def is_mostly_numeric(text: str) -> bool:
    """
    中文：判断一段文字是否以数字为主——图表里这种框直接不翻译。

    Decide whether a fragment is mostly a number.

    A label inside a chart that is mostly numbers is a measurement, and the
    words beside them are its units. Translating one breaks it: 「(+12.6 pts)」
    came back with 「+3」 held in a formula placeholder, the digits split across
    two type sizes and 「pts」 rendered as 「分」; 「uRE 0.285」 became 「为0.285」;
    and 「AIME 2025」 has nothing to gain from being touched at all. Leaving such
    a label in English is always at least as good as translating it.

    Only used inside figures and tables. In prose a sentence quoting several
    numbers is still a sentence.

    Args:
        text (str): Fragment about to be translated, placeholders included.

    Returns:
        bool: ``True`` when the fragment holds at least one number and at least
            as many numbers as words.
    """
    numbers = len(_NUMBER_TOKEN.findall(text or ""))
    if not numbers:
        return False
    return numbers >= len(_WORD_TOKEN.findall(text or ""))


# Regions where a whitelist term must survive whatever it costs. A chart's axis
# label or a table cell has no prose to disturb, so the marker protection can be
# applied there without the word-order risk it carries in a sentence.
# 白名单术语必须不计代价保住的区域。图表的坐标轴标签或表格单元格里没有可被打乱的行文，
# 因此可以在那里使用标记保护，而不必承担它在句子里带来的语序风险。
_STRICT_TERM_LABELS = frozenset({"figure", "table", "table_footnote"})


def needs_strict_terms(paragraph) -> bool:
    """
    中文：判断一个段落是否属于「白名单必须硬保护」的区域（图、表）。

    Decide whether a paragraph sits where whitelist terms must be protected.

    Args:
        paragraph (PdfParagraph): The paragraph about to be translated.

    Returns:
        bool: ``True`` for text inside a figure or a table. Captions are not
            included — they read as prose and the user asked for the softer
            treatment there.
    """
    label = (getattr(paragraph, "layout_label", "") or "").strip().lower()
    if label in _STRICT_TERM_LABELS:
        return True
    # Text drawn inside a Form XObject is a figure's own labelling even when the
    # layout model gave it no useful label.
    # 画在 Form XObject 里的文字属于图自身的标注，即使版面模型没给出有用的标签。
    return bool(getattr(paragraph, "xobj_id", None))


def _install_formula_probe(cls) -> None:
    """
    中文：调试用——在 worker 进程内记录段落被拆成了哪些公式单元。

    Log how a paragraph decomposes into formula units, inside the worker.

    Enabled only when ``LOCALTRANS_FORMULA_PROBE`` names a substring to look for.
    The worker runs in its own process, so a probe installed in the parent never
    applies; this hook lives where the class is actually built.

    Args:
        cls (type): The filtering translator class to wrap.
    """
    import os

    needle = os.environ.get("LOCALTRANS_FORMULA_PROBE")
    if not needle:
        return

    real = cls.translate_paragraph

    def probe(self, paragraph, page, *args, **kwargs):
        text = paragraph.unicode or ""
        if needle in text:
            logger.warning("PROBE paragraph %r", text[:70])
            for index, comp in enumerate(paragraph.pdf_paragraph_composition):
                formula = comp.pdf_formula
                if formula is not None:
                    chars = formula.pdf_character
                    logger.warning(
                        "  [%d] FORMULA %r x=%s top=%s bot=%s x_offset=%s y_offset=%s",
                        index,
                        "".join(c.char_unicode or "?" for c in chars)[:24],
                        [round(c.box.x, 1) for c in chars][:8],
                        [round(c.box.y2, 1) for c in chars][:8],
                        [round(c.box.y, 1) for c in chars][:8],
                        getattr(formula, "x_offset", None),
                        getattr(formula, "y_offset", None),
                    )
                elif comp.pdf_line is not None:
                    line = "".join(c.char_unicode or "?" for c in comp.pdf_line.pdf_character)
                    logger.warning("  [%d] LINE %r", index, line[:60])
        return real(self, paragraph, page, *args, **kwargs)

    cls.translate_paragraph = probe


def install(skip_labels: set[str], skip_bibliography: bool = True,
            anchors=None, skip_content: bool = True,
            skip_pseudocode: bool = True) -> type:
    """
    中文：把 BabelDOC 的 ILTranslator 替换为过滤子类。

    Activate selective translation for the current process.

    Rebinds ``babeldoc.format.pdf.high_level.ILTranslator`` to a filtering
    subclass configured with ``skip_labels``. Call this in the worker process
    before invoking BabelDOC's translate entry point.

    The class is installed even when there is nothing to skip. It also carries
    the narrowed rich-text rule (see :meth:`pre_translate_paragraph`), which is
    what keeps a run-in bold heading such as 「致谢」 bold; returning early with
    selective translation switched off silently took that away too.

    Args:
        skip_labels (set[str]): Layout labels to leave untranslated.
        skip_bibliography (bool): Also skip paragraphs that read as citations,
            whatever label the layout model gave them.
        anchors (DocumentAnchors | None): Structural anchors from the source
            PDF. When present, a paragraph holding a citation anchor is skipped
            regardless of its label or text.
        skip_content (bool): Whether the content-based rules — bibliography,
            display equations, pseudocode — apply at all. Cleared when the
            settings page turns selective translation off.
        skip_pseudocode (bool): Also skip paragraphs that read as pseudocode.
            Its own switch now, so a reader who wants translated algorithm
            blocks can have them without giving up the other content rules.

    Returns:
        type: The installed class.

    Raises:
        RuntimeError: If BabelDOC no longer exposes ``ILTranslator`` on its
            ``high_level`` module, which would mean this patch has silently
            stopped applying.
    """
    import babeldoc.format.pdf.high_level as high_level

    if not hasattr(high_level, "ILTranslator"):
        raise RuntimeError(
            "babeldoc.format.pdf.high_level.ILTranslator not found. "
            "Selective translation relies on this symbol; the installed BabelDOC "
            "version is incompatible."
        )

    if not hasattr(high_level, "ILTranslatorLLMOnly"):
        raise RuntimeError(
            "babeldoc.format.pdf.high_level.ILTranslatorLLMOnly not found. "
            "Selective translation relies on this symbol; the installed BabelDOC "
            "version is incompatible."
        )

    _BUILT.clear()
    cls = _build_class()
    _install_formula_probe(cls)
    llm_cls = _build_llm_only_class()

    # Both pipelines share the configuration and the counter, so a run reports
    # one total however BabelDOC chose to translate it.
    # 两条管线共用配置与计数器，因此无论 BabelDOC 选了哪一条，一次运行只报一份总数。
    counter = Counter()
    for target in (cls, llm_cls):
        target.skip_labels = set(skip_labels)
        target.skip_bibliography = skip_bibliography and skip_content
        target.skip_pseudocode = skip_pseudocode and skip_content
        target.skip_content = skip_content
        target.anchors = anchors if skip_content else None
        target.skipped = counter

    high_level.ILTranslator = cls
    high_level.ILTranslatorLLMOnly = llm_cls

    # Assert the patch took. Replacing a module attribute only works while
    # BabelDOC reads the class through that attribute; if a future version
    # imports it directly at the call site, this module would go on logging
    # 「selective translation active」 while doing nothing — which is exactly how
    # the cloud path stayed broken. Better to refuse to start.
    # 断言补丁确实生效。替换模块属性只在 BabelDOC 通过该属性读取类时有效；若将来某个版本
    # 在调用处直接导入，本模块会继续打印「选择性翻译已启用」却什么也不做——云端那条路径正是
    # 这样一直坏着的。宁可拒绝启动。
    for name, expected in (("ILTranslator", cls), ("ILTranslatorLLMOnly", llm_cls)):
        if getattr(high_level, name) is not expected:
            raise RuntimeError(
                f"选择性翻译未能接入 BabelDOC 的 {name}：替换后读回的仍是原类。"
                "这一版 BabelDOC 可能改了引用方式。"
            )
    if skip_labels or skip_content:
        logger.info("Selective translation active; skipping labels: %s%s",
                    sorted(skip_labels),
                    "" if skip_content else " (content rules off)")
    else:
        logger.info("Selective translation off; every paragraph will be translated")
    return cls


def skipped_summary(cls: type | None) -> dict[str, int]:
    """
    中文：返回本次运行各版面标签被跳过的段落数。

    Report how many paragraphs were skipped, per layout label.

    Args:
        cls (type | None): The class returned by :func:`install`.

    Returns:
        dict[str, int]: Label to skipped-paragraph count. Empty when selective
            translation was not active.
    """
    if cls is None:
        return {}
    return dict(cls.skipped)
