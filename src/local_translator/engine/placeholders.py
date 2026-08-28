"""
中文：模型漏掉一个公式占位符，那段公式就没了，而且没有任何一处会发现。

Refuse a translation that dropped a formula placeholder.

**The defect.** Inline mathematics is sent to the model as ``{v1}``-style
placeholders and put back where the model leaves them. If the model does not
return one, the formula it stood for is simply gone. Measured on page 2 of the
Nature Communications paper: the source reads ``The perovskite precursor
Nd₀.₈Sr₀.₂NiO₃ thin films``, and the translation came back as
``钙钛矿前驱体Sr0.2NiO3 薄膜`` — the ``Nd₀.₈`` placeholder was not returned, and
a chemical formula lost its first element without a word of complaint.

**Why nothing caught it.** The translation is checked three ways before it is
accepted — that it differs from the input, that its length is within a factor
of three, and that its edit distance is not tiny. A missing placeholder trips
none of them: the sentence is fully translated, the right length, and quite
different from the input. It just says something else.

**The fix.** The placeholders in the input are known exactly, so the ones in the
output can be compared against them. A translation missing any of them is
refused, which leaves the paragraph in the source language — where the retry
pass finds it and sends it again, and where, if it fails a second time, it is at
least *visible*. Losing a formula silently is the worse failure: nothing on the
page shows that anything is missing.

**Why this matters more than it used to.** The more of a formula is protected,
the more placeholders a paragraph carries, and the more chances there are for
one to go missing. Protecting inline mathematics properly — variables, marks
and the coefficients between them — roughly triples the count on a page like
this one, so the guard has to exist for that to be safe.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# The placeholder forms BabelDOC hands to the model. Only these are checked:
# a style tag may legitimately be dropped when the model merges two runs, but a
# formula placeholder stands for content that exists nowhere else.
# BabelDOC 交给模型的占位符形式。只检查这些：模型合并两段样式时，样式标签可以合理地消失，
# 但公式占位符代表的内容在别处并不存在。
_PLACEHOLDER = re.compile(r"\{v\d+\}")


def missing(translate_input: str, translated_text: str) -> list[str]:
    """
    中文：找出输入里有、输出里没有的占位符。

    List the placeholders the input carried and the output did not.

    Args:
        translate_input (str): The text sent to the model, placeholders and all.
        translated_text (str): What the model returned.

    Returns:
        list[str]: Each placeholder that went missing, once per lost
            occurrence, so a paragraph that used ``{v1}`` twice and got it back
            once is still reported.
    """
    if not translate_input or not translated_text:
        return []
    wanted = _PLACEHOLDER.findall(translate_input)
    if not wanted:
        return []
    got = _PLACEHOLDER.findall(translated_text)
    lost = []
    for name in wanted:
        if got.count(name) < wanted.count(name) - lost.count(name):
            lost.append(name)
    return lost


def install() -> None:
    """
    中文：应用译文之前先查占位符，缺了就拒收。

    Check the placeholders before a translation is applied, and refuse if any
    are missing.

    Wraps ``ILTranslator.post_translate_paragraph``, which is where a
    translation is written onto its paragraph and the only place that sees both
    the input and the output. Refusing returns ``False``, which is what
    BabelDOC already returns when it declines a translation, so the paragraph
    keeps its source text and the retry pass can find it.

    Idempotent, so a second call is harmless.

    Raises:
        AttributeError: If BabelDOC no longer exposes the method, so an upgrade
            that moves it fails at startup rather than silently letting
            formulas disappear again.
    """
    from babeldoc.format.pdf.document_il.midend.il_translator import ILTranslator

    if getattr(ILTranslator.post_translate_paragraph, "_placeholders_patched", False):
        return

    original = ILTranslator.post_translate_paragraph

    def post_translate_paragraph(self, paragraph, tracker, translate_input,
                                 translated_text, *args, **kwargs):
        """中文：缺占位符就不采用这条译文。 / Refuse a translation that lost one."""
        try:
            lost = missing(translate_input, translated_text)
        except Exception:  # noqa: BLE001
            lost = []
        if lost:
            logger.warning(
                "译文漏掉了 %d 个公式占位符（%s），已拒收，该段保留原文等待重试",
                len(lost), "、".join(lost[:4]))
            try:
                tracker.set_output(translated_text)
            except Exception:  # noqa: BLE001
                pass
            return False
        return original(self, paragraph, tracker, translate_input,
                        translated_text, *args, **kwargs)

    post_translate_paragraph._placeholders_patched = True
    ILTranslator.post_translate_paragraph = post_translate_paragraph
