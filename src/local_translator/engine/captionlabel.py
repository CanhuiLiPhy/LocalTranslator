"""
中文：图注编号后面跟一个全角句号，在版面上留下一个六点宽的洞。

A full stop after a figure number leaves a six-point hole in the caption.

**The symptom.** The caption of figure 3 in the IEEE-format paper reads::

    图3。      CSI反馈模型解空间。曲线表示收敛路径

with a visible gap between the label and the text the caption's own sentences do
not have.

**Why it is not a space.** There is no space glyph there. Read character by
character, the label ends at x=332.5 and the body starts at x=342.4, and nothing
is drawn between them. The gap is inside the ``。`` itself: an ideographic full
stop occupies a full em — eight points at this size — with the dot at the left
of it, so it carries about six points of blank on its right. The source's ``.``
is two points wide and carries none. Whenever a run happened to render the label
as ``图3.`` instead, the caption closed up and the gap was gone.

**Why the full stop is wrong anyway.** ``。`` ends a sentence. A figure number
is a label, not a sentence, and Chinese practice writes ``图 3`` or ``图3.``.
The model produces one or the other from run to run — the source it is
translating is ``Fig. 3.``, and a full stop is a defensible reading of the final
period — so the label is normalised here rather than argued about in a prompt.

**Scope.** Only the caption's own leading label, only when a numbering word is
followed by a number and then the full stop, and only for the caption labels
this pipeline emits. A ``。`` anywhere else in the caption, including the one
that ends its first sentence, is left alone.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# A caption's leading label: a numbering word, a number, then an ideographic
# full stop. Anchored at the start, so only the label is ever matched.
# 图注开头的标签：编号词、编号、一个全角句号。锚定在开头，因此只会匹配到标签。
_LABEL = re.compile(r"^((?:图|表|算法|附图|附表)\s*(?:[0-9]+|[IVXLC]+)\s*)。")


def normalise(text: str) -> str:
    """
    中文：把图注开头的「图3。」改成「图3.」。

    Replace the ideographic full stop that closes a caption's label.

    Args:
        text (str): The translated caption.

    Returns:
        str: The same text with the label's ``。`` replaced by ``.``. Text that
            does not open with a label is returned unchanged.
    """
    if not text:
        return text
    return _LABEL.sub(lambda m: f"{m.group(1)}.", text, count=1)


def install() -> None:
    """
    中文：译文写回段落之前，先把图注标签的全角句号改掉。

    Normalise a caption's label as the translation is written back.

    Wraps ``ILTranslator.post_translate_paragraph``, which both pipelines go
    through — the LLM-only one calls the same method on its inner translator —
    and which is where the translated text is turned into the characters the
    page will carry. Editing here means the label is normalised once, before
    anything measures or lays it out.

    Idempotent, so a second call is harmless.

    Raises:
        AttributeError: If BabelDOC no longer exposes the method, so an upgrade
            that moves it fails at startup rather than silently leaving the hole
            back in every caption.
    """
    from babeldoc.format.pdf.document_il.midend.il_translator import ILTranslator

    if getattr(ILTranslator.post_translate_paragraph, "_captionlabel_patched", False):
        return

    original = ILTranslator.post_translate_paragraph

    def post_translate_paragraph(self, paragraph, tracker, translate_input,
                                 translated_text, *args, **kwargs):
        """中文：先规整标签，再照常写回。 / Normalise the label, then proceed."""
        try:
            fixed = normalise(translated_text)
            if fixed != translated_text:
                logger.debug("Normalised a caption label: %r", translated_text[:24])
                translated_text = fixed
        except Exception:  # noqa: BLE001
            # A caption that cannot be examined keeps its label, which is how
            # every caption behaved before this existed.
            # 看不了的图注保留原标签，也就是本模块存在之前所有图注的行为。
            logger.warning("Could not normalise a caption label", exc_info=True)
        return original(self, paragraph, tracker, translate_input, translated_text,
                        *args, **kwargs)

    post_translate_paragraph._captionlabel_patched = True
    ILTranslator.post_translate_paragraph = post_translate_paragraph
