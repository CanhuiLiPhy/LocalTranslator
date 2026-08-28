"""
中文：有些子集字体只用名字后缀 .B 表示粗体，MuPDF 认不出来，于是加粗的小标题变成了正体。

Recover the weight of a subset font that records it only in its name.

**The symptom.** In the Nature Communications paper the section headings —
``Results``, ``Discussion`` — are bold in the source and come out in the regular
face, so the body text and the headings above it are set identically and the
page loses its structure.

**Why.** The output face is chosen by asking the source font whether it is
bold::

    bold = original_font.bold

and that flag comes from MuPDF, which reads the weight out of the font
programme and the name. Springer's subsetted CFF fonts declare nothing usable —
measured on this paper, the heading font ``FIDMKH+AdvOT3d287b35.B`` and the body
font ``FIDMAB+AdvOTdd63dae3`` carry the *same* descriptor: ``Flags 34``,
``StemV 0``, no ``FontWeight``. MuPDF reports ``is_bold=0`` for both, and it is
right to: nothing in the file says otherwise.

Nothing except the name. These fonts follow the convention of appending ``.B``
for bold and ``.I`` for italic, and across this document it separates them
exactly: the three ``.B`` faces are the title and the headings, the two ``.I``
faces are the italic runs, and the four with no suffix are body text. MuPDF
looks for ``Bold`` and ``Black`` spelled out, so a single letter goes past it.

**Scope.** Only the abbreviated dotted forms are added, and only to say *yes*
where MuPDF said no — a font MuPDF has already called bold is left alone. The
letter must not be followed by another, so ``.Book`` and ``.Italic`` are not
mistaken for ``.B`` and ``.I``.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# The dotted single-letter weight suffixes, at the end of a name or before the
# encoding tag some subsets append (``AdvOTa2fa4214.B+fb``). The negative
# lookahead keeps ``.Book`` and ``.Italic`` out.
# 点号加单字母的字重后缀，位于名字末尾，或位于某些子集追加的编码标记之前
# （如 AdvOTa2fa4214.B+fb）。后面的否定预查把 .Book、.Italic 挡在外面。
_BOLD = re.compile(r"\.B(?:I)?(?![A-Za-z])")
_ITALIC = re.compile(r"\.(?:I|BI)(?![A-Za-z])")


def bold_by_name(name: str) -> bool:
    """
    中文：名字后缀是否表示粗体。

    Report whether a font name's suffix says the face is bold.

    Args:
        name (str): The font's name, subset prefix and all.

    Returns:
        bool: ``True`` for a ``.B`` or ``.BI`` suffix.
    """
    return bool(name) and bool(_BOLD.search(name))


def italic_by_name(name: str) -> bool:
    """
    中文：名字后缀是否表示斜体。

    Report whether a font name's suffix says the face is italic.

    Args:
        name (str): The font's name, subset prefix and all.

    Returns:
        bool: ``True`` for an ``.I`` or ``.BI`` suffix.
    """
    return bool(name) and bool(_ITALIC.search(name))


def install() -> None:
    """
    中文：字体样式判定之后，再按名字后缀补上 MuPDF 漏掉的粗体与斜体。

    Add the weight a font's name declares and MuPDF did not find.

    Wraps ``ActiveILCreater._compute_font_style_flags``, whose result becomes
    the IL's font metadata and is what the font mapper later reads to choose an
    output face. The original runs first and its answers are kept; this only
    turns a ``False`` into a ``True``, never the reverse.

    Idempotent, so a second call is harmless.

    Raises:
        AttributeError: If BabelDOC no longer exposes the method, so an upgrade
            that moves it fails at startup rather than silently setting every
            heading in the regular face again.
    """
    from babeldoc.format.pdf.document_il.frontend.il_creater_active import (
        ActiveILCreater,
    )

    if getattr(ActiveILCreater._compute_font_style_flags, "_boldface_patched", False):
        return

    original = ActiveILCreater._compute_font_style_flags

    def _compute_font_style_flags(self, xref_id, *args, **kwargs):
        """中文：先按原逻辑判定，再用名字后缀补足。 / Original first, then the name."""
        flags = original(self, xref_id, *args, **kwargs)
        try:
            name = self.mupdf.extract_font(xref_id)[0] or ""
        except Exception:  # noqa: BLE001
            return flags
        if not flags.get("bold") and bold_by_name(name):
            flags["bold"] = True
            logger.debug("Font %r is bold by name; MuPDF did not say so", name)
        if not flags.get("italic") and italic_by_name(name):
            flags["italic"] = True
        return flags

    _compute_font_style_flags._boldface_patched = True
    ActiveILCreater._compute_font_style_flags = _compute_font_style_flags
