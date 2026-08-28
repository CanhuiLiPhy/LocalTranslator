"""
中文：把自备的字体接进 BabelDOC 的字体族，用来替换默认的手写体。

Register a font of one's own into BabelDOC's font families.

BabelDOC picks a Chinese face per character from a hard-coded family table: an
upright body face for ordinary text, and a **script** face used for exactly one
thing — a run that was *italic* in the source, because Chinese has no true
italic and a calligraphic face is the conventional stand-in. The shipped choice
for Simplified Chinese is LXGW WenKai GB.

Substituting a different face is not a matter of dropping a file somewhere.
``FontMapper`` resolves every name through ``assets.get_font_and_metadata``,
which checks the cached file against a SHA3-256 recorded in
``EMBEDDING_FONT_METADATA`` and downloads from BabelDOC's own asset servers when
it does not match. A face those servers have never heard of has to be entered
into that table first, hash included, or the lookup ends in a download that
cannot succeed.

So this module does three things: reads the metrics BabelDOC needs out of the
font file, puts the file where the cache expects it, and writes the entry — after
which the ordinary lookup finds it and never reaches the network.

One face is bundled — Huiwen Fangsong, whose own version string reads
「本字体文件开源且免费商用」 and whose ``fsType`` is 0, so it may be shipped and
embedded. It is the default. Any other file is still the reader's to provide
and its licence theirs to observe, which is why this takes a path at all:
several of the faces people ask for may be used but not redistributed.
"""

from __future__ import annotations

import hashlib
import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

# CJK fonts are two bytes per code in the encoding BabelDOC writes, matching
# every CJK entry in its own metadata table.
# BabelDOC 写出的编码里，中日韩字体每个码位两字节，与它自己元数据表中所有中日韩
# 条目一致。
_CJK_ENCODING_LENGTH = 2

# The bundled face, relative to the package root. Ships with the wheel; absent
# from a checkout that has not fetched it, which is why every use is guarded.
# 随包发布的字体，相对于包根目录。未取回该文件的检出里会缺失，所以每处使用都做了防护。
_BUNDLED = "assets/fonts/HuiwenFangSong.ttf"


def bundled() -> Path | None:
    """
    中文：返回随程序携带的汇文仿宋路径，没有则返回 None。

    Locate the bundled Huiwen Fangsong file.

    Returns:
        Path | None: Absolute path to the shipped font, or ``None`` when this
            installation does not have it — a source checkout without the
            binary, say — in which case BabelDOC's own choice stands.
    """
    path = Path(__file__).resolve().parent.parent / _BUNDLED
    return path if path.is_file() else None


def _sha3_256(path: Path) -> str:
    """
    中文：算文件的 SHA3-256，格式与 BabelDOC 元数据一致。

    Hash a file the way BabelDOC's asset table records it.

    Args:
        path (Path): File to hash.

    Returns:
        str: Lower-case hexadecimal digest.
    """
    digest = hashlib.sha3_256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metrics(path: Path) -> tuple[int, int, str]:
    """
    中文：从字体文件读出 ascent、descent 与字体名。

    Read the metrics BabelDOC stores alongside a font.

    Args:
        path (Path): Font file.

    Returns:
        tuple[int, int, str]: Ascent, descent and the font's own name. Ascent
            and descent are in the thousandths BabelDOC's table uses, and the
            descent is negative.
    """
    import pymupdf

    font = pymupdf.Font(fontfile=str(path))
    ascent = int(round(font.ascender * 1000))
    descent = int(round(font.descender * 1000))
    return ascent, descent, font.name or path.stem


def install(font_path: str | Path | None, *, lang_out: str = "zh",
            bold: bool = False, italic: bool = False,
            serif: bool = True) -> str | None:
    """
    中文：把指定字体注册为中文手写体（原文斜体处使用的那一档）。

    Register a font as the Simplified Chinese script face.

    Args:
        font_path (str | Path | None): Path to a TTF or OTF. ``None`` falls back
            to the bundled Huiwen Fangsong; a path that does not exist leaves
            BabelDOC's own choice in place, because a translation with the wrong
            calligraphic face is a far better outcome than one that fails to
            start.
        lang_out (str): Target language code, used to look up the family the
            run will actually consult. This matters: BabelDOC resolves a family
            by substring, so ``"zh"`` matches neither ``CN`` nor any other
            branch and lands on the English family. Writing into ``CN`` alone
            left the substitution with no effect at all.
        bold (bool): Whether the face is bold. Affects which source runs it can
            be matched against.
        italic (bool): Whether the face is italic.
        serif (bool): Whether the face is serif. A Fangsong or Song face is;
            a Kai or Hei face is not. ``FontMapper`` matches on this, so it has
            to describe the file honestly.

    Returns:
        str | None: The registered file name, or ``None`` when nothing changed.
    """
    source = Path(font_path).expanduser() if font_path else bundled()
    if source is None:
        return None
    if not source.is_file():
        logger.warning("Script font %s not found; keeping BabelDOC's own choice", source)
        return None

    try:
        from babeldoc.assets.assets import get_cache_file_path
        from babeldoc.assets.embedding_assets_metadata import (
            ALL_FONT_FAMILY,
            EMBEDDING_FONT_METADATA,
            get_font_family,
        )
    except Exception:  # noqa: BLE001
        logger.warning("Could not reach BabelDOC's font tables; script font unchanged",
                       exc_info=True)
        return None

    name = source.name
    try:
        cached = Path(get_cache_file_path(name, "fonts"))
        cached.parent.mkdir(parents=True, exist_ok=True)
        if not cached.exists() or cached.stat().st_size != source.stat().st_size:
            # Contents only: a reader may well point this at a font of the
            # operating system's, and those carry flags that cannot be
            # reproduced on a copy.
            # 只复制内容：读者完全可能把这里指向一款系统自带的字体，而那些字体带着无法
            # 在副本上重建的文件标志。
            shutil.copyfile(source, cached)

        ascent, descent, font_name = _metrics(cached)
        EMBEDDING_FONT_METADATA[name] = {
            "ascent": ascent,
            "bold": int(bold),
            "descent": descent,
            "encoding_length": _CJK_ENCODING_LENGTH,
            "file_name": name,
            "font_name": font_name,
            "italic": int(italic),
            "monospace": 0,
            "serif": int(serif),
            "sha3_256": _sha3_256(cached),
            "size": cached.stat().st_size,
        }

        # Whichever family this run will actually consult, and only that one.
        # The others belong to targets this substitution was not asked about,
        # and quietly restyling Japanese output because someone chose a Chinese
        # face would be surprising.
        #
        # The family has to be resolved rather than assumed. ``get_font_family``
        # tests for "CN", "TW", "JP" and so on as substrings of the upper-cased
        # code, so the ordinary ``zh`` matches none of them and falls through to
        # the English family — a substitution written into ``CN`` would sit there
        # unused, which is exactly what happened.
        # 只改本次运行真正会用到的那个字体族。其它族对应的是本次替换并未涉及的目标
        # 语种，因为有人选了一款中文字体就顺手改掉日文排版，会让人意外。
        #
        # 字体族必须查出来而不能想当然。get_font_family 是把语言码大写后按子串匹配
        # "CN"、"TW"、"JP" 等，因此常见的 "zh" 一个都匹配不上，最终落到英文族——
        # 写进 CN 的替换就那么闲置着，先前正是如此。
        family = get_font_family(lang_out)
        if isinstance(family, str):
            family = ALL_FONT_FAMILY[family]
        family["script"] = [name]
    except Exception:  # noqa: BLE001
        logger.warning("Could not register script font %s; keeping the default", name,
                       exc_info=True)
        return None

    logger.info("Script font for %s (used where the source is italic) set to %s",
                lang_out, font_name)
    return name
