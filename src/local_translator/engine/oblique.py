"""
中文：原文的斜体用同一款字体倾斜写出，而不是换成另一款书法体。

Set an italic run in the same face, slanted, rather than in a different one.

**What BabelDOC does.** Chinese has no italic, so its font table carries a
fourth tier beside serif, sans and fallback: a **script** face, used for exactly
one thing — a run the source set in italic. For Simplified Chinese the shipped
choice is LXGW WenKai GB, a handwriting face.

**Why that is the wrong answer here.** An italic run in a paper is not
calligraphy. ``understand, generate, and optimize materials`` on page 1 of the
document this was reported on is set in ``NimbusRomNo9L-MediItal`` — the same
Times the paragraph around it uses, slanted. Italic there means emphasis,
exactly as bold does, and answering it with a different typeface says something
the source never said.

**What this does instead.** The body face is registered a second time under its
own name, marked italic, and installed as the script tier. The font matcher then
picks it for italic runs — same outlines, same metrics, same colour on the page
— and the renderer gives those characters a sheared text matrix, so they come
out slanted. ``1 0 0 1 x y Tm`` becomes ``1 0 c 1 x y Tm``, which is how a PDF
obliques type and what a viewer does for "faux italic".

Naming the face twice is what makes the second half possible: after translation
a character carries a font id and nothing else, so the id is the only thing left
that can say "this run was italic".

Setting ``pdf.script_font`` overrides all of it — someone who wants a
calligraphic face for italic can still have one.
"""

from __future__ import annotations

import logging
import math
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

# Slant, in degrees. Ten to twelve is what a faux-italic routine applies and
# what the eye reads as italic; more looks like a mistake.
# 倾斜角度。仿斜体通常取十到十二度，眼睛读到的正是"斜体"；再大就像出错了。
_DEGREES = 11.0

# The shear that produces it, as the `c` element of the text matrix.
# 对应的错切量，即文本矩阵的 c 分量。
_SHEAR = math.tan(math.radians(_DEGREES))

# Suffix marking the second registration of a face. The name reaches the PDF as
# a name object, where "#" introduces a hex escape, so it must not appear here.
# 同一款字体第二次注册时所用的后缀。这个名字会作为 PDF 名字对象写进文件，而 "#" 在
# 名字对象里是十六进制转义的引导符，因此不能出现在这里。
_SUFFIX = "-BabelOblique.ttf"

# Font ids registered as oblique, filled in by :func:`install_face`.
# 已注册为倾斜档的字体 id，由 install_face 填入。
_OBLIQUE: set[str] = set()

# How many characters were actually slanted, so a run that quietly does nothing
# is visible in the log rather than looking like a document with no italics.
# 实际被倾斜的字符数。若某次运行悄悄什么都没做，这个数会在日志里显出来，而不是让人
# 误以为文档里本来就没有斜体。
_COUNT = [0]


def slanted_count() -> int:
    """
    中文：返回本进程中被倾斜的字符数。

    Return how many characters have been slanted in this process.

    Returns:
        int: The running count.
    """
    return _COUNT[0]


# A character the target language cannot do without, used to find which of the
# normal-tier faces will actually be rendering it. The tier is ordered
# best-first for Latin, so for a CJK target the first entry is a Latin face that
# has no CJK glyph at all and would be slanted to no effect.
# 目标语言绕不开的一个字符，用来找出 normal 档里真正会被用来排它的那款字体。该档是
# 按拉丁文优先排序的，因此中日韩目标语言的第一个条目是一款完全没有汉字的拉丁字体，
# 把它倾斜等于什么都没做。
_PROBE = {"zh": "字", "ja": "字", "ko": "한", "ru": "б", "ar": "ا"}


def _upright_faces(family: dict, lang_out: str, metadata: dict) -> list[str]:
    """
    中文：在正文档里找出真正会用来排目标语言的那些字体（含粗细各一）。

    Find the normal-tier faces that will actually render the target language.

    Every weight is returned, not just the first. The font matcher picks a face
    by weight as well as by tier, so a script tier holding only the bold face
    leaves an ordinary italic run with nothing to match and falls back to the
    upright body text.

    Args:
        family (dict): The font family this run resolved to.
        lang_out (str): Target language code.
        metadata (dict): BabelDOC's font metadata table.

    Returns:
        list[str]: Names of the faces to slant, in the family's own order, or
            an empty list when none can be read.
    """
    import pymupdf
    from babeldoc.assets.assets import get_cache_file_path

    probe = _PROBE.get(lang_out.lower().split("-")[0], "A")
    found: list[str] = []
    for name in family.get("normal", ()):
        if name not in metadata:
            continue
        path = Path(get_cache_file_path(name, "fonts"))
        if not path.is_file():
            continue
        try:
            if pymupdf.Font(fontfile=str(path)).has_glyph(ord(probe)):
                found.append(name)
        except Exception:  # noqa: BLE001
            logger.debug("Could not read %s while looking for the upright face",
                         name, exc_info=True)
        # One weight of each is enough; the family lists other scripts' faces
        # after them and slanting those would do nothing here.
        # 每种字重各取一款即可；字体族在其后列出的是其它文种的字体，倾斜它们在这里
        # 没有意义。
        if len(found) >= 2:
            break
    return found


def install_face(lang_out: str = "zh") -> str | None:
    """
    中文：把正文字体再注册一份作为「斜体档」，返回它的名字。

    Register the body face a second time, as the script tier.

    Args:
        lang_out (str): Target language code, used to find the family this run
            will consult. Resolved rather than assumed: BabelDOC matches family
            names as substrings of the upper-cased code, so ``zh`` reaches the
            English family, not ``CN``.

    Returns:
        str | None: The name the oblique face is registered under, or ``None``
            when the family could not be read, in which case BabelDOC's own
            script face stays in place.
    """
    try:
        from babeldoc.assets.assets import get_cache_file_path
        from babeldoc.assets.embedding_assets_metadata import (
            ALL_FONT_FAMILY,
            EMBEDDING_FONT_METADATA,
            get_font_family,
        )
    except Exception:  # noqa: BLE001
        logger.warning("Could not reach BabelDOC's font tables; italic unchanged",
                       exc_info=True)
        return None

    try:
        family = get_font_family(lang_out)
        if isinstance(family, str):
            family = ALL_FONT_FAMILY[family]

        sources = _upright_faces(family, lang_out, EMBEDDING_FONT_METADATA)
        if not sources:
            return None

        names = []
        for source in sources:
            name = _register(source, EMBEDDING_FONT_METADATA, get_cache_file_path)
            if name is not None:
                names.append(name)
                _OBLIQUE.add(name)
        if not names:
            return None

        # Regular before bold. ``map_in_type`` matches weight properly, but
        # ``map`` follows it with a loop that does not:
        #
        #     for script_font in self.script_fonts:
        #         if italic and script_font.has_glyph(current_char):
        #             return script_font
        #
        # which is what an italic run reaches whenever the tier's serif-ness
        # does not match its own. First in the list wins there, so an ordinary
        # italic run in a sans-serif context was coming out bold.
        # 常规体排在粗体之前。map_in_type 会正确匹配字重，但 map 在它之后还有一个不
        # 匹配字重的循环（见上），斜体段落只要档位的衬线属性与自身不符就会落到那里。
        # 那里是「列表第一个胜出」，于是无衬线上下文里一段普通斜体就排成了粗体。
        names.sort(key=lambda name: "-Bold" in name)
        family["script"] = names
    except Exception:  # noqa: BLE001
        logger.warning("Could not register an oblique face; italic unchanged",
                       exc_info=True)
        return None

    logger.info("Italic runs will be set in %s, slanted %.0f°",
                ", ".join(sources), _DEGREES)
    return names[0]


def _link(original: Path, alias: Path) -> bool:
    """
    中文：把倾斜档指向正文字体本体，优先用符号链接，不行再拷贝一份。

    Point the oblique alias at the upright file it duplicates.

    The alias exists only so the asset loader finds *a* file under the second
    name; its contents are the upright face byte for byte. A symbolic link
    satisfies every reader of it — ``verify_file`` opens the path and hashes
    what it reads, ``pymupdf.Font`` opens it the same way — while costing no
    disk at all, where a copy costs another 13 MB per weight.

    A link is not always available, so a copy remains the fallback: some file
    systems refuse them, and on Windows creating one needs a privilege an
    ordinary account does not have.

    Args:
        original (Path): The upright face already in the cache.
        alias (Path): Where the second registration expects to find a file.

    Returns:
        bool: Whether the alias now resolves to the upright face's bytes.
    """
    alias.parent.mkdir(parents=True, exist_ok=True)

    # An alias left by an earlier run may be a link whose target has since been
    # deleted. It still passes ``exists()`` in the negative — a dangling link is
    # not "exists" — but it must be cleared before a new one can be made, and
    # leaving it there would send the loader to the network for a name its
    # servers have never heard of, which ends in exit().
    # 上一次运行留下的别名可能是一条目标已被删除的链接。它过不了 exists()——断链不算
    # "存在"——但必须先清掉才能建新的；留着它会让加载器为一个其服务器从未听说过的名字
    # 去联网，最终以 exit() 收场。
    if alias.is_symlink() and not alias.exists():
        alias.unlink(missing_ok=True)

    if alias.exists():
        if alias.is_symlink() or alias.stat().st_size == original.stat().st_size:
            return True
        alias.unlink(missing_ok=True)

    try:
        alias.symlink_to(original)
        return True
    except OSError:
        logger.debug("Could not link %s to %s; copying instead", alias, original,
                     exc_info=True)

    try:
        shutil.copy2(original, alias)
        return True
    except OSError:
        logger.warning("Could not create %s; italic unchanged", alias, exc_info=True)
        return False


def _register(source: str, metadata: dict, get_cache_file_path) -> str | None:
    """
    中文：把一款字体再注册一份作为倾斜档，返回新名字。

    Register one face a second time, under a name marked italic.

    Args:
        source (str): Name of the upright face to copy.
        metadata (dict): BabelDOC's font metadata table, modified in place.
        get_cache_file_path (callable): BabelDOC's cache path resolver.

    Returns:
        str | None: The new name, or ``None`` when the upright face is not in
            the cache yet and so cannot be copied.
    """
    name = source.removesuffix(".ttf") + _SUFFIX
    if name not in metadata:
            # The asset loader walks every key in the table and checks the file
            # its *name* resolves to, then goes to the network for anything
            # missing — and calls exit() when it cannot reach it. So the second
            # registration needs a real file of its own.
            # 资源加载器会遍历表中每一个键，按「键名」去找对应文件，缺什么就去联网取，
            # 取不到就直接 exit()。因此第二次注册必须有自己的实体文件。
        original_path = Path(get_cache_file_path(source, "fonts"))
        if not original_path.is_file():
            logger.debug("%s is not in the cache yet; italic unchanged", source)
            return None
        alias_path = Path(get_cache_file_path(name, "fonts"))
        if not _link(original_path, alias_path):
            return None

        entry = dict(metadata[source])
        # Same bytes, so the same hash and size still describe it; only the
        # italic flag and the name differ, because only they need to.
        # 字节完全相同，因此原有的哈希与大小依然成立；只有斜体标志与名字不同，
        # 因为需要不同的只有这两样。
        entry["italic"] = 1
        metadata[name] = entry
    return name


def install() -> None:
    """
    中文：让渲染器给斜体档的字符写一个错切的文本矩阵。

    Make the renderer slant the characters set in the oblique face.

    Wraps ``CharacterRenderUnit.render``. The wrapper swaps the text matrix the
    original writes — ``1 0 0 1 x y Tm`` — for a sheared one, and only for
    characters whose font id was registered by :func:`install_face`. Everything
    else, including the encoding, the colour and the glyph id, is left to
    BabelDOC.

    Idempotent, so a second call is harmless.

    Raises:
        AttributeError: If BabelDOC no longer exposes the class, so an upgrade
            that moves it fails at startup rather than quietly setting italic
            upright.
    """
    from babeldoc.format.pdf.document_il.backend.pdf_creater import CharacterRenderUnit

    if getattr(CharacterRenderUnit.render, "_oblique_patched", False):
        return

    original = CharacterRenderUnit.render

    def render(self, draw_op, context):
        """中文：斜体档的字符改用错切矩阵。 / Slant the characters set in the oblique face."""
        char = self.char
        style = getattr(char, "pdf_style", None)
        font_id = getattr(style, "font_id", None) if style is not None else None
        if font_id not in _OBLIQUE or getattr(char, "vertical", False):
            return original(self, draw_op, context)

        # The stream is a bit sequence, not a list of the pieces appended to
        # it, so the operations are collected first and edited before they go in.
        # 该流是比特序列，不是「已追加片段」的列表，因此先收集操作、改好再放进去。
        recorder = _Recorder()
        original(self, recorder, context)
        pieces, slanted = recorder.slanted(_SHEAR)
        for piece in pieces:
            draw_op.append(piece)
        _COUNT[0] += int(slanted)

    render._oblique_patched = True
    CharacterRenderUnit.render = render


class _Recorder:
    """
    中文：冒充绘制流，把原方法写出的操作收下来。

    Stand in for the draw stream and keep what the original wrote.

    ``CharacterRenderUnit.render`` only ever calls ``append`` on the stream it
    is handed — directly and through ``render_graphic_state`` — so recording
    that one method is enough to intercept everything it emits.
    """

    def __init__(self):
        self.pieces: list[bytes] = []

    def append(self, piece) -> None:
        """
        中文：收下一段操作。

        Keep one appended operation.

        Args:
            piece (bytes): PDF content-stream operations.
        """
        self.pieces.append(piece)

    def slanted(self, shear: float) -> list[bytes]:
        """
        中文：把其中的文本矩阵换成错切矩阵后返回。

        Return the recorded operations with the text matrix sheared.

        Args:
            shear (float): The ``c`` element of the text matrix, ``tan`` of the
                slant angle.

        Returns:
            tuple[list[bytes], bool]: The operations ready to append, and
                whether a text matrix was actually sheared. If none was found
                the operations come back unchanged, leaving the character as
                BabelDOC drew it.
        """
        upright = b"1 0 0 1 "
        replacement = f"1 0 {shear:f} 1 ".encode()
        out = []
        done = False
        for piece in self.pieces:
            if not done and isinstance(piece, bytes) and upright in piece and b"Tm" in piece:
                piece = piece.replace(upright, replacement, 1)
                done = True
            out.append(piece)
        return out, done
