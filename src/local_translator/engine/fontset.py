"""
中文：只装载目标语言真正用得上的字体；其余按需下载，实在取不到就退回系统字体。

Keep only the faces the language pair actually needs.

**What goes wrong without this.** BabelDOC's family table is per-language —
``CN`` names the four Source Han CN faces, ``JP`` names the four Japanese ones,
and so on — but ``__add_fallback_to_font_family()`` then appends *every other
language's* faces to *every* family, so that any character at all can be drawn
by something. The result is that one family lists 34 files, and ``FontMapper``
opens all 34 in its constructor:

    for font_file_name in self.font_file_names:
        font_path, font_metadata = assets.get_font_and_metadata(font_file_name)
        pymupdf_font = pymupdf.Font(fontfile=str(font_path))

``get_font_and_metadata`` downloads what is missing, and ``pymupdf.Font`` reads
the file into memory. Measured on this machine, for a Simplified Chinese target:
323 MB fetched to disk and 253 MB resident, of which the Japanese, Korean,
Hong Kong and Traditional Chinese faces — 123 MB — are never consulted, because
every character they could draw is already in the Simplified faces ahead of them
in the list.

**What this does.** Rebuilds the family the run will consult so it holds the
target language's own faces, the Latin faces, and the fallback tier, and nothing
else. The lists are the only thing that changes; the matching, the ordering
within what remains, and the tier semantics are all still BabelDOC's.

Three consequences follow, in the order they matter:

- a first run downloads what the language pair needs rather than the whole set;
- a translation holds the faces it uses rather than five languages' worth;
- a language chosen later fetches its faces then, which is what
  :func:`ensure` is for.

**When the network is not there.** BabelDOC answers a font it cannot fetch with
``logger.critical`` followed by ``exit(1)``, which on a server takes the process
with it. :func:`install_fallback` turns that into a substitution: a face from
the operating system that covers the same script is registered under the name
that could not be fetched, and the run continues in a different typeface rather
than not at all.
"""

from __future__ import annotations

import hashlib
import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

# The four Source Han faces each CJK family names, as a template over the
# region tag. BabelDOC's own tables are written out longhand; this is the same
# set, stated once.
# 每个中日韩字体族列出的四款思源字体，按地区标记参数化。BabelDOC 自己的表是逐条写
# 死的；这里把同一组内容只写一遍。
_HAN_SERIF = (
    "SourceHanSerif{0}-Bold.ttf",
    "SourceHanSerif{0}-Regular.ttf",
)

# The sans-serif pair. Kept apart because they are almost always dead weight:
# ``map_in_type`` chooses between the two by whether the *source* run was
# serif, and an academic paper is set in Times, XCharter or Computer Modern
# throughout, so every translated run asks for serif. Across 148 output PDFs
# from 42 papers on this machine, Source Han Sans appears on zero pages.
#
# And when a sans-serif run does turn up, dropping these costs nothing:
# matching falls through ``normal`` to the fallback tier, where
# GoNotoKurrent — which has no "serif" in its name and so answers the
# sans branch — draws it in Noto Sans CJK, which *is* Source Han Sans under
# its other name. Same outlines, one file instead of two.
# 无衬线的那一对。单独列出，是因为它们几乎总是白占地方：map_in_type 在两者之间的取舍
# 取决于「原文那一段」是不是衬线体，而学术论文通篇是 Times、XCharter 或 Computer
# Modern，于是每一段译文要的都是衬线。本机 42 篇论文共 148 个产物 PDF 里，思源黑体
# 出现在 0 页上。
#
# 而真出现无衬线段落时，去掉它们也没有代价：匹配会从 normal 落到备用档，那里的
# GoNotoKurrent 名字里没有 "serif"，正好应答无衬线这一支，用 Noto Sans CJK 画出来
# ——那本来就是思源黑体的另一个名字。同样的字形，只是少存一份。
_HAN_SANS = (
    "SourceHanSans{0}-Bold.ttf",
    "SourceHanSans{0}-Regular.ttf",
)

# Every region BabelDOC ships Source Han faces for.
# BabelDOC 提供思源字体的全部地区。
_REGIONS = ("CN", "TW", "HK", "JP", "KR")

# The handwriting face each region uses for a run the source set in italic.
# Listed here only so a region's own face survives the trim and the other
# regions' do not; which one is actually installed is decided elsewhere
# (see :mod:`engine.fonts` and :mod:`engine.oblique`).
# 各地区用于「原文斜体处」的手写体。列在这里只是为了让本地区那款留下、其它地区那些
# 被裁掉；真正装哪一款由别处决定（见 engine.fonts 与 engine.oblique）。
_SCRIPT = {
    "CN": ("LXGWWenKaiGB-Regular.1.520.ttf",),
    "TW": ("LXGWWenKaiTC-Regular.1.520.ttf",),
    "HK": ("LXGWWenKaiTC-Regular.1.520.ttf",),
    "JP": ("KleeOne-Regular.ttf",),
    "KR": ("MaruBuri-Regular.ttf",),
}

# The Latin faces, kept for every target: a paper translated into Chinese is
# still full of English — author names, units, code, every citation key — and
# these are the faces that draw it well.
# 拉丁字体，任何目标语言都保留：译成中文的论文里仍然满是英文——作者名、单位、代码、
# 每一个引用标记——把它们排得好看的正是这几款。
_LATIN = (
    "NotoSerif-Regular.ttf",
    "NotoSerif-Bold.ttf",
    "NotoSans-Regular.ttf",
    "NotoSans-Bold.ttf",
    "NotoSans-Italic.ttf",
    "NotoSans-BoldItalic.ttf",
    "NotoSerif-Italic.ttf",
    "NotoSerif-BoldItalic.ttf",
)

# The fallback tier: one face covering nearly every script there is, consulted
# only for a character none of the faces above can draw. Thirty megabytes for a
# case that may never arise in a given paper, which is why it can be switched
# off — see :func:`restrict`.
# 备用档：一款几乎覆盖所有文种的字体，只在上面那些都画不出某个字符时才会被查到。
# 为一个也许根本不会发生的情况付出三十兆，所以它可以关掉——见 restrict。
_FALLBACK = ("GoNotoKurrent-Regular.ttf", "GoNotoKurrent-Bold.ttf")

# Language code to region, matched as a substring of the upper-cased code,
# most specific first. This is deliberately *not* BabelDOC's own
# ``get_font_family`` rule: that one tests for "CN" and so on, which the
# ordinary code ``zh`` matches nowhere, landing it on the English family. Which
# family the run consults is still BabelDOC's answer; which faces belong in it
# is this one's.
# 语言码到地区的映射，按大写后的子串匹配，越具体的排越前。这里刻意不复用 BabelDOC
# 自己的 get_font_family 规则：那条规则匹配的是 "CN" 之类，而常见的 "zh" 一个都
# 匹配不上，最终落到英文族。「查哪个族」仍由 BabelDOC 决定，「族里该有哪些字体」由
# 这里决定。
_LANGUAGE_REGION = (
    ("ZH-HANT", "TW"),
    ("ZH-HANS", "CN"),
    ("ZH-TW", "TW"),
    ("ZH-HK", "HK"),
    ("ZH-MO", "HK"),
    ("ZH-CN", "CN"),
    ("ZH-SG", "CN"),
    ("HANT", "TW"),
    ("HANS", "CN"),
    ("TW", "TW"),
    ("HK", "HK"),
    ("KO", "KR"),
    ("KR", "KR"),
    ("JA", "JP"),
    ("JP", "JP"),
    ("CN", "CN"),
    ("ZH", "CN"),
)

# The four tiers a family is made of, in the order ``FontMapper`` reads them.
# 字体族的四个档位，顺序与 FontMapper 读取时一致。
_TIERS = ("normal", "script", "fallback", "base")

# The untouched family lists, keyed by the identity of the family dict, taken
# before the first trim. Restricting always starts from these rather than from
# whatever a previous call left behind, so calling it twice with two different
# languages gives the second language its own faces instead of the intersection.
# 首次裁剪之前的原始字体族列表，按字体族字典的身份存放。每次裁剪都从这份原件出发，
# 而不是上一次裁剪的残留，因此用两种不同语言各调一次，第二次得到的是该语言自己的
# 字体，而不是两次的交集。
_ORIGINAL: dict[int, dict[str, list[str]]] = {}


def region_of(lang_code: str | None) -> str | None:
    """
    中文：把语言码归到一个地区（CN/TW/HK/JP/KR），认不出来就返回 None。

    Map a language code to the region whose faces it needs.

    Args:
        lang_code (str | None): A language code as the settings carry it, such
            as ``"zh"``, ``"zh-TW"``, ``"ja"`` or ``"en"``. Case and separator
            are both insignificant.

    Returns:
        str | None: One of ``"CN"``, ``"TW"``, ``"HK"``, ``"JP"``, ``"KR"``, or
            ``None`` for a language written in the Latin alphabet, which needs
            no CJK face of its own.
    """
    code = (lang_code or "").upper().replace("_", "-")
    for prefix, region in _LANGUAGE_REGION:
        if prefix in code:
            return region
    return None


def faces(region: str | None, *, script: bool = True,
          sans: bool = False) -> set[str]:
    """
    中文：返回某个地区自己的字体文件名（衬线两款，可选加上黑体与手写体）。

    Return the font files that belong to one region.

    Args:
        region (str | None): A region tag as returned by :func:`region_of`.
            ``None`` yields the empty set.
        script (bool): Include the region's handwriting face, used where the
            source was italic. This application normally supplies its own — see
            :mod:`engine.fonts` and :mod:`engine.oblique` — so the shipped one
            is 24 MB that would be fetched and never consulted.
        sans (bool): Include the sans-serif pair. Off by default; see
            :data:`_HAN_SANS` for why, and what draws sans-serif runs instead.

    Returns:
        set[str]: File names as BabelDOC's metadata table keys them.
    """
    if region not in _REGIONS:
        return set()
    own = {name.format(region) for name in _HAN_SERIF}
    if sans:
        own |= {name.format(region) for name in _HAN_SANS}
    if script:
        own |= set(_SCRIPT.get(region, ()))
    return own


def script_faces(region: str | None) -> set[str]:
    """
    中文：返回某个地区的手写体文件名。

    Return the handwriting faces one region uses.

    Args:
        region (str | None): A region tag, or ``None``.

    Returns:
        set[str]: File names, empty for a region with none.
    """
    return set(_SCRIPT.get(region, ())) if region else set()


def every_script_face() -> set[str]:
    """
    中文：返回所有地区的手写体文件名。

    Return the handwriting faces of every region.

    Returns:
        set[str]: File names across all regions.
    """
    out: set[str] = set()
    for names in _SCRIPT.values():
        out |= set(names)
    return out


def needed(lang_out: str, lang_in: str | None = None, *,
           fallback: bool = True, script: bool = False,
           sans: bool = False) -> set[str]:
    """
    中文：列出这对语言真正需要的字体文件。

    List the font files this language pair actually needs.

    Args:
        lang_out (str): Target language code.
        lang_in (str | None): Source language code. Included because a bilingual
            output can be asked to re-draw the source side, and because the
            reader may set the pair either way round.
        fallback (bool): Keep the universal fallback face. Costs about 30 MB of
            memory and of disk, and is consulted only for a character none of
            the other faces can draw.
        script (bool): Include the region's shipped handwriting face. Off by
            default because this application installs one of its own over it.
        sans (bool): Include the sans-serif pair. Off by default; see
            :data:`_HAN_SANS`.

    Returns:
        set[str]: File names, always including the Latin faces.
    """
    wanted = set(_LATIN)
    for code in (lang_out, lang_in):
        wanted |= faces(region_of(code), script=script, sans=sans)
    if fallback:
        wanted |= set(_FALLBACK)
    return wanted


def _families(lang_out: str, lang_in: str | None):
    """
    中文：取出本次运行会被查到的那些字体族字典。

    Collect the family dicts this run will consult.

    Args:
        lang_out (str): Target language code.
        lang_in (str | None): Source language code.

    Returns:
        list[dict]: The distinct family dicts, de-duplicated by identity —
            several language codes routinely resolve to the same one.
    """
    from babeldoc.assets.embedding_assets_metadata import ALL_FONT_FAMILY, get_font_family

    found: dict[int, dict] = {}
    for code in (lang_out, lang_in):
        if not code:
            continue
        try:
            family = get_font_family(code)
        except Exception:  # noqa: BLE001
            logger.debug("Could not resolve a font family for %r", code, exc_info=True)
            continue
        if isinstance(family, str):
            family = ALL_FONT_FAMILY[family]
        found[id(family)] = family
    return list(found.values())


def restrict(lang_out: str, lang_in: str | None = None, *,
             fallback: bool = True) -> int:
    """
    中文：把字体族裁到这对语言真正用得上的那些字体，返回裁掉了几款。

    Trim the family lists down to the faces this language pair needs.

    Must run before ``FontMapper`` is constructed: it reads the lists once and
    keeps whatever it found, so a trim afterwards changes nothing.

    The ``normal`` and ``fallback`` tiers are filtered against :func:`needed`.
    The ``script`` tier is treated differently — only the *other* regions' faces
    are removed from it — because by the time this runs the tier usually holds a
    face this application put there (Huiwen Fangsong, or the body face
    registered a second time to be slanted), and filtering it against a list of
    BabelDOC's own names would throw that away. The ``base`` face is never
    removed: it is what the mapper falls back to when nothing else matches.

    If filtering would empty the ``normal`` tier the family is left exactly as
    it was — a run in the wrong typeface is a far better outcome than one that
    cannot start.

    Args:
        lang_out (str): Target language code.
        lang_in (str | None): Source language code.
        fallback (bool): Keep the universal fallback face; see :func:`needed`.

    Returns:
        int: How many distinct files were dropped, counted across all the
            families this run would consult. Zero means there was nothing to
            trim, which is also what a failure returns.
    """
    try:
        families = _families(lang_out, lang_in)
    except Exception:  # noqa: BLE001
        logger.warning("Could not reach BabelDOC's font tables; every face stays loaded",
                       exc_info=True)
        return 0

    wanted = needed(lang_out, lang_in, fallback=fallback)
    mine = set()
    for code in (lang_out, lang_in):
        mine |= script_faces(region_of(code))
    foreign_script = every_script_face() - mine
    dropped: set[str] = set()

    for family in families:
        # The trim always starts from the untouched lists, so that a second call
        # for another language is not intersected with the first.
        # 每次裁剪都从原始列表出发，避免为另一种语言的第二次调用与第一次求了交集。
        snapshot = _ORIGINAL.setdefault(
            id(family), {tier: list(family.get(tier, ())) for tier in _TIERS}
        )
        # The script tier is read live rather than from the snapshot: whatever
        # is in it now is what this application chose to put there.
        # script 档要读当前值而非快照：现在里面是什么，就是本程序选定放进去的。
        snapshot["script"] = list(family.get("script", ()))
        keep = set(wanted)

        trimmed = {tier: [n for n in snapshot[tier] if n in keep]
                   for tier in ("normal", "fallback")}
        trimmed["script"] = [n for n in snapshot["script"] if n not in foreign_script]
        if not trimmed["normal"]:
            logger.warning("Trimming would leave no body face for %r; keeping them all",
                           lang_out)
            continue
        trimmed["base"] = _base(snapshot.get("base", ()), keep, trimmed["normal"],
                                region_of(lang_out))

        for tier in _TIERS:
            before = set(snapshot[tier])
            dropped |= before - set(trimmed[tier])
            family[tier] = trimmed[tier]

    # A face dropped from one family may still be named by another this run
    # consults, in which case it was not really dropped.
    # 从某个字体族里裁掉的字体，可能仍被本次运行会查到的另一个族列出，那就不算裁掉。
    still: set[str] = set()
    for family in families:
        for tier in _TIERS:
            still |= set(family.get(tier, ()))
    dropped -= still

    if dropped:
        logger.info("Font set trimmed to %s: %d faces left out (%s)",
                    lang_out, len(dropped), ", ".join(sorted(dropped)))
    return len(dropped)


def _base(original, keep: set[str], body: list[str],
          region: str | None) -> list[str]:
    """
    中文：确定裁剪后的 base 档；原来的那款被裁掉时另选一款。

    Choose the base face that survives the trim.

    Every CJK family names ``SourceHanSansCN-Regular.ttf`` as its base, and the
    sans-serif pair is normally trimmed away, so the base has to move.

    Moving it is safe, and provably so: ``base`` never draws anything. Its one
    use in BabelDOC is ``typesetting.py``, which asks it how wide 「你」 is at a
    given point size in order to work out a scale. Every full-width CJK face
    answers with exactly the point size — measured here for Source Han Serif CN
    and Source Han Sans CN at 9, 10, 10.5 and 12 pt, identical to the last
    decimal — so the number the typesetter gets does not change.

    Args:
        original (Iterable[str]): The base tier before trimming.
        keep (set[str]): The faces that survive.
        body (list[str]): The trimmed ``normal`` tier, used as a last resort.
        region (str | None): The target's region, used to name its serif face.

    Returns:
        list[str]: A one-element base tier. Empty only when the family had none,
            which BabelDOC's own validator already rejects.
    """
    existing = list(original)
    if existing and existing[0] in keep:
        return existing

    if region:
        preferred = f"SourceHanSerif{region}-Regular.ttf"
        if preferred in keep:
            return [preferred]

    if body:
        return body[:1]
    return existing


def ensure(lang_out: str, lang_in: str | None = None, *,
           fallback: bool = True) -> tuple[int, int]:
    """
    中文：把这对语言需要而本地还没有的字体下载下来。

    Fetch the faces this language pair needs and does not have yet.

    Called when a language is chosen rather than when a translation starts, so
    that the download happens while the reader is still in the settings page
    instead of in the middle of a run. Safe to call repeatedly: a face already
    in the cache is verified against its recorded hash and left alone.

    Args:
        lang_out (str): Target language code.
        lang_in (str | None): Source language code.
        fallback (bool): Include the universal fallback face.

    Returns:
        tuple[int, int]: How many faces are now present, and how many were
            asked for. The two differ when the network could not be reached, in
            which case the missing ones are fetched at translation time instead
            — or substituted, if :func:`install_fallback` is in place.
    """
    try:
        from babeldoc.assets.assets import get_font_and_metadata
        from babeldoc.assets.embedding_assets_metadata import EMBEDDING_FONT_METADATA
    except Exception:  # noqa: BLE001
        logger.warning("Could not reach BabelDOC's font tables; nothing fetched",
                       exc_info=True)
        return 0, 0

    wanted = sorted(n for n in needed(lang_out, lang_in, fallback=fallback)
                    if n in EMBEDDING_FONT_METADATA)
    have = 0
    for name in wanted:
        try:
            get_font_and_metadata(name)
            have += 1
        except SystemExit:
            # BabelDOC answers an unreachable font with exit(); here that is a
            # missing file to report, not a reason to end the process.
            # BabelDOC 取不到字体时直接 exit()；在这里那只是「少一个文件」需要上报，
            # 而不是结束进程的理由。
            logger.warning("Could not fetch %s", name)
        except Exception:  # noqa: BLE001
            logger.warning("Could not fetch %s", name, exc_info=True)
    logger.info("Fonts for %s: %d of %d present", lang_out, have, len(wanted))
    return have, len(wanted)


# Faces of the operating system's own to fall back on, best first. Every path
# is checked before use and then probed for the script it is needed for, so
# entries for the other platforms — and for fonts a given machine happens not to
# have — cost nothing but a ``stat``.
#
# Order matters more than it looks. A ``.ttc`` is a collection and pymupdf opens
# face 0 of it, which is not always the weight one would choose:
# ``Songti.ttc`` opens as *Songti SC Black*, and setting a paper's body text in
# a black weight is worse than setting it in the wrong family. Single-face files
# come first for that reason, and ``Arial Unicode.ttf`` first of all — 22 MB,
# regular weight, and the only file here that covers Chinese, Japanese, Korean
# and Latin together.
# 用来顶替的系统自带字体，越靠前越优先。每个路径使用前都会检查存在性，再针对所需文种
# 探测覆盖情况，因此其它平台的条目、以及本机恰好没有的字体，代价只是一次 stat。
#
# 顺序比看上去重要。.ttc 是字体集合，pymupdf 打开的是其中第 0 号，而那一号未必是想要
# 的字重：Songti.ttc 打开是「宋体简 Black」，把正文排成黑体字重比排错字体族更糟。单
# 面文件因此排在前面，其中 Arial Unicode.ttf 又排第一——22 MB，常规字重，而且是这里
# 唯一一款同时覆盖中日韩与拉丁文的文件。
_SYSTEM_CANDIDATES = (
    # macOS
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
    "/System/Library/Fonts/STHeiti Light.ttc",
    "/System/Library/Fonts/Supplemental/Songti.ttc",
    # Linux
    "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/wqy-zenhei/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/arphic/uming.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
    # Windows
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simsun.ttc",
    "C:/Windows/Fonts/malgun.ttf",
    "C:/Windows/Fonts/meiryo.ttc",
    "C:/Windows/Fonts/times.ttf",
)

# A character each region cannot do without, used to test a candidate rather
# than assume it. Apple SD Gothic Neo covers Hangul and no Han at all, which a
# list of paths alone would never reveal.
# 各地区绕不开的字符，用来「测试」候选字体而不是「假定」它。Apple SD Gothic Neo 覆盖
# 谚文却完全没有汉字，光靠一张路径表是看不出来的。
_PROBE_CHAR = {"CN": "翻译", "TW": "翻譯", "HK": "翻譯", "JP": "翻訳", "KR": "번역"}

# CJK fonts are two bytes per code in the encoding BabelDOC writes; Latin ones
# are one. Getting this wrong garbles the text rather than the typeface.
# BabelDOC 写出的编码里，中日韩字体每码两字节，拉丁字体每码一字节。这个值弄错会把
# 文字本身弄乱，而不只是换个字形。
_CJK_ENCODING_LENGTH = 2
_LATIN_ENCODING_LENGTH = 1

# Prefix for the copy a substitution leaves in the cache. Deliberately not the
# name that was asked for: writing there would overwrite the genuine font with
# a stand-in, and a later run with the network back would go on using the
# stand-in because a file of that name is present and hashes to what the table
# was just told to expect.
# 顶替时留在缓存里的那份拷贝的前缀。刻意不用被请求的那个名字：写到那里等于用替身覆盖
# 真字体，而等网络恢复之后，那一趟仍会继续用替身——因为那个名字下确实有文件，且哈希
# 正好等于刚刚被写进表里的值。
_SUBSTITUTE_PREFIX = "system-fallback-"

# The faces kept on disk for languages this installation does not translate into
# by default: one each, the smallest that covers its script. Downloading them on
# demand works and is quick — measured at 4.6 s for a language's pair over the
# GitHub mirror — but it needs a network, and BabelDOC's four upstreams are
# ``raw.githubusercontent.com``, ``huggingface.co``, ``hf-mirror.com`` and
# ``modelscope.cn``, the first two of which are not reachable everywhere.
#
# Between them these three also carry 229 of the CJK Compatibility Ideographs
# (U+F900–FAFF) that the Simplified faces lack — which is, measured, the entire
# coverage the per-language trim gives up. The remaining 146 of that block are
# in no face BabelDOC ships at all.
# 为「本机默认不译入」的那些语言留在盘上的字体：每种语言一款，取覆盖该文种的最小的
# 那一款。按需下载是可行的，也很快——实测走 GitHub 镜像取一种语言的两款用 4.6 秒——
# 但它需要网络，而 BabelDOC 的四个上游是 raw.githubusercontent.com、huggingface.co、
# hf-mirror.com 与 modelscope.cn，前两个并非处处可达。
#
# 这三款合起来还补上了简体字体所缺的 229 个中日韩兼容汉字（U+F900–FAFF）——实测那正
# 是按语言裁剪所放弃的全部覆盖。该区段剩下的 146 个，BabelDOC 提供的任何字体都没有。
# One face, and a serif one.
#
# *One*, because of what each candidate adds over the Simplified faces and the
# universal fallback: measured across the 229 code points those leave uncovered,
# the Japanese face carries 227, the Korean face adds two (U+F95A 讀 and U+F9F3
# 麟, Korean-reading duplicates whose NFC forms are covered anyway), and the Hong
# Kong face adds none at all.
#
# *Serif*, because the slot it stands in for is a serif one. Choosing the
# smallest file — the sans face, 1.9 MB less — would have left the substitution
# self-contradictory: :func:`substitute` inherits the style flags from the slot,
# so the mapper would go on treating it as serif while the reader saw a sans
# face set as body text. The two cover the same 227 code points, so the smaller
# file buys nothing but the mismatch.
# 一款，而且是衬线的那款。
#
# 「一款」的依据是各候选比简体字体加通用备用档多覆盖多少：在那 229 个未覆盖码位上实测，
# 日文款带着其中 227 个，韩文款只多出两个（U+F95A 讀 与 U+F9F3 麟，韩文读音的重复形，
# 其 NFC 形本来就覆盖），港版款一个都不多。
#
# 「衬线」的依据是它顶替的那个槽位就是衬线槽。按「文件最小」去挑会选中黑体（小 1.9 MB），
# 那会让这次替换自相矛盾：substitute 从槽位继承样式标志，于是映射器仍当它是衬线，而读者
# 看到的是一款被当正文排的黑体。两者覆盖的是同样 227 个码位，小的那份除了这处错位什么都
# 没换来。
OFFLINE_FALLBACK = (
    "SourceHanSerifJP-Regular.ttf",   # 7.4 MB, covers 227 of the 229
)


def _region_of_file(name: str) -> str | None:
    """
    中文：从字体文件名推断它属于哪个地区。

    Infer which region a font file name belongs to.

    Args:
        name (str): A file name as BabelDOC's metadata table keys it.

    Returns:
        str | None: The region tag, or ``None`` for a Latin or universal face.
    """
    for region in _REGIONS:
        if any(name == pattern.format(region)
               for pattern in _HAN_SERIF + _HAN_SANS):
            return region
        if name in _SCRIPT.get(region, ()):
            return region
    return None


def _cached_face(region: str | None, exclude: str = "") -> Path | None:
    """
    中文：在字体缓存里找一款同地区、还留着的字体，作为下不下来时的第一顺位替补。

    Find another face for the same region that is already in the cache.

    This is the first rung of the ladder a failed download falls onto, and the
    one worth having: a face BabelDOC itself ships has the right metrics, the
    right encoding length and a design meant for the same script, where a font
    of the operating system's is merely whatever happened to be installed.

    What makes the rung exist is a small set kept on purpose — one face per
    language, the smallest that covers the script, listed in
    :data:`OFFLINE_FALLBACK`. Between them they carry the CJK Compatibility
    Ideographs (U+F900–FAFF) that the Simplified faces do not, which is the only
    coverage the per-language trim actually gives up.

    Args:
        region (str | None): Region tag. ``None`` returns ``None``: a Latin face
            that could not be fetched is a different problem, handled by the
            system fallback.
        exclude (str): A file name to skip — the one that could not be fetched,
            which may be present but corrupt.

    Returns:
        Path | None: The smallest usable face for that region, or ``None``.
    """
    if not region:
        return None

    from babeldoc.assets.assets import get_cache_file_path

    directory = Path(get_cache_file_path("probe.ttf", "fonts")).parent
    if not directory.is_dir():
        return None

    wanted = {name.format(region) for name in _HAN_SERIF + _HAN_SANS}
    wanted |= set(_SCRIPT.get(region, ()))
    wanted.discard(exclude)

    found = [directory / name for name in sorted(wanted)
             if (directory / name).is_file()]
    if not found:
        return None
    # Smallest first, and a regular weight ahead of a bold one of the same size:
    # this face is standing in for a whole family, and body text is the bulk of
    # what it will draw.
    # 先小后大，同样大小时常规体优先于粗体：这一款要顶替整个字体族，而它将要排的绝大
    # 部分是正文。
    found.sort(key=lambda f: (f.stat().st_size, "-Bold" in f.name))
    return found[0]


def _system_face(region: str | None) -> Path | None:
    """
    中文：在本机系统字体里找一款真能画出该文种的。

    Find a face of the system's own that can actually draw the script.

    Candidates are probed, not assumed: Apple SD Gothic Neo sits in the same
    directory as the Chinese faces and covers Hangul with no Han at all, and a
    Linux box may have any subset of what is listed.

    Args:
        region (str | None): Region tag, or ``None`` for Latin.

    Returns:
        Path | None: The first candidate that exists and covers the script, or
            ``None`` when the system has nothing suitable.
    """
    import pymupdf

    probe = _PROBE_CHAR.get(region or "", "Ag")
    for candidate in _SYSTEM_CANDIDATES:
        path = Path(candidate)
        if not path.is_file():
            continue
        try:
            font = pymupdf.Font(fontfile=str(path))
        except Exception:  # noqa: BLE001
            logger.debug("Could not open the system face at %s", path, exc_info=True)
            continue
        if all(font.has_glyph(ord(ch)) for ch in probe):
            return path
        logger.debug("%s does not cover %r; trying the next", path.name, probe)
    return None


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


def substitute(name: str) -> tuple[Path, dict] | None:
    """
    中文：用系统字体顶替一款取不到的字体，返回路径与元数据。

    Stand a system face in for one that could not be fetched.

    The substitute is copied into the cache under the name that was asked for
    and entered into BabelDOC's metadata table with its own metrics and hash, so
    everything downstream — the mapper, the embedder, the CMap reproducer —
    goes on addressing it by the original name and never learns the difference.

    Args:
        name (str): The font file name BabelDOC could not obtain.

    Returns:
        tuple[Path, dict] | None: The cached path and the metadata entry, or
            ``None`` when the system has no face covering that script either,
            in which case there is nothing to be done and the caller should let
            the original failure stand.
    """
    import pymupdf
    from babeldoc.assets.assets import get_cache_file_path
    from babeldoc.assets.embedding_assets_metadata import (
        EMBEDDING_FONT_METADATA,
        FONT_NAMES,
    )

    region = _region_of_file(name)
    source = _cached_face(region, exclude=name)
    origin = "already-cached"
    if source is None:
        source = _system_face(region)
        origin = "the system's"
    if source is None:
        logger.error("Nothing on this machine can stand in for %s", name)
        return None

    original = EMBEDDING_FONT_METADATA.get(name, {})
    directory = Path(get_cache_file_path("probe.ttf", "fonts")).parent
    try:
        if source.parent == directory:
            # Already where the loader looks. Copying it beside itself under
            # another name would double a 6 MB file to no purpose — this rung
            # of the ladder exists precisely because the file is right here.
            # 它本来就在加载器要找的位置。在旁边再存一份换个名字，等于把一个 6 MB 的
            # 文件白白翻倍——这一级台阶存在的理由恰恰是「文件就在这儿」。
            cached = source
        else:
            cached = directory / f"{_SUBSTITUTE_PREFIX}{source.stem}{source.suffix}"
            cached.parent.mkdir(parents=True, exist_ok=True)
            if not cached.exists() or cached.stat().st_size != source.stat().st_size:
                # Contents only. ``copy2`` also copies the file's flags, and a
                # macOS system font carries the ones System Integrity Protection
                # sets, so reproducing them on the copy fails with EPERM — which
                # would turn "the network is down" into "no font at all".
                # 只复制内容。copy2 连文件标志一起复制，而 macOS 系统字体带着 SIP 设
                # 的那些标志，在副本上重建它们会以 EPERM 失败——那会把「网络不通」变成
                # 「完全没有字体可用」。
                shutil.copyfile(source, cached)

        font = pymupdf.Font(fontfile=str(cached))
        entry = {
            "ascent": int(round(font.ascender * 1000)),
            "descent": int(round(font.descender * 1000)),
            "encoding_length": _CJK_ENCODING_LENGTH if region else _LATIN_ENCODING_LENGTH,
            "file_name": cached.name,
            "font_name": font.name or cached.stem,
            # The style flags describe the *slot*, not the substitute: the
            # mapper picks a face by them, and answering "bold" with a regular
            # file is still better than leaving the bold slot unfillable.
            # 这几个样式标志描述的是「档位」而非替代字体本身：映射器靠它们挑字体，
            # 用一款常规字体去应答「粗体」，总好过让粗体档无人可用。
            "bold": int(original.get("bold", 0)),
            "italic": int(original.get("italic", 0)),
            "serif": int(original.get("serif", 1)),
            "monospace": int(original.get("monospace", 0)),
            "sha3_256": _sha3_256(cached),
            "size": cached.stat().st_size,
        }
    except Exception:  # noqa: BLE001
        logger.error("Could not install a system face for %s", name, exc_info=True)
        return None

    EMBEDDING_FONT_METADATA[name] = entry
    # FONT_NAMES was frozen at import from the shipped table. A face missing
    # from it has its ToUnicode CMap left alone by ``reproduce_cmap``, and the
    # text it draws comes out uncopyable — visible only when someone tries to
    # select it.
    # FONT_NAMES 是 import 时从内置表冻结下来的。不在其中的字体，reproduce_cmap 不
    # 会为它重建 ToUnicode 表，它排出的文字就复制不出来——只有等人去框选时才看得见。
    FONT_NAMES.add(entry["font_name"])

    logger.warning("Could not fetch %s; using %s %s instead",
                   name, origin, entry["font_name"])
    return cached, entry


def install_fallback() -> None:
    """
    中文：让「字体取不到」变成换一款字体，而不是让进程退出。

    Turn an unobtainable font into a substitution rather than an exit.

    BabelDOC's ``get_font_and_metadata`` ends a failed lookup with
    ``logger.critical`` and ``exit(1)``. That is defensible for a command-line
    run and wrong for a long-lived server, where one unreachable asset server
    would take down a process that is holding a queue. This wraps the function
    so the failure becomes :func:`substitute`, and only re-raises when the
    system has no face to offer either.

    Idempotent, so a second call is harmless.
    """
    from babeldoc.assets import assets

    if getattr(assets.get_font_and_metadata, "_localtranslator_guarded", False):
        return

    original = assets.get_font_and_metadata

    def guarded(font_file_name: str):
        """中文：取字体失败时改用系统字体。 / Substitute on a failed font fetch."""
        try:
            return original(font_file_name)
        except SystemExit:
            # exit() raises SystemExit, which is catchable; the process only
            # ends because nothing normally catches it.
            # exit() 抛的是 SystemExit，是可以捕获的；进程之所以结束，只是因为通常
            # 没人去捕获它。
            logger.warning("BabelDOC gave up on %s", font_file_name)
        except Exception:  # noqa: BLE001
            logger.warning("Fetching %s failed", font_file_name, exc_info=True)

        replacement = substitute(font_file_name)
        if replacement is None:
            raise RuntimeError(
                f"字体 {font_file_name} 既取不到，本机系统里也没有能顶替它的字体。"
                "请检查网络，或在设置里换一个目标语言。"
            )
        return replacement

    guarded._localtranslator_guarded = True
    assets.get_font_and_metadata = guarded
