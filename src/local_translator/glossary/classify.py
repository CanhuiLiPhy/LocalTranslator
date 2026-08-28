"""
中文：术语库的学科归类与「翻译/保持英文」判定。

Sorting the glossary into subjects, and deciding what stays in English.

Two problems this module exists for, both visible in the management page.

**The subjects were wrong.** The Wikidata seed derives a domain by walking the
item's ontology, and the walk lands far from where a reader would put the term:
「Brook rearrangement」 (an organic reaction), 「fibrinolysis」, 「torpor」 and
「Seven Military Classics」 were all filed under *physics*; 「Lindblad equation」
and 「optical equivalence theorem」 under *mathematics*. A quarter of the live
glossary carried no subject at all.

**The keep-in-English list was mostly junk.** Every entry in it came from
automatic extraction and none carried a subject. Among them: author names
(「Théophane Vallaeys」, 「Schiele, B.」), journal abbreviations
(「J. Comput. Phys.」), and bare tokens (「ps16」, 「VGAS」). A whitelist entry
earns its place only if a translator would otherwise get the term wrong, and
nothing translates 「ps16」.

Rules do the part that is decidable — a name is a name — and a model does the
part that needs knowing what a term *is*. The split matters because the model
pass costs an API call per batch, so it is worth spending only on entries no
rule can settle.
"""

from __future__ import annotations

import logging
import re

from local_translator.glossary.store import DOMAINS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------- #
# Rules / 规则
# ---------------------------------------------------------------------- #

# A personal name as automatic extraction writes it: "Schiele, B.",
# "Théophane Vallaeys", "Yann LeCun". Capitalised words, optionally with
# initials, and nothing that looks like a common noun phrase.
# 自动抽取写出的人名形式："Schiele, B."、"Théophane Vallaeys"、"Yann LeCun"。
# 首字母大写的词，可能带缩写点，且不含普通名词短语的迹象。
_PARTICLE = r"(?:van |von |de |del |della |da |di |le |la |du |den |der |ten |ter )"
_SURNAME_INITIAL = re.compile(
    rf"^(?:{_PARTICLE})*[A-ZÀ-Þ][\w'’\-]+,\s*(?:[A-Z]\.\s*){{1,3}}$")
_GIVEN_SURNAME = re.compile(
    r"^(?:[A-ZÀ-Þ][\w'’\-]{1,}\.?\s+){1,2}"          # given name(s) or initials
    r"(?:van |von |de |del |da |di |le |la |Mc|Mac|O')?"
    r"[A-ZÀ-Þ][\w'’\-]{1,}$"
)

# Words this short are not evidence either way — "He" is both a surname and the
# commonest English pronoun, so testing it would reject 「Kaiming He」.
# 这么短的词两边都不构成证据——"He" 既是姓氏又是最常见的英文代词，拿它去判会把
# 「Kaiming He」误判成非人名。
_TELLING_LENGTH = 3

# Being in the common-word list at all rules a name out. The list is the ten
# thousand commonest English words; a surname is not in it, and "Transform"
# (rank 7649) is — so a rank threshold would have to sit near the end of the
# list anyway. Erring this way costs a missed name, which leaves an entry in the
# glossary; erring the other way deletes a real term.
# 只要出现在常用词表里就排除人名。该表收录英语最常用的一万词，姓氏不在其中，
# 而 "Transform"（第 7649 位）在——所以按名次设阈值本来也得设到表尾。往这个方向出错
# 只是漏掉一个人名、条目留下；往反方向出错则会删掉真术语。

# Journal and conference abbreviations: "J. Comput. Phys.", "Phys. Rev. B",
# "IEEE Trans. Pattern Anal.". The giveaway is abbreviated words with stops.
# 期刊与会议缩写："J. Comput. Phys."、"Phys. Rev. B"、"IEEE Trans. Pattern Anal."。
# 特征是带句点的缩写词。
_ABBREVIATED_WORD = re.compile(r"\b[A-Z][a-z]{0,7}\.")
_VENUE_HINT = frozenset({
    "j", "journal", "proc", "proceedings", "trans", "transactions", "rev",
    "review", "reviews", "lett", "letters", "ann", "annals", "bull", "adv",
    "advances", "conf", "conference", "symp", "symposium", "workshop", "arxiv",
    "preprint", "acta", "annu", "int", "intl", "soc", "society", "assoc",
})

# Publishers and preprint servers that arrive as bare tokens.
# 以裸标记形式出现的出版社与预印本平台。
_PUBLISHER = frozenset({
    "pmlr", "neurips", "nips", "icml", "iclr", "cvpr", "iccv", "eccv", "acl",
    "emnlp", "naacl", "aaai", "ijcai", "kdd", "sigir", "siggraph", "usenix",
    "acm", "ieee", "springer", "elsevier", "wiley", "biorxiv", "medrxiv",
    "chemrxiv", "openreview", "github", "huggingface",
})

# A token with no internal space that is not a word: "ps16", "VGAS", "THFM",
# "Wan2.2-T2V-14B". Kept apart from acronyms a reader would recognise, which
# the model pass decides on.
# 不含空格且不成词的标记："ps16"、"VGAS"、"THFM"、"Wan2.2-T2V-14B"。与读者认得的
# 缩写区分开，后者交给模型判定。
_HAS_DIGIT = re.compile(r"\d")
_MIN_MEANINGFUL = 2


def is_person(source: str) -> bool:
    """
    中文：判断一条术语是否只是人名。

    Decide whether a term is nothing but a personal name.

    Args:
        source (str): The source term.

    Returns:
        bool: ``True`` for "Schiele, B.", "Théophane Vallaeys" and the like.
    """
    text = source.strip()
    if not text or " " not in text and "," not in text:
        return False
    if _SURNAME_INITIAL.match(text):
        return True
    words = text.split()
    if len(words) > 3 or not _GIVEN_SURNAME.match(text):
        return False
    if any(word.lower() in _VENUE_HINT for word in words):
        return False

    # A capitalised two-word phrase is just as often a real term — "Fourier
    # Transform", "Gibbs Free Energy" — and the pattern cannot tell them apart.
    # What can is vocabulary: a surname is not among the commonest words of
    # English, and "Transform" is. Short words are exempt because "He" is both.
    # 首字母大写的双词短语同样常常是真术语——"Fourier Transform"、"Gibbs Free
    # Energy"——单靠形式分辨不了。能分辨的是词汇：姓氏不会出现在英语最常用词里，
    # 而 "Transform" 会。短词豁免，因为 "He" 两者都是。
    from local_translator.glossary.clean import _frequency_rank

    ranks = _frequency_rank()
    for word in words:
        plain = word.lower().strip(".,'’-")
        if len(plain) < _TELLING_LENGTH:
            continue
        if plain in ranks:
            return False
    return True


def is_venue(source: str) -> bool:
    """
    中文：判断一条术语是否是期刊/会议/出版社名称。

    Decide whether a term names a journal, conference or publisher.

    Args:
        source (str): The source term.

    Returns:
        bool: ``True`` for "J. Comput. Phys.", "PmLR", "IEEE Trans. …".
    """
    text = source.strip()
    if not text:
        return False
    if text.lower().strip(".") in _PUBLISHER:
        return True
    words = [word.lower().strip(".") for word in text.split()]
    if any(word in _VENUE_HINT for word in words):
        # One abbreviated word alongside a venue word settles it; "Journal of
        # Chemical Physics" spelled out does too.
        # 有一个缩写词加上一个刊物词即可判定；完整写出的 "Journal of Chemical
        # Physics" 同样成立。
        return len(words) <= 8
    return len(_ABBREVIATED_WORD.findall(text)) >= 2


def is_bare_token(source: str) -> bool:
    """
    中文：判断一条术语是否只是一个无意义的裸标记（含数字的短标识、随机大写串）。

    Decide whether a term is a meaningless bare token.

    A whitelist entry earns its place only if a translator would otherwise get
    the term wrong. Nothing translates "ps16" or "Wan2.2-T2V-14B", so keeping
    them costs prompt space and buys nothing.

    Args:
        source (str): The source term.

    Returns:
        bool: ``True`` for a single token containing digits, or one shorter than
            two characters.
    """
    text = source.strip()
    if not text or " " in text:
        return False
    if len(text) < _MIN_MEANINGFUL:
        return True
    return bool(_HAS_DIGIT.search(text))


def junk(terms: list[dict]) -> list[dict]:
    """
    中文：挑出可由规则判定为垃圾的条目（人名、刊物名、裸标记）。

    Select the entries the rules can condemn without asking a model.

    Args:
        terms (list[dict]): Entries with at least ``id`` and ``src``.

    Returns:
        list[dict]: The subset that is a name, a venue or a bare token.
    """
    caught = []
    for term in terms:
        source = term["src"]
        target = term.get("tgt") or ""
        if is_venue(source) or is_bare_token(source):
            caught.append(term)
            continue
        # The name rule is the one that can misfire on a real term: an
        # eponymous pair like 「Navier–Stokes」 is name-shaped. It is therefore
        # applied only where there is no Chinese rendering to lose — a
        # keep-in-English entry stores its source as its target. An entry that
        # someone or something has actually rendered into Chinese is left to
        # the model.
        # 人名规则是唯一可能误伤真术语的：像「Navier–Stokes」这种以人名命名的词组
        # 形状上就是人名。因此只在没有中文译法可失去时才应用它——保持英文的条目其
        # 译文字段存的就是原文。已经有人（或有程序）给出过中文译法的条目交给模型。
        if is_person(source) and target in ("", source):
            caught.append(term)
    return caught


# ---------------------------------------------------------------------- #
# Model pass / 模型判定
# ---------------------------------------------------------------------- #

# Batch size. Larger batches cost fewer calls but make the model's numbering
# drift; forty is where the replies stayed aligned in testing.
# 批量大小。批越大调用越少，但模型的编号越容易错位；实测四十条时回复仍能对齐。
_BATCH = 40

_DOMAIN_LIST = ", ".join(DOMAINS)

# A rendering is only accepted if it actually contains Chinese; the model
# sometimes echoes the English instead of the dash.
# 只有真的含中文才当作译法采纳；模型有时会回显英文而不是那个短横。
_CJK = re.compile(r"[\u4e00-\u9fff]")

_CLASSIFY_SYSTEM = (
    "You sort scientific glossary entries for a Chinese-language reading tool.\n"
    "For each numbered entry, reply with one line:\n"
    "  <number>. <subject> <policy>\n"
    f"<subject> is exactly one of: {_DOMAIN_LIST}\n"
    "<policy> is exactly one of: translate, keep, drop\n\n"
    "Choose the subject a working researcher would file the term under, not the "
    "broadest one that technically contains it. A named organic reaction is "
    "chemistry, not physics. An equation from quantum optics is physics, not "
    "math. A clinical sign is medicine, not biology. An algorithm is cs unless "
    "it is a theorem or a pure-mathematics construction, which is math. Machine "
    "learning, neural networks and language models are ai, not cs. Use general "
    "only when no subject fits.\n\n"
    "policy=keep means the term must appear in the Chinese text in its original "
    "English form. Be strict. Chinese scientific writing leaves a term in "
    "English only when it is an established acronym or symbol (DNA, GPU, SMILES, "
    "pH), or the proper name of a piece of software, a file format or a "
    "standard. An ordinary English noun phrase is not one of those, however "
    "technical: 「Deep Network」, 「Symbolic Learning」, 「token sequence」 and "
    "「Rosenbrock function」 all have settled Chinese renderings and must be "
    "translated. If a competent Chinese paper would write the idea in Chinese, "
    "choose translate.\n"
    "policy=translate means the term should be rendered in Chinese. Where the "
    "entry's Chinese rendering is missing, or identical to the English, append "
    "the correct Chinese rendering after the policy. Otherwise append a dash.\n"
    "policy=drop means the entry does not belong in a glossary at all: a "
    "person's name, a paper or conference title, a journal abbreviation, a "
    "product or model name, an arbitrary identifier, or an ordinary word that "
    "no translator would get wrong unaided.\n\n"
    "Reply with one line per entry and nothing else. Example:\n"
    "  1. chemistry translate 消融实验\n"
    "  2. ai keep -\n"
    "  3. general drop -"
)

_ANSWER = re.compile(
    r"^\s*(\d+)\s*[.、:)]\s*([a-zA-Z]+)\s+(translate|keep|drop)\b[ \t]*(.*)$",
    re.IGNORECASE | re.MULTILINE,
)


def classify(terms: list[dict], complete, *, batch: int = _BATCH) -> dict[int, tuple[str, str]]:
    """
    中文：请模型给每条术语判定学科与「翻译/保持英文/丢弃」。

    Ask a model for each entry's subject and policy.

    An entry the model does not answer for is simply absent from the result, so
    the caller leaves it exactly as it was — the default is always to change
    nothing.

    Args:
        terms (list[dict]): Entries to classify, each with ``id``, ``src`` and
            ``tgt``.
        complete (Callable[[str, str], str]): Sends ``(system, user)`` to the
            model and returns its reply. Supplied by the caller so this module
            never reaches for credentials itself.
        batch (int): Entries per request.

    Returns:
        dict[int, tuple[str, str, str]]: Term id to
            ``(domain, policy, rendering)``, where ``domain`` is one of
            :data:`~local_translator.glossary.store.DOMAINS`, ``policy`` is
            ``"translate"``, ``"keep"`` or ``"drop"``, and ``rendering`` is a
            Chinese rendering for an entry that had none — empty otherwise.
    """
    out: dict[int, tuple[str, str, str]] = {}
    valid = set(DOMAINS)

    for start in range(0, len(terms), batch):
        chunk = terms[start:start + batch]
        listing = "\n".join(
            f"{index + 1}. {term['src']} = {term['tgt']}"
            for index, term in enumerate(chunk)
        )
        try:
            reply = complete(_CLASSIFY_SYSTEM, listing)
        except Exception:  # noqa: BLE001
            logger.warning("Classification request failed; leaving %d entries as they are",
                           len(chunk), exc_info=True)
            continue

        for match in _ANSWER.finditer(reply or ""):
            index = int(match.group(1)) - 1
            domain = match.group(2).lower()
            policy = match.group(3).lower()
            if not (0 <= index < len(chunk)) or domain not in valid:
                continue
            rendering = (match.group(4) or "").strip().strip("-–—").strip()
            if not _CJK.search(rendering):
                rendering = ""
            out[chunk[index]["id"]] = (domain, policy, rendering)

    return out
