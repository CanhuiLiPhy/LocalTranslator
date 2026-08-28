"""
中文：术语库清洗——先用确定性规则，规则判不了的再交给 API 辅助判别。

Clean the cumulative glossary, rules first and a judgement only where needed.

**What was actually wrong.** Measured over 1,544 paragraphs taken from real
papers, the glossary injected ``for=循环`` into **20%** of them and ``are=公亩``
into **6.9%**. Both come from a source term that is an ordinary English function
word: ``for`` had been recorded from a programming context, ``are`` is the
Wikidata entry for the unit of area. Neither can ever be the right reading of
the word in running prose, and the model is told about them on every fifth
paragraph.

**What was not wrong.** The same measurement says the per-paragraph injection
budget is nearly never the binding constraint: the median paragraph matches 4
terms against a cap of 24, and only 0.8% of paragraphs exceed the cap. So the
usual instinct — prune hard to make room — buys nothing here. A generic but
*correct* entry such as ``Model=模型`` costs one slot out of twenty-four and
does no harm. The thing worth removing is an entry that is **wrong**, not one
that is merely ordinary.

That is why this module is a scalpel:

* **Rules** remove what can be decided mechanically — function words, sources
  with no letters at all, and the everyday short words the Wikidata seed
  attached obscure units and currencies to.
* **Exposure ranking** measures how often each term actually matches a corpus of
  real papers, so an expensive judgement can be spent on the terms that are
  injected often rather than on all fourteen thousand.
* **Judgement** is then asked only about that shortlist, and only when the
  caller opts in.

Nothing is ever deleted. Entries are disabled, which the settings page shows and
which can be undone by setting ``enabled`` back to 1.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

# Prepositions, conjunctions, articles, pronouns and auxiliaries: the classes
# where no member can ever be a technical term, whatever the target says. A
# closed class does not gain members, so this cannot go stale.
#
# Determiners, quantifiers and degree adverbs — ``same``, ``every``, ``several``,
# ``very`` — are deliberately **not** here. Their translations are usually
# correct (``Same=相同``), and since the per-paragraph budget is not the binding
# constraint, a correct-but-generic entry is not worth removing on a rule. Those
# go to the judgement pass instead, if anywhere.
# 介词、连词、冠词、代词与助动词：无论目标词写的是什么，这些词类的成员都不可能是
# 术语。封闭词类不会新增成员，因此这份清单不会过时。
#
# 限定词、量词与程度副词（same、every、several、very）刻意不在其中。它们的译法
# 通常是对的（Same=相同），而每段注入预算又不是瓶颈，因此不值得用规则删掉一条正确
# 条目。这类词若要处理，交给判别环节。
_FUNCTION_WORDS = frozenset("""
a an the and or but nor for so yet
of in on at by to from with without within into onto upon over under above below
between among through during before after since until while across against along
around behind beside besides beyond despite except inside outside per via versus
toward towards throughout underneath whether about
is are was were be been being am do does did done has have had having
this that these those it its they them their there here we our us you your
he she his her him who whom whose what when where why how which
not no can could may might must shall should will would
""".split())

# The shortest run of letters that can carry a technical meaning of its own once
# the source is lower case. Longer everyday words are left alone: the collisions
# the Wikidata seed produced are all very short.
# 源词为小写时，能独立承载专业含义的最短字母数。更长的日常词不予处理：Wikidata
# 种子造成的冲突全都非常短。
_SHORT = 4

# Any letter, in any script. A source with none of these is a year or a symbol.
# 任意文种的字母。源词里一个都没有，说明它是年份或符号。
_LETTER = re.compile(r"[^\W\d_]", re.UNICODE)

# Where macOS keeps its English word list. Absent on other systems, in which
# case the rule that needs it simply does not fire.
# macOS 的英语词表位置。其它系统上没有，需要它的规则便不会生效。
_WORDS = Path("/usr/share/dict/words")

# A frequency-ordered list of ordinary English, shipped beside this module. Rank
# is what the rule below needs and a dictionary cannot give: /usr/share/dict
# holds 236,000 words with no notion of which are common.
# 与本模块一同发布的英语常用词频表。下面的规则需要的是词频排名，而词典给不了：
# /usr/share/dict 收了 23.6 万个词，却不区分哪些常用。
_FREQUENCY = Path(__file__).resolve().parent / "common-english.txt"

# How far down that list a single word still counts as ordinary. A term is
# domain-specific vocabulary by definition, and nothing in the three thousand
# commonest English words is domain-specific — a translator does not need to be
# told that "data" is 数据. Measured on the glossary this was written against,
# the cutoff catches 289 entries; raising it to 10,000 catches 617 and starts
# reaching words like "lattice" and "kernel" that a paper does use as terms.
# 单词排在该名次以内仍算普通词。术语按定义是领域特有词汇，而英语最常用的三千词里
# 没有任何领域特有词——不需要告诉翻译模型 data 是「数据」。在本次清洗的术语库上，
# 该阈值命中 289 条；放宽到 10000 会命中 617 条，并开始波及 lattice、kernel 这类
# 论文里确实当术语用的词。
_ORDINARY_RANK = 3000

# Sources whose entries a person chose by hand, for this subject. The rule above
# asks "is this word ordinary English?", which is the wrong question about a term
# a terminologist put in an AI glossary on purpose: 「Agent=智能体」 and
# 「Cell=单元」 are exactly the renderings a translation model gets wrong on its
# own, and both source words are ordinary. Automatic imports get no such benefit
# of the doubt.
# 由人工为本领域挑选的来源。上面的规则问的是「这是不是普通英语词」，而对一个术语学家
# 特意收进 AI 术语表的词来说，这个问题问错了：Agent=智能体、Cell=单元 正是翻译模型自己
# 会译错的地方，而这两个源词都很普通。自动导入的条目不享受这种推定。
_CURATED = frozenset({"jiqizhixin", "builtin", "manual"})

# Longest a term may be, in words. Real terminology is short — "equivariant
# message passing neural networks" is five and is about as long as they get.
# What sits above this is the shape automatic extraction pulls out of a
# bibliography: "An image is worth 16x16 words: Transformers for image
# recognition at scale", "Python materials genomics (pymatgen): A robust,
# open-source python library for materials analysis".
# 术语最长几个词。真正的术语都短——equivariant message passing neural networks
# 五个词，已接近上限。超过这个长度的是自动抽取从参考文献里捞出来的形态：
# 「An image is worth 16x16 words: Transformers for image recognition at scale」。
_MAX_WORDS = 5

# A colon splits a paper's title from its subtitle. Combined with any real
# length it is the single most reliable sign of a citation rather than a term.
# 冒号把论文标题与副标题分开。配上一点长度，它就是「这是一条文献而非术语」最可靠的
# 单一信号。
_TITLE_COLON_WORDS = 4

# Words that name a venue, a publisher or an institution. None of these can be
# part of a term; all of them are what a reference list is made of.
# 表示会议、期刊、出版方或机构的词。它们不可能是术语的一部分，却正是参考文献的构成。
# Conclusive on their own: no term contains any of these, at any length.
# 单独出现即可定案：任何长度的术语都不会包含这些词。
_CITATION = frozenset({"preprint", "proceedings", "arxiv", "doi", "isbn", "issn"})

# Need some length beside them, because each can be part of a real term.
# 需要配合长度，因为它们都可能是某个真术语的一部分。
_VENUE = frozenset({
    "journal", "conference", "workshop", "symposium", "transactions",
    "advances", "university", "institute", "institut", "laboratory",
    "department", "press", "edition", "volume", "society", "summit",
    "college", "school", "academy", "academie", "universite", "universität",
    "polytechnique", "politecnico", "foundation", "association", "committee",
    "council", "ministry", "programme", "consortium", "initiative", "centre",
    "fellowship", "scholarship", "grant", "award", "prize", "chair",
})

# A finite verb makes a clause, and a clause is a sentence rather than a term.
# 出现限定动词就构成了小句，而小句是句子，不是术语。
_FINITE_VERB = frozenset({
    "is", "are", "was", "were", "be", "been", "can", "could", "will", "would",
    "may", "might", "must", "should", "does", "do", "did", "has", "have", "had",
    "makes", "make", "shows", "show", "enables", "enable", "allows", "allow",
    "induces", "induce", "requires", "require", "uses", "use", "gives", "give",
    "improves", "improve", "learns", "learn", "predicts", "predict",
})

# A section number opening the string: "B.2.4. Performance…", "1 Introduction",
# "3.2 Scalability". Written so it cannot claim "l2 loss" or "H100 peak", which
# are terms whose first token happens to mix a letter and a digit: the number
# must either carry a dot or be followed by a capitalised word.
# 串首的章节编号："B.2.4. Performance…"、"1 Introduction"、"3.2 Scalability"。
# 写法上刻意认不上 "l2 loss"、"H100 peak" 这类首词恰好字母数字混排的真术语：
# 编号要么自带小数点，要么其后跟一个首字母大写的词。
# A pointer into the document rather than a term: "Figure 1", "Table 3",
# "Section 4.2", "Algorithm 1". The translation is mechanical, the model gets it
# right unaided, and pinning it invites the pairing to fire mid-sentence.
# 这是指向文档某处的指针，不是术语："Figure 1"、"Table 3"、"Section 4.2"。译法是机械的，
# 模型自己就能译对，而把它钉死反而可能在句中误触。
_CROSS_REFERENCE = re.compile(
    r"^(?:figure|fig|table|tab|eq|equation|section|sec|appendix|alg|algorithm|"
    r"chapter|chap|part|step|listing|scheme)\.?\s*[A-Z]?\d",
    re.IGNORECASE,
)

# A person's name: capitalised words, optionally hyphenated or with a middle
# initial, two to four of them. Only ever consulted for entries the extractor
# left untranslated, which is what a name looks like after extraction — 623 of
# them on the glossary this was written against, all harvested from reference
# author lists.
# 人名：两到四个首字母大写的词，可带连字符或中间名缩写。只对「抽取器没有翻译」的条目
# 使用该判据，而那正是人名被抽出来后的样子——本次清洗的术语库里有 623 条，全部来自
# 参考文献的作者列表。
_PERSON = re.compile(
    r"^[A-Z][a-z]+(?:[-'][A-Z][a-z]+)*"
    r"(?: (?:[A-Z]\.|[A-Z][a-z]+(?:[-'][A-Z][a-z]+)*)){1,3}$"
)

_SECTION_NUMBER = re.compile(
    r"^\(?(?:[A-Z]\.)?\d{1,3}(?:\.\d{1,3})+\.?\)?\s+\S"
    r"|^\(?(?:[A-Z]\.)?\d{1,3}\.?\)?\s+[A-Z][A-Za-z]"
)

# Smallest number of words before a venue keyword condemns a term. "autonomous
# laboratory" is a term in materials science; "Laboratory of Artificial Chemical
# Intelligence" is an address.
# 出现会议/机构关键词时，至少要有几个词才判为文献。autonomous laboratory 是材料学的
# 术语；Laboratory of Artificial Chemical Intelligence 是一个地址。
_VENUE_WORDS = 3


def _english_words() -> frozenset[str]:
    """
    中文：读取系统英语词表，读不到就返回空集合。

    Load the system English word list.

    Returns:
        frozenset[str]: Lower-cased words, or an empty set when the file is not
            present, which disables the rule that consults it rather than
            guessing.
    """
    try:
        return frozenset(
            line.strip().lower() for line in _WORDS.read_text(encoding="utf-8", errors="ignore").splitlines()
            if line.strip()
        )
    except OSError:
        logger.debug("No system word list at %s; skipping the collision rule", _WORDS)
        return frozenset()


def _frequency_rank() -> dict[str, int]:
    """
    中文：读取常用词频表，返回「词 -> 名次」。

    Load the frequency-ordered English word list.

    Returns:
        dict[str, int]: Lower-cased word to its rank, 0 being the commonest.
            Empty when the list is not installed, which disables the rule that
            consults it rather than guessing at what is ordinary.
    """
    try:
        words = _FREQUENCY.read_text(encoding="utf-8", errors="ignore").split()
    except OSError:
        logger.debug("No frequency list at %s; skipping the ordinary-word rule",
                     _FREQUENCY)
        return {}
    return {word.lower(): index for index, word in enumerate(words)}


def _is_ordinary_word(term: dict, ranks: dict[str, int]) -> bool:
    """
    中文：源词是一个常用英语单词，不是术语。

    The source is one ordinary English word rather than a term.

    ``for`` → 循环 was the case that made this necessary — it reached a fifth of
    every paragraph — but it is not alone: ``gallery`` → 画廊, ``price`` → 价格,
    ``video`` → 视频 and ``family`` → 家族 were all recorded as terminology.

    All-capitals sources are exempt. ``ADD``, ``ALL`` and ``AS`` are acronyms
    that happen to spell common words, and the capitalisation is what says so.
    So are the hand-curated sources, for the reason given at :data:`_CURATED`.

    Args:
        term (dict): Row from ``glossary_terms``.
        ranks (dict[str, int]): Word to frequency rank.

    Returns:
        bool: ``True`` when the source is a single word inside
            :data:`_ORDINARY_RANK` of the frequency list.
    """
    if not ranks or term["source"] in _CURATED:
        return False
    src = term["src"].strip()
    if " " in src or src.isupper():
        return False
    return ranks.get(src.lower(), 1 << 30) < _ORDINARY_RANK


def _is_title_or_sentence(term: dict) -> bool:
    """
    中文：源词其实是一条文献标题、一个章节标题或一句话，不是术语。

    The source is a citation, a heading or a sentence rather than a term.

    Automatic extraction reads the whole document, bibliography and headings
    included, and hands back whatever looked salient. What comes out is a paper
    title, a venue, an institution or a clause — none of which a translator
    needs pinned, and all of which crowd out the terms that do.

    Args:
        term (dict): Row from ``glossary_terms``.

    Returns:
        bool: ``True`` when the source has the shape of a citation, a numbered
            heading or a sentence.
    """
    if term["source"] in _CURATED:
        return False
    src = term["src"].strip()
    words = src.split()
    if len(words) > _MAX_WORDS:
        return True
    if ":" in src and len(words) >= _TITLE_COLON_WORDS:
        return True
    if _SECTION_NUMBER.match(src) or _CROSS_REFERENCE.match(src):
        return True
    lowered = {w.strip(",.;:()").lower() for w in words}
    if lowered & _CITATION:
        return True
    if len(words) >= _VENUE_WORDS and lowered & _VENUE:
        return True
    return bool(len(words) >= 3 and lowered & _FINITE_VERB)


def _is_function_word(term: dict) -> bool:
    """
    中文：源词是英语虚词。

    The source is an English function word.

    Args:
        term (dict): Row from ``glossary_terms``.

    Returns:
        bool: ``True`` when the source, lower-cased and stripped, is closed-class.
    """
    return term["src"].strip().lower() in _FUNCTION_WORDS


def _has_no_letters(term: dict) -> bool:
    """
    中文：源词里没有任何字母（年份、符号）。

    The source contains no letter at all.

    Args:
        term (dict): Row from ``glossary_terms``.

    Returns:
        bool: ``True`` for a year such as ``2025`` or a bare symbol.
    """
    return _LETTER.search(term["src"]) is None


def _collides_with_everyday_word(term: dict, words: frozenset[str]) -> bool:
    """
    中文：Wikidata 种子把冷僻单位或货币挂到了一个日常英文短词上。

    A Wikidata entry attached an obscure unit or currency to an everyday word.

    ``are`` → 公亩, ``as`` → 阿斯 (硬币), ``mu`` → 亩, ``won`` → 韩圆,
    ``som`` → 索姆 (货币单位), ``chi`` → 尺. Each is a real Wikidata item and each
    is the wrong reading of the word in a scientific paper written in English.

    Args:
        term (dict): Row from ``glossary_terms``.
        words (frozenset[str]): The system English word list.

    Returns:
        bool: ``True`` when the source is a short all-lower-case English word
            that came from the Wikidata seed.
    """
    if not words or term["source"] != "wikidata":
        return False
    src = term["src"].strip()
    return (
        src.isalpha()
        and src.islower()
        and len(src) <= _SHORT
        and src in words
    )


def shortlist(terms: list[dict]) -> list[dict]:
    """
    中文：挑出规则判不了、但值得送去判别的可疑条目。

    Collect the entries a rule cannot settle but a judgement should look at.

    These are the Wikidata seed's short everyday words. Some carry a unit a
    paper really does use — ``mole`` → 摩尔, ``gray`` → 戈瑞, ``mass`` → 质量 —
    and some carry one it never will: ``are`` → 公亩, ``won`` → 韩圆,
    ``som`` → 索姆 (货币单位), ``chi`` → 尺. Nothing in the row separates the two,
    so the pattern is used to bound what gets asked rather than to decide.

    Args:
        terms (list[dict]): Rows from ``glossary_terms``.

    Returns:
        list[dict]: The suspicious entries, in input order.
    """
    words = _english_words()
    return [t for t in terms if _collides_with_everyday_word(t, words)]


def people(terms: list[dict]) -> list[dict]:
    """
    中文：挑出疑似人名的条目，交给判别确认。

    Collect the entries that look like a person's name.

    Automatic extraction reads the bibliography along with everything else and
    hands back author names, which it leaves untranslated because a name should
    not be translated. That is exactly right about the name and exactly wrong
    about it being terminology.

    Shape alone is not proof — "Gram Newton-Schulz" has it too — so this bounds
    what gets asked rather than deciding.

    Args:
        terms (list[dict]): Rows from ``glossary_terms``.

    Returns:
        list[dict]: The candidates, in input order.
    """
    return [
        term for term in terms
        if term["source"] not in _CURATED
        and term["src"].strip() == (term["tgt"] or "").strip()
        and _PERSON.match(term["src"].strip())
    ]


def apply_rules(terms: list[dict]) -> dict[str, list[dict]]:
    """
    中文：按确定性规则给术语分类，返回「规则名 -> 命中的术语」。

    Sort terms into the rules that condemn them.

    Args:
        terms (list[dict]): Rows from ``glossary_terms``, each carrying at least
            ``id``, ``src``, ``tgt``, ``policy``, ``source``.

    Returns:
        dict[str, list[dict]]: Rule name to the terms it caught. A term appears
            under the first rule that matches it, so the counts add up.
    """
    ranks = _frequency_rank()
    checks = (
        ("function word", _is_function_word),
        ("no letters", _has_no_letters),
        ("ordinary English word", lambda t: _is_ordinary_word(t, ranks)),
        ("a title, a heading or a sentence", _is_title_or_sentence),
    )
    caught: dict[str, list[dict]] = {name: [] for name, _ in checks}
    for term in terms:
        for name, check in checks:
            if check(term):
                caught[name].append(term)
                break
    return caught


def is_junk(term: dict) -> bool:
    """
    中文：单条术语是否可由确定性规则判为垃圾（用于自动提取的入库过滤）。

    Decide whether one term is condemned by the deterministic rules.

    The same rules :func:`apply_rules` uses, plus the name, venue and bare-token
    rules, applied to a single entry. Automatic extraction runs after every
    document and its output goes straight into the glossary, so without this the
    glossary refills with what the last cleaning removed: five test runs added
    2,087 entries, none with a subject, among them 「compute → 计算」.

    Args:
        term (dict): An entry with at least ``src``, ``tgt``, ``policy`` and
            ``source``.

    Returns:
        bool: ``True`` when the entry should not enter the glossary.
    """
    from local_translator.glossary.classify import is_bare_token, is_person, is_venue

    source = term.get("src") or ""
    if is_venue(source) or is_bare_token(source):
        return True
    if is_person(source) and (term.get("tgt") or "") in ("", source):
        return True

    ranks = _frequency_rank()
    return bool(
        _is_function_word(term)
        or _has_no_letters(term)
        or _is_ordinary_word(term, ranks)
        or _is_title_or_sentence(term)
    )


def exposure(terms: list[dict], paragraphs: list[str], *, max_terms: int = 10 ** 6):
    """
    中文：量出每条术语在真实语料里的命中段落数，用于决定判别预算花在哪。

    Measure how many paragraphs each term matches.

    Exposure is what makes a wrong entry expensive: an entry nothing matches is
    clutter, while ``for=循环`` reaches a fifth of everything the model is asked
    to translate.

    Args:
        terms (list[dict]): Rows from ``glossary_terms``.
        paragraphs (list[str]): Source-language paragraphs from real documents.
        max_terms (int): Cap passed to the matcher; the default lifts the
            per-paragraph injection limit so the measurement sees every match.

    Returns:
        dict[str, int]: Source term to the number of paragraphs it matched.
    """
    from collections import Counter

    from local_translator.glossary.matcher import GlossaryMatcher

    matcher = GlossaryMatcher(
        [(t["src"], t["tgt"], t["policy"]) for t in terms], max_terms=max_terms
    )
    hits: Counter[str] = Counter()
    for paragraph in paragraphs:
        pairs, keeps = matcher(paragraph)
        for src, _ in pairs:
            hits[src] += 1
        for src in keeps:
            hits[src] += 1
    return dict(hits)


def paragraphs_from(directory: Path, *, low: int = 60, high: int = 3000) -> list[str]:
    """
    中文：从一个目录里的 PDF 抽出正文段落，作为衡量命中率的语料。

    Collect body paragraphs from the PDFs in a directory.

    Args:
        directory (Path): Folder of source-language PDFs. Files whose name
            contains ``.zh.`` are skipped, since those are this tool's own
            output rather than input.
        low (int): Shortest paragraph to keep, in characters. Below this a block
            is a heading, a page number or a caption fragment.
        high (int): Longest paragraph to keep. Above this a block is a whole
            page that failed to split.

    Returns:
        list[str]: One string per paragraph.
    """
    import pymupdf

    out: list[str] = []
    for pdf in sorted(directory.glob("*.pdf")):
        if ".zh." in pdf.name:
            continue
        try:
            with pymupdf.open(pdf) as doc:
                for page in doc:
                    for block in page.get_text("blocks"):
                        text = re.sub(r"\s+", " ", block[4]).strip()
                        if low <= len(text) <= high:
                            out.append(text)
        except Exception:  # noqa: BLE001
            logger.warning("Could not read %s", pdf, exc_info=True)
    return out


# How many entries go into one judgement request. Small enough that the answer
# stays aligned with the question, large enough that eighty entries cost two
# calls rather than eighty.
# 一次判别请求里放多少条。足够小以保证答案与问题对齐，足够大以使八十条只花两次调用
# 而不是八十次。
_BATCH = 40

_PEOPLE_SYSTEM = (
    "You audit a glossary built for translating scientific papers. Each entry "
    "below was extracted from a paper and left untranslated. Decide whether it "
    "is a **person's name**.\n"
    "Answer 'drop' when it names a person — an author, a researcher, anyone.\n"
    "Answer 'keep' when it names something else that happens to be capitalised: "
    "a method, an algorithm, a dataset, a model, a laboratory, a chemical, a "
    "place, an organisation.\n"
    "Reply with one line per entry, in the given order, as 'N. keep' or "
    "'N. drop'. No other text."
)

_USEFUL_SYSTEM = (
    "You audit a glossary used when translating English scientific papers "
    "into Chinese. Papers in physics, chemistry, materials, biology, medicine, "
    "mathematics and computer science.\n"
    "For each entry decide whether pinning this translation would ever help "
    "such a paper.\n"
    "Answer 'keep' for terminology a paper in those fields would use: "
    "quantities, units, reactions, algorithms, theorems, anatomy, diseases, "
    "materials, methods.\n"
    "Answer 'drop' for anything else, however correct the translation is: "
    "named events, crimes, spacecraft, censuses, festivals, currencies, "
    "administrative divisions, foods, games, works of art, people, places, "
    "companies, calendar terms.\n"
    "When in doubt, drop: a wrong or irrelevant entry costs more than a "
    "missing one.\n"
    "Reply with one line per entry, in the given order, as 'N. keep' or "
    "'N. drop'. No other text."
)

_MISTRANSLATED_SYSTEM = (
    "You audit a glossary used when translating English scientific papers into "
    "Chinese. The glossary exists for one purpose: to stop a competent "
    "translation model from getting a term wrong. Every entry it holds is "
    "injected into the prompt, so an entry that teaches the model nothing is "
    "pure cost.\n"
    "For each entry, ask: would a competent English-to-Chinese translation "
    "model, given this term in a scientific paper and no glossary, plausibly "
    "get it WRONG or render it inconsistently?\n"
    "Answer 'keep' only when the answer is yes. Typically:\n"
    "- a common word with a fixed technical sense that differs from its "
    "everyday one (agent=智能体 not 代理; kernel=核 not 内核);\n"
    "- an acronym or proper name that must stay in English or has one "
    "established Chinese form;\n"
    "- a term with several competing Chinese renderings, where consistency "
    "across a document matters.\n"
    "Answer 'drop' when the model would get it right unaided. Typically:\n"
    "- a compositional phrase that is the sum of its parts "
    "(evolution of sexual reproduction = 有性生殖的演化);\n"
    "- a transliterated proper noun (Austri, Vestri = 奥斯特里、威斯特里);\n"
    "- ordinary vocabulary, place names, everyday objects;\n"
    "- anything outside physics, chemistry, materials, biology, medicine, "
    "mathematics and computer science.\n"
    "When in doubt, drop: a needless entry crowds out a useful one.\n"
    "Reply with one line per entry, in the given order, as 'N. keep' or "
    "'N. drop'. No other text."
)

_JUDGE_SYSTEM = (
    "You audit a bilingual glossary used when translating English scientific "
    "papers into Chinese. For each entry decide whether the Chinese target is a "
    "sense the English source plausibly carries in a scientific paper.\n"
    "Answer 'keep' when it is — including domain units and symbols a paper may "
    "genuinely use.\n"
    "Answer 'drop' when the target is an unrelated sense: a currency, a "
    "traditional measure, a place, a colour name or an everyday meaning that no "
    "paper would intend.\n"
    "Reply with one line per entry, in the given order, as 'N. keep' or "
    "'N. drop'. No other text."
)

_VERDICT = re.compile(r"^\s*(\d+)\s*[.、:)]\s*(keep|drop)\b", re.IGNORECASE | re.MULTILINE)


def judge(terms: list[dict], complete, system: str | None = None) -> dict[int, str]:
    """
    中文：请 API 判别一批规则拿不准的条目，返回「术语 id -> keep/drop」。

    Ask a model to settle the entries no rule can.

    Only ever used on :func:`shortlist`, so the cost is bounded by that list
    rather than by the size of the glossary. An entry the model does not answer
    for is simply absent from the result and is therefore left enabled — the
    default is always to keep.

    Args:
        terms (list[dict]): Entries to judge, each with ``id``, ``src``, ``tgt``.
        complete (Callable[[str, str], str]): Sends ``(system, user)`` to the
            model and returns its reply. Supplied by the caller so this module
            does not reach for credentials itself.
        system (str | None): Which question to ask. Defaults to the sense check
            used for the short-word shortlist; :data:`_PEOPLE_SYSTEM` and
            :data:`_USEFUL_SYSTEM` ask the other two.

    Returns:
        dict[int, str]: Term id to ``"keep"`` or ``"drop"``, for the entries the
            model answered about.
    """
    verdicts: dict[int, str] = {}
    for start in range(0, len(terms), _BATCH):
        batch = terms[start:start + _BATCH]
        listing = "\n".join(
            f"{index + 1}. {term['src']} = {term['tgt']}"
            for index, term in enumerate(batch)
        )
        try:
            reply = complete(system or _JUDGE_SYSTEM, listing)
        except Exception:  # noqa: BLE001
            logger.warning("Judgement request failed; leaving %d entries enabled",
                           len(batch), exc_info=True)
            continue
        for match in _VERDICT.finditer(reply or ""):
            index = int(match.group(1)) - 1
            if 0 <= index < len(batch):
                verdicts[batch[index]["id"]] = match.group(2).lower()
    return verdicts
