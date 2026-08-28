r"""
中文：给 LaTeX 源码翻译换一套提示词——PDF 那一套会让整段原样退回。

Give source translation a prompt of its own.

The PDF pipeline's prompt comes from BabelDOC, and one clause in it is fatal
here::

    If translation is unnecessary (e.g. proper nouns, codes, {{1}}, etc.),
    return the original text.

A paragraph of LaTeX opens with a command and carries ``{v1}`` placeholders
through the middle of every sentence, so it reads to a model as exactly the
"codes, {{1}}" the prompt names — and the model does what it was told: it
returns the English, unchanged. Measured on the IEEEtran paper, its opening
paragraph came back verbatim on three independent attempts, the only difference
being that ``\IEEEPARstart`` had quietly become ``\IEEEParstart``. The block was
then refused for the renamed command, which made a prompt problem look like a
model problem.

The same clause is what left whole paragraphs in English elsewhere: when the
echo happens to break nothing, the check passes and the English is shipped as
though it were a translation.

**中文：** PDF 管线的提示词里那句「如果不需要翻译就原样返回」，在源码翻译里是致命的。
一段 LaTeX 以命令开头、句子中间又都是 {v1} 占位符，在模型看来正是那句话点名的
「代码、{{1}}」，于是它照做——把英文原样退回来。这里换一套提示词：明说这是散文、一定
要翻，占位符照抄，命令不许改名。
"""

from __future__ import annotations

import logging
import threading

from local_translator.engine.translators import language_name

logger = logging.getLogger(__name__)

# What the next request on this thread has to be reminded about. Thread-local
# because a document is translated by a pool of threads and a reminder meant
# for one block must not reach another.
# 本线程的下一次请求需要被提醒什么。按线程保存，因为一篇文档由线程池并行翻译，给这一块
# 的提醒不能跑到另一块上去。
_state = threading.local()


# Extra instructions for regions that are not running prose. A table is the one
# that matters: its cells arrive separated by placeholders standing for ``&``
# and ``\\``, and a model told nothing about them will happily merge two columns
# into one Chinese phrase.
# 给「不是行文散文」的区域加的额外说明。要紧的是表格：单元格之间隔着代表 & 和 \\ 的
# 占位符，而一个对此一无所知的模型会很乐意把两列并成一个中文短语。
_REGION_NOTES: dict[str, str] = {
    "table": (
        "This passage is a table. The placeholders between cells are column "
        "separators and row endings: keep every one, in the same order, so the "
        "cells stay in their columns. Translate the headings and any words in "
        "the cells; copy every number exactly as it is written."
    ),
    "table_footnote": (
        "This is a note under a table. Copy every number exactly as written."
    ),
    "algorithm": (
        "This passage is pseudocode. Each placeholder is a keyword, a line "
        "break or a block delimiter: keep every one, in the same order, on the "
        "same line it is on now, and never merge two lines into one. Translate "
        "the comments and the English words inside conditions and statements; "
        "leave identifiers, variable names and numbers exactly as they are."
    ),
    "reference": (
        "These are bibliography entries. Translate nothing but the words a "
        "reader reads as prose; copy author names, titles, journal names, "
        "years, volumes and page numbers exactly as they are."
    ),
    "author": (
        "This is an author block. Translate only the words around the names — "
        "roles, departments, the word for 'and' — and copy every personal name, "
        "institution name and email address exactly as written."
    ),
    "affiliation": (
        "This is an affiliation. Translate the institution's description if "
        "there is one, and copy names, addresses and email addresses exactly."
    ),
}


def region_note(label: str) -> str:
    """
    中文：取这个区域要额外交代的话。

    The extra instruction this region needs, if any.

    Args:
        label (str): The region label, as the parser assigned it.

    Returns:
        str: One instruction line, or empty for ordinary prose.
    """
    return _REGION_NOTES.get(label, "")


def about(label: str = "", terminology: str = "") -> None:
    """
    中文：告诉本线程的下一次请求，它翻的是哪种区域、要守哪些术语。

    Tell this thread's next request what region it is translating and which
    terms are fixed.

    Args:
        label (str): The region label, from the parser.
        terminology (str): A terminology line, or empty.

    Returns:
        None
    """
    _state.label = label
    _state.terminology = terminology


def insist(tokens) -> None:
    """
    中文：让本线程的下一次请求把这些记号点名列出来。

    Ask the next request on this thread to name these tokens explicitly.

    Args:
        tokens (Iterable[str]): The placeholders that must come back.

    Returns:
        None
    """
    _state.tokens = list(dict.fromkeys(tokens))[:12]


def _reminder() -> str:
    """
    中文：取出并清空本线程待发的提醒。

    Take this thread's pending reminder, clearing it.

    Returns:
        str: The region note, the terminology line and any dropped-placeholder
            reminder, or empty when there is nothing to add. The reminder is
            cleared on read so it applies to one request; the region and the
            terminology are not, because they belong to the block and a retry
            translates the same block.
    """
    parts = []
    note = region_note(getattr(_state, "label", "") or "")
    if note:
        parts.append(note)
    terminology = getattr(_state, "terminology", "") or ""
    if terminology:
        parts.append(terminology)
    tokens = getattr(_state, "tokens", None)
    _state.tokens = None
    if tokens:
        parts.append("CRITICAL: the previous answer dropped a placeholder. "
                     "These must each appear once in your answer, exactly as "
                     "written: " + ", ".join(tokens) + ".")
    return ("\n" + "\n".join(parts)) if parts else ""

# The rules a translation of LaTeX prose has to follow, written as constraints
# rather than as prohibitions where possible: a model told what to do with a
# placeholder places it correctly far more often than one told not to lose it.
#
# No rule names a real command. An earlier draft illustrated rule 4 with
# ``\IEEEPARstart{A}{ccording}`` — a command that appears in one paper out of
# five — and three paragraphs of the Gemini report came back containing
# ``\IEEEPARstart``, copied out of the instructions. An example in a prompt is
# something the model may reach for; in a prompt about LaTeX, reaching for it
# means writing a command the document has never defined.
# 翻译 LaTeX 散文要守的规矩。能写成「该怎么做」就不写成「不许怎样」：告诉模型占位符该
# 放哪里，比告诉它别弄丢，放对的次数多得多。
#
# 规则里不出现任何真实命令。早先的版本用 \IEEEPARstart{A}{ccording} 举例说明第 4 条——
# 那是五篇论文里只有一篇会用的命令——结果 Gemini 那篇有三段译文里出现了 \IEEEPARstart，
# 是从指令里抄过去的。提示词里的例子是模型可能顺手拿来用的东西；在一段讲 LaTeX 的提示词
# 里，顺手拿来用就意味着写出一个文档从未定义过的命令。
_RULES = (
    "This passage is prose from an academic paper. It always needs "
    "translating: never return it unchanged, and never answer that no "
    "translation is required.",
    "It may be a two-word heading or several paragraphs. Translate exactly "
    "what is there and no more: keep the line and paragraph breaks the source "
    "has, and never add a sentence, a line or a paragraph it does not have.",
    "Tokens like {v1} and {v12} stand for formulas, citations and "
    "cross-references. Reproduce each one exactly once, verbatim, at the point "
    "in the Chinese sentence where it belongs. They are not code to skip over.",
    "Backslash commands are LaTeX. Keep every command name spelled exactly as "
    "it appears — letter for letter, capital for capital — keep its braces, and "
    "translate only the words inside them. Use no command the passage does not "
    "already contain.",
    "Output the translation and nothing else — no explanation, no notes, no "
    "code fence.",
)

# A heading is not a passage, and the rules above read to a model as licence to
# write one. Asked for the two words of ``\subsection{Code generation}`` under
# the full instruction, it returned five numbered paragraphs inventing an
# architecture the paper does not have, and the essay went inside the braces of
# \subsection. A short input gets an instruction that says what a short output
# is.
# 标题不是段落，而上面那几条规矩在模型看来像是允许它写一段。用完整指令去问
# \subsection{Code generation} 那两个词，它回来的是五个编号段落，凭空造出一个论文里
# 没有的架构，而这段文章进了 \subsection 的花括号里。短输入配一条说明短输出该是什么
# 样子的指令。
_SHORT_RULES = (
    "This is a heading or a short phrase from an academic paper, not a "
    "passage. Answer with the heading in {language} and nothing else: a few "
    "words, no sentence, no explanation, no list.",
    # Without this line a model leaves the field's own vocabulary in English:
    # "Instruction Finetuning" and "Preference Tree with Look-Ahead Algorithm"
    # both came back untouched, three attempts running, while the sentences
    # around them translated fine. It reads a technical heading as a name.
    # 没有这一行，模型会把本领域自己的词汇留在英文里：Instruction Finetuning 和
    # Preference Tree with Look-Ahead Algorithm 连问三次都是原样退回，而它们周围的句子
    # 翻得好好的。它把技术性的标题读成了专有名词。
    "The answer must be in {language}. Translate technical terms too; where the "
    "English term is worth keeping, write the {language} and put the English in "
    "parentheses after it.",
    "Tokens like {{v1}} stand for formulas and references. Reproduce each one "
    "verbatim, exactly once.",
    "Keep any backslash command spelled exactly as it appears.",
)

# Below this many characters a block is a heading rather than a passage, unless
# it ends a sentence. Section titles, captions and table headers all sit here.
# 短于这个长度的块是标题而不是段落——除非它以句号收尾。小节标题、图注和表头都在这个范围。
_SHORT = 80

_SENTENCE_END = ("。", ".", "!", "?", "！", "？")

_SYSTEM = "You are a professional, faithful translation engine for academic papers."


def instruction(language: str) -> str:
    """
    中文：拼出这套提示词的正文部分。

    Compose the instruction that precedes the passage.

    Args:
        language (str): Target language name, e.g. ``"Chinese"``.

    Returns:
        str: The instruction line followed by the numbered rules.
    """
    lines = [f"Translate the passage below into {language}."]
    lines += [f"{index}. {rule}" for index, rule in enumerate(_RULES, 1)]
    return "\n".join(lines)


def short_instruction(language: str) -> str:
    """
    中文：拼出给标题用的短指令。

    Compose the instruction used for a heading.

    Args:
        language (str): Target language name.

    Returns:
        str: One line per rule, with the language filled in.
    """
    return "\n".join(rule.format(language=language) for rule in _SHORT_RULES)


def is_heading(text: str) -> bool:
    """
    中文：这段文字是标题还是段落。

    Report whether a block reads as a heading rather than a passage.

    Args:
        text (str): The block about to be sent.

    Returns:
        bool: ``True`` for a short run that does not close a sentence.
    """
    stripped = text.strip()
    return len(stripped) <= _SHORT and not stripped.endswith(_SENTENCE_END)


def _target(translator) -> str:
    """
    中文：问出翻译器的目标语言全名。

    Ask a translator what language it is translating into.

    Args:
        translator: Any of this project's translators.

    Returns:
        str: A language name suitable for a prompt, falling back to
            ``"Chinese"`` when the translator declares neither.
    """
    named = getattr(translator, "target_language", None)
    if named:
        return named
    code = getattr(translator, "lang_out", "") or ""
    return language_name(code) if code else "Chinese"


def use_latex_prompt(translator):
    r"""
    中文：把翻译器的提示词换成 LaTeX 专用的一套，原地生效。

    Replace a translator's prompt with the one written for LaTeX source.

    Args:
        translator: The translator the LaTeX path is about to use. Patched in
            place, because it is built for this run and used by nothing else.

    Returns:
        The same translator, so the call reads as a conversion.

    Note:
        The change is registered as a cache-impact parameter. Without that, a
        block translated under the old prompt is served from cache under the
        new one, and the fix appears not to work — which is exactly what a
        stale cache looks like.
        这次改动会登记为影响缓存的参数。不登记的话，旧提示词下译过的块会在新提示词下
        被缓存命中，修复看起来就像没生效——而那正是缓存过期时的样子。
    """
    language = _target(translator)
    head = instruction(language)
    brief = short_instruction(language)

    def _lead(text: str) -> str:
        """中文：按长短挑一条指令。 / Pick the instruction this block needs."""
        return brief if is_heading(text) else head

    if hasattr(translator, "prompt"):
        def prompt(_self, text):
            """中文：LaTeX 版提示词。 / The LaTeX prompt."""
            # The per-block constraints go in the *system* message, never in
            # the user turn. A translation model reads everything after the
            # instruction line as material to translate, and a line that reads
            # 「Terminology (use exactly these renderings): Optimization=最优化」
            # is prose to it: measured end to end through the application, an
            # ICLR paper came back with its own terminology list printed above
            # the title on page one. The constraint still applies from the
            # system message; it just stops being content.
            # 每一块的附加约束放在 **system** 消息里，绝不放进 user 轮。翻译模型会把指令
            # 行之后的一切都当作要翻译的材料，而「Terminology (use exactly these
            # renderings): Optimization=最优化」这样一行在它看来就是散文：端到端跑完整个
            # 应用实测，一篇 ICLR 论文的第一页标题上方印着它自己的术语表。约束放在 system
            # 里照样生效，只是不再是内容。
            notes = _reminder()
            return [
                {"role": "system", "content": _SYSTEM + notes},
                {"role": "user", "content": f"{_lead(text)}\n\n{text}"},
            ]

        translator.prompt = prompt.__get__(translator, type(translator))

    if hasattr(translator, "build_messages"):
        original = translator.build_messages

        def build_messages(text, *, strict=False, _original=original):
            """中文：在本地翻译器的消息里换上 LaTeX 版指令。 / Swap in the LaTeX instruction."""
            messages = _original(text, strict=strict)
            lead = _lead(text)
            rules = "\n".join(f"- {line}" for line in lead.splitlines()[1:])
            rules += _reminder()
            for message in messages:
                if message["role"] == "system":
                    message["content"] += "\n" + rules
                    break
            else:
                messages.insert(0, {"role": "system",
                                    "content": f"{_SYSTEM}\n{rules}"})
            for message in messages:
                if message["role"] == "user":
                    message["content"] = f"{lead}\n\n{text}"
            return messages

        translator.build_messages = build_messages

    with_params = getattr(translator, "add_cache_impact_parameters", None)
    if with_params:
        with_params("latex_prompt", 3)
    logger.debug("LaTeX 提示词已启用，目标语言 %s", language)
    return translator
