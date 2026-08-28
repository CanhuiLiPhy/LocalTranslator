"""
中文：英汉 / 汉英词典页——离线查词，词库缺词时回退到术语库。

Bilingual dictionary page, English to Chinese and back.

Two sources answer a lookup, in this order. ECDICT covers general English with
about 770,000 entries, phonetics and frequency ranks. The project's own glossary
covers what ECDICT does not: it was seeded for physics, chemistry, materials,
machine learning, computer science and mathematics, so it holds *tokenizer* and
*ablation study* where a general dictionary holds neither.

The order matters. ECDICT gives senses, pronunciation and register; the glossary
gives one preferred rendering and nothing else. Asking the richer source first
and falling back to the domain list gives a full entry when one exists and a
usable answer when it does not.

Direction is inferred from the query rather than chosen with a switch: a query
containing a Chinese character can only be a Chinese one. Looking a Chinese word
up runs against a reverse index built from the same entries (see
:meth:`DictionaryStore.build_reverse_index`), which returns headwords rather
than a single entry — 注意力 is a sense of *attention*, *attentiveness* and
*concentration*, and picking one of them for the reader would be guessing.
"""

from __future__ import annotations

import logging
import re

from nicegui import ui

from local_translator.dictionary import get_dictionary
from local_translator.service import get_service
from local_translator.web.theme import (
    BLUE,
    CARD,
    GREEN,
    HEADING,
    INK_SOFT,
    INPUT_PROPS,
    META,
    MUTED,
    PURPLE,
    SURFACE_ALT,
)

logger = logging.getLogger(__name__)

# Exam and register tags ECDICT stores as bare codes.
# ECDICT 以裸代码存储的考试与语域标签。
TAG_NAMES = {
    "zk": "中考", "gk": "高考", "cet4": "四级", "cet6": "六级",
    "ky": "考研", "toefl": "托福", "ielts": "雅思", "gre": "GRE",
    "sat": "SAT", "bbc": "BBC 常用",
}

POS_NAMES = {
    "n": "名词", "v": "动词", "adj": "形容词", "adv": "副词",
    "prep": "介词", "conj": "连词", "pron": "代词", "int": "感叹词",
    "art": "冠词", "num": "数词", "u": "不可数", "c": "可数",
}

MATCH_NOTES = {
    "lemma": "原形",
    "exchange": "原形",
    "suffix": "推测原形",
}

REVERSE_NOTES = {
    "exact": "释义相符",
    "prefix": "释义以此开头",
    "contains": "释义中含此词",
}

# Any Chinese character means the query is Chinese; no language switch needed.
# 出现汉字即判定为中文查询，无需切换语种。
_CJK = re.compile(r"[\u4e00-\u9fff]")


def _glossary_fallback(word: str, chinese: bool = False) -> list[tuple[str, str]]:
    """
    中文：在术语库中查找该词，返回匹配的中英对照。

    Look a word up in the project glossary.

    Args:
        word (str): Word to find.
        chinese (bool): Match against the Chinese side instead of the English.

    Returns:
        list[tuple[str, str]]: ``(source, target)`` pairs, empty when absent.
    """
    try:
        from local_translator.glossary.store import GlossaryStore

        rows = GlossaryStore(get_service().db).search(query=word, limit=8)
    except Exception:  # noqa: BLE001
        logger.debug("Glossary fallback failed", exc_info=True)
        return []

    # The store searches both columns, so the side that actually matched has to
    # be checked here — otherwise an English query returns rows whose Chinese
    # happens to contain it, and vice versa.
    # 存储层两列都会搜，因此真正命中的是哪一侧要在这里判定，否则英文查询会带回
    # 只是中文里含有该串的条目，反之亦然。
    needle = word.strip().lower()
    field = "tgt" if chinese else "src"
    return [(row["src"], row["tgt"]) for row in rows if needle in row[field].lower()]


def _render_reverse(hits, on_pick) -> None:
    """
    中文：渲染汉英查询的候选词列表。

    Render the candidate headwords of a Chinese lookup.

    Args:
        hits (list[ReverseHit]): Matches, best first.
        on_pick (Callable[[str], None]): Called with a headword when the reader
            selects one, to show its full entry.
    """
    for hit in hits:
        with ui.row().classes(
            "w-full items-baseline gap-3 py-2 border-b border-gray-100 cursor-pointer"
        ).on("click", lambda h=hit: on_pick(h.word)):
            ui.label(hit.word).classes("text-[15px] font-medium").style(f"color: {BLUE}")
            if hit.phonetic:
                ui.label(f"/{hit.phonetic}/").classes("text-[12px]").style(
                    f"color: {INK_SOFT}"
                )
            note = REVERSE_NOTES.get(hit.match)
            if note and hit.match != "exact":
                ui.label(f"{note}：{hit.term}").classes(META).style(f"color: {INK_SOFT}")
            ui.space()
            # The first sense line is the useful preview: it tells the reader
            # whether this headword is the sense they meant before they open it.
            # 第一行释义就是有用的预览：读者据此判断这个词条是否是自己要的义项。
            preview = (hit.translation or "").split("\n")[0].strip()
            if preview:
                ui.label(preview[:48]).classes("text-[13px] truncate max-w-md").style(
                    "color: #37352f"
                )


def _render_entry(entry) -> None:
    """
    中文：渲染一条词典释义。

    Render one dictionary entry.

    Args:
        entry (Entry): Entry to display.
    """
    with ui.row().classes("items-baseline gap-3 flex-wrap"):
        ui.label(entry.word).classes("text-[22px] font-semibold").style("color: #37352f")
        if entry.phonetic:
            ui.label(f"/{entry.phonetic}/").classes("text-[14px]").style(f"color: {INK_SOFT}")
        note = MATCH_NOTES.get(entry.matched_via)
        if note:
            ui.label(note).classes(META).style(f"color: {BLUE}")

    chips = []
    if entry.collins:
        chips.append(("柯林斯 " + "★" * entry.collins, GREEN))
    if entry.oxford:
        chips.append(("牛津核心词", GREEN))
    for tag in (entry.tag or "").split():
        chips.append((TAG_NAMES.get(tag, tag), PURPLE))
    if entry.frq:
        chips.append((f"词频 #{entry.frq}", INK_SOFT))
    if chips:
        with ui.row().classes("gap-2 flex-wrap mt-1"):
            for text, colour in chips:
                ui.label(text).classes("text-[11px] px-2 py-0.5 rounded").style(
                    f"background: {SURFACE_ALT}; color: {colour}"
                )

    if entry.translation:
        ui.separator().classes("my-3")
        for line in entry.translation.split("\n"):
            if line.strip():
                ui.label(line.strip()).classes("text-[14px] leading-6").style("color: #37352f")

    if entry.definition:
        ui.separator().classes("my-3")
        ui.label("英文释义").classes(MUTED)
        for line in entry.definition.split("\n"):
            if line.strip():
                ui.label(line.strip()).classes("text-[13px] leading-6").style(
                    f"color: {INK_SOFT}"
                )

    if entry.exchange:
        ui.separator().classes("my-3")
        ui.label(f"词形变化 · {entry.exchange}").classes(MUTED)


def _chinese_results(store, word: str, on_pick) -> None:
    """
    中文：渲染一次汉英查询的结果。

    Render the result of a Chinese-to-English lookup.

    The reverse index is built on demand for anyone who installed the dictionary
    before this direction existed, so an old installation gains the feature
    without re-downloading anything.

    Args:
        store (DictionaryStore): Dictionary to query.
        word (str): Chinese query.
        on_pick (Callable[[str], None]): Called with a headword the reader
            selects.
    """
    if not store.reverse_ready():
        with ui.card().classes(CARD):
            ui.label("汉英索引尚未建立").classes(HEADING)
            ui.label(
                "汉英查询需要先从已有词条反向建立索引，约需数秒、增加约 30 MB，"
                "不需要重新下载词库。"
            ).classes(MUTED)

            async def build() -> None:
                import asyncio

                button.disable()
                spinner.visible = True
                try:
                    rows = await asyncio.to_thread(store.build_reverse_index)
                except Exception as exc:  # noqa: BLE001
                    logger.exception("Reverse index build failed")
                    ui.notify(f"建立失败：{exc}", type="negative")
                    button.enable()
                    spinner.visible = False
                    return
                ui.notify(f"已建立 {rows:,} 条索引", type="positive")
                on_pick(word)

            with ui.row().classes("items-center gap-3 mt-3"):
                button = ui.button("建立汉英索引", on_click=build).classes("lt-btn-primary")
                spinner = ui.spinner(size="sm")
                spinner.visible = False
        return

    hits = store.lookup_zh(word)
    if hits:
        with ui.card().classes(CARD):
            with ui.row().classes("items-baseline gap-3"):
                ui.label(word).classes("text-[22px] font-semibold").style("color: #37352f")
                ui.label(f"{len(hits)} 个候选词 · 点按查看完整词条").classes(MUTED)
            with ui.column().classes("w-full gap-0 mt-2"):
                _render_reverse(hits, on_pick)

    terms = _glossary_fallback(word, chinese=True)
    if terms:
        with ui.card().classes(CARD):
            ui.label("术语库" if hits else "术语库（词典未收录）").classes(HEADING)
            for src, tgt in terms:
                with ui.row().classes("items-baseline gap-3 cursor-pointer").on(
                    "click", lambda w=src: on_pick(w)
                ):
                    ui.label(tgt).classes("text-[14px]").style(f"color: {PURPLE}")
                    ui.label("→").classes(MUTED)
                    ui.label(src).classes("text-[14px] font-medium").style("color: #37352f")

    if not hits and not terms:
        with ui.card().classes(CARD):
            ui.label(f"没有找到「{word}」").classes(HEADING)
            ui.label(
                "汉英查询按释义匹配，只有词典里某个词条的中文释义与之相符才会命中；"
                "冷僻或较新的术语可以先用文本翻译页试试。"
            ).classes(MUTED)


@ui.page("/dictionary")
def dictionary_page() -> None:
    """
    中文：渲染词典页。

    Render the dictionary page.
    """
    from local_translator.web.app import _header, _page_body

    _header("dictionary")
    store = get_dictionary()

    with _page_body():
        if not store.installed():
            with ui.card().classes(CARD):
                ui.label("词库尚未安装").classes(HEADING)
                ui.label(
                    "英汉词典使用开源的 ECDICT 词库（MIT 许可，约 77 万词条，"
                    "含音标、词频与考试标签）。下载约 70 MB，导入后占用约 112 MB。"
                ).classes(MUTED)

                bar = ui.linear_progress(value=0, show_value=False).classes("w-full mt-3")
                bar.visible = False
                label = ui.label("").classes(MUTED)

                async def install() -> None:
                    import asyncio

                    from local_translator.dictionary.install import install_dictionary

                    button.disable()
                    bar.visible = True

                    def report(stage: str, frac: float) -> None:
                        bar.value = frac
                        label.text = f"{stage} · {frac:.0%}"

                    try:
                        result = await asyncio.to_thread(install_dictionary, None, report)
                    except Exception as exc:  # noqa: BLE001
                        logger.exception("Dictionary install failed")
                        ui.notify(f"安装失败：{exc}", type="negative")
                        button.enable()
                        return
                    ui.notify(f"已导入 {result['entries']:,} 条词条", type="positive")
                    ui.navigate.reload()

                button = ui.button("下载并安装词库", on_click=install).classes("lt-btn-primary mt-3")
            return

        with ui.card().classes(CARD):
            with ui.row().classes("w-full items-center gap-2"):
                box = ui.input(placeholder="输入英文单词或中文词…").props(
                    f"{INPUT_PROPS} autofocus clearable"
                ).classes("flex-1")
                ui.button("查询", on_click=lambda: results.refresh()).classes("lt-btn-primary")
            ui.label(
                f"离线查询 · {store.count():,} 条词条 · "
                "英汉与汉英同一个输入框，输入中文即按汉英查 · "
                "查不到时自动回退到术语库"
            ).classes(MUTED)

        def pick(word: str) -> None:
            """中文：改查某个候选词。 / Look the chosen headword up instead."""
            box.value = word
            results.refresh()

        @ui.refreshable
        def results() -> None:
            """中文：渲染查询结果。 / Render the lookup result."""
            word = (box.value or "").strip()
            if not word:
                return

            if _CJK.search(word):
                _chinese_results(store, word, pick)
                return

            entry = store.lookup(word)
            if entry:
                with ui.card().classes(CARD):
                    _render_entry(entry)

            terms = _glossary_fallback(word)
            if terms:
                with ui.card().classes(CARD):
                    ui.label("术语库" if entry else "术语库（词典未收录）").classes(HEADING)
                    for src, tgt in terms:
                        with ui.row().classes("items-baseline gap-3"):
                            ui.label(src).classes("text-[14px] font-medium").style(
                                "color: #37352f"
                            )
                            ui.label("→").classes(MUTED)
                            ui.label(tgt).classes("text-[14px]").style(f"color: {PURPLE}")

            if not entry and not terms:
                with ui.card().classes(CARD):
                    ui.label(f"没有找到「{word}」").classes(HEADING)
                    hints = store.suggest(word[:4], limit=10)
                    if hints:
                        ui.label("你是否想查：").classes(MUTED)
                        with ui.row().classes("gap-2 flex-wrap mt-1"):
                            for hint in hints:
                                ui.button(
                                    hint,
                                    on_click=lambda h=hint: (
                                        setattr(box, "value", h), results.refresh()
                                    ),
                                ).classes("lt-btn").props("dense no-caps")

        # Both events are needed. Quasar syncs an input's model on `change`,
        # which Enter also fires, but the keydown reaches the server first and
        # would read the previous value; binding `change` as well means the
        # refresh runs once the new value has arrived.
        # 两个事件都要绑。Quasar 在 change 时同步输入值，Enter 也会触发 change，
        # 但 keydown 先到服务端、读到的还是旧值；同时绑 change 可确保新值到达后再刷新。
        box.on("keydown.enter", lambda: results.refresh())
        box.on("change", lambda: results.refresh())
        results()
