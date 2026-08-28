"""
中文：术语库管理页——搜索、编辑、切换策略、导入导出。

Glossary management page.

The glossary is the one part of the system the user is expected to curate by
hand, because terminology preference is a judgement call no import can make: the
seed sources supply a defensible rendering, not necessarily the one this reader
wants. So the page is built around correcting and protecting individual terms
rather than around browsing the collection.

Editing a term bumps ``glossary_rev``, which is mixed into the translation cache
key — otherwise a correction would appear to do nothing when a document is
retranslated, because every paragraph would be served from cache.
"""

from __future__ import annotations

import logging

from nicegui import ui

from local_translator.glossary.store import POLICY_KEEP, POLICY_TRANSLATE, GlossaryStore
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
    NOTION_CSS,
    PURPLE,
    ROW,
    SURFACE_ALT,
    apply_palette,
)

logger = logging.getLogger(__name__)

DOMAIN_LABELS: dict[str, str] = {
    "": "全部学科",
    "physics": "物理",
    "chemistry": "化学",
    "materials": "材料",
    "biology": "生物",
    "medicine": "医学",
    "math": "数学",
    "cs": "计算机",
    "ai": "人工智能",
    "engineering": "工程",
    "general": "通用",
}

POLICY_LABELS: dict[str, str] = {
    "": "全部",
    POLICY_TRANSLATE: "翻译",
    POLICY_KEEP: "保持英文",
}



def _bump_revision() -> None:
    """
    中文：手工编辑后提升术语版本号，使旧缓存失效。

    Invalidate cached translations after a manual glossary edit.

    BabelDOC caches each paragraph's translation keyed by its text plus the
    engine's declared parameters. ``glossary_rev`` is one of those parameters, so
    incrementing it is what makes a corrected term actually take effect the next
    time a document is retranslated instead of replaying the old output.

    Only manual edits bump it. Terms accumulated automatically do not, because a
    new document's paragraphs miss the cache anyway and invalidating everything
    on every learned term would throw away the cache's whole value.
    """
    service = get_service()
    service.settings.glossary.glossary_rev += 1
    service.update_settings(service.settings)


def _stat_tile(label: str, value: str, colour: str) -> None:
    """
    中文：渲染一个统计小块。

    Render one summary tile.

    Args:
        label (str): Caption beneath the figure.
        value (str): The figure itself.
        colour (str): Accent colour for the figure.
    """
    with ui.column().classes("gap-0 px-4 py-2 rounded-md").style(f"background: {SURFACE_ALT}"):
        ui.label(value).classes("text-[19px] font-semibold lt-num").style(f"color: {colour}")
        ui.label(label).classes(MUTED)


@ui.page("/glossary")
def glossary_page() -> None:
    """
    中文：术语库管理页面。

    Render the glossary management page.
    """
    from local_translator.web.app import _header, _page_body

    service = get_service()
    store = GlossaryStore(service.db)

    _header("glossary")
    ui.add_head_html(NOTION_CSS)
    apply_palette()

    state = {"query": "", "domain": "", "policy": "", "offset": 0}
    page_size = 40

    with _page_body():

        @ui.refreshable
        def summary() -> None:
            """中文：渲染术语库概览。 / Render the glossary summary tiles."""
            stats = store.stats()
            by_policy = stats["by_policy"]
            by_domain = stats["by_domain"]

            with ui.card().classes(CARD):
                with ui.row().classes("items-center gap-2 mb-4"):
                    ui.label("术语库").classes(HEADING)
                    ui.label("翻译时自动匹配注入，随使用积累").classes(MUTED)

                # Only the live glossary is shown. A retired entry stays in the
                # table as a tombstone so a re-import cannot resurrect it, but
                # that is bookkeeping — this page is for the terms in use.
                # 页面只展示启用中的部分。停用的条目留在表里充当墓碑，防止重新导入时
                # 复活，但那是内部记账；这个页面面向的是正在使用的术语。
                with ui.row().classes("gap-3 flex-wrap"):
                    _stat_tile("术语总数", str(stats["total"]), BLUE)
                    _stat_tile("保持英文", str(by_policy.get(POLICY_KEEP, 0)), PURPLE)
                    for key in ("ai", "physics", "math", "cs", "chemistry", "materials"):
                        if by_domain.get(key):
                            _stat_tile(DOMAIN_LABELS[key], str(by_domain[key]), INK_SOFT)

                if stats["total"] == 0:
                    with ui.column().classes("w-full items-center py-8 gap-2"):
                        ui.label("术语库还是空的").classes("text-sm text-[#9b9a97]")
                        # There is nothing for the reader to do here any more.
                        # The glossary ships with the program and is installed
                        # at startup, so an empty one means that read failed —
                        # which the log will say, and a restart will retry.
                        # Sending them to the wizard would be sending them to a
                        # step that no longer exists.
                        # 这里已经没有需要读者操作的事了。术语库随程序发布、启动时导入，
                        # 空的说明那次读取失败——日志里会写明，重启会再试一次。把人指向
                        # 向导等于指向一个已经不存在的步骤。
                        ui.label("术语库随程序一同发布，会在启动时自动导入；"
                                 "如果一直是空的，请重启程序并查看日志。").classes(MUTED)

        @ui.refreshable
        def term_list() -> None:
            """中文：渲染术语列表。 / Render the filtered term list."""
            rows = store.search(
                state["query"],
                domain=state["domain"] or None,
                policy=state["policy"] or None,
                limit=page_size,
                offset=state["offset"],
            )

            with ui.card().classes(CARD):
                if not rows:
                    ui.label("没有匹配的术语").classes("text-sm text-[#9b9a97] py-4")
                    return

                with ui.column().classes("w-full gap-2"):
                    for row in rows:
                        _term_row(store, row, term_list.refresh, summary.refresh)

                with ui.row().classes("w-full items-center justify-between mt-4"):
                    ui.label(
                        f"第 {state['offset'] + 1}–{state['offset'] + len(rows)} 条"
                    ).classes(META)
                    with ui.row().classes("gap-2"):

                        def prev_page() -> None:
                            state["offset"] = max(0, state["offset"] - page_size)
                            term_list.refresh()

                        def next_page() -> None:
                            state["offset"] += page_size
                            term_list.refresh()

                        ui.button("上一页", on_click=prev_page).props(
                            "flat dense no-caps"
                        ).set_enabled(state["offset"] > 0)
                        ui.button("下一页", on_click=next_page).props(
                            "flat dense no-caps"
                        ).set_enabled(len(rows) == page_size)

        summary()

        # Filters / 过滤器
        with ui.card().classes(CARD):
            with ui.row().classes("w-full items-end gap-3 flex-wrap"):
                search_input = (
                    ui.input("搜索术语", placeholder="中英文均可")
                    .classes("flex-1 min-w-[220px]")
                    .props(f"{INPUT_PROPS} clearable")
                )

                def on_search(event) -> None:
                    state["query"] = (event.value or "").strip()
                    state["offset"] = 0
                    term_list.refresh()

                search_input.on_value_change(on_search)

                domain_select = ui.select(DOMAIN_LABELS, value="").props(INPUT_PROPS).classes("w-40")
                policy_select = ui.select(POLICY_LABELS, value="").props(INPUT_PROPS).classes("w-32")

                def on_domain(event) -> None:
                    state["domain"] = event.value or ""
                    state["offset"] = 0
                    term_list.refresh()

                def on_policy(event) -> None:
                    state["policy"] = event.value or ""
                    state["offset"] = 0
                    term_list.refresh()

                domain_select.on_value_change(on_domain)
                policy_select.on_value_change(on_policy)

                ui.button(
                    "添加术语",
                    on_click=lambda: _add_dialog(store, term_list.refresh, summary.refresh),
                ).props("no-caps unelevated").classes("lt-btn lt-btn-primary")

        term_list()


def _term_row(store: GlossaryStore, row: dict, refresh, refresh_summary) -> None:
    """
    中文：渲染一条术语，支持就地编辑、切换策略、停用/恢复与删除。

    Render one glossary entry with its inline controls.

    A retired entry is drawn faded and tagged, so that browsing the retired half
    of the table never looks like browsing the live glossary.

    Args:
        store (GlossaryStore): Store used for edits.
        row (dict): The glossary row.
        refresh (Callable[[], None]): Refresh the list after a change.
        refresh_summary (Callable[[], None]): Refresh the summary tiles.
    """
    is_keep = row["policy"] == POLICY_KEEP

    with ui.row().classes(f"{ROW} items-center gap-3"):
        ui.label(row["src"]).classes("text-[13.5px] font-medium w-64 truncate").style(
            "color: #37352f"
        )
        ui.icon("arrow_forward", size="0.8rem").classes("text-[#9b9a97]")

        if is_keep:
            ui.label("保持英文原样").classes("text-[13px] flex-1").style(f"color: {PURPLE}")
        else:
            ui.label(row["tgt"]).classes("text-[13.5px] flex-1 truncate").style(
                f"color: {GREEN}"
            )

        # DOMAIN_LABELS[""] is the filter dropdown's "全部学科", which is not a
        # subject a term can belong to — an unclassified term shows a dash.
        # DOMAIN_LABELS[""] 是筛选框的「全部学科」，不是术语可归属的学科；未分类显示破折号。
        domain = row["domain"] or ""
        ui.label(DOMAIN_LABELS.get(domain, domain) if domain else "—").classes(META)
        ui.label(row["source"] or "-").classes(MUTED)
        if row["hit_count"]:
            ui.label(f"命中 {row['hit_count']}").classes(f"{MUTED} lt-num")

        def toggle_policy() -> None:
            store.set_policy(row["id"], POLICY_TRANSLATE if is_keep else POLICY_KEEP)
            _bump_revision()
            ui.notify("已改为翻译" if is_keep else "已改为保持英文", type="positive")
            refresh()
            refresh_summary()

        ui.button(
            icon="translate" if is_keep else "lock",
            on_click=toggle_policy,
        ).props("flat dense round size=sm").tooltip(
            "改为翻译此术语" if is_keep else "改为保持英文原样"
        )

        ui.button(
            icon="edit",
            on_click=lambda: _edit_dialog(store, row, refresh, refresh_summary),
        ).props("flat dense round size=sm").tooltip("编辑")

        def delete() -> None:
            # Retire rather than remove. The row is what stops a later
            # `glossary seed` re-import from adding the same term again, since
            # the importer matches on (source, target) and does not look at the
            # flag. A deleted term would come back; a retired one does not.
            # 停用而非删除。这一行正是防止下次 `glossary seed` 重新导入同一个术语的
            # 依据——导入按 (原文, 译文) 判重，并不看这个标志位。删掉的会回来，停用的
            # 不会。
            store.set_enabled(row["id"], False)
            _bump_revision()
            ui.notify(f"已删除：{row['src']}", type="warning")
            refresh()
            refresh_summary()

        ui.button(icon="delete_outline", on_click=delete).props(
            "flat dense round size=sm"
        ).classes("text-[#eb5757]").tooltip("删除")


def _edit_dialog(store: GlossaryStore, row: dict, refresh, refresh_summary) -> None:
    """
    中文：打开术语编辑对话框。

    Open the term editing dialog.

    Args:
        store (GlossaryStore): Store used for the update.
        row (dict): The row being edited.
        refresh (Callable[[], None]): Refresh the list afterwards.
        refresh_summary (Callable[[], None]): Refresh the summary afterwards.
    """
    with ui.dialog() as dialog, ui.card().classes("w-[420px] p-5 gap-3"):
        ui.label("编辑术语").classes(HEADING)
        src = ui.input("英文术语", value=row["src"]).classes("w-full").props(INPUT_PROPS)
        tgt = ui.input("中文译法", value=row["tgt"]).classes("w-full").props(INPUT_PROPS)
        domain = ui.select(
            {k: v for k, v in DOMAIN_LABELS.items() if k},
            label="学科",
            value=row["domain"] or "general",
        ).classes("w-full").props(INPUT_PROPS)

        def save() -> None:
            if not src.value or not tgt.value:
                ui.notify("英文术语与中文译法都不能为空", type="warning")
                return
            store.update(
                row["id"],
                src=src.value.strip(),
                tgt=tgt.value.strip(),
                domain=domain.value,
                source="manual",
            )
            _bump_revision()
            ui.notify("已保存", type="positive")
            dialog.close()
            refresh()
            refresh_summary()

        with ui.row().classes("w-full justify-end gap-2 mt-2"):
            ui.button("取消", on_click=dialog.close).props("flat no-caps")
            ui.button("保存", on_click=save).props("no-caps unelevated").classes(
                "lt-btn lt-btn-primary"
            )

    dialog.open()


def _add_dialog(store: GlossaryStore, refresh, refresh_summary) -> None:
    """
    中文：打开新增术语对话框。

    Open the "add term" dialog.

    Args:
        store (GlossaryStore): Store used for the insert.
        refresh (Callable[[], None]): Refresh the list afterwards.
        refresh_summary (Callable[[], None]): Refresh the summary afterwards.
    """
    from local_translator.glossary.store import TermRecord

    with ui.dialog() as dialog, ui.card().classes("w-[420px] p-5 gap-3"):
        ui.label("添加术语").classes(HEADING)
        src = ui.input("英文术语").classes("w-full").props(INPUT_PROPS)
        tgt = ui.input("中文译法").classes("w-full").props(INPUT_PROPS)
        keep = ui.checkbox("保持英文原样（不翻译）")
        domain = ui.select(
            {k: v for k, v in DOMAIN_LABELS.items() if k},
            label="学科",
            value="general",
        ).classes("w-full").props(INPUT_PROPS)

        def on_keep(event) -> None:
            tgt.set_enabled(not event.value)
            if event.value:
                tgt.value = src.value

        keep.on_value_change(on_keep)

        def save() -> None:
            source_term = (src.value or "").strip()
            if not source_term:
                ui.notify("请填写英文术语", type="warning")
                return
            target = source_term if keep.value else (tgt.value or "").strip()
            if not target:
                ui.notify("请填写中文译法", type="warning")
                return
            store.add(
                TermRecord(
                    src=source_term,
                    tgt=target,
                    policy=POLICY_KEEP if keep.value else POLICY_TRANSLATE,
                    domain=domain.value,
                    source="manual",
                    confidence=1.0,
                )
            )
            _bump_revision()
            ui.notify(f"已添加：{source_term}", type="positive")
            dialog.close()
            refresh()
            refresh_summary()

        with ui.row().classes("w-full justify-end gap-2 mt-2"):
            ui.button("取消", on_click=dialog.close).props("flat no-caps")
            ui.button("添加", on_click=save).props("no-caps unelevated").classes(
                "lt-btn lt-btn-primary"
            )

    dialog.open()
