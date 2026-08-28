"""
中文：Notion 风格主题——设计令牌、Quasar 样式覆盖与常用类名常量。

Notion-style visual theme.

The design language is deliberately restrained: a white ground with a warm
secondary surface, hairline borders instead of shadows, and small corner radii.
It was chosen over the neumorphic alternative for three reasons specific to this
application:

- Status has to be readable at a glance in a list that may hold dozens of jobs.
  A coloured badge does that; a raised-versus-inset shadow does not.
- Every element in a neumorphic design needs clearance for its shadow to spread,
  which costs vertical space in exactly the place that matters most here — a
  dense queue of translation jobs.
- The reader sits beside a rendered PDF page, which is white. A white interface
  reads as continuous with the document; a grey one reads as a separate panel.

NiceGUI renders through Quasar, whose components ship with their own shadows,
radii and uppercase button labels. :data:`NOTION_CSS` overrides those defaults
so the component library follows this theme rather than fighting it.
"""

from __future__ import annotations

# ── Design tokens ─────────────────────────────────────────────────────────── #
# Values come from the Notion style specification and are kept here as the
# single source of truth so pages never hardcode a colour.
# 设计令牌集中于此，页面不得硬编码颜色。

INK = "#37352f"          # Primary text / 主文字
INK_SOFT = "#6b7280"     # Secondary text / 次级文字
INK_FAINT = "#9b9a97"    # Captions, metadata / 说明与元信息
SURFACE = "#ffffff"      # Page and card ground / 页面与卡片底色
SURFACE_ALT = "#f7f6f3"  # Notion's signature warm beige / 标志性米色
LINE = "#e5e7eb"         # Hairline border / 发丝边框
HOVER = "#f1f1ef"        # Hover wash / 悬停底色

BLUE = "#2eaadc"         # Accent, in-progress / 强调色、进行中
RED = "#eb5757"          # Failure, destructive / 失败与危险操作
GREEN = "#0f7b6c"        # Success / 成功
AMBER = "#d9730d"        # Interrupted, warning / 中断与警告
PURPLE = "#a083ff"       # Reserved for glossary accents / 术语库强调色

# Badge colour pairs (text, background) per job status.
# 各任务状态对应的（文字色, 背景色）。
STATUS_COLOURS: dict[str, tuple[str, str]] = {
    "queued": (INK_SOFT, SURFACE_ALT),
    "running": (BLUE, "#e8f6fb"),
    "completed": (GREEN, "#e7f3f1"),
    "failed": (RED, "#fdeded"),
    "cancelled": (AMBER, "#fdf3e7"),
    "interrupted": (AMBER, "#fdf3e7"),
}


# ── Class-name constants ──────────────────────────────────────────────────── #
# Shared Tailwind class strings, so a spacing or border decision is made once.
# 共用类名串，间距与边框只在一处决定。

CARD = "lt-card w-full"
CARD_TIGHT = "lt-card lt-card-tight w-full"
HEADING = "text-[14.5px] font-semibold"
MUTED = "text-xs text-[#9b9a97]"
META = "text-[12px] text-[#6b7280]"
ROW = "lt-row w-full"
BTN = "lt-btn"
BTN_PRIMARY = "lt-btn lt-btn-primary"
BTN_DANGER = "lt-btn lt-btn-danger"
INPUT_PROPS = "dense outlined"


NOTION_CSS = f"""
<style>
  /* ── Ground and type ────────────────────────────────────────────────── */
  :root {{
    --lt-ink: {INK};
    --lt-ink-soft: {INK_SOFT};
    --lt-ink-faint: {INK_FAINT};
    --lt-surface: {SURFACE};
    --lt-surface-alt: {SURFACE_ALT};
    --lt-line: {LINE};
    --lt-hover: {HOVER};
    --lt-blue: {BLUE};
    --lt-red: {RED};
  }}

  body, .nicegui-content {{
    background: var(--lt-surface);
    color: var(--lt-ink);
    font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI",
                 "PingFang SC", "Hiragino Sans GB", sans-serif;
    -webkit-font-smoothing: antialiased;
  }}
  .nicegui-content {{ padding: 0; }}

  /* Digits in progress counts and token totals must not jitter as they tick.
     进度计数与 token 数字跳动时不应抖动。 */
  .lt-num, .lt-usage, .lt-progress-label {{ font-variant-numeric: tabular-nums; }}

  /* ── Header ─────────────────────────────────────────────────────────── */
  .q-header {{
    background: var(--lt-surface) !important;
    color: var(--lt-ink) !important;
    border-bottom: 1px solid var(--lt-line);
    box-shadow: none !important;
  }}

  .lt-navlink {{
    padding: 5px 11px; border-radius: 6px;
    font-size: 13.5px; font-weight: 500; color: var(--lt-ink-soft);
    text-decoration: none; transition: background .15s, color .15s;
  }}
  .lt-navlink:hover {{ background: var(--lt-hover); color: var(--lt-ink); }}
  .lt-navlink-active {{
    background: var(--lt-surface-alt); color: var(--lt-ink); font-weight: 600;
  }}

  /* ── Cards: hairline border, never a drop shadow ────────────────────── */
  .lt-card {{
    background: var(--lt-surface);
    border: 1px solid var(--lt-line);
    border-radius: 8px;
    box-shadow: 0 1px 2px 0 rgb(0 0 0 / .05) !important;
    padding: 18px;
  }}
  .lt-card-tight {{ padding: 14px; }}
  .q-card {{ border-radius: 8px; }}
  .q-card--bordered {{ border-color: var(--lt-line); }}

  .lt-row {{
    border: 1px solid var(--lt-line);
    border-radius: 6px;
    padding: 14px 16px;
    transition: border-color .15s;
  }}
  .lt-row:hover {{ border-color: #d6d5d2; }}

  /* ── Buttons: no uppercase, no elevation ────────────────────────────── */
  .q-btn {{
    text-transform: none !important;
    font-weight: 500;
    border-radius: 6px;
    box-shadow: none !important;
  }}
  .q-btn:before {{ box-shadow: none !important; }}

  .lt-btn {{
    background: var(--lt-surface) !important;
    color: var(--lt-ink) !important;
    border: 1px solid var(--lt-line);
    padding: 6px 14px;
    font-size: 13.5px;
  }}
  .lt-btn:hover {{ background: var(--lt-hover) !important; }}
  .lt-btn-primary {{
    background: var(--lt-ink) !important;
    color: #fff !important;
    border-color: var(--lt-ink);
  }}
  .lt-btn-primary:hover {{ background: #4b4843 !important; }}
  .lt-btn-danger {{ color: var(--lt-red) !important; }}

  /* ── Inputs ─────────────────────────────────────────────────────────── */
  .q-field--outlined .q-field__control {{
    border-radius: 6px;
    background: var(--lt-surface);
  }}
  .q-field--outlined .q-field__control:before {{
    border: 1px solid var(--lt-line);
  }}
  .q-field--outlined.q-field--focused .q-field__control:after {{
    border-width: 1px;
    border-color: var(--lt-blue);
  }}
  .q-field__native, .q-field__input {{ color: var(--lt-ink); font-size: 13.5px; }}

  /* ── Tabs: underline, not a pill ────────────────────────────────────── */
  .q-tabs {{
    border-bottom: 1px solid var(--lt-line);
    color: var(--lt-ink-soft);
  }}
  .q-tab {{
    text-transform: none !important;
    font-size: 13.5px; font-weight: 500;
    min-height: 38px; padding: 0 13px;
  }}
  .q-tab--active {{ color: var(--lt-ink); font-weight: 600; }}
  .q-tab__indicator {{ background: var(--lt-ink); height: 2px; }}
  .q-tab-panel {{ padding: 16px 0 0; }}
  .q-tab-panels {{ background: transparent; }}

  /* ── Progress: a thin rule, not a chunky bar ────────────────────────── */
  .q-linear-progress {{
    height: 4px !important;
    border-radius: 2px;
    color: var(--lt-blue);
    background: var(--lt-surface-alt);
  }}

  /* ── Badges ─────────────────────────────────────────────────────────── */
  .q-badge {{
    border-radius: 4px;
    font-size: 11px;
    font-weight: 600;
    padding: 2px 8px;
    box-shadow: none;
  }}

  /* ── Uploader ───────────────────────────────────────────────────────── */
  .q-uploader {{
    border: 1px dashed var(--lt-line);
    border-radius: 8px;
    background: var(--lt-surface-alt);
    box-shadow: none;
    width: 100%; max-width: 100%;
  }}
  .q-uploader:hover {{ border-color: var(--lt-blue); }}
  .q-uploader__header {{ background: transparent; color: var(--lt-ink-soft); }}
  .q-uploader__list {{ display: none; }}

  /* ── Expansion, dialogs, separators ─────────────────────────────────── */
  .q-expansion-item {{ border-radius: 6px; }}
  .q-dialog__inner > .q-card {{
    border-radius: 8px;
    border: 1px solid var(--lt-line);
    box-shadow: 0 4px 12px rgb(0 0 0 / .08) !important;
  }}
  .q-separator {{ background: var(--lt-line); }}

  /* ── Scrollbar ──────────────────────────────────────────────────────── */
  ::-webkit-scrollbar {{ width: 10px; height: 10px; }}
  ::-webkit-scrollbar-track {{ background: transparent; }}
  ::-webkit-scrollbar-thumb {{
    background: #dcdad5; border-radius: 5px; border: 2px solid var(--lt-surface);
  }}
  ::-webkit-scrollbar-thumb:hover {{ background: #c4c1ba; }}

  /* ── Focus visibility, kept even though Quasar hides its own ────────── */
  a:focus-visible, button:focus-visible, .q-btn:focus-visible {{
    outline: 2px solid var(--lt-blue);
    outline-offset: 2px;
  }}

  @media (prefers-reduced-motion: reduce) {{
    * {{ transition-duration: .01ms !important; animation-duration: .01ms !important; }}
  }}
</style>
"""


def apply_palette() -> None:
    """
    中文：把 Quasar 的调色板换成本主题的颜色。

    Rebind Quasar's semantic colour slots to this theme.

    Quasar emits utility classes such as ``bg-primary`` with ``!important``, so
    a stylesheet override races its own rules and wins or loses depending on
    load order. Replacing the palette itself is deterministic and reaches every
    component — buttons, badges, progress bars, focus rings — at once.

    ``primary`` is bound to the ink colour rather than a brand hue because the
    primary action in this interface is a plain dark button, and the blue accent
    is reserved for in-progress state so it means one thing wherever it appears.
    """
    from nicegui import ui

    ui.colors(
        primary=INK,
        secondary=INK_SOFT,
        accent=PURPLE,
        info=BLUE,
        positive=GREEN,
        negative=RED,
        warning=AMBER,
        dark=INK,
    )


def badge_style(status: str) -> str:
    """
    中文：返回某个任务状态对应的徽章内联样式。

    Build the inline style for a status badge.

    Quasar's ``color`` prop only accepts its own palette names, so the Notion
    accents are applied as an inline style instead.

    Args:
        status (str): Job status code.

    Returns:
        str: A CSS declaration string setting colour and background.
    """
    fg, bg = STATUS_COLOURS.get(status, (INK_SOFT, SURFACE_ALT))
    return f"color: {fg}; background: {bg};"
