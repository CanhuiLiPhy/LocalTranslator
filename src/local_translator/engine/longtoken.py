"""
中文：一个断不开又比栏宽还长的词（比如网址），会被排成每行一个字母。

Stop a word longer than the column from being set one letter per line.

**The symptom.** The abstract of the ICML paper in the bench ends with a link,
and the translated page sets it like this::

    调整阶段。博客：
    h
    t
    t
    p
    s://
    aigeeksgroup.github.io/DontAbuseRL/。

**Why.** Before placing a unit, the typesetter looks ahead to the next point the
line may break, and wraps if the unit plus everything up to that point would
overrun the box::

    width_before_next_break_point = self._get_width_before_next_break_point(
        typesetting_units[i:], scale)
    ...
    if ... or (use_english_line_break
               and current_x + unit_width + width_before_next_break_point > box.x2):

That is the right rule for an ordinary word: move it down whole rather than
split it. A URL has no break points, so the run it measures is the entire
address — 240 points against a column 198 wide. The test is true wherever the
line starts, so the unit wraps; one character is then placed, the measured run
is one character shorter, and the test is true again. The loop walks down the
page a letter at a time until the remainder finally fits, which is why the tail
appears whole on the last line.

**The fix.** A look-ahead is a choice between placing a run here and placing it
on the next line. When the run does not fit on an empty line, there is no such
choice — no line will hold it — and the look-ahead has nothing to decide, so it
stands down and the unit is placed by the ordinary fit test. The word then fills
the line and breaks at the margin, which is what every typesetter does with a
word wider than its measure.

**And the run is measured once too many.** ``_get_width_before_next_break_point``
sums from the current unit forward, so its result already contains that unit;
the caller then adds ``unit_width`` to it again. Every look-ahead is therefore
one character too wide. It shows up as a word wrapping a hair early, and on this
URL it stranded a single letter: with the run at 185.3 and the box 271.4 wide,
the honest sum from x=81.2 is 266.5 and fits, while the doubled one is 272.5 and
does not. Reporting the run without its first unit leaves the caller's arithmetic
correct.

**Scope.** The suppression applies only while the measured run exceeds the
column, and the correction is one unit wide. Runs that fit keep their behaviour:
ordinary words still move down whole rather than splitting.
"""

from __future__ import annotations

import logging
import os
import threading

logger = logging.getLogger(__name__)

# Set LOCALTRANS_LINEBREAK_PROBE to a substring to have every layout unit of the
# paragraph containing it logged with the decision made about it.
# 把 LOCALTRANS_LINEBREAK_PROBE 设成一个子串，即可把含该子串的段落里每个排版单元
# 及其换行判定写进日志。
_PROBE = os.getenv("LOCALTRANS_LINEBREAK_PROBE")

# The column width in force for the layout call now running, kept per thread so
# a caller cannot see another's box.
# 当前正在执行的那次布局所用的栏宽，按线程保存，以免相互串用。
_state = threading.local()


def install() -> None:
    """
    中文：让「断不开的词」的前瞻在该词根本放不下时让位给普通换行。

    Make the break-point look-ahead stand down for a run no line can hold.

    Two methods are wrapped. ``_layout_typesetting_units`` publishes the usable
    width of the box it was handed — the look-ahead is computed from the units
    alone and otherwise has no way to know what it must fit inside. Its own
    width is saved and restored around the call, so a nested layout leaves the
    outer one's width intact.

    ``_get_width_before_next_break_point`` then returns zero for any run wider
    than that, which switches the caller's second wrap condition off for exactly
    those units and leaves the plain "does this unit fit" test to place them. For
    every other run it returns the width without the current unit, which the
    caller adds back itself.

    Idempotent, so a second call is harmless.

    Raises:
        AttributeError: If BabelDOC no longer exposes either method, so an
            upgrade that renames them fails at startup rather than silently
            restoring the one-letter-per-line layout.
    """
    from babeldoc.format.pdf.document_il.midend.typesetting import Typesetting

    if getattr(Typesetting._get_width_before_next_break_point,
               "_longtoken_patched", False):
        return

    layout = Typesetting._layout_typesetting_units
    measure = Typesetting._get_width_before_next_break_point

    def _layout_typesetting_units(self, typesetting_units, box, *args, **kwargs):
        """中文：记下本次布局的栏宽再照常执行。 / Publish the width, then proceed."""
        previous = getattr(_state, "width", None)
        try:
            _state.width = box.x2 - box.x
        except AttributeError:
            _state.width = None
        if _PROBE:
            text = "".join(u.try_get_unicode() or "\uFFFD" for u in typesetting_units)
            if _PROBE in text:
                logger.warning("PROBE box x=%.1f..%.1f width=%.1f  %r",
                               box.x, box.x2, box.x2 - box.x, text[:120])
        try:
            return layout(self, typesetting_units, box, *args, **kwargs)
        finally:
            _state.width = previous

    def _get_width_before_next_break_point(self, typesetting_units, scale):
        """中文：放不下的整段不参与前瞻。 / A run that cannot fit skips the look-ahead."""
        width = measure(self, typesetting_units, scale)
        if width <= 0:
            return width
        limit = getattr(_state, "width", None)
        if limit and width > limit:
            return 0.0
        # The sum starts at the current unit, and the caller adds that unit's
        # width to the result. Hand back the run without it.
        # 求和是从当前单元开始的，而调用方还会把这个单元的宽度再加一次。这里先减掉它。
        return max(0.0, width - typesetting_units[0].width * scale)

    _get_width_before_next_break_point._longtoken_patched = True
    Typesetting._layout_typesetting_units = _layout_typesetting_units
    Typesetting._get_width_before_next_break_point = _get_width_before_next_break_point
