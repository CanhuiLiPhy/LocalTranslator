"""
中文：把随源码发布的术语库快照装进本机。

Install the glossary that ships with the source.

**What this module used to be.** Six hundred lines that built the glossary at
first run by asking the network for it: a SPARQL query per subject class against
query.wikidata.org with a four-second pause between classes and a forty-five
second backoff whenever it was throttled, and a Markdown file fetched from
raw.githubusercontent.com. On the run it was measured against that took twenty-
five minutes thirty seconds, two classes gave up after four attempts, and the
packaged application began the whole thing with nobody pressing anything.

**Why none of that is here now.** Because the result could not be relied on.
Wikidata moves, so two installations a month apart did not get the same
glossary, and no one could say what either of them had. A glossary is editorial
work — somebody decides that 「fidelity」 is 「保真度」 — and editorial work belongs
in the source tree where it can be reviewed, diffed and rolled back, not
re-derived from a moving target on each machine.

The corpus now lives beside this module as a file, and this module has one job:
put it into the database. See :mod:`local_translator.glossary.bundled` for the
file's format and :func:`local_translator.cli.glossary_snapshot` for how it is
produced.

**Recovering the old fetchers.** They are in git history, removed in the commit
that added the bundled snapshot. Nothing imports them, and the module they
served, ``glossary/wikipedia.py``, went with them for the same reason.
"""

from __future__ import annotations

import logging

from local_translator.glossary.store import GlossaryStore

logger = logging.getLogger(__name__)


def seed(
    store: GlossaryStore,
    *,
    overwrite: bool = False,
    **legacy,
) -> dict[str, dict[str, int]]:
    """
    中文：把随源码发布的术语库快照导入本机，不联网。

    Load the glossary that ships with the source into this installation.

    This used to contact two hosts and take twenty-five minutes. It now reads
    one file beside this module — see
    :mod:`local_translator.glossary.bundled` for why that changed and what was
    lost by changing it (nothing that could be relied on: Wikidata moves, so no
    two installations ever got the same glossary).

    Existing rows are left alone unless ``overwrite`` is set, so a person's own
    corrections and the terms an installation learned survive an upgrade. Rows
    the snapshot marks as retired arrive retired, which is what stops a term
    somebody removed from coming back.

    Args:
        store (GlossaryStore): Destination store.
        overwrite (bool): Replace a row that already exists with the shipped
            one. Off by default, so local edits win.
        **legacy: Accepted and ignored. The old signature took ``wikidata``,
            ``jiqizhixin``, ``whitelist``, ``limit_per_class`` and ``refresh``,
            all of which named parts of a network import that no longer exists.
            Swallowing them keeps an old settings file or an old command line
            from failing on an argument that has simply stopped meaning
            anything.

    Returns:
        dict[str, dict[str, int]]: ``{"bundled": {"inserted": n, "updated": n,
            "skipped": n}}``, in the shape the previous per-source report had.
    """
    from local_translator.glossary import bundled

    if legacy:
        logger.debug("Ignoring settings from the network seeding era: %s",
                     ", ".join(sorted(legacy)))

    records = bundled.load()
    if not records:
        logger.warning("The bundled glossary is empty or missing; nothing was seeded")
        return {bundled.BUNDLED: {"inserted": 0, "updated": 0, "skipped": 0}}

    report = store.upsert_many(records, overwrite=overwrite)
    logger.info("Seeded %d terms from the bundled glossary (digest %s)",
                len(records), bundled.digest())
    return {bundled.BUNDLED: report}
