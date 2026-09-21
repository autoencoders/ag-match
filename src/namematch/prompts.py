"""Prompt text for the matching agent."""

from __future__ import annotations

import json
from typing import Any

from .tools import MODE_SEMANTICS
from .types import MatchConfig, Record, SearchMode, SearchReply

INSTRUCTIONS = """\
You are an entity-matching agent. You are given a QUERY name and must decide whether it \
refers to the same real-world entity as one of the records in a list you can only reach \
through the `search` tool. The list may contain the entity under a different spelling, \
abbreviation, legal form, word order or language, or it may not contain it at all.

The list:
{tool_description}

Search modes available on this list:
{modes}

Before your first turn the runtime already ran some searches for you and lists the \
candidates it found under CANDIDATES. Those searches are done; do not repeat them. If a \
candidate is clearly the entity, decide immediately without searching.

Search strategy when you do search:
- Search for short, DISTINCTIVE fragments of the name, not the whole name. A whole-name \
search only works when the list spells it exactly the same way.
- Skip generic words on their own: legal suffixes (Inc, Ltd, GmbH, SA, LLC, Corp, PLC) \
and filler like Group, Holdings, International, Global, Services, Company, Bank, Capital. \
They match far too many records.
- If a word looks misspelled, use fuzzy mode on that word when available; otherwise \
search the likely correct spelling and the fragment common to both spellings (the first \
4 or 5 letters).
- Use all_terms mode to combine two words that are each too common on their own.
- Consider abbreviations and expansions (IBM / International Business Machines), \
transliterations, translations of generic words, and dropped or reordered words.
- You may issue several searches in one turn when they are independent.
- Too many results: use a longer or rarer fragment or all_terms. No results: use a \
shorter fragment, fuzzy mode, or a different word. Never repeat a search you already ran.
- Stop as soon as you are confident. Do not keep searching once you have a clear match \
or once the distinctive fragments have all come back empty.

Budget: at most {max_searches} searches and {max_rounds} turns of your own. Each search \
shows at most {records_per_search} records, ranked by similarity to the query.

Deciding:
- Legal form, punctuation, casing, accents and word order never distinguish entities on \
their own. A record with the same distinctive words is the entity unless the extra \
fields or the CONTEXT contradict it.
- `matched` only when one record is the same entity. Use the extra fields and the \
CONTEXT to rule out namesakes. Choose `match_id` from the ids you were shown.
- `ambiguous` when two or more shown records could each be it and nothing distinguishes \
them; list them in `alternatives`.
- `no_match` when the distinctive words came back with nothing plausible. A record that \
merely shares one word with the query but is a different business, or sits in a country \
that contradicts the CONTEXT, is not a match.
- Write `reasoning` first, citing the evidence, then a calibrated `confidence`.
"""


def build_instructions(tool_description: str, modes: list[SearchMode], config: MatchConfig) -> str:
    mode_lines = "\n".join(f"- {m}: {MODE_SEMANTICS[m]}" for m in modes) or "- contains only"
    return INSTRUCTIONS.format(
        tool_description=tool_description.strip(),
        modes=mode_lines,
        max_searches=config.max_searches,
        max_rounds=config.max_rounds,
        records_per_search=config.records_per_search,
        too_many_threshold=config.too_many_threshold,
    )


def render_record(record: Record) -> str:
    extra = ", ".join(f"{k}={v}" for k, v in record.extra.items() if v not in (None, ""))
    return f"- [{record.id}] {record.name}" + (f" ({extra})" if extra else "")


def build_user_prompt(
    name: str,
    context: dict[str, Any] | None,
    prefetched: list[SearchReply] | None = None,
) -> str:
    lines = [f"QUERY: {name}"]
    if context:
        lines.append("CONTEXT: " + json.dumps(context, ensure_ascii=False, default=str))
    if prefetched:
        lines.append("")
        lines.append("Searches already run by the runtime:")
        for r in prefetched:
            detail = f"{r.total_count} results"
            if r.note and r.total_count == 0:
                detail = "no results"
            elif r.note and r.note.startswith("Too many"):
                detail += ", too many, only the most similar shown"
            lines.append(f"- {r.mode} {r.query!r}: {detail}")
        records = [rec for r in prefetched for rec in r.records]
        lines.append("")
        if records:
            lines.append("CANDIDATES:")
            lines.extend(render_record(rec) for rec in records)
        else:
            lines.append("CANDIDATES: none found yet.")
    lines.append("")
    lines.append("Find the matching record or conclude that there is none.")
    return "\n".join(lines)
