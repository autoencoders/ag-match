"""Search tool protocol, the reference in-memory tool, and result shaping.

`SearchTool` is the extension point: implement it against any backend and mount it on
a `Matcher`. `SearchSession` wraps a tool for one match run and applies caps, the
too-many threshold, dedupe, repeat detection and the search budget. The LLM only ever
sees what `SearchSession` produces.
"""

from __future__ import annotations

import asyncio
import inspect
import unicodedata
from collections.abc import Awaitable, Iterable
from typing import Protocol, runtime_checkable

from .types import MatchConfig, Record, SearchReply, SearchResult, SearchTrace


@runtime_checkable
class SearchTool(Protocol):
    """A backend that finds records by an exact search string.

    `description` is shown to the LLM and must state the matching semantics:
    what fields are searched, case sensitivity, substring vs prefix vs token match,
    and roughly how big the list is. `search` may be sync or async and returns
    `total_count` (all matches) plus at most `limit` records.
    """

    description: str

    def search(self, query: str, limit: int) -> SearchResult | Awaitable[SearchResult]: ...


def normalize_query(query: str) -> str:
    """Key used for repeat detection. Case, accents and whitespace folded."""
    text = unicodedata.normalize("NFKD", query)
    text = "".join(c for c in text if not unicodedata.combining(c))
    return " ".join(text.casefold().split())


class InMemorySearchTool:
    """Case- and accent-insensitive substring search over an in-memory list.

    Reference implementation and test double. Matches on `name` and, optionally,
    on the listed `extra` fields.
    """

    def __init__(
        self,
        records: Iterable[Record],
        *,
        extra_fields: Iterable[str] = (),
        description: str | None = None,
    ) -> None:
        self._records = list(records)
        self._extra_fields = tuple(extra_fields)
        self._haystacks = [self._haystack(r) for r in self._records]
        fields = ", ".join(("name", *self._extra_fields))
        self.description = description or (
            f"Case- and accent-insensitive substring match over the fields [{fields}] of "
            f"{len(self._records)} records. A query matches a record when the query text "
            "appears anywhere inside one of those fields."
        )

    def _haystack(self, record: Record) -> str:
        parts = [record.name, *(str(record.extra.get(f) or "") for f in self._extra_fields)]
        return normalize_query(" \x1f ".join(parts))

    def search(self, query: str, limit: int) -> SearchResult:
        needle = normalize_query(query)
        if not needle:
            return SearchResult(total_count=0, records=[])
        hits = [r for r, hay in zip(self._records, self._haystacks, strict=True) if needle in hay]
        return SearchResult(total_count=len(hits), records=hits[:limit])


NOTE_TOO_MANY = (
    "Too many results ({count} > {threshold}); none shown. Search a longer or more "
    "distinctive string."
)
NOTE_TRUNCATED = "Showing {shown} of {count} matches. Narrow the string if none of these fit."
NOTE_NO_RESULTS = (
    "No results. Try a shorter fragment, a different spelling, or another distinctive word "
    "from the name."
)
NOTE_EMPTY_QUERY = "Empty query ignored."
NOTE_REPEAT = "Already searched this string (same results). Try a different one."
NOTE_BUDGET = "Search budget exhausted ({max} searches). Decide now from the records already shown."
NOTE_LAST_SEARCH = "This was the last allowed search. Decide next."


class SearchSession:
    """Per-run wrapper around a `SearchTool` that shapes results and enforces budgets."""

    def __init__(self, tool: SearchTool, config: MatchConfig) -> None:
        self.tool = tool
        self.config = config
        self.seen: dict[str, Record] = {}
        self.trace: list[SearchTrace] = []
        self._queries: dict[str, SearchTrace] = {}

    @property
    def searches_used(self) -> int:
        return sum(1 for t in self.trace if t.executed)

    @property
    def searches_left(self) -> int:
        return max(0, self.config.max_searches - self.searches_used)

    async def run(self, query: str) -> SearchReply:
        key = normalize_query(query)
        if not key:
            return self._short_circuit(query, key, NOTE_EMPTY_QUERY)
        if key in self._queries:
            prior = self._queries[key]
            return self._short_circuit(query, key, NOTE_REPEAT, total_count=prior.total_count)
        if self.searches_left == 0:
            return self._short_circuit(query, key, NOTE_BUDGET.format(max=self.config.max_searches))

        result = await self._call_tool(query, self.config.records_per_search)
        count = result.total_count
        cfg = self.config

        fresh: list[Record] = []
        already: list[str] = []
        note: str | None = None
        if count > cfg.too_many_threshold:
            note = NOTE_TOO_MANY.format(count=count, threshold=cfg.too_many_threshold)
        else:
            for record in result.records[: cfg.records_per_search]:
                if record.id in self.seen:
                    already.append(record.id)
                else:
                    self.seen[record.id] = record
                    fresh.append(record)
            if count == 0:
                note = NOTE_NO_RESULTS
            elif count > len(result.records):
                note = NOTE_TRUNCATED.format(shown=len(result.records), count=count)

        trace = SearchTrace(
            query=query,
            normalized=key,
            total_count=count,
            shown_ids=[r.id for r in fresh],
            already_shown_ids=already,
            note=note,
            executed=True,
        )
        self.trace.append(trace)
        self._queries[key] = trace
        if self.searches_left == 0:
            note = f"{note} {NOTE_LAST_SEARCH}" if note else NOTE_LAST_SEARCH
            trace.note = note
        return SearchReply(
            query=query,
            total_count=count,
            records=fresh,
            already_shown_ids=already,
            note=note,
        )

    async def _call_tool(self, query: str, limit: int) -> SearchResult:
        fn = self.tool.search
        if inspect.iscoroutinefunction(fn):
            result = await fn(query, limit)
        else:
            result = await asyncio.to_thread(fn, query, limit)
            if inspect.isawaitable(result):
                result = await result
        if not isinstance(result, SearchResult):
            result = SearchResult.model_validate(result)
        return result

    def _short_circuit(
        self, query: str, key: str, note: str, *, total_count: int = 0
    ) -> SearchReply:
        self.trace.append(
            SearchTrace(
                query=query,
                normalized=key,
                total_count=total_count,
                shown_ids=[],
                already_shown_ids=[],
                note=note,
                executed=False,
            )
        )
        return SearchReply(query=query, total_count=total_count, note=note)
