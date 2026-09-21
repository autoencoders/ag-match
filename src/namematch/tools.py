"""Search tool protocol, the reference in-memory tool, and result shaping.

`SearchTool` is the extension point: implement it against any backend and mount it on
a `Matcher`. Only `search` (mode `contains`) is required; `search_all_terms` and
`search_fuzzy` are optional and unlock the other modes. `SearchSession` wraps a tool
for one match run and applies fetching, similarity ranking, caps, the too-many
threshold, dedupe, repeat detection, prefetch and the search budget. The LLM only
ever sees what `SearchSession` produces.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable, Iterable
from typing import Any, Protocol, runtime_checkable

from .text import (
    default_max_distance,
    distinctive_words,
    fuzzy_word_distance,
    normalize,
    similarity,
    words,
)
from .types import MatchConfig, Record, SearchMode, SearchReply, SearchResult, SearchTrace

SearchFn = Callable[..., "SearchResult | Awaitable[SearchResult]"]


@runtime_checkable
class SearchTool(Protocol):
    """A backend that finds records by a search string.

    `description` is shown to the LLM and must state the matching semantics: what
    fields are searched and roughly how big the list is. Methods may be sync or async
    and return `total_count` (all matches) plus at most `limit` records.

    Required: `search(query, limit)` implements mode `contains`.
    Optional: `search_all_terms(terms, limit)` implements `all_terms`;
    `search_fuzzy(query, limit)` implements `fuzzy`.
    """

    description: str

    def search(self, query: str, limit: int) -> SearchResult | Awaitable[SearchResult]: ...


MODE_METHODS: dict[SearchMode, str] = {
    "contains": "search",
    "all_terms": "search_all_terms",
    "fuzzy": "search_fuzzy",
}

MODE_SEMANTICS: dict[SearchMode, str] = {
    "contains": "the whole string appears inside a name (case and accent insensitive)",
    "all_terms": "every space-separated term appears somewhere in the name, in any order",
    "fuzzy": (
        "every space-separated term matches some word of the name within a small edit "
        "distance (1 for short terms, 2 for longer); use it for suspected misspellings"
    ),
}


def available_modes(tool: object) -> list[SearchMode]:
    return [m for m, attr in MODE_METHODS.items() if callable(getattr(tool, attr, None))]


def normalize_query(query: str) -> str:
    """Key used for repeat detection. Case, accents and whitespace folded."""
    return normalize(query)


class InMemorySearchTool:
    """Reference tool over an in-memory list; implements all three modes.

    Matches on `name` and, optionally, on the listed `extra` fields. Fuzzy search is a
    per-word Levenshtein scan and is meant for lists up to a few tens of thousands of
    rows; larger lists belong in a database-backed tool.
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
        self._words = [set(words(h)) for h in self._haystacks]
        fields = ", ".join(("name", *self._extra_fields))
        self.description = description or (
            f"In-memory list of {len(self._records)} records; searches cover the fields [{fields}]."
        )

    def _haystack(self, record: Record) -> str:
        parts = [record.name, *(str(record.extra.get(f) or "") for f in self._extra_fields)]
        return normalize(" \x1f ".join(parts))

    def search(self, query: str, limit: int) -> SearchResult:
        needle = normalize(query)
        if not needle:
            return SearchResult(total_count=0, records=[])
        hits = [r for r, hay in zip(self._records, self._haystacks, strict=True) if needle in hay]
        return SearchResult(total_count=len(hits), records=hits[:limit])

    def search_all_terms(self, terms: list[str], limit: int) -> SearchResult:
        needles = [normalize(t) for t in terms if normalize(t)]
        if not needles:
            return SearchResult(total_count=0, records=[])
        hits = [
            r
            for r, hay in zip(self._records, self._haystacks, strict=True)
            if all(n in hay for n in needles)
        ]
        return SearchResult(total_count=len(hits), records=hits[:limit])

    def search_fuzzy(self, query: str, limit: int) -> SearchResult:
        terms = words(query)
        if not terms:
            return SearchResult(total_count=0, records=[])
        scored: list[tuple[int, float, Record]] = []
        for r, hay in zip(self._records, self._haystacks, strict=True):
            total = 0
            for t in terms:
                d = fuzzy_word_distance(t, hay, max_distance=default_max_distance(t))
                if d is None:
                    break
                total += d
            else:
                scored.append((total, -similarity(query, r.name), r))
        scored.sort(key=lambda x: (x[0], x[1]))
        return SearchResult(total_count=len(scored), records=[r for _, _, r in scored[:limit]])


NOTE_TOO_MANY = (
    "Too many results ({count} > {threshold}); none shown. Search a longer or more "
    "distinctive string."
)
NOTE_TOO_MANY_SHOWING = (
    "Too many results ({count} > {threshold}). Showing the {shown} most similar of the first "
    "{fetched} fetched; the right record may be outside them. Narrow the string to see more."
)
NOTE_TRUNCATED = (
    "Showing the {shown} most similar of {count} matches. Narrow the string if none fit."
)
NOTE_NO_RESULTS = (
    "No results. Try a shorter fragment, a different spelling, another distinctive word, "
    "or fuzzy mode if available."
)
NOTE_EMPTY_QUERY = "Empty query ignored."
NOTE_REPEAT = "Already searched this string in this mode (same results). Try a different one."
NOTE_BUDGET = "Search budget exhausted ({max} searches). Decide now from the records already shown."
NOTE_LAST_SEARCH = "This was the last allowed search. Decide next."
NOTE_MODE_UNSUPPORTED = "Mode {mode!r} is not supported by this backend. Available: {modes}."


class SearchSession:
    """Per-run wrapper around a `SearchTool` that shapes results and enforces budgets.

    `query_name` is the name being matched. It drives similarity ranking of fetched
    rows and the prefetch step.
    """

    def __init__(self, tool: SearchTool, config: MatchConfig, query_name: str = "") -> None:
        self.tool = tool
        self.config = config
        self.query_name = query_name
        self.modes = available_modes(tool)
        self.seen: dict[str, Record] = {}
        self.trace: list[SearchTrace] = []
        self._queries: dict[str, SearchTrace] = {}

    @property
    def searches_used(self) -> int:
        return sum(1 for t in self.trace if t.executed and t.source == "agent")

    @property
    def searches_left(self) -> int:
        return max(0, self.config.max_searches - self.searches_used)

    async def prefetch(self) -> list[SearchReply]:
        """Deterministic candidate generation before the LLM's first turn."""
        replies: list[SearchReply] = []
        if not self.config.prefetch or not self.query_name:
            return replies
        terms = distinctive_words(self.query_name)[: self.config.prefetch_terms]
        for term in terms:
            replies.append(await self.run(term, "contains", source="prefetch"))
        if "fuzzy" in self.modes and terms:
            joined = await self.run(" ".join(terms), "fuzzy", source="prefetch")
            replies.append(joined)
            if joined.total_count == 0 and len(terms) > 1:
                # One term may be noise (a word the list does not carry); retry on the
                # single most distinctive term so a typo in it can still be recovered.
                longest = max(terms, key=len)
                replies.append(await self.run(longest, "fuzzy", source="prefetch"))
        return replies

    async def run(
        self,
        query: str,
        mode: SearchMode = "contains",
        *,
        source: str = "agent",
    ) -> SearchReply:
        key = f"{mode}:{normalize(query)}"
        if not normalize(query):
            return self._short_circuit(query, mode, key, NOTE_EMPTY_QUERY, source=source)
        if mode not in self.modes:
            note = NOTE_MODE_UNSUPPORTED.format(mode=mode, modes=", ".join(self.modes))
            return self._short_circuit(query, mode, key, note, source=source)
        if key in self._queries:
            prior = self._queries[key]
            return self._short_circuit(
                query, mode, key, NOTE_REPEAT, total_count=prior.total_count, source=source
            )
        if source == "agent" and self.searches_left == 0:
            note = NOTE_BUDGET.format(max=self.config.max_searches)
            return self._short_circuit(query, mode, key, note, source=source)

        cfg = self.config
        result = await self._call_tool(query, mode, cfg.effective_fetch_limit)
        count = result.total_count
        ranked = self._rank(query, result.records)

        fresh: list[Record] = []
        already: list[str] = []
        note: str | None = None
        show: list[Record] = []
        if count > cfg.too_many_threshold:
            if cfg.show_on_too_many and ranked:
                show = ranked[: cfg.records_per_search]
                note = NOTE_TOO_MANY_SHOWING.format(
                    count=count,
                    threshold=cfg.too_many_threshold,
                    shown=len(show),
                    fetched=len(ranked),
                )
            else:
                note = NOTE_TOO_MANY.format(count=count, threshold=cfg.too_many_threshold)
        else:
            show = ranked[: cfg.records_per_search]
            if count == 0:
                note = NOTE_NO_RESULTS
            elif count > len(show):
                note = NOTE_TRUNCATED.format(shown=len(show), count=count)
        for record in show:
            if record.id in self.seen:
                already.append(record.id)
            else:
                self.seen[record.id] = record
                fresh.append(record)

        trace = SearchTrace(
            query=query,
            mode=mode,
            normalized=key,
            total_count=count,
            shown_ids=[r.id for r in fresh],
            already_shown_ids=already,
            note=note,
            executed=True,
            source=source,  # type: ignore[arg-type]
        )
        self.trace.append(trace)
        self._queries[key] = trace
        if source == "agent" and self.searches_left == 0:
            note = f"{note} {NOTE_LAST_SEARCH}" if note else NOTE_LAST_SEARCH
            trace.note = note
        return SearchReply(
            query=query,
            mode=mode,
            total_count=count,
            records=fresh,
            already_shown_ids=already,
            note=note,
        )

    def _rank(self, query: str, records: list[Record]) -> list[Record]:
        """Most similar to the query name first; stable so backend order breaks ties."""
        reference = self.query_name or query
        return sorted(records, key=lambda r: -similarity(reference, r.name))

    async def _call_tool(self, query: str, mode: SearchMode, limit: int) -> SearchResult:
        fn: SearchFn = getattr(self.tool, MODE_METHODS[mode])
        arg: Any = query.split() if mode == "all_terms" else query
        if inspect.iscoroutinefunction(fn):
            result = await fn(arg, limit)
        else:
            result = await asyncio.to_thread(fn, arg, limit)
            if inspect.isawaitable(result):
                result = await result
        if not isinstance(result, SearchResult):
            result = SearchResult.model_validate(result)
        return result

    def _short_circuit(
        self,
        query: str,
        mode: SearchMode,
        key: str,
        note: str,
        *,
        total_count: int = 0,
        source: str = "agent",
    ) -> SearchReply:
        self.trace.append(
            SearchTrace(
                query=query,
                mode=mode,
                normalized=key,
                total_count=total_count,
                shown_ids=[],
                already_shown_ids=[],
                note=note,
                executed=False,
                source=source,  # type: ignore[arg-type]
            )
        )
        return SearchReply(query=query, mode=mode, total_count=total_count, note=note)
