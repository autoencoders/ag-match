from __future__ import annotations

import pytest

from ag_match import (
    InMemorySearchTool,
    MatchConfig,
    Record,
    SearchResult,
    SearchSession,
    normalize_query,
)
from ag_match.tools import (
    NOTE_BUDGET,
    NOTE_EMPTY_QUERY,
    NOTE_LAST_SEARCH,
    NOTE_NO_RESULTS,
    NOTE_REPEAT,
)


def test_normalize_query_folds_case_accents_whitespace():
    assert normalize_query("  Acmé   HOLDINGS ") == "acme holdings"


def test_in_memory_substring_search_is_case_and_accent_insensitive(tool):
    result = tool.search("acme", limit=10)
    assert result.total_count == 3
    assert {r.id for r in result.records} == {"c1", "c2", "c3"}


def test_in_memory_search_respects_limit_but_counts_everything(tool):
    result = tool.search("acme", limit=2)
    assert result.total_count == 3
    assert len(result.records) == 2


def test_in_memory_search_over_extra_fields(tool):
    assert tool.search("GB", limit=10).total_count == 3
    plain = InMemorySearchTool(tool._records)
    assert plain.search("GB", limit=10).total_count == 0


def test_in_memory_empty_query_matches_nothing(tool):
    assert tool.search("   ", limit=10).total_count == 0


async def test_session_dedupes_records_across_searches(tool):
    session = SearchSession(tool, MatchConfig())
    first = await session.run("Acme")
    assert {r.id for r in first.records} == {"c1", "c2", "c3"}
    assert first.note is None

    second = await session.run("Acme Widgets")
    assert second.total_count == 1
    assert second.records == []
    assert second.already_shown_ids == ["c2"]
    assert set(session.seen) == {"c1", "c2", "c3"}


async def test_session_too_many_hides_records_and_asks_to_narrow(tool):
    session = SearchSession(tool, MatchConfig(records_per_search=2, too_many_threshold=2))
    reply = await session.run("Acme")
    assert reply.total_count == 3
    assert reply.records == []
    assert "Too many results" in reply.note
    assert session.seen == {}


async def test_session_truncates_at_records_per_search(tool):
    session = SearchSession(tool, MatchConfig(records_per_search=2, too_many_threshold=5))
    reply = await session.run("Acme")
    assert reply.total_count == 3
    assert len(reply.records) == 2
    assert "Showing 2 of 3" in reply.note


async def test_session_no_results_hint(tool):
    session = SearchSession(tool, MatchConfig())
    reply = await session.run("Zzyzx")
    assert reply.total_count == 0
    assert reply.note == NOTE_NO_RESULTS


async def test_session_repeat_query_does_not_hit_backend(tool):
    calls = []
    original = tool.search

    def counting(query, limit):
        calls.append(query)
        return original(query, limit)

    tool.search = counting
    session = SearchSession(tool, MatchConfig())
    await session.run("Acme")
    reply = await session.run("  ACMÉ ")
    assert reply.note == NOTE_REPEAT
    assert reply.total_count == 3
    assert calls == ["Acme"]
    assert session.searches_used == 1
    assert [t.executed for t in session.trace] == [True, False]


async def test_session_empty_query_short_circuits(tool):
    session = SearchSession(tool, MatchConfig())
    reply = await session.run("  ")
    assert reply.note == NOTE_EMPTY_QUERY
    assert session.searches_used == 0


async def test_session_search_budget(tool):
    session = SearchSession(tool, MatchConfig(max_searches=2))
    await session.run("Acme")
    last = await session.run("Globex")
    assert last.note == NOTE_LAST_SEARCH
    assert session.searches_left == 0
    denied = await session.run("Initech")
    assert denied.note == NOTE_BUDGET.format(max=2)
    assert denied.records == []
    assert session.searches_used == 2


async def test_session_accepts_async_tools_and_plain_dicts():
    class AsyncTool:
        description = "async backend"

        async def search(self, query: str, limit: int):
            return {"total_count": 1, "records": [{"id": "x", "name": query}]}

    session = SearchSession(AsyncTool(), MatchConfig())
    reply = await session.run("hello")
    assert reply.records == [Record(id="x", name="hello")]


def test_config_threshold_must_cover_cap():
    with pytest.raises(ValueError):
        MatchConfig(records_per_search=20, too_many_threshold=10)


def test_search_result_rejects_negative_count():
    with pytest.raises(ValueError):
        SearchResult(total_count=-1)
