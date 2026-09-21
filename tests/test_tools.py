from __future__ import annotations

import pytest

from namematch import (
    InMemorySearchTool,
    MatchConfig,
    Record,
    SearchResult,
    SearchSession,
    normalize_query,
)
from namematch.tools import (
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


async def test_session_too_many_shows_most_similar_by_default(tool):
    cfg = MatchConfig(records_per_search=2, too_many_threshold=2, fetch_limit=10)
    session = SearchSession(tool, cfg, query_name="Acme Software")
    reply = await session.run("Acme")
    assert reply.total_count == 3
    assert [r.id for r in reply.records][0] == "c3" and len(reply.records) == 2
    assert "Too many results (3 > 2)" in reply.note and "2 most similar" in reply.note


async def test_session_too_many_can_hide_records(tool):
    cfg = MatchConfig(records_per_search=2, too_many_threshold=2, show_on_too_many=False)
    session = SearchSession(tool, cfg)
    reply = await session.run("Acme")
    assert reply.records == []
    assert "Too many results" in reply.note
    assert session.seen == {}


async def test_session_ranks_by_similarity_to_query_name(tool):
    session = SearchSession(tool, MatchConfig(), query_name="Acme Widgets Limited")
    reply = await session.run("Acme")
    assert [r.id for r in reply.records][0] == "c2"


async def test_session_modes(tool):
    session = SearchSession(tool, MatchConfig())
    assert session.modes == ["contains", "all_terms", "fuzzy"]
    both = await session.run("widgets acme", "all_terms")
    assert [r.id for r in both.records] == ["c2"]
    fuzzy = await session.run("Acmee Widgts", "fuzzy")
    assert fuzzy.already_shown_ids == ["c2"] and fuzzy.total_count == 1
    same = await session.run("ACME", "contains")
    assert same.total_count == 3  # a different mode is not a repeat


async def test_session_unsupported_mode_short_circuits():
    class ContainsOnly:
        description = "contains only"

        def search(self, query, limit):
            return SearchResult(total_count=0)

    session = SearchSession(ContainsOnly(), MatchConfig())
    reply = await session.run("x", "fuzzy")
    assert "not supported" in reply.note and "contains" in reply.note
    assert session.searches_used == 0


async def test_prefetch_runs_distinctive_words_and_fuzzy(tool):
    session = SearchSession(tool, MatchConfig(), query_name="Sirius Cybernetics Group Ltd")
    replies = await session.prefetch()
    assert [(r.query, r.mode) for r in replies] == [
        ("sirius", "contains"),
        ("cybernetics", "contains"),
        ("sirius cybernetics", "fuzzy"),
    ]
    assert replies[0].records[0].id == "c10"
    assert replies[1].already_shown_ids == ["c10"]
    assert session.searches_used == 0
    assert all(t.source == "prefetch" for t in session.trace)


async def test_prefetch_can_be_disabled(tool):
    session = SearchSession(tool, MatchConfig(prefetch=False), query_name="Sirius")
    assert await session.prefetch() == []


async def test_session_truncates_at_records_per_search(tool):
    session = SearchSession(tool, MatchConfig(records_per_search=2, too_many_threshold=5))
    reply = await session.run("Acme")
    assert reply.total_count == 3
    assert len(reply.records) == 2
    assert "Showing the 2 most similar of 3 matches" in reply.note


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
    with pytest.raises(ValueError):
        MatchConfig(records_per_search=20, fetch_limit=10)
    assert MatchConfig().effective_fetch_limit == 60


def test_in_memory_fuzzy_and_all_terms(tool):
    fuzzy = tool.search_fuzzy("Cybernetcs", 5)
    assert [r.id for r in fuzzy.records] == ["c10"]
    assert tool.search_fuzzy("Zzyzx", 5).total_count == 0
    both = tool.search_all_terms(["holdings", "GROUP"], 5)
    assert [r.id for r in both.records] == ["c7"]


def test_search_result_rejects_negative_count():
    with pytest.raises(ValueError):
        SearchResult(total_count=-1)
