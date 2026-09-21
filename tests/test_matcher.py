from __future__ import annotations

from conftest import scripted_model

from namematch import MatchConfig, Matcher

NO_PREFETCH = MatchConfig(prefetch=False)


def decide(status, match_id=None, confidence=0.9, reasoning="because", alternatives=()):
    return {
        "status": status,
        "match_id": match_id,
        "confidence": confidence,
        "reasoning": reasoning,
        "alternatives": list(alternatives),
    }


async def test_direct_match_in_one_round(tool):
    model, infos = scripted_model(
        [
            [("search", {"query": "Acme"})],
            decide("matched", "c1", reasoning="Acme Holdings International is the query"),
        ]
    )
    run = await Matcher(tool, model=model, config=NO_PREFETCH).match_async(
        "Acme Holdngs Intl.", context={"country": "US"}
    )
    assert run.decision.status == "matched"
    assert run.match is not None and run.match.id == "c1"
    assert [t.query for t in run.searches] == ["Acme"]
    assert run.usage.requests == 2
    assert run.usage.tool_calls == 1
    assert set(run.seen_records) == {"c1", "c2", "c3"}

    first = infos()[0]
    tool_names = {t.name for t in first.function_tools}
    assert tool_names == {"search"}
    search_def = next(t for t in first.function_tools if t.name == "search")
    assert "contains, all_terms, fuzzy" in search_def.description
    assert "mode" in search_def.parameters_json_schema["properties"]
    assert "In-memory list of 10 records" in first.instructions


async def test_parallel_searches_in_one_turn(tool):
    model, _ = scripted_model(
        [
            [("search", {"query": "Globex"}), ("search", {"query": "Initech"})],
            decide("matched", "c4"),
        ]
    )
    run = await Matcher(tool, model=model, config=NO_PREFETCH).match_async("Globex Corp")
    assert [t.query for t in run.searches] == ["Globex", "Initech"]
    assert run.usage.requests == 2


async def test_too_many_shows_most_similar_then_narrow(tool):
    config = MatchConfig(records_per_search=2, too_many_threshold=2, prefetch=False)
    model, _ = scripted_model(
        [
            [("search", {"query": "Acme"})],
            [("search", {"query": "Acme Widgets"})],
            decide("matched", "c2"),
        ]
    )
    run = await Matcher(tool, model=model, config=config).match_async("Acme Widgets")
    assert run.searches[0].shown_ids == ["c2", "c1"]  # most similar to the query first
    assert "Too many" in run.searches[0].note and "most similar" in run.searches[0].note
    assert run.searches[1].already_shown_ids == ["c2"]
    assert run.match.id == "c2"


async def test_too_many_can_hide_everything(tool):
    config = MatchConfig(
        records_per_search=2, too_many_threshold=2, prefetch=False, show_on_too_many=False
    )
    model, _ = scripted_model(
        [
            [("search", {"query": "Acme"})],
            [("search", {"query": "Widgets"})],
            decide("matched", "c2"),
        ]
    )
    run = await Matcher(tool, model=model, config=config).match_async("Acme Widgets")
    assert run.searches[0].shown_ids == []
    assert run.searches[0].note.startswith("Too many results (3 > 2); none shown")


async def test_prefetch_presents_candidates_and_allows_immediate_decision(tool):
    model, infos = scripted_model([decide("matched", "c1", reasoning="prefetched candidate")])
    run = await Matcher(tool, model=model).match_async("Acmee Holdngs Intl", {"country": "US"})
    assert run.match.id == "c1"
    assert run.usage.requests == 1 and run.usage.tool_calls == 0
    assert all(t.source == "prefetch" for t in run.searches)
    assert [(t.query, t.mode) for t in run.searches] == [
        ("acmee", "contains"),
        ("holdngs", "contains"),
        ("acmee holdngs", "fuzzy"),
    ]
    assert [t.total_count for t in run.searches] == [0, 0, 1]
    assert run.searches[2].shown_ids == ["c1"]
    prompt = infos()[0]  # instructions only; the user prompt is in the messages
    assert "do not repeat them" in prompt.instructions
    user_text = run.messages[0].parts[0].content
    assert (
        "CANDIDATES:" in user_text
        and "[c1] Acme Holdings International Inc (country=US)" in user_text
    )
    assert "fuzzy 'acmee holdngs': 1 results" in user_text
    assert "contains 'acmee': no results" in user_text


async def test_prefetch_searches_do_not_consume_agent_budget(tool):
    model, _ = scripted_model([[("search", {"query": "Widg"})], decide("matched", "c2")])
    config = MatchConfig(max_searches=1)
    run = await Matcher(tool, model=model, config=config).match_async("Acme Widgets")
    assert sum(1 for t in run.searches if t.source == "prefetch") == 3
    agent = [t for t in run.searches if t.source == "agent"]
    assert len(agent) == 1 and agent[0].executed
    assert "last allowed search" in agent[0].note


async def test_search_modes_reach_the_backend(tool):
    model, _ = scripted_model(
        [
            [
                ("search", {"query": "widgets acme", "mode": "all_terms"}),
                ("search", {"query": "Acmee", "mode": "fuzzy"}),
            ],
            decide("matched", "c2"),
        ]
    )
    run = await Matcher(tool, model=model, config=NO_PREFETCH).match_async("Acme Widgets")
    by_mode = {t.mode: t for t in run.searches}
    assert by_mode["all_terms"].shown_ids == ["c2"]
    fuzzy_ids = set(by_mode["fuzzy"].shown_ids) | set(by_mode["fuzzy"].already_shown_ids)
    assert fuzzy_ids == {"c1", "c2", "c3"}
    assert run.match.id == "c2"


async def test_invalid_mode_is_retried_not_fatal(tool):
    model, _ = scripted_model(
        [
            [("search", {"query": "Acme", "mode": "nope"})],
            [("search", {"query": "Acme", "mode": "contains"})],
            decide("matched", "c2"),
        ]
    )
    run = await Matcher(tool, model=model, config=NO_PREFETCH).match_async("Acme Widgets")
    assert run.match.id == "c2"
    assert [t.mode for t in run.searches] == ["contains"]
    assert run.usage.requests == 3


async def test_no_results_then_widen_then_no_match(tool):
    model, _ = scripted_model(
        [
            [("search", {"query": "Vandelay"})],
            [("search", {"query": "Vand"})],
            decide("no_match", confidence=0.85, reasoning="nothing plausible"),
        ]
    )
    run = await Matcher(tool, model=model, config=NO_PREFETCH).match_async("Vandelay Industries")
    assert run.decision.status == "no_match"
    assert run.match is None
    assert all(t.total_count == 0 for t in run.searches)


async def test_hallucinated_match_id_is_rejected_and_retried(tool):
    model, _ = scripted_model(
        [
            [("search", {"query": "Stark"})],
            decide("matched", "c99"),
            decide("matched", "c8"),
        ]
    )
    run = await Matcher(tool, model=model, config=NO_PREFETCH).match_async("Stark Industries")
    assert run.match.id == "c8"
    assert run.usage.requests == 3


async def test_match_before_any_search_is_rejected(tool):
    model, _ = scripted_model(
        [
            decide("matched", "c8"),
            [("search", {"query": "Stark"})],
            decide("matched", "c8"),
        ]
    )
    run = await Matcher(tool, model=model, config=NO_PREFETCH).match_async("Stark Industries")
    assert run.match.id == "c8"


async def test_retries_exhausted_becomes_inconclusive(tool):
    model, _ = scripted_model(
        [
            [("search", {"query": "Stark"})],
            decide("matched", "bogus1"),
            decide("matched", "bogus2"),
            decide("matched", "bogus3"),
        ]
    )
    run = await Matcher(
        tool, model=model, config=MatchConfig(output_retries=2, prefetch=False)
    ).match_async("Stark Industries")
    assert run.decision.status == "inconclusive"
    assert "valid decision" in run.decision.reasoning
    assert set(run.seen_records) == {"c8"}
    assert run.messages


async def test_ambiguous_requires_alternatives(tool):
    model, _ = scripted_model(
        [
            [("search", {"query": "Acme"})],
            decide("ambiguous", alternatives=[]),
            decide("ambiguous", alternatives=["c1", "c2"], confidence=0.4),
        ]
    )
    run = await Matcher(tool, model=model, config=NO_PREFETCH).match_async("Acme")
    assert run.decision.status == "ambiguous"
    assert [r.id for r in run.alternatives] == ["c1", "c2"]


async def test_unknown_alternative_is_rejected(tool):
    model, _ = scripted_model(
        [
            [("search", {"query": "Acme"})],
            decide("matched", "c1", alternatives=["c2", "nope"]),
            decide("matched", "c1", alternatives=["c2", "c1"]),
        ]
    )
    run = await Matcher(tool, model=model, config=NO_PREFETCH).match_async("Acme Holdings")
    assert run.decision.alternatives == ["c2"]


async def test_no_match_drops_stray_match_id(tool):
    model, _ = scripted_model(
        [
            [("search", {"query": "Acme"})],
            decide("no_match", "c1"),
        ]
    )
    run = await Matcher(tool, model=model, config=NO_PREFETCH).match_async("Acme")
    assert run.decision.match_id is None
    assert run.match is None


async def test_round_budget_exhausted_becomes_inconclusive(tool):
    model, _ = scripted_model(
        [
            [("search", {"query": "Acme"})],
            [("search", {"query": "Globex"})],
            [("search", {"query": "Initech"})],
            decide("matched", "c1"),
        ]
    )
    run = await Matcher(
        tool, model=model, config=MatchConfig(max_rounds=2, prefetch=False)
    ).match_async("x")
    assert run.decision.status == "inconclusive"
    assert "round budget" in run.decision.reasoning
    assert len(run.searches) == 2
    assert set(run.seen_records) == {"c1", "c2", "c3", "c4"}


async def test_search_budget_note_and_repeat_reach_the_model(tool):
    model, _ = scripted_model(
        [
            [("search", {"query": "Acme"})],
            [("search", {"query": "acme"}), ("search", {"query": "Globex"})],
            decide("matched", "c1"),
        ]
    )
    run = await Matcher(
        tool, model=model, config=MatchConfig(max_searches=2, prefetch=False)
    ).match_async("x")
    notes = [t.note for t in run.searches]
    assert notes[1].startswith("Already searched")
    assert "last allowed search" in notes[2]
    assert run.match.id == "c1"


async def test_extra_tools_are_mounted(tool):
    def lookup_ticker(company_id: str) -> str:
        """Return the stock ticker for a record id."""
        return {"c4": "GBX"}.get(company_id, "unknown")

    model, infos = scripted_model(
        [
            [("search", {"query": "Globex"})],
            [("lookup_ticker", {"company_id": "c4"})],
            decide("matched", "c4"),
        ]
    )
    run = await Matcher(
        tool, model=model, config=NO_PREFETCH, extra_tools=[lookup_ticker]
    ).match_async("Globex")
    assert {t.name for t in infos()[0].function_tools} == {"search", "lookup_ticker"}
    assert run.match.id == "c4"
    assert run.usage.tool_calls == 2


async def test_match_many_runs_each_name(tool):
    model, _ = scripted_model(
        [
            [("search", {"query": "Globex"})],
            decide("matched", "c4"),
            [("search", {"query": "Initech"})],
            decide("matched", "c6"),
        ]
    )
    runs = await Matcher(tool, model=model, config=NO_PREFETCH).match_many(
        ["Globex", "Initech"], concurrency=1
    )
    assert [r.match.id for r in runs] == ["c4", "c6"]


def test_sync_wrapper(tool):
    model, _ = scripted_model([[("search", {"query": "Wayne"})], decide("matched", "c9")])
    run = Matcher(tool, model=model, config=NO_PREFETCH).match("Wayne Enterprises")
    assert run.match.id == "c9"


async def test_custom_instructions_template(tool):
    model, infos = scripted_model([decide("no_match")])
    matcher = Matcher(
        tool,
        model=model,
        config=NO_PREFETCH,
        instructions="CUSTOM {max_searches} :: {tool_description} :: {modes}",
    )
    await matcher.match_async("x")
    text = infos()[0].instructions
    assert text.startswith("CUSTOM 8 :: In-memory list of 10 records")
    assert "- fuzzy:" in text
