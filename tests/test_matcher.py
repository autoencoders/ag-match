from __future__ import annotations

from conftest import scripted_model

from ag_match import MatchConfig, Matcher


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
    run = await Matcher(tool, model=model).match_async(
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
    assert "substring" in search_def.description
    assert "substring" in first.instructions


async def test_parallel_searches_in_one_turn(tool):
    model, _ = scripted_model(
        [
            [("search", {"query": "Globex"}), ("search", {"query": "Initech"})],
            decide("matched", "c4"),
        ]
    )
    run = await Matcher(tool, model=model).match_async("Globex Corp")
    assert [t.query for t in run.searches] == ["Globex", "Initech"]
    assert run.usage.requests == 2


async def test_too_many_then_narrow(tool):
    config = MatchConfig(records_per_search=2, too_many_threshold=2)
    model, _ = scripted_model(
        [
            [("search", {"query": "Acme"})],
            [("search", {"query": "Acme Widgets"})],
            decide("matched", "c2"),
        ]
    )
    run = await Matcher(tool, model=model, config=config).match_async("Acme Widgets")
    assert run.searches[0].shown_ids == []
    assert "Too many" in run.searches[0].note
    assert run.searches[1].shown_ids == ["c2"]
    assert run.match.id == "c2"


async def test_no_results_then_widen_then_no_match(tool):
    model, _ = scripted_model(
        [
            [("search", {"query": "Vandelay"})],
            [("search", {"query": "Vand"})],
            decide("no_match", confidence=0.85, reasoning="nothing plausible"),
        ]
    )
    run = await Matcher(tool, model=model).match_async("Vandelay Industries")
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
    run = await Matcher(tool, model=model).match_async("Stark Industries")
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
    run = await Matcher(tool, model=model).match_async("Stark Industries")
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
    run = await Matcher(tool, model=model, config=MatchConfig(output_retries=2)).match_async(
        "Stark Industries"
    )
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
    run = await Matcher(tool, model=model).match_async("Acme")
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
    run = await Matcher(tool, model=model).match_async("Acme Holdings")
    assert run.decision.alternatives == ["c2"]


async def test_no_match_drops_stray_match_id(tool):
    model, _ = scripted_model(
        [
            [("search", {"query": "Acme"})],
            decide("no_match", "c1"),
        ]
    )
    run = await Matcher(tool, model=model).match_async("Acme")
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
    run = await Matcher(tool, model=model, config=MatchConfig(max_rounds=2)).match_async("x")
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
    run = await Matcher(tool, model=model, config=MatchConfig(max_searches=2)).match_async("x")
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
    run = await Matcher(tool, model=model, extra_tools=[lookup_ticker]).match_async("Globex")
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
    runs = await Matcher(tool, model=model).match_many(["Globex", "Initech"], concurrency=1)
    assert [r.match.id for r in runs] == ["c4", "c6"]


def test_sync_wrapper(tool):
    model, _ = scripted_model([[("search", {"query": "Wayne"})], decide("matched", "c9")])
    run = Matcher(tool, model=model).match("Wayne Enterprises")
    assert run.match.id == "c9"


async def test_custom_instructions_template(tool):
    model, infos = scripted_model([decide("no_match")])
    matcher = Matcher(tool, model=model, instructions="CUSTOM {max_searches} :: {tool_description}")
    await matcher.match_async("x")
    assert infos()[0].instructions.startswith("CUSTOM 8 :: Case-")
