from __future__ import annotations

from conftest import scripted_model

from ag_match import InMemorySearchTool, MatchConfig, Matcher
from evals.baseline import fuzzy_decide, heuristic_model, rank, score
from evals.dataset import ALL_KINDS, NEGATIVE_KINDS, build_dataset, perturb
from evals.run import classify, result_from_run, run_baseline, summarize


def test_dataset_is_deterministic_and_well_formed():
    a = build_dataset(seed=3, filler=50)
    b = build_dataset(seed=3, filler=50)
    assert [c.query for c in a.cases] == [c.query for c in b.cases]
    ids = {r.id for r in a.records}
    assert len(ids) == len(a.records)
    for c in a.cases:
        assert c.kind in ALL_KINDS
        if c.expected_id is not None:
            assert c.expected_id in ids
        else:
            assert c.kind in NEGATIVE_KINDS
    kinds = {c.kind for c in a.cases}
    assert {"typo", "suffix", "abbrev", "absent", "decoy", "generic_only"} <= kinds


def test_negatives_are_absent_from_the_list():
    ds = build_dataset(seed=1, filler=100)
    names = {r.name.casefold() for r in ds.records}
    for c in ds.cases:
        if c.kind == "absent":
            assert c.query.casefold() not in names
            absent_word = c.query.split()[0].casefold()
            assert not any(absent_word in n for n in names)


def test_perturbations_change_the_name():
    import random

    rng = random.Random(0)
    name = "Acme Holdings International Inc"
    for kind in ("typo", "two_typos", "suffix", "abbrev", "drop_word", "case_punct", "combo"):
        assert perturb(name, kind, rng) != name, kind
    assert perturb(name, "exact", rng) == name
    assert perturb("Nestlé SA", "accent", rng) == "Nestle SA"


def test_fuzzy_baseline_scores_and_decides():
    ds = build_dataset(seed=0, filler=200)
    acme = next(r for r in ds.records if r.name == "Acme Holdings International Inc")
    assert score("Acmee Holdngs Intl", acme.name) > 0.6
    assert rank("Acmee Holdngs Intl Inc", ds.records)[0][1].id == acme.id
    d = fuzzy_decide("Acmee Holdngs Intl Inc", ds.records, threshold=0.75)
    assert d["status"] == "matched" and d["match_id"] == acme.id
    assert fuzzy_decide("Xanthippe Freight Global Ltd", ds.records)["status"] == "no_match"


def test_classify_outcomes():
    from evals.dataset import Case

    pos = Case(id="p", query="q", kind="typo", expected_id="t1")
    neg = Case(id="n", query="q", kind="absent", expected_id=None)
    assert classify(pos, "matched", "t1", []) == "tp"
    assert classify(pos, "matched", "t2", []) == "fp_wrong"
    assert classify(pos, "no_match", None, []) == "fn_missed"
    assert classify(pos, "ambiguous", None, ["t1"]) == "fn_ambiguous_hit"
    assert classify(pos, "inconclusive", None, []) == "fn_inconclusive"
    assert classify(neg, "no_match", None, []) == "tn"
    assert classify(neg, "matched", "t1", []) == "fp_spurious"
    assert classify(neg, "ambiguous", None, ["t1"]) == "neg_ambiguous"


async def test_result_from_run_and_summary_with_scripted_model():
    ds = build_dataset(seed=0, filler=30, per_target=1, kinds=["exact", "absent"])
    tool = InMemorySearchTool(ds.records, extra_fields=["country"])
    case = next(c for c in ds.cases if c.expected_id and "Globex" in c.query)
    model, _ = scripted_model(
        [
            [("search", {"query": "Globex"})],
            {
                "status": "matched",
                "match_id": case.expected_id,
                "confidence": 0.95,
                "reasoning": "r",
            },
        ]
    )
    run = await Matcher(tool, model=model, config=MatchConfig(prefetch=False)).match_async(
        case.query
    )
    res = result_from_run(case, run, 0.1, ds)
    assert res.outcome == "tp" and res.correct and res.searches == 1
    assert res.matched_name == "Globex Corporation"

    base = run_baseline(ds, ds.cases[:20], threshold=0.85)
    summary = summarize(base + [res])
    assert summary["n"] == 21
    assert 0 <= summary["precision"] <= 1 and 0 <= summary["recall"] <= 1
    assert "exact" in summary["per_kind"]


async def test_heuristic_model_drives_the_agent_loop():
    ds = build_dataset(seed=0, filler=100, per_target=1, kinds=["typo"])
    tool = InMemorySearchTool(ds.records, extra_fields=["country"])
    matcher = Matcher(tool, model=heuristic_model())
    run = await matcher.match_async("Globex Corporaton")
    assert run.decision.status == "matched"
    assert run.match.name == "Globex Corporation"
    assert run.usage.tool_calls <= 4
    absent = await matcher.match_async("Xanthippe Freight Global Ltd")
    assert absent.decision.status == "no_match"
