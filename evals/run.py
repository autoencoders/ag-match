"""Run the eval: `uv run python -m evals.run --help`.

Examples:
    uv run python -m evals.run --baseline                  # fuzzy baseline, no LLM
    uv run python -m evals.run --model heuristic           # agent loop with a fake LLM
    uv run python -m evals.run --model gemini --sample 60  # real model on a subset
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ag_match import InMemorySearchTool, MatchConfig, Matcher, MatchRun

from .baseline import fuzzy_decide, heuristic_model
from .dataset import ALL_KINDS, Case, Dataset, build_dataset


@dataclass
class CaseResult:
    case: dict[str, Any]
    status: str
    match_id: str | None
    confidence: float
    alternatives: list[str]
    outcome: str
    correct: bool
    searches: int
    search_queries: list[str]
    too_many: int
    zero_hits: int
    requests: int
    input_tokens: int
    output_tokens: int
    seconds: float
    reasoning: str = ""
    error: str | None = None
    matched_name: str | None = None


def classify(case: Case, status: str, match_id: str | None, alternatives: list[str]) -> str:
    if case.expected_id is None:
        return {
            "no_match": "tn",
            "matched": "fp_spurious",
            "ambiguous": "neg_ambiguous",
        }.get(status, "neg_inconclusive")
    if status == "matched":
        return "tp" if match_id == case.expected_id else "fp_wrong"
    if status == "ambiguous":
        return "fn_ambiguous_hit" if case.expected_id in alternatives else "fn_ambiguous_miss"
    if status == "no_match":
        return "fn_missed"
    return "fn_inconclusive"


def result_from_run(case: Case, run: MatchRun, seconds: float, dataset: Dataset) -> CaseResult:
    d = run.decision
    outcome = classify(case, d.status, d.match_id, d.alternatives)
    executed = [t for t in run.searches if t.executed]
    return CaseResult(
        case=case.to_dict(),
        status=d.status,
        match_id=d.match_id,
        confidence=d.confidence,
        alternatives=list(d.alternatives),
        outcome=outcome,
        correct=outcome in ("tp", "tn"),
        searches=len(executed),
        search_queries=[t.query for t in run.searches],
        too_many=sum(1 for t in executed if (t.note or "").startswith("Too many")),
        zero_hits=sum(1 for t in executed if t.total_count == 0),
        requests=run.usage.requests,
        input_tokens=run.usage.input_tokens,
        output_tokens=run.usage.output_tokens,
        seconds=seconds,
        reasoning=d.reasoning,
        matched_name=dataset.by_id[d.match_id].name if d.match_id in dataset.by_id else None,
    )


async def run_llm(
    dataset: Dataset,
    cases: list[Case],
    *,
    model: Any,
    config: MatchConfig,
    concurrency: int,
    progress: bool,
) -> list[CaseResult]:
    tool = InMemorySearchTool(dataset.records, extra_fields=["country"])
    matcher = Matcher(tool, model=model, config=config)
    sem = asyncio.Semaphore(concurrency)
    done = 0

    async def one(case: Case) -> CaseResult:
        nonlocal done
        async with sem:
            t0 = time.perf_counter()
            try:
                run = await matcher.match_async(case.query, case.context)
                res = result_from_run(case, run, time.perf_counter() - t0, dataset)
            except Exception as exc:  # noqa: BLE001 - keep the eval going
                res = CaseResult(
                    case=case.to_dict(),
                    status="error",
                    match_id=None,
                    confidence=0.0,
                    alternatives=[],
                    outcome="error",
                    correct=False,
                    searches=0,
                    search_queries=[],
                    too_many=0,
                    zero_hits=0,
                    requests=0,
                    input_tokens=0,
                    output_tokens=0,
                    seconds=time.perf_counter() - t0,
                    error=f"{type(exc).__name__}: {exc}",
                )
            done += 1
            if progress and (done % 10 == 0 or done == len(cases)):
                print(f"  {done}/{len(cases)}", file=sys.stderr)
            return res

    return list(await asyncio.gather(*(one(c) for c in cases)))


def run_baseline(dataset: Dataset, cases: list[Case], *, threshold: float) -> list[CaseResult]:
    out = []
    for case in cases:
        t0 = time.perf_counter()
        d = fuzzy_decide(case.query, dataset.records, threshold=threshold)
        outcome = classify(case, d["status"], d["match_id"], d["alternatives"])
        out.append(
            CaseResult(
                case=case.to_dict(),
                status=d["status"],
                match_id=d["match_id"],
                confidence=d["confidence"],
                alternatives=d["alternatives"],
                outcome=outcome,
                correct=outcome in ("tp", "tn"),
                searches=0,
                search_queries=[],
                too_many=0,
                zero_hits=0,
                requests=0,
                input_tokens=0,
                output_tokens=0,
                seconds=time.perf_counter() - t0,
                matched_name=dataset.by_id[d["match_id"]].name if d["match_id"] else None,
            )
        )
    return out


# --- metrics and report -------------------------------------------------------------


def summarize(results: list[CaseResult]) -> dict[str, Any]:
    n = len(results)
    outcomes = Counter(r.outcome for r in results)
    positives = sum(1 for r in results if r.case["expected_id"] is not None)
    negatives = n - positives
    tp = outcomes["tp"]
    fp = outcomes["fp_wrong"] + outcomes["fp_spurious"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / positives if positives else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    llm = [r for r in results if r.requests > 0]

    def mean(xs: list[float]) -> float:
        return statistics.fmean(xs) if xs else 0.0

    def p95(xs: list[float]) -> float:
        if not xs:
            return 0.0
        s = sorted(xs)
        return s[min(len(s) - 1, int(round(0.95 * (len(s) - 1))))]

    per_kind: dict[str, dict[str, Any]] = {}
    by_kind: dict[str, list[CaseResult]] = defaultdict(list)
    for r in results:
        by_kind[r.case["kind"]].append(r)
    for kind in ALL_KINDS:
        rs = by_kind.get(kind)
        if not rs:
            continue
        per_kind[kind] = {
            "n": len(rs),
            "accuracy": sum(r.correct for r in rs) / len(rs),
            "searches": mean([r.searches for r in rs]),
            "outcomes": dict(Counter(r.outcome for r in rs)),
        }

    return {
        "n": n,
        "positives": positives,
        "negatives": negatives,
        "accuracy": sum(r.correct for r in results) / n if n else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "outcomes": dict(outcomes),
        "searches_mean": mean([r.searches for r in llm]),
        "searches_p95": p95([r.searches for r in llm]),
        "searches_max": max((r.searches for r in llm), default=0),
        "requests_mean": mean([r.requests for r in llm]),
        "too_many_total": sum(r.too_many for r in llm),
        "zero_hit_total": sum(r.zero_hits for r in llm),
        "input_tokens_mean": mean([r.input_tokens for r in llm]),
        "output_tokens_mean": mean([r.output_tokens for r in llm]),
        "seconds_mean": mean([r.seconds for r in results]),
        "errors": outcomes["error"],
        "per_kind": per_kind,
    }


def render_report(summary: dict[str, Any], results: list[CaseResult], label: str) -> str:
    s = summary
    lines = [
        f"# ag-match eval: {label}",
        "",
        f"{s['n']} cases ({s['positives']} positive, {s['negatives']} negative)",
        "",
        "| metric | value |",
        "|---|---|",
        f"| accuracy | {s['accuracy']:.3f} |",
        f"| precision | {s['precision']:.3f} |",
        f"| recall | {s['recall']:.3f} |",
        f"| f1 | {s['f1']:.3f} |",
        "| searches per case (mean / p95 / max) | "
        f"{s['searches_mean']:.2f} / {s['searches_p95']:.0f} / {s['searches_max']} |",
        f"| LLM requests per case | {s['requests_mean']:.2f} |",
        "| tokens per case (in / out) | "
        f"{s['input_tokens_mean']:.0f} / {s['output_tokens_mean']:.0f} |",
        f"| too-many replies / zero-hit searches | {s['too_many_total']} / {s['zero_hit_total']} |",
        f"| seconds per case | {s['seconds_mean']:.2f} |",
        f"| errors | {s['errors']} |",
        "",
        "## Outcomes",
        "",
        "| outcome | count |",
        "|---|---|",
    ]
    for k, v in sorted(s["outcomes"].items()):
        lines.append(f"| {k} | {v} |")
    lines += [
        "",
        "## Per kind",
        "",
        "| kind | n | accuracy | searches | outcomes |",
        "|---|---|---|---|---|",
    ]
    for kind, pk in s["per_kind"].items():
        outs = ", ".join(f"{k}={v}" for k, v in sorted(pk["outcomes"].items()))
        lines.append(
            f"| {kind} | {pk['n']} | {pk['accuracy']:.2f} | {pk['searches']:.1f} | {outs} |"
        )

    failures = [r for r in results if not r.correct]
    if failures:
        lines += [
            "",
            f"## Failures ({len(failures)})",
            "",
            "| case | kind | query | expected | got | searches |",
            "|---|---|---|---|---|---|",
        ]
        for r in failures[:80]:
            exp = r.case.get("source_name") or "-"
            got = f"{r.status}" + (f": {r.matched_name}" if r.matched_name else "")
            if r.error:
                got = f"error: {r.error[:60]}"
            q = " → ".join(r.search_queries) or "-"
            lines.append(
                f"| {r.case['id']} | {r.case['kind']} | {r.case['query']} | {exp} | {got} | {q} |"
            )
    return "\n".join(lines) + "\n"


# --- CLI ----------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--model", default="gemini", help="model spec, or 'heuristic' for the offline fake LLM"
    )
    p.add_argument(
        "--baseline", action="store_true", help="run the fuzzy-string baseline instead of an LLM"
    )
    p.add_argument("--baseline-threshold", type=float, default=0.85)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--filler", type=int, default=1500, help="number of filler records")
    p.add_argument("--per-target", type=int, default=3, help="positive cases per target name")
    p.add_argument("--kinds", default=None, help="comma-separated case kinds to include")
    p.add_argument("--sample", type=int, default=None, help="random subset of cases")
    p.add_argument("--limit", type=int, default=None, help="first N cases")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--records-per-search", type=int, default=None)
    p.add_argument("--too-many", type=int, default=None)
    p.add_argument("--max-searches", type=int, default=None)
    p.add_argument("--max-rounds", type=int, default=None)
    p.add_argument("--out-dir", type=Path, default=Path("evals/results"))
    p.add_argument("--label", default=None, help="name for the report files")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    kinds = args.kinds.split(",") if args.kinds else None
    dataset = build_dataset(
        seed=args.seed, filler=args.filler, kinds=kinds, per_target=args.per_target
    )
    cases = dataset.cases
    if args.sample:
        cases = random.Random(args.seed).sample(cases, min(args.sample, len(cases)))
    if args.limit:
        cases = cases[: args.limit]

    overrides = {
        k: v
        for k, v in {
            "records_per_search": args.records_per_search,
            "too_many_threshold": args.too_many,
            "max_searches": args.max_searches,
            "max_rounds": args.max_rounds,
        }.items()
        if v is not None
    }
    config = MatchConfig(**overrides)

    if args.baseline:
        label = args.label or f"baseline-{args.baseline_threshold}"
        results = run_baseline(dataset, cases, threshold=args.baseline_threshold)
    else:
        model: Any = heuristic_model() if args.model == "heuristic" else args.model
        label = args.label or str(args.model).replace(":", "_").replace("/", "_")
        if not args.quiet:
            print(
                f"running {len(cases)} cases with {args.model} (concurrency {args.concurrency})",
                file=sys.stderr,
            )
        results = asyncio.run(
            run_llm(
                dataset,
                cases,
                model=model,
                config=config,
                concurrency=args.concurrency,
                progress=not args.quiet,
            )
        )

    summary = summarize(results)
    report = render_report(summary, results, label)
    print(report)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    base = args.out_dir / f"{stamp}-{label}"
    base.with_suffix(".md").write_text(report, encoding="utf-8")
    base.with_suffix(".json").write_text(
        json.dumps(
            {
                "label": label,
                "args": vars(args) | {"out_dir": str(args.out_dir)},
                "config": config.model_dump(),
                "summary": summary,
                "results": [asdict(r) for r in results],
            },
            indent=1,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )
    if not args.quiet:
        print(f"wrote {base}.md and .json", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
