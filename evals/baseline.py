"""Offline references: a fuzzy-string baseline and a heuristic fake LLM.

The baseline answers "does the agent beat plain fuzzy matching". The heuristic model
drives the real agent loop and tools without an API, so the harness can be exercised
end to end offline. Both are lower bounds, not targets.
"""

from __future__ import annotations

import difflib
import json
import re
from collections.abc import Iterable
from typing import Any

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

from ag_match import Record

from .dataset import distinctive_tokens, strip_accents


def normalize(name: str) -> str:
    text = strip_accents(name).casefold()
    text = re.sub(r"[^\w\s&]", " ", text)
    return " ".join(text.split())


def core(name: str) -> str:
    return normalize(" ".join(distinctive_tokens(name))) or normalize(name)


def score(query: str, name: str) -> float:
    a, b = normalize(query), normalize(name)
    full = difflib.SequenceMatcher(None, a, b)
    if full.real_quick_ratio() < 0.5 or full.quick_ratio() < 0.5:
        best = 0.0
    else:
        best = full.ratio()
    qa, qb = core(query), core(name)
    if qa and qb:
        best = max(best, difflib.SequenceMatcher(None, qa, qb).ratio())
    return best


def rank(query: str, records: Iterable[Record]) -> list[tuple[float, Record]]:
    scored = [(score(query, r.name), r) for r in records]
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored


def fuzzy_decide(
    query: str, records: Iterable[Record], *, threshold: float = 0.85, margin: float = 0.03
) -> dict[str, Any]:
    ranked = rank(query, records)
    if not ranked or ranked[0][0] < threshold:
        return {
            "status": "no_match",
            "match_id": None,
            "confidence": 1 - (ranked[0][0] if ranked else 0),
            "alternatives": [],
        }
    best, second = ranked[0], ranked[1] if len(ranked) > 1 else (0.0, None)
    if second[1] is not None and best[0] - second[0] < margin and second[0] >= threshold:
        ties = [r.id for s, r in ranked if best[0] - s < margin and s >= threshold]
        return {
            "status": "ambiguous",
            "match_id": None,
            "confidence": best[0],
            "alternatives": ties,
        }
    return {"status": "matched", "match_id": best[1].id, "confidence": best[0], "alternatives": []}


# --- heuristic fake LLM -------------------------------------------------------------


def _query_from(messages: list[ModelMessage]) -> str:
    for m in messages:
        if isinstance(m, ModelRequest):
            for p in m.parts:
                if isinstance(p, UserPromptPart) and isinstance(p.content, str):
                    for line in p.content.splitlines():
                        if line.startswith("QUERY: "):
                            return line[len("QUERY: ") :]
    return ""


_CANDIDATE = re.compile(r"^- \[(?P<id>[^\]]+)\] (?P<name>.*?)(?: \((?P<extra>[^)]*)\))?$")
_PREFETCH = re.compile(r"^- (?P<mode>\w+) '(?P<query>.*)': ")


def _search_returns(messages: list[ModelMessage]) -> tuple[list[str], list[Record], int]:
    """(queries searched, records seen, count of too-many replies), prefetch included."""
    queries: list[str] = []
    seen: dict[str, Record] = {}
    too_many = 0
    for m in messages:
        if isinstance(m, ModelRequest):
            for p in m.parts:
                if isinstance(p, UserPromptPart) and isinstance(p.content, str):
                    for line in p.content.splitlines():
                        if cm := _CANDIDATE.match(line):
                            seen[cm["id"]] = Record(id=cm["id"], name=cm["name"])
                        elif pm := _PREFETCH.match(line):
                            queries.append(pm["query"])
        if isinstance(m, ModelResponse):
            for p in m.parts:
                if isinstance(p, ToolCallPart) and p.tool_name == "search":
                    args = p.args if isinstance(p.args, dict) else json.loads(p.args or "{}")
                    queries.append(str(args.get("query", "")))
        elif isinstance(m, ModelRequest):
            for p in m.parts:
                if isinstance(p, ToolReturnPart) and p.tool_name == "search":
                    content = p.content
                    if isinstance(content, str):
                        content = json.loads(content)
                    if isinstance(content, dict):
                        for r in content.get("records", []):
                            rec = Record.model_validate(r)
                            seen[rec.id] = rec
                        if str(content.get("note", "")).startswith("Too many"):
                            too_many += 1
    return queries, list(seen.values()), too_many


def heuristic_model(*, threshold: float = 0.85, max_turns: int = 4) -> FunctionModel:
    """Searches distinctive tokens, then their stems, then decides by fuzzy score."""

    def fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        query = _query_from(messages)
        searched, seen, _ = _search_returns(messages)
        output_tool = info.output_tools[0].name
        dist = sorted(distinctive_tokens(query), key=len, reverse=True)
        candidates = [strip_accents(t.strip(".,'")) for t in dist[:2]]
        stems = [c[:5] for c in candidates if len(c) > 5]
        plan = [
            c
            for c in candidates + stems
            if c and c.casefold() not in {s.casefold() for s in searched}
        ]

        turn = sum(1 for m in messages if isinstance(m, ModelResponse)) + 1
        decision = fuzzy_decide(query, seen, threshold=threshold)
        if decision["status"] == "matched" or not plan or turn >= max_turns:
            if not seen and not plan and turn < max_turns and not searched:
                return ModelResponse(parts=[ToolCallPart("search", {"query": query.split()[0]})])
            decision["reasoning"] = f"fuzzy score {decision['confidence']:.2f}"
            return ModelResponse(parts=[ToolCallPart(output_tool, decision)])
        batch = plan[:2] if turn == 1 else plan[:1]
        return ModelResponse(parts=[ToolCallPart("search", {"query": q}) for q in batch])

    return FunctionModel(fn, model_name="heuristic")
