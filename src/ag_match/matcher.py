"""The matching agent: wires a mounted `SearchTool` into a pydantic-ai agent loop."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from pydantic_ai import Agent, ModelRetry, RunContext, Tool
from pydantic_ai.exceptions import AgentRunError, UnexpectedModelBehavior, UsageLimitExceeded
from pydantic_ai.models import Model
from pydantic_ai.usage import UsageLimits

from .models import GoogleCloudAuth, resolve_model
from .prompts import build_instructions, build_user_prompt
from .tools import SearchSession, SearchTool
from .types import MatchConfig, MatchDecision, MatchRun, SearchReply, Usage

SEARCH_TOOL_DESCRIPTION = """\
Search the list for records containing an exact string. Returns the total match count, \
the matching records not shown before, ids of matching records already shown, and a note \
from the runtime. Above the too-many threshold no records are returned: narrow the string.

Backend semantics:
{backend}
"""


@dataclass
class RunDeps:
    session: SearchSession


class _LLMDecision(MatchDecision):
    """Model-facing output schema. `inconclusive` is reserved for the runtime."""

    status: Literal["matched", "no_match", "ambiguous"]  # type: ignore[assignment]


class Matcher:
    """Matches names against a list reachable through a runtime-mounted `SearchTool`.

    Args:
        search: The search backend. Its `description` is shown to the LLM.
        model: pydantic-ai model instance or spec string. Defaults to `DEFAULT_MODEL`,
            currently Gemini 3.8 Flash.
        config: Budgets and shaping thresholds.
        google_cloud: Vertex AI credentials (service account, explicit credentials or
            ADC settings) for `google-cloud:` and `gemini:` model specs.
        extra_tools: Additional tools exposed to the LLM: plain typed callables or
            pydantic-ai `Tool` objects.
        instructions: Replace the default system instructions. Receives the same
            `str.format` fields as `prompts.INSTRUCTIONS`.
    """

    def __init__(
        self,
        search: SearchTool,
        *,
        model: str | Model | None = None,
        config: MatchConfig | None = None,
        google_cloud: GoogleCloudAuth | None = None,
        extra_tools: Sequence[Tool[RunDeps] | Callable[..., Any]] = (),
        instructions: str | None = None,
    ) -> None:
        self.search_tool = search
        self.config = config or MatchConfig()
        self.model = resolve_model(model, google_cloud=google_cloud)
        if instructions is None:
            text = build_instructions(search.description, self.config)
        else:
            text = instructions.format(
                tool_description=search.description.strip(),
                **self.config.model_dump(),
            )

        search_tool = Tool(
            _search,
            name="search",
            description=SEARCH_TOOL_DESCRIPTION.format(backend=search.description.strip()),
            max_retries=0,
        )
        self._agent: Agent[RunDeps, _LLMDecision] = Agent(
            self.model,
            deps_type=RunDeps,
            output_type=_LLMDecision,
            instructions=text,
            tools=[search_tool, *extra_tools],
            retries=self.config.output_retries,
        )
        self._agent.output_validator(_validate_decision)

    async def match_async(self, name: str, context: dict[str, Any] | None = None) -> MatchRun:
        session = SearchSession(self.search_tool, self.config)
        deps = RunDeps(session=session)
        limits = UsageLimits(request_limit=self.config.max_rounds)
        decision: MatchDecision | None = None
        failure: str | None = None

        async with self._agent.iter(
            build_user_prompt(name, context), deps=deps, usage_limits=limits
        ) as run:
            try:
                async for _node in run:
                    pass
            except UsageLimitExceeded:
                failure = f"LLM round budget exhausted ({self.config.max_rounds} requests)."
            except UnexpectedModelBehavior as exc:
                failure = f"Model did not produce a valid decision: {exc}"
            except AgentRunError as exc:
                failure = f"Agent run failed: {exc}"
            if run.result is not None:
                decision = MatchDecision.model_validate(run.result.output.model_dump())
            messages = run.all_messages()
            raw_usage = run.usage

        if decision is None:
            decision = MatchDecision(
                status="inconclusive",
                match_id=None,
                confidence=0.0,
                reasoning=failure or "Run ended without a decision.",
            )

        return MatchRun(
            query=name,
            context=dict(context or {}),
            decision=decision,
            searches=list(session.trace),
            seen_records=dict(session.seen),
            usage=Usage(
                requests=raw_usage.requests,
                tool_calls=raw_usage.tool_calls,
                input_tokens=raw_usage.input_tokens,
                output_tokens=raw_usage.output_tokens,
            ),
            messages=list(messages),
        )

    def match(self, name: str, context: dict[str, Any] | None = None) -> MatchRun:
        """Sync wrapper. Use `match_async` from inside a running event loop."""
        return asyncio.run(self.match_async(name, context))

    async def match_many(
        self,
        names: Sequence[str],
        contexts: Sequence[dict[str, Any] | None] | None = None,
        *,
        concurrency: int = 4,
    ) -> list[MatchRun]:
        if contexts is not None and len(contexts) != len(names):
            raise ValueError("contexts must be the same length as names")
        sem = asyncio.Semaphore(concurrency)

        async def one(i: int) -> MatchRun:
            async with sem:
                return await self.match_async(names[i], contexts[i] if contexts else None)

        return list(await asyncio.gather(*(one(i) for i in range(len(names)))))


async def _search(ctx: RunContext[RunDeps], query: str) -> dict[str, Any]:
    """Search the list for records containing this exact string."""
    reply: SearchReply = await ctx.deps.session.run(query)
    # Drop empty lists and nulls so the model reads fewer tokens per search.
    return reply.model_dump(exclude_none=True, exclude_defaults=True)


def _validate_decision(ctx: RunContext[RunDeps], decision: _LLMDecision) -> _LLMDecision:
    seen = ctx.deps.session.seen
    valid = ", ".join(seen) or "(none shown yet)"
    if decision.status == "matched":
        if not decision.match_id:
            raise ModelRetry("status is matched but match_id is missing.")
        if decision.match_id not in seen:
            raise ModelRetry(
                f"match_id {decision.match_id!r} was never returned by search. "
                f"Valid ids: {valid}. Pick one of them or change the status."
            )
    elif decision.match_id is not None:
        decision.match_id = None
    unknown = [a for a in decision.alternatives if a not in seen]
    if unknown:
        raise ModelRetry(
            f"alternatives contain ids never returned by search: {unknown}. Valid ids: {valid}."
        )
    if decision.match_id in decision.alternatives:
        decision.alternatives = [a for a in decision.alternatives if a != decision.match_id]
    if decision.status == "ambiguous" and not decision.alternatives:
        raise ModelRetry("status is ambiguous but alternatives is empty. List the candidates.")
    return decision
