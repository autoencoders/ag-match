from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from namematch import InMemorySearchTool, Record

COMPANIES = [
    Record(id="c1", name="Acme Holdings International Inc", extra={"country": "US"}),
    Record(id="c2", name="Acme Widgets Ltd", extra={"country": "GB"}),
    Record(id="c3", name="Acmé Software GmbH", extra={"country": "DE"}),
    Record(id="c4", name="Globex Corporation", extra={"country": "US"}),
    Record(id="c5", name="International Business Machines Corp", extra={"country": "US"}),
    Record(id="c6", name="Initech LLC", extra={"country": "US"}),
    Record(id="c7", name="Umbrella Group Holdings", extra={"country": "GB"}),
    Record(id="c8", name="Stark Industries Group", extra={"country": "US"}),
    Record(id="c9", name="Wayne Enterprises Holdings", extra={"country": "US"}),
    Record(id="c10", name="Sirius Cybernetics Group", extra={"country": "GB"}),
]


@pytest.fixture
def companies() -> list[Record]:
    return list(COMPANIES)


@pytest.fixture
def tool(companies: list[Record]) -> InMemorySearchTool:
    return InMemorySearchTool(companies, extra_fields=["country"])


Step = list[tuple[str, dict[str, Any]]] | dict[str, Any] | str
"""One scripted LLM turn: a list of (tool_name, args) calls, a final-output dict, or text."""


def scripted_model(steps: list[Step]) -> tuple[FunctionModel, Callable[[], list[AgentInfo]]]:
    """A FunctionModel that replays `steps` one per request. Also returns the seen AgentInfos."""
    queue = list(steps)
    infos: list[AgentInfo] = []

    def fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        infos.append(info)
        if not queue:
            raise AssertionError("scripted model ran out of steps")
        step = queue.pop(0)
        if isinstance(step, str):
            return ModelResponse(parts=[TextPart(step)])
        if isinstance(step, dict):
            output_tool = info.output_tools[0].name
            return ModelResponse(parts=[ToolCallPart(output_tool, step)])
        return ModelResponse(parts=[ToolCallPart(name, args) for name, args in step])

    return FunctionModel(fn), lambda: infos
