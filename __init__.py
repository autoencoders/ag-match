"""ag-match: LLM-driven name matching over a runtime-mounted search tool."""

from .matcher import Matcher, RunDeps
from .models import DEFAULT_MODEL, GoogleCloudAuth, resolve_model
from .tools import InMemorySearchTool, SearchSession, SearchTool, normalize_query
from .types import (
    MatchConfig,
    MatchDecision,
    MatchRun,
    Record,
    SearchReply,
    SearchResult,
    SearchTrace,
    Usage,
)

__all__ = [
    "DEFAULT_MODEL",
    "GoogleCloudAuth",
    "InMemorySearchTool",
    "MatchConfig",
    "MatchDecision",
    "MatchRun",
    "Matcher",
    "Record",
    "RunDeps",
    "SearchReply",
    "SearchResult",
    "SearchSession",
    "SearchTool",
    "SearchTrace",
    "Usage",
    "normalize_query",
    "resolve_model",
]
