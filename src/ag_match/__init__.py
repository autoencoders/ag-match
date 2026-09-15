"""ag-match: LLM-driven name matching over a runtime-mounted search tool."""

from .matcher import Matcher, RunDeps
from .models import DEFAULT_MODEL, GoogleCloudAuth, resolve_model
from .text import distinctive_words, normalize, similarity
from .tools import InMemorySearchTool, SearchSession, SearchTool, available_modes, normalize_query
from .types import (
    MatchConfig,
    MatchDecision,
    MatchRun,
    Record,
    SearchMode,
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
    "SearchMode",
    "SearchTool",
    "SearchTrace",
    "Usage",
    "available_modes",
    "distinctive_words",
    "normalize",
    "normalize_query",
    "resolve_model",
    "similarity",
]
