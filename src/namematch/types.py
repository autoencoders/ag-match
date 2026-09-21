"""Public data types: records, search results, decisions, run traces, config."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

Scalar = str | int | float | bool | None

SearchMode = Literal["contains", "all_terms", "fuzzy"]
"""How a search string is interpreted by the backend.

contains: the whole string appears inside a name (case and accent insensitive).
all_terms: every whitespace-separated term appears somewhere in the name, any order.
fuzzy: every term matches some word of the name within a small edit distance.
"""


class Record(BaseModel):
    """One entry from the list being matched against.

    `name` is the primary matching field. `extra` carries small disambiguating
    fields (country, ticker, city, ...). Keep it small: everything here is shown
    to the LLM verbatim.
    """

    id: str
    name: str
    extra: dict[str, Scalar] = Field(default_factory=dict)


class SearchResult(BaseModel):
    """What a `SearchTool` returns for one query.

    `total_count` is the number of records matching the query in the whole list,
    not just the ones returned. `records` holds at most the requested `limit`.
    """

    total_count: int = Field(ge=0)
    records: list[Record] = Field(default_factory=list)


class SearchReply(BaseModel):
    """What the LLM sees after a search. Built by `SearchSession`, not by tools."""

    query: str
    mode: SearchMode = "contains"
    total_count: int
    records: list[Record] = Field(
        default_factory=list, description="Matching records not shown in an earlier search."
    )
    already_shown_ids: list[str] = Field(
        default_factory=list,
        description="Ids of matching records that an earlier search already returned.",
    )
    note: str | None = Field(
        default=None, description="Guidance from the runtime, e.g. too many results, narrow it."
    )


class MatchConfig(BaseModel):
    """Budgets and shaping thresholds. All enforced by the runtime."""

    records_per_search: int = Field(default=15, ge=1, description="Max records shown per search.")
    too_many_threshold: int = Field(
        default=60,
        ge=1,
        description="Above this count no records are shown; the LLM is asked to narrow.",
    )
    max_searches: int = Field(default=8, ge=1, description="Search tool calls per match run.")
    max_rounds: int = Field(default=6, ge=1, description="LLM requests per match run.")
    output_retries: int = Field(
        default=2, ge=0, description="Retries when the final decision fails validation."
    )
    fetch_limit: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Rows fetched from the backend per search before ranking by similarity to the "
            "query and cutting to records_per_search. Defaults to too_many_threshold."
        ),
    )
    show_on_too_many: bool = Field(
        default=True,
        description=(
            "Above the threshold, still show the most similar fetched rows alongside the "
            "narrow-it note instead of nothing."
        ),
    )
    prefetch: bool = Field(
        default=True,
        description=(
            "Before the first LLM turn, search each distinctive word of the query (and run a "
            "fuzzy search when the backend supports it) and show the candidates up front."
        ),
    )
    prefetch_terms: int = Field(
        default=3, ge=0, description="Max distinctive words searched during prefetch."
    )
    temperature: float | None = Field(default=0.0, description="Sampling temperature.")

    @model_validator(mode="after")
    def _limits_consistent(self) -> MatchConfig:
        if self.too_many_threshold < self.records_per_search:
            raise ValueError("too_many_threshold must be >= records_per_search")
        if self.fetch_limit is not None and self.fetch_limit < self.records_per_search:
            raise ValueError("fetch_limit must be >= records_per_search")
        return self

    @property
    def effective_fetch_limit(self) -> int:
        return self.fetch_limit or self.too_many_threshold


MatchStatus = Literal["matched", "no_match", "ambiguous", "inconclusive"]


class MatchDecision(BaseModel):
    """Final verdict for one query name. `reasoning` comes first on purpose: the model
    writes its evidence before it commits to a status and an id."""

    reasoning: str = Field(description="One or two sentences. Name the evidence.")
    status: MatchStatus = Field(
        description=(
            "matched: exactly one record is the same entity. "
            "no_match: searched adequately, nothing matches. "
            "ambiguous: more than one shown record could be it; list them in alternatives."
        )
    )
    match_id: str | None = Field(
        default=None, description="Id of the matched record. Required when status is matched."
    )
    confidence: float = Field(ge=0.0, le=1.0, description="0 to 1.")
    alternatives: list[str] = Field(
        default_factory=list,
        description="Ids of other shown records that could plausibly be the match, best first.",
    )


class SearchTrace(BaseModel):
    query: str
    mode: SearchMode = "contains"
    normalized: str
    total_count: int
    shown_ids: list[str]
    already_shown_ids: list[str]
    note: str | None
    executed: bool = Field(description="False when short-circuited (repeat or budget).")
    source: Literal["agent", "prefetch"] = "agent"


class Usage(BaseModel):
    requests: int = 0
    tool_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


class MatchRun(BaseModel):
    """Everything about one match: the decision plus a full trace."""

    query: str
    context: dict[str, Any] = Field(default_factory=dict)
    decision: MatchDecision
    searches: list[SearchTrace] = Field(default_factory=list)
    seen_records: dict[str, Record] = Field(default_factory=dict)
    usage: Usage = Field(default_factory=Usage)
    messages: list[Any] = Field(default_factory=list, exclude=True, repr=False)

    @property
    def match(self) -> Record | None:
        if self.decision.status == "matched" and self.decision.match_id:
            return self.seen_records.get(self.decision.match_id)
        return None

    @property
    def alternatives(self) -> list[Record]:
        return [self.seen_records[i] for i in self.decision.alternatives if i in self.seen_records]
