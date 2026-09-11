"""Public data types: records, search results, decisions, run traces, config."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

Scalar = str | int | float | bool | None


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

    @model_validator(mode="after")
    def _threshold_above_cap(self) -> MatchConfig:
        if self.too_many_threshold < self.records_per_search:
            raise ValueError("too_many_threshold must be >= records_per_search")
        return self


MatchStatus = Literal["matched", "no_match", "ambiguous", "inconclusive"]


class MatchDecision(BaseModel):
    """Final verdict for one query name."""

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
    reasoning: str = Field(description="One or two sentences. Name the evidence.")
    alternatives: list[str] = Field(
        default_factory=list,
        description="Ids of other shown records that could plausibly be the match, best first.",
    )


class SearchTrace(BaseModel):
    query: str
    normalized: str
    total_count: int
    shown_ids: list[str]
    already_shown_ids: list[str]
    note: str | None
    executed: bool = Field(description="False when short-circuited (repeat or budget).")


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
