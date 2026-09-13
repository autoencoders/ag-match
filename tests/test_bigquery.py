from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

pytest.importorskip("google.cloud.bigquery")

from ag_match import MatchConfig, SearchSession, available_modes  # noqa: E402
from ag_match.bigquery import BigQuerySearchTool  # noqa: E402


@dataclass
class FakeJob:
    rows: list[dict[str, Any]]

    def result(self):
        return self.rows


@dataclass
class FakeClient:
    rows: list[dict[str, Any]] = field(default_factory=list)
    calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)

    def query(self, sql, job_config=None):
        params = {p.name: p.value for p in job_config.query_parameters}
        self.calls.append((sql, params))
        return FakeJob(self.rows)


@pytest.fixture
def tool():
    client = FakeClient(
        rows=[
            {
                "id": 7,
                "name": "Acme Holdings International Inc",
                "x_country": "US",
                "total_count": 2,
            },
            {"id": 9, "name": "Acme Widgets Ltd", "x_country": None, "total_count": 2},
        ]
    )
    return BigQuerySearchTool(
        client,
        table="proj.ds.companies",
        id_column="company_id",
        name_column="legal_name",
        extra_columns=["country"],
        alias_columns=["trade_name"],
        where="status = 'active'",
    )


def test_all_modes_available(tool):
    assert available_modes(tool) == ["contains", "all_terms", "fuzzy"]
    assert "proj.ds.companies" in tool.description
    assert "legal_name, trade_name" in tool.description


def test_contains_sql_and_rows(tool):
    result = tool.search("Acmé", 15)
    sql, params = tool.client.calls[-1]
    assert params == {"q": "acme", "limit": 15}
    assert "STRPOS(LOWER(REGEXP_REPLACE(NORMALIZE(COALESCE(legal_name, ''), NFKD)" in sql
    assert "COALESCE(trade_name, '')" in sql
    assert "(status = 'active')" in sql
    assert "COUNT(*) OVER() AS total_count" in sql
    assert "CAST(company_id AS STRING) AS id" in sql
    assert result.total_count == 2
    assert [r.id for r in result.records] == ["7", "9"]
    assert result.records[0].extra == {"country": "US"}
    assert result.records[1].extra == {}


def test_all_terms_sql(tool):
    tool.search_all_terms(["Delta", " Engineering ", ""], 10)
    sql, params = tool.client.calls[-1]
    assert params == {"t0": "delta", "t1": "engineering", "limit": 10}
    assert sql.count("STRPOS(") == 4  # two terms x two name columns
    assert ") AND (" in sql


def test_fuzzy_sql_uses_edit_distance_with_length_bound(tool):
    tool.search_fuzzy("Fabnekam Widgts", 10)
    sql, params = tool.client.calls[-1]
    assert params == {"t0": "fabnekam", "d0": 2, "t1": "widgts", "d1": 2, "limit": 10}
    assert "EDIT_DISTANCE(w, @t0, max_distance => @d0)" in sql
    assert "UNNEST(SPLIT(" in sql
    assert "ORDER BY score" in sql
    assert "dist0 IS NOT NULL AND dist1 IS NOT NULL" in sql


def test_empty_queries_do_not_hit_bigquery(tool):
    assert tool.search("  ", 5).total_count == 0
    assert tool.search_all_terms([" "], 5).total_count == 0
    assert tool.search_fuzzy("", 5).total_count == 0
    assert tool.client.calls == []


async def test_session_drives_bigquery_tool_in_a_thread(tool):
    session = SearchSession(tool, MatchConfig(prefetch=False), query_name="Acme Widgets")
    reply = await session.run("acme", "all_terms")
    assert [r.id for r in reply.records] == ["9", "7"]  # ranked by similarity to the query
    assert tool.client.calls[-1][1]["t0"] == "acme"
