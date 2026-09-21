"""BigQuery-backed search tool. Requires the `bigquery` extra (google-cloud-bigquery).

All three modes are implemented in SQL over a normalized form of the name columns:
accents stripped, case folded. `contains` and `all_terms` use STRPOS; `fuzzy` uses
BigQuery's EDIT_DISTANCE per word with a small bound. Each call is one query that also
returns the total match count via a window function.

Example:
    from google.cloud import bigquery
    tool = BigQuerySearchTool(
        bigquery.Client(),
        table="proj.dataset.companies",
        id_column="company_id",
        name_column="legal_name",
        extra_columns=["country", "city"],
        alias_columns=["trade_name"],
    )
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any

from .text import default_max_distance, normalize, words
from .types import Record, SearchResult

if TYPE_CHECKING:
    from google.cloud import bigquery


def _norm_sql(column: str) -> str:
    """SQL for the same normalization as `namematch.text.normalize`."""
    return f"LOWER(REGEXP_REPLACE(NORMALIZE(COALESCE({column}, ''), NFKD), r'\\pM', ''))"


class BigQuerySearchTool:
    def __init__(
        self,
        client: bigquery.Client,
        *,
        table: str,
        id_column: str,
        name_column: str,
        extra_columns: Sequence[str] = (),
        alias_columns: Sequence[str] = (),
        description: str | None = None,
        where: str | None = None,
    ) -> None:
        """Args:
        client: A `google.cloud.bigquery.Client`. Sync; the runtime runs it in a thread.
        table: Fully qualified `project.dataset.table` or a backtick-safe view name.
        id_column: Unique id column.
        name_column: Primary name column.
        extra_columns: Small disambiguating columns shown to the LLM (country, city, ...).
        alias_columns: Other name-like columns searched alongside `name_column`.
        description: Overrides the generated tool description shown to the LLM.
        where: Optional extra SQL filter, e.g. "status = 'active'".
        """
        self.client = client
        self.table = table
        self.id_column = id_column
        self.name_column = name_column
        self.extra_columns = tuple(extra_columns)
        self.alias_columns = tuple(alias_columns)
        self.where = where
        searched = ", ".join((name_column, *alias_columns))
        self.description = description or (
            f"BigQuery table {table}; searches cover the columns [{searched}] and the "
            f"extra fields [{', '.join(extra_columns) or 'none'}] are shown for context."
        )

    # --- SQL building -----------------------------------------------------------------

    @property
    def _name_columns(self) -> tuple[str, ...]:
        return (self.name_column, *self.alias_columns)

    def _select(self) -> str:
        cols = ", ".join(
            [f"CAST({self.id_column} AS STRING) AS id", f"{self.name_column} AS name"]
            + [f"{c} AS `x_{c}`" for c in self.extra_columns]
        )
        return f"SELECT {cols}, COUNT(*) OVER() AS total_count"

    def _base_where(self) -> str:
        return f"({self.where})" if self.where else "TRUE"

    def _contains_clause(self, param: str) -> str:
        return " OR ".join(f"STRPOS({_norm_sql(c)}, @{param}) > 0" for c in self._name_columns)

    def _fuzzy_min_distance(self, param: str, bound: str) -> str:
        """SQL for the smallest per-word edit distance over all searched name columns."""
        unions = " UNION ALL ".join(
            f"SELECT w FROM UNNEST(SPLIT({_norm_sql(c)}, ' ')) AS w" for c in self._name_columns
        )
        return (
            f"(SELECT MIN(EDIT_DISTANCE(w, @{param}, max_distance => @{bound})) "
            f"FROM ({unions}) WHERE w != '')"
        )

    def build_contains(self, query: str, limit: int) -> tuple[str, list[Any]]:
        from google.cloud import bigquery

        sql = (
            f"{self._select()} FROM `{self.table}` "
            f"WHERE {self._base_where()} AND ({self._contains_clause('q')}) "
            f"LIMIT @limit"
        )
        params = [
            bigquery.ScalarQueryParameter("q", "STRING", normalize(query)),
            bigquery.ScalarQueryParameter("limit", "INT64", limit),
        ]
        return sql, params

    def build_all_terms(self, terms: Iterable[str], limit: int) -> tuple[str, list[Any]]:
        from google.cloud import bigquery

        clean = [normalize(t) for t in terms if normalize(t)]
        clauses = " AND ".join(f"({self._contains_clause(f't{i}')})" for i in range(len(clean)))
        sql = (
            f"{self._select()} FROM `{self.table}` "
            f"WHERE {self._base_where()} AND {clauses or 'FALSE'} "
            f"LIMIT @limit"
        )
        params = [bigquery.ScalarQueryParameter(f"t{i}", "STRING", t) for i, t in enumerate(clean)]
        params.append(bigquery.ScalarQueryParameter("limit", "INT64", limit))
        return sql, params

    def build_fuzzy(self, query: str, limit: int) -> tuple[str, list[Any]]:
        from google.cloud import bigquery

        terms = words(query)
        dist_exprs = [self._fuzzy_min_distance(f"t{i}", f"d{i}") for i in range(len(terms))]
        inner_cols = ", ".join(f"{e} AS dist{i}" for i, e in enumerate(dist_exprs)) or "0 AS dist0"
        conds = " AND ".join(f"dist{i} IS NOT NULL" for i in range(len(terms))) or "FALSE"
        score = " + ".join(f"dist{i}" for i in range(len(terms))) or "0"
        sql = (
            f"WITH scored AS (SELECT *, {inner_cols} FROM `{self.table}` "
            f"WHERE {self._base_where()}) "
            f"{self._select()}, ({score}) AS score FROM scored "
            f"WHERE {conds} ORDER BY score LIMIT @limit"
        )
        params: list[Any] = []
        for i, t in enumerate(terms):
            params.append(bigquery.ScalarQueryParameter(f"t{i}", "STRING", t))
            params.append(bigquery.ScalarQueryParameter(f"d{i}", "INT64", default_max_distance(t)))
        params.append(bigquery.ScalarQueryParameter("limit", "INT64", limit))
        return sql, params

    # --- execution --------------------------------------------------------------------

    def _run(self, sql: str, params: list[Any]) -> SearchResult:
        from google.cloud import bigquery

        job = self.client.query(sql, job_config=bigquery.QueryJobConfig(query_parameters=params))
        rows = list(job.result())
        total = int(rows[0]["total_count"]) if rows else 0
        records = [
            Record(
                id=str(r["id"]),
                name=str(r["name"] or ""),
                extra={c: r[f"x_{c}"] for c in self.extra_columns if r[f"x_{c}"] is not None},
            )
            for r in rows
        ]
        return SearchResult(total_count=total, records=records)

    def search(self, query: str, limit: int) -> SearchResult:
        if not normalize(query):
            return SearchResult(total_count=0)
        return self._run(*self.build_contains(query, limit))

    def search_all_terms(self, terms: list[str], limit: int) -> SearchResult:
        if not any(normalize(t) for t in terms):
            return SearchResult(total_count=0)
        return self._run(*self.build_all_terms(terms, limit))

    def search_fuzzy(self, query: str, limit: int) -> SearchResult:
        if not words(query):
            return SearchResult(total_count=0)
        return self._run(*self.build_fuzzy(query, limit))
