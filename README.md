# ag-match

LLM-driven name matching. Give it a name and a search tool over your list, and an agent
figures out which record, if any, is the same entity. It copes with misspellings,
abbreviations, legal-suffix noise, reordered words and transliterations where plain
string matching fails.

How it works: before the LLM's first turn the runtime searches each distinctive word of
the query (and runs a fuzzy search when the backend supports it) and presents the
candidates. If none is clearly right, the LLM proposes further short, distinctive
strings, the runtime runs them through the tool you mounted, and the LLM narrows,
widens or decides. The runtime enforces all budgets and shaping: fetched rows are ranked
by similarity to the query, only a capped number are shown per search, and the LLM is
told to narrow when a string matches too much. See [docs/PLAN.md](docs/PLAN.md) for the
design.

## Install

```sh
uv sync
export GOOGLE_API_KEY=...   # Gemini is the default model
```

## Use

```python
from ag_match import InMemorySearchTool, Matcher, Record

records = [
    Record(id="c1", name="Acme Holdings International Inc", extra={"country": "US"}),
    Record(id="c2", name="Acme Widgets Ltd", extra={"country": "GB"}),
]
matcher = Matcher(InMemorySearchTool(records, extra_fields=["country"]))

run = matcher.match("Acmee Holdngs Intl.", context={"country": "US"})
run.decision        # status, match_id, confidence, reasoning, alternatives
run.match           # the Record, or None
run.searches        # every search string, its count, what was shown, runtime notes
run.usage           # requests, tool calls, tokens
```

`match_async` is the primary API; `match` is a sync wrapper; `match_many` runs a batch
with bounded concurrency.

Pick a model with a pydantic-ai spec string: `model="google:gemini-3.8-flash"`,
`"openai:gpt-5"`, `"anthropic:claude-sonnet-4-5"`. `"gemini"` means the default, currently
`google:gemini-3.8-flash`.

### Google: API key or service account

Two routes reach Gemini:

- `google:<model>` is the Gemini Developer API with an API key in `GOOGLE_API_KEY`.
- `google-cloud:<model>` (alias `vertex:<model>`) is Vertex AI, which takes a service
  account or any Google Cloud credential.

```python
from ag_match import GoogleCloudAuth, Matcher

# Service-account key file. Project comes from the file, location defaults to us-central1.
matcher = Matcher(
    tool,
    model="google-cloud:gemini-3.8-flash",
    google_cloud=GoogleCloudAuth(service_account_file="sa.json", location="global"),
)

# Already-loaded credentials, or Application Default Credentials with no argument at all.
matcher = Matcher(tool, model="google-cloud:gemini-3.8-flash", google_cloud=GoogleCloudAuth(credentials=creds, project="my-proj"))
matcher = Matcher(tool, model="google-cloud:gemini-3.8-flash")   # ADC: GOOGLE_APPLICATION_CREDENTIALS, gcloud, or metadata server
```

`gemini:<model>` (and the bare default) chooses the route: the API key route when
`GOOGLE_API_KEY` is set and no `google_cloud` is given, otherwise Vertex AI through ADC.
So on a GCE, GKE, Cloud Run or Cloud Functions host with an attached service account,
`Matcher(tool)` with no configuration at all uses that default service account; the
project comes from the metadata server. `GOOGLE_CLOUD_PROJECT` and
`GOOGLE_CLOUD_LOCATION` override the defaults. With no credentials anywhere the ADC
probe takes a few seconds before failing with a message listing the options.

## Search modes

The LLM-facing `search` tool takes a string and a mode. A backend implements `contains`
and may add the other two; the runtime tells the LLM which modes exist.

| mode | backend method | meaning |
|---|---|---|
| `contains` | `search(query, limit)` | the whole string appears inside a name, case and accent insensitive |
| `all_terms` | `search_all_terms(terms, limit)` | every term appears somewhere in the name, any order |
| `fuzzy` | `search_fuzzy(query, limit)` | every term matches a word of the name within a small edit distance (1 up to 5 letters, else 2) |

Fuzzy is what recovers misspellings the LLM cannot guess. On the synthetic eval set,
exact substring search can surface the right record for 90% of queries at best; fuzzy
search reaches 100%.

## Mount your own search backend

Implement a `description` that tells the LLM what your backend holds, and
`search(query, limit)` returning the total match count plus up to `limit` records. Add
`search_all_terms` and `search_fuzzy` to unlock those modes. Sync or async both work.

```python
from ag_match import Record, SearchResult

class SqlNameSearch:
    description = "~40k companies; searches cover the `legal_name` and `trade_name` columns."

    def __init__(self, conn):
        self.conn = conn

    def search(self, query: str, limit: int) -> SearchResult:
        pattern = f"%{query}%"
        count = self.conn.execute(
            "SELECT count(*) FROM companies WHERE legal_name ILIKE ? OR trade_name ILIKE ?",
            (pattern, pattern),
        ).fetchone()[0]
        rows = self.conn.execute(
            "SELECT id, legal_name, country FROM companies "
            "WHERE legal_name ILIKE ? OR trade_name ILIKE ? LIMIT ?",
            (pattern, pattern, limit),
        ).fetchall()
        return SearchResult(
            total_count=count,
            records=[Record(id=str(r[0]), name=r[1], extra={"country": r[2]}) for r in rows],
        )

matcher = Matcher(SqlNameSearch(conn))
```

### BigQuery

`ag_match.bigquery.BigQuerySearchTool` implements all three modes in SQL. Install the
extra with `uv sync --extra bigquery` (or add `google-cloud-bigquery` to your app).

```python
from google.cloud import bigquery
from ag_match import Matcher
from ag_match.bigquery import BigQuerySearchTool

tool = BigQuerySearchTool(
    bigquery.Client(),
    table="proj.dataset.companies",
    id_column="company_id",
    name_column="legal_name",
    extra_columns=["country", "city"],   # shown to the LLM for disambiguation
    alias_columns=["trade_name"],        # searched alongside legal_name
    where="status = 'active'",           # optional filter
)
matcher = Matcher(tool)
```

Names are normalized in SQL the same way as in Python (accents stripped via
`NORMALIZE(..., NFKD)` and `REGEXP_REPLACE`, then lower-cased). `contains` and
`all_terms` use `STRPOS`; `fuzzy` uses `EDIT_DISTANCE(word, term, max_distance => n)`
over the words of each name and orders by total distance. Each search is one query that
also returns the total count through `COUNT(*) OVER()`. Every call scans the table, so
for very large tables put the normalized name in a column or a materialized view.

Extra tools are plain typed callables with a docstring:

```python
def lookup_ticker(company_id: str) -> str:
    """Return the stock ticker for a record id."""
    ...

matcher = Matcher(SqlNameSearch(conn), extra_tools=[lookup_ticker])
```

## Tuning

`MatchConfig` holds every budget and threshold:

| field | default | meaning |
|---|---|---|
| `records_per_search` | 15 | max records shown per search, the most similar to the query first |
| `too_many_threshold` | 60 | above this count the LLM is asked to narrow; with `show_on_too_many` it still sees the most similar rows fetched |
| `fetch_limit` | = threshold | rows fetched from the backend per search before ranking |
| `show_on_too_many` | true | show the most similar fetched rows even above the threshold |
| `prefetch` | true | search each distinctive word (and fuzzy) before the first LLM turn |
| `prefetch_terms` | 3 | max distinctive words searched during prefetch |
| `max_searches` | 8 | search calls the LLM may make per match; prefetch does not count |
| `max_rounds` | 6 | LLM requests per match; exceeding it ends the run as `inconclusive` |
| `output_retries` | 2 | retries when the final decision fails validation |
| `temperature` | 0.0 | sampling temperature; `None` leaves the model default |

A decision is validated before it is accepted: `match_id` and `alternatives` must be
ids the LLM was actually shown.

## CLI

```sh
uv run ag-match "Acmee Holdngs Intl." --list companies.csv --context country=US --trace
uv run ag-match "Acmee Holdngs Intl." --list companies.csv --service-account sa.json --location global
```

The CSV needs `id` and `name` columns; other columns become `extra` fields.

## Develop

```sh
uv run pytest                     # unit tests, no API calls
AG_MATCH_LIVE=1 uv run pytest tests/test_live.py -s   # live smoke test against Gemini
uv run ruff check . && uv run ruff format .
```

## Evaluate

`evals/` builds a seeded synthetic dataset (about 130 curated targets, 1,500 filler
records, 500 queries with typos, suffix changes, abbreviations, reorderings, accents,
namesakes needing context, and three kinds of negatives) and scores a run: accuracy,
precision, recall, searches and tokens per case, and a per-kind breakdown with every
failure listed.

```sh
uv run python -m evals.run --baseline                    # fuzzy-string reference, no LLM
uv run python -m evals.run --model heuristic             # agent loop driven by a fake LLM, no API
uv run python -m evals.run --model gemini --sample 100   # real model on a subset
uv run python -m evals.run --model gemini --too-many 30 --max-searches 5   # try other defaults
```

Reports land in `evals/results/` as markdown plus a JSON file with every case, decision,
search string and token count. Use `--kinds typo,reorder` to focus on one failure mode.

## Vendor into an app

The `package` branch holds only the contents of `src/ag_match` at its root. Copy it
into an app under any name with git subtree; the result is plain committed files with
no reference back to this repo:

```sh
git subtree add --prefix=myapp/namematch https://github.com/autoencoders/ag-match.git package --squash
```

Then add `pydantic` and `pydantic-ai-slim[google]` to the app's dependencies and import
from the new path. `docs/` travels with it: `ag_match/docs/ag-match-confluence.txt` is a
plain-text component overview, with the architecture diagram beside it. To pick up later changes, repeat with `git subtree pull`. Refresh the
branch after merging to main with `./scripts/publish-package-branch.sh main`.
