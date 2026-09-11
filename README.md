# ag-match

LLM-driven name matching. Give it a name and a search tool over your list, and an agent
figures out which record, if any, is the same entity. It copes with misspellings,
abbreviations, legal-suffix noise, reordered words and transliterations where plain
string matching fails.

How it works: the LLM proposes short, distinctive strings to search; the runtime runs
them through the tool you mounted and hands back the hits; the LLM narrows, widens or
decides. The runtime enforces all budgets and shaping, so the LLM never sees more than
a capped number of records per search and gets told to narrow when a string matches too
much. See [docs/PLAN.md](docs/PLAN.md) for the design.

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

## Mount your own search backend

Implement two things: a `description` that tells the LLM how matching works on your
backend, and `search(query, limit)` returning the total match count plus up to `limit`
records. Sync or async both work.

```python
from ag_match import Record, SearchResult

class SqlNameSearch:
    description = (
        "Case-insensitive substring match (SQL ILIKE) over the `legal_name` and "
        "`trade_name` columns of ~40k companies."
    )

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
| `records_per_search` | 15 | max records shown per search |
| `too_many_threshold` | 60 | above this count, no records are shown and the LLM is asked to narrow |
| `max_searches` | 8 | search calls per match |
| `max_rounds` | 6 | LLM requests per match; exceeding it ends the run as `inconclusive` |
| `output_retries` | 2 | retries when the final decision fails validation |

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

## Vendor into an app

The `package` branch holds only the contents of `src/ag_match` at its root. Copy it
into an app under any name with git subtree; the result is plain committed files with
no reference back to this repo:

```sh
git subtree add --prefix=myapp/namematch git@github.com:autoencoders/ag-match.git package --squash
```

Then add `pydantic` and `pydantic-ai-slim[google]` to the app's dependencies and import
from the new path. To pick up later changes, repeat with `git subtree pull`. Refresh the
branch after merging to main with `./scripts/publish-package-branch.sh main`.
