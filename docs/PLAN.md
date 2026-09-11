# ag-match plan

Goal: match one name (typically a company name) against a list of names using an LLM,
where plain string matching fails: misspellings, abbreviations, legal-suffix noise,
word reordering, transliteration.

## Workflow (agentic mode)

1. **Input.** A query name, optional context (country, sector, ...), and a search tool
   mounted at runtime. The tool takes an exact string and returns matching records from
   the list, wherever it lives (memory, SQL, search index).
2. **Propose.** The LLM proposes a small set of exact strings to search. It corrects
   likely misspellings, strips legal suffixes and generic words, and prefers the most
   distinctive fragments.
3. **Search.** The runtime executes each string through the tool and returns the hits.
   Above a threshold the LLM gets only the count plus a request to narrow. On zero hits
   it gets a hint to widen or try another spelling.
4. **Decide or iterate.** The LLM picks a record, declares no match, or searches again,
   inside a budget of LLM rounds and searches.
5. **Output.** A structured decision (status, matched id, confidence, reasoning,
   alternatives) plus a trace of every search and what came back.

## Design decisions

- **Python 3.12, uv, pydantic-ai** for the agent loop: one model string covers Gemini,
  OpenAI, Anthropic and others; native function calling; typed tools; structured output;
  usage limits; fake models for tests. Our own interfaces are framework-free and
  pydantic-ai sits behind `matcher.py`.
- **`SearchTool` protocol** is the extension point. `description` tells the LLM the
  backend's matching semantics. `search(query, limit)` may be sync or async and returns
  `total_count` plus up to `limit` records. Extra tools are plain typed callables.
- **Runtime, not prompt, keeps things lean**: per-search record cap, too-many threshold
  (count only, no records, narrow-it note), dedupe by record id across searches,
  repeat-query short circuit, search budget, LLM round budget. On exhaustion the run
  ends `inconclusive`, never with a guess.
- **Output validation**: a chosen `match_id` must be a record the LLM actually saw,
  otherwise the model is asked to retry. Closes the hallucinated-match hole.
- **Async first**, sync wrapper, `match_many` with bounded concurrency.
- **Output shape**: single best match plus ranked alternatives.

## Phases

1. Scaffold: uv project, ruff, pytest, README.
2. Core types, `SearchTool`, `InMemorySearchTool`, `SearchSession` shaping. Unit tests, no LLM.
3. `Matcher` on pydantic-ai: prompt, tool wiring, budgets, output validation, trace.
   Tests script a fake LLM through every path.
4. Providers: model spec resolution (Gemini default, OpenAI, Anthropic), env config,
   live Gemini smoke test behind an env flag.
5. Batch + CLI: `match_many`, `ag-match "name" --list names.csv`.
6. Eval harness: synthetic perturbations of a seed list, precision / recall / searches
   per match / tokens per match; tune prompt and defaults.
7. Docs: README with a worked SQL-backed tool example.
