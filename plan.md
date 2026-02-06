# LitGraph Implementation Plan

Project root: `/Users/shipan/Documents/workspace_automate_life/literature_graph/LitGraph/`
Architecture doc: `/Users/shipan/Documents/workspace_automate_life/literature_graph/LiteGRAPH_ARCHITECTURE.md`

---

## Key Implementation Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Config | `dataclass` + `python-dotenv` | No pydantic, flat config is sufficient |
| Retry | Hand-rolled decorator + `RateLimiter` | Tenacity not in deps; simple enough |
| LLM bridge | Sync `OpenAI` + `asyncio.to_thread` for nano-graphrag async | One client, no dual sync/async |
| KG schema | Custom `GraphStorage` subclass, override `upsert_node`/`upsert_edge` | Intercept before write, no post-processing |
| Template | Jinja2 `{{ }}` with `StrictUndefined` | Safe from paper content `{ }` |
| PDF | `tempfile` + `pymupdf` + immediate delete | No persistent PDF per architecture |
| Build | `hatchling` + PEP 621 | Minimal config |

---

## Phase 0: Scaffolding + Settings + Retry

**Test first**: `tests/dummy/test_settings.py`

### Files to create:

1. `pyproject.toml` — hatchling build, all deps, `[project.scripts] litgraph = "litgraph.cli:main"`, pytest markers
2. `.env.example` — from architecture doc
3. Directory tree + `__init__.py`:
   - `src/litgraph/{__init__,settings,retry}.py`
   - `src/litgraph/{search,llm,kg,analysis,output}/__init__.py`
   - `tests/{__init__,conftest}.py`, `tests/{dummy,live}/__init__.py`
4. `config/questions.yaml` — 6 questions, version: 1
5. `config/schema.yaml` — node_types, relation_types, aliases
6. `config/config.default.yaml` — search/filter/analysis defaults
7. `config/prompts/*.yaml` — paper_analysis, relevance_filter, entity_extraction, innovation, keyword_expansion
8. `tests/conftest.py` — fixtures: `fixture_papers` (3 papers), `data_dir` (tmp_path), `mock_llm`, `small_graph` (5-node nx.Graph), `sample_schema`
9. `tests/dummy/test_settings.py` — env loading, path resolution, mode switch, retry/rate_limit defaults

### Implementation:

**`src/litgraph/settings.py`**: `get_settings(project_root=None, force_reload=False) -> Settings` singleton. `@dataclass` for `Settings`, `LLMConfig`, `RetryConfig`, `RateLimitConfig`. Auto-detect `project_root` from `__file__`. Lite mode: `best_model == cheap_model`.

**`src/litgraph/retry.py`**: `with_retry(func)` decorator (handles both sync/async via `iscoroutinefunction`), exponential backoff `base^attempt`. `RateLimiter` class with sliding-window `deque[float]`, `acquire_sync()` and `acquire()`.

### Verify & Commit:
```bash
pip install -e ".[dev]"
pytest tests/dummy/test_settings.py -v
```
Commit: `"Phase 0: scaffolding, config YAMLs, settings + retry"`

---

## Phase 1: LLM Client + Prompt Loader

**Test first**: `tests/dummy/test_prompts.py`

### Files to create:

1. `tests/dummy/test_prompts.py` — load YAML, Jinja2 render, missing var raises UndefinedError, `{curly}` in content safe, optional `full_text` conditional, load_questions returns 6 items
2. `src/litgraph/llm/prompts.py` — `load_prompt(name, **kwargs) -> (system, user)`, `load_questions() -> list[dict]`, `get_questions_version() -> int`, `format_questions_block(questions) -> str`. Uses `jinja2.Environment(loader=BaseLoader(), undefined=StrictUndefined)`. System prompt is plain text (not templated).
3. `src/litgraph/llm/client.py`:
   - `complete(prompt, system_prompt=None, model="best") -> str` — sync, `@with_retry`, uses `OpenAI(base_url, api_key)`
   - `best_model_complete(prompt, system_prompt=None, history_messages=[], **kwargs) -> str` — async, for nano-graphrag. Handles `hashing_kv` caching kwarg. Uses `asyncio.to_thread` to call sync OpenAI.
   - `cheap_model_complete(...)` — same pattern, uses cheap_model
   - `_maybe_warn_lite()` — `logging.warning` on first Lite mode call
4. `tests/dummy/test_cli_args.py` (partial) — mode routing, lite best==cheap
5. `tests/live/test_llm_client.py` — pro/lite real request, verify non-empty return

### Verify & Commit:
```bash
pytest tests/dummy/test_prompts.py tests/dummy/test_cli_args.py -v
```
Commit: `"Phase 1: LLM client + Jinja2 prompt loader"`

---

## Phase 2: Paper Search + Dedup

**Test first**: `tests/dummy/test_search_dedup.py`

### Files to create:

1. `tests/dummy/test_search_dedup.py` — dedup_key priority (arxiv > doi > title_hash), title normalization (lowercase + strip punctuation + SHA256[:16]), single-run dedup, cross-run merge into index.json (add/update), empty results
2. `src/litgraph/search/dedup.py`:
   - `dedup_key(paper) -> str` — `"arxiv:{id}"` / `"doi:{doi}"` / `"title:{hash}"`
   - `_title_hash(title) -> str` — lower → strip punct → collapse whitespace → SHA256[:16]
   - `dedup_paper_list(papers) -> list[dict]` — single-run, keeps first occurrence
   - `merge_into_index(new_papers, index_path) -> (added, updated)` — cross-run, updates meta fields (citations, doi, pdf_url)
3. `src/litgraph/search/arxiv.py` — `search_arxiv(keywords, max_results) -> list[dict]`, uses `paperscraper.arxiv.get_and_dump_arxiv_papers` → JSONL → normalize to standard dict format
4. `src/litgraph/search/semantic.py` — `search_semantic_scholar(keywords, max_results, year_from) -> list[dict]`, uses `semanticscholar.SemanticScholar().search_paper`, with `RateLimiter`
5. `src/litgraph/search/filters.py` — `filter_papers(papers, keywords, min_citations, use_llm_filter) -> list[dict]`, citation threshold + optional LLM relevance scoring via `relevance_filter.yaml`
6. `tests/live/test_search_real.py` — real arxiv + semantic scholar search, verify structure

### Verify & Commit:
```bash
pytest tests/dummy/test_search_dedup.py -v
```
Commit: `"Phase 2: paper search (arxiv + semantic scholar), dedup, filters"`

---

## Phase 3: Single Paper Analysis

**Test first**: `tests/dummy/test_pdf_parse.py`, `tests/dummy/test_analysis_format.py`

### Files to create:

1. `tests/dummy/test_pdf_parse.py` — create fixture PDF with pymupdf programmatically, test text extraction, line-break merge (`founda-\ntion` → `foundation`), page number cleanup
2. `tests/dummy/test_analysis_format.py` — mock LLM, verify Markdown with YAML front matter (all fields), skip existing, questions_version mismatch triggers re-analysis + rename old to `.v{N}.md`, `source_type: "abstract_only"` vs `"full_text"`
3. `src/litgraph/analysis/paper.py`:
   - `analyze_paper(paper, data_dir) -> Path | None` — full flow: check existing → download PDF to tempfile → pymupdf extract → cleanup → delete PDF → Jinja2 prompt → LLM → write Markdown with YAML front matter
   - `extract_pdf_text(pdf_path, max_pages=50) -> str`
   - `cleanup_pdf_text(text) -> str` — strip page numbers, merge hyphenated breaks
   - `_download_and_extract_pdf(pdf_url) -> str | None` — `@with_retry`, urllib download to tempfile, extract, delete
   - `_extract_questions_version(md_path) -> int | None`
4. `src/litgraph/analysis/batch.py` — `analyze_batch(paper_ids, all_pending, data_dir) -> dict` — sequential loop with `tqdm`, returns `{analyzed, skipped, failed, errors}`
5. Add `fixture_pdf` to `tests/conftest.py`

### Verify & Commit:
```bash
pytest tests/dummy/test_pdf_parse.py tests/dummy/test_analysis_format.py -v
```
Commit: `"Phase 3: paper analysis (PDF extraction, LLM analysis, Markdown output)"`

---

## Phase 4: Knowledge Graph

**Test first**: `tests/dummy/test_kg_schema.py`, `tests/dummy/test_kg_direct.py`

### Files to create:

1. `tests/dummy/test_kg_schema.py` — normalize alias (case-insensitive), unknown passthrough, validate entity type, validate relation, invalid relation degrades to `"related_to"`
2. `tests/dummy/test_kg_direct.py` — expand_keywords from seed (BFS neighbors), subgraph extraction, get_stats, empty graph
3. `src/litgraph/kg/schema.py`:
   - `normalize_entity(name, schema_dict=None) -> str` — case-insensitive alias lookup
   - `validate_entity(name, entity_type, schema_dict=None) -> bool`
   - `validate_relation(from_type, rel_type, to_type, schema_dict=None) -> bool`
   - `normalize_relation_type(...) -> str` — valid → keep, invalid → `"related_to"`
   - All functions accept `schema_dict` for testing without filesystem
4. `src/litgraph/kg/direct.py`:
   - `load_graph(graphml_path) -> nx.Graph`
   - `expand_keywords(graph, seeds, max_hops=2, max_results=20) -> list[str]` — BFS on Concept/Method/Task nodes
   - `get_subgraph(graph, center_node, max_hops=2) -> nx.Graph`
   - `get_stats(graph) -> dict`
5. `src/litgraph/kg/graph.py`:
   - `SchemaAwareStorage(NetworkXStorage)` — override `upsert_node`, `upsert_edge`, batch variants → call `normalize_entity` before `super()`
   - `get_rag(working_dir) -> GraphRAG` — inject `best_model_complete`, `cheap_model_complete`, local `SentenceTransformer` embedding via `wrap_embedding_func_with_attrs`, `SchemaAwareStorage`, override `PROMPTS["entity_extraction"]` + `PROMPTS["DEFAULT_ENTITY_TYPES"]`, `max_async=1`
   - `insert_texts(texts, working_dir)`, `query_graph(question, mode, working_dir)`
6. `tests/live/test_kg_insert.py` — insert 1 paper text, query, verify non-empty response

### Verify & Commit:
```bash
pytest tests/dummy/test_kg_schema.py tests/dummy/test_kg_direct.py -v
```
Commit: `"Phase 4: knowledge graph (schema, direct ops, nano-graphrag integration)"`

---

## Phase 5: Innovation + Reports

**Test first**: `tests/dummy/test_innovation.py` (new)

### Files to create:

1. `tests/dummy/test_innovation.py` — innovation prompt includes KG context + papers summary, report has Markdown structure, empty KG still works
2. `src/litgraph/analysis/innovation.py` — `identify_innovations(scope, data_dir) -> str`, gathers analysis .md files + KG global query → renders `innovation.yaml` prompt → LLM
3. `src/litgraph/output/report.py` — `save_report(content, report_type, data_dir) -> Path`, writes to `DATA/reports/{timestamp}_{type}.md` with metadata comment

### Verify & Commit:
```bash
pytest tests/dummy/test_innovation.py -v
```
Commit: `"Phase 5: innovation analysis + report generation"`

---

## Phase 6: Pipeline + CLI

**Test first**: `tests/dummy/test_pipeline_resume.py`, `tests/dummy/test_cli_args.py` (complete)

### Files to create:

1. `tests/dummy/test_pipeline_resume.py` — skip completed search (index.json exists), skip analyzed papers, detect pending, run record saved
2. `tests/dummy/test_cli_args.py` (complete) — all subcommands via `CliRunner`: search, filter, analyze, kg update/query/expand/stats, innovate, run, config show/validate, `--mode lite` warning
3. `src/litgraph/cli.py`:
   - `@click.group() main(mode)` — setup logging, optional mode override via `os.environ`, lite warning
   - `search` — `--keywords` (multiple), `--sources`, `--max-results`, `--year-from`
   - `filter` — `--min-citations`, `--relevance-check`
   - `analyze` — `--paper-ids` (multiple), `--all-pending`
   - `kg` group → `update`, `query`, `expand`, `stats`
   - `innovate` — `--scope`
   - `run` — full pipeline: keyword expand → search → filter → analyze → KG update → innovate → save run record
   - `config` group → `show`, `validate`
   - Exit codes: 0 success, 1 partial fail, 2 config error
4. `tests/live/test_pipeline_e2e.py` — 2 papers end-to-end lite mode

### Verify & Commit:
```bash
pytest tests/dummy/test_pipeline_resume.py tests/dummy/test_cli_args.py -v
```
Commit: `"Phase 6: CLI + pipeline with resume support"`

---

## Phase 7: Polish

1. Structured `logging` throughout (replace any `print`)
2. Error handling: PDF download timeout (30s), `HTTPError(429)` in search, disk space
3. `config validate` — ping LLM endpoint, test embedding model load
4. Update `README.md` — install, configure, quick start, CLI reference

### Verify & Commit:
```bash
pytest tests/dummy/ -v  # all dummy tests green
```
Commit: `"Phase 7: logging, error handling, README"`

---

## Commit Summary (8 commits)

| # | Phase | Commit Message |
|---|-------|---------------|
| 1 | 0 | Phase 0: scaffolding, config YAMLs, settings + retry |
| 2 | 1 | Phase 1: LLM client + Jinja2 prompt loader |
| 3 | 2 | Phase 2: paper search (arxiv + semantic scholar), dedup, filters |
| 4 | 3 | Phase 3: paper analysis (PDF extraction, LLM analysis, Markdown output) |
| 5 | 4 | Phase 4: knowledge graph (schema, direct ops, nano-graphrag integration) |
| 6 | 5 | Phase 5: innovation analysis + report generation |
| 7 | 6 | Phase 6: CLI + pipeline with resume support |
| 8 | 7 | Phase 7: logging, error handling, README |
