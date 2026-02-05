# LitGraph

Literature analysis knowledge graph service. Automatically searches academic databases, builds a knowledge graph via GraphRAG, analyzes papers with LLM, and identifies innovation opportunities.

## Install

```bash
pip install -e ".[dev]"
```

## Configure

```bash
cp .env.example .env
# Edit .env with your settings
```

**Pro mode** (Claude via proxy):
```
LITGRAPH_MODE=pro
LITGRAPH_PROXY_BASE_URL=http://localhost:3456/v1
```

**Lite mode** (local Ollama):
```
LITGRAPH_MODE=lite
LITGRAPH_OLLAMA_BASE_URL=http://localhost:11434/v1
LITGRAPH_OLLAMA_MODEL=qwen2.5:7b
```

## Quick Start

```bash
# Full pipeline: search → filter → analyze → KG → innovate
litgraph run --keywords "single-cell foundation model" --mode pro

# Or step by step:
litgraph search --keywords "spatial transcriptomics" --year-from 2023
litgraph filter --min-citations 5
litgraph analyze --all-pending
litgraph kg update --all-pending
litgraph innovate --scope all
```

## CLI Reference

```
litgraph search    -k KEYWORDS [--sources arxiv,semantic] [--max-results 50] [--year-from YEAR]
litgraph filter    [--min-citations 5] [--relevance-check]
litgraph analyze   [-p PAPER_IDS...] [--all-pending]
litgraph kg update [-p PAPER_IDS...] [--all-pending]
litgraph kg query  QUESTION [--mode local|global|naive]
litgraph kg expand -k KEYWORDS [--max-hops 2]
litgraph kg stats
litgraph innovate  [--scope all|last-run]
litgraph run       -k KEYWORDS [--resume] [--min-citations 5]
litgraph config show
litgraph config validate
```

Exit codes: 0 = success, 1 = partial failure, 2 = config error.

## Testing

```bash
# Offline tests (no LLM/network required)
pytest tests/dummy/ -v

# Live tests (requires running proxy or Ollama)
pytest tests/live/ -m live -v
```

## Architecture

- **Pure Python, zero external databases** — NetworkX + JSON for graph, nano-graphrag for vector storage
- **Single-process sequential execution** — simple, no concurrency issues
- **Code/data separation** — code in repo, runtime data in DATA/ directory
- **Dual mode** — Pro (Claude via proxy) and Lite (local Ollama)
- **Test-driven** — dummy tests (offline) and live tests (real services) in separate directories
