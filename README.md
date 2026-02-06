# LitGraph

Literature analysis knowledge graph service. Automatically searches academic databases, builds a knowledge graph via GraphRAG, analyzes papers with LLM, and identifies innovation opportunities.

## Prerequisites

- Python 3.9+
- An [Anthropic API key](https://console.anthropic.com/settings/keys) (for Pro mode)
- Or [Ollama](https://ollama.com) installed locally (for Lite mode)

## Install

```bash
git clone <repo-url> && cd LitGraph
pip install -e ".[dev]"
```

## Configure

```bash
cp .env.example .env
```

Edit `.env` and fill in your settings. See the two mode options below.

### Pro Mode: Claude via litellm proxy

Pro mode uses Claude models through a local [litellm](https://github.com/BerriAI/litellm) proxy that translates OpenAI-format API calls into Anthropic API calls.

**Step 1: Get an Anthropic API key**

Go to https://console.anthropic.com/settings/keys, create a key (starts with `sk-ant-api03-...`), and add credits to your account.

> Note: This is separate from a Claude Code / Claude Pro subscription. The Anthropic API has its own billing at https://console.anthropic.com.

**Step 2: Install litellm**

```bash
pip install litellm
```

**Step 3: Configure `.env`**

```bash
LITGRAPH_MODE=pro
LITGRAPH_ANTHROPIC_API_KEY=sk-ant-api03-your-key-here
LITGRAPH_PROXY_BASE_URL=http://localhost:3456/v1
LITGRAPH_PROXY_API_KEY=not-needed
LITGRAPH_PRO_BEST_MODEL=anthropic/claude-sonnet-4-20250514
LITGRAPH_PRO_CHEAP_MODEL=anthropic/claude-haiku-4-20250414
```

**Step 4: Start the litellm proxy**

```bash
# In a separate terminal (keep it running)
ANTHROPIC_API_KEY=sk-ant-api03-your-key-here litellm --model anthropic/claude-sonnet-4-20250514 --port 3456
```

You should see output like:
```
INFO:     Uvicorn running on http://0.0.0.0:3456
```

**Step 5: Verify the connection**

```bash
litgraph config validate
```

### Lite Mode: local Ollama

Lite mode uses a local Ollama model. Free, offline, but lower quality — recommended for prompt debugging and small-scale testing only.

```bash
# Install Ollama: https://ollama.com
ollama pull qwen2.5:7b
```

In `.env`:
```bash
LITGRAPH_MODE=lite
LITGRAPH_OLLAMA_BASE_URL=http://localhost:11434/v1
LITGRAPH_OLLAMA_MODEL=qwen2.5:7b
```

## Quick Start

```bash
# Make sure the litellm proxy is running (Pro mode), then:

# Full pipeline: expand keywords → search → filter → analyze → KG → innovate
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

## Project Structure

```
LitGraph/
├── .env.example              # Configuration template
├── config/
│   ├── prompts/*.yaml        # LLM prompt templates (Jinja2)
│   ├── questions.yaml        # Paper analysis questions
│   ├── schema.yaml           # KG node/relation type definitions
│   └── config.default.yaml   # Default search/filter parameters
├── src/litgraph/
│   ├── settings.py           # Config loading (.env + YAML → Settings singleton)
│   ├── retry.py              # Retry decorator + rate limiter
│   ├── cli.py                # All CLI commands
│   ├── llm/
│   │   ├── client.py         # Unified LLM interface (OpenAI SDK)
│   │   └── prompts.py        # Jinja2 template loader
│   ├── search/
│   │   ├── arxiv.py          # arXiv search via paperscraper
│   │   ├── semantic.py       # Semantic Scholar API
│   │   ├── dedup.py          # Cross-run deduplication
│   │   └── filters.py        # Citation + LLM relevance filtering
│   ├── kg/
│   │   ├── graph.py          # nano-graphrag integration
│   │   ├── direct.py         # NetworkX direct operations
│   │   └── schema.py         # Entity normalization + validation
│   ├── analysis/
│   │   ├── paper.py          # Single paper: PDF → extract → LLM → Markdown
│   │   ├── batch.py          # Batch analysis with progress bar
│   │   └── innovation.py     # Innovation identification
│   └── output/
│       └── report.py         # Report generation
└── tests/
    ├── dummy/                # Offline tests (no LLM/network)
    └── live/                 # Real service tests (@pytest.mark.live)
```

Runtime data is stored in a separate `DATA/` directory (default: `../DATA`, configurable via `LITGRAPH_DATA_DIR`).

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
- **Dual mode** — Pro (Claude via litellm proxy) and Lite (local Ollama)
- **Test-driven** — 118 dummy tests (offline) and live tests (real services) in separate directories
