"""CLI entry point — all subcommands for LitGraph."""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import click

logger = logging.getLogger(__name__)


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _warn_lite_mode() -> None:
    """Print Lite mode warning."""
    from litgraph.settings import get_settings
    settings = get_settings()
    if settings.mode == "lite":
        click.secho(
            f"WARNING: Running in Lite mode ({settings.llm.best_model})\n"
            "  - Analysis quality will be lower than Pro mode\n"
            "  - KG entity extraction accuracy may be significantly reduced\n"
            "  - Recommended: use Lite mode only for prompt debugging and small-scale testing\n"
            "  - For production analysis, switch to Pro mode: --mode pro",
            fg="yellow",
            err=True,
        )


@click.group()
@click.option("--mode", type=click.Choice(["pro", "lite"]), default=None,
              help="Override LLM mode (pro or lite).")
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging.")
@click.pass_context
def main(ctx, mode, verbose):
    """LitGraph: Literature analysis knowledge graph service."""
    _setup_logging(verbose)

    if mode:
        os.environ["LITGRAPH_MODE"] = mode

    from litgraph.settings import get_settings, reset_settings
    reset_settings()
    get_settings()

    _warn_lite_mode()

    ctx.ensure_object(dict)


@main.command()
@click.option("--keywords", "-k", multiple=True, required=True, help="Search keywords.")
@click.option("--sources", default="arxiv,semantic", help="Comma-separated sources.")
@click.option("--max-results", default=50, type=int, help="Max results per source.")
@click.option("--year-from", default=None, type=int, help="Minimum publication year.")
def search(keywords, sources, max_results, year_from):
    """Search for papers across academic databases."""
    from litgraph.search.dedup import dedup_paper_list, merge_into_index
    from litgraph.settings import get_settings

    settings = get_settings()
    source_list = [s.strip() for s in sources.split(",")]
    all_papers = []

    if "arxiv" in source_list:
        from litgraph.search.arxiv import search_arxiv
        click.echo(f"Searching arxiv for: {', '.join(keywords)}")
        papers = search_arxiv(list(keywords), max_results=max_results)
        click.echo(f"  Found {len(papers)} papers from arxiv")
        all_papers.extend(papers)

    if "semantic" in source_list:
        from litgraph.search.semantic import search_semantic_scholar
        click.echo(f"Searching Semantic Scholar for: {', '.join(keywords)}")
        papers = search_semantic_scholar(list(keywords), max_results=max_results, year_from=year_from)
        click.echo(f"  Found {len(papers)} papers from Semantic Scholar")
        all_papers.extend(papers)

    # Dedup within this run
    deduped = dedup_paper_list(all_papers)
    click.echo(f"After dedup: {len(deduped)} unique papers")

    # Merge into index
    index_path = settings.data_dir / "papers" / "index.json"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    added, updated = merge_into_index(deduped, index_path)
    click.echo(f"Index: {len(added)} new, {len(updated)} updated")


@main.command("filter")
@click.option("--min-citations", default=5, type=int, help="Minimum citation count.")
@click.option("--relevance-check", is_flag=True, help="Use LLM for relevance scoring.")
def filter_cmd(min_citations, relevance_check):
    """Filter papers by citation count and optional LLM relevance."""
    from litgraph.search.filters import filter_papers
    from litgraph.settings import get_settings

    settings = get_settings()
    index_path = settings.data_dir / "papers" / "index.json"

    if not index_path.exists():
        click.secho("No index.json found. Run 'search' first.", fg="red", err=True)
        sys.exit(2)

    with open(index_path) as f:
        papers = json.load(f)

    filtered = filter_papers(papers, min_citations=min_citations, use_llm_filter=relevance_check)

    # Update index with relevance markers
    with open(index_path, "w") as f:
        json.dump(papers, f, indent=2, ensure_ascii=False)

    click.echo(f"Filtered: {len(filtered)}/{len(papers)} papers passed")


@main.command()
@click.option("--paper-ids", "-p", multiple=True, help="Specific paper IDs to analyze.")
@click.option("--all-pending", is_flag=True, help="Analyze all papers without analysis.")
def analyze(paper_ids, all_pending):
    """Analyze papers using LLM."""
    from litgraph.analysis.batch import analyze_batch
    from litgraph.settings import get_settings

    settings = get_settings()
    stats = analyze_batch(
        paper_ids=list(paper_ids) if paper_ids else None,
        all_pending=all_pending,
        data_dir=settings.data_dir,
    )

    click.echo(f"Analyzed: {stats['analyzed']}, Skipped: {stats['skipped']}, Failed: {stats['failed']}")
    if stats["errors"]:
        for e in stats["errors"]:
            click.secho(f"  Error: {e['paper_id']}: {e['error']}", fg="red", err=True)
        sys.exit(1)


@main.group()
def kg():
    """Knowledge graph operations."""
    pass


@kg.command()
@click.option("--paper-ids", "-p", multiple=True, help="Specific paper IDs.")
@click.option("--all-pending", is_flag=True, help="Insert all unprocessed papers.")
def update(paper_ids, all_pending):
    """Update the knowledge graph with paper analyses."""
    from litgraph.kg.graph import insert_texts
    from litgraph.settings import get_settings

    settings = get_settings()
    analysis_dir = settings.data_dir / "analysis"

    if not analysis_dir.exists():
        click.secho("No analysis directory. Run 'analyze' first.", fg="red", err=True)
        sys.exit(2)

    md_files = sorted(analysis_dir.glob("*.md"))
    md_files = [f for f in md_files if not any(f.stem.endswith(f".v{i}") for i in range(100))]

    if paper_ids:
        selected = []
        for f in md_files:
            for pid in paper_ids:
                safe_pid = pid.replace(":", "_").replace("/", "_")
                if f.stem == safe_pid:
                    selected.append(f)
        md_files = selected

    if not md_files:
        click.echo("No papers to insert into KG.")
        return

    texts = []
    for f in md_files:
        texts.append(f.read_text(encoding="utf-8"))

    click.echo(f"Inserting {len(texts)} papers into KG...")
    insert_texts(texts)
    click.echo("KG update complete.")


@kg.command()
@click.argument("question")
@click.option("--mode", "query_mode", default="global", type=click.Choice(["local", "global", "naive"]))
def query(question, query_mode):
    """Query the knowledge graph."""
    from litgraph.kg.graph import query_graph

    result = query_graph(question, mode=query_mode)
    click.echo(result)


@kg.command()
@click.option("--keywords", "-k", multiple=True, required=True, help="Seed keywords.")
@click.option("--max-hops", default=2, type=int)
@click.option("--max-results", default=20, type=int)
def expand(keywords, max_hops, max_results):
    """Expand keywords using the knowledge graph."""
    from litgraph.kg.direct import expand_keywords, load_graph
    from litgraph.settings import get_settings

    settings = get_settings()
    kg_dir = settings.data_dir / "kg_store"
    graphml_files = list(kg_dir.glob("*.graphml")) if kg_dir.exists() else []

    if not graphml_files:
        click.echo("No graph file found. Run 'kg update' first.")
        return

    graph = load_graph(graphml_files[0])
    results = expand_keywords(graph, list(keywords), max_hops=max_hops, max_results=max_results)
    click.echo(f"Expanded keywords ({len(results)}):")
    for kw in results:
        click.echo(f"  - {kw}")


@kg.command()
def stats():
    """Show knowledge graph statistics."""
    from litgraph.kg.direct import get_stats, load_graph
    from litgraph.settings import get_settings

    settings = get_settings()
    kg_dir = settings.data_dir / "kg_store"
    graphml_files = list(kg_dir.glob("*.graphml")) if kg_dir.exists() else []

    if not graphml_files:
        click.echo("No graph file found.")
        return

    graph = load_graph(graphml_files[0])
    s = get_stats(graph)
    click.echo(f"Nodes: {s['node_count']}")
    click.echo(f"Edges: {s['edge_count']}")
    click.echo("Node types:")
    for k, v in s["node_types"].items():
        click.echo(f"  {k}: {v}")
    click.echo("Relation types:")
    for k, v in s["relation_types"].items():
        click.echo(f"  {k}: {v}")


@main.command()
@click.option("--scope", default="all", type=click.Choice(["all", "last-run"]))
def innovate(scope):
    """Identify innovation opportunities."""
    from litgraph.analysis.innovation import identify_innovations
    from litgraph.output.report import save_report
    from litgraph.settings import get_settings

    settings = get_settings()
    click.echo(f"Running innovation analysis (scope: {scope})...")
    report = identify_innovations(scope=scope, data_dir=settings.data_dir)
    path = save_report(report, "innovation", settings.data_dir)
    click.echo(f"Report saved: {path}")


@main.command()
@click.option("--keywords", "-k", multiple=True, required=True, help="Seed keywords.")
@click.option("--sources", default="arxiv,semantic")
@click.option("--max-results", default=50, type=int)
@click.option("--year-from", default=None, type=int)
@click.option("--min-citations", default=5, type=int)
@click.option("--resume", is_flag=True, help="Skip completed steps.")
def run(keywords, sources, max_results, year_from, min_citations, resume):
    """Run the full pipeline: search → filter → analyze → KG → innovate."""
    from litgraph.analysis.batch import analyze_batch
    from litgraph.analysis.innovation import identify_innovations
    from litgraph.kg.graph import insert_texts
    from litgraph.output.report import save_report
    from litgraph.search.dedup import dedup_paper_list, merge_into_index
    from litgraph.search.filters import filter_papers
    from litgraph.settings import get_settings

    settings = get_settings()
    data_dir = settings.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)

    run_record = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "keywords": list(keywords),
        "mode": settings.mode,
        "steps": {},
    }

    # Step 1: Keyword expansion (if KG exists)
    search_keywords = list(keywords)
    kg_dir = data_dir / "kg_store"
    graphml_files = list(kg_dir.glob("*.graphml")) if kg_dir.exists() else []
    if graphml_files:
        click.echo("Step 1: Expanding keywords from KG...")
        from litgraph.kg.direct import expand_keywords, load_graph
        graph = load_graph(graphml_files[0])
        expanded = expand_keywords(graph, search_keywords, max_hops=2, max_results=20)
        new_kws = [kw for kw in expanded if kw not in search_keywords]
        if new_kws:
            search_keywords.extend(new_kws)
            click.echo(f"  Expanded: +{len(new_kws)} keywords → {search_keywords}")
        run_record["steps"]["expand"] = {"original": list(keywords), "expanded": new_kws}
    else:
        click.echo("Step 1: Keyword expansion — skipping (no KG yet)")
        run_record["steps"]["expand"] = "skipped"

    # Step 2: Search
    index_path = data_dir / "papers" / "index.json"
    if resume and index_path.exists():
        click.echo("Step 2: Search — skipping (index.json exists)")
        run_record["steps"]["search"] = "skipped"
    else:
        click.echo("Step 2: Searching papers...")
        source_list = [s.strip() for s in sources.split(",")]
        all_papers = []

        if "arxiv" in source_list:
            from litgraph.search.arxiv import search_arxiv
            all_papers.extend(search_arxiv(search_keywords, max_results=max_results))

        if "semantic" in source_list:
            from litgraph.search.semantic import search_semantic_scholar
            all_papers.extend(search_semantic_scholar(search_keywords, max_results=max_results, year_from=year_from))

        deduped = dedup_paper_list(all_papers)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        added, updated = merge_into_index(deduped, index_path)
        click.echo(f"  Found {len(deduped)} papers ({len(added)} new)")
        run_record["steps"]["search"] = {"found": len(deduped), "added": len(added)}

    # Step 2: Filter
    click.echo("Step 2: Filtering papers...")
    with open(index_path) as f:
        papers = json.load(f)
    filtered = filter_papers(papers, list(keywords), min_citations=min_citations)
    # Write back index with relevance markers
    with open(index_path, "w") as f:
        json.dump(papers, f, indent=2, ensure_ascii=False)
    click.echo(f"  {len(filtered)} papers passed filter")
    run_record["steps"]["filter"] = {"passed": len(filtered)}

    # Step 3: Analyze — only filtered papers
    click.echo("Step 3: Analyzing papers...")
    filtered_ids = [p.get("paper_id") or p.get("dedup_key") for p in filtered]
    stats = analyze_batch(paper_ids=filtered_ids, data_dir=data_dir)
    click.echo(f"  Analyzed: {stats['analyzed']}, Skipped: {stats['skipped']}, Failed: {stats['failed']}")
    run_record["steps"]["analyze"] = stats

    # Step 4: KG Update
    click.echo("Step 4: Updating knowledge graph...")
    analysis_dir = data_dir / "analysis"
    if analysis_dir.exists():
        md_files = sorted(analysis_dir.glob("*.md"))
        md_files = [f for f in md_files if not any(f.stem.endswith(f".v{i}") for i in range(100))]
        if md_files:
            texts = [f.read_text(encoding="utf-8") for f in md_files]
            insert_texts(texts)
            click.echo(f"  Inserted {len(texts)} texts into KG")
            run_record["steps"]["kg_update"] = {"inserted": len(texts)}
        else:
            click.echo("  No analyses to insert")
            run_record["steps"]["kg_update"] = "no_content"
    else:
        click.echo("  No analysis directory found")
        run_record["steps"]["kg_update"] = "skipped"

    # Step 5: Innovation
    click.echo("Step 5: Innovation analysis...")
    try:
        report = identify_innovations(scope="last-run", data_dir=data_dir)
        path = save_report(report, "innovation", data_dir)
        click.echo(f"  Report saved: {path}")
        run_record["steps"]["innovate"] = str(path)
    except Exception as e:
        click.secho(f"  Innovation analysis failed: {e}", fg="yellow", err=True)
        run_record["steps"]["innovate"] = f"failed: {e}"

    # Save run record
    run_record["completed_at"] = datetime.now(timezone.utc).isoformat()
    runs_dir = data_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    run_path = runs_dir / f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
    with open(run_path, "w") as f:
        json.dump(run_record, f, indent=2, ensure_ascii=False)
    click.echo(f"Run record: {run_path}")


@main.group()
def config():
    """Configuration management."""
    pass


@config.command("show")
def config_show():
    """Show current configuration."""
    from litgraph.settings import get_settings

    settings = get_settings()
    click.echo(f"Mode: {settings.mode}")
    click.echo(f"Data dir: {settings.data_dir}")
    click.echo(f"LLM base URL: {settings.llm.base_url}")
    click.echo(f"Best model: {settings.llm.best_model}")
    click.echo(f"Cheap model: {settings.llm.cheap_model}")
    click.echo(f"Embedding model: {settings.embedding_model}")
    click.echo(f"Max retries: {settings.retry.max_retries}")
    click.echo(f"Rate limit: {settings.rate_limit.search_max_calls}/{settings.rate_limit.search_period}s")


@config.command("validate")
def config_validate():
    """Validate configuration (ping LLM, check embedding model)."""
    from litgraph.settings import get_settings

    settings = get_settings()
    errors = []

    # Test LLM endpoint
    click.echo(f"Testing LLM endpoint ({settings.llm.base_url})...")
    try:
        from litgraph.llm.client import complete, reset_client
        reset_client()
        result = complete("Say 'ok'.", model="cheap")
        if result:
            click.secho("  LLM: OK", fg="green")
        else:
            errors.append("LLM returned empty response")
            click.secho("  LLM: empty response", fg="red")
    except Exception as e:
        errors.append(f"LLM: {e}")
        click.secho(f"  LLM: FAILED ({e})", fg="red")

    # Test embedding model
    click.echo(f"Testing embedding model ({settings.embedding_model})...")
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(settings.embedding_model)
        emb = model.encode(["test"])
        if emb is not None and len(emb) > 0:
            click.secho(f"  Embedding: OK (dim={emb.shape[1]})", fg="green")
        else:
            errors.append("Embedding returned empty")
    except Exception as e:
        errors.append(f"Embedding: {e}")
        click.secho(f"  Embedding: FAILED ({e})", fg="red")

    if errors:
        click.secho(f"\n{len(errors)} validation error(s)", fg="red")
        sys.exit(2)
    else:
        click.secho("\nAll checks passed!", fg="green")
