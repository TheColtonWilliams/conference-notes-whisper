"""Phase 5.4: Hybrid search with optional RAG.

Usage:
    python -m pipeline.search "what were the action items?"
    python -m pipeline.search "threat modeling" --speaker "Jane Doe" --all
    python -m pipeline.search "sprint planning" --rag
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from pipeline.utils import (
    discover_collections,
    load_config,
    resolve_device,
    resolve_path,
    setup_logging,
)

app = typer.Typer(add_completion=False)
console = Console()


def _hybrid_search(
    query: str,
    cfg: dict,
    collections: list[str],
    speaker: str | None = None,
    top_k: int = 50,
) -> list[dict]:
    """Perform hybrid vector + FTS search across one or more collection indexes."""
    import lancedb
    from sentence_transformers import SentenceTransformer

    index_dir = resolve_path("paths.index", cfg)

    # Encode query once, reuse across collection indexes
    embedder = SentenceTransformer(
        cfg["models"]["embedder"],
        device=resolve_device(cfg),
    )
    query_vec = embedder.encode(query).tolist()

    alpha = cfg.get("retrieval", {}).get("hybrid_alpha", 0.6)
    k_rrf = 60

    scores: dict[str, float] = {}
    docs: dict[str, dict] = {}

    for collection in collections:
        db_path = Path(index_dir) / collection / "lancedb"
        if not db_path.exists():
            continue
        db = lancedb.connect(str(db_path))
        if "transcripts" not in db.table_names():
            continue
        tbl = db.open_table("transcripts")

        try:
            vector_results = tbl.search(query_vec).limit(top_k).to_list()
        except Exception:
            vector_results = []
        try:
            fts_results = tbl.search(query, query_type="fts").limit(top_k).to_list()
        except Exception:
            fts_results = []

        # RRF, with collection-prefixed keys so same-named chunks don't collide
        for rank, doc in enumerate(vector_results):
            doc.setdefault("collection", collection)
            doc_id = f"{collection}::{doc.get('id', rank)}"
            scores[doc_id] = scores.get(doc_id, 0) + alpha * (1.0 / (k_rrf + rank + 1))
            docs[doc_id] = doc

        for rank, doc in enumerate(fts_results):
            doc.setdefault("collection", collection)
            doc_id = f"{collection}::{doc.get('id', rank)}"
            scores[doc_id] = scores.get(doc_id, 0) + (1 - alpha) * (1.0 / (k_rrf + rank + 1))
            if doc_id not in docs:
                docs[doc_id] = doc

    # Sort by RRF score
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    results = [docs[doc_id] for doc_id, _ in ranked[:top_k]]

    # Filter by speaker if specified
    if speaker:
        results = [
            r for r in results
            if any(speaker.lower() in s.lower() for s in json.loads(r.get("speakers") or "[]"))
        ]

    # Free embedder
    from pipeline.utils import gpu_cleanup
    gpu_cleanup(embedder)

    return results


def _rerank(
    query: str,
    results: list[dict],
    cfg: dict,
    rerank_k: int = 10,
) -> list[dict]:
    """Rerank results using BGE cross-encoder."""
    if not results:
        return results

    from sentence_transformers import CrossEncoder

    reranker = CrossEncoder(
        cfg["models"]["reranker"],
        device=resolve_device(cfg),
    )

    pairs = [(query, r["text"]) for r in results]
    scores = reranker.predict(pairs)

    for r, score in zip(results, scores):
        r["rerank_score"] = float(score)

    results.sort(key=lambda r: r.get("rerank_score", 0), reverse=True)

    from pipeline.utils import gpu_cleanup
    gpu_cleanup(reranker)

    return results[:rerank_k]


def _rag_answer(
    query: str,
    context_chunks: list[dict],
    cfg: dict,
    log,
    collection: str | None = None,
) -> str:
    """Generate RAG answer using Ollama."""
    import ollama as ollama_client

    context_parts = []
    for c in context_chunks:
        context_parts.append(
            f"[{c.get('source', c.get('file', '?'))} @ {c.get('start_ts', '?')}]\n{c['text']}"
        )
    context = "\n\n---\n\n".join(context_parts)

    prompt = f"""Answer the question based ONLY on the transcript excerpts below.
Cite your sources using [filename @ HH:MM:SS] format.
If the answer cannot be found in the excerpts, say "I couldn't find relevant information."

## Transcript excerpts

{context}

## Question

{query}

## Answer
"""

    model = cfg["models"]["llm"]
    response = ollama_client.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options={
            "num_ctx": cfg.get("ollama", {}).get("num_ctx", 32768),
            "temperature": 0.1,
        },
    )

    answer = response["message"]["content"]

    # Log to qa_log.md (per-collection when scoped to one, else at kb root)
    kb_dir = resolve_path("paths.kb", cfg)
    if collection:
        log_dir = Path(kb_dir) / collection
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "qa_log.md"
    else:
        log_path = Path(kb_dir) / "qa_log.md"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"\n---\n\n**Q ({datetime.now().isoformat()}):** {query}\n\n")
        f.write(f"**A:** {answer}\n\n")
        f.write(f"**Sources:** {', '.join(c.get('id', '?') for c in context_chunks)}\n")

    return answer


@app.command()
def main(
    query: str = typer.Argument(..., help="Search query"),
    collection: str = typer.Option(None, "--collection", help="Collection to search"),
    all_collections: bool = typer.Option(False, "--all", help="Search across all collections"),
    speaker: str = typer.Option(None, "--speaker", help="Filter by speaker name"),
    after: str = typer.Option(None, "--after", help="Filter results after timestamp HH:MM:SS"),
    rag: bool = typer.Option(False, "--rag", help="Generate RAG answer"),
    verbose: bool = typer.Option(False, "--verbose", help="Debug logging"),
) -> None:
    """Search transcripts with hybrid vector + BM25 retrieval."""
    log = setup_logging("search", verbose)
    cfg = load_config()

    retrieval = cfg.get("retrieval", {})
    top_k = retrieval.get("top_k", 50)
    rerank_k = retrieval.get("rerank_k", 10)
    rag_k = retrieval.get("rag_context_k", 5)

    # Resolve which collection(s) to search
    index_dir = resolve_path("paths.index", cfg)
    available = discover_collections(index_dir)
    if not available:
        console.print("[yellow]No indexed collections found. Run 'python -m pipeline.index' first.[/yellow]")
        raise typer.Exit(1)

    if all_collections:
        target_collections = available
        scope_label = "all"
    elif collection:
        if collection not in available:
            console.print(
                f"[red]Collection '{collection}' not indexed.[/red] Available: {', '.join(available)}"
            )
            raise typer.Exit(1)
        target_collections = [collection]
        scope_label = collection
    elif len(available) == 1:
        target_collections = available
        scope_label = available[0]
    else:
        console.print(
            "[red]Multiple collections exist.[/red] Specify --collection <name> or --all. "
            f"Available: {', '.join(available)}"
        )
        raise typer.Exit(1)

    # Search
    log.info("Searching [%s]: '%s'", scope_label, query)
    results = _hybrid_search(query, cfg, target_collections, speaker=speaker, top_k=top_k)
    log.info("Found %d initial results", len(results))

    # Rerank
    if results:
        results = _rerank(query, results, cfg, rerank_k=rerank_k)
        log.info("Reranked to %d results", len(results))

    # Filter by timestamp
    if after and results:
        h, m, s = map(int, after.split(":"))
        after_sec = h * 3600 + m * 60 + s
        results = [r for r in results if r.get("start", 0) >= after_sec]

    # Display
    if not results:
        console.print("[yellow]No results found.[/yellow]")
        return

    table = Table(title=f"Search [{scope_label}]: {query}", show_lines=True)
    table.add_column("#", style="dim", width=3)
    table.add_column("Collection", width=16)
    table.add_column("File", width=18)
    table.add_column("Time", width=12)
    table.add_column("Score", width=8)
    table.add_column("Text", width=54)

    for i, r in enumerate(results):
        table.add_row(
            str(i + 1),
            r.get("collection", "?"),
            r.get("file", "?"),
            f"{r.get('start_ts', '?')} - {r.get('end_ts', '?')}",
            f"{r.get('rerank_score', 0):.3f}",
            r["text"][:200] + ("..." if len(r["text"]) > 200 else ""),
        )

    console.print(table)

    # RAG mode
    if rag:
        console.print("\n[bold]Generating answer...[/bold]")
        context = results[:rag_k]
        qa_collection = target_collections[0] if len(target_collections) == 1 else None
        answer = _rag_answer(query, context, cfg, log, collection=qa_collection)
        console.print(Panel(answer, title="RAG Answer", border_style="green"))


if __name__ == "__main__":
    app()
