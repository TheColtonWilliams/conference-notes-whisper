"""Phase 5: Chunk + embed + index transcripts into LanceDB.

Usage:
    python -m pipeline.index                 # incremental index
    python -m pipeline.index --rebuild       # full rebuild
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import typer

from pipeline.utils import (
    collection_transcripts,
    discover_collections,
    ensure_dir,
    format_timestamp,
    load_config,
    resolve_device,
    resolve_path,
    setup_logging,
    sha256_file,
)

app = typer.Typer(add_completion=False)


def _chunk_segments(
    segments: list[dict],
    tokens_per_chunk: int = 512,
    overlap_fraction: float = 0.15,
    respect_speaker_turns: bool = True,
) -> list[dict]:
    """Recursively chunk segments into ~512-token chunks with overlap.

    Returns list of chunk dicts with keys: text, start, end, speakers, segment_ids.
    """
    chars_per_chunk = tokens_per_chunk * 4  # rough estimate
    overlap_chars = int(chars_per_chunk * overlap_fraction)
    chunks = []
    current_texts = []
    current_len = 0
    current_start = None
    current_end = None
    current_speakers = set()
    current_ids = []

    for seg in segments:
        text = seg.get("text", "").strip()
        if not text:
            continue

        seg_len = len(text)

        # Check if adding this segment would exceed chunk size
        if current_len + seg_len > chars_per_chunk and current_texts:
            # Emit chunk
            chunks.append({
                "text": " ".join(current_texts),
                "start": current_start,
                "end": current_end,
                "speakers": list(current_speakers),
                "segment_ids": current_ids[:],
            })

            # Calculate overlap: keep last segments that fit in overlap
            overlap_texts = []
            overlap_len = 0
            overlap_ids = []
            overlap_speakers = set()
            for t, sid, spk in reversed(list(zip(current_texts, current_ids, [seg.get("speaker")] * len(current_ids)))):
                if overlap_len + len(t) > overlap_chars:
                    break
                overlap_texts.insert(0, t)
                overlap_ids.insert(0, sid)
                overlap_len += len(t)

            current_texts = overlap_texts
            current_len = overlap_len
            current_ids = overlap_ids
            current_speakers = overlap_speakers
            current_start = seg["start"] if not overlap_ids else current_start

        if current_start is None:
            current_start = seg.get("start", 0)

        current_texts.append(text)
        current_len += seg_len
        current_end = seg.get("end", 0)
        current_speakers.add(seg.get("speaker_display") or seg.get("speaker", "UNKNOWN"))
        current_ids.append(seg.get("id", 0))

    # Emit final chunk
    if current_texts:
        chunks.append({
            "text": " ".join(current_texts),
            "start": current_start,
            "end": current_end,
            "speakers": list(current_speakers),
            "segment_ids": current_ids[:],
        })

    return chunks


def _load_manifest(manifest_path: Path) -> dict:
    """Load the index manifest tracking what's been indexed."""
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"files": {}, "updated_at": None}


def _save_manifest(manifest_path: Path, manifest: dict) -> None:
    """Save the index manifest."""
    manifest["updated_at"] = datetime.now().isoformat()
    ensure_dir(manifest_path.parent)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


@app.command()
def main(
    rebuild: bool = typer.Option(False, "--rebuild", help="Full rebuild"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be indexed"),
    verbose: bool = typer.Option(False, "--verbose", help="Debug logging"),
) -> None:
    """Index transcript chunks into LanceDB with BGE embeddings."""
    log = setup_logging("index", verbose)
    cfg = load_config()

    transcript_dir = resolve_path("paths.transcripts", cfg)

    chunking = cfg.get("chunking", {})
    tokens_per_chunk = chunking.get("tokens_per_chunk", 512)
    overlap = chunking.get("overlap_fraction", 0.15)
    respect_turns = chunking.get("respect_speaker_turns", True)

    collections = discover_collections(transcript_dir)
    if not collections:
        log.warning("No collections found under %s", transcript_dir)
        raise typer.Exit(1)

    # Plan work per collection: each gets its own LanceDB at index/<collection>/lancedb
    plan: list[tuple[str, dict, dict, Path, Path]] = []  # (collection, to_index, manifest, db_path, manifest_path)
    index_base = Path(resolve_path("paths.index", cfg))
    for collection in collections:
        col_tr = Path(transcript_dir) / collection
        # Don't create the dir here -- planning must be side-effect-free for
        # --dry-run. lancedb.connect() creates it when we actually write.
        index_col = index_base / collection
        db_path = index_col / "lancedb"
        manifest_path = index_col / "manifest.json"

        transcript_paths = collection_transcripts(col_tr)

        manifest = _load_manifest(manifest_path)
        if rebuild:
            manifest = {"files": {}, "updated_at": None}

        to_index = {}
        for stem, tp in transcript_paths.items():
            file_hash = sha256_file(tp)
            if manifest["files"].get(stem, {}).get("sha256") == file_hash and not rebuild:
                log.info("  %s/%s: up to date", collection, stem)
                continue
            to_index[stem] = tp

        if to_index:
            plan.append((collection, to_index, manifest, db_path, manifest_path))

    if not plan:
        log.info("All files up to date, nothing to index")
        return

    if dry_run:
        for collection, to_index, _, _, _ in plan:
            log.info("Would index %d file(s) in [%s]:", len(to_index), collection)
            for stem in to_index:
                log.info("  %s", stem)
        return

    # Load embedding model once, reuse across collections
    log.info("Loading embedding model: %s", cfg["models"]["embedder"])
    from sentence_transformers import SentenceTransformer

    embedder = SentenceTransformer(
        cfg["models"]["embedder"],
        device=resolve_device(cfg),
    )

    import lancedb

    total_records = 0
    total_files = 0
    for collection, to_index, manifest, db_path, manifest_path in plan:
        db = lancedb.connect(str(db_path))

        all_records = []
        for stem, tp in to_index.items():
            with open(tp, "r", encoding="utf-8") as f:
                transcript = json.load(f)

            log.info("  %s/%s: chunking...", collection, stem)
            chunks = _chunk_segments(
                transcript["segments"],
                tokens_per_chunk=tokens_per_chunk,
                overlap_fraction=overlap,
                respect_speaker_turns=respect_turns,
            )
            log.info("  %s/%s: %d chunks, embedding...", collection, stem, len(chunks))

            texts = [c["text"] for c in chunks]
            embeddings = embedder.encode(texts, batch_size=128, show_progress_bar=verbose)

            for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
                all_records.append({
                    "id": f"{stem}_{i}",
                    "file": stem,
                    "collection": collection,
                    "source": transcript.get("source", ""),
                    "chunk_idx": i,
                    "text": chunk["text"],
                    "start": chunk["start"],
                    "end": chunk["end"],
                    "start_ts": format_timestamp(chunk["start"]),
                    "end_ts": format_timestamp(chunk["end"]),
                    "speakers": json.dumps(chunk["speakers"]),
                    "vector": emb.tolist(),
                })

            manifest["files"][stem] = {
                "sha256": sha256_file(tp),
                "chunks": len(chunks),
                "indexed_at": datetime.now().isoformat(),
            }

        if not all_records:
            continue

        log.info("[%s] writing %d records to LanceDB...", collection, len(all_records))
        table_name = "transcripts"
        if rebuild or table_name not in db.table_names():
            tbl = db.create_table(table_name, data=all_records, mode="overwrite")
        else:
            tbl = db.open_table(table_name)
            for stem in to_index:
                try:
                    tbl.delete(f'file = "{stem}"')
                except Exception:
                    pass
            tbl.add(all_records)

        log.info("[%s] creating FTS index...", collection)
        try:
            tbl.create_fts_index("text", replace=True)
        except Exception as e:
            log.warning("[%s] FTS index creation failed (may work on query): %s", collection, e)

        _save_manifest(manifest_path, manifest)
        total_records += len(all_records)
        total_files += len(to_index)

    # Free GPU
    from pipeline.utils import gpu_cleanup
    gpu_cleanup(embedder)

    log.info("Indexing complete: %d records from %d file(s) across %d collection(s)",
             total_records, total_files, len(plan))


if __name__ == "__main__":
    app()
