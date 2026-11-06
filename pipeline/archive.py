"""Final phase: archive completed recordings out of the inbox.

A recording is "complete" only when it has passed the *entire* chain --
preprocess, transcribe, knowledge-base extraction, and indexing -- with every
output current for the source's current bytes. Completed source audio is moved
from ``recordings/<collection>/`` to ``processed/<collection>/`` so it is never
rediscovered (the primary "don't reprocess" gate). Anything that failed partway
stays in the inbox and is retried on the next run.

Usage:
    python -m pipeline.archive                # move all completed recordings
    python -m pipeline.archive --dry-run      # show what would move, change nothing
    python -m pipeline.archive "keynote.m4a"           # only this file
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from pipeline.utils import (
    collection_root,
    discover_sources,
    load_config,
    preferred_transcript,
    resolve_path,
    setup_logging,
    sha256_file,
)

app = typer.Typer(add_completion=False)


def _meta_sha_matches(meta_path: Path, expected_sha: str) -> bool:
    """True if a stage's .meta.json records source_sha256 == expected_sha."""
    if not meta_path.exists():
        return False
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except (json.JSONDecodeError, OSError):
        return False
    return meta.get("source_sha256") == expected_sha


def _completion_status(source: Path, cfg: dict) -> tuple[bool, str]:
    """Check whether every downstream artifact for ``source`` is present & current.

    Returns (complete, reason). ``reason`` names the first missing/stale stage.
    """
    collection = source.parent.name
    stem = source.stem
    src_sha = sha256_file(source)

    norm_dir = Path(resolve_path("paths.normalized", cfg)) / collection
    tr_dir = Path(resolve_path("paths.transcripts", cfg)) / collection
    kb_dir = Path(resolve_path("paths.kb", cfg)) / collection
    index_dir = Path(resolve_path("paths.index", cfg)) / collection

    # 1. preprocess -- normalized WAV current for the source bytes
    if not (norm_dir / f"{stem}.wav").exists():
        return False, "preprocess (no WAV)"
    if not _meta_sha_matches(norm_dir / f"{stem}.meta.json", src_sha):
        return False, "preprocess (stale)"

    # 2. transcribe -- transcript current for the source bytes
    if not (tr_dir / f"{stem}.json").exists():
        return False, "transcribe (no transcript)"
    if not _meta_sha_matches(tr_dir / f"{stem}.meta.json", src_sha):
        return False, "transcribe (stale)"

    # The transcript actually consumed downstream (prefer LLM-corrected)
    tp = preferred_transcript(tr_dir, stem)
    tp_sha = sha256_file(tp)

    # 3. knowledge base -- entry current for the transcript it was built from
    if not (kb_dir / f"{stem}.kb.json").exists() or not (kb_dir / f"{stem}.md").exists():
        return False, "kb (no entry)"
    if not _meta_sha_matches(kb_dir / f"{stem}.meta.json", tp_sha):
        return False, "kb (stale)"

    # 4. index -- manifest records this stem at the transcript's current hash
    manifest_path = index_dir / "manifest.json"
    if not manifest_path.exists():
        return False, "index (no manifest)"
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except (json.JSONDecodeError, OSError):
        return False, "index (unreadable manifest)"
    if manifest.get("files", {}).get(stem, {}).get("sha256") != tp_sha:
        return False, "index (not indexed / stale)"

    return True, "complete"


@app.command()
def main(
    files: list[str] = typer.Argument(None, help="Specific files to consider"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would move, change nothing"),
    verbose: bool = typer.Option(False, "--verbose", help="Debug logging"),
) -> None:
    """Move fully-processed recordings from the inbox to the processed archive."""
    log = setup_logging("archive", verbose)
    cfg = load_config()

    raw_dir = resolve_path("paths.raw_audio", cfg)
    extensions = cfg.get("audio_extensions", [".m4a", ".mp3", ".wav", ".flac", ".ogg"])

    sources = discover_sources(raw_dir, extensions=extensions, specific_files=files)
    if not sources:
        log.info("Inbox is empty -- nothing to archive")
        return

    moved = 0
    kept = 0
    for source in sources:
        collection = source.parent.name
        complete, reason = _completion_status(source, cfg)

        if not complete:
            kept += 1
            log.info("  KEEP  %s/%s -- incomplete: %s", collection, source.name, reason)
            continue

        dest_dir = collection_root(cfg, "paths.processed", collection)
        dest = dest_dir / source.name

        if dry_run:
            log.info("  MOVE  %s/%s -> %s", collection, source.name, dest)
            moved += 1
            continue

        # Overwrite a stale archive copy if one somehow exists
        if dest.exists():
            dest.unlink()
        source.replace(dest)
        log.info("  MOVE  %s/%s -> processed/%s/", collection, source.name, collection)
        moved += 1

    verb = "would move" if dry_run else "moved"
    log.info("Archive complete -- %s %d, kept %d in inbox", verb, moved, kept)


if __name__ == "__main__":
    app()
