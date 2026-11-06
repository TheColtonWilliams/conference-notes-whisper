"""Apply people.yaml display names to every transcript in every collection.

Names come from the global people database (``data/speaker-id/people.yaml``),
so naming a speaker once labels them in every talk they appear in. The
``.md``/``.vtt`` transcripts are re-rendered from the JSON rather than patched
with regexes, so renaming the same person twice works as expected.

Usage:
    python -m pipeline.rename_speakers              # rename all
    python -m pipeline.rename_speakers --dry-run    # show what would change
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from pipeline import people as ppl
from pipeline.utils import (
    discover_collections,
    load_config,
    resolve_path,
    setup_logging,
)

app = typer.Typer(add_completion=False)


def _apply_names(data: dict, names: dict[str, tuple[str, str]]) -> bool:
    """Set display_name/global_id on speakers and speaker_display on segments.

    ``names`` maps a transcript-local speaker id to ``(person_id, display)``.
    Returns True if anything changed.
    """
    changed = False
    for spk in data.get("speakers", []):
        if spk["id"] not in names:
            continue
        pid, display = names[spk["id"]]
        if spk.get("display_name") != display or spk.get("global_id") != pid:
            spk["display_name"] = display
            spk["global_id"] = pid
            changed = True
    for seg in data.get("segments", []):
        if seg.get("speaker") in names:
            display = names[seg["speaker"]][1]
            if seg.get("speaker_display") != display:
                seg["speaker_display"] = display
                changed = True
    return changed


def _write_if_changed(path: Path, content: str) -> bool:
    if path.exists() and path.read_text(encoding="utf-8") == content:
        return False
    path.write_text(content, encoding="utf-8")
    return True


def _rerender(raw: dict, col_dir: Path, stem: str, cfg: dict) -> int:
    """Re-render <stem>.md / .vtt from the raw transcript with display names."""
    from pipeline.transcribe import _render_markdown, _render_vtt

    view = dict(raw)
    view["segments"] = [
        {**seg, "speaker": seg.get("speaker_display") or seg.get("speaker")}
        for seg in raw.get("segments", [])
    ]
    changes = 0
    changes += _write_if_changed(col_dir / f"{stem}.md", _render_markdown(view, cfg))
    changes += _write_if_changed(col_dir / f"{stem}.vtt", _render_vtt(view))
    return changes


@app.command()
def main(
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would change"),
    verbose: bool = typer.Option(False, "--verbose", help="Debug logging"),
) -> None:
    """Apply people.yaml speaker names to all transcript files."""
    log = setup_logging("rename_speakers", verbose)
    cfg = load_config()

    transcript_dir = resolve_path("paths.transcripts", cfg)
    collections = discover_collections(transcript_dir)
    if not collections:
        log.warning("No collections found under %s", transcript_dir)
        raise typer.Exit(1)

    people = ppl.load_people(cfg)
    if not people:
        log.info("No people database yet -- skipping (run 'python -m pipeline.speakers')")
        return
    index = ppl.appearance_index(people)

    total_changes = 0
    for collection in collections:
        col_dir = Path(transcript_dir) / collection
        for raw_path in sorted(col_dir.glob("*.json")):
            if raw_path.name.endswith((".meta.json", ".clean.json")):
                continue
            stem = raw_path.stem
            names = {
                local_id: (pid, ppl.display_name(pid, people[pid]))
                for (col, file, local_id), pid in index.items()
                if col == collection and file == stem
            }
            if not names:
                continue

            for json_path in (raw_path, col_dir / f"{stem}.clean.json"):
                if not json_path.exists():
                    continue
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if not _apply_names(data, names):
                    continue
                if dry_run:
                    log.info("  Would rename speakers in %s/%s", collection, json_path.name)
                    continue
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                total_changes += 1

            if not dry_run:
                with open(raw_path, "r", encoding="utf-8") as f:
                    total_changes += _rerender(json.load(f), col_dir, stem, cfg)

        if not dry_run and any(col == collection for col, _, _ in index):
            ppl.write_collection_view(cfg, collection, people)

    log.info("Renamed speakers: %d file(s) changed across %d collection(s)", total_changes, len(collections))


if __name__ == "__main__":
    app()
