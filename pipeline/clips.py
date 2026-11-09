"""Phase 3.6: Short audio clips that help a human put a name to a voice.

``names`` proposes names from what is said. When nobody introduces a speaker, a
person has to listen, and hunting through hours of audio for one voice is slow.
This stage cuts one MP3 per voice from its longest turns -- one per recording
before a second from the same recording -- and writes an INDEX.md listing each
snippet's recording, timestamp and words. Naming a voice becomes one listen.

    data/speaker-id/<collection>/clips/<pid>.mp3          <- one per unnamed voice
    data/speaker-id/<collection>/clips/INDEX.md           <- what each clip holds, longest talker first
    data/speaker-id/<collection>/clips/.meta.json         <- idempotency + ownership
    data/speaker-id/<collection>/clips/selected/...       <- same layout; holds the latest --person /
                                                             --all-people run that touched this collection

A clip is evidence, not proof: diarization can merge two similar voices, so if the
snippets in one clip sound like different people, note that before naming it.

Re-running with no change writes nothing. When something changes, only files this
stage wrote (listed in .meta.json) are replaced or removed. If INDEX.md was edited
by hand (people write names straight into it), it is kept as
INDEX.edited-<timestamp>.md first. A clip that is open in a media player and can't
be replaced is left alone and retried on the next run.

Usage:
    python -m pipeline.clips                       # every collection, unnamed voices
    python -m pipeline.clips defcon-2026           # only these collections
    python -m pipeline.clips --person spk_0006     # specific people, into clips/selected/
    python -m pipeline.clips --all-people          # every voice, into clips/selected/
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import typer

from pipeline import people as ppl
from pipeline.utils import format_timestamp, load_config, project_root, resolve_path, setup_logging

app = typer.Typer(add_completion=False)

_DEFAULTS = {
    "per_voice": 4,
    "min_seconds": 3.0,
    "max_seconds": 15.0,
    "pad_seconds": 0.25,
    "gap_seconds": 1.0,
    "mp3_quality": 4,
}
_SAMPLE_RATE = 16000
_MIN_SEGMENT = 0.3  # turns shorter than this are never used (and ffmpeg reads -t 0 as "no limit")


def _clips_cfg(cfg: dict) -> dict:
    """Merged clips settings, validated. Raises ValueError with a readable message."""
    c = {**_DEFAULTS, **(cfg.get("clips") or {})}
    try:
        out = {
            "per_voice": int(c["per_voice"]),
            "min_seconds": float(c["min_seconds"]),
            "max_seconds": float(c["max_seconds"]),
            "pad_seconds": float(c["pad_seconds"]),
            "gap_seconds": float(c["gap_seconds"]),
            "mp3_quality": int(c["mp3_quality"]),
        }
    except (TypeError, ValueError) as e:
        raise ValueError(f"clips settings must be numbers: {e}") from e
    problems = []
    if out["per_voice"] < 1:
        problems.append("per_voice must be >= 1")
    if out["min_seconds"] < _MIN_SEGMENT:
        problems.append(f"min_seconds must be >= {_MIN_SEGMENT}")
    if out["max_seconds"] < out["min_seconds"]:
        problems.append("max_seconds must be >= min_seconds")
    if out["pad_seconds"] < 0 or out["gap_seconds"] < 0:
        problems.append("pad_seconds and gap_seconds must be >= 0")
    if not 0 <= out["mp3_quality"] <= 9:
        problems.append("mp3_quality must be 0..9")
    if problems:
        raise ValueError("; ".join(problems))
    return out


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _audio_source(cfg: dict, collection: str, stem: str) -> Path | None:
    """Prefer the normalized WAV; fall back to the original recording (inbox or archive)."""
    wav = Path(resolve_path("paths.normalized", cfg)) / collection / f"{stem}.wav"
    if wav.exists():
        return wav
    exts = cfg.get("audio_extensions", [".m4a", ".mp3", ".wav", ".flac", ".ogg"])
    for key in ("paths.raw_audio", "paths.processed"):
        base = Path(resolve_path(key, cfg)) / collection
        for ext in exts:
            p = base / f"{stem}{ext}"
            if p.exists():
                return p
    return None


def _source_id(path: Path) -> str:
    """Location-independent id for the signature: path below the project root, else the name."""
    try:
        return path.resolve().relative_to(project_root().resolve()).as_posix()
    except ValueError:
        return path.name


def pick_snippets(appearances: list[dict], seg_lookup, ccfg: dict) -> list[dict]:
    """Choose up to ``per_voice`` snippets for one voice.

    Pass 1 takes one snippet per recording, preferring a turn of
    [min_seconds, max_seconds], then a longer turn trimmed to max_seconds, then a
    shorter turn. Pass 2 fills the remaining slots in the same order. Longest
    first within each tier. Returned in recording/time order.
    """
    n, lo, hi = ccfg["per_voice"], ccfg["min_seconds"], ccfg["max_seconds"]
    fits, longer, shorter = [], [], []
    for a in appearances:
        for s in seg_lookup(a["file"]):
            if s.get("speaker") != a["local_id"] or not (s.get("text") or "").strip():
                continue
            d = float(s["end"]) - float(s["start"])
            if d < _MIN_SEGMENT:
                continue
            cand = {"file": a["file"], "start": float(s["start"]), "end": float(s["end"]),
                    "text": s["text"].strip(), "dur": d}
            (fits if lo <= d <= hi else longer if d > hi else shorter).append(cand)
    tiers = []
    for pool, trim in ((fits, False), (longer, True), (shorter, False)):
        pool.sort(key=lambda c: (-c["dur"], c["file"], c["start"]))
        tiers.append((pool, trim))

    picks: list[dict] = []
    taken: set[tuple[str, float]] = set()
    for first_pass in (True, False):
        for pool, trim in tiers:
            for c in pool:
                if len(picks) >= n:
                    break
                key = (c["file"], c["start"])
                if key in taken or (first_pass and any(p["file"] == c["file"] for p in picks)):
                    continue
                taken.add(key)
                picks.append({**c, "end": c["start"] + hi, "dur": hi} if trim else c)
    return sorted(picks, key=lambda c: (c["file"], c["start"]))


def _ffmpeg_concat(snips: list[dict], out: Path, ccfg: dict) -> None:
    """Cut each snippet from its source and join them, with short silences, into one MP3."""
    pad, gap = ccfg["pad_seconds"], ccfg["gap_seconds"]
    args = ["ffmpeg", "-y", "-v", "error"]
    total = 0.0
    for s in snips:
        st = max(0.0, s["start"] - pad)
        dur = max(_MIN_SEGMENT, (s["end"] + pad) - st)
        total += dur
        args += ["-ss", f"{st:.2f}", "-t", f"{dur:.2f}", "-i", str(s["source"])]
    k = len(snips)
    use_gap = k > 1 and gap >= 0.01  # "-t 0.00" on anullsrc would never end
    norm = f"aresample={_SAMPLE_RATE},aformat=sample_fmts=fltp:channel_layouts=mono"
    graph = [f"[{i}:a]{norm}[c{i}]" for i in range(k)]
    if use_gap:
        args += ["-f", "lavfi", "-t", f"{gap:.2f}", "-i", f"anullsrc=r={_SAMPLE_RATE}:cl=mono"]
        graph.append(f"[{k}:a]{norm},asplit={k - 1}" + "".join(f"[g{i}]" for i in range(k - 1)))
        chain = "".join(f"[c{i}][g{i}]" if i < k - 1 else f"[c{i}]" for i in range(k))
        n_parts = 2 * k - 1
        total += gap * (k - 1)
    else:
        chain = "".join(f"[c{i}]" for i in range(k))
        n_parts = k
    graph.append(f"{chain}concat=n={n_parts}:v=0:a=1[out]")
    args += ["-filter_complex", ";".join(graph), "-map", "[out]",
             "-c:a", "libmp3lame", "-q:a", str(ccfg["mp3_quality"]), "-f", "mp3", str(out)]
    subprocess.run(args, check=True, capture_output=True, text=True, encoding="utf-8",
                   errors="replace", timeout=120 + 3 * total)


def _render_index(collection: str, voices: list[dict], selected: bool) -> str:
    title = "Selected voices" if selected else "Voices to identify"
    lines = [
        f"# {title}: {collection}",
        "",
        "Listen to a clip, then name the voice in `people.yaml` (`display_name`) "
        "or accept a proposal with `python -m pipeline.names --accept <id>`, and run "
        "`python -m pipeline.rename_speakers`. Same person under two ids? "
        "`python -m pipeline.speakers --merge <id> --into <id>`.",
        "",
        "Snippets come from different recordings where possible. If one clip sounds like "
        "two different people, diarization merged two voices: note it before naming.",
        "",
        "_Generated by `python -m pipeline.clips`. If you write notes in this file, they are "
        "kept as `INDEX.edited-<timestamp>.md` when the clips are rebuilt._",
        "",
    ]
    for rank, v in enumerate(voices, 1):
        name = f" ({v['name']})" if v["name"] != v["pid"] else ""
        lines.append(f"## {rank:02d} · {v['clip']}")
        meta = (f"**{v['pid']}**{name} · {v['n_recordings']} recording(s) here · "
                f"{v['minutes']:.1f} min total")
        if v.get("proposal"):
            meta += f" · proposed: {v['proposal']['name']} (score {v['proposal']['score']})"
        lines += [meta, ""]
        for i, s in enumerate(v["snippets"], 1):
            text = s["text"] if len(s["text"]) <= 160 else s["text"][:157] + "..."
            lines.append(f"{i}. `{s['file']}` @ {format_timestamp(s['start'])}: \"{text}\"")
        if not v["snippets"]:
            lines.append(f"_No clip: {v['why_empty']}._")
        lines.append("")
    return "\n".join(lines)


def _read_meta(path: Path) -> dict:
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(meta, dict):
        return {}
    if not isinstance(meta.get("clips"), dict):
        meta["clips"] = {}
    if not isinstance(meta.get("pending_removal"), list):
        meta["pending_removal"] = []
    return meta


def build_collection(cfg: dict, ccfg: dict, collection: str, people: dict, only: set[str] | None,
                     all_people: bool, force: bool, dry_run: bool, log) -> int:
    """Build or refresh one collection's clips. Returns the number of files written."""
    selected = only is not None or all_people
    out_dir = Path(resolve_path("paths.speaker_id", cfg)) / collection / "clips"
    if selected:
        out_dir = out_dir / "selected"
    meta_path = out_dir / ".meta.json"
    index_path = out_dir / "INDEX.md"
    seg_cache: dict = {}

    def segments(stem: str) -> list[dict]:
        if stem not in seg_cache:
            tp = Path(resolve_path("paths.transcripts", cfg)) / collection / f"{stem}.json"
            try:
                with open(tp, "r", encoding="utf-8") as f:
                    seg_cache[stem] = json.load(f).get("segments", [])
            except (OSError, json.JSONDecodeError) as e:
                log.warning("  %s: transcript for %s unreadable (%s) -- skipped", collection, stem, e)
                seg_cache[stem] = []
        return seg_cache[stem]

    voices = []
    for pid, person in people.items():
        if only is not None and pid not in only:
            continue
        if not selected and ppl.is_named(pid, person):
            continue
        apps = [a for a in (person or {}).get("appearances", []) if a["collection"] == collection]
        if not apps:
            continue
        sources = {a["file"]: _audio_source(cfg, collection, a["file"]) for a in apps}
        no_audio = sorted(f for f, s in sources.items() if s is None)
        if no_audio:
            log.warning("  %s/%s: no audio for %s -- skipped", collection, pid, ", ".join(no_audio))
        usable = [a for a in apps if sources[a["file"]] is not None]
        snips = pick_snippets(usable, segments, ccfg)
        for s in snips:
            s["source"] = sources[s["file"]]
        total = sum(a.get("speaking_time_sec", 0) for a in apps)
        props = person.get("proposed_names") or []
        voices.append({
            "pid": pid, "name": ppl.display_name(pid, person), "total": total, "minutes": total / 60,
            "n_recordings": len({a["file"] for a in apps}), "clip": f"{pid}.mp3", "snippets": snips,
            "proposal": {"name": props[0]["name"], "score": props[0]["score"]} if props else None,
            "why_empty": ("no audio found for its recordings" if not usable
                          else "no usable turns in the transcripts"),
            "sig": _sha(json.dumps({"cfg": ccfg, "snips": [
                (s["file"], round(s["start"], 2), round(s["end"], 2), _source_id(s["source"]),
                 s["source"].stat().st_size) for s in snips]}, sort_keys=True)),
        })
    voices.sort(key=lambda v: (-v["total"], v["pid"]))

    index_text = _render_index(collection, voices, selected)
    old = _read_meta(meta_path)
    old_clips: dict = old.get("clips", {})
    new_clips = {v["clip"]: v["sig"] for v in voices if v["snippets"]}
    # A name that is wanted again is never removed (its rebuild replaces the file atomically).
    pending = [c for c in set(old.get("pending_removal", [])) | (set(old_clips) - set(new_clips))
               if c not in new_clips and (out_dir / c).exists()]
    todo = [v for v in voices if v["snippets"]
            and (force or old_clips.get(v["clip"]) != v["sig"] or not (out_dir / v["clip"]).exists())]
    on_disk = index_path.read_bytes().decode("utf-8", "replace") if index_path.exists() else None
    if not (force or todo or pending or on_disk is None or old.get("index_sha") != _sha(index_text)
            or old_clips != new_clips):
        log.info("  %s: %d clip(s) up to date%s", collection, len(new_clips), " (selected)" if selected else "")
        return 0
    edited = on_disk is not None and old.get("index_sha") != _sha(on_disk)

    if dry_run:
        for v in todo:
            log.info("  %s: would write %s (%d snippet(s))", collection, v["clip"], len(v["snippets"]))
        for c in pending:
            log.info("  %s: would remove %s (no longer needed)", collection, c)
        if edited:
            log.info("  %s: INDEX.md has hand edits -- would keep it as INDEX.edited-<timestamp>.md", collection)
        log.info("  %s: would write INDEX.md (%d voice(s))", collection, len(voices))
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    for part in out_dir.glob("*.part.mp3"):  # leftovers from an interrupted run (ours by name)
        try:
            part.unlink()
        except OSError:
            pass

    writes = 0
    still_pending = []
    for c in pending:  # only files this stage wrote (listed in its own meta)
        try:
            (out_dir / c).unlink()
            log.debug("  %s: removed %s", collection, c)
        except OSError as e:
            still_pending.append(c)
            log.warning("  %s: can't remove %s (%s) -- close it in your player; retried next run",
                        collection, c, e.strerror or e)

    done = {c: s for c, s in old_clips.items() if c in new_clips and (out_dir / c).exists()}
    for v in todo:
        final, part = out_dir / v["clip"], out_dir / (v["clip"][:-4] + ".part.mp3")
        log.info("  %s: %s (%d snippet(s))", collection, v["clip"], len(v["snippets"]))
        try:
            _ffmpeg_concat(v["snippets"], part, ccfg)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            part.unlink(missing_ok=True)
            err = getattr(e, "stderr", "") or ""
            log.error("ffmpeg failed for %s/%s: %s", collection, v["pid"], (err or str(e)).strip()[-400:])
            raise typer.Exit(1)
        try:
            part.replace(final)
        except OSError as e:
            part.unlink(missing_ok=True)
            log.warning("  %s: can't replace %s (%s) -- close it in your player; retried next run",
                        collection, v["clip"], e.strerror or e)
            continue
        done[v["clip"]] = v["sig"]
        writes += 1

    index_sha = old.get("index_sha")
    try:
        if edited:
            keep = out_dir / f"INDEX.edited-{datetime.now().strftime('%Y%m%d-%H%M%S')}.md"
            index_path.replace(keep)
            log.warning("  %s: INDEX.md had hand edits -- kept as %s", collection, keep.name)
        # newline="\n" so the stored hash matches the bytes on disk on Windows too
        index_path.write_text(index_text, encoding="utf-8", newline="\n")
        index_sha = _sha(index_text)
        writes += 1
    except OSError as e:
        log.warning("  %s: couldn't update INDEX.md (%s) -- retried next run", collection, e.strerror or e)

    meta_path.write_text(json.dumps({
        "generated_at": datetime.now().isoformat(),
        "index_sha": index_sha,
        "clips": done,
        "pending_removal": still_pending,
    }, indent=2), encoding="utf-8", newline="\n")
    log.info("  %s: %d file(s) written, %d voice(s) listed -> %s", collection, writes + 1, len(voices), index_path)
    return writes + 1


@app.command()
def main(
    collections: list[str] = typer.Argument(None, help="Only these collections (default: all)"),
    person: list[str] = typer.Option(None, "--person", help="Only this person id (repeatable); writes clips/selected/"),
    all_people: bool = typer.Option(False, "--all-people", help="Every voice, named or not; writes clips/selected/"),
    force: bool = typer.Option(False, "--force", help="Rebuild clips even if nothing changed"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be written"),
    verbose: bool = typer.Option(False, "--verbose", help="Debug logging"),
) -> None:
    """Cut one short MP3 per voice (plus an INDEX.md) so a human can name it by ear."""
    log = setup_logging("clips", verbose)
    cfg = load_config()
    try:
        ccfg = _clips_cfg(cfg)
    except ValueError as e:
        log.error("Invalid clips settings in config: %s", e)
        raise typer.Exit(1)
    if shutil.which("ffmpeg") is None:
        log.error("ffmpeg not found on PATH")
        raise typer.Exit(1)

    people = ppl.load_people(cfg)
    if not people:
        log.info("No people database yet. Run 'python -m pipeline.speakers' first.")
        return
    only = set(person) if person else None
    if only:
        unknown = sorted(only - set(people))
        if unknown:
            log.error("Unknown person id(s): %s", ", ".join(unknown))
            raise typer.Exit(1)

    pool = [p for pid, p in people.items() if only is None or pid in only]
    known = sorted({a["collection"] for p in pool for a in (p or {}).get("appearances", [])})
    for c in collections or []:
        if c not in known:
            log.warning("No matching voices recorded for collection %r", c)
    targets = [c for c in known if not collections or c in collections]

    writes = 0
    for collection in targets:
        writes += build_collection(cfg, ccfg, collection, people, only, all_people, force, dry_run, log)
    if not dry_run:
        log.info("Clips complete -- %d file(s) written", writes)


if __name__ == "__main__":
    app()
