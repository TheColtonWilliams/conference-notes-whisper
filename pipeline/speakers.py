"""Phase 3: Speaker identification via SpeechBrain ECAPA-TDNN.

Voices are clustered within each collection, then every cluster is matched
against the global people database (``data/speaker-id/people.yaml``, see
``pipeline/people.py``) so the same speaker is recognized across conferences.
Unmatched clusters become new, unnamed people.

Usage:
    python -m pipeline.speakers                              # cluster + match all collections
    python -m pipeline.speakers --list-unknown               # unnamed people with sample quotes
    python -m pipeline.speakers --merge spk_0007 --into spk_0002   # same person, two ids
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import typer
import yaml

from pipeline import people as ppl
from pipeline.utils import (
    discover_collections,
    load_config,
    resolve_device,
    resolve_path,
    setup_logging,
)

app = typer.Typer(add_completion=False)


def _extract_speaker_audio_segments(
    wav_path: Path,
    transcript: dict,
    speaker_id: str,
    cfg: dict,
) -> list[tuple[float, float]]:
    """Get time segments for a speaker from a transcript."""
    segments = []
    for seg in transcript.get("segments", []):
        if seg.get("speaker") == speaker_id:
            segments.append((seg["start"], seg["end"]))
    return segments


def _compute_speaker_embedding(
    wav_path: Path,
    time_segments: list[tuple[float, float]],
    classifier,
    cfg: dict,
) -> np.ndarray | None:
    """Compute a single embedding for a speaker by averaging over their segments."""
    import soundfile as sf
    import torch

    speaker_cfg = cfg.get("speakers", {})
    min_seconds = speaker_cfg.get("min_speech_seconds", 30)
    max_seconds = speaker_cfg.get("max_speech_seconds", 90)

    # Collect audio for this speaker
    data, sr = sf.read(str(wav_path))
    total_seconds = 0
    chunks = []

    for start, end in sorted(time_segments):
        s_idx = int(start * sr)
        e_idx = int(end * sr)
        chunk = data[s_idx:e_idx]
        chunks.append(chunk)
        total_seconds += end - start
        if total_seconds >= max_seconds:
            break

    if total_seconds < min_seconds:
        return None

    audio = np.concatenate(chunks)
    waveform = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        embedding = classifier.encode_batch(waveform)

    return embedding.squeeze().cpu().numpy()


def _cluster_embeddings(
    embeddings: dict[str, np.ndarray],
    cfg: dict,
) -> dict[str, str]:
    """Cluster speaker embeddings and return mapping of file_speaker -> cluster_id."""
    from sklearn.cluster import AgglomerativeClustering

    speaker_cfg = cfg.get("speakers", {})
    threshold = speaker_cfg.get("distance_threshold", 0.40)

    keys = list(embeddings.keys())
    if len(keys) <= 1:
        return {k: "c0" for k in keys}

    X = np.array([embeddings[k] for k in keys])

    # Normalize embeddings for cosine distance
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1
    X_norm = X / norms

    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric="cosine",
        linkage="average",
        distance_threshold=threshold,
    )
    labels = clustering.fit_predict(X_norm)

    return {k: f"c{l}" for k, l in zip(keys, labels)}


def _get_sample_quotes(transcript: dict, speaker_id: str, n: int = 3) -> list[str]:
    """Get sample quotes from a speaker for identification."""
    quotes = []
    for seg in transcript.get("segments", []):
        if seg.get("speaker") == speaker_id and len(seg.get("text", "")) > 20:
            quotes.append(seg["text"][:100])
            if len(quotes) >= n:
                break
    return quotes


def _inputs_fingerprint(
    transcript_dir: Path, collection: str, transcript_paths: list[Path], cfg: dict
) -> str:
    """Fingerprint a collection's clustering inputs for idempotency.

    Built from each transcript's *audio* sha256 (read from the transcribe
    .meta.json) plus the speaker config -- NOT the transcript JSON itself, so
    that rename_speakers rewriting transcripts does not invalidate the cache.
    """
    col_dir = Path(transcript_dir) / collection
    audio_shas: dict[str, str | None] = {}
    for tp in transcript_paths:
        meta = col_dir / f"{tp.stem}.meta.json"
        sha = None
        if meta.exists():
            try:
                with open(meta, "r", encoding="utf-8") as f:
                    sha = json.load(f).get("source_sha256")
            except (json.JSONDecodeError, OSError):
                sha = None
        audio_shas[tp.stem] = sha

    speaker_cfg = cfg.get("speakers", {})
    payload = {
        "audio": audio_shas,
        "model": speaker_cfg.get("model"),
        "distance_threshold": speaker_cfg.get("distance_threshold"),
        "match_threshold": speaker_cfg.get("match_threshold"),
        "min_speech_seconds": speaker_cfg.get("min_speech_seconds"),
        "max_speech_seconds": speaker_cfg.get("max_speech_seconds"),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def _merge_people(cfg: dict, src: str, dst: str, dry_run: bool, log) -> None:
    """Fold person ``src`` into ``dst`` (appearances, aliases, centroid)."""
    people = ppl.load_people(cfg)
    for pid in (src, dst):
        if pid not in people:
            log.error("Unknown person id: %s", pid)
            raise typer.Exit(1)
    if src == dst:
        log.error("--merge and --into are the same id")
        raise typer.Exit(1)

    s, d = people[src], people[dst]
    log.info("Merging %s (%s, %d appearance(s)) into %s (%s)",
             src, ppl.display_name(src, s), len(s.get("appearances", [])),
             dst, ppl.display_name(dst, d))
    if dry_run:
        return

    d.setdefault("appearances", []).extend(s.get("appearances", []))
    aliases = d.setdefault("aliases", [])
    for name in [ppl.display_name(src, s) if ppl.is_named(src, s) else None, *s.get("aliases", [])]:
        if name and name not in aliases and name != d.get("display_name"):
            aliases.append(name)
    if not ppl.is_named(dst, d) and ppl.is_named(src, s):
        d["display_name"] = s["display_name"]
    for field in ("affiliation", "notes"):
        if not d.get(field) and s.get(field):
            d[field] = s[field]

    del people[src]
    ppl.drop_centroid(cfg, src)
    ppl.recompute_centroid(cfg, dst, d)
    ppl.save_people(cfg, people)
    for collection in sorted({a["collection"] for a in d["appearances"]}):
        ppl.write_collection_view(cfg, collection, people)
    log.info("Merged. Run 'python -m pipeline.rename_speakers' to update transcripts.")


@app.command()
def main(
    list_unknown: bool = typer.Option(False, "--list-unknown", help="Show unnamed people"),
    merge: str = typer.Option("", "--merge", help="Person id to fold into --into"),
    into: str = typer.Option("", "--into", help="Person id that absorbs --merge"),
    force: bool = typer.Option(False, "--force", help="Recluster even if inputs are unchanged"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be processed"),
    verbose: bool = typer.Option(False, "--verbose", help="Debug logging"),
) -> None:
    """Cluster speakers per collection and match them to the global people database."""
    log = setup_logging("speakers", verbose)
    cfg = load_config()

    if merge or into:
        if not (merge and into):
            log.error("--merge and --into must be given together")
            raise typer.Exit(1)
        _merge_people(cfg, merge, into, dry_run, log)
        return

    transcript_dir = resolve_path("paths.transcripts", cfg)
    norm_dir = resolve_path("paths.normalized", cfg)
    speaker_dir = resolve_path("paths.speaker_id", cfg)
    speaker_cfg = cfg.get("speakers", {})
    match_threshold = speaker_cfg.get("match_threshold", 0.35)

    if list_unknown:
        people = ppl.load_people(cfg)
        if not people:
            log.error("No people database yet. Run without --list-unknown first.")
            raise typer.Exit(1)
        unnamed = [(pid, p) for pid, p in sorted(people.items()) if not ppl.is_named(pid, p)]
        log.info("%d of %d people unnamed (edit %s)", len(unnamed), len(people), ppl.people_path(cfg))
        for pid, person in unnamed:
            apps = person.get("appearances", [])
            cols = sorted({a["collection"] for a in apps})
            log.info("  %s -- %d appearance(s) in %s", pid, len(apps), ", ".join(cols))
            for p in (person.get("proposed_names") or [])[:2]:
                log.info("    ? %s (score %d) -- accept: python -m pipeline.names --accept %s",
                         p["name"], p["score"], pid)
            for q in [q for a in apps for q in a.get("sample_quotes", [])][:3]:
                log.info("    > %s", q)
        return

    collections = discover_collections(transcript_dir)
    if not collections:
        log.warning("No collections found under %s", transcript_dir)
        raise typer.Exit(1)

    def _transcripts_in(collection: str) -> list[Path]:
        col_dir = Path(transcript_dir) / collection
        return sorted(
            p for p in col_dir.glob("*.json")
            if not p.name.endswith(".clean.json") and not p.name.endswith(".meta.json")
        )

    def _collection_stale(collection: str, transcript_paths: list[Path]) -> bool:
        """True if the collection needs (re)clustering."""
        meta_path = Path(speaker_dir) / collection / "registry.meta.json"
        if force or not meta_path.exists() or not ppl.people_path(cfg).exists():
            return True
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                prev = json.load(f).get("inputs_fingerprint")
        except (json.JSONDecodeError, OSError):
            return True
        return prev != _inputs_fingerprint(transcript_dir, collection, transcript_paths, cfg)

    # Determine which collections actually need (re)clustering (idempotency)
    todo: list[tuple[str, list[Path]]] = []
    for collection in collections:
        tps = _transcripts_in(collection)
        if not tps:
            continue
        if _collection_stale(collection, tps):
            todo.append((collection, tps))
        else:
            log.info("[%s] up to date -- skipping (%d transcript(s))", collection, len(tps))

    if dry_run:
        for collection, tps in todo:
            log.info("[%s] would cluster + match %d transcript(s)", collection, len(tps))
        if not todo:
            log.info("All collections up to date")
        return

    if not todo:
        log.info("All collections up to date -- nothing to cluster")
        return

    # Load ECAPA-TDNN model once; the model cache is collection-independent.
    log.info("Loading ECAPA-TDNN model...")
    from speechbrain.inference.speaker import EncoderClassifier
    from speechbrain.utils.fetching import LocalStrategy

    # local_strategy=COPY is a DIRECT kwarg of from_hparams (forwarded to fetch);
    # the old fetch_from={...} wrapper was an unrecognized kwarg and was silently
    # ignored, so fetch fell back to SYMLINK -> WinError 1314 on Windows without
    # Developer Mode. COPY copies the model files into savedir instead.
    classifier = EncoderClassifier.from_hparams(
        source=speaker_cfg.get("model", "speechbrain/spkrec-ecapa-voxceleb"),
        savedir=str(Path(speaker_dir) / "models" / "ecapa"),
        run_opts={"device": resolve_device(cfg)},
        local_strategy=LocalStrategy.COPY,
    )

    people = ppl.load_people(cfg)
    total_new = total_matched = 0
    # Cluster within each collection first (same room, same mic: reliable), then
    # match each cluster against the global database (cross-event: stricter).
    for collection, transcript_paths in todo:
        log.info("[%s] %d transcript(s)", collection, len(transcript_paths))
        norm_col = Path(norm_dir) / collection

        all_embeddings: dict[str, np.ndarray] = {}
        speaker_info: dict[str, dict] = {}  # file_speaker -> metadata

        for tp in transcript_paths:
            with open(tp, "r", encoding="utf-8") as f:
                transcript = json.load(f)

            stem = tp.stem
            wav_path = norm_col / f"{stem}.wav"
            if not wav_path.exists():
                log.warning("  %s: WAV not found, skipping", stem)
                continue

            for spk in transcript.get("speakers", []):
                spk_id = spk["id"]
                if spk_id == "UNKNOWN":
                    continue

                key = f"{stem}::{spk_id}"
                time_segs = _extract_speaker_audio_segments(wav_path, transcript, spk_id, cfg)

                log.info("  %s/%s: %d segments, computing embedding...", stem, spk_id, len(time_segs))
                emb = _compute_speaker_embedding(wav_path, time_segs, classifier, cfg)

                if emb is not None:
                    all_embeddings[key] = emb
                    np.save(str(ppl.appearance_embedding_path(cfg, collection, stem, spk_id)), emb)
                    speaker_info[key] = {
                        "collection": collection,
                        "file": stem,
                        "local_id": spk_id,
                        "speaking_time_sec": spk.get("speaking_time_sec", 0),
                        "sample_quotes": _get_sample_quotes(transcript, spk_id),
                    }
                else:
                    log.info("  %s/%s: insufficient speech, skipping", stem, spk_id)

        if not all_embeddings:
            log.warning("[%s] no speaker embeddings computed (need diarized transcripts)", collection)
            continue

        log.info("[%s] clustering %d speaker embeddings...", collection, len(all_embeddings))
        clusters = _cluster_embeddings(all_embeddings, cfg)

        members: dict[str, list[str]] = {}
        for key, cid in clusters.items():
            members.setdefault(cid, []).append(key)
        cluster_centroids = {
            cid: np.mean([all_embeddings[k] / (np.linalg.norm(all_embeddings[k]) or 1) for k in keys], axis=0)
            for cid, keys in members.items()
        }

        # Match against the database as it stood *before* this run, which still
        # includes this collection's previous appearances -- so re-running a
        # collection after adding a talk keeps everyone's existing person id.
        matches = ppl.match_clusters(cluster_centroids, ppl.load_centroids(cfg, people), match_threshold)

        # Replace this collection's appearances wholesale.
        touched: set[str] = set()
        for pid, person in people.items():
            before = len(person.get("appearances", []))
            person["appearances"] = [a for a in person.get("appearances", []) if a["collection"] != collection]
            if len(person["appearances"]) != before:
                touched.add(pid)

        for cid, keys in sorted(members.items()):
            if cid in matches:
                pid, dist = matches[cid]
                total_matched += 1
                log.info("  %s -> %s (%s), distance %.3f", cid, pid,
                         ppl.display_name(pid, people[pid]), dist)
            else:
                pid, dist = ppl.new_person_id(people), None
                people[pid] = ppl.new_person(pid)
                total_new += 1
                log.info("  %s -> new person %s", cid, pid)
            for key in keys:
                app_rec = dict(speaker_info[key])
                if dist is not None:
                    app_rec["match_distance"] = dist
                people[pid]["appearances"].append(app_rec)
            touched.add(pid)

        for pid in sorted(touched):
            person = people[pid]
            if person["appearances"]:
                ppl.recompute_centroid(cfg, pid, person)
            elif not ppl.is_named(pid, person):
                # An unnamed voice that no longer appears anywhere carries no value.
                del people[pid]
                ppl.drop_centroid(cfg, pid)

        # Drop appearance embeddings for files that left this collection.
        live = {f"{info['file']}_{info['local_id']}.npy" for info in speaker_info.values()}
        for stale in Path(speaker_dir, "embeddings", collection).glob("*.npy"):
            if stale.name not in live:
                stale.unlink()

        ppl.save_people(cfg, people)
        view = ppl.write_collection_view(cfg, collection, people)

        # Record the inputs fingerprint so re-runs skip unchanged collections
        meta_path = Path(speaker_dir) / collection / "registry.meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({
                "inputs_fingerprint": _inputs_fingerprint(transcript_dir, collection, transcript_paths, cfg),
                "generated_at": datetime.now().isoformat(),
                "clusters": len(members),
                "matched": len(matches),
            }, f, indent=2)

        log.info("[%s] %d voice(s): %d known, %d new -- view: %s",
                 collection, len(members), len(matches), len(members) - len(matches), view)

    # Free model
    from pipeline.utils import gpu_cleanup
    gpu_cleanup(classifier)

    log.info("Speaker identification complete -- %d matched to known people, %d new; "
             "database has %d people (%s)", total_matched, total_new, len(people), ppl.people_path(cfg))


if __name__ == "__main__":
    app()
