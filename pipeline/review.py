"""Phase 6: Human-in-the-loop review of entities/acronyms and speakers.

After the pipeline processes recordings, the knowledge base contains entities and
acronyms the LLM only *guessed* at, plus unnamed speaker clusters. This stage:

  build  -> flag every entity/acronym (with file + timestamp + speaker provenance)
            into the GLOBAL data/glossary.yaml (status: unverified), list unnamed
            speakers, and write a human-readable data/REVIEW.md guide + a marker.
  status -> summarize what is still pending.
  apply  -> propagate verified data: merge verified acronyms into the local,
            gitignored domain_corrections.local.yaml (spelling; event vocabulary
            never goes in the shared tracked file), then re-run the back half
            (correct -> rename_speakers -> kb extract -> index -> kb index-md),
            which is staleness-gated so only affected files recompute.

The glossary is GLOBAL with multiple "senses" per term, each scoped by
`collections`, so the same acronym can mean different things in different subjects.
Speakers live in the GLOBAL people database (data/speaker-id/people.yaml) and are
matched across collections, so naming someone once labels them in every talk.

Usage:
    python -m pipeline.review build
    python -m pipeline.review status
    python -m pipeline.review apply
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import typer
import yaml

from pipeline import people
from pipeline.utils import (
    discover_collections,
    format_timestamp,
    load_config,
    preferred_transcript,
    project_root,
    resolve_path,
    setup_logging,
    write_yaml_atomic,
)

app = typer.Typer(add_completion=False)

_HEARD_CAP = 20  # max provenance entries kept per sense


# --- paths ---

def _data_root(cfg: dict) -> Path:
    """The data/ root (parent of the per-stage dirs)."""
    return Path(resolve_path("paths.kb", cfg)).parent


def _glossary_path(cfg: dict) -> Path:
    return _data_root(cfg) / "glossary.yaml"


def _review_md_path(cfg: dict) -> Path:
    return _data_root(cfg) / "REVIEW.md"


def _marker_path(cfg: dict) -> Path:
    return _data_root(cfg) / ".review_pending"


def _corrections_path(cfg: dict) -> Path:
    """Where verified spellings are written: the local (gitignored) overlay, never the
    shared tracked file, so event vocabulary stays out of the repo."""
    rel = cfg.get("correction", {}).get("local_corrections_file",
                                         "pipeline/prompts/domain_corrections.local.yaml")
    return project_root() / rel


# --- provenance ---

def find_provenance(entity: str, transcript: dict, max_hits: int = 5) -> list[dict]:
    """Locate where an entity is said: returns [{at, by, context}, ...].

    Matching, in order: exact case-insensitive substring; alphanumeric-normalized
    substring (handles "A.S.R." / spacing); all significant words present (multi-
    word). Empty list = no transcript source (likely an LLM hallucination).
    """
    el = (entity or "").strip().lower()
    if not el:
        return []
    norm = re.sub(r"[^a-z0-9]+", "", el)
    words = [w for w in re.findall(r"[a-z0-9]+", el) if len(w) > 2]
    hits: list[dict] = []
    for seg in transcript.get("segments", []):
        text = seg.get("text", "")
        tl = text.lower()
        matched = (
            el in tl
            or (len(norm) >= 3 and norm in re.sub(r"[^a-z0-9]+", "", tl))
            or (len(words) >= 2 and all(w in tl for w in words))
        )
        if matched:
            hits.append({
                "at": format_timestamp(seg.get("start", 0.0)),
                "by": seg.get("speaker_display") or seg.get("speaker") or "UNKNOWN",
                "context": text.strip()[:160],
            })
            if len(hits) >= max_hits:
                break
    return hits


# --- glossary load/save/merge ---

def _load_glossary(cfg: dict) -> dict:
    gpath = _glossary_path(cfg)
    if not gpath.exists():
        return {}
    try:
        return yaml.safe_load(gpath.read_text(encoding="utf-8")) or {}
    except (yaml.YAMLError, OSError):
        return {}


def _merge_heard(sense: dict, heard: list[dict]) -> None:
    """Append-dedupe provenance by (file, at); cap length."""
    existing = sense.setdefault("heard", [])
    seen = {(h.get("file"), h.get("at")) for h in existing}
    for h in heard:
        key = (h.get("file"), h.get("at"))
        if key not in seen:
            existing.append(h)
            seen.add(key)
    if len(existing) > _HEARD_CAP:
        del existing[_HEARD_CAP:]


def _merge_term(glossary: dict, collection: str, term: str, auto_def: str,
                heard: list[dict]) -> dict:
    """Merge one (collection, term) observation into the global glossary.

    Never overwrites human-owned status/definition. Returns a small report dict
    describing what happened for REVIEW.md.
    """
    senses = glossary.setdefault(term, [])
    applicable = [s for s in senses
                  if not s.get("collections") or collection in s.get("collections", [])]

    if applicable:
        s = applicable[0]
        _merge_heard(s, heard)
        status = s.get("status", "unverified")
        # Only fill an auto definition when still unverified, auto-sourced, empty.
        if (status == "unverified" and s.get("definition_source", "auto") == "auto"
                and not (s.get("definition") or "").strip() and auto_def):
            s["definition"] = auto_def
        return {"term": term, "status": status, "sense": s, "new": False}

    # No applicable sense: new sense for this collection.
    other_verified_cols = sorted({
        c for s in senses if s.get("status") == "verified" for c in (s.get("collections") or [])
    })
    new_sense = {
        "definition": auto_def or "",
        "context": "",
        "collections": [collection],
        "status": "unverified",
        "definition_source": "auto",
        "heard": [],
    }
    _merge_heard(new_sense, heard)
    if other_verified_cols:
        new_sense["possible_new_meaning_of"] = other_verified_cols
    senses.append(new_sense)
    return {"term": term, "status": "unverified", "sense": new_sense, "new": True,
            "possible_new_meaning_of": other_verified_cols}


# --- build ---

def _collection_observations(cfg: dict, collection: str, log) -> dict[str, dict]:
    """For one collection: {entity -> {auto_def, heard:[{file,at,by,context}]}}."""
    kb_dir = Path(resolve_path("paths.kb", cfg)) / collection
    tr_dir = Path(resolve_path("paths.transcripts", cfg)) / collection
    obs: dict[str, dict] = {}
    for kb_json in sorted(kb_dir.glob("*.kb.json")):
        stem = kb_json.name[: -len(".kb.json")]
        try:
            data = json.loads(kb_json.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        entities = data.get("entities", []) or []
        kb_gloss = {k.lower(): v for k, v in (data.get("glossary", {}) or {}).items()}
        tp = preferred_transcript(tr_dir, stem)
        transcript = {}
        if tp.exists():
            try:
                transcript = json.loads(tp.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                transcript = {}
        for ent in entities:
            rec = obs.setdefault(ent, {"auto_def": "", "heard": []})
            if not rec["auto_def"]:
                rec["auto_def"] = kb_gloss.get(ent.lower(), "")
            for h in find_provenance(ent, transcript):
                rec["heard"].append({"file": stem, **h})
    return obs


def _unnamed_speakers(cfg: dict, collection: str) -> list[dict]:
    """Unnamed people (global database) who appear in a collection."""
    return people.unnamed_in_collection(cfg, collection)


def _write_review_md(cfg: dict, report: dict) -> Path:
    """Render the human-readable REVIEW.md guide from the build report."""
    g = _glossary_path(cfg)
    lines = [
        "# Review queue",
        "",
        "Edit the two source-of-truth files below, then run "
        "`python -m pipeline.review apply` (or press Enter in the paused pipeline / "
        "`.\\pipeline\\run.ps1 -Resume`).",
        "",
        f"1. **Glossary** — `{g}`: for each term set `status: verified` and fix the "
        "`definition` (or `status: ignore` to drop it). Same acronym with a different "
        "meaning in another subject? It's listed under *possible new meaning* — give it "
        "its own sense.",
        "2. **Speakers** — `data/speaker-id/people.yaml`: set `display_name` (and "
        "optionally `affiliation`) for each unnamed person below, or accept a proposed name "
        "with `python -m pipeline.names --accept <id>`. Same person under two "
        "ids? `python -m pipeline.speakers --merge <id> --into <id>`.",
        "",
        "_This file is generated; your edits go in the files above, not here._",
        "",
    ]
    for collection in sorted(report):
        rep = report[collection]
        lines.append(f"## {collection}")
        lines.append("")
        terms = rep["terms"]
        found = [t for t in terms if t["heard"] and not t.get("possible_new_meaning_of")]
        newmeaning = [t for t in terms if t.get("possible_new_meaning_of")]
        notfound = [t for t in terms if not t["heard"]]

        lines.append(f"### Acronyms / entities to verify ({len(terms)})")
        lines.append("")
        for t in found:
            lines.append(f"- **{t['term']}** — guess: _{t['auto_def'] or '(none)'}_")
            for h in t["heard"][:3]:
                lines.append(f"    - [{h['at']}] {h['by']} ({h['file']}): \"{h['context']}\"")
        lines.append("")
        if newmeaning:
            lines.append("#### Possible new meanings (already verified in other subjects)")
            for t in newmeaning:
                lines.append(f"- **{t['term']}** — also used in: "
                             f"{', '.join(t['possible_new_meaning_of'])}. Guess here: "
                             f"_{t['auto_def'] or '(none)'}_")
                for h in t["heard"][:2]:
                    lines.append(f"    - [{h['at']}] {h['by']} ({h['file']}): \"{h['context']}\"")
            lines.append("")
        if notfound:
            lines.append("#### No transcript source found (verify spelling / likely hallucination)")
            for t in notfound:
                lines.append(f"- **{t['term']}** — guess: _{t['auto_def'] or '(none)'}_")
            lines.append("")

        speakers = rep["speakers"]
        lines.append(f"### Unnamed speakers ({len(speakers)})")
        lines.append("")
        if speakers:
            lines.append(f"Listen first: one clip per voice in `data/speaker-id/{collection}/clips/` "
                         "(see its `INDEX.md`; build with `python -m pipeline.clips`).")
            lines.append("")
        for s in speakers:
            also = f" (also heard in: {', '.join(s['also_in'])})" if s.get("also_in") else ""
            lines.append(f"- **{s['cluster']}** — {s['speaking_time']} across "
                         f"{', '.join(s['files'])}{also}")
            for p in s.get("proposals", []):
                aff = f" ({p['affiliation']})" if p.get("affiliation") else ""
                ev = p["evidence"][0] if p.get("evidence") else {}
                lines.append(f"    - proposed: **{p['name']}**{aff}, score {p['score']} "
                             f"({', '.join(p['basis'])}): \"{ev.get('quote', '')}\" [{ev.get('source', '')}]")
            if s.get("proposals"):
                lines.append(f"    - accept: `python -m pipeline.names --accept {s['cluster']}`")
            for q in s["samples"]:
                lines.append(f"    - \"{q}\"")
        lines.append("")

    path = _review_md_path(cfg)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


@app.command()
def build(
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be flagged; write nothing"),
    verbose: bool = typer.Option(False, "--verbose", help="Debug logging"),
) -> None:
    """Flag entities/acronyms + unnamed speakers for human verification."""
    log = setup_logging("review", verbose)
    cfg = load_config()

    kb_root = resolve_path("paths.kb", cfg)
    collections = discover_collections(kb_root)
    if not collections:
        log.warning("No KB collections found -- run the pipeline first.")
        raise typer.Exit(1)

    glossary = _load_glossary(cfg)
    report: dict[str, dict] = {}
    total_new = 0

    for collection in collections:
        obs = _collection_observations(cfg, collection, log)
        term_reports = []
        for term in sorted(obs):
            rec = obs[term]
            r = _merge_term(glossary, collection, term, rec["auto_def"], rec["heard"])
            if r["new"]:
                total_new += 1
            term_reports.append({
                "term": term,
                "auto_def": rec["auto_def"],
                "heard": rec["heard"],
                "possible_new_meaning_of": r.get("possible_new_meaning_of") or [],
            })
        speakers = _unnamed_speakers(cfg, collection)
        report[collection] = {"terms": term_reports, "speakers": speakers}
        log.info("[%s] %d entity term(s), %d unnamed speaker(s)",
                 collection, len(term_reports), len(speakers))

    if dry_run:
        log.info("Dry run -- would flag %d new term(s) across %d collection(s); nothing written.",
                 total_new, len(collections))
        return

    write_yaml_atomic(_glossary_path(cfg), glossary)
    md = _write_review_md(cfg, report)
    _marker_path(cfg).write_text("pending\n", encoding="utf-8")

    unverified = sum(1 for senses in glossary.values() for s in senses
                     if s.get("status") == "unverified")
    unnamed = sum(len(r["speakers"]) for r in report.values())
    log.info("Review queue ready:")
    log.info("  glossary: %s (%d unverified term-sense(s))", _glossary_path(cfg), unverified)
    log.info("  guide:    %s", md)
    log.info("  speakers: %d unnamed cluster(s) -- edit each registry.yaml", unnamed)


@app.command()
def status(
    verbose: bool = typer.Option(False, "--verbose", help="Debug logging"),
) -> None:
    """Show pending review items."""
    log = setup_logging("review", verbose)
    cfg = load_config()
    glossary = _load_glossary(cfg)
    unverified = [(t, s) for t, senses in glossary.items() for s in senses
                  if s.get("status") == "unverified"]
    verified = sum(1 for senses in glossary.values() for s in senses
                   if s.get("status") == "verified")
    log.info("Glossary: %d verified, %d unverified term-sense(s)", verified, len(unverified))
    for t, s in unverified[:50]:
        log.info("  [ ] %s (%s)", t, ", ".join(s.get("collections", []) or ["global"]))

    kb_root = resolve_path("paths.kb", cfg)
    for collection in discover_collections(kb_root):
        unnamed = _unnamed_speakers(cfg, collection)
        if unnamed:
            log.info("[%s] %d unnamed speaker(s): %s", collection, len(unnamed),
                     ", ".join(s["cluster"] for s in unnamed))
    log.info("Pending marker: %s", "present" if _marker_path(cfg).exists() else "none")


def _merge_domain_corrections(cfg: dict, terms: list[str], log, dry_run: bool) -> int:
    """Idempotently add `lowercased -> Canonical` spelling fixes for verified terms."""
    path = _corrections_path(cfg)
    # This file is hand-maintained: only a missing file counts as empty. An unreadable
    # one stops the apply rather than being overwritten (and its rules lost).
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except FileNotFoundError:
        data = {}
    except (yaml.YAMLError, OSError) as e:
        log.error("Can't read %s (%s) -- fix it, then re-run apply. Nothing was written.", path, e)
        raise typer.Exit(1)
    if not isinstance(data, dict) or not isinstance(data.get("corrections") or {}, dict):
        log.error("%s must be a mapping with a 'corrections:' mapping -- fix it, then re-run apply.", path)
        raise typer.Exit(1)
    corr = data.get("corrections") or {}
    added = 0
    for term in terms:
        key = term.lower()
        if key != term and corr.get(key) != term:  # only when it actually normalizes
            corr[key] = term
            added += 1
    if added and not dry_run:
        data["corrections"] = corr
        write_yaml_atomic(path, data)
    log.info("domain_corrections: %d spelling rule(s) %s", added,
             "would be added" if dry_run else "added")
    return added


def _run_stage(module_args: list[str], log) -> None:
    """Run a pipeline stage as a subprocess (reuses each stage's idempotency)."""
    log.info("  $ python -m %s", " ".join(module_args))
    r = subprocess.run([sys.executable, "-m", *module_args], cwd=str(project_root()))
    if r.returncode != 0:
        raise RuntimeError(f"stage failed: {' '.join(module_args)} (exit {r.returncode})")


@app.command()
def apply(
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would happen; change nothing"),
    verbose: bool = typer.Option(False, "--verbose", help="Debug logging"),
) -> None:
    """Propagate verified glossary + speaker names into all artifacts."""
    log = setup_logging("review", verbose)
    cfg = load_config()

    glossary = _load_glossary(cfg)
    verified_terms = sorted({t for t, senses in glossary.items()
                             for s in senses if s.get("status") == "verified"})
    log.info("%d verified glossary term(s)", len(verified_terms))

    # Feedback loop 1: verified acronyms -> spelling corrections.
    _merge_domain_corrections(cfg, verified_terms, log, dry_run)

    if dry_run:
        log.info("Dry run -- would re-run: correct -> rename_speakers -> kb extract "
                 "-> index -> kb index-md")
        return

    # Re-run the back half. ORDER IS LOAD-BEARING: `correct` regenerates
    # .clean.json from the raw .json (dropping speaker_display), so rename_speakers
    # must run AFTER it to re-apply names, and kb reads the final .clean.json.
    # Each stage is staleness-gated, so only files affected by verified changes
    # (corrections hash / glossary hash / renamed transcripts) actually recompute.
    verbose_args = ["--verbose"] if verbose else []
    _run_stage(["pipeline.correct", *verbose_args], log)
    _run_stage(["pipeline.rename_speakers", *verbose_args], log)
    _run_stage(["pipeline.kb", "extract", *verbose_args], log)
    _run_stage(["pipeline.index", *verbose_args], log)
    _run_stage(["pipeline.kb", "index-md", *verbose_args], log)

    _marker_path(cfg).unlink(missing_ok=True)
    log.info("Review applied -- verified knowledge propagated into transcripts + KB.")


if __name__ == "__main__":
    app()
