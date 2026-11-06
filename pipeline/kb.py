"""Phase 4: Knowledge base extraction via instructor + Ollama.

Usage:
    python -m pipeline.kb                                    # extract all
    python -m pipeline.kb "keynote.m4a"                      # single file
    python -m pipeline.kb --validate                         # verify KB against transcripts
    python -m pipeline.kb index-md                           # generate INDEX.md
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import jinja2
import typer
import yaml
from pydantic import BaseModel, Field

from pipeline.utils import (
    collection_root,
    collection_transcripts,
    discover_collections,
    ensure_dir,
    format_timestamp,
    is_stale,
    load_config,
    project_root,
    read_meta,
    resolve_path,
    setup_logging,
    write_meta,
)

app = typer.Typer(add_completion=False)

_MODEL_KEYS = ["models.llm"]


def _glossary_path(cfg: dict) -> Path:
    """Path to the global glossary (data/glossary.yaml), beside the data dirs."""
    return Path(resolve_path("paths.kb", cfg)).parent / "glossary.yaml"


def _applicable_senses(cfg: dict, collection: str) -> dict[str, str]:
    """``term -> definition`` for VERIFIED glossary senses applicable to ``collection``.

    Applicable = the sense's ``collections`` list includes this collection, or is
    empty/missing (a global sense). Only ``term`` + ``definition`` are returned
    (and hashed) — provenance/context are intentionally excluded so re-running
    ``review build`` (which appends provenance) does not force re-extraction.
    """
    gpath = _glossary_path(cfg)
    if not gpath.exists():
        return {}
    try:
        data = yaml.safe_load(gpath.read_text(encoding="utf-8")) or {}
    except (yaml.YAMLError, OSError):
        return {}
    out: dict[str, str] = {}
    for term, senses in data.items():
        if not isinstance(senses, list):
            continue
        for sense in senses:
            if not isinstance(sense, dict) or sense.get("status") != "verified":
                continue
            cols = sense.get("collections") or []
            if cols and collection not in cols:
                continue
            defn = (sense.get("definition") or "").strip()
            if defn:
                out.setdefault(term, defn)  # first applicable sense wins
    return out


def _glossary_senses_hash(senses: dict[str, str]) -> str:
    """Stable hash of applicable verified senses, for KB re-extraction staleness."""
    return hashlib.sha256(json.dumps(senses, sort_keys=True).encode()).hexdigest()[:16]


# --- Pydantic schemas ---

class ActionItem(BaseModel):
    task: str
    owner: str
    due: str = "unspecified"
    source_timestamp: str


class Quote(BaseModel):
    text: str
    speaker: str
    timestamp: str


class KBEntry(BaseModel):
    summary: str = Field(min_length=50, max_length=2000)
    participants: list[str]
    decisions: list[str]
    action_items: list[ActionItem]
    open_questions: list[str]
    topics: list[str]
    entities: list[str]
    notable_quotes: list[Quote] = Field(default_factory=list)
    glossary: dict[str, str] = Field(default_factory=dict)


# --- Extraction ---

def _transcript_to_text(transcript: dict) -> str:
    """Convert transcript JSON to plain text with timestamps for LLM input."""
    lines = []
    for seg in transcript.get("segments", []):
        ts = format_timestamp(seg["start"])
        speaker = seg.get("speaker_display", seg.get("speaker", "UNKNOWN"))
        lines.append(f"[{ts}] {speaker}: {seg['text']}")
    return "\n".join(lines)


def _chunk_transcript(transcript: dict, max_tokens: int = 3000) -> list[str]:
    """Chunk transcript into segments respecting speaker turns.

    Rough token estimate: 1 token ~ 4 chars.
    """
    chars_per_chunk = max_tokens * 4
    chunks = []
    current_lines = []
    current_len = 0

    for seg in transcript.get("segments", []):
        ts = format_timestamp(seg["start"])
        speaker = seg.get("speaker_display", seg.get("speaker", "UNKNOWN"))
        line = f"[{ts}] {speaker}: {seg['text']}"
        line_len = len(line)

        if current_len + line_len > chars_per_chunk and current_lines:
            chunks.append("\n".join(current_lines))
            # Keep last line as overlap
            current_lines = [current_lines[-1]] if current_lines else []
            current_len = len(current_lines[0]) if current_lines else 0

        current_lines.append(line)
        current_len += line_len

    if current_lines:
        chunks.append("\n".join(current_lines))

    return chunks


def _create_client(cfg: dict):
    """Return the native Ollama client module.

    KB uses the native /api/chat path (like correct.py), NOT the OpenAI /v1
    endpoint. On this machine /v1 wedges on a cloud-auth check (POST /api/me)
    and chat requests hang before reaching the model, while /api/chat works.
    Structured output is enforced below with format="json" + pydantic
    validation + a self-correcting retry loop (replacing instructor's role).
    """
    import ollama
    return ollama


def _extract_kb(
    transcript: dict,
    cfg: dict,
    log,
    glossary: dict[str, str] | None = None,
) -> KBEntry:
    """Extract KB entry from transcript using map-reduce if needed.

    ``glossary`` (verified term -> definition, applicable to this collection) is
    injected into the extract prompt as ground-truth so the model uses verified
    definitions/spellings instead of guessing.
    """
    client = _create_client(cfg)
    model = cfg["models"]["llm"]
    ollama_cfg = cfg.get("ollama", {})

    # KB extraction needs more context/output headroom than correction: the
    # reduce step merges many partial extractions into one large JSON. The
    # shared ollama.num_ctx is kept small so correction's 8B fits on the GPU,
    # so request a larger context just for KB here (via Ollama options), plus a
    # generous output cap. Without this, the largest file's reduce truncates
    # ("output incomplete due to max_tokens length limit").
    kb_num_ctx = max(int(ollama_cfg.get("num_ctx", 8192)), 16384)
    kb_max_out = 6144

    def _create(messages) -> KBEntry:
        """Call the model and validate into KBEntry, retrying with feedback."""
        msgs = list(messages)
        last_err = None
        for _ in range(3):
            resp = client.chat(
                model=model,
                messages=msgs,
                format="json",
                options={
                    "temperature": 0,
                    "num_ctx": kb_num_ctx,
                    "num_predict": kb_max_out,
                },
            )
            content = resp["message"]["content"]
            try:
                return KBEntry.model_validate_json(content)
            except Exception as e:  # JSON or schema validation error -> retry
                last_err = e
                msgs = list(messages) + [
                    {"role": "assistant", "content": content},
                    {"role": "user", "content":
                        f"Your previous response failed validation:\n{e}\n"
                        "Return ONLY a corrected JSON object matching the required schema."},
                ]
        raise RuntimeError(f"KB extraction failed validation after 3 attempts: {last_err}")

    template_dir = project_root() / "pipeline" / "prompts"
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(template_dir)),
        keep_trailing_newline=True,
    )
    template = env.get_template("kb_extract.j2")

    text = _transcript_to_text(transcript)
    # Rough token estimate
    est_tokens = len(text) / 4

    glossary = glossary or {}

    if est_tokens < 6000:
        # Single pass
        log.info("    Single-pass extraction (~%d tokens)", int(est_tokens))
        prompt = template.render(transcript_text=text, mode="extract", glossary=glossary)
        return _create([{"role": "user", "content": prompt}])
    else:
        # Map-reduce
        chunks = _chunk_transcript(transcript)
        log.info("    Map-reduce: %d chunks (~%d tokens total)", len(chunks), int(est_tokens))

        partial_entries = []
        for i, chunk in enumerate(chunks):
            log.info("    Map chunk %d/%d...", i + 1, len(chunks))
            prompt = template.render(transcript_text=chunk, mode="extract", glossary=glossary)
            entry = _create([{"role": "user", "content": prompt}])
            partial_entries.append(entry.model_dump())

        # Reduce
        log.info("    Reduce pass...")
        reduce_prompt = template.render(
            chunks_json=json.dumps(partial_entries, indent=2),
            mode="reduce",
        )
        return _create([{"role": "user", "content": reduce_prompt}])


def _render_kb_markdown(kb: KBEntry, transcript: dict, cfg: dict) -> str:
    """Render KB entry to markdown via Jinja2 template."""
    template_dir = project_root() / "pipeline" / "templates"
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(template_dir)),
        keep_trailing_newline=True,
    )
    template = env.get_template("kb.md.j2")

    return template.render(
        source=transcript["source"],
        duration=format_timestamp(transcript["duration_sec"]),
        participants=kb.participants,
        topics=kb.topics,
        generated_at=datetime.now().isoformat(),
        model=cfg["models"]["llm"],
        title=Path(transcript["source"]).stem,
        summary=kb.summary,
        decisions=kb.decisions,
        action_items=[ai.model_dump() for ai in kb.action_items],
        open_questions=kb.open_questions,
        entities=kb.entities,
        notable_quotes=[q.model_dump() for q in kb.notable_quotes],
        glossary=kb.glossary,
    )


# --- Validation ---

def _validate_kb(
    kb_path: Path,
    transcript_path: Path,
    log,
) -> list[str]:
    """Validate KB entry against transcript. Returns list of warnings."""
    with open(transcript_path, "r", encoding="utf-8") as f:
        transcript = json.load(f)

    # Parse KB markdown to extract structured data
    with open(kb_path, "r", encoding="utf-8") as f:
        kb_content = f.read()

    warnings_list = []
    transcript_text = " ".join(
        seg["text"].lower() for seg in transcript.get("segments", [])
    )

    # Check entities exist in transcript
    lines = kb_content.split("\n")
    in_entities = False
    for line in lines:
        if line.strip() == "## Entities":
            in_entities = True
            continue
        if line.startswith("## ") and in_entities:
            break
        if in_entities and line.startswith("- "):
            entity = line[2:].strip()
            if entity and entity.lower() not in transcript_text:
                warnings_list.append(f"Entity '{entity}' not found in transcript")

    return warnings_list


# --- Commands ---

@app.command("extract")
def extract(
    files: list[str] = typer.Argument(None, help="Specific source files"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be processed"),
    verbose: bool = typer.Option(False, "--verbose", help="Debug logging"),
) -> None:
    """Extract knowledge base entries from transcripts."""
    log = setup_logging("kb", verbose)
    cfg = load_config()

    transcript_dir = resolve_path("paths.transcripts", cfg)

    wanted_stems = {Path(n).stem for n in files} if files else None

    # Build (collection, transcript_path) work list across all collections,
    # preferring <stem>.clean.json over <stem>.json.
    work: list[tuple[str, Path]] = []
    for collection in discover_collections(transcript_dir):
        col_dir = Path(transcript_dir) / collection
        for stem, tp in collection_transcripts(col_dir).items():
            if wanted_stems is not None and stem not in wanted_stems:
                continue
            work.append((collection, tp))

    if files and not work:
        log.warning("No transcripts found matching: %s", ", ".join(files))
    if not work:
        log.warning("No transcripts found")
        raise typer.Exit(1)

    log.info("Found %d transcript(s) for KB extraction across %d collection(s)",
             len(work), len({c for c, _ in work}))

    failures = 0
    for collection, tp in work:
        kb_dir = collection_root(cfg, "paths.kb", collection)
        stem = tp.stem.replace(".clean", "")
        out_md = Path(kb_dir) / f"{stem}.md"
        out_json = Path(kb_dir) / f"{stem}.kb.json"
        meta_path = Path(kb_dir) / f"{stem}.meta.json"

        # Verified glossary senses applicable to this collection (feedback loop 2).
        gloss = _applicable_senses(cfg, collection)
        gloss_hash = _glossary_senses_hash(gloss)

        # Re-extract when the source/model is stale OR the applicable verified
        # glossary changed (so verifying a term refreshes this collection's KB).
        if not is_stale(tp, out_md, meta_path, cfg, _MODEL_KEYS):
            if (read_meta(meta_path) or {}).get("glossary_senses_hash") == gloss_hash:
                log.info("  %s: skipped (up to date)", stem)
                continue
            log.info("  %s: verified glossary changed -- re-extracting", stem)

        if dry_run:
            log.info("  %s: would extract", stem)
            continue

        try:
            with open(tp, "r", encoding="utf-8") as f:
                transcript = json.load(f)

            log.info("  %s: extracting...", stem)
            kb = _extract_kb(transcript, cfg, log, glossary=gloss)

            # Write KB JSON
            ensure_dir(Path(kb_dir))
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(kb.model_dump(), f, indent=2, ensure_ascii=False)

            # Write KB markdown
            md_content = _render_kb_markdown(kb, transcript, cfg)
            with open(out_md, "w", encoding="utf-8") as f:
                f.write(md_content)

            write_meta(
                meta_path, tp, cfg, _MODEL_KEYS,
                summary_length=len(kb.summary),
                n_action_items=len(kb.action_items),
                n_topics=len(kb.topics),
                glossary_senses_hash=gloss_hash,
            )

            log.info("  %s: done -- %d topics, %d action items", stem, len(kb.topics), len(kb.action_items))

        except Exception as e:
            log.error("  %s: FAILED -- %s", stem, e, exc_info=verbose)
            failures += 1

    if failures:
        log.error("%d file(s) failed", failures)
        raise typer.Exit(1)

    log.info("KB extraction complete")


@app.command("validate")
def validate(
    verbose: bool = typer.Option(False, "--verbose", help="Debug logging"),
) -> None:
    """Validate KB entries against their source transcripts."""
    log = setup_logging("kb_validate", verbose)
    cfg = load_config()

    transcript_dir = resolve_path("paths.transcripts", cfg)
    kb_dir = resolve_path("paths.kb", cfg)

    collections = discover_collections(kb_dir)
    if not collections:
        log.warning("No KB collections found")
        raise typer.Exit(1)

    total_warnings = 0
    for collection in collections:
        col_kb = Path(kb_dir) / collection
        col_tr = Path(transcript_dir) / collection
        kb_files = sorted(col_kb.glob("*.md"))
        for kb_path in kb_files:
            if kb_path.name in ("INDEX.md", "qa_log.md"):
                continue

            stem = kb_path.stem
            # Find matching transcript within the same collection
            clean = col_tr / f"{stem}.clean.json"
            raw = col_tr / f"{stem}.json"
            tp = clean if clean.exists() else raw

            if not tp.exists():
                log.warning("  %s/%s: transcript not found", collection, stem)
                continue

            warnings = _validate_kb(kb_path, tp, log)
            if warnings:
                log.warning("  %s/%s: %d warning(s)", collection, stem, len(warnings))
                for w in warnings:
                    log.warning("    - %s", w)
                total_warnings += len(warnings)
            else:
                log.info("  %s/%s: OK", collection, stem)

    if total_warnings:
        log.warning("%d total warning(s)", total_warnings)
    else:
        log.info("All KB entries validated successfully")


@app.command("index-md")
def index_md(
    verbose: bool = typer.Option(False, "--verbose", help="Debug logging"),
) -> None:
    """Generate INDEX.md from all KB entries."""
    log = setup_logging("kb_index", verbose)
    cfg = load_config()

    kb_dir = resolve_path("paths.kb", cfg)

    collections = discover_collections(kb_dir)
    if not collections:
        log.warning("No KB collections found")
        raise typer.Exit(1)

    total = 0
    for collection in collections:
        col_kb = Path(kb_dir) / collection
        entries = []
        for kb_path in sorted(col_kb.glob("*.md")):
            if kb_path.name in ("INDEX.md", "qa_log.md"):
                continue
            json_path = kb_path.with_suffix(".kb.json")
            if json_path.exists():
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                entries.append({
                    "file": kb_path.name,
                    "stem": kb_path.stem,
                    **data,
                })
        if not entries:
            log.info("[%s] no KB entries -- skipping INDEX.md", collection)
            continue

        index_path = _write_index_md(col_kb, collection, entries)
        log.info("[%s] INDEX.md written: %s (%d entries)", collection, index_path, len(entries))
        total += len(entries)

    log.info("INDEX.md generation complete -- %d entr(ies) across %d collection(s)",
             total, len(collections))


def _write_index_md(kb_dir: Path, collection: str, entries: list[dict]) -> Path:
    """Build and write INDEX.md for one collection. Returns the index path."""
    # Build INDEX.md
    lines = [f"# Knowledge Base Index -- {collection}", "",
             f"_Generated: {datetime.now().isoformat()}_", ""]

    # Recordings table
    lines.append("## Recordings")
    lines.append("")
    lines.append("| Recording | Participants | Topics | Action Items |")
    lines.append("|---|---|---|---|")
    for e in entries:
        participants = ", ".join(e.get("participants", []))
        topics = ", ".join(e.get("topics", [])[:3])
        n_actions = len(e.get("action_items", []))
        lines.append(f"| [{e['stem']}]({e['file']}) | {participants} | {topics} | {n_actions} |")
    lines.append("")

    # Topics index
    lines.append("## Topics")
    lines.append("")
    topic_map: dict[str, list[str]] = {}
    for e in entries:
        for t in e.get("topics", []):
            topic_map.setdefault(t, []).append(e["stem"])
    for topic in sorted(topic_map.keys()):
        files = ", ".join(f"[{f}]({f}.md)" for f in topic_map[topic])
        lines.append(f"- **{topic}**: {files}")
    lines.append("")

    # Entities index
    lines.append("## Entities")
    lines.append("")
    entity_map: dict[str, list[str]] = {}
    for e in entries:
        for ent in e.get("entities", []):
            entity_map.setdefault(ent, []).append(e["stem"])
    for entity in sorted(entity_map.keys()):
        files = ", ".join(f"[{f}]({f}.md)" for f in entity_map[entity])
        lines.append(f"- **{entity}**: {files}")
    lines.append("")

    # All action items
    lines.append("## All Action Items")
    lines.append("")
    all_actions = []
    for e in entries:
        for ai in e.get("action_items", []):
            all_actions.append({**ai, "source": e["stem"]})

    # Sort: dated first (by date), then unspecified
    dated = [a for a in all_actions if a.get("due", "unspecified") != "unspecified"]
    undated = [a for a in all_actions if a.get("due", "unspecified") == "unspecified"]
    dated.sort(key=lambda a: a.get("due", ""))

    for a in dated + undated:
        lines.append(
            f"- [ ] {a['task']} -- owner: {a['owner']}, due: {a['due']}, "
            f"source: [{a['source']}]({a['source']}.md) [{a.get('source_timestamp', '')}]"
        )
    lines.append("")

    index_path = Path(kb_dir) / "INDEX.md"
    with open(index_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return index_path


if __name__ == "__main__":
    app()
