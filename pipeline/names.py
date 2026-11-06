"""Phase 3.5: Propose real names for speakers from what is said in the talk.

Conference audio usually says who is speaking: hosts introduce speakers ("please
welcome Jane Doe from Example Corp"), speakers introduce themselves, and Q&A
addresses people by name. This stage sends only those passages (the opening and
closing minutes plus lines with introduction cues) to the LLM, which attributes
names to diarization labels. Claims are then rolled up per *person* in the global
database and written to people.yaml as ``proposed_names`` -- a human still
confirms them (``--accept``), because a wrong name pollutes every talk.

Extraction is optional enrichment: if Ollama is down or a file fails validation,
it is skipped with a warning (no cache written, retried next run) and the
pipeline continues.

    data/speaker-id/<collection>/names/<stem>.json   <- per-talk claims cache

Usage:
    python -m pipeline.names                          # extract (stale talks) + roll up
    python -m pipeline.names --accept spk_0004        # take spk_0004's top proposal
    python -m pipeline.names --accept spk_0004 --pick 2
    python -m pipeline.names --accept-all             # unambiguous, well-supported proposals only
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Literal

import jinja2
import typer
from pydantic import BaseModel

from pipeline import people as ppl
from pipeline.utils import (
    collection_transcripts,
    discover_collections,
    format_timestamp,
    load_config,
    project_root,
    resolve_path,
    setup_logging,
)

app = typer.Typer(add_completion=False)

# Evidence weights: a self-introduction is near-certain; being addressed by name
# in Q&A is often a first name only and easy to misattribute.
_WEIGHTS = {"self_introduction": 3, "introduced_by_other": 2, "addressed_by_name": 1}

_CUES = re.compile(
    r"\b(my name(?:'s| is)|please welcome|welcome (?:to the stage|back|up)|introduc\w*|"
    r"joining us|our (?:next|first|last|final) (?:speaker|panelist|guest)|next (?:up|speaker)|"
    r"give it up for|round of applause|thank you,|thanks,|great question,|go ahead,)",
    re.IGNORECASE,
)
# "I'm"/"I am"/"this is" are too common on their own; require a capitalized word after.
_SELF = re.compile(r"\b(?:I'm|I am|this is|it's)\s+[A-Z][a-z]+")


class NameClaim(BaseModel):
    speaker: str
    name: str
    affiliation: str = ""
    basis: Literal["self_introduction", "introduced_by_other", "addressed_by_name"]
    timestamp: str = ""
    evidence: str = ""


class NameClaims(BaseModel):
    claims: list[NameClaim]


def _names_cfg(cfg: dict) -> dict:
    return cfg.get("names", {})


def build_excerpt(transcript: dict, cfg: dict) -> str:
    """Select the passages most likely to name speakers, in transcript order."""
    ncfg = _names_cfg(cfg)
    head = ncfg.get("head_minutes", 5) * 60
    tail = ncfg.get("tail_minutes", 2) * 60
    ctx = ncfg.get("context_segments", 2)
    max_chars = ncfg.get("max_chars", 12000)

    segs = transcript.get("segments", [])
    if not segs:
        return ""
    end = max(s.get("end", 0) for s in segs)

    keep: set[int] = set()
    for i, s in enumerate(segs):
        text = s.get("text", "")
        if s.get("start", 0) < head or s.get("end", 0) > end - tail:
            keep.add(i)
        elif _CUES.search(text) or _SELF.search(text):
            # Neighbors matter: the person introduced is usually the next speaker.
            keep.update(range(max(0, i - ctx), min(len(segs), i + ctx + 1)))

    lines, prev, size = [], None, 0
    for i in sorted(keep):
        s = segs[i]
        line = f"[{format_timestamp(s.get('start', 0))}] {s.get('speaker', 'UNKNOWN')}: {s.get('text', '').strip()}"
        if prev is not None and i != prev + 1:
            line = "...\n" + line
        if size + len(line) > max_chars:
            lines.append("...")
            break
        lines.append(line)
        size += len(line) + 1
        prev = i
    return "\n".join(lines)


def _prompt_hash() -> str:
    path = project_root() / "pipeline" / "prompts" / "names.j2"
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _input_hash(excerpt: str, cfg: dict) -> str:
    """Fingerprint of exactly what the LLM sees, so rename_speakers rewriting
    the transcript JSON (speaker_display) does not trigger re-extraction."""
    payload = {"excerpt": excerpt, "model": _names_cfg(cfg).get("model"), "prompt": _prompt_hash()}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def _claims_ok(claims: list[NameClaim], transcript: dict, excerpt: str) -> list[dict]:
    """Drop claims for unknown labels or names that never occur in the text."""
    labels = {s["id"] for s in transcript.get("speakers", [])} - {"UNKNOWN"}
    low = excerpt.lower()
    out = []
    for c in claims:
        name = c.name.strip()
        if c.speaker not in labels or not name:
            continue
        # Hallucination guard: at least one name token must actually be in the text.
        if not any(len(t) >= 3 and t.lower() in low for t in re.findall(r"[\w'-]+", name)):
            continue
        out.append({**c.model_dump(), "name": name, "affiliation": c.affiliation.strip()})
    return out


def _extract(excerpt: str, cfg: dict) -> list[NameClaim]:
    import ollama

    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(project_root() / "pipeline" / "prompts")),
        keep_trailing_newline=True,
    )
    messages = [{"role": "user", "content": env.get_template("names.j2").render(excerpt=excerpt)}]
    models = cfg.get("models", {})
    model = _names_cfg(cfg).get("model") or models.get("llm_fallback") or models["llm"]
    num_ctx = max(int(cfg.get("ollama", {}).get("num_ctx", 8192)), 8192)

    last_err = None
    for _ in range(3):
        resp = ollama.chat(
            model=model,
            messages=messages,
            format="json",
            options={"temperature": 0, "num_ctx": num_ctx, "num_predict": 2048},
        )
        content = resp["message"]["content"]
        try:
            return NameClaims.model_validate_json(content).claims
        except Exception as e:  # JSON or schema validation error -> retry with feedback
            last_err = e
            messages = messages[:1] + [
                {"role": "assistant", "content": content},
                {"role": "user", "content":
                    f"Your previous response failed validation:\n{e}\n"
                    'Return ONLY a JSON object of the form {"claims": [...]}.'},
            ]
    raise RuntimeError(f"name extraction failed validation after 3 attempts: {last_err}")


def _norm(name: str) -> str:
    return " ".join(re.sub(r"[^\w\s'-]", " ", name.lower()).split())


def roll_up(cfg: dict, people: dict) -> bool:
    """Recompute every person's ``proposed_names`` from the per-talk caches.

    Returns True if people.yaml content changed.
    """
    speaker_dir = Path(resolve_path("paths.speaker_id", cfg))
    index = ppl.appearance_index(people)

    per_person: dict[str, dict[str, dict]] = {}
    for cache in sorted(speaker_dir.glob("*/names/*.json")):
        collection, stem = cache.parent.parent.name, cache.stem
        try:
            claims = json.loads(cache.read_text(encoding="utf-8")).get("claims", [])
        except (json.JSONDecodeError, OSError):
            continue
        for c in claims:
            pid = index.get((collection, stem, c["speaker"]))
            if not pid:
                continue
            bucket = per_person.setdefault(pid, {})
            rec = bucket.setdefault(_norm(c["name"]), {
                "name": c["name"], "affiliation": "", "score": 0, "basis": set(), "evidence": [],
            })
            rec["score"] += _WEIGHTS[c["basis"]]
            rec["basis"].add(c["basis"])
            if c.get("affiliation") and not rec["affiliation"]:
                rec["affiliation"] = c["affiliation"]
            # Prefer the fullest spelling seen ("Jane Doe" over "jane doe").
            if len(c["name"]) >= len(rec["name"]) and c["name"] != c["name"].lower():
                rec["name"] = c["name"]
            if len(rec["evidence"]) < 2:
                rec["evidence"].append({
                    "source": f"{collection}/{stem} @ {c.get('timestamp', '')}".rstrip(" @"),
                    "quote": c.get("evidence", "")[:200],
                })

    changed = False
    for pid, person in people.items():
        bucket = per_person.get(pid, {})
        # Fold a lone first name ("Jane", from Q&A) into a full name with that first token.
        for key in [k for k in bucket if " " not in k]:
            full = [k for k in bucket if " " in k and k.split()[0] == key]
            if len(full) == 1:
                tgt, src = bucket[full[0]], bucket.pop(key)
                tgt["score"] += src["score"]
                tgt["basis"] |= src["basis"]
        proposals = [
            {**r, "basis": sorted(r["basis"])}
            for r in sorted(bucket.values(), key=lambda r: (-r["score"], r["name"]))
        ]
        if proposals:
            if person.get("proposed_names") != proposals:
                person["proposed_names"] = proposals
                changed = True
        elif "proposed_names" in person:
            del person["proposed_names"]
            changed = True
    return changed


def _accept(people: dict, pid: str, pick: int, log) -> bool:
    person = people.get(pid)
    if person is None:
        log.error("Unknown person id: %s", pid)
        return False
    props = person.get("proposed_names") or []
    if not 1 <= pick <= len(props):
        log.error("%s has %d proposal(s); --pick %d is out of range", pid, len(props), pick)
        return False
    choice = props[pick - 1]
    old = ppl.display_name(pid, person)
    if ppl.is_named(pid, person) and old != choice["name"]:
        aliases = person.setdefault("aliases", [])
        if old not in aliases:
            aliases.append(old)
    person["display_name"] = choice["name"]
    if choice.get("affiliation") and not person.get("affiliation"):
        person["affiliation"] = choice["affiliation"]
    log.info("  %s -> %s%s", pid, choice["name"],
             f" ({choice['affiliation']})" if choice.get("affiliation") else "")
    return True


@app.command()
def main(
    files: list[str] = typer.Argument(None, help="Only these recordings (name or stem)"),
    accept: str = typer.Option("", "--accept", help="Person id whose proposal to accept"),
    pick: int = typer.Option(1, "--pick", help="Which proposal to accept (1 = top)"),
    accept_all: bool = typer.Option(False, "--accept-all",
                                    help="Accept every unnamed person's top proposal if unambiguous"),
    force: bool = typer.Option(False, "--force", help="Re-extract even if inputs are unchanged"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be processed"),
    verbose: bool = typer.Option(False, "--verbose", help="Debug logging"),
) -> None:
    """Propose speaker names from introductions and roll them up into people.yaml."""
    log = setup_logging("names", verbose)
    cfg = load_config()
    people = ppl.load_people(cfg)

    if accept or accept_all:
        if not people:
            log.error("No people database yet. Run 'python -m pipeline.speakers' first.")
            raise typer.Exit(1)
        min_score = _names_cfg(cfg).get("auto_accept_score", 3)
        if accept:
            targets = [(accept, pick)]
        else:
            targets = []
            for pid, person in sorted(people.items()):
                props = person.get("proposed_names") or []
                if ppl.is_named(pid, person) or not props or props[0]["score"] < min_score:
                    continue
                if len(props) > 1 and props[1]["score"] * 2 > props[0]["score"]:
                    log.info("  %s: ambiguous (%s vs %s) -- skipping", pid, props[0]["name"], props[1]["name"])
                    continue
                targets.append((pid, 1))
        ok = [t for t in targets if dry_run or _accept(people, *t, log)]
        if dry_run:
            log.info("Would accept %d proposal(s): %s", len(targets), ", ".join(p for p, _ in targets))
            return
        if accept and not ok:
            raise typer.Exit(1)
        if ok:
            ppl.save_people(cfg, people)
            log.info("Accepted %d. Run 'python -m pipeline.rename_speakers' to update transcripts.", len(ok))
        else:
            log.info("Nothing to accept (min score %d, unambiguous only)", min_score)
        return

    if not _names_cfg(cfg).get("enabled", True):
        log.info("Name proposals disabled (names.enabled: false)")
        return

    transcript_dir = resolve_path("paths.transcripts", cfg)
    speaker_dir = Path(resolve_path("paths.speaker_id", cfg))
    wanted = {Path(f).stem for f in files} if files else None

    todo: list[tuple[str, str, dict, str, Path, str]] = []
    for collection in discover_collections(transcript_dir):
        for stem, tp in collection_transcripts(Path(transcript_dir) / collection).items():
            if wanted is not None and stem not in wanted:
                continue
            with open(tp, "r", encoding="utf-8") as f:
                transcript = json.load(f)
            excerpt = build_excerpt(transcript, cfg)
            cache = speaker_dir / collection / "names" / f"{stem}.json"
            h = _input_hash(excerpt, cfg)
            if not force and cache.exists():
                try:
                    if json.loads(cache.read_text(encoding="utf-8")).get("input_hash") == h:
                        log.debug("  %s/%s: up to date", collection, stem)
                        continue
                except (json.JSONDecodeError, OSError):
                    pass
            todo.append((collection, stem, transcript, excerpt, cache, h))

    if dry_run:
        for collection, stem, *_ in todo:
            log.info("  %s/%s: would extract name claims", collection, stem)
        log.info("%d talk(s) to extract", len(todo))
        return

    llm_down = False
    for collection, stem, transcript, excerpt, cache, h in todo:
        if llm_down:
            break
        claims: list[dict] = []
        if excerpt:
            try:
                log.info("  %s/%s: extracting name claims (%d chars)...", collection, stem, len(excerpt))
                claims = _claims_ok(_extract(excerpt, cfg), transcript, excerpt)
            except ConnectionError as e:
                log.warning("Ollama unreachable (%s) -- skipping name extraction this run", e)
                llm_down = True
                continue
            except Exception as e:
                if "connect" in str(e).lower():
                    log.warning("Ollama unreachable (%s) -- skipping name extraction this run", e)
                    llm_down = True
                else:
                    log.warning("  %s/%s: skipped -- %s", collection, stem, e)
                continue
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps({
            "input_hash": h,
            "generated_at": datetime.now().isoformat(),
            "claims": claims,
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        for c in claims:
            log.info("    %s = %s (%s)", c["speaker"], c["name"], c["basis"])

    if not people:
        log.info("No people database yet -- claims cached; they roll up after 'pipeline.speakers'")
        return

    if roll_up(cfg, people):
        ppl.save_people(cfg, people)

    # Report: proposals for unnamed people, and named people whose talks disagree.
    pending = 0
    for pid, person in sorted(people.items()):
        props = person.get("proposed_names") or []
        if not props:
            continue
        top = props[0]
        if not ppl.is_named(pid, person):
            pending += 1
            log.info("  %s: proposed %s (score %d)%s", pid, top["name"], top["score"],
                     f", also: {', '.join(p['name'] for p in props[1:3])}" if len(props) > 1 else "")
        else:
            known = {_norm(person["display_name"]), *(_norm(a) for a in person.get("aliases", []))}
            if _norm(top["name"]) not in known and top["score"] >= _names_cfg(cfg).get("auto_accept_score", 3):
                log.warning("  %s is named %r but talks suggest %r -- check for a wrong match",
                            pid, person["display_name"], top["name"])
    log.info("Name proposals: %d unnamed people have proposals (accept with --accept <id> or --accept-all)",
             pending)


if __name__ == "__main__":
    app()
