"""Phase 2.5: LLM-based transcript correction -- rule-based + Ollama qwen3.

Usage:
    python -m pipeline.correct                          # all transcripts
    python -m pipeline.correct "keynote.m4a"            # single file
    python -m pipeline.correct --diff                   # show corrections without writing
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import jinja2
import typer
import yaml

from pipeline.utils import (
    discover_collections,
    is_stale,
    load_config,
    project_root,
    read_meta,
    resolve_path,
    setup_logging,
    write_meta,
)

app = typer.Typer(add_completion=False)

_MODEL_KEYS = ["correction.model"]


def _corrections_hash(corrections: dict[str, str]) -> str:
    """Stable hash of the domain corrections, for re-correction staleness."""
    return hashlib.sha256(
        json.dumps(corrections, sort_keys=True).encode()
    ).hexdigest()[:16]


def _load_corrections(cfg: dict) -> dict[str, str]:
    """Load domain corrections from YAML file."""
    corrections_file = project_root() / cfg.get("correction", {}).get(
        "corrections_file", "pipeline/prompts/domain_corrections.yaml"
    )
    if not corrections_file.exists():
        return {}
    with open(corrections_file, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("corrections", {})


def _apply_rule_based(segments: list[dict], corrections: dict[str, str]) -> tuple[list[dict], int]:
    """Apply rule-based corrections. Returns (corrected_segments, correction_count)."""
    count = 0
    corrected = []
    for seg in segments:
        text = seg["text"]
        for wrong, right in corrections.items():
            pattern = re.compile(re.escape(wrong), re.IGNORECASE)
            new_text = pattern.sub(right, text)
            if new_text != text:
                count += 1
                text = new_text
        corrected.append({**seg, "text": text})
    return corrected, count


def _apply_llm_corrections(
    segments: list[dict],
    corrections: dict[str, str],
    cfg: dict,
    log,
) -> tuple[list[dict], int]:
    """Apply LLM-based corrections via Ollama. Returns (corrected_segments, correction_count)."""
    try:
        import ollama as ollama_client
    except ImportError:
        log.warning("ollama package not available, skipping LLM corrections")
        return segments, 0

    model = cfg.get("correction", {}).get("model", cfg["models"]["llm"])
    template_dir = project_root() / "pipeline" / "prompts"
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(template_dir)),
        keep_trailing_newline=True,
    )
    template = env.get_template("correct.j2")

    # Process in batches of ~20 segments (~3000 tokens)
    batch_size = 20
    all_corrected = []
    total_corrections = 0

    n_batches = (len(segments) + batch_size - 1) // batch_size
    for i in range(0, len(segments), batch_size):
        batch_num = i // batch_size + 1
        log.info("    LLM batch %d/%d (%d segments)...", batch_num, n_batches, len(segments[i:i + batch_size]))
        batch = segments[i:i + batch_size]
        batch_simple = [
            {"id": s["id"], "start": s["start"], "end": s["end"],
             "speaker": s["speaker"], "text": s["text"]}
            for s in batch
        ]

        prompt = template.render(
            corrections=corrections,
            segments_json=json.dumps(batch_simple, indent=2),
        )

        try:
            response = ollama_client.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                options={
                    "num_ctx": cfg.get("ollama", {}).get("num_ctx", 32768),
                    "temperature": 0,
                },
            )

            response_text = response["message"]["content"]

            # Extract JSON from response (handle markdown code blocks)
            json_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", response_text, re.DOTALL)
            if json_match:
                response_text = json_match.group(1)

            corrected_batch = json.loads(response_text)

            # Validate and merge
            for orig, corrected in zip(batch, corrected_batch):
                new_text = corrected.get("text", orig["text"])

                # Safety: word count within 20%
                orig_words = len(orig["text"].split())
                new_words = len(new_text.split())
                if orig_words > 0:
                    ratio = new_words / orig_words
                    if ratio < 0.8 or ratio > 1.2:
                        log.debug(
                            "Rejected LLM correction for seg %d: word count ratio %.2f",
                            orig["id"], ratio,
                        )
                        new_text = orig["text"]

                if new_text != orig["text"]:
                    total_corrections += 1

                all_corrected.append({**orig, "text": new_text})

        except Exception as e:
            log.warning("LLM batch %d failed: %s -- keeping originals", i // batch_size, e)
            all_corrected.extend(batch)

    return all_corrected, total_corrections


def _correct_file(
    transcript_path: Path,
    cfg: dict,
    corrections: dict[str, str],
    log,
    dry_run: bool = False,
    diff_mode: bool = False,
) -> bool:
    """Correct one transcript file. Returns True on success."""
    stem = transcript_path.stem
    out_path = transcript_path.parent / f"{stem}.clean.json"
    meta_path = transcript_path.parent / f"{stem}.clean.meta.json"

    # Idempotency. Beyond the usual source-sha/model-config check, also re-correct
    # when domain_corrections.yaml changed (feedback loop 1): a newly-verified
    # acronym must fix its spelling in existing transcripts. _MODEL_KEYS can't see
    # file contents, so we track a content hash of the corrections separately.
    corr_hash = _corrections_hash(corrections)
    if not diff_mode and not is_stale(transcript_path, out_path, meta_path, cfg, _MODEL_KEYS):
        meta = read_meta(meta_path) or {}
        if meta.get("corrections_sha256") == corr_hash:
            log.info("  %s: skipped (up to date)", stem)
            return True
        log.info("  %s: domain corrections changed -- re-correcting", stem)

    with open(transcript_path, "r", encoding="utf-8") as f:
        transcript = json.load(f)

    segments = transcript.get("segments", [])
    if not segments:
        log.warning("  %s: no segments found", stem)
        return True

    # Pass 1: Rule-based corrections
    corrected, rule_count = _apply_rule_based(segments, corrections)
    log.info("  %s: rule-based -- %d corrections", stem, rule_count)

    # Pass 2: LLM corrections (if enabled and not dry-run)
    llm_count = 0
    correction_cfg = cfg.get("correction", {})
    if correction_cfg.get("enabled", True) and not dry_run:
        try:
            corrected, llm_count = _apply_llm_corrections(corrected, corrections, cfg, log)
            log.info("  %s: LLM -- %d corrections", stem, llm_count)
        except Exception as e:
            log.warning("  %s: LLM corrections failed: %s -- using rule-based only", stem, e)

    total = rule_count + llm_count

    if diff_mode:
        # Show diffs
        changes = 0
        for orig, corr in zip(segments, corrected):
            if orig["text"] != corr["text"]:
                changes += 1
                log.info("  [%d] ORIG: %s", orig["id"], orig["text"])
                log.info("  [%d]  NEW: %s", orig["id"], corr["text"])
        log.info("  %s: %d changes shown", stem, changes)
        return True

    if dry_run:
        log.info("  %s: would apply %d rule-based corrections", stem, rule_count)
        return True

    # Build clean transcript
    clean = {
        **transcript,
        "correction_model": cfg.get("correction", {}).get("model", cfg["models"]["llm"]),
        "corrections_applied": total,
        "segments": corrected,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(clean, f, indent=2, ensure_ascii=False)

    write_meta(
        meta_path,
        transcript_path,
        cfg,
        _MODEL_KEYS,
        corrections_rule=rule_count,
        corrections_llm=llm_count,
        corrections_total=total,
        corrections_sha256=corr_hash,
    )

    log.info("  %s: done -- %d total corrections (%d rule, %d LLM)", stem, total, rule_count, llm_count)
    return True


@app.command()
def main(
    files: list[str] = typer.Argument(None, help="Specific source files to correct"),
    diff: bool = typer.Option(False, "--diff", help="Show corrections without writing"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be processed"),
    verbose: bool = typer.Option(False, "--verbose", help="Debug logging"),
) -> None:
    """Correct transcripts: rule-based domain fixes + LLM disfluency cleanup."""
    log = setup_logging("correct", verbose)
    cfg = load_config()

    corrections = _load_corrections(cfg)
    log.info("Loaded %d domain corrections", len(corrections))

    transcript_dir = resolve_path("paths.transcripts", cfg)

    # Find transcript JSON files to correct across all collections
    wanted_stems = {Path(n).stem for n in files} if files else None
    transcript_paths = []
    for collection in discover_collections(transcript_dir):
        col_dir = Path(transcript_dir) / collection
        for p in sorted(col_dir.glob("*.json")):
            if p.name.endswith(".clean.json") or p.name.endswith(".meta.json"):
                continue
            if wanted_stems is not None and p.stem not in wanted_stems:
                continue
            transcript_paths.append(p)

    if files and not transcript_paths:
        log.warning("No transcripts found matching: %s", ", ".join(files))
    if not transcript_paths:
        log.warning("No transcripts found to correct")
        raise typer.Exit(1)

    log.info("Found %d transcript(s) to correct", len(transcript_paths))

    failures = 0
    for tp in transcript_paths:
        try:
            success = _correct_file(tp, cfg, corrections, log, dry_run=dry_run, diff_mode=diff)
            if not success:
                failures += 1
        except Exception as e:
            log.error("  %s: FAILED -- %s", tp.stem, e, exc_info=verbose)
            failures += 1

    if failures:
        log.error("%d file(s) failed", failures)
        raise typer.Exit(1)

    log.info("Correction complete -- %d file(s)", len(transcript_paths))


if __name__ == "__main__":
    app()
