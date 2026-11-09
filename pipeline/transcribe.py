"""Phase 2: Transcription -- WhisperX ASR + wav2vec2 alignment + pyannote diarization.

Usage:
    python -m pipeline.transcribe                          # all files
    python -m pipeline.transcribe "keynote.m4a"            # single file
    python -m pipeline.transcribe --smoke                  # 60s fixture test
"""

from __future__ import annotations

import json
from pathlib import Path

import jinja2
import torch
import typer

from pipeline.utils import (
    collection_root,
    discover_sources,
    ensure_dir,
    format_timestamp,
    format_vtt_timestamp,
    gpu_cleanup,
    is_stale,
    load_config,
    project_root,
    resolve_device,
    resolve_path,
    setup_logging,
    sha256_file,
    hf_model_cached,
    verify_hf_token,
    write_meta,
)

app = typer.Typer(add_completion=False)

_MODEL_KEYS = ["models.asr", "models.asr_compute", "models.diarization", "asr"]


def _build_transcript_json(
    result: dict,
    source_path: Path,
    wav_path: Path,
    cfg: dict,
    duration_sec: float,
) -> dict:
    """Build the canonical transcript JSON from WhisperX output."""
    # Collect speaker stats
    speaker_stats: dict[str, dict] = {}
    segments_out = []

    for i, seg in enumerate(result.get("segments", [])):
        speaker = seg.get("speaker", "UNKNOWN")
        text = seg.get("text", "").strip()
        start = seg.get("start", 0.0)
        end = seg.get("end", 0.0)

        if speaker not in speaker_stats:
            speaker_stats[speaker] = {
                "id": speaker,
                "global_id": None,
                "display_name": speaker,
                "turns": 0,
                "speaking_time_sec": 0.0,
            }
        speaker_stats[speaker]["turns"] += 1
        speaker_stats[speaker]["speaking_time_sec"] += end - start

        words_out = []
        for w in seg.get("words", []):
            words_out.append({
                "start": w.get("start"),
                "end": w.get("end"),
                "word": w.get("word", ""),
                "score": w.get("score"),
                "speaker": w.get("speaker", speaker),
            })

        segments_out.append({
            "id": i,
            "start": round(start, 3),
            "end": round(end, 3),
            "speaker": speaker,
            "text": text,
            "words": words_out,
        })

    return {
        "source": source_path.name,
        "sha256": sha256_file(source_path),
        "duration_sec": round(duration_sec, 2),
        "language": result.get("language", cfg.get("asr", {}).get("language", "en")),
        "language_probability": result.get("language_probability"),
        "sample_rate": cfg.get("preprocess", {}).get("sample_rate", 16000),
        "model": {
            "asr": f"faster-whisper-{cfg['models']['asr']}",
            "asr_compute": cfg["models"]["asr_compute"],
            "align": cfg["models"].get("align_model", "wav2vec2"),
            "diar": cfg["models"]["diarization"],
        },
        "speakers": sorted(speaker_stats.values(), key=lambda s: s["id"]),
        "segments": segments_out,
    }


def _merge_same_speaker_turns(segments: list[dict]) -> list[dict]:
    """Merge consecutive segments from the same speaker into turns for markdown."""
    if not segments:
        return []

    turns = []
    current = {
        "speaker": segments[0]["speaker"],
        "start": segments[0]["start"],
        "texts": [segments[0]["text"]],
    }

    for seg in segments[1:]:
        if seg["speaker"] == current["speaker"]:
            current["texts"].append(seg["text"])
        else:
            turns.append({
                "timestamp": format_timestamp(current["start"]),
                "speaker": current["speaker"],
                "text": " ".join(current["texts"]),
            })
            current = {
                "speaker": seg["speaker"],
                "start": seg["start"],
                "texts": [seg["text"]],
            }

    turns.append({
        "timestamp": format_timestamp(current["start"]),
        "speaker": current["speaker"],
        "text": " ".join(current["texts"]),
    })
    return turns


def _render_markdown(transcript: dict, cfg: dict) -> str:
    """Render transcript JSON to markdown via Jinja2 template."""
    template_dir = project_root() / "pipeline" / "templates"
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(template_dir)),
        keep_trailing_newline=True,
    )
    template = env.get_template("transcript.md.j2")

    turns = _merge_same_speaker_turns(transcript["segments"])
    duration_str = format_timestamp(transcript["duration_sec"])

    return template.render(
        source=transcript["source"],
        duration=duration_str,
        language=transcript["language"],
        speakers=transcript["speakers"],
        model=transcript["model"],
        generated_at=transcript.get("generated_at", ""),
        title=Path(transcript["source"]).stem,
        turns=turns,
    )


def _render_vtt(transcript: dict) -> str:
    """Render transcript JSON to WebVTT format."""
    lines = ["WEBVTT", ""]
    for seg in transcript["segments"]:
        start_vtt = format_vtt_timestamp(seg["start"])
        end_vtt = format_vtt_timestamp(seg["end"])
        lines.append(str(seg["id"]))
        lines.append(f"{start_vtt} --> {end_vtt}")
        lines.append(f"<v {seg['speaker']}>{seg['text']}")
        lines.append("")
    return "\n".join(lines)


def _transcribe_file(
    source_path: Path,
    wav_path: Path,
    out_dir: Path,
    raw_dir: Path,
    cfg: dict,
    log,
    dry_run: bool = False,
) -> bool:
    """Run the full WhisperX pipeline on one file. Returns True on success."""
    import whisperx

    stem = source_path.stem
    out_json = out_dir / f"{stem}.json"
    out_md = out_dir / f"{stem}.md"
    out_vtt = out_dir / f"{stem}.vtt"
    meta_path = out_dir / f"{stem}.meta.json"

    # Idempotency
    if not is_stale(source_path, out_json, meta_path, cfg, _MODEL_KEYS):
        log.info("  %s: skipped (up to date)", source_path.name)
        return True

    if dry_run:
        log.info("  %s: would process", source_path.name)
        return True

    device = resolve_device(cfg)
    compute_type = cfg["models"]["asr_compute"]
    asr_cfg = cfg.get("asr", {})

    log.info("  %s: loading audio...", stem)
    audio = whisperx.load_audio(str(wav_path))
    duration_sec = len(audio) / cfg.get("preprocess", {}).get("sample_rate", 16000)

    # --- Stage 1: ASR ---
    log.info("  %s: ASR (model=%s, compute=%s, device=%s)...", stem, cfg["models"]["asr"], compute_type, device)
    asr_model = whisperx.load_model(
        cfg["models"]["asr"],
        device,
        compute_type=compute_type,
        asr_options={
            "beam_size": asr_cfg.get("beam_size", 5),
            "initial_prompt": asr_cfg.get("initial_prompt", ""),
        },
        vad_options={
            "vad_onset": 0.500,
            "vad_offset": 0.363,
        },
        language=asr_cfg.get("language", "en"),
        threads=asr_cfg.get("inter_threads", 4),
    )
    result = asr_model.transcribe(
        audio,
        batch_size=asr_cfg.get("batch_size", 16),
        language=asr_cfg.get("language", "en"),
    )

    # Archive raw ASR output
    ensure_dir(raw_dir)
    with open(raw_dir / f"{stem}.whisperx.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)

    # Free ASR model before loading alignment
    gpu_cleanup(asr_model)
    log.info("  %s: ASR complete, %d segments", stem, len(result.get("segments", [])))

    # --- Stage 2: Alignment ---
    log.info("  %s: aligning (wav2vec2)...", stem)
    align_model, align_meta = whisperx.load_align_model(
        language_code=result.get("language", "en"),
        device=device,
    )
    result = whisperx.align(
        result["segments"],
        align_model,
        align_meta,
        audio,
        device,
        return_char_alignments=False,
    )

    # Free alignment model before diarization
    gpu_cleanup(align_model)
    log.info("  %s: alignment complete", stem)

    # --- Stage 3: Diarization ---
    hf_token = verify_hf_token()
    if not hf_token and not hf_model_cached(cfg["models"]["diarization"]):
        log.warning("  %s: HF_TOKEN not set and model not cached -- skipping diarization", stem)
    else:
        diar_cfg = cfg.get("diarization", {})
        diar_device = diar_cfg.get("run_on", "cuda") if torch.cuda.is_available() else "cpu"

        log.info("  %s: diarizing (pyannote, device=%s)...", stem, diar_device)
        from whisperx.diarize import DiarizationPipeline
        diarize_model = DiarizationPipeline(
            model_name=cfg["models"]["diarization"],
            token=hf_token,
            device=torch.device(diar_device),
        )

        diar_kwargs = {}
        if diar_cfg.get("min_speakers"):
            diar_kwargs["min_speakers"] = diar_cfg["min_speakers"]
        if diar_cfg.get("max_speakers"):
            diar_kwargs["max_speakers"] = diar_cfg["max_speakers"]

        diarize_segments = diarize_model(audio, **diar_kwargs)

        # Free diarization model
        gpu_cleanup(diarize_model)
        log.info("  %s: diarization complete", stem)

        # --- Stage 4: Assign speakers to words ---
        result = whisperx.assign_word_speakers(diarize_segments, result)

        # Archive diarization
        if hasattr(diarize_segments, "to_csv"):
            diarize_segments.to_csv(str(raw_dir / f"{stem}.diar.rttm"))

    # --- Build output ---
    log.info("  %s: building outputs...", stem)
    transcript = _build_transcript_json(result, source_path, wav_path, cfg, duration_sec)

    from datetime import datetime
    transcript["generated_at"] = datetime.now().isoformat()

    # Write JSON
    ensure_dir(out_dir)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(transcript, f, indent=2, ensure_ascii=False)

    # Write markdown
    md_content = _render_markdown(transcript, cfg)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(md_content)

    # Write VTT
    vtt_content = _render_vtt(transcript)
    with open(out_vtt, "w", encoding="utf-8") as f:
        f.write(vtt_content)

    # Write metadata
    write_meta(
        meta_path,
        source_path,
        cfg,
        _MODEL_KEYS,
        duration_sec=duration_sec,
        segments=len(transcript["segments"]),
        speakers=len(transcript["speakers"]),
    )

    n_speakers = len(transcript["speakers"])
    n_segments = len(transcript["segments"])
    log.info(
        "  %s: done -- %d segments, %d speakers, %.0fs",
        stem, n_segments, n_speakers, duration_sec,
    )
    return True


@app.command()
def main(
    files: list[str] = typer.Argument(None, help="Specific source files to transcribe"),
    smoke: bool = typer.Option(False, "--smoke", help="Run on 60s test fixture only"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be processed"),
    verbose: bool = typer.Option(False, "--verbose", help="Debug logging"),
) -> None:
    """Transcribe audio files with WhisperX: ASR + alignment + diarization."""
    log = setup_logging("transcribe", verbose)
    cfg = load_config()

    # Validate HF token early (only needed to download a gated model once)
    hf_token = verify_hf_token()
    if not hf_token and not smoke and not hf_model_cached(cfg["models"]["diarization"]):
        log.warning(
            "HF_TOKEN not set in .env and diarization model not cached. "
            "Diarization will be skipped. Set HF_TOKEN once to download it."
        )

    if smoke:
        fixture = project_root() / "tests" / "fixtures" / "short_sample.wav"
        if not fixture.exists():
            log.error("Smoke test fixture not found: %s", fixture)
            raise typer.Exit(1)

        log.info("Smoke test: processing %s", fixture.name)
        out_dir = collection_root(cfg, "paths.transcripts", "smoke")
        raw_dir = Path(out_dir) / "raw"

        # For smoke test, use the fixture as both source and wav
        success = _transcribe_file(
            source_path=fixture,
            wav_path=fixture,
            out_dir=Path(out_dir),
            raw_dir=raw_dir,
            cfg=cfg,
            log=log,
            dry_run=dry_run,
        )
        if success:
            log.info("Smoke test passed!")
        else:
            log.error("Smoke test failed!")
            raise typer.Exit(1)
        return

    # Normal mode: process audio_norm WAVs
    raw_audio_dir = resolve_path("paths.raw_audio", cfg)
    extensions = cfg.get("audio_extensions", [".m4a", ".mp3", ".wav", ".flac", ".ogg"])

    # Find source files (original audio for sha256 tracking)
    sources = discover_sources(raw_audio_dir, extensions=extensions, specific_files=files)
    if not sources:
        log.warning("No audio files found under %s/<collection>/", raw_audio_dir)
        raise typer.Exit(1)

    log.info("Found %d audio file(s) to transcribe across %d collection(s)",
             len(sources), len({s.parent.name for s in sources}))

    failures = 0
    for source in sources:
        collection = source.parent.name
        norm_dir = collection_root(cfg, "paths.normalized", collection)
        out_dir = collection_root(cfg, "paths.transcripts", collection)
        raw_dir = Path(out_dir) / "raw"
        wav_path = Path(norm_dir) / f"{source.stem}.wav"
        if not wav_path.exists():
            log.error("  %s/%s: normalized WAV not found. Run preprocess first.",
                      collection, source.name)
            failures += 1
            continue

        try:
            success = _transcribe_file(
                source_path=source,
                wav_path=wav_path,
                out_dir=Path(out_dir),
                raw_dir=raw_dir,
                cfg=cfg,
                log=log,
                dry_run=dry_run,
            )
            if not success:
                failures += 1
        except Exception as e:
            log.error("  %s: FAILED -- %s", source.name, e, exc_info=verbose)
            failures += 1

    if failures:
        log.error("%d file(s) failed", failures)
        raise typer.Exit(1)

    log.info("Transcription complete -- %d file(s)", len(sources))


if __name__ == "__main__":
    app()
