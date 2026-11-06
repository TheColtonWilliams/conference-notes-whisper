"""Phase 1: Audio preprocessing -- m4a/mp3/wav -> 16 kHz mono WAV, loudness-normalized.

Usage:
    python -m pipeline.preprocess                     # all files
    python -m pipeline.preprocess "keynote.m4a"           # single file
    python -m pipeline.preprocess --verify            # check for missing/stale outputs
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pyloudnorm as pyln
import soundfile as sf
import typer

from pipeline.utils import (
    collection_root,
    discover_sources,
    ensure_dir,
    is_stale,
    load_config,
    resolve_path,
    setup_logging,
    write_meta,
)

app = typer.Typer(add_completion=False)

# Model keys relevant to preprocessing (none -- pure signal processing)
_MODEL_KEYS: list[str] = []

FFMPEG_PATH: str | None = None


def _find_ffmpeg() -> str:
    """Find ffmpeg executable, checking PATH and known install locations."""
    global FFMPEG_PATH
    if FFMPEG_PATH is not None:
        return FFMPEG_PATH

    # Check PATH first
    path = shutil.which("ffmpeg")
    if path:
        FFMPEG_PATH = path
        return FFMPEG_PATH

    # Known winget install location
    import glob

    patterns = [
        str(Path.home() / "AppData/Local/Microsoft/WinGet/Packages/Gyan.FFmpeg*/ffmpeg-*-full_build/bin/ffmpeg.exe"),
        "C:/ProgramData/chocolatey/bin/ffmpeg.exe",
    ]
    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            FFMPEG_PATH = matches[0]
            return FFMPEG_PATH

    raise FileNotFoundError(
        "ffmpeg not found. Install with: winget install Gyan.FFmpeg"
    )


def _preprocess_single(args: tuple[Path, Path, dict]) -> tuple[str, bool, str]:
    """Preprocess one audio file. Returns (filename, success, message)."""
    source, out_dir, cfg = args
    stem = source.stem
    out_wav = out_dir / f"{stem}.wav"
    meta_path = out_dir / f"{stem}.meta.json"

    # Idempotency check
    if not is_stale(source, out_wav, meta_path, cfg, _MODEL_KEYS):
        return (source.name, True, "skipped (up to date)")

    target_lufs = cfg.get("preprocess", {}).get("target_lufs", -23)
    peak_limit = cfg.get("preprocess", {}).get("peak_limit_dbtp", -1)
    target_sr = cfg.get("preprocess", {}).get("sample_rate", 16000)

    try:
        ffmpeg = _find_ffmpeg()

        # Step 1: Convert to 16kHz mono WAV via ffmpeg
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        cmd = [
            ffmpeg,
            "-y",
            "-i", str(source),
            "-ac", "1",
            "-ar", str(target_sr),
            "-sample_fmt", "s16",
            str(tmp_path),
        ]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
        )
        if not tmp_path.exists() or tmp_path.stat().st_size == 0:
            return (source.name, False, f"ffmpeg failed: {result.stderr[-500:]}")

        # Step 2: Load and measure loudness
        data, sr = sf.read(str(tmp_path))
        meter = pyln.Meter(sr)
        loudness_before = meter.integrated_loudness(data)

        # Step 3: Normalize to target LUFS
        if np.isinf(loudness_before) or np.isnan(loudness_before):
            # Silent or nearly silent audio -- skip normalization
            loudness_after = loudness_before
        else:
            data = pyln.normalize.loudness(data, loudness_before, target_lufs)

            # Peak limiting: if peak exceeds limit, attenuate
            peak_db = peak_limit
            peak_linear = 10 ** (peak_db / 20.0)
            max_peak = np.max(np.abs(data))
            if max_peak > peak_linear:
                data = data * (peak_linear / max_peak)

            loudness_after = meter.integrated_loudness(data)

        # Step 4: Get duration
        duration_sec = len(data) / sr

        # Step 5: Write output
        ensure_dir(out_dir)
        sf.write(str(out_wav), data, sr, subtype="PCM_16")

        # Step 6: Write metadata
        write_meta(
            meta_path,
            source,
            cfg,
            _MODEL_KEYS,
            duration_sec=duration_sec,
            sample_rate=sr,
            channels=1,
            lufs_before=round(loudness_before, 2) if not np.isinf(loudness_before) else None,
            lufs_after=round(loudness_after, 2) if not np.isinf(loudness_after) else None,
            target_lufs=target_lufs,
            peak_limit_dbtp=peak_limit,
        )

        # Cleanup temp file
        tmp_path.unlink(missing_ok=True)

        return (
            source.name,
            True,
            f"OK ({duration_sec:.0f}s, {loudness_before:.1f} -> {loudness_after:.1f} LUFS)",
        )

    except Exception as e:
        return (source.name, False, str(e))


@app.command()
def main(
    files: list[str] = typer.Argument(None, help="Specific files to process"),
    verify: bool = typer.Option(False, "--verify", help="Check for missing/stale outputs"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be processed"),
    verbose: bool = typer.Option(False, "--verbose", help="Debug logging"),
    workers: int = typer.Option(None, "--workers", help="Override worker count"),
) -> None:
    """Preprocess audio files: convert to 16kHz mono WAV + EBU R128 loudness normalization."""
    log = setup_logging("preprocess", verbose)
    cfg = load_config()

    raw_dir = resolve_path("paths.raw_audio", cfg)
    extensions = cfg.get("audio_extensions", [".m4a", ".mp3", ".wav", ".flac", ".ogg"])

    sources = discover_sources(raw_dir, extensions=extensions, specific_files=files)
    if not sources:
        log.warning("No audio files found under %s/<collection>/", raw_dir)
        raise typer.Exit(1)

    # Per-collection output dir = audio_norm/<collection>/ (collection = parent folder name)
    def _out_dir(src: Path) -> Path:
        return collection_root(cfg, "paths.normalized", src.parent.name)

    log.info("Found %d audio file(s) across %d collection(s)",
             len(sources), len({s.parent.name for s in sources}))

    if verify:
        stale = []
        for src in sources:
            out_dir = _out_dir(src)
            stem = src.stem
            out_wav = out_dir / f"{stem}.wav"
            meta_path = out_dir / f"{stem}.meta.json"
            if is_stale(src, out_wav, meta_path, cfg, _MODEL_KEYS):
                stale.append(src.name)
                log.warning("STALE or MISSING: %s/%s", src.parent.name, src.name)
            else:
                log.info("OK: %s/%s", src.parent.name, src.name)
        if stale:
            log.warning("%d file(s) need preprocessing", len(stale))
            raise typer.Exit(1)
        else:
            log.info("All files are up to date")
        return

    if dry_run:
        for src in sources:
            out_dir = _out_dir(src)
            stem = src.stem
            out_wav = out_dir / f"{stem}.wav"
            meta_path = out_dir / f"{stem}.meta.json"
            status = "SKIP" if not is_stale(src, out_wav, meta_path, cfg, _MODEL_KEYS) else "PROCESS"
            log.info("[%s] %s/%s -> %s", status, src.parent.name, src.name, out_wav)
        return

    # Find ffmpeg early to fail fast
    try:
        ffmpeg = _find_ffmpeg()
        log.info("Using ffmpeg: %s", ffmpeg)
    except FileNotFoundError as e:
        log.error(str(e))
        raise typer.Exit(1)

    # Process files in parallel
    n_workers = workers or cfg.get("preprocess", {}).get("workers", 4)
    n_workers = min(n_workers, len(sources))
    tasks = [(src, _out_dir(src), cfg) for src in sources]

    log.info("Processing %d file(s) with %d worker(s)", len(sources), n_workers)

    if n_workers == 1:
        results = [_preprocess_single(t) for t in tasks]
    else:
        with Pool(n_workers) as pool:
            results = pool.map(_preprocess_single, tasks)

    # Report results
    failures = 0
    for name, success, msg in results:
        if success:
            log.info("  %s: %s", name, msg)
        else:
            log.error("  %s: FAILED -- %s", name, msg)
            failures += 1

    if failures:
        log.error("%d file(s) failed", failures)
        raise typer.Exit(1)

    log.info("Preprocessing complete -- %d file(s) processed", len(results))


if __name__ == "__main__":
    app()
