"""Phase 0: Pull the audio out of video recordings so they enter the normal pipeline.

Drop a talk video (``.mp4`` etc.) into ``data/recordings/<collection>/`` like any
recording. This stage:

  1. extracts its audio track to ``data/recordings/<collection>/<stem>.m4a``
     (stream copy when the codec fits; otherwise re-encoded to AAC),
  2. moves the original video, untouched, to ``data/video/<collection>/``.

From there the ``.m4a`` is an ordinary recording: preprocess -> transcribe -> ...
-> archive. The audio is written under a temporary name and renamed into place
only when complete, so the watcher/preprocess never see a half-written file.

Idempotent: once a video has been moved out of the inbox there is nothing left to
do. If the run dies between steps 1 and 2, the sidecar
``data/video/<collection>/<stem>.extract.json`` lets the next run finish the move
instead of treating the existing ``.m4a`` as a name collision.

Usage:
    python -m pipeline.extract                  # every video in every collection
    python -m pipeline.extract "keynote.mp4"    # one file
    python -m pipeline.extract --dry-run
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import typer

from pipeline.utils import (
    collection_root,
    discover_sources,
    load_config,
    resolve_path,
    setup_logging,
    sha256_file,
)

app = typer.Typer(add_completion=False)

_DEFAULT_VIDEO_EXTS = [".mp4", ".mkv", ".mov", ".webm", ".avi"]


def _ffmpeg_audio(ffmpeg: str, video: Path, out: Path, track: int, reencode: bool) -> subprocess.CompletedProcess:
    codec = ["-c:a", "aac", "-b:a", "160k"] if reencode else ["-c:a", "copy"]
    cmd = [
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(video),
        "-map", f"0:a:{track}",
        "-vn", *codec,
        "-f", "ipod",  # the .m4a muxer; explicit because the temp name has no .m4a suffix
        str(out),
    ]
    return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")


def _extract_one(ffmpeg: str, video: Path, audio: Path, track: int, log) -> tuple[bool, str]:
    """Write ``audio`` from ``video`` atomically. Returns (ok, how-or-error)."""
    tmp = audio.with_name(f".{audio.name}.part")
    try:
        r = _ffmpeg_audio(ffmpeg, video, tmp, track, reencode=False)
        how = "stream copy"
        if r.returncode != 0:
            # e.g. Opus/Vorbis audio (common in .webm/.mkv) can't go in an .m4a as-is.
            log.debug("  stream copy failed (%s); re-encoding to AAC", r.stderr.strip()[-200:])
            r = _ffmpeg_audio(ffmpeg, video, tmp, track, reencode=True)
            how = "re-encoded to AAC"
        if r.returncode != 0:
            err = r.stderr.strip()[-300:]
            if "matches no streams" in err:
                err = f"no audio track #{track} in this file"
            return False, err
        os.replace(tmp, audio)
        return True, how
    finally:
        if tmp.exists():
            tmp.unlink()


@app.command()
def main(
    files: list[str] = typer.Argument(None, help="Specific video files (name or stem)"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be extracted"),
    verbose: bool = typer.Option(False, "--verbose", help="Debug logging"),
) -> None:
    """Extract audio from videos in the inbox and move the videos aside."""
    log = setup_logging("extract", verbose)
    cfg = load_config()

    raw_dir = resolve_path("paths.raw_audio", cfg)
    video_exts = cfg.get("video_extensions", _DEFAULT_VIDEO_EXTS)
    track = int(cfg.get("video", {}).get("audio_track", 0))

    videos = discover_sources(raw_dir, extensions=video_exts, specific_files=files)
    if not videos:
        # Not an error: most runs have only audio.
        log.info("No videos to extract under %s/<collection>/", raw_dir)
        return

    log.info("Found %d video(s) across %d collection(s)",
             len(videos), len({v.parent.name for v in videos}))

    ffmpeg = None
    if not dry_run:
        from pipeline.preprocess import _find_ffmpeg
        try:
            ffmpeg = _find_ffmpeg()
        except FileNotFoundError as e:
            log.error(str(e))
            raise typer.Exit(1)

    failures = 0
    for video in videos:
        collection = video.parent.name
        audio = video.with_suffix(".m4a")
        # resolve_path/collection_root create directories; a dry run must not write.
        video_dir = (Path(cfg["paths"]["video"]) / collection) if dry_run \
            else collection_root(cfg, "paths.video", collection)
        dest = video_dir / video.name
        sidecar = video_dir / f"{video.stem}.extract.json"

        resuming = False
        if sidecar.exists():
            try:
                resuming = json.loads(sidecar.read_text(encoding="utf-8")).get("audio") == audio.name
            except (json.JSONDecodeError, OSError):
                resuming = False

        # Never overwrite a different recording that happens to share the name.
        if audio.exists() and not resuming:
            log.error("  %s/%s: FAILED -- %s already exists; rename the video or the audio",
                      collection, video.name, audio.name)
            failures += 1
            continue
        if dest.exists():
            log.error("  %s/%s: FAILED -- %s already exists; rename the video",
                      collection, video.name, dest)
            failures += 1
            continue

        if dry_run:
            log.info("  %s/%s -> %s (+ move video to %s)", collection, video.name, audio.name, video_dir)
            continue

        if resuming and audio.exists():
            how = "already extracted (resuming)"
        else:
            sha = sha256_file(video)
            ok, how = _extract_one(ffmpeg, video, audio, track, log)
            if not ok:
                log.error("  %s/%s: FAILED -- %s", collection, video.name, how)
                failures += 1
                continue
            sidecar.write_text(json.dumps({
                "video": video.name,
                "video_sha256": sha,
                "audio": audio.name,
                "audio_track": track,
                "method": how,
                "extracted_at": datetime.now().isoformat(),
            }, indent=2), encoding="utf-8")

        shutil.move(str(video), str(dest))
        log.info("  %s/%s -> %s (%s); video moved to %s", collection, video.name, audio.name, how, video_dir)

    if failures:
        log.error("%d video(s) failed", failures)
        raise typer.Exit(1)

    if not dry_run:
        log.info("Audio extraction complete -- %d video(s)", len(videos))


if __name__ == "__main__":
    app()
