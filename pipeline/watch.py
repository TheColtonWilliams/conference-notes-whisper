"""File watcher: auto-run pipeline on new audio files.

Usage:
    python -m pipeline.watch              # watch for new files
    python -m pipeline.watch --interval 60  # check every 60s
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

import typer
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from pipeline.utils import load_config, project_root, resolve_path, setup_logging

app = typer.Typer(add_completion=False)


class AudioHandler(FileSystemEventHandler):
    """Handle new audio files by running the pipeline."""

    def __init__(self, cfg: dict, log, cooldown: int = 30):
        self.cfg = cfg
        self.log = log
        self.cooldown = cooldown
        self.extensions = set(cfg.get("audio_extensions", [".m4a", ".mp3", ".wav"]))
        self._pending: dict[str, float] = {}

    def on_created(self, event):
        if event.is_directory:
            return
        path = Path(event.src_path)
        if path.suffix.lower() in self.extensions:
            self._pending[str(path)] = time.time()
            self.log.info("Detected new file: %s (waiting %ds for stability)", path.name, self.cooldown)

    def on_modified(self, event):
        if event.is_directory:
            return
        path = Path(event.src_path)
        if str(path) in self._pending:
            self._pending[str(path)] = time.time()

    def process_pending(self):
        """Process files that have been stable for cooldown seconds."""
        now = time.time()
        ready = [
            p for p, t in self._pending.items()
            if now - t >= self.cooldown
        ]
        for path_str in ready:
            del self._pending[path_str]
            path = Path(path_str)
            if path.exists():
                self.log.info("Processing stable file: %s", path.name)
                self._run_pipeline(path)

    def _run_pipeline(self, path: Path):
        """Run the full pipeline on a single file.

        The human-in-the-loop `review` stage is intentionally skipped here: the
        watcher is unattended/per-file, so it can't block on human edits. Run
        `python -m pipeline.review build/apply` (or `run.ps1`) for review.
        """
        venv_python = str(project_root() / ".venv" / "Scripts" / "python.exe")
        stages = [
            ("preprocess", [venv_python, "-m", "pipeline.preprocess", path.name]),
            ("transcribe", [venv_python, "-m", "pipeline.transcribe", path.name]),
            ("correct", [venv_python, "-m", "pipeline.correct", path.name]),
            ("kb", [venv_python, "-m", "pipeline.kb", "extract", path.name]),
            ("index", [venv_python, "-m", "pipeline.index"]),
            ("archive", [venv_python, "-m", "pipeline.archive", path.name]),
        ]

        for stage_name, cmd in stages:
            self.log.info("  Running %s...", stage_name)
            result = subprocess.run(
                cmd,
                cwd=str(project_root()),
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                self.log.error("  %s failed: %s", stage_name, result.stderr[-500:])
                break
            self.log.info("  %s complete", stage_name)


@app.command()
def main(
    interval: int = typer.Option(5, "--interval", help="Poll interval in seconds"),
    cooldown: int = typer.Option(30, "--cooldown", help="Seconds file must be stable before processing"),
    verbose: bool = typer.Option(False, "--verbose", help="Debug logging"),
) -> None:
    """Watch for new audio files and auto-run the pipeline."""
    log = setup_logging("watch", verbose)
    cfg = load_config()

    watch_dir = resolve_path("paths.raw_audio", cfg)
    log.info("Watching %s/<collection>/ for new audio files...", watch_dir)
    log.info("Extensions: %s", cfg.get("audio_extensions", []))
    log.info("Cooldown: %ds, Poll: %ds", cooldown, interval)

    handler = AudioHandler(cfg, log, cooldown=cooldown)
    observer = Observer()
    # recursive: audio now lives one level deep under recordings/<collection>/
    observer.schedule(handler, str(watch_dir), recursive=True)
    observer.start()

    try:
        while True:
            time.sleep(interval)
            handler.process_pending()
    except KeyboardInterrupt:
        log.info("Stopping watcher...")
        observer.stop()

    observer.join()
    log.info("Watcher stopped")


if __name__ == "__main__":
    app()
