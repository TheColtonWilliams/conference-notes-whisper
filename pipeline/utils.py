"""Shared utilities for the pipeline. Every script imports from here."""

from __future__ import annotations

import pipeline._patch  # noqa: F401  — must be first

import gc
import hashlib
import json
import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from rich.logging import RichHandler

_PROJECT_ROOT: Path | None = None

# Subfolder names that are never real collections: "raw" archives intermediate
# ASR output; "smoke" holds throwaway smoke-test output. Both must be skipped
# wherever collections are iterated so they don't get clustered/indexed/searched.
_RESERVED_DIRS = {"raw", "smoke"}


def project_root() -> Path:
    """Return the project root (the directory containing pipeline/)."""
    global _PROJECT_ROOT
    if _PROJECT_ROOT is None:
        # Walk up from this file (pipeline/utils.py) to find the project root
        _PROJECT_ROOT = Path(__file__).resolve().parent.parent
    return _PROJECT_ROOT


def load_config(override_path: Path | None = None) -> dict:
    """Load config.yaml and resolve all paths relative to project root.

    Also loads .env for environment variables (HF_TOKEN, etc.).
    """
    load_dotenv(project_root() / ".env")

    config_path = override_path or (project_root() / "pipeline" / "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Resolve all paths relative to project root
    root = project_root()
    for key, relpath in cfg.get("paths", {}).items():
        cfg["paths"][key] = str(root / relpath)

    return cfg


def resolve_path(dotted_key: str, cfg: dict) -> Path:
    """Resolve a dotted config key (e.g., 'paths.transcripts') to a Path.

    Creates the directory if it doesn't exist.
    """
    keys = dotted_key.split(".")
    val = cfg
    for k in keys:
        val = val[k]
    p = Path(val)
    if not p.suffix:  # looks like a directory, not a file
        p.mkdir(parents=True, exist_ok=True)
    return p


def ensure_dir(path: Path) -> Path:
    """Create directory (and parents) if it doesn't exist. Returns the path."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def sha256_file(path: Path) -> str:
    """Compute hex SHA-256 of a file, reading in 64KB chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def _model_config_hash(cfg: dict, model_keys: list[str]) -> str:
    """Hash the subset of config relevant to a pipeline stage's models."""
    relevant = {}
    for key in model_keys:
        keys = key.split(".")
        val = cfg
        for k in keys:
            val = val.get(k, {}) if isinstance(val, dict) else None
        relevant[key] = val
    return hashlib.sha256(json.dumps(relevant, sort_keys=True).encode()).hexdigest()[:16]


def is_stale(
    source: Path,
    output: Path,
    meta_path: Path,
    cfg: dict,
    model_keys: list[str],
) -> bool:
    """Check if output needs regeneration.

    Returns True if:
    - output or meta file is missing
    - source sha256 has changed
    - model config has changed
    """
    if not output.exists() or not meta_path.exists():
        return True

    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except (json.JSONDecodeError, OSError):
        return True

    # Check source file hash
    if meta.get("source_sha256") != sha256_file(source):
        return True

    # Check model config hash
    if model_keys and meta.get("model_config_hash") != _model_config_hash(cfg, model_keys):
        return True

    return False


def write_meta(
    meta_path: Path,
    source: Path,
    cfg: dict,
    model_keys: list[str],
    **extra: Any,
) -> None:
    """Write metadata JSON for idempotency tracking."""
    meta = {
        "source": source.name,
        "source_sha256": sha256_file(source),
        "model_config_hash": _model_config_hash(cfg, model_keys) if model_keys else None,
        "generated_at": datetime.now().isoformat(),
        **extra,
    }
    ensure_dir(meta_path.parent)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def read_meta(meta_path: Path) -> dict | None:
    """Load a stage's ``.meta.json`` defensively, or None if missing/unreadable."""
    meta_path = Path(meta_path)
    if not meta_path.exists():
        return None
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def write_yaml_atomic(path: Path, data: Any) -> None:
    """Write YAML atomically (temp file in the same dir + os.replace).

    Used for human-edited files like data/glossary.yaml so a concurrent reader
    never sees a half-written file. Comments are not preserved (pyyaml).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def setup_logging(script_name: str, verbose: bool = False) -> logging.Logger:
    """Configure dual logging: file (always DEBUG) + rich console.

    Returns a logger for the script.
    """
    log_dir = ensure_dir(project_root() / "pipeline" / "logs")
    date_str = datetime.now().strftime("%Y-%m-%d")
    log_file = log_dir / f"{script_name}-{date_str}.log"

    logger = logging.getLogger(f"pipeline.{script_name}")
    logger.setLevel(logging.DEBUG)

    # Avoid duplicate handlers on re-import
    if logger.handlers:
        return logger

    # File handler (always DEBUG)
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(fh)

    # Rich console handler
    ch = RichHandler(rich_tracebacks=True, show_path=False)
    ch.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.addHandler(ch)

    return logger


def gpu_cleanup(*models: Any) -> None:
    """Delete model references and free GPU memory."""
    for m in models:
        del m
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def discover_sources(
    directory: str | Path,
    extensions: list[str] | None = None,
    specific_files: list[str] | None = None,
) -> list[Path]:
    """List source audio files, grouped by collection (immediate subfolder).

    The pipeline is collection-based: audio lives one level deep under
    ``directory`` and the subfolder name *is* the collection. For example,
    ``recordings/bsides-2026/keynote.m4a`` belongs to collection
    ``bsides-2026`` (derivable as ``path.parent.name``).

    Args:
        directory: Inbox root to scan (e.g. ``recordings``).
        extensions: File extensions to include (e.g., ['.m4a', '.mp3']).
        specific_files: If provided, keep only files whose name or stem matches
            one of these (searched across every collection). Absolute/relative
            paths that exist are also accepted directly.

    Returns:
        Sorted list of Path objects (each ``<directory>/<collection>/<file>``).
    """
    directory = Path(directory)
    if extensions is None:
        extensions = [".m4a", ".mp3", ".wav", ".flac", ".ogg"]
    exts = {e.lower() for e in extensions}

    found: list[Path] = []
    if directory.exists():
        for sub in sorted(directory.iterdir()):
            if not sub.is_dir() or sub.name in _RESERVED_DIRS:
                continue
            for p in sorted(sub.iterdir()):
                if p.is_file() and p.suffix.lower() in exts:
                    found.append(p)

    if specific_files:
        wanted_names = {Path(n).name for n in specific_files}
        wanted_stems = {Path(n).stem for n in specific_files}
        filtered = [p for p in found if p.name in wanted_names or p.stem in wanted_stems]
        # Also accept explicit paths that exist on disk
        for n in specific_files:
            pn = Path(n)
            if pn.is_file() and pn not in filtered:
                filtered.append(pn)
        return sorted(filtered)

    return found


def discover_collections(base_dir: str | Path) -> list[str]:
    """Return sorted collection names = immediate subdirectories of ``base_dir``.

    Excludes reserved subfolders (``raw`` intermediate output, ``smoke`` test
    output) so they are never treated as real collections.
    """
    base = Path(base_dir)
    if not base.exists():
        return []
    return sorted(
        p.name for p in base.iterdir()
        if p.is_dir() and p.name not in _RESERVED_DIRS
    )


def collection_root(cfg: dict, dotted_key: str, collection: str) -> Path:
    """Resolve ``paths.<key>/<collection>`` to a Path, creating it.

    Use this everywhere a stage needs the per-collection output directory so a
    single recording's outputs never collide with a same-named recording in a
    different collection.
    """
    base = resolve_path(dotted_key, cfg)
    p = Path(base) / collection
    p.mkdir(parents=True, exist_ok=True)
    return p


def preferred_transcript(col_dir: Path, stem: str) -> Path:
    """Return the transcript to consume for ``stem``: ``.clean.json`` if it
    exists (LLM-corrected), else the raw ``.json``."""
    clean = Path(col_dir) / f"{stem}.clean.json"
    return clean if clean.exists() else Path(col_dir) / f"{stem}.json"


def collection_transcripts(col_dir: Path) -> dict[str, Path]:
    """Map ``stem -> transcript path`` for one collection directory, preferring
    ``.clean.json`` over ``.json``. Skips ``.meta.json`` sidecars.

    Centralizes the "prefer clean" discovery shared by kb.py and index.py.
    """
    col_dir = Path(col_dir)
    stems = set()
    for p in col_dir.glob("*.json"):
        if p.name.endswith(".meta.json"):
            continue
        stems.add(p.stem.replace(".clean", ""))
    return {stem: preferred_transcript(col_dir, stem) for stem in sorted(stems)}


def resolve_device(cfg: dict) -> str:
    """Resolve the compute device from config (``models.device``: auto|cuda|cpu).

    ``auto`` (the default) picks cuda when available, else cpu.
    """
    pref = cfg.get("models", {}).get("device", "auto")
    if pref and pref != "auto":
        return pref
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def verify_hf_token() -> str | None:
    """Check that HF_TOKEN is set and return it. Returns None if not set."""
    import os
    token = os.environ.get("HF_TOKEN")
    if not token:
        return None
    return token


def format_timestamp(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def format_vtt_timestamp(seconds: float) -> str:
    """Format seconds as HH:MM:SS.mmm for WebVTT."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"
