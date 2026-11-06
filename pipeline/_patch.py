"""Monkey-patches for third-party library bugs. Import early.

Patches applied:
- SpeechBrain LazyModule: Windows backslash path check (GH speechbrain/speechbrain#xxxx)
  The guard that prevents inspect.py from triggering lazy imports checks for
  "/inspect.py" but Windows uses backslashes. We patch ensure_module to also
  check "\\inspect.py".
- FFmpeg PATH: Adds winget FFmpeg install location to PATH if not already present.
"""

from __future__ import annotations

import glob
import os
import sys

# Force SpeechBrain to use COPY strategy instead of symlinks on Windows
os.environ["SPEECHBRAIN_LOCAL_STRATEGY"] = "COPY"


def _ensure_ffmpeg_on_path() -> None:
    """Add ffmpeg to PATH if installed via winget but not on PATH."""
    if sys.platform != "win32":
        return

    import shutil

    if shutil.which("ffmpeg"):
        return

    # Search known winget install locations
    home = os.path.expanduser("~")
    patterns = [
        os.path.join(home, "AppData", "Local", "Microsoft", "WinGet", "Packages",
                     "Gyan.FFmpeg*", "ffmpeg-*-full_build", "bin"),
    ]
    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            os.environ["PATH"] = matches[0] + os.pathsep + os.environ.get("PATH", "")
            return


def _apply_speechbrain_lazy_module_patch() -> None:
    """Fix SpeechBrain LazyModule.ensure_module for Windows paths."""
    if sys.platform != "win32":
        return

    try:
        from speechbrain.utils.importutils import LazyModule
    except ImportError:
        return

    import inspect
    import importlib
    import warnings

    _original_ensure_module = LazyModule.ensure_module

    def _patched_ensure_module(self, stacklevel=0):
        importer_frame = None
        try:
            importer_frame = inspect.getframeinfo(
                sys._getframe(stacklevel + 1)
            )
        except AttributeError:
            warnings.warn(
                "Failed to inspect frame to check if we should ignore "
                "importing a module lazily."
            )

        if importer_frame is not None and (
            importer_frame.filename.endswith("/inspect.py")
            or importer_frame.filename.endswith("\\inspect.py")
        ):
            raise AttributeError()

        if self.lazy_module is None:
            try:
                if self.package is None:
                    self.lazy_module = importlib.import_module(self.target)
                else:
                    self.lazy_module = importlib.import_module(
                        f".{self.target}", self.package
                    )
            except Exception as e:
                raise ImportError(
                    f"Lazy import of {repr(self)} failed"
                ) from e

        return self.lazy_module

    LazyModule.ensure_module = _patched_ensure_module


_ensure_ffmpeg_on_path()
_apply_speechbrain_lazy_module_patch()
