"""Apex RAG retrieval engine."""

from __future__ import annotations

import os
from pathlib import Path

# Keep Matplotlib's cache inside the project instead of the user profile.
_MPL_CACHE = Path(".cache") / "matplotlib"
_MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE))
