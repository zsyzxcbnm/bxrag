"""Apex RAG retrieval engine."""

from __future__ import annotations

import os
from pathlib import Path

# Keep Matplotlib's cache inside the project instead of the user profile,
# while still allowing an explicit MPLCONFIGDIR override for read-only checkouts.
_MPL_CACHE = Path(os.environ.get("MPLCONFIGDIR", Path(".cache") / "matplotlib"))
_MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE))
