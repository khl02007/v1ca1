from __future__ import annotations

"""Shared pytest bootstrap for environment-dependent test configuration."""

import os
import tempfile
from pathlib import Path


def pytest_configure() -> None:
    """Point Numba caching at a writable temp directory for pynapple imports."""
    cache_dir = Path(tempfile.gettempdir()) / "v1ca1-numba-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("NUMBA_CACHE_DIR", str(cache_dir))
