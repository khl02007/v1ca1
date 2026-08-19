"""Share Figure 3 canvas and dark-rate cache primitives."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


DEFAULT_FIGURE_WIDTH_MM = 165.0
DEFAULT_FIGURE_HEIGHT_MM = 110.0
DARK_MOVEMENT_FR_CACHE_VERSION = 1
DARK_MOVEMENT_FR_CACHE_COLUMNS = ("unit", "dark_firing_rate_hz")


def _dark_movement_firing_rate_metadata_path(cache_path: Path) -> Path:
    """Return the JSON sidecar path for one dark movement-rate cache."""
    return cache_path.with_suffix(".json")


def save_dark_movement_firing_rate_cache(
    cache_path: Path,
    table: Any,
    metadata: Mapping[str, Any],
) -> None:
    """Write one dark movement-rate cache and its metadata sidecar."""
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_table = table.loc[:, list(DARK_MOVEMENT_FR_CACHE_COLUMNS)].copy()
    cache_table.to_parquet(cache_path, index=False)
    _dark_movement_firing_rate_metadata_path(cache_path).write_text(
        json.dumps(dict(metadata), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_dark_movement_firing_rate_cache(
    cache_path: Path,
    expected_metadata: Mapping[str, Any],
) -> Any | None:
    """Return cached dark movement rates when their metadata matches."""
    import pandas as pd

    cache_path = Path(cache_path)
    metadata_path = _dark_movement_firing_rate_metadata_path(cache_path)
    if not cache_path.exists() or not metadata_path.exists():
        return None

    try:
        cached_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if cached_metadata != dict(expected_metadata):
            print(f"Ignoring stale dark movement firing-rate cache at {cache_path}.")
            return None

        table = pd.read_parquet(cache_path)
        missing_columns = [
            column
            for column in DARK_MOVEMENT_FR_CACHE_COLUMNS
            if column not in table.columns
        ]
        if missing_columns:
            print(
                "Ignoring invalid dark movement firing-rate cache at "
                f"{cache_path}: missing columns {missing_columns!r}."
            )
            return None
        return table.loc[:, list(DARK_MOVEMENT_FR_CACHE_COLUMNS)].copy()
    except Exception as exc:
        print(
            "Ignoring unreadable dark movement firing-rate cache at "
            f"{cache_path}: {exc}"
        )
        return None
