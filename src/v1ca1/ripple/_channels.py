from __future__ import annotations

"""Session-specific ripple channel assignments."""

RIPPLE_CHANNELS_BY_SESSION: dict[str, dict[str, list[int]]] = {
    "L14": {
        "20240611": [
            2 + 32 * 2 + 128 * 1,
            3 + 32 * 2 + 128 * 1,
            4 + 32 * 2 + 128 * 1,
            15 + 32 * 2 + 128 * 2,
            16 + 32 * 2 + 128 * 2,
            17 + 32 * 2 + 128 * 2,
        ],
    },
    "L15": {
        "20241121": [
            29 + 128,
            28 + 128,
            27 + 128,
            63 + 128,
            63 + 128 * 2,
            62 + 128 * 2,
            61 + 128 * 2,
            26 + 128 * 2,
        ],
    },
}


def get_session_ripple_channels(animal_name: str, date: str) -> list[int]:
    """Return the configured ripple channels for one animal/date session."""
    channels_by_date = RIPPLE_CHANNELS_BY_SESSION.get(str(animal_name))
    if channels_by_date is None:
        raise KeyError(f"No ripple channel mapping configured for animal {animal_name!r}.")

    channels = channels_by_date.get(str(date))
    if channels is None:
        raise KeyError(
            "No ripple channel mapping configured for session "
            f"{animal_name!r} / {date!r}."
        )
    return list(channels)
