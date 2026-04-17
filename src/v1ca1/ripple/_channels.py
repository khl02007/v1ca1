from __future__ import annotations

"""Session-specific ripple channel assignments."""

RIPPLE_CHANNELS_BY_SESSION: dict[str, dict[str, list[int]]] = {
    "L12": {
        "20240421": [
            16 + 32 * 0 + 128 * 1,
            17 + 32 * 0 + 128 * 1,
            18 + 32 * 1 + 128 * 1,
            19 + 32 * 1 + 128 * 1,
            22 + 32 * 0 + 128 * 2,
            21 + 32 * 0 + 128 * 2,
            23 + 32 * 1 + 128 * 2,
            22 + 32 * 1 + 128 * 2,
            31 + 32 * 2 + 128 * 2,
            30 + 32 * 2 + 128 * 2,
        ],
    },
    "L14": {
        "20240606": [
            0 + 32 * 2 + 128 * 1,
            1 + 32 * 2 + 128 * 1,
            2 + 32 * 2 + 128 * 1,
            9 + 32 * 2 + 128 * 2,
            10 + 32 * 2 + 128 * 2,
            11 + 32 * 2 + 128 * 2,
        ],
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
            29 + 32 * 0 + 128 * 1,
            28 + 32 * 0 + 128 * 1,
            27 + 32 * 0 + 128 * 1,
            31 + 32 * 1 + 128 * 1,
            31 + 32 * 1 + 128 * 2,
            30 + 32 * 1 + 128 * 2,
            29 + 32 * 1 + 128 * 2,
            26 + 32 * 0 + 128 * 2,
        ],
    },
    "L19": {
        "20250930": [
            7 + 32 * 1 + 128 * 2,
            8 + 32 * 1 + 128 * 2,
            9 + 32 * 2 + 128 * 2,
            10 + 32 * 2 + 128 * 2,
            15 + 32 * 3 + 128 * 2,
            16 + 32 * 3 + 128 * 2,
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
