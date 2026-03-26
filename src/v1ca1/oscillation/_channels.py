from __future__ import annotations

"""Session-specific theta channel assignments."""

THETA_CHANNEL_BY_SESSION: dict[str, dict[str, int]] = {
    "L15": {
        "20241121": 128 + 63,
    },
}


def get_session_theta_channel(animal_name: str, date: str) -> int:
    """Return the configured theta channel for one animal/date session."""
    channels_by_date = THETA_CHANNEL_BY_SESSION.get(str(animal_name))
    if channels_by_date is None:
        raise KeyError(f"No theta channel mapping configured for animal {animal_name!r}.")

    channel = channels_by_date.get(str(date))
    if channel is None:
        raise KeyError(
            "No theta channel mapping configured for session "
            f"{animal_name!r} / {date!r}."
        )
    return int(channel)
