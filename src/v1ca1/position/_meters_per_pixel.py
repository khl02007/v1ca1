from __future__ import annotations

"""Session-specific position meters-per-pixel assignments."""

METERS_PER_PIXEL_BY_SESSION: dict[str, dict[str, float]] = {
    "L14": {
        "20240611": 0.00189,
    },
    "L15": {
        "20241121": 0.00189,
    },
}


def get_session_meters_per_pixel(animal_name: str, date: str) -> float:
    """Return the configured meters-per-pixel value for one animal/date session."""
    scales_by_date = METERS_PER_PIXEL_BY_SESSION.get(str(animal_name))
    if scales_by_date is None:
        raise KeyError(
            f"No meters-per-pixel mapping configured for animal {animal_name!r}."
        )

    meters_per_pixel = scales_by_date.get(str(date))
    if meters_per_pixel is None:
        raise KeyError(
            "No meters-per-pixel mapping configured for session "
            f"{animal_name!r} / {date!r}."
        )
    return float(meters_per_pixel)
