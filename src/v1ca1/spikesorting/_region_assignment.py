from __future__ import annotations

"""Session-specific curated-unit region assignment rules."""

from typing import Any

EXPECTED_PROBE_COUNT = 4
EXPECTED_SHANK_COUNT = 4

SESSION_REGION_ASSIGNMENTS: dict[str, dict[str, list[dict[str, Any]]]] = {
    "L12": {
        "20240421": [
            {
                "probe_idx": 0,
                "shank_idx": 0,
                "mode": "all",
                "region": "v1",
            },
            {
                "probe_idx": 0,
                "shank_idx": 1,
                "mode": "all",
                "region": "v1",
            },
            {
                "probe_idx": 0,
                "shank_idx": 2,
                "mode": "all",
                "region": "v1",
            },
            {
                "probe_idx": 0,
                "shank_idx": 3,
                "mode": "all",
                "region": "v1",
            },
            {
                "probe_idx": 1,
                "shank_idx": 0,
                "mode": "threshold_by_extremum_depth",
                "reference_depth": -286.0,
                "below_region": "ca1",
                "above_or_equal_region": "s1",
            },
            {
                "probe_idx": 1,
                "shank_idx": 1,
                "mode": "threshold_by_extremum_depth",
                "reference_depth": -286.0,
                "below_region": "ca1",
                "above_or_equal_region": "s1",
            },
            {
                "probe_idx": 1,
                "shank_idx": 2,
                "mode": "threshold_by_extremum_depth",
                "reference_depth": -390.0,
                "below_region": "ca1",
                "above_or_equal_region": "s1",
            },
            {
                "probe_idx": 1,
                "shank_idx": 3,
                "mode": "threshold_by_extremum_depth",
                "reference_depth": -390.0,
                "below_region": "ca1",
                "above_or_equal_region": "s1",
            },
            {
                "probe_idx": 2,
                "shank_idx": 0,
                "mode": "threshold_by_extremum_depth",
                "reference_depth": -286.0,
                "below_region": "ca1",
                "above_or_equal_region": "s1",
            },
            {
                "probe_idx": 2,
                "shank_idx": 1,
                "mode": "threshold_by_extremum_depth",
                "reference_depth": -286.0,
                "below_region": "ca1",
                "above_or_equal_region": "s1",
            },
            {
                "probe_idx": 2,
                "shank_idx": 2,
                "mode": "threshold_by_extremum_depth",
                "reference_depth": -390.0,
                "below_region": "ca1",
                "above_or_equal_region": "s1",
            },
            {
                "probe_idx": 2,
                "shank_idx": 3,
                "mode": "threshold_by_extremum_depth",
                "reference_depth": -390.0,
                "below_region": "ca1",
                "above_or_equal_region": "s1",
            },
            {
                "probe_idx": 3,
                "shank_idx": 0,
                "mode": "all",
                "region": "v1",
            },
            {
                "probe_idx": 3,
                "shank_idx": 1,
                "mode": "all",
                "region": "v1",
            },
            {
                "probe_idx": 3,
                "shank_idx": 2,
                "mode": "all",
                "region": "v1",
            },
            {
                "probe_idx": 3,
                "shank_idx": 3,
                "mode": "all",
                "region": "v1",
            },
        ],
    },
    "L14": {
        "20240611": [
            {
                "probe_idx": 0,
                "shank_idx": 0,
                "mode": "all",
                "region": "v1",
            },
            {
                "probe_idx": 0,
                "shank_idx": 1,
                "mode": "all",
                "region": "v1",
            },
            {
                "probe_idx": 0,
                "shank_idx": 2,
                "mode": "all",
                "region": "v1",
            },
            {
                "probe_idx": 0,
                "shank_idx": 3,
                "mode": "all",
                "region": "v1",
            },
            {
                "probe_idx": 1,
                "shank_idx": 0,
                "mode": "all",
                "region": "ca1",
            },
            {
                "probe_idx": 1,
                "shank_idx": 1,
                "mode": "all",
                "region": "ca1",
            },
            {
                "probe_idx": 1,
                "shank_idx": 2,
                "mode": "all",
                "region": "ca1",
            },
            {
                "probe_idx": 1,
                "shank_idx": 3,
                "mode": "all",
                "region": "ca1",
            },
            {
                "probe_idx": 2,
                "shank_idx": 0,
                "mode": "all",
                "region": "ca1",
            },
            {
                "probe_idx": 2,
                "shank_idx": 1,
                "mode": "all",
                "region": "ca1",
            },
            {
                "probe_idx": 2,
                "shank_idx": 2,
                "mode": "all",
                "region": "ca1",
            },
            {
                "probe_idx": 2,
                "shank_idx": 3,
                "mode": "all",
                "region": "ca1",
            },
            {
                "probe_idx": 3,
                "shank_idx": 0,
                "mode": "all",
                "region": "v1",
            },
            {
                "probe_idx": 3,
                "shank_idx": 1,
                "mode": "all",
                "region": "v1",
            },
            {
                "probe_idx": 3,
                "shank_idx": 2,
                "mode": "all",
                "region": "v1",
            },
            {
                "probe_idx": 3,
                "shank_idx": 3,
                "mode": "all",
                "region": "v1",
            },
        ],
    },
    "L15": {
        "20241121": [
            {
                "probe_idx": 0,
                "shank_idx": 0,
                "mode": "all",
                "region": "v1",
            },
            {
                "probe_idx": 0,
                "shank_idx": 1,
                "mode": "all",
                "region": "v1",
            },
            {
                "probe_idx": 0,
                "shank_idx": 2,
                "mode": "all",
                "region": "v1",
            },
            {
                "probe_idx": 0,
                "shank_idx": 3,
                "mode": "all",
                "region": "v1",
            },
            {
                "probe_idx": 1,
                "shank_idx": 0,
                "mode": "all",
                "region": "ca1",
            },
            {
                "probe_idx": 1,
                "shank_idx": 1,
                "mode": "all",
                "region": "ca1",
            },
            {
                "probe_idx": 1,
                "shank_idx": 2,
                "mode": "all",
                "region": "ca1",
            },
            {
                "probe_idx": 1,
                "shank_idx": 3,
                "mode": "all",
                "region": "ca1",
            },
            {
                "probe_idx": 2,
                "shank_idx": 0,
                "mode": "all",
                "region": "ca1",
            },
            {
                "probe_idx": 2,
                "shank_idx": 1,
                "mode": "all",
                "region": "ca1",
            },
            {
                "probe_idx": 2,
                "shank_idx": 2,
                "mode": "all",
                "region": "ca1",
            },
            {
                "probe_idx": 2,
                "shank_idx": 3,
                "mode": "all",
                "region": "ca1",
            },
            {
                "probe_idx": 3,
                "shank_idx": 0,
                "mode": "all",
                "region": "v1",
            },
            {
                "probe_idx": 3,
                "shank_idx": 1,
                "mode": "all",
                "region": "v1",
            },
            {
                "probe_idx": 3,
                "shank_idx": 2,
                "mode": "all",
                "region": "v1",
            },
            {
                "probe_idx": 3,
                "shank_idx": 3,
                "mode": "all",
                "region": "v1",
            },
        ],
    },
    "L16": {
        "20250302": [
            {
                "probe_idx": 0,
                "shank_idx": 0,
                "mode": "all",
                "region": "v1",
            },
            {
                "probe_idx": 0,
                "shank_idx": 1,
                "mode": "all",
                "region": "v1",
            },
            {
                "probe_idx": 0,
                "shank_idx": 2,
                "mode": "all",
                "region": "v1",
            },
            {
                "probe_idx": 0,
                "shank_idx": 3,
                "mode": "all",
                "region": "v1",
            },
            {
                "probe_idx": 1,
                "shank_idx": 0,
                "mode": "threshold_by_extremum_depth",
                "reference_depth": -620.0,
                "below_region": "ca1",
                "above_or_equal_region": "s1",
            },
            {
                "probe_idx": 1,
                "shank_idx": 1,
                "mode": "threshold_by_extremum_depth",
                "reference_depth": -620.0,
                "below_region": "ca1",
                "above_or_equal_region": "s1",
            },
            {
                "probe_idx": 1,
                "shank_idx": 2,
                "mode": "threshold_by_extremum_depth",
                "reference_depth": -620.0,
                "below_region": "ca1",
                "above_or_equal_region": "s1",
            },
            {
                "probe_idx": 1,
                "shank_idx": 3,
                "mode": "threshold_by_extremum_depth",
                "reference_depth": -620.0,
                "below_region": "ca1",
                "above_or_equal_region": "s1",
            },
            {
                "probe_idx": 2,
                "shank_idx": 0,
                "mode": "threshold_by_extremum_depth",
                "reference_depth": -620.0,
                "below_region": "ca1",
                "above_or_equal_region": "s1",
            },
            {
                "probe_idx": 2,
                "shank_idx": 1,
                "mode": "threshold_by_extremum_depth",
                "reference_depth": -620.0,
                "below_region": "ca1",
                "above_or_equal_region": "s1",
            },
            {
                "probe_idx": 2,
                "shank_idx": 2,
                "mode": "threshold_by_extremum_depth",
                "reference_depth": -620.0,
                "below_region": "ca1",
                "above_or_equal_region": "s1",
            },
            {
                "probe_idx": 2,
                "shank_idx": 3,
                "mode": "threshold_by_extremum_depth",
                "reference_depth": -620.0,
                "below_region": "ca1",
                "above_or_equal_region": "s1",
            },
            {
                "probe_idx": 3,
                "shank_idx": 0,
                "mode": "all",
                "region": "v1",
            },
            {
                "probe_idx": 3,
                "shank_idx": 1,
                "mode": "all",
                "region": "v1",
            },
            {
                "probe_idx": 3,
                "shank_idx": 2,
                "mode": "all",
                "region": "v1",
            },
            {
                "probe_idx": 3,
                "shank_idx": 3,
                "mode": "all",
                "region": "v1",
            },
        ],
    },
}


def get_expected_probe_shank_pairs() -> list[tuple[int, int]]:
    """Return the full set of expected probe/shank pairs for one session."""
    return [
        (probe_idx, shank_idx)
        for probe_idx in range(EXPECTED_PROBE_COUNT)
        for shank_idx in range(EXPECTED_SHANK_COUNT)
    ]


def validate_session_region_assignment_rules(
    rules: list[dict[str, Any]],
) -> None:
    """Require one valid region-assignment rule for every probe/shank pair."""
    expected_pairs = set(get_expected_probe_shank_pairs())
    seen_pairs: set[tuple[int, int]] = set()

    for rule in rules:
        probe_idx = int(rule["probe_idx"])
        shank_idx = int(rule["shank_idx"])
        pair = (probe_idx, shank_idx)
        if pair not in expected_pairs:
            raise ValueError(
                "Region assignment rule references an unexpected probe/shank pair: "
                f"{pair!r}."
            )
        if pair in seen_pairs:
            raise ValueError(
                f"Duplicate region assignment rule configured for probe {probe_idx} "
                f"shank {shank_idx}."
            )
        seen_pairs.add(pair)

        mode = str(rule["mode"])
        if mode == "all":
            if "region" not in rule:
                raise ValueError(
                    "Region assignment rule with mode 'all' must define 'region'."
                )
        elif mode == "threshold_by_extremum_depth":
            required_keys = ("reference_depth", "below_region", "above_or_equal_region")
            missing_keys = [key for key in required_keys if key not in rule]
            if missing_keys:
                raise ValueError(
                    "Region assignment rule with mode 'threshold_by_extremum_depth' "
                    f"is missing required keys {missing_keys!r}."
                )
        else:
            raise ValueError(
                "Unsupported region assignment mode "
                f"{mode!r} for probe {probe_idx} shank {shank_idx}."
            )

    missing_pairs = sorted(expected_pairs - seen_pairs)
    if missing_pairs:
        raise ValueError(
            "Region assignment rules must cover every expected probe/shank pair. "
            f"Missing pairs: {missing_pairs!r}."
        )


def get_session_region_assignment_rules(
    animal_name: str,
    date: str,
) -> list[dict[str, Any]]:
    """Return validated region-assignment rules for one animal/date session."""
    rules_by_date = SESSION_REGION_ASSIGNMENTS.get(str(animal_name))
    if rules_by_date is None:
        raise KeyError(
            f"No region assignment rules configured for animal {animal_name!r}."
        )

    rules = rules_by_date.get(str(date))
    if rules is None:
        raise KeyError(
            "No region assignment rules configured for session "
            f"{animal_name!r} / {date!r}."
        )

    rules_copy = [dict(rule) for rule in rules]
    validate_session_region_assignment_rules(rules_copy)
    return rules_copy


def index_rules_by_probe_shank(
    rules: list[dict[str, Any]],
) -> dict[tuple[int, int], dict[str, Any]]:
    """Return one validated rule keyed by `(probe_idx, shank_idx)`."""
    validate_session_region_assignment_rules(rules)
    return {
        (int(rule["probe_idx"]), int(rule["shank_idx"])): dict(rule)
        for rule in rules
    }


def get_region_names_from_rules(rules: list[dict[str, Any]]) -> list[str]:
    """Return all configured region names referenced by one session's rules."""
    validate_session_region_assignment_rules(rules)
    region_names: set[str] = set()
    for rule in rules:
        mode = str(rule["mode"])
        if mode == "all":
            region_names.add(str(rule["region"]))
        else:
            region_names.add(str(rule["below_region"]))
            region_names.add(str(rule["above_or_equal_region"]))
    return sorted(region_names)
