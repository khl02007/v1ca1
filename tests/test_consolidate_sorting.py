from __future__ import annotations

import pytest

from v1ca1.spikesorting import _region_assignment
from v1ca1.spikesorting.consolidate_sorting import (
    assign_regions_to_depths,
    build_curated_unit_groups,
    compute_curated_unit_extremum_depths,
    get_shank_channel_depths_from_arrays,
)


def _make_complete_rules() -> list[dict[str, object]]:
    """Return one valid full-session region-rule configuration."""
    return [
        {
            "probe_idx": probe_idx,
            "shank_idx": shank_idx,
            "mode": "all",
            "region": "ca1",
        }
        for probe_idx in range(_region_assignment.EXPECTED_PROBE_COUNT)
        for shank_idx in range(_region_assignment.EXPECTED_SHANK_COUNT)
    ]


def test_validate_session_region_assignment_rules_requires_all_pairs() -> None:
    rules = _make_complete_rules()[:-1]

    with pytest.raises(ValueError, match="Missing pairs"):
        _region_assignment.validate_session_region_assignment_rules(rules)


def test_get_session_region_assignment_rules_returns_valid_rules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rules = _make_complete_rules()
    monkeypatch.setattr(
        _region_assignment,
        "SESSION_REGION_ASSIGNMENTS",
        {"L16": {"20250302": rules}},
    )

    loaded_rules = _region_assignment.get_session_region_assignment_rules(
        animal_name="L16",
        date="20250302",
    )

    assert loaded_rules == rules


def test_build_curated_unit_groups_mirrors_merge_then_filter_behavior() -> None:
    curation_payload = {
        "labelsByUnit": {
            "1": ["noise"],
            "2": [],
            "3": [],
            "4": ["reject"],
        },
        "mergeGroups": [[2, 4]],
    }

    curated_groups = build_curated_unit_groups(
        original_unit_ids=[1, 2, 3, 4],
        curation_payload=curation_payload,
    )

    assert curated_groups == [[3], [2, 4]]


def test_compute_curated_unit_extremum_depths_uses_spike_count_weights() -> None:
    extremum_depths = compute_curated_unit_extremum_depths(
        curated_unit_groups=[[10], [11, 12]],
        original_unit_extremum_depths={10: -200.0, 11: -100.0, 12: -400.0},
        original_unit_spike_counts={10: 5, 11: 1, 12: 3},
    )

    assert extremum_depths[0] == pytest.approx(-200.0)
    assert extremum_depths[1] == pytest.approx((-100.0 * 1 + -400.0 * 3) / 4.0)


def test_assign_regions_to_depths_threshold_rule() -> None:
    assigned_regions = assign_regions_to_depths(
        extremum_depths=[-500.0, -200.0, 0.0],
        rule={
            "probe_idx": 0,
            "shank_idx": 0,
            "mode": "threshold_by_extremum_depth",
            "reference_depth": -250.0,
            "below_region": "ca1",
            "above_or_equal_region": "v1",
        },
    )

    assert assigned_regions == ["ca1", "v1", "v1"]


def test_get_shank_channel_depths_from_arrays_prefers_rel_y() -> None:
    channel_depths = get_shank_channel_depths_from_arrays(
        electrode_ids=[0, 1, 2, 3],
        probe_group_labels=["0", "0", "0", "0"],
        probe_shanks=[0, 0, 0, 0],
        rel_y_values=[0.0, -40.0, -80.0, -120.0],
        y_values=[0.0, 0.0, 0.0, 0.0],
        probe_idx=0,
        shank_idx=0,
    )

    assert channel_depths == {
        0: pytest.approx(0.0),
        1: pytest.approx(-40.0),
        2: pytest.approx(-80.0),
        3: pytest.approx(-120.0),
    }


def test_get_shank_channel_depths_from_arrays_falls_back_to_y() -> None:
    channel_depths = get_shank_channel_depths_from_arrays(
        electrode_ids=[32, 33, 34, 35],
        probe_group_labels=["0", "0", "0", "0"],
        probe_shanks=[1, 1, 1, 1],
        rel_y_values=[0.0, 0.0, 0.0, 0.0],
        y_values=[10.0, 20.0, 30.0, 40.0],
        probe_idx=0,
        shank_idx=1,
    )

    assert channel_depths == {
        32: pytest.approx(10.0),
        33: pytest.approx(20.0),
        34: pytest.approx(30.0),
        35: pytest.approx(40.0),
    }
