from __future__ import annotations

import matplotlib
import pandas as pd
from matplotlib.axes import Axes

from v1ca1.spikesorting.plot_extremum_channel_depths import (
    build_premerge_depth_rows,
    plot_premerge_extremum_depth_distribution,
)


matplotlib.use("Agg")


def test_build_premerge_depth_rows_keeps_merge_components_separate() -> None:
    rows = build_premerge_depth_rows(
        animal_name="L15",
        date="20241121",
        region="ca1",
        probe_idx=1,
        shank_idx=0,
        original_unit_ids=[1, 2, 3, 4],
        extremum_channel_ids={1: 128, 2: 129, 3: 130, 4: 131},
        channel_depths={128: 0.0, 129: -26.0, 130: -52.0, 131: -78.0},
        curation_payload={
            "labelsByUnit": {
                "1": ["noise"],
                "2": [],
                "3": [],
                "4": ["reject"],
            },
            "mergeGroups": [[2, 4]],
        },
        rule={"mode": "all", "region": "ca1"},
    )

    assert [row["original_unit_id"] for row in rows] == [3, 2, 4]
    assert [row["extremum_depth_um"] for row in rows] == [-52.0, -26.0, -78.0]
    assert [row["is_merge_component"] for row in rows] == [False, True, True]
    assert [row["curation_group_size"] for row in rows] == [1, 2, 2]


def test_build_premerge_depth_rows_applies_region_threshold() -> None:
    rows = build_premerge_depth_rows(
        animal_name="L15",
        date="20241121",
        region="ca1",
        probe_idx=1,
        shank_idx=0,
        original_unit_ids=[1, 2],
        extremum_channel_ids={1: 128, 2: 129},
        channel_depths={128: -500.0, 129: -100.0},
        curation_payload={"labelsByUnit": {}, "mergeGroups": []},
        rule={
            "mode": "threshold_by_extremum_depth",
            "reference_depth": -250.0,
            "below_region": "ca1",
            "above_or_equal_region": "s1",
        },
    )

    assert [row["original_unit_id"] for row in rows] == [1]


def test_plot_premerge_extremum_depth_distribution_uses_raw_counts_per_shank(
    tmp_path,
    monkeypatch,
) -> None:
    histogram_calls = []
    original_hist = Axes.hist

    def capture_hist(axis, values, *args, **kwargs):
        histogram_calls.append((axis, values, kwargs))
        return original_hist(axis, values, *args, **kwargs)

    monkeypatch.setattr(Axes, "hist", capture_hist)
    table = pd.DataFrame(
        {
            "animal_name": ["L15"] * 8,
            "date": ["20241121"] * 8,
            "region": ["ca1"] * 8,
            "probe_idx": [1, 1, 1, 1, 2, 2, 2, 2],
            "shank_idx": [0, 0, 1, 1, 0, 0, 1, 1],
            "extremum_depth_um": [
                -26.0,
                -52.0,
                -52.0,
                -78.0,
                -26.0,
                -78.0,
                -52.0,
                -104.0,
            ],
        }
    )
    output_path = tmp_path / "depths.png"

    returned_path = plot_premerge_extremum_depth_distribution(
        table,
        output_path,
        dpi=100,
    )

    assert returned_path == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0
    assert len(histogram_calls) == 4
    assert len({id(axis) for axis, _values, _kwargs in histogram_calls}) == 4
    assert all("weights" not in kwargs for _axis, _values, kwargs in histogram_calls)
    positions = [axis.get_position() for axis, _values, _kwargs in histogram_calls]
    assert positions[0].y0 == positions[1].y0
    assert positions[2].y0 == positions[3].y0
    assert positions[0].y0 > positions[2].y0
