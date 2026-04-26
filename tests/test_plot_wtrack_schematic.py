from __future__ import annotations

import numpy as np
import pytest

from v1ca1.helper.plot_wtrack_schematic import (
    get_oval_specs,
    get_w_track_geometry,
    plot_w_track_trajectory,
    sample_path,
    trajectory_points,
    trajectory_segments,
)


def test_w_track_geometry_has_expected_named_points() -> None:
    outline, points, dims = get_w_track_geometry()

    assert len(outline) == 12
    assert set(points) == {
        "left",
        "center",
        "right",
        "bottom_left",
        "bottom_center",
        "bottom_right",
    }
    assert dims["corridor_w"] == 0.7
    assert points["left"][0] < points["center"][0] < points["right"][0]


def test_trajectory_segments_follow_path_order() -> None:
    _outline, points, _dims = get_w_track_geometry()

    path = trajectory_points("left_to_center", points)
    segments = trajectory_segments("left_to_center", points)

    assert len(segments) == 3
    assert segments[0][0] == path[0]
    assert segments[-1][-1] == path[-1]


def test_sample_path_includes_endpoint_and_deduplicates_segment_boundary() -> None:
    centers = sample_path([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)], spacing=1.0)

    assert np.allclose(centers, [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])


def test_sample_path_rejects_nonpositive_spacing() -> None:
    with pytest.raises(ValueError, match="spacing must be positive"):
        sample_path([(0.0, 0.0), (1.0, 0.0)], spacing=0.0)


def test_get_oval_specs_contains_all_regions() -> None:
    _outline, _points, dims = get_w_track_geometry()

    specs = get_oval_specs(dims)

    assert set(specs) == {
        "left_arm",
        "center_arm",
        "right_arm",
        "left_center_connector",
        "center_right_connector",
    }


def test_plot_w_track_trajectory_smoke() -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plot_w_track_trajectory(
        trajectory_name="center_to_left",
        show_basis=True,
        basis_segment_styles=[
            {"edge_color": "black", "fill_color": "none"},
            {"edge_color": "black", "fill_color": "none"},
            {"edge_color": "red", "fill_color": "pink", "fill_alpha": 0.35},
        ],
        show_large_ovals=True,
        oval_regions=["center_arm"],
    )

    assert fig is not None
    assert len(ax.patches) > 1
    plt.close(fig)
