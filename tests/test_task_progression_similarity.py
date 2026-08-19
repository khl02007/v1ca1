"""Focused tests for shared segmented tuning-shape helpers."""

from __future__ import annotations

import numpy as np
import pytest

from v1ca1.helper.wtrack import get_wtrack_segment_edges
from v1ca1.task_progression._session import build_task_progression_bins
from v1ca1.task_progression.similarity import (
    compute_segmented_shape_overlap,
    make_segment_masks,
)


@pytest.mark.parametrize(
    ("animal_name", "expected_indices"),
    (
        (
            "L12",
            (range(0, 18), range(18, 26), range(26, 44)),
        ),
        (
            "L14",
            (range(0, 17), range(17, 24), range(24, 41)),
        ),
        (
            "L19",
            (range(0, 17), range(17, 24), range(24, 41)),
        ),
    ),
)
def test_make_segment_masks_uses_wtrack_geometry(
    animal_name: str,
    expected_indices: tuple[range, range, range],
) -> None:
    """Physical W-track boundaries assign every bin center exactly once."""
    bin_edges = build_task_progression_bins(animal_name)
    progression = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    masks = make_segment_masks(
        progression,
        get_wtrack_segment_edges(animal_name),
    )

    assert len(masks) == 3
    assert [np.flatnonzero(mask).tolist() for mask in masks] == [
        list(indices) for indices in expected_indices
    ]
    assert np.all(np.sum(np.vstack(masks), axis=0) == 1)


def test_make_segment_masks_assigns_boundary_to_following_segment() -> None:
    """Interior boundaries are left-closed and right-open."""
    edges = get_wtrack_segment_edges("L14")
    progression = np.asarray(
        [
            0.0,
            np.nextafter(edges[1], 0.0),
            edges[1],
            0.5,
            edges[2],
            1.0,
        ]
    )

    masks = make_segment_masks(progression, edges)

    assert [np.flatnonzero(mask).tolist() for mask in masks] == [
        [0, 1],
        [2, 3],
        [4, 5],
    ]


def test_segmented_overlap_scores_one_silent_curve_as_zero() -> None:
    """A segment active in only one curve is maximally dissimilar."""
    progression = np.asarray([0.1, 0.2, 0.4, 0.5, 0.8, 0.9])
    edges = np.asarray([0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0])

    result = compute_segmented_shape_overlap(
        np.asarray([1.0, 3.0, 2.0, 4.0, 5.0, 5.0]),
        np.asarray([2.0, 6.0, 0.0, 0.0, 10.0, 10.0]),
        progression,
        edges,
    )

    assert result["scores"] == pytest.approx([1.0, 0.0, 1.0])
    assert result["statuses"] == ["valid", "valid", "valid"]
    assert result["mean_rates_a_hz"] == pytest.approx([2.0, 3.0, 5.0])
    assert result["mean_rates_b_hz"] == pytest.approx([4.0, 0.0, 10.0])
    assert result["areas_a"] == pytest.approx([4.0, 6.0, 10.0])
    assert result["areas_b"] == pytest.approx([8.0, 0.0, 20.0])


def test_segmented_overlap_marks_both_silent_segment_undefined() -> None:
    """Two silent curves carry no within-segment shape information."""
    progression = np.asarray([0.1, 0.2, 0.4, 0.5, 0.8, 0.9])
    edges = np.asarray([0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0])

    result = compute_segmented_shape_overlap(
        np.asarray([1.0, 3.0, 0.0, 0.0, 5.0, 5.0]),
        np.asarray([2.0, 6.0, 0.0, 0.0, 10.0, 10.0]),
        progression,
        edges,
    )

    assert result["scores"][0] == pytest.approx(1.0)
    assert np.isnan(result["scores"][1])
    assert result["scores"][2] == pytest.approx(1.0)
    assert result["statuses"] == ["valid", "both_silent", "valid"]
    assert result["mean_rates_a_hz"][1] == pytest.approx(0.0)
    assert result["mean_rates_b_hz"][1] == pytest.approx(0.0)
