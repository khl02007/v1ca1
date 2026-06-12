from __future__ import annotations

import numpy as np
import pytest

from v1ca1.paper_figures.supplementary_figure_4_scalar_candidates import (
    compute_segment_curve_similarity,
    select_top_similarity_candidates,
)


def test_compute_segment_curve_similarity_interpolates_model_to_empirical_bins() -> None:
    observed_position = np.asarray([0.35, 0.40, 0.50, 0.60, 0.65])
    observed_rate = np.asarray([9.0, 1.0, 2.0, 3.0, 9.0])
    model_position = np.asarray([0.40, 0.45, 0.50, 0.55, 0.60])
    model_rate = np.asarray([2.0, 4.0, 6.0, 8.0, 10.0])

    score = compute_segment_curve_similarity(
        observed_position,
        observed_rate,
        model_position,
        model_rate,
        0.40,
        0.60,
    )

    assert score is not None
    assert score["similarity_r"] == pytest.approx(1.0)
    assert score["n_similarity_points"] == pytest.approx(3.0)
    assert score["rmse_hz"] > 0.0


def test_select_top_similarity_candidates_keeps_best_trajectory_per_cell() -> None:
    candidates = [
        {
            "animal_name": "L14",
            "date": "20240611",
            "unit_id": 11,
            "trajectory": "center_to_left",
            "similarity_r": 0.95,
            "rmse_hz": 0.5,
            "delta_ll_bits_per_spike": 0.1,
        },
        {
            "animal_name": "L14",
            "date": "20240611",
            "unit_id": 11,
            "trajectory": "center_to_right",
            "similarity_r": 0.99,
            "rmse_hz": 0.4,
            "delta_ll_bits_per_spike": 0.2,
        },
        {
            "animal_name": "L15",
            "date": "20241121",
            "unit_id": 12,
            "trajectory": "center_to_left",
            "similarity_r": 0.96,
            "rmse_hz": 0.3,
            "delta_ll_bits_per_spike": 0.0,
        },
    ]

    selected = select_top_similarity_candidates(candidates, top_n=2)

    assert [(row["unit_id"], row["trajectory"]) for row in selected] == [
        (11, "center_to_right"),
        (12, "center_to_left"),
    ]
