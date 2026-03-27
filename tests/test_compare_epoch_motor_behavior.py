from __future__ import annotations

import numpy as np
import pytest

from v1ca1.motor.compare_epoch_motor_behavior import (
    MOTOR_VARIABLES,
    build_pairwise_distance_table,
    compute_motor_variables,
    filter_finite_position_samples,
    summarize_progression_values,
)


def test_compute_motor_variables_preserves_length_and_returns_finite_outputs() -> None:
    pytest.importorskip("position_tools")

    timestamps = np.linspace(0.0, 1.0, 11)
    head_position = np.column_stack(
        (
            np.linspace(0.0, 10.0, timestamps.size),
            np.sin(np.linspace(0.0, np.pi, timestamps.size)),
        )
    )
    body_position = head_position - np.array([1.0, 0.5])

    variables = compute_motor_variables(head_position, body_position, timestamps)

    assert set(variables) == set(MOTOR_VARIABLES)
    for values in variables.values():
        assert values.shape == timestamps.shape
        assert np.isfinite(values).all()


def test_filter_finite_position_samples_drops_nan_rows_and_preserves_order() -> None:
    timestamps = np.array([0.0, 0.1, 0.2, 0.3])
    head_position = np.array(
        [
            [0.0, 1.0],
            [np.nan, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
        ]
    )
    body_position = np.array(
        [
            [-1.0, 0.0],
            [0.0, 1.0],
            [1.0, 2.0],
            [2.0, np.nan],
        ]
    )

    filtered_timestamps, filtered_head, filtered_body, dropped_count = (
        filter_finite_position_samples(
            timestamps,
            head_position,
            body_position,
        )
    )

    assert dropped_count == 2
    assert np.allclose(filtered_timestamps, [0.0, 0.2])
    assert np.allclose(filtered_head, [[0.0, 1.0], [2.0, 3.0]])
    assert np.allclose(filtered_body, [[-1.0, 0.0], [1.0, 2.0]])


def test_summarize_progression_values_drops_empty_bins_and_preserves_labels() -> None:
    table = summarize_progression_values(
        epoch="02_r1",
        trajectory_type="center_to_left",
        variable_name="speed_cm_s",
        progression=np.array([0.05, 0.10, 0.82, 0.95]),
        values=np.array([10.0, 12.0, 20.0, 24.0]),
        progression_bin_edges=np.array([0.0, 0.25, 0.50, 0.75, 1.0]),
    )

    assert table["epoch"].tolist() == ["02_r1", "02_r1"]
    assert table["trajectory_type"].tolist() == ["center_to_left", "center_to_left"]
    assert table["variable"].tolist() == ["speed_cm_s", "speed_cm_s"]
    assert table["progression_bin_index"].tolist() == [0, 3]
    assert table["sample_count"].tolist() == [2, 2]
    assert np.allclose(table["median"], [11.0, 22.0])


def test_build_pairwise_distance_table_is_symmetric_and_epoch_ordered() -> None:
    selected_epochs = ["02_r1", "04_r2", "06_r3"]
    values_by_epoch = {
        "02_r1": {
            variable_name: np.array([0.0, 0.0, 1.0, 1.0], dtype=float)
            for variable_name in MOTOR_VARIABLES
        },
        "04_r2": {
            variable_name: np.array([0.0, 0.5, 1.0, 1.5], dtype=float)
            for variable_name in MOTOR_VARIABLES
        },
        "06_r3": {
            variable_name: np.array([1.0, 1.0, 2.0, 2.0], dtype=float)
            for variable_name in MOTOR_VARIABLES
        },
    }
    bin_edges_by_variable = {
        variable_name: np.array([-180.0, 0.0, 180.0], dtype=float)
        if variable_name == "head_direction_deg"
        else np.array([0.0, 1.0, 2.0], dtype=float)
        for variable_name in MOTOR_VARIABLES
    }

    table = build_pairwise_distance_table(
        selected_epochs=selected_epochs,
        values_by_epoch=values_by_epoch,
        bin_edges_by_variable=bin_edges_by_variable,
    )

    speed_table = table.loc[table["variable"].astype(str) == "speed_cm_s"].reset_index(drop=True)
    assert speed_table["epoch_a"].astype(str).tolist()[:3] == ["02_r1", "02_r1", "02_r1"]
    assert speed_table["epoch_b"].astype(str).tolist()[:3] == ["02_r1", "04_r2", "06_r3"]

    divergence_ab = speed_table.loc[
        (speed_table["epoch_a"].astype(str) == "02_r1")
        & (speed_table["epoch_b"].astype(str) == "06_r3"),
        "jensen_shannon_divergence",
    ].item()
    divergence_ba = speed_table.loc[
        (speed_table["epoch_a"].astype(str) == "06_r3")
        & (speed_table["epoch_b"].astype(str) == "02_r1"),
        "jensen_shannon_divergence",
    ].item()
    divergence_diag = speed_table.loc[
        (speed_table["epoch_a"].astype(str) == "04_r2")
        & (speed_table["epoch_b"].astype(str) == "04_r2"),
        "jensen_shannon_divergence",
    ].item()

    assert divergence_ab == divergence_ba
    assert divergence_diag == 0.0
