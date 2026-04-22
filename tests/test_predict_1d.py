from __future__ import annotations

"""Tests for 1D decoder prediction helpers."""

from pathlib import Path
from typing import Any

import numpy as np

from v1ca1.decoding import predict_1d
from v1ca1.decoding._1d import (
    POSTERIOR_KIND_ACAUSAL,
    POSTERIOR_KIND_CAUSAL,
    build_prediction_output_paths,
)


def test_causal_prediction_paths_have_distinct_suffix(tmp_path: Path) -> None:
    """Causal outputs should not overwrite default acausal outputs."""
    base_kwargs = {
        "regions": ("ca1",),
        "epoch": "02_r1",
        "n_folds": 5,
        "random_state": 47,
        "direction": True,
        "movement": True,
        "speed_threshold_cm_s": 4.0,
        "position_std": 4.0,
        "discrete_var": "switching",
        "place_bin_size": 2.0,
        "movement_var": 6.0,
        "branch_gap_cm": 15.0,
    }

    acausal_path = build_prediction_output_paths(
        tmp_path,
        **base_kwargs,
        posterior_kind=POSTERIOR_KIND_ACAUSAL,
    )["ca1"]
    causal_path = build_prediction_output_paths(
        tmp_path,
        **base_kwargs,
        posterior_kind=POSTERIOR_KIND_CAUSAL,
    )["ca1"]

    assert "_posterior_causal" not in acausal_path.name
    assert causal_path.name.endswith("_posterior_causal.nc")
    assert causal_path != acausal_path


def test_predict_region_disables_acausal_pass_for_causal_only(
    monkeypatch: Any,
) -> None:
    """Causal-only prediction should pass is_compute_acausal=False to RTC."""
    predict_calls = []

    class FakeClassifier:
        place_fields_ = {"place": object()}

        def predict(self, spikes: np.ndarray, **kwargs: Any) -> dict[str, Any]:
            predict_calls.append(
                {
                    "spikes": spikes.copy(),
                    **kwargs,
                }
            )
            return {"time": kwargs["time"]}

    monkeypatch.setattr(
        predict_1d,
        "get_spike_indicator",
        lambda *args, **kwargs: np.ones((4, 2), dtype=np.uint8),
    )
    monkeypatch.setattr(predict_1d, "load_classifier", lambda _path: FakeClassifier())
    monkeypatch.setattr(
        predict_1d,
        "validate_classifier_place_fields",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        predict_1d,
        "concatenate_fold_results",
        lambda fold_results, **kwargs: fold_results,
    )

    result, spike_indicator = predict_1d.predict_region(
        region="ca1",
        sorting=object(),
        classifier_paths={
            ("ca1", 0): Path("fold0.pkl"),
            ("ca1", 1): Path("fold1.pkl"),
        },
        timestamps_ephys_all=np.arange(4, dtype=float),
        time_grid=np.array([0.0, 0.002, 0.004, 0.006]),
        unit_ids=[1, 2],
        fold_by_time=np.array([0, 0, 1, 1]),
        linear_position=np.arange(4, dtype=float),
        speed=np.ones(4, dtype=float),
        n_folds=2,
        state_names=["Continuous", "Fragmented"],
        causal_only=True,
    )

    assert len(result) == 2
    assert spike_indicator.shape == (4, 2)
    assert [call["is_compute_acausal"] for call in predict_calls] == [False, False]
    assert [call["use_gpu"] for call in predict_calls] == [True, True]
    assert [call["spikes"].shape for call in predict_calls] == [(2, 2), (2, 2)]
