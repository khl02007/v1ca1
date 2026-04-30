from __future__ import annotations

import numpy as np
import pandas as pd

from v1ca1.signal_dim import meme


def test_normalize_meme_tensor_raw_returns_copy() -> None:
    tensor = np.arange(24, dtype=float).reshape(2, 4, 3)

    normalized = meme.normalize_meme_tensor(
        tensor,
        normalization="raw",
        min_scale=1e-6,
    )

    assert np.allclose(normalized, tensor)
    assert normalized is not tensor


def test_normalize_meme_tensor_zscores_repeat_averaged_tuning() -> None:
    base = np.asarray(
        [
            [1.0, 10.0, 5.0],
            [2.0, 20.0, 5.0 + 1e-12],
            [3.0, 30.0, 5.0 - 1e-12],
            [4.0, 40.0, 5.0],
        ]
    )
    tensor = np.stack([base, base], axis=0)

    normalized = meme.normalize_meme_tensor(
        tensor,
        normalization="zscore",
        min_scale=1e-6,
    )
    mean_tuning = normalized.mean(axis=0)

    assert np.allclose(mean_tuning[:, :2].mean(axis=0), 0.0)
    assert np.allclose(mean_tuning[:, :2].std(axis=0), 1.0)
    assert np.max(np.abs(normalized[:, :, 2])) < 1e-6


def test_save_epoch_summary_tables_excludes_repeat_averaged_pca(tmp_path) -> None:
    settings = {
        "normalization": "zscore",
        "output_suffix": "_zscore",
        "unit_filter_mode": "shared-active",
        "min_condition_sd_hz": 1e-6,
        "bin_size_cm": 4.0,
        "n_groups": 4,
        "n_random_repeats": 5,
        "random_repeat_ci": 95.0,
        "full_n_pairings": 10,
        "full_n_bin_perms": 2,
        "random_repeat_n_pairings": 5,
        "random_repeat_n_bin_perms": 2,
    }

    paths = meme.save_epoch_summary_tables(
        tmp_path,
        regions=["v1"],
        analysis_epochs=["08_r4"],
        epoch_to_condition={"08_r4": "dark"},
        n_neurons_by_region={"v1": {"08_r4": 12}},
        meme_pr={"v1": {"08_r4": 3.5}},
        meme_repeat_summary={
            "v1": {
                "08_r4": meme.PRSummary(
                    pr_center=3.4,
                    ci_low=3.1,
                    ci_high=3.8,
                    n_eff=5,
                )
            }
        },
        min_firing_rate_by_region={"v1": 0.5},
        n_units_by_class_by_region={
            "v1": {
                "shared_active": 12,
                "dark_only": 0,
                "light_only": 0,
                "inactive_or_low_mod": 3,
            }
        },
        settings=settings,
    )

    table = pd.read_parquet(paths[0])

    assert paths[0].name == "v1_08_r4_zscore_meme_summary.parquet"
    assert "repeat_averaged_pca_pr" not in table.columns
    assert table["normalization"].iloc[0] == "zscore"
    assert table["meme_pr"].iloc[0] == 3.5
    assert table["random_repeat_pr_center"].iloc[0] == 3.4
