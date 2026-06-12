import numpy as np
import pytest

from v1ca1.task_progression import decoding_comparison as module


def test_decoding_transfer_specs_include_same_inbound_outbound_cross_arm() -> None:
    assert module.SAME_INBOUND_OUTBOUND_CROSS_ARM_FAMILY in (
        module.DECODING_TP_TRANSFER_FAMILY_ORDER
    )

    new_specs = [
        pair_spec
        for pair_spec in module.DECODING_TP_TRANSFER_PAIR_SPECS
        if pair_spec["transfer_family"] == module.SAME_INBOUND_OUTBOUND_CROSS_ARM_FAMILY
    ]
    assert [
        (pair_spec["source_trajectory"], pair_spec["target_trajectory"])
        for pair_spec in new_specs
    ] == list(module.SAME_INBOUND_OUTBOUND_CROSS_ARM_PAIRS)
    assert all(not pair_spec.get("flip_tuning_curve", False) for pair_spec in new_specs)


def test_decoding_transfer_specs_do_not_change_shared_task_progression_specs() -> None:
    shared_pairs = {
        (
            pair_spec["transfer_family"],
            pair_spec["source_trajectory"],
            pair_spec["target_trajectory"],
        )
        for pair_spec in module.TP_TRANSFER_PAIR_SPECS
    }
    decoding_pairs = {
        (
            pair_spec["transfer_family"],
            pair_spec["source_trajectory"],
            pair_spec["target_trajectory"],
        )
        for pair_spec in module.DECODING_TP_TRANSFER_PAIR_SPECS
    }

    assert len(decoding_pairs) == len(shared_pairs) + len(
        module.SAME_INBOUND_OUTBOUND_CROSS_ARM_PAIRS
    )
    assert all(
        (
            module.SAME_INBOUND_OUTBOUND_CROSS_ARM_FAMILY,
            source_trajectory,
            target_trajectory,
        )
        not in shared_pairs
        for source_trajectory, target_trajectory in module.SAME_INBOUND_OUTBOUND_CROSS_ARM_PAIRS
    )


def test_filter_epochs_with_count_bins_drops_short_empty_fragments() -> None:
    nap = pytest.importorskip("pynapple")

    time_support = nap.IntervalSet(start=np.asarray([0.0]), end=np.asarray([1.0]))
    spikes = nap.TsGroup(
        {
            0: nap.Ts(
                t=np.asarray([0.11, 0.51]),
                time_support=time_support,
                time_units="s",
            )
        },
        time_support=time_support,
        time_units="s",
    )
    epochs = nap.IntervalSet(
        start=np.asarray([0.0, 0.10, 0.50]),
        end=np.asarray([0.001, 0.13, 0.56]),
        time_units="s",
    )

    filtered = module._filter_epochs_with_count_bins(
        spikes,
        epochs,
        bin_size_s=0.02,
    )

    assert np.asarray(filtered.start).tolist() == pytest.approx([0.10, 0.50])
    assert np.asarray(filtered.end).tolist() == pytest.approx([0.13, 0.56])


def test_parse_arguments_accepts_optional_tuning_stability_threshold() -> None:
    args = module.parse_arguments(
        [
            "--animal-name",
            "L14",
            "--date",
            "20240611",
            "--dark-epoch",
            "08_r4",
        ]
    )

    assert args.min_tuning_stability_correlation is None

    args = module.parse_arguments(
        [
            "--animal-name",
            "L14",
            "--date",
            "20240611",
            "--dark-epoch",
            "08_r4",
            "--min-tuning-stability-correlation",
            "0.5",
        ]
    )

    assert args.min_tuning_stability_correlation == pytest.approx(0.5)
    assert module.build_stability_filter_token(
        args.min_tuning_stability_correlation
    ) == "stable0p5"


def test_cross_trajectory_unit_selection_uses_epoch_firing_rate_and_stability() -> None:
    pd = pytest.importorskip("pandas")

    stability_table = pd.DataFrame(
        {
            "region": ["v1", "v1", "v1", "v1", "v1", "ca1", "v1"],
            "epoch": ["08_r4", "08_r4", "08_r4", "08_r4", "06_r3", "08_r4", "08_r4"],
            "trajectory_type": [
                "center_to_left",
                "right_to_center",
                "center_to_right",
                "left_to_center",
                "center_to_left",
                "center_to_left",
                "not_a_trajectory",
            ],
            "unit": [10, 10, 11, 12, 13, 14, 15],
            "stability_correlation": [0.2, 0.5, 0.9, 0.49, 0.95, 0.99, 1.0],
        }
    )

    table = module.build_cross_trajectory_unit_selection_table(
        animal_name="L14",
        date="20240611",
        region="v1",
        epoch="08_r4",
        unit_ids=[10, 11, 12, 13],
        movement_firing_rates=np.asarray([0.6, 0.4, 0.8, 0.9]),
        cross_traj_fr_threshold_hz=0.5,
        min_tuning_stability_correlation=0.5,
        stability_table=stability_table,
    )

    assert table["passes_firing_rate_threshold"].tolist() == [
        True,
        False,
        True,
        True,
    ]
    assert table["passes_tuning_stability_threshold"].tolist() == [
        True,
        True,
        False,
        False,
    ]
    assert table["selected_for_cross_trajectory_decoding"].tolist() == [
        True,
        False,
        False,
        False,
    ]
    assert table["max_stability_correlation"].tolist() == pytest.approx(
        [0.5, 0.9, 0.49, np.nan],
        nan_ok=True,
    )
    assert module.get_selected_cross_trajectory_unit_ids(table) == [10]


def test_load_tuning_stability_table_reports_missing_artifact(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="task_progression.stability"):
        module.load_tuning_stability_table(
            analysis_path=tmp_path,
            animal_name="L14",
            date="20240611",
        )


def test_save_cross_trajectory_tsds_adds_stability_filter_token(tmp_path) -> None:
    class FakeTsd:
        def __init__(self, label: str) -> None:
            self.label = label

        def save(self, path) -> None:
            path.write_text(self.label, encoding="utf-8")

    key = (
        "same_turn_cross_arm",
        "center_to_left",
        "right_to_center",
    )
    true_by_region = {"v1": {"08_r4": {key: FakeTsd("true")}}}
    decoded_by_region = {"v1": {"08_r4": {key: FakeTsd("decoded")}}}

    paths = module.save_cross_trajectory_tsds(
        true_by_region,
        decoded_by_region,
        save_dir=tmp_path,
        output_token="stable0p5",
    )

    assert [path.name for path in paths] == [
        (
            "v1_08_r4_stable0p5_same_turn_cross_arm_center_to_left_to_"
            "right_to_center_true_tp_cross_traj.npz"
        ),
        (
            "v1_08_r4_stable0p5_same_turn_cross_arm_center_to_left_to_"
            "right_to_center_decoded_tp_cross_traj.npz"
        ),
    ]


def test_save_selected_unit_tables_only_when_stability_filter_is_tokenized(tmp_path) -> None:
    pd = pytest.importorskip("pandas")

    tables = {
        "v1": {
            "08_r4": pd.DataFrame(
                {
                    "unit": [10],
                    "selected_for_cross_trajectory_decoding": [True],
                }
            )
        }
    }

    assert module.save_cross_trajectory_selected_unit_tables(
        tables,
        save_dir=tmp_path,
        output_token=None,
    ) == []

    paths = module.save_cross_trajectory_selected_unit_tables(
        tables,
        save_dir=tmp_path,
        output_token="stable0p5",
    )

    assert [path.name for path in paths] == [
        "v1_08_r4_stable0p5_cross_trajectory_selected_units.parquet"
    ]
