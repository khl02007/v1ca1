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
