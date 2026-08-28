"""Tests for the manuscript Spyglass population campaign planner."""

from __future__ import annotations

import pytest

from v1ca1.spyglass.populate_figures import (
    STAGE_ORDER,
    _register_analysis_nwbfile_table,
    _require_expected_result_primary_key,
    _require_one_pending_populate_job,
    figure_dataset_specs,
    planned_result_counts,
    select_figure_datasets,
    stages_through,
)


class _FakePendingRelation:
    def __init__(self, count: int):
        self.count = count

    def __and__(self, restriction):
        return self

    def __sub__(self, table):
        return self

    def __len__(self):
        return self.count


class _FakeResultTable:
    primary_key = ("result_id",)

    def __init__(self, pending_count: int):
        self.key_source = _FakePendingRelation(pending_count)


class _FakeAnalysisNwbfile:
    def __init__(self):
        self.registration_count = 0

    def __call__(self):
        return self

    def register_with_spyglass(self):
        self.registration_count += 1


def test_figure_dataset_specs_match_current_manuscript_cohort() -> None:
    specs = figure_dataset_specs()

    assert [
        (spec["animal_name"], spec["date"], spec["dark_epoch"])
        for spec in specs
    ] == [
        ("L12", "20240421", "08_r4"),
        ("L14", "20240611", "08_r4"),
        ("L15", "20241121", "10_r5"),
        ("L19", "20250930", "08_r4"),
    ]


def test_select_figure_datasets_requires_one_explicit_mode() -> None:
    selected = select_figure_datasets(
        animal_name="L14",
        date="20240611",
        all_datasets=False,
    )
    assert len(selected) == 1
    assert selected[0]["light_epoch"] == "02_r1"

    assert len(
        select_figure_datasets(
            animal_name=None,
            date=None,
            all_datasets=True,
        )
    ) == 4
    with pytest.raises(ValueError, match="Pass both"):
        select_figure_datasets(
            animal_name="L14",
            date=None,
            all_datasets=False,
        )
    with pytest.raises(ValueError, match="cannot be combined"):
        select_figure_datasets(
            animal_name="L14",
            date="20240611",
            all_datasets=True,
        )


def test_planned_result_counts_cover_exact_figure_dependency_closure() -> None:
    specs = figure_dataset_specs()
    totals = [sum(planned_result_counts(spec).values()) for spec in specs]

    assert totals == [98, 98, 99, 98]
    assert sum(totals) == 393
    assert planned_result_counts(specs[0])["dpp_encoding"] == 1
    assert "dpp_tuning_curve" not in planned_result_counts(specs[0])
    assert planned_result_counts(specs[2])["ripple_cross_region_xcorr"] == 1


def test_stages_are_explicit_and_dependency_ordered() -> None:
    assert stages_through("all") == STAGE_ORDER
    assert stages_through("tuning") == ("tuning",)
    assert stages_through("preflight") == ()
    assert stages_through("status") == ()
    with pytest.raises(ValueError, match="Unknown population stage"):
        stages_through("unknown")


def test_runtime_registers_custom_analysis_nwbfile_table() -> None:
    analysis_nwbfile = _FakeAnalysisNwbfile()

    _register_analysis_nwbfile_table({"analysis_nwbfile": analysis_nwbfile})

    assert analysis_nwbfile.registration_count == 1
    with pytest.raises(RuntimeError, match="did not return"):
        _register_analysis_nwbfile_table({})


def test_population_guard_rejects_unexpected_live_primary_key() -> None:
    table = _FakeResultTable(pending_count=1)
    table.primary_key = ("result_id", "analysis_file_name")

    with pytest.raises(RuntimeError, match="Repair the live schema"):
        _require_expected_result_primary_key(
            table,
            result_table_key="result",
            id_field="result_id",
        )


def test_population_guard_requires_exactly_one_pending_job() -> None:
    table = _FakeResultTable(pending_count=1)
    _require_one_pending_populate_job(
        table,
        {"result_id": "selection"},
        result_table_key="result",
    )

    table.key_source.count = 2
    with pytest.raises(RuntimeError, match="2 pending populate jobs"):
        _require_one_pending_populate_job(
            table,
            {"result_id": "selection"},
            result_table_key="result",
        )
