from __future__ import annotations

import hashlib
import os
from pathlib import Path
import subprocess
import sys
from typing import Any
import uuid

import pytest

from v1ca1.spyglass import region_sorted_spikes
from v1ca1.spyglass.selection import provenance_sha256, unit_identity_sha256


def _adapter_output(
    *,
    region: str = "v1",
    members: tuple[str, ...] = ("merge-b", "merge-a"),
    include_labels: tuple[str, ...] = ("good", "accepted"),
    exclude_labels: tuple[str, ...] = ("noise",),
    unit_ids: tuple[tuple[str, int], ...] = (
        ("merge-a", 11),
        ("merge-b", 21),
    ),
) -> dict[str, Any]:
    sorted_members = sorted(members)
    filter_snapshot = {
        "unit_filter_params_name": "exclude_noise",
        "include_labels": sorted(include_labels),
        "exclude_labels": sorted(exclude_labels),
    }
    identities = [
        {"spikesorting_merge_id": merge_id, "unit_id": unit_id}
        for merge_id, unit_id in unit_ids
    ]
    return {
        "group_key": {
            "nwb_file_name": "L1420240102_.nwb",
            "unit_filter_params_name": "exclude_noise",
            "sorted_spikes_group_name": "all shanks",
        },
        "region": region,
        "sorting_group_members": sorted_members,
        "sorting_group_members_sha256": hashlib.sha256(
            "\0".join(sorted_members).encode("utf-8")
        ).hexdigest(),
        "unit_selection_params": filter_snapshot,
        "unit_selection_params_sha256": provenance_sha256(filter_snapshot),
        "n_units": len(identities),
        "unit_ids": identities,
        "spike_times_s": [[0.1], [0.2]],
        "unit_metadata": [{"ignored": True}],
        "member_provenance": [{"ignored": True}],
    }


def test_build_registration_row_is_deterministic_and_minimal() -> None:
    first = region_sorted_spikes.build_region_sorted_spikes_group_row(
        _adapter_output(region=" MEC ")
    )
    reordered = region_sorted_spikes.build_region_sorted_spikes_group_row(
        _adapter_output(
            region="mec",
            members=("merge-a", "merge-b"),
            include_labels=("accepted", "good"),
            unit_ids=(("merge-b", 21), ("merge-a", 11)),
        )
    )

    assert set(first) == set(region_sorted_spikes.REGISTRATION_FIELDS)
    assert first == reordered
    assert first["region_name"] == "mec"
    assert first["sorting_group_members"] == ["merge-a", "merge-b"]
    assert first["unit_filter_include_labels"] == ["accepted", "good"]
    assert first["n_units"] == 2
    assert first["selected_units_sha256"] == unit_identity_sha256(
        _adapter_output()["unit_ids"]
    )
    assert isinstance(first["region_sorted_spikes_group_id"], uuid.UUID)
    assert first["region_sorted_spikes_group_id"].version == 5
    assert "unit_ids" not in first
    assert "spike_times_s" not in first


def test_registration_uuid_changes_with_selected_unit_identity() -> None:
    first = region_sorted_spikes.build_region_sorted_spikes_group_row(
        _adapter_output()
    )
    changed = region_sorted_spikes.build_region_sorted_spikes_group_row(
        _adapter_output(unit_ids=(("merge-a", 11), ("merge-b", 22)))
    )

    assert first["n_units"] == changed["n_units"]
    assert first["selected_units_sha256"] != changed["selected_units_sha256"]
    assert first["region_sorted_spikes_group_id"] != changed[
        "region_sorted_spikes_group_id"
    ]


@pytest.mark.parametrize("region", [None, "", "   ", "x" * 65])
def test_region_must_be_nonempty(region: Any) -> None:
    with pytest.raises(ValueError, match="region must be"):
        region_sorted_spikes.build_region_sorted_spikes_group_row(
            _adapter_output(region=region)
        )


def test_builder_rejects_internally_inconsistent_adapter_output() -> None:
    bad_membership = _adapter_output()
    bad_membership["sorting_group_members_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="sorting_group_members_sha256"):
        region_sorted_spikes.build_region_sorted_spikes_group_row(bad_membership)

    bad_filter = _adapter_output()
    bad_filter["unit_selection_params"]["include_labels"] = ["other"]
    with pytest.raises(ValueError, match="unit_selection_params_sha256"):
        region_sorted_spikes.build_region_sorted_spikes_group_row(bad_filter)

    bad_count = _adapter_output()
    bad_count["n_units"] = 1
    with pytest.raises(ValueError, match="n_units must equal"):
        region_sorted_spikes.build_region_sorted_spikes_group_row(bad_count)

    missing_group_field = _adapter_output()
    del missing_group_field["group_key"]["nwb_file_name"]
    with pytest.raises(ValueError, match="nwb_file_name must be"):
        region_sorted_spikes.build_region_sorted_spikes_group_row(
            missing_group_field
        )


def test_validate_detects_current_source_changes() -> None:
    loaded = _adapter_output()
    row = region_sorted_spikes.build_region_sorted_spikes_group_row(loaded)
    region_sorted_spikes.validate_region_sorted_spikes_group_row(row, loaded)

    changed_units = _adapter_output(unit_ids=(("merge-a", 12), ("merge-b", 21)))
    with pytest.raises(ValueError, match="selected_units_sha256"):
        region_sorted_spikes.validate_region_sorted_spikes_group_row(
            row,
            changed_units,
        )

    changed_members = _adapter_output(members=("merge-a", "merge-c"))
    with pytest.raises(ValueError, match="sorting_group_members"):
        region_sorted_spikes.validate_region_sorted_spikes_group_row(
            row,
            changed_members,
        )


def test_validate_rejects_tampered_registration_uuid() -> None:
    loaded = _adapter_output()
    row = region_sorted_spikes.build_region_sorted_spikes_group_row(loaded)
    row["region_sorted_spikes_group_id"] = uuid.uuid4()

    with pytest.raises(ValueError, match="registration UUID"):
        region_sorted_spikes.validate_region_sorted_spikes_group_row(row, loaded)


def test_reload_uses_registered_group_and_verifies_source(monkeypatch: Any) -> None:
    loaded = _adapter_output(region="ca1")
    row = region_sorted_spikes.build_region_sorted_spikes_group_row(loaded)
    calls: list[dict[str, Any]] = []

    def _load(*args: Any, **kwargs: Any) -> dict[str, Any]:
        calls.append({"args": args, "kwargs": kwargs})
        return loaded

    monkeypatch.setattr(region_sorted_spikes, "load_sorted_spikes_group", _load)
    result = region_sorted_spikes.reload_region_sorted_spikes_group(
        row,
        sorted_spikes_group="group table",
        unit_selection_params="filter table",
        spike_sorting_output="merge table",
        time_support=(0.0, 1.0),
        pynapple_module="pynapple",
    )

    assert result is loaded
    assert calls[0]["args"][:3] == (
        "group table",
        "filter table",
        "merge table",
    )
    assert calls[0]["args"][3] == loaded["group_key"]
    assert calls[0]["kwargs"] == {
        "region": "ca1",
        "region_validator": None,
        "time_support": (0.0, 1.0),
        "allow_empty": True,
        "pynapple_module": "pynapple",
    }


def test_module_import_does_not_import_datajoint_or_spyglass() -> None:
    source_root = Path(__file__).resolve().parents[1] / "src"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(source_root), environment.get("PYTHONPATH", "")]
    )
    code = """
import sys
import v1ca1.spyglass.region_sorted_spikes
assert 'datajoint' not in sys.modules
assert 'spyglass' not in sys.modules
"""
    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        env=environment,
        capture_output=True,
        text=True,
    )
