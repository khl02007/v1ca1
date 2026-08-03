from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from v1ca1.spyglass import spikes


class _FakeSpikeSortingOutput:
    def __init__(
        self,
        *,
        source: str,
        nwb_payloads: list[dict[str, Any]],
        merge_ids: list[str],
        restriction: dict[str, Any] | None = None,
        sorting: Any = None,
        recording: Any = None,
        sort_group_region: str | None = "v1",
    ) -> None:
        self.source = source
        self.nwb_payloads = nwb_payloads
        self.merge_ids = merge_ids
        self.restriction = restriction
        self.sorting = sorting
        self.recording = recording
        self.sort_group_region = sort_group_region
        self.get_sorting_keys: list[dict[str, Any]] = []

    def __and__(self, restriction: dict[str, Any]) -> "_FakeSpikeSortingOutput":
        return _FakeSpikeSortingOutput(
            source=self.source,
            nwb_payloads=self.nwb_payloads,
            merge_ids=self.merge_ids,
            restriction=dict(restriction),
            sorting=self.sorting,
            recording=self.recording,
            sort_group_region=self.sort_group_region,
        )

    def fetch1(self, attribute: str) -> Any:
        assert attribute == "source"
        return self.source

    def fetch_nwb(self, *, return_merge_ids: bool = False) -> Any:
        assert self.restriction is not None
        if return_merge_ids:
            return self.nwb_payloads, self.merge_ids
        return self.nwb_payloads

    def get_sorting(self, key: dict[str, Any]) -> Any:
        self.get_sorting_keys.append(key)
        return self.sorting

    def get_recording(self, key: dict[str, Any]) -> Any:
        assert key == {"merge_id": "merge-a"}
        return self.recording

    def get_sort_group_info(self, key: dict[str, Any]) -> dict[str, Any]:
        assert key["merge_id"] in self.merge_ids
        if self.sort_group_region is None:
            return {}
        return {"region_name": self.sort_group_region}


class _FakeRecording:
    def get_num_segments(self) -> int:
        return 1

    def has_time_vector(self, *, segment_index: int) -> bool:
        assert segment_index == 0
        return True


class _FakeSortingWithRecording:
    def __init__(self) -> None:
        self._recording: Any = None

    def register_recording(self, recording: Any) -> None:
        self._recording = recording

    def has_recording(self) -> bool:
        return self._recording is not None

    def get_num_segments(self) -> int:
        return 1

    def has_time_vector(self, *, segment_index: int) -> bool:
        assert segment_index == 0
        return self._recording is not None and self._recording.has_time_vector(
            segment_index=segment_index
        )


class _FakeIntervalSet:
    def __init__(self, *, start: float, end: float, **_kwargs: Any) -> None:
        self.start = start
        self.end = end


class _FakeTs:
    def __init__(self, *, t: np.ndarray, **_kwargs: Any) -> None:
        self.t = np.asarray(t, dtype=float)


class _FakeTsGroup(dict[int, _FakeTs]):
    def __init__(self, data: dict[int, _FakeTs], **kwargs: Any) -> None:
        super().__init__(data)
        self.metadata = kwargs["metadata"]


class _FakePynapple:
    IntervalSet = _FakeIntervalSet
    Ts = _FakeTs
    TsGroup = _FakeTsGroup


def _units_table(regions: tuple[str, str] = ("v1", "v1")) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "spike_times": [
                np.array([0.1, 0.4, 0.7], dtype=float),
                np.array([0.2, 0.8], dtype=float),
            ],
            "region": list(regions),
            "sorting_unit_id": [101, 102],
        },
        index=pd.Index([11, 12], name="id"),
    )


class _FakeGroupUnits:
    def __init__(
        self,
        merge_ids: list[str],
        restriction: dict[str, Any] | None = None,
    ) -> None:
        self.merge_ids = list(merge_ids)
        self.restriction = restriction

    def __and__(self, restriction: dict[str, Any]) -> "_FakeGroupUnits":
        return _FakeGroupUnits(self.merge_ids, dict(restriction))

    def fetch(self, attribute: str) -> list[str]:
        assert attribute == "spikesorting_merge_id"
        assert self.restriction is not None
        return list(self.merge_ids)


class _FakeSortedSpikesGroup:
    def __init__(self, merge_ids: list[str]) -> None:
        self.Units = _FakeGroupUnits(merge_ids)


class _FakeUnitSelectionParams:
    def __init__(
        self,
        *,
        include_labels: list[str] | None,
        exclude_labels: list[str] | None,
        restriction: dict[str, Any] | None = None,
    ) -> None:
        self.include_labels = include_labels
        self.exclude_labels = exclude_labels
        self.restriction = restriction

    def __and__(self, restriction: dict[str, Any]) -> "_FakeUnitSelectionParams":
        return _FakeUnitSelectionParams(
            include_labels=self.include_labels,
            exclude_labels=self.exclude_labels,
            restriction=dict(restriction),
        )

    def fetch1(self, *attributes: str) -> tuple[Any, Any]:
        assert attributes == ("include_labels", "exclude_labels")
        assert self.restriction is not None
        return self.include_labels, self.exclude_labels


class _FakeSortingParent:
    def __init__(self, nwb_file_name: str) -> None:
        self.nwb_file_name = nwb_file_name

    def fetch(self, attribute: str) -> list[str]:
        assert attribute == "nwb_file_name"
        return [self.nwb_file_name]


class _FakeRestrictedMultiOutput:
    def __init__(self, source: "_FakeMultiOutput", merge_id: str) -> None:
        self.source = source
        self.merge_id = merge_id

    def fetch1(self, attribute: str) -> str:
        assert attribute == "source"
        return str(self.source.members[self.merge_id]["source"])

    def fetch_nwb(self, *, return_merge_ids: bool = False) -> Any:
        member = self.source.members[self.merge_id]
        payload = {str(member.get("units_field", "object_id")): member["units"]}
        if return_merge_ids:
            return [payload], [self.merge_id]
        return [payload]


class _FakeMultiOutput:
    def __init__(self, members: dict[str, dict[str, Any]]) -> None:
        self.members = members

    def __and__(self, restriction: dict[str, Any]) -> _FakeRestrictedMultiOutput:
        return _FakeRestrictedMultiOutput(self, str(restriction["merge_id"]))

    def merge_get_parent(self, key: dict[str, Any]) -> _FakeSortingParent:
        member = self.members[str(key["merge_id"])]
        return _FakeSortingParent(str(member["nwb_file_name"]))

    def get_sort_group_info(self, key: dict[str, Any]) -> dict[str, Any]:
        member = self.members[str(key["merge_id"])]
        region = member.get("sort_group_region")
        return {} if region is None else {"region_name": region}


def _group_key() -> dict[str, str]:
    return {
        "nwb_file_name": "L1420240102_.nwb",
        "unit_filter_params_name": "exclude_noise",
        "sorted_spikes_group_name": "all shanks",
    }


def _labeled_units(
    *,
    unit_ids: list[int],
    regions: list[str] | None,
    labels: list[Any] | None,
) -> pd.DataFrame:
    table = pd.DataFrame(
        {
            "spike_times": [np.asarray([unit_id / 100.0]) for unit_id in unit_ids],
            "sorting_unit_id": [unit_id + 1000 for unit_id in unit_ids],
        },
        index=pd.Index(unit_ids, name="id"),
    )
    if regions is not None:
        table["region"] = regions
    if labels is not None:
        table["curation_label"] = labels
    return table


@pytest.mark.parametrize(
    ("source", "units_field"),
    [
        ("ImportedSpikeSorting", "object_id"),
        ("CurationV1", "object_id"),
        ("CuratedSpikeSorting", "units"),
    ],
)
def test_fetch_spike_times_seconds_uses_nwb_units_for_every_parent(
    source: str,
    units_field: str,
) -> None:
    relation = _FakeSpikeSortingOutput(
        source=source,
        nwb_payloads=[{units_field: _units_table()}],
        merge_ids=["merge-a"],
    )

    spike_times_s, unit_ids = spikes.fetch_spike_times_seconds(
        relation,
        {"merge_id": "merge-a"},
    )

    assert np.allclose(spike_times_s[0], [0.1, 0.4, 0.7])
    assert np.allclose(spike_times_s[1], [0.2, 0.8])
    assert unit_ids == [
        {"spikesorting_merge_id": "merge-a", "unit_id": 11},
        {"spikesorting_merge_id": "merge-a", "unit_id": 12},
    ]


def test_load_spikes_returns_tsgroup_and_list_compatibility() -> None:
    relation = _FakeSpikeSortingOutput(
        source="CurationV1",
        nwb_payloads=[{"object_id": _units_table()}],
        merge_ids=["merge-a"],
    )

    result = spikes.load_spikes(
        relation,
        {"merge_id": "merge-a"},
        region="V1",
        time_support=(0.0, 1.0),
        pynapple_module=_FakePynapple,
    )

    assert result["merge_parent"] == "CurationV1"
    assert result["region"] == "v1"
    assert list(result["ts_group"]) == [0, 1]
    assert result["ts_group"].metadata == {
        "spikesorting_merge_id": ["merge-a", "merge-a"],
        "unit_id": [11, 12],
    }
    assert result["compatibility"]["time_units"] == "s"
    assert result["compatibility"]["unit_ids"] == result["unit_ids"]
    assert np.allclose(result["compatibility"]["spike_times"][0], [0.1, 0.4, 0.7])


def test_tsgroup_keys_do_not_reorder_nonmonotonic_nwb_unit_ids() -> None:
    group = spikes.build_spike_tsgroup(
        [np.asarray([1.0]), np.asarray([2.0])],
        [
            {"spikesorting_merge_id": "merge-a", "unit_id": 20},
            {"spikesorting_merge_id": "merge-a", "unit_id": 10},
        ],
        time_support=(0.0, 3.0),
        pynapple_module=_FakePynapple,
    )

    assert list(group) == [0, 1]
    np.testing.assert_allclose(group[0].t, [1.0])
    np.testing.assert_allclose(group[1].t, [2.0])
    assert group.metadata["unit_id"] == [20, 10]


def test_fetch_seconds_metadata_preserves_legacy_sorting_unit_id() -> None:
    relation = _FakeSpikeSortingOutput(
        source="ImportedSpikeSorting",
        nwb_payloads=[{"object_id": _units_table(("v1", "ca1"))}],
        merge_ids=["merge-a"],
    )

    spike_times, unit_ids, metadata = (
        spikes.fetch_spike_times_seconds_with_metadata(
            relation,
            {"merge_id": "merge-a"},
            region="ca1",
        )
    )

    assert len(spike_times) == 1
    assert unit_ids == [
        {"spikesorting_merge_id": "merge-a", "unit_id": 12}
    ]
    assert metadata == [
        {
            "spikesorting_merge_id": "merge-a",
            "unit_id": 12,
            "sorting_unit_id": 102,
            "region": "ca1",
        }
    ]


def test_get_spikeinterface_sorting_dispatches_curated_parent() -> None:
    sorting = _FakeSortingWithRecording()
    recording = _FakeRecording()
    relation = _FakeSpikeSortingOutput(
        source="CurationV1",
        nwb_payloads=[{"object_id": _units_table()}],
        merge_ids=["merge-a"],
        sorting=sorting,
        recording=recording,
    )

    result = spikes.get_spikeinterface_sorting(
        relation,
        {"merge_id": "merge-a"},
    )

    assert result is sorting
    assert sorting._recording is recording


def test_imported_sorting_requires_and_uses_injected_factory() -> None:
    relation = _FakeSpikeSortingOutput(
        source="ImportedSpikeSorting",
        nwb_payloads=[{"object_id": _units_table(("v1", "ca1"))}],
        merge_ids=["merge-a"],
    )
    with pytest.raises(NotImplementedError, match="cannot infer"):
        spikes.get_spikeinterface_sorting(relation, {"merge_id": "merge-a"})

    calls: list[dict[str, Any]] = []

    def _factory(**kwargs: Any) -> _FakeSortingWithRecording:
        calls.append(kwargs)
        sorting = _FakeSortingWithRecording()
        sorting.register_recording(_FakeRecording())
        return sorting

    result = spikes.get_spikeinterface_sorting(
        relation,
        {"merge_id": "merge-a"},
        region="ca1",
        sorting_factory=_factory,
    )

    assert isinstance(result, _FakeSortingWithRecording)
    assert calls[0]["merge_parent"] == "ImportedSpikeSorting"
    assert calls[0]["region"] == "ca1"
    assert calls[0]["key"] == {"merge_id": "merge-a"}
    assert calls[0]["nwb_payloads"][0]["object_id"].index.tolist() == [12]


def test_spikeinterface_times_seconds_explicitly_requests_return_times() -> None:
    class _Sorting:
        def __init__(self) -> None:
            self.calls: list[tuple[int, bool]] = []
            self._recording = _FakeRecording()

        def has_recording(self) -> bool:
            return True

        def get_num_segments(self) -> int:
            return 1

        def has_time_vector(self, *, segment_index: int) -> bool:
            assert segment_index == 0
            return True

        def get_unit_ids(self) -> list[int]:
            return [4, 7]

        def get_unit_spike_train(
            self,
            *,
            unit_id: int,
            return_times: bool,
        ) -> np.ndarray:
            self.calls.append((unit_id, return_times))
            return np.array([unit_id / 10.0], dtype=float)

    sorting = _Sorting()
    spike_times_s, unit_ids = spikes.spikeinterface_times_seconds(
        sorting,
        spikesorting_merge_id="merge-b",
    )

    assert sorting.calls == [(4, True), (7, True)]
    assert np.allclose(spike_times_s[0], [0.4])
    assert unit_ids == [
        {"spikesorting_merge_id": "merge-b", "unit_id": 4},
        {"spikesorting_merge_id": "merge-b", "unit_id": 7},
    ]


def test_spikeinterface_seconds_rejects_unregistered_relative_sorting() -> None:
    class _RelativeSorting:
        def has_recording(self) -> bool:
            return False

    with pytest.raises(ValueError, match="return_times=True would not retain"):
        spikes.spikeinterface_times_seconds(
            _RelativeSorting(),
            spikesorting_merge_id="merge-a",
        )


def test_unsupported_merge_parent_fails_before_fetching_nwb() -> None:
    relation = _FakeSpikeSortingOutput(
        source="UnknownSorting",
        nwb_payloads=[{"object_id": _units_table()}],
        merge_ids=["merge-a"],
    )

    with pytest.raises(ValueError, match="Unsupported spike-sorting merge parent"):
        spikes.fetch_spike_times_seconds(relation, {"merge_id": "merge-a"})


def test_imported_units_are_filtered_by_augmented_nwb_region() -> None:
    relation = _FakeSpikeSortingOutput(
        source="ImportedSpikeSorting",
        nwb_payloads=[{"object_id": _units_table(("v1", "ca1"))}],
        merge_ids=["merge-a"],
    )

    spike_times_s, unit_ids = spikes.fetch_spike_times_seconds(
        relation,
        {"merge_id": "merge-a"},
        region="CA1",
    )

    assert len(spike_times_s) == 1
    assert np.allclose(spike_times_s[0], [0.2, 0.8])
    assert unit_ids == [
        {"spikesorting_merge_id": "merge-a", "unit_id": 12}
    ]


def test_curated_sort_group_region_must_match_requested_region() -> None:
    relation = _FakeSpikeSortingOutput(
        source="CurationV1",
        nwb_payloads=[{"object_id": _units_table()}],
        merge_ids=["merge-a"],
        sort_group_region="ca1",
    )

    with pytest.raises(ValueError, match="does not match requested region"):
        spikes.fetch_spike_times_seconds(
            relation,
            {"merge_id": "merge-a"},
            region="v1",
        )


def test_curated_region_validator_is_an_explicit_fallback() -> None:
    relation = _FakeSpikeSortingOutput(
        source="CuratedSpikeSorting",
        nwb_payloads=[{"units": _units_table()}],
        merge_ids=["merge-a"],
        sort_group_region=None,
    )
    calls: list[dict[str, Any]] = []

    def _validate(**kwargs: Any) -> bool:
        calls.append(kwargs)
        return kwargs["region"] == "v1"

    spike_times_s, _ = spikes.fetch_spike_times_seconds(
        relation,
        {"merge_id": "merge-a"},
        region="V1",
        region_validator=_validate,
    )

    assert len(spike_times_s) == 2
    assert calls[0]["merge_parent"] == "CuratedSpikeSorting"
    assert calls[0]["region"] == "v1"


def test_sorted_group_provenance_is_order_independent_and_deterministic() -> None:
    first = spikes.resolve_sorted_spikes_group_provenance(
        _FakeSortedSpikesGroup(["merge-b", "merge-a"]),
        _FakeUnitSelectionParams(
            include_labels=["good", "accepted", "good"],
            exclude_labels=["mua", "noise"],
        ),
        _group_key(),
    )
    second = spikes.resolve_sorted_spikes_group_provenance(
        _FakeSortedSpikesGroup(["merge-a", "merge-b"]),
        _FakeUnitSelectionParams(
            include_labels=["accepted", "good"],
            exclude_labels=["noise", "mua"],
        ),
        _group_key(),
    )

    assert first["merge_ids"] == ["merge-a", "merge-b"]
    assert first["sorting_group_members"] == ["merge-a", "merge-b"]
    assert len(first["sorting_group_members_sha256"]) == 64
    assert first["sorting_group_members_sha256"] == second[
        "sorting_group_members_sha256"
    ]
    assert first["unit_selection_params"] == {
        "unit_filter_params_name": "exclude_noise",
        "include_labels": ["accepted", "good"],
        "exclude_labels": ["mua", "noise"],
    }
    assert first["unit_selection_params_sha256"] == second[
        "unit_selection_params_sha256"
    ]


def test_load_sorted_group_filters_labels_and_mixed_imported_regions() -> None:
    members = {
        "merge-a": {
            "source": "ImportedSpikeSorting",
            "nwb_file_name": "L1420240102_.nwb",
            "units": _labeled_units(
                unit_ids=[11],
                regions=["v1"],
                labels=[["good"]],
            ),
        },
        "merge-b": {
            "source": "ImportedSpikeSorting",
            "nwb_file_name": "L1420240102_.nwb",
            "units": _labeled_units(
                unit_ids=[21, 22],
                regions=["ca1", "ca1"],
                labels=[["good"], ["noise"]],
            ),
        },
    }
    result = spikes.load_sorted_spikes_group(
        _FakeSortedSpikesGroup(["merge-b", "merge-a"]),
        _FakeUnitSelectionParams(include_labels=[], exclude_labels=["noise"]),
        _FakeMultiOutput(members),
        _group_key(),
        region="CA1",
        time_support=(0.0, 1.0),
        pynapple_module=_FakePynapple,
    )

    assert result["status"] == "valid"
    assert result["n_units"] == 1
    assert result["sorting_group_members"] == ["merge-a", "merge-b"]
    assert result["unit_ids"] == [
        {"spikesorting_merge_id": "merge-b", "unit_id": 21}
    ]
    assert list(result["ts_group"]) == [0]
    assert result["ts_group"].metadata == {
        "spikesorting_merge_id": ["merge-b"],
        "unit_id": [21],
    }
    assert [row["n_selected_units"] for row in result["member_provenance"]] == [
        0,
        1,
    ]
    assert result["member_provenance"][0]["region_sources"] == [
        "nwb_units.region"
    ]


def test_load_sorted_group_requires_labels_for_nonempty_filter() -> None:
    members = {
        "merge-a": {
            "source": "ImportedSpikeSorting",
            "nwb_file_name": "L1420240102_.nwb",
            "units": _labeled_units(
                unit_ids=[11],
                regions=["v1"],
                labels=None,
            ),
        }
    }
    with pytest.raises(ValueError, match="Nonempty UnitSelectionParams"):
        spikes.load_sorted_spikes_group(
            _FakeSortedSpikesGroup(["merge-a"]),
            _FakeUnitSelectionParams(include_labels=[], exclude_labels=["noise"]),
            _FakeMultiOutput(members),
            _group_key(),
            region="ca1",
            time_support=(0.0, 1.0),
            allow_empty=True,
            pynapple_module=_FakePynapple,
        )


def test_load_sorted_group_skips_nonmatching_curated_members() -> None:
    members = {
        "merge-a": {
            "source": "CurationV1",
            "nwb_file_name": "L1420240102_.nwb",
            "sort_group_region": "v1",
            "units": _labeled_units(
                unit_ids=[11],
                regions=None,
                labels=["good"],
            ),
        },
        "merge-b": {
            "source": "CurationV1",
            "nwb_file_name": "L1420240102_.nwb",
            "sort_group_region": "ca1",
            "units": _labeled_units(
                unit_ids=[21],
                regions=None,
                labels=["good"],
            ),
        },
    }
    result = spikes.load_sorted_spikes_group(
        _FakeSortedSpikesGroup(["merge-a", "merge-b"]),
        _FakeUnitSelectionParams(include_labels=["good"], exclude_labels=[]),
        _FakeMultiOutput(members),
        _group_key(),
        region="ca1",
        time_support=(0.0, 1.0),
        pynapple_module=_FakePynapple,
    )

    assert result["unit_ids"] == [
        {"spikesorting_merge_id": "merge-b", "unit_id": 21}
    ]
    assert [row["region_sources"] for row in result["member_provenance"]] == [
        ["sort_group_info.region_name"],
        ["sort_group_info.region_name"],
    ]


def test_load_sorted_group_preserves_composite_identity_across_members() -> None:
    members = {
        merge_id: {
            "source": "ImportedSpikeSorting",
            "nwb_file_name": "L1420240102_.nwb",
            "units": _labeled_units(
                unit_ids=[11],
                regions=["ca1"],
                labels=None,
            ),
        }
        for merge_id in ("merge-a", "merge-b")
    }
    result = spikes.load_sorted_spikes_group(
        _FakeSortedSpikesGroup(["merge-b", "merge-a"]),
        _FakeUnitSelectionParams(include_labels=[], exclude_labels=[]),
        _FakeMultiOutput(members),
        {**_group_key(), "unit_filter_params_name": "all_units"},
        region="ca1",
        time_support=(0.0, 1.0),
        pynapple_module=_FakePynapple,
    )

    assert result["unit_ids"] == [
        {"spikesorting_merge_id": "merge-a", "unit_id": 11},
        {"spikesorting_merge_id": "merge-b", "unit_id": 11},
    ]
    assert list(result["ts_group"]) == [0, 1]
    assert result["ts_group"].metadata == {
        "spikesorting_merge_id": ["merge-a", "merge-b"],
        "unit_id": [11, 11],
    }


def test_load_sorted_group_validates_every_member_session() -> None:
    members = {
        "merge-a": {
            "source": "ImportedSpikeSorting",
            "nwb_file_name": "L1420240102_.nwb",
            "units": _labeled_units(
                unit_ids=[11],
                regions=["v1"],
                labels=None,
            ),
        },
        "merge-b": {
            "source": "ImportedSpikeSorting",
            "nwb_file_name": "L1520240102_.nwb",
            "units": _labeled_units(
                unit_ids=[21],
                regions=["v1"],
                labels=None,
            ),
        },
    }
    with pytest.raises(ValueError, match="not sorting-group session"):
        spikes.load_sorted_spikes_group(
            _FakeSortedSpikesGroup(["merge-a", "merge-b"]),
            _FakeUnitSelectionParams(include_labels=[], exclude_labels=[]),
            _FakeMultiOutput(members),
            {**_group_key(), "unit_filter_params_name": "all_units"},
            region="v1",
            time_support=(0.0, 1.0),
            pynapple_module=_FakePynapple,
        )


def test_load_sorted_group_can_return_explicit_empty_result() -> None:
    members = {
        "merge-a": {
            "source": "ImportedSpikeSorting",
            "nwb_file_name": "L1420240102_.nwb",
            "units": _labeled_units(
                unit_ids=[11],
                regions=["v1"],
                labels=None,
            ),
        }
    }
    arguments = (
        _FakeSortedSpikesGroup(["merge-a"]),
        _FakeUnitSelectionParams(include_labels=[], exclude_labels=[]),
        _FakeMultiOutput(members),
        {**_group_key(), "unit_filter_params_name": "all_units"},
    )
    with pytest.raises(ValueError, match="has no units after"):
        spikes.load_sorted_spikes_group(
            *arguments,
            region="ca1",
            time_support=(0.0, 1.0),
            pynapple_module=_FakePynapple,
        )

    result = spikes.load_sorted_spikes_group(
        *arguments,
        region="ca1",
        time_support=(0.0, 1.0),
        allow_empty=True,
        pynapple_module=_FakePynapple,
    )
    assert result["status"] == "no_units"
    assert result["n_units"] == 0
    assert result["unit_ids"] == []
    assert result["spike_times_s"] == []
    assert list(result["ts_group"]) == []
    assert result["sorting_group_members"] == ["merge-a"]
