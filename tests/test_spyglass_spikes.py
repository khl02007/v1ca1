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
