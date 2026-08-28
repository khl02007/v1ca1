"""Tests for database-backed manuscript-figure path resolution."""

from pathlib import Path

from v1ca1.paper_figures._spyglass_database import SpyglassFigureDatabase


class _FakeNwbfile:
    def __init__(self, path: Path, nwbfile: object) -> None:
        self.path = Path(path)
        self.nwbfile = nwbfile
        self.restrictions = []
        self.fetch_count = 0

    def __and__(self, restriction):
        self.restrictions.append(dict(restriction))
        return self

    def __len__(self) -> int:
        return 1

    def fetch_nwb(self):
        self.fetch_count += 1
        return [self.nwbfile]


def test_source_nwb_uses_spyglass_fetch_and_cache(tmp_path):
    registered_path = tmp_path / "L1420240611_augmented_.nwb"
    source_nwb = object()
    nwbfile = _FakeNwbfile(registered_path, source_nwb)
    database = object.__new__(SpyglassFigureDatabase)
    database.nwb_root = tmp_path
    database.runtime = {
        "get_nwb_copy_filename": lambda name: name.replace(
            ".nwb", "_.nwb"
        ),
        "Nwbfile": nwbfile,
    }
    database._cache = {}
    spec = {"animal_name": "L14", "date": "20240611"}

    assert not database.raw_nwb_path(spec).exists()
    assert database.nwb_file_name(spec) == registered_path.name
    with database.open_source_nwb(spec) as fetched:
        assert fetched is source_nwb
    with database.open_source_nwb(spec) as fetched:
        assert fetched is source_nwb
    assert nwbfile.restrictions == [
        {"nwb_file_name": registered_path.name}
    ]
    assert nwbfile.fetch_count == 1
