"""Tests for database-backed manuscript-figure path resolution."""

from pathlib import Path

from v1ca1.paper_figures._spyglass_database import SpyglassFigureDatabase


class _FakeNwbfile:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.restrictions = []
        self.fetch_count = 0

    def __and__(self, restriction):
        self.restrictions.append(dict(restriction))
        return self

    def __len__(self) -> int:
        return 1

    def fetch1(self):
        self.fetch_count += 1
        return {"nwb_file_name": self.path.name}

    def get_abs_path(self, nwb_file_name: str) -> str:
        assert nwb_file_name == self.path.name
        return str(self.path)


def test_registered_nwb_path_does_not_require_original_source(tmp_path):
    registered_path = tmp_path / "L1420240611_augmented_.nwb"
    registered_path.touch()
    nwbfile = _FakeNwbfile(registered_path)
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
    assert database.registered_nwb_path(spec) == registered_path
    assert database.registered_nwb_path(spec) == registered_path
    assert nwbfile.restrictions == [
        {"nwb_file_name": registered_path.name}
    ]
    assert nwbfile.fetch_count == 1
