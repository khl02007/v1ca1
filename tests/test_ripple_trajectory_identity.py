from __future__ import annotations

import pytest

from v1ca1.ripple.ripple_trajectory_identity import parse_arguments


def test_parse_arguments_requires_animal_name_and_date(monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["ripple_trajectory_identity.py"])

    with pytest.raises(SystemExit):
        parse_arguments()
