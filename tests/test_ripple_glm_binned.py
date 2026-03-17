from __future__ import annotations

import pytest

from v1ca1.ripple.ripple_glm_binned import (
    parse_arguments,
    validate_arguments,
    validate_selected_epochs,
)


def test_parse_arguments_requires_animal_name_and_date() -> None:
    with pytest.raises(SystemExit):
        parse_arguments([])


def test_validate_arguments_rejects_invalid_n_splits() -> None:
    args = parse_arguments(["--animal-name", "L14", "--date", "20240611"])
    args.n_splits = 1

    with pytest.raises(ValueError, match="--n-splits"):
        validate_arguments(args)


def test_parse_arguments_overwrites_by_default_and_supports_no_overwrite() -> None:
    default_args = parse_arguments(["--animal-name", "L14", "--date", "20240611"])
    no_overwrite_args = parse_arguments(
        ["--animal-name", "L14", "--date", "20240611", "--no-overwrite"]
    )

    assert default_args.overwrite is True
    assert no_overwrite_args.overwrite is False


def test_validate_selected_epochs_rejects_unknown_epoch() -> None:
    with pytest.raises(ValueError, match="Requested epochs were not found"):
        validate_selected_epochs(["01_s1", "02_r1"], ["03_s2"])
