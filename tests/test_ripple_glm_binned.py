from __future__ import annotations

from types import SimpleNamespace

import pytest

import v1ca1.ripple.ripple_glm_binned as ripple_glm_binned_module
from v1ca1.ripple.ripple_glm_binned import (
    _format_nemos_solver_selection_message,
    _resolve_nemos_population_glm_solver,
)


def test_resolve_nemos_population_glm_solver_keeps_lbfgs_for_nemos_0_2_5() -> None:
    version_text, solver_name = _resolve_nemos_population_glm_solver(
        SimpleNamespace(__version__="0.2.5")
    )

    message = _format_nemos_solver_selection_message(version_text, solver_name)

    assert version_text == "0.2.5"
    assert solver_name == "LBFGS"
    assert "0.2.5" in message
    assert "solver_name='LBFGS'" in message


def test_resolve_nemos_population_glm_solver_uses_jaxopt_after_0_2_5() -> None:
    version_text, solver_name = _resolve_nemos_population_glm_solver(
        SimpleNamespace(__version__="0.2.7")
    )

    message = _format_nemos_solver_selection_message(version_text, solver_name)

    assert version_text == "0.2.7"
    assert solver_name == "LBFGS[jaxopt]"
    assert "0.2.7" in message
    assert "solver_name='LBFGS[jaxopt]'" in message


def test_resolve_nemos_population_glm_solver_falls_back_to_package_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        ripple_glm_binned_module,
        "package_version",
        lambda name: "0.2.6",
    )

    version_text, solver_name = _resolve_nemos_population_glm_solver(SimpleNamespace())

    message = _format_nemos_solver_selection_message(version_text, solver_name)

    assert version_text == "0.2.6"
    assert solver_name == "LBFGS[jaxopt]"
    assert "0.2.6" in message
    assert "solver_name='LBFGS[jaxopt]'" in message
