from __future__ import annotations

from types import SimpleNamespace

import pytest

import v1ca1.spyglass.deploy as deploy_module


class _FakeAnalysisNwbfile:
    full_table_name = "`kyuv1ca1_nwbfile`.`analysis_nwbfile`"
    registration_calls = 0

    def register_with_spyglass(self) -> None:
        type(self).registration_calls += 1


def test_deploy_sets_prefix_activates_and_registers(monkeypatch) -> None:
    connection = object()
    fake_dj = SimpleNamespace(
        config={"custom": {"existing_setting": "preserved"}},
        conn=lambda: connection,
    )
    activation_calls = []
    tables = {
        "analysis_nwbfile": _FakeAnalysisNwbfile,
        "epoch_intervals": object(),
    }

    def activate(**kwargs):
        activation_calls.append(kwargs)
        return tables

    monkeypatch.setattr(
        deploy_module,
        "_load_runtime_dependencies",
        lambda: (fake_dj, activate),
    )
    _FakeAnalysisNwbfile.registration_calls = 0

    deployed = deploy_module.deploy_spyglass_tables()

    assert deployed is tables
    assert fake_dj.config["custom"] == {
        "existing_setting": "preserved",
        "database.prefix": "kyuv1ca1",
    }
    assert activation_calls == [
        {
            "schema_name": "kyuv1ca1",
            "analysis_nwbfile_schema_name": "kyuv1ca1_nwbfile",
            "connection": connection,
            "create_schema": True,
            "create_tables": True,
        }
    ]
    assert _FakeAnalysisNwbfile.registration_calls == 1


@pytest.mark.parametrize(
    "schema_name",
    ["nwbfile", "team_extra_nwbfile", "team_analysis"],
)
def test_deploy_rejects_invalid_analysis_schema_name(schema_name: str) -> None:
    with pytest.raises(ValueError, match="<prefix>_nwbfile"):
        deploy_module.deploy_spyglass_tables(
            analysis_nwbfile_schema_name=schema_name,
        )


def test_main_accepts_explicit_schema_names(monkeypatch, capsys) -> None:
    calls = []

    def deploy_spyglass_tables(**kwargs):
        calls.append(kwargs)
        return {"analysis_nwbfile": _FakeAnalysisNwbfile}

    monkeypatch.setattr(
        deploy_module,
        "deploy_spyglass_tables",
        deploy_spyglass_tables,
    )

    deploy_module.main(
        [
            "--schema-name",
            "testv1ca1",
            "--analysis-nwbfile-schema-name",
            "testv1ca1_nwbfile",
        ]
    )

    assert calls == [
        {
            "schema_name": "testv1ca1",
            "analysis_nwbfile_schema_name": "testv1ca1_nwbfile",
        }
    ]
    assert "Activated 1 table classes in 'testv1ca1'" in capsys.readouterr().out
