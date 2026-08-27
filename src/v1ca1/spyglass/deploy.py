"""Create and register the project-owned Spyglass DataJoint tables."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from v1ca1.spyglass import table_specs


def _analysis_prefix(analysis_nwbfile_schema_name: str) -> str:
    """Return the custom prefix from one ``<prefix>_nwbfile`` schema name."""
    if analysis_nwbfile_schema_name.count("_") != 1:
        raise ValueError(
            "AnalysisNwbfile schema must have the form '<prefix>_nwbfile'."
        )
    prefix, suffix = analysis_nwbfile_schema_name.split("_", 1)
    if not prefix or suffix != "nwbfile":
        raise ValueError(
            "AnalysisNwbfile schema must have the form '<prefix>_nwbfile'."
        )
    return prefix


def _load_runtime_dependencies() -> tuple[Any, Callable[..., Mapping[str, Any]]]:
    """Import DataJoint and activation only when deployment is requested."""
    import datajoint as dj

    from v1ca1.spyglass.tables import activate

    return dj, activate


def deploy_spyglass_tables(
    *,
    schema_name: str = table_specs.DEFAULT_SCHEMA_NAME,
    analysis_nwbfile_schema_name: str = (
        table_specs.DEFAULT_ANALYSIS_NWBFILE_SCHEMA_NAME
    ),
) -> Mapping[str, Any]:
    """Create missing project tables and register the analysis-NWB table.

    Re-running this deployment is safe. It does not insert parameter defaults,
    ingest source rows, or populate computed tables.
    """
    prefix = _analysis_prefix(analysis_nwbfile_schema_name)
    dj, activate = _load_runtime_dependencies()

    custom_config = dict(dj.config.get("custom", {}))
    custom_config["database.prefix"] = prefix
    dj.config["custom"] = custom_config

    tables = activate(
        schema_name=schema_name,
        analysis_nwbfile_schema_name=analysis_nwbfile_schema_name,
        connection=dj.conn(),
        create_schema=True,
        create_tables=True,
    )
    analysis_nwbfile_table = tables.get("analysis_nwbfile")
    if analysis_nwbfile_table is None:
        raise RuntimeError("Activation did not return the AnalysisNwbfile table.")
    analysis_nwbfile_table().register_with_spyglass()
    return tables


def _parser() -> argparse.ArgumentParser:
    """Build the explicit Spyglass deployment CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--schema-name",
        default=table_specs.DEFAULT_SCHEMA_NAME,
    )
    parser.add_argument(
        "--analysis-nwbfile-schema-name",
        default=table_specs.DEFAULT_ANALYSIS_NWBFILE_SCHEMA_NAME,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Create and register the configured project-owned Spyglass tables."""
    args = _parser().parse_args(argv)
    tables = deploy_spyglass_tables(
        schema_name=args.schema_name,
        analysis_nwbfile_schema_name=args.analysis_nwbfile_schema_name,
    )
    full_table_name = tables["analysis_nwbfile"].full_table_name
    print(
        f"Activated {len(tables)} table classes in {args.schema_name!r} and "
        f"registered {full_table_name}."
    )


if __name__ == "__main__":
    main()


__all__ = ["deploy_spyglass_tables", "main"]
