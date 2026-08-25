from __future__ import annotations

"""Plot pre-merge template-extremum depths for curated units in one region."""

import argparse
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from v1ca1.helper.run_logging import write_run_log
from v1ca1.helper.session import DEFAULT_NWB_ROOT
from v1ca1.spikesorting._region_assignment import (
    get_region_names_from_rules,
    get_session_region_assignment_rules,
    index_rules_by_probe_shank,
)
from v1ca1.spikesorting.consolidate_sorting import (
    DEFAULT_ANALYSIS_ROOT,
    DEFAULT_CURATION_ROOT,
    assign_regions_to_depths,
    build_curated_unit_groups,
    get_analysis_path,
    get_sorting_curation_path,
    get_template_extremum_channel_ids,
    load_curation_payload,
    load_shank_channel_depths,
    load_sorting_analyzer,
)


DEFAULT_REGION = "ca1"
DEFAULT_OUTPUT_FORMAT = "png"
OUTPUT_FORMATS = ("png", "pdf", "svg")
PREMERGE_DEPTH_TABLE_COLUMNS = (
    "animal_name",
    "date",
    "region",
    "probe_idx",
    "shank_idx",
    "original_unit_id",
    "extremum_channel_id",
    "extremum_depth_um",
    "curation_group_index",
    "curation_group_size",
    "is_merge_component",
    "curation_labels",
)


def get_rule_region_names(rule: dict[str, Any]) -> set[str]:
    """Return the region names referenced by one probe/shank rule."""
    if str(rule["mode"]) == "all":
        return {str(rule["region"])}
    return {
        str(rule["below_region"]),
        str(rule["above_or_equal_region"]),
    }


def build_premerge_depth_rows(
    *,
    animal_name: str,
    date: str,
    region: str,
    probe_idx: int,
    shank_idx: int,
    original_unit_ids: Iterable[int],
    extremum_channel_ids: dict[int, int],
    channel_depths: dict[int, float],
    curation_payload: dict[str, Any],
    rule: dict[str, Any],
) -> list[dict[str, Any]]:
    """Build rows for retained original units without collapsing merge groups."""
    curated_unit_groups = build_curated_unit_groups(
        original_unit_ids=original_unit_ids,
        curation_payload=curation_payload,
    )
    labels_by_unit = curation_payload.get("labelsByUnit", {})
    rows: list[dict[str, Any]] = []
    for group_index, unit_group in enumerate(curated_unit_groups):
        group_size = len(unit_group)
        for original_unit_id in unit_group:
            original_unit_id = int(original_unit_id)
            if original_unit_id not in extremum_channel_ids:
                raise ValueError(
                    "Template extremum channel is missing for original unit "
                    f"{original_unit_id} in probe {probe_idx} shank {shank_idx}."
                )
            extremum_channel_id = int(extremum_channel_ids[original_unit_id])
            if extremum_channel_id not in channel_depths:
                raise ValueError(
                    f"Channel {extremum_channel_id} has no depth coordinate for "
                    f"probe {probe_idx} shank {shank_idx}."
                )
            extremum_depth_um = float(channel_depths[extremum_channel_id])
            assigned_region = assign_regions_to_depths(
                extremum_depths=[extremum_depth_um],
                rule=rule,
            )[0]
            if assigned_region != region:
                continue

            unit_labels = labels_by_unit.get(str(original_unit_id), [])
            rows.append(
                {
                    "animal_name": str(animal_name),
                    "date": str(date),
                    "region": str(region),
                    "probe_idx": int(probe_idx),
                    "shank_idx": int(shank_idx),
                    "original_unit_id": original_unit_id,
                    "extremum_channel_id": extremum_channel_id,
                    "extremum_depth_um": extremum_depth_um,
                    "curation_group_index": int(group_index),
                    "curation_group_size": int(group_size),
                    "is_merge_component": bool(group_size > 1),
                    "curation_labels": ";".join(
                        sorted(str(label) for label in unit_labels)
                    ),
                }
            )
    return rows


def load_premerge_extremum_depth_table(
    *,
    animal_name: str,
    date: str,
    region: str = DEFAULT_REGION,
    analysis_root: Path = DEFAULT_ANALYSIS_ROOT,
    curation_root: Path = DEFAULT_CURATION_ROOT,
    nwb_root: Path = DEFAULT_NWB_ROOT,
):
    """Load retained pre-merge unit depths across one session region."""
    import pandas as pd

    session_rules = get_session_region_assignment_rules(
        animal_name=animal_name,
        date=date,
    )
    known_regions = get_region_names_from_rules(session_rules)
    if region not in known_regions:
        raise ValueError(
            f"Region {region!r} is not configured for {animal_name} {date}. "
            f"Expected one of {known_regions!r}."
        )

    rows: list[dict[str, Any]] = []
    rules_by_probe_shank = index_rules_by_probe_shank(session_rules)
    for (probe_idx, shank_idx), rule in sorted(rules_by_probe_shank.items()):
        if region not in get_rule_region_names(rule):
            continue
        print(f"Loading probe {probe_idx} shank {shank_idx}...")
        sorting_analyzer = load_sorting_analyzer(
            animal_name=animal_name,
            date=date,
            probe_idx=probe_idx,
            shank_idx=shank_idx,
            analysis_root=analysis_root,
        )
        original_unit_ids = [
            int(unit_id) for unit_id in sorting_analyzer.sorting.get_unit_ids()
        ]
        extremum_channel_ids = get_template_extremum_channel_ids(sorting_analyzer)
        channel_depths = load_shank_channel_depths(
            animal_name=animal_name,
            date=date,
            probe_idx=probe_idx,
            shank_idx=shank_idx,
            nwb_root=nwb_root,
        )
        curation_payload = load_curation_payload(
            get_sorting_curation_path(
                animal_name=animal_name,
                date=date,
                probe_idx=probe_idx,
                shank_idx=shank_idx,
                curation_root=curation_root,
            )
        )
        rows.extend(
            build_premerge_depth_rows(
                animal_name=animal_name,
                date=date,
                region=region,
                probe_idx=probe_idx,
                shank_idx=shank_idx,
                original_unit_ids=original_unit_ids,
                extremum_channel_ids=extremum_channel_ids,
                channel_depths=channel_depths,
                curation_payload=curation_payload,
                rule=rule,
            )
        )

    table = pd.DataFrame(rows, columns=PREMERGE_DEPTH_TABLE_COLUMNS)
    if table.empty:
        raise ValueError(
            f"No retained pre-merge {region} units found for {animal_name} {date}."
        )
    return table.sort_values(
        ["probe_idx", "shank_idx", "original_unit_id"],
        kind="stable",
        ignore_index=True,
    )


def get_depth_bin_edges(depths: np.ndarray) -> np.ndarray:
    """Return contact-centered histogram edges for discrete channel depths."""
    unique_depths = np.unique(np.asarray(depths, dtype=float))
    unique_depths = unique_depths[np.isfinite(unique_depths)]
    if unique_depths.size == 0:
        raise ValueError("Cannot plot an empty or non-finite depth distribution.")
    if unique_depths.size == 1:
        return np.asarray([unique_depths[0] - 0.5, unique_depths[0] + 0.5])
    spacing = float(np.min(np.diff(unique_depths)))
    return np.concatenate(
        (
            [unique_depths[0] - 0.5 * spacing],
            unique_depths + 0.5 * spacing,
        )
    )


def plot_premerge_extremum_depth_distribution(
    table: Any,
    output_path: Path,
    *,
    dpi: int,
) -> Path:
    """Plot one raw-count depth distribution per probe and shank."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    probe_indices = sorted(int(value) for value in table["probe_idx"].unique())
    shank_indices = sorted(int(value) for value in table["shank_idx"].unique())
    fig, axes = plt.subplots(
        len(probe_indices),
        len(shank_indices),
        figsize=(2.45 * len(shank_indices), 3.15 * len(probe_indices)),
        sharex="row",
        sharey="row",
        squeeze=False,
    )
    colors = plt.get_cmap("tab10").colors
    for row_index, probe_idx in enumerate(probe_indices):
        probe_rows = table.loc[table["probe_idx"] == probe_idx]
        depth_bin_edges = get_depth_bin_edges(
            probe_rows["extremum_depth_um"].to_numpy(dtype=float)
        )
        for column_index, shank_idx in enumerate(shank_indices):
            axis = axes[row_index, column_index]
            values = probe_rows.loc[
                probe_rows["shank_idx"] == shank_idx,
                "extremum_depth_um",
            ].to_numpy(dtype=float)
            if values.size == 0:
                axis.axis("off")
                continue
            axis.hist(
                values,
                bins=depth_bin_edges,
                orientation="horizontal",
                color=colors[shank_idx % len(colors)],
                edgecolor="white",
                linewidth=0.5,
            )
            axis.set_title(f"Shank {shank_idx} (n={values.size})")
            axis.set_xlim(left=0.0)
            axis.xaxis.set_major_locator(MaxNLocator(integer=True))
            axis.grid(axis="x", color="0.88", linewidth=0.6)
            if column_index == 0:
                axis.set_ylabel(
                    f"Probe {probe_idx}\nExtremum-channel depth (µm)"
                )
    animal_name = str(table["animal_name"].iloc[0])
    date = str(table["date"].iloc[0])
    region = str(table["region"].iloc[0]).upper()
    fig.suptitle(
        f"{animal_name} {date} {region}: pre-merge template extrema",
        fontsize=11,
    )
    fig.supxlabel("Number of pre-merge units")
    fig.tight_layout(rect=(0.0, 0.04, 1.0, 0.97))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def generate_premerge_extremum_depth_outputs(
    *,
    animal_name: str,
    date: str,
    region: str = DEFAULT_REGION,
    analysis_root: Path = DEFAULT_ANALYSIS_ROOT,
    curation_root: Path = DEFAULT_CURATION_ROOT,
    nwb_root: Path = DEFAULT_NWB_ROOT,
    output_dir: Path | None = None,
    output_name: str | None = None,
    output_format: str = DEFAULT_OUTPUT_FORMAT,
    dpi: int = 300,
) -> dict[str, Path]:
    """Load pre-merge depths and save an audit table and distribution plot."""
    analysis_path = get_analysis_path(animal_name, date, analysis_root)
    output_dir = analysis_path / "figs" if output_dir is None else Path(output_dir)
    output_name = (
        f"{animal_name}_{date}_{region}_premerge_extremum_channel_depths"
        if output_name is None
        else str(output_name)
    )
    table = load_premerge_extremum_depth_table(
        animal_name=animal_name,
        date=date,
        region=region,
        analysis_root=analysis_root,
        curation_root=curation_root,
        nwb_root=nwb_root,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    table_path = output_dir / f"{output_name}.parquet"
    figure_path = output_dir / f"{output_name}.{output_format}"
    table.to_parquet(table_path, index=False)
    plot_premerge_extremum_depth_distribution(
        table,
        figure_path,
        dpi=dpi,
    )

    for (probe_idx, shank_idx), rows in table.groupby(
        ["probe_idx", "shank_idx"], sort=True
    ):
        depths = rows["extremum_depth_um"].to_numpy(dtype=float)
        print(
            f"probe {probe_idx} shank {shank_idx}: n={depths.size}, "
            f"median={np.median(depths):.1f} µm, "
            f"range=[{np.min(depths):.1f}, {np.max(depths):.1f}] µm"
        )

    outputs = {
        "premerge_extremum_depth_table": table_path,
        "premerge_extremum_depth_figure": figure_path,
    }
    log_path = write_run_log(
        analysis_path=analysis_path,
        script_name="v1ca1.spikesorting.plot_extremum_channel_depths",
        parameters={
            "animal_name": animal_name,
            "date": date,
            "region": region,
            "analysis_root": analysis_root,
            "curation_root": curation_root,
            "nwb_root": nwb_root,
            "output_dir": output_dir,
            "output_name": output_name,
            "output_format": output_format,
            "dpi": dpi,
            "template_peak_sign": "neg",
            "merge_handling": "original pre-merge template components",
        },
        outputs=outputs,
    )
    print(f"Saved depth table to {table_path}")
    print(f"Saved depth distribution to {figure_path}")
    print(f"Saved run metadata to {log_path}")
    return outputs


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the depth-distribution plot."""
    parser = argparse.ArgumentParser(
        description=(
            "Plot pre-merge template-extremum depths for retained units in one region."
        )
    )
    parser.add_argument("--animal-name", required=True, help="Animal name")
    parser.add_argument(
        "--date",
        required=True,
        help="Recording date in YYYYMMDD format",
    )
    parser.add_argument(
        "--region",
        default=DEFAULT_REGION,
        help=f"Region to plot. Default: {DEFAULT_REGION}",
    )
    parser.add_argument(
        "--analysis-root",
        type=Path,
        default=DEFAULT_ANALYSIS_ROOT,
        help=f"Base directory for analysis outputs. Default: {DEFAULT_ANALYSIS_ROOT}",
    )
    parser.add_argument(
        "--curation-root",
        type=Path,
        default=DEFAULT_CURATION_ROOT,
        help=f"Base directory for sorting curations. Default: {DEFAULT_CURATION_ROOT}",
    )
    parser.add_argument(
        "--nwb-root",
        type=Path,
        default=DEFAULT_NWB_ROOT,
        help=f"Base directory for source NWB files. Default: {DEFAULT_NWB_ROOT}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Default: <analysis-root>/<animal>/<date>/figs",
    )
    parser.add_argument(
        "--output-name",
        default=None,
        help="Output basename. Default: derived from animal, date, and region.",
    )
    parser.add_argument(
        "--format",
        dest="output_format",
        choices=OUTPUT_FORMATS,
        default=DEFAULT_OUTPUT_FORMAT,
        help=f"Figure format. Default: {DEFAULT_OUTPUT_FORMAT}",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Raster output resolution. Default: 300",
    )
    return parser.parse_args()


def main() -> None:
    """Run the CLI entrypoint."""
    args = parse_arguments()
    generate_premerge_extremum_depth_outputs(
        animal_name=args.animal_name,
        date=args.date,
        region=args.region,
        analysis_root=args.analysis_root,
        curation_root=args.curation_root,
        nwb_root=args.nwb_root,
        output_dir=args.output_dir,
        output_name=args.output_name,
        output_format=args.output_format,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
