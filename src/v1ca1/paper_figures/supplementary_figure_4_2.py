from __future__ import annotations

"""Generate Supplementary Figure 4_2 without the cross-correlogram row."""

from collections.abc import Sequence
from pathlib import Path

from v1ca1.paper_figures.datasets import (
    DatasetId,
    get_processed_datasets,
)
from v1ca1.paper_figures.figure_3 import (
    DEFAULT_FIGURE_CACHE_DIR,
    DEFAULT_RIDGE_STRENGTH,
    DEFAULT_RIPPLE_WINDOW_OFFSET_S,
    DEFAULT_RIPPLE_WINDOW_S,
    add_aligned_panel_headers,
    filter_epoch_payloads,
    load_glm_dark_activity_devexp_tables,
    load_glm_epoch_summary_tables,
    load_glm_source_predictor_comparison_tables,
    load_pooled_ripple_heatmap_epoch_tables,
    plot_glm_behavior_association_panel,
    plot_glm_source_predictor_comparison_panel,
    plot_epoch_modulation_histogram_panel,
)
from v1ca1.paper_figures.style import (
    apply_paper_style,
    figure_size,
    label_axis,
    save_figure,
)
from v1ca1.paper_figures.supplementary_figure_4 import (
    DEFAULT_DATA_ROOT,
    DEFAULT_FIGURE_4_PANEL_HEIGHT_MM,
    DEFAULT_FIGURE_WIDTH_MM,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_RIPPLE_THRESHOLD_ZSCORE,
    DEFAULT_REGIONS,
    PANEL_A_EPOCH_ORDER,
    PANEL_C_EPOCH_ORDER,
    PANEL_C_SOURCE_COMPARISON_LIMITS,
    build_output_path,
    parse_arguments as parse_supplementary_figure_4_arguments,
    plot_glm_scatter_box_panel,
    resolve_primary_ripple_selection,
)


DEFAULT_OUTPUT_NAME = "supplementary_figure_4_2"
DEFAULT_FIGURE_HEIGHT_MM = DEFAULT_FIGURE_4_PANEL_HEIGHT_MM / 2.0
PANEL_A_HISTOGRAM_BOTTOM = 0.30
PANEL_A_HISTOGRAM_HEIGHT = 0.58


def make_supplementary_figure_4_2(
    *,
    data_root: Path,
    output_path: Path,
    datasets: Sequence[DatasetId],
    regions: Sequence[str],
    light_epoch: str | None,
    dark_epoch: str | None,
    sleep_epoch: str | None,
    ripple_threshold_zscore: float,
    ripple_selection_modes: Sequence[str],
    ripple_window_s: float,
    ridge_strength: float,
    dpi: int,
    ripple_window_offset_s: float = DEFAULT_RIPPLE_WINDOW_OFFSET_S,
    dark_movement_fr_cache_dir: Path | None = DEFAULT_FIGURE_CACHE_DIR,
    refresh_dark_movement_fr_cache: bool = False,
) -> Path:
    """Build and save Supplementary Figure 4_2."""
    import matplotlib.pyplot as plt

    ripple_selection = resolve_primary_ripple_selection(ripple_selection_modes)
    heatmap_epoch_tables = load_pooled_ripple_heatmap_epoch_tables(
        data_root,
        datasets,
        light_epoch=light_epoch,
        dark_epoch=dark_epoch,
        sleep_epoch=sleep_epoch,
        ripple_threshold_zscore=ripple_threshold_zscore,
    )
    glm_epoch_tables = load_glm_epoch_summary_tables(
        data_root,
        datasets,
        epoch_types=PANEL_C_EPOCH_ORDER,
        ripple_selection=ripple_selection,
        ripple_window_s=ripple_window_s,
        ripple_window_offset_s=ripple_window_offset_s,
        ridge_strength=ridge_strength,
    )
    source_comparison_payload = load_glm_source_predictor_comparison_tables(
        data_root,
        datasets,
        ripple_selection=ripple_selection,
        ripple_window_s=ripple_window_s,
        ripple_window_offset_s=ripple_window_offset_s,
        ridge_strength=ridge_strength,
    )
    behavior_payload = load_glm_dark_activity_devexp_tables(
        data_root,
        datasets,
        ripple_selection=ripple_selection,
        ripple_window_s=ripple_window_s,
        ripple_window_offset_s=ripple_window_offset_s,
        ridge_strength=ridge_strength,
        dark_movement_fr_cache_dir=dark_movement_fr_cache_dir,
        refresh_dark_movement_fr_cache=refresh_dark_movement_fr_cache,
    )
    panel_a_epoch_tables = filter_epoch_payloads(
        heatmap_epoch_tables,
        PANEL_A_EPOCH_ORDER,
    )
    panel_b_epoch_tables = filter_epoch_payloads(
        glm_epoch_tables,
        PANEL_C_EPOCH_ORDER,
    )

    apply_paper_style()
    fig = plt.figure(
        figsize=figure_size(DEFAULT_FIGURE_WIDTH_MM, DEFAULT_FIGURE_HEIGHT_MM),
        constrained_layout=True,
    )
    outer_grid = fig.add_gridspec(nrows=1, ncols=2, width_ratios=[1.0, 4.0])
    modulation_ax = fig.add_subplot(outer_grid[0, 0])
    lower_grid = outer_grid[0, 1].subgridspec(
        nrows=2,
        ncols=8,
        height_ratios=[0.46, 0.54],
        width_ratios=[1.0] * 8,
    )
    panel_b_ax = fig.add_subplot(lower_grid[:, :5])
    source_comparison_ax = fig.add_subplot(lower_grid[0, 5:])
    behavior_ax = fig.add_subplot(lower_grid[1, 5:])

    plot_epoch_modulation_histogram_panel(
        modulation_ax,
        panel_a_epoch_tables,
        regions=regions,
        bottom=PANEL_A_HISTOGRAM_BOTTOM,
        height=PANEL_A_HISTOGRAM_HEIGHT,
    )
    modulation_ax.set_title("Ripple modulation index", fontsize=7.2, pad=2)

    plot_glm_scatter_box_panel(
        panel_b_ax,
        panel_b_epoch_tables,
    )
    panel_b_ax.set_title(
        "Predicting V1 activity during ripples\nwith CA1 activity",
        fontsize=7.2,
        pad=2,
    )
    plot_glm_source_predictor_comparison_panel(
        source_comparison_ax,
        source_comparison_payload,
        include_per_animal=False,
        include_pooled=True,
        compact_labels=True,
        show_color_note=False,
        axis_limits=PANEL_C_SOURCE_COMPARISON_LIMITS,
    )
    source_comparison_title = "CA1 spike vector vs.\nmean CA1 activity"
    source_comparison_ax.set_title(source_comparison_title, fontsize=7.2, pad=2)
    plot_glm_behavior_association_panel(
        behavior_ax,
        behavior_payload,
        show_note=False,
        show_significance_marker=False,
    )
    behavior_ax.set_title(
        "Relationship to dark-active DPP cells",
        fontsize=7.2,
        pad=2,
    )
    label_axis(behavior_ax, "D", x=-0.06, y=1.04)

    fig.canvas.draw()
    fig.set_constrained_layout(False)
    add_aligned_panel_headers(
        fig,
        (modulation_ax, panel_b_ax, source_comparison_ax),
        labels=("A", "B", "C"),
        titles=(
            "Ripple modulation index",
            "Predicting V1 activity during ripples\nwith CA1 activity",
            source_comparison_title,
        ),
        label_x_offsets=(-0.08, -0.06, -0.06),
        fontsize=7.2,
    )

    save_figure(fig, output_path, dpi=dpi)
    plt.close(fig)
    for missing in source_comparison_payload["missing_artifacts"]:
        print(
            "Supplementary Figure 4_2 source-comparison missing "
            f"{missing['artifact']} for {missing['animal_name']} "
            f"{missing['date']} {missing['epoch']} "
            f"({missing['source_predictor_mode']}): {missing['path']}"
        )
    for missing in behavior_payload["missing_artifacts"]:
        print(
            "Supplementary Figure 4_2 behavior-association missing "
            f"{missing['artifact']} for {missing['animal_name']} "
            f"{missing['date']} {missing['epoch']}: {missing.get('path', '')}"
        )
    print(f"Saved Supplementary Figure 4_2 to {output_path}")
    return output_path


def parse_arguments(argv: Sequence[str] | None = None):
    """Parse command-line arguments for Supplementary Figure 4_2 generation."""
    return parse_supplementary_figure_4_arguments(
        argv,
        default_output_name=DEFAULT_OUTPUT_NAME,
        description="Generate Supplementary Figure 4_2 without the cross-correlogram row.",
        include_xcorr_options=False,
    )


def main(argv: Sequence[str] | None = None) -> None:
    """Run Supplementary Figure 4_2 generation."""
    args = parse_arguments(argv)
    datasets = args.dataset if args.dataset is not None else get_processed_datasets()
    regions = tuple(args.region) if args.region is not None else DEFAULT_REGIONS
    output_path = build_output_path(
        args.output_dir,
        args.output_name,
        args.output_format,
    )
    make_supplementary_figure_4_2(
        data_root=args.data_root,
        output_path=output_path,
        datasets=datasets,
        regions=regions,
        light_epoch=args.light_epoch,
        dark_epoch=args.dark_epoch,
        sleep_epoch=args.sleep_epoch,
        ripple_threshold_zscore=args.ripple_threshold_zscore,
        ripple_selection_modes=tuple(args.ripple_selection),
        ripple_window_s=args.ripple_window_s,
        ripple_window_offset_s=args.ripple_window_offset_s,
        ridge_strength=args.ridge_strength,
        dark_movement_fr_cache_dir=args.dark_movement_fr_cache_dir,
        refresh_dark_movement_fr_cache=args.refresh_dark_movement_fr_cache,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
