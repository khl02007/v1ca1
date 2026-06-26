from __future__ import annotations

"""Generate Figure 4_3 with DPP overlap in the behavior-association panel."""

from collections.abc import Sequence

from v1ca1.paper_figures.datasets import get_processed_datasets
from v1ca1.paper_figures.figure_4 import (
    DEFAULT_OUTPUT_NAME as FIGURE_4_OUTPUT_NAME,
    DEFAULT_REGIONS,
    build_output_path,
    make_figure_4,
    parse_arguments as parse_figure_4_arguments,
)


DEFAULT_OUTPUT_NAME = "figure_4_3"
PANEL_D_TUNING_SIMILARITY_METRIC = "absolute_overlap"


def main(argv: Sequence[str] | None = None) -> None:
    """Run Figure 4_3 generation."""
    args = parse_figure_4_arguments(argv)
    if args.output_name == FIGURE_4_OUTPUT_NAME:
        args.output_name = DEFAULT_OUTPUT_NAME

    datasets = args.dataset if args.dataset is not None else get_processed_datasets()
    regions = tuple(args.region) if args.region is not None else DEFAULT_REGIONS
    output_path = build_output_path(
        args.output_dir,
        args.output_name,
        args.output_format,
    )
    make_figure_4(
        data_root=args.data_root,
        output_path=output_path,
        datasets=datasets,
        example_dataset=args.example_dataset,
        xcorr_dataset=args.xcorr_dataset,
        xcorr_state=args.xcorr_state,
        xcorr_top_ca1_units=args.xcorr_top_ca1_units,
        xcorr_bin_size_s=args.xcorr_bin_size_s,
        xcorr_max_lag_s=args.xcorr_max_lag_s,
        xcorr_display_vmax=args.xcorr_display_vmax,
        light_epoch=args.light_epoch,
        dark_epoch=args.dark_epoch,
        sleep_epoch=args.sleep_epoch,
        regions=regions,
        ripple_threshold_zscore=args.ripple_threshold_zscore,
        ripple_window_s=args.ripple_window_s,
        ripple_window_offset_s=args.ripple_window_offset_s,
        ripple_selection=args.ripple_selection,
        ridge_strength=args.ridge_strength,
        dark_movement_fr_cache_dir=args.dark_movement_fr_cache_dir,
        refresh_dark_movement_fr_cache=args.refresh_dark_movement_fr_cache,
        refresh_panel_b_schematic_cache=args.refresh_panel_b_schematic_cache,
        dpi=args.dpi,
        panel_d_tuning_similarity_metric=PANEL_D_TUNING_SIMILARITY_METRIC,
    )


if __name__ == "__main__":
    main()
