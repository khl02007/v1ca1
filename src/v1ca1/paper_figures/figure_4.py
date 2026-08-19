"""Generate Figure 4 from Figure 3 panels B, C, and E."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from v1ca1.paper_figures import figure_3_old as _figure_3
from v1ca1.paper_figures.datasets import DatasetId, get_processed_datasets
from v1ca1.paper_figures.style import PANEL_LABEL_KWARGS


DEFAULT_OUTPUT_DIR = _figure_3.DEFAULT_OUTPUT_DIR
DEFAULT_OUTPUT_NAME = "figure_4"
DEFAULT_OUTPUT_FORMAT = _figure_3.DEFAULT_OUTPUT_FORMAT
DEFAULT_FIGURE_WIDTH_MM = _figure_3.DEFAULT_FIGURE_WIDTH_MM
DEFAULT_FIGURE_HEIGHT_MM = 75.0
FIGURE_FORMATS = _figure_3.FIGURE_FORMATS
PANEL_WIDTH_RATIOS = (1.0, 2.0, 1.0)
PANEL_LABELS = ("A", "B", "C", "D", "E", "F")
PANEL_HEADER_LABELS = (PANEL_LABELS[0], "", PANEL_LABELS[4])
PANEL_B_COLUMN_LABELS = PANEL_LABELS[1:4]
PANEL_TITLES = (
    "CA1-V1 correlogram\nduring ripples",
    "Predicting V1 activity during ripples with CA1 activity",
    "Relationship to dark activity",
)
PANEL_F_TITLE = "Relationship to path-invariance"
PANEL_F_XLABEL = "Dark path-invariance index"
PANEL_HEADER_LABEL_X_OFFSETS = (-0.18, 0.0, -0.08)
PANEL_E_SINGLE_EPOCH_COLUMN_BOUNDS = (
    (0.02, 0.38),
    (0.53, 0.39),
    (0.0, 0.0),
)
PANEL_E_SINGLE_EPOCH_AXIS_VERTICAL_BOUNDS = (0.58, 0.29)
PANEL_E_DPPI_AXIS_BOUNDS = (0.23, 0.10, 0.54, 0.29)


def _add_aligned_panel_e_f_headers(
    fig: Any,
    panel_f_axis: Any,
) -> None:
    """Align the E/F labels and each panel's title in figure coordinates."""
    panel_e_label = next(
        text for text in reversed(fig.texts) if text.get_text() == PANEL_LABELS[4]
    )
    panel_e_title = next(
        text for text in reversed(fig.texts) if text.get_text() == PANEL_TITLES[2]
    )
    panel_e_label_x, _panel_e_label_y = panel_e_label.get_position()
    panel_e_title_x, panel_e_header_y = panel_e_title.get_position()
    panel_e_label.set_position((panel_e_label_x, panel_e_header_y))
    panel_e_title.set_position((panel_e_title_x, panel_e_header_y))

    panel_f_title_display = panel_f_axis.title.get_transform().transform(
        panel_f_axis.title.get_position()
    )
    panel_f_header_y = fig.transFigure.inverted().transform(
        panel_f_title_display
    )[1]
    panel_f_axis.set_title("")
    panel_f_box = panel_f_axis.get_position()
    label_kwargs = PANEL_LABEL_KWARGS.copy()
    label_kwargs["va"] = "top"
    fig.text(
        panel_e_label_x,
        panel_f_header_y,
        PANEL_LABELS[5],
        transform=fig.transFigure,
        **label_kwargs,
    )
    fig.text(
        (panel_f_box.x0 + panel_f_box.x1) / 2.0,
        panel_f_header_y,
        PANEL_F_TITLE,
        ha="center",
        va="top",
        fontsize=7.2,
        transform=fig.transFigure,
    )


def _align_panel_f_histogram_to_panel_d_bottom(
    panel_d_parent_axis: Any,
    panel_f_axis: Any,
) -> None:
    """Align Panel F's x-axis with the bottom plot in Panel D."""
    panel_d_bottom_axes = [
        child_axis
        for child_axis in panel_d_parent_axis.child_axes
        if child_axis.get_xlabel() == "Deviance explained"
    ]
    if len(panel_d_bottom_axes) != 1:
        raise RuntimeError(
            "Expected one Panel D bottom axis labeled 'Deviance explained'."
        )

    panel_d_bottom = panel_d_bottom_axes[0].get_position().y0
    panel_f_box = panel_f_axis.get_position()
    panel_f_axis.set_axes_locator(None)
    panel_f_axis.set_position(
        (
            panel_f_box.x0,
            panel_d_bottom,
            panel_f_box.width,
            panel_f_box.height,
        )
    )


def build_output_path(
    output_dir: Path,
    output_name: str,
    output_format: str,
) -> Path:
    """Return the Figure 4 output path for a supported format."""
    return _figure_3.build_output_path(output_dir, output_name, output_format)


def load_figure_4_panel_data(
    *,
    data_root: Path,
    datasets: Sequence[DatasetId],
    example_dataset: DatasetId,
    light_epoch: str | None,
    dark_epoch: str | None,
    sleep_epoch: str | None,
    ripple_threshold_zscore: float | None,
    ripple_window_s: float,
    ripple_window_offset_s: float,
    ripple_selection: str,
    ridge_strength: float,
    dark_movement_fr_cache_dir: Path | None,
    refresh_dark_movement_fr_cache: bool,
    refresh_panel_b_schematic_cache: bool,
    panel_d_tuning_similarity_metric: str = (
        _figure_3.DEFAULT_PANEL_D_TUNING_SIMILARITY_METRIC
    ),
) -> dict[str, Any]:
    """Load only the data used by retained Figure 3 panels B, C, and E."""
    dataset_ids = tuple(datasets)
    glm_epoch_tables = _figure_3.load_glm_epoch_summary_tables(
        data_root,
        dataset_ids,
        light_epoch=light_epoch,
        dark_epoch=dark_epoch,
        sleep_epoch=sleep_epoch,
        epoch_types=_figure_3.PANEL_C_EPOCH_ORDER,
        ripple_window_s=ripple_window_s,
        ripple_window_offset_s=ripple_window_offset_s,
        ripple_selection=ripple_selection,
        ridge_strength=ridge_strength,
    )

    schematic_animal, schematic_date, schematic_epoch = (
        _figure_3.normalize_dataset_id(_figure_3.DEFAULT_PANEL_B_SCHEMATIC_DATASET)
    )
    ripple_schematic_trace: dict[str, Any] | None = None
    try:
        ripple_schematic_trace = _figure_3.load_or_build_panel_b_schematic_example(
            data_root,
            cache_dir=_figure_3.DEFAULT_FIGURE_CACHE_DIR,
            animal_name=schematic_animal,
            date=schematic_date,
            epoch=schematic_epoch,
            ripple_threshold_zscore=ripple_threshold_zscore,
            time_before_s=_figure_3.DEFAULT_PANEL_B_SCHEMATIC_TIME_BEFORE_S,
            time_after_s=_figure_3.DEFAULT_PANEL_B_SCHEMATIC_TIME_AFTER_S,
            n_units_per_region=_figure_3.DEFAULT_PANEL_B_SCHEMATIC_N_UNITS_PER_REGION,
            target_ripple_duration_s=(
                _figure_3.DEFAULT_PANEL_B_SCHEMATIC_TARGET_DURATION_S
            ),
            refresh_cache=refresh_panel_b_schematic_cache,
        )
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(
            "Panel C using fallback schematic spikes because the real-spike cache "
            f"could not be built for {schematic_animal} {schematic_date} "
            f"{schematic_epoch}: {exc}"
        )
        example_animal, example_date, example_epoch = (
            _figure_3.normalize_dataset_id(example_dataset)
        )
        try:
            ripple_schematic_trace = _figure_3.load_example_ripple_lfp_trace(
                data_root,
                animal_name=example_animal,
                date=example_date,
                epoch=example_epoch,
                ripple_threshold_zscore=ripple_threshold_zscore,
                time_before_s=_figure_3.DEFAULT_PANEL_B_SCHEMATIC_TIME_BEFORE_S,
                time_after_s=_figure_3.DEFAULT_PANEL_B_SCHEMATIC_TIME_AFTER_S,
            )
        except (FileNotFoundError, KeyError, ValueError) as fallback_exc:
            print(
                "Panel C using fully synthetic schematic because saved ripple-band "
                f"LFP was unavailable for {example_animal} {example_date} "
                f"{example_epoch}: {fallback_exc}"
            )

    panel_c_prediction_examples: list[dict[str, Any]] = []
    try:
        panel_c_prediction_examples = _figure_3.load_panel_b_prediction_examples(
            data_root,
            ripple_window_s=ripple_window_s,
            ripple_window_offset_s=ripple_window_offset_s,
            ripple_selection=ripple_selection,
            ridge_strength=ridge_strength,
        )
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(f"Panel C prediction examples unavailable: {exc}")

    panel_e_behavior_payload = _figure_3.load_glm_dark_activity_devexp_tables(
        data_root,
        dataset_ids,
        light_epoch=light_epoch,
        dark_epoch=dark_epoch,
        sleep_epoch=sleep_epoch,
        ripple_window_s=ripple_window_s,
        ripple_window_offset_s=ripple_window_offset_s,
        ripple_selection=ripple_selection,
        ridge_strength=ridge_strength,
        dark_movement_fr_cache_dir=dark_movement_fr_cache_dir,
        refresh_dark_movement_fr_cache=refresh_dark_movement_fr_cache,
        tuning_similarity_metric=panel_d_tuning_similarity_metric,
    )

    panel_b_xcorr_payload: dict[str, Any] | None = None
    xcorr_animal, xcorr_date, xcorr_epoch = _figure_3.normalize_dataset_id(
        _figure_3.DEFAULT_XCORR_DATASET
    )
    try:
        panel_b_xcorr_payload = _figure_3.load_top_ca1_xcorr_panel_data(
            data_root,
            animal_name=xcorr_animal,
            date=xcorr_date,
            epoch=xcorr_epoch,
            state=_figure_3.DEFAULT_XCORR_STATE,
            top_n_ca1_units=_figure_3.DEFAULT_XCORR_TOP_CA1_UNITS,
            bin_size_s=_figure_3.DEFAULT_XCORR_BIN_SIZE_S,
            max_lag_s=_figure_3.DEFAULT_XCORR_MAX_LAG_S,
            display_vmax=_figure_3.DEFAULT_XCORR_DISPLAY_VMAX,
        )
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(
            "Panel B xcorr unavailable for "
            f"{xcorr_animal} {xcorr_date} {xcorr_epoch}: {exc}"
        )

    return {
        "panel_b_xcorr_payload": panel_b_xcorr_payload,
        "panel_c_epoch_tables": _figure_3.filter_epoch_payloads(
            glm_epoch_tables,
            _figure_3.PANEL_C_EPOCH_ORDER,
        ),
        "panel_c_ripple_trace": ripple_schematic_trace,
        "panel_c_prediction_examples": panel_c_prediction_examples,
        "panel_e_behavior_payload": panel_e_behavior_payload,
    }


def make_figure_4(
    *,
    data_root: Path,
    output_path: Path,
    datasets: Sequence[DatasetId],
    example_dataset: DatasetId,
    light_epoch: str | None,
    dark_epoch: str | None,
    sleep_epoch: str | None,
    ripple_threshold_zscore: float | None,
    ripple_window_s: float,
    ripple_window_offset_s: float,
    ripple_selection: str,
    ridge_strength: float,
    dark_movement_fr_cache_dir: Path | None,
    refresh_dark_movement_fr_cache: bool,
    refresh_panel_b_schematic_cache: bool,
    dpi: int,
    panel_d_tuning_similarity_metric: str = (
        _figure_3.DEFAULT_PANEL_D_TUNING_SIMILARITY_METRIC
    ),
) -> Path:
    """Build and save Figure 4 as a one-row B, C, and E composition."""
    import matplotlib.pyplot as plt

    panel_data = load_figure_4_panel_data(
        data_root=data_root,
        datasets=datasets,
        example_dataset=example_dataset,
        light_epoch=light_epoch,
        dark_epoch=dark_epoch,
        sleep_epoch=sleep_epoch,
        ripple_threshold_zscore=ripple_threshold_zscore,
        ripple_window_s=ripple_window_s,
        ripple_window_offset_s=ripple_window_offset_s,
        ripple_selection=ripple_selection,
        ridge_strength=ridge_strength,
        dark_movement_fr_cache_dir=dark_movement_fr_cache_dir,
        refresh_dark_movement_fr_cache=refresh_dark_movement_fr_cache,
        refresh_panel_b_schematic_cache=refresh_panel_b_schematic_cache,
        panel_d_tuning_similarity_metric=panel_d_tuning_similarity_metric,
    )

    _figure_3.apply_paper_style()
    fig = plt.figure(
        figsize=_figure_3.figure_size(
            DEFAULT_FIGURE_WIDTH_MM,
            DEFAULT_FIGURE_HEIGHT_MM,
        ),
        constrained_layout=True,
    )
    outer_grid = fig.add_gridspec(
        nrows=1,
        ncols=3,
        width_ratios=PANEL_WIDTH_RATIOS,
    )
    axes = [fig.add_subplot(outer_grid[0, index]) for index in range(3)]

    panel_b_xcorr_payload = panel_data["panel_b_xcorr_payload"]
    if panel_b_xcorr_payload is not None:
        _figure_3.plot_top_ca1_xcorr_panel(
            axes[0],
            _figure_3.prepare_xcorr_payload_for_display(panel_b_xcorr_payload),
            lag_label_y=-0.055,
            compact_unit_titles=True,
        )
    else:
        axes[0].axis("off")
        axes[0].text(
            0.5,
            0.5,
            "No xcorr data",
            ha="center",
            va="center",
            fontsize=6,
            transform=axes[0].transAxes,
        )
    _figure_3.plot_glm_analysis_panel(
        axes[1],
        panel_data["panel_c_epoch_tables"],
        ripple_trace=panel_data["panel_c_ripple_trace"],
        prediction_examples=panel_data["panel_c_prediction_examples"],
        column_panel_labels=PANEL_B_COLUMN_LABELS,
    )
    _figure_3.plot_glm_dark_epoch_properties_panel(
        axes[2],
        panel_data["panel_e_behavior_payload"],
        single_epoch_column_bounds=PANEL_E_SINGLE_EPOCH_COLUMN_BOUNDS,
        single_epoch_axis_vertical_bounds=(
            PANEL_E_SINGLE_EPOCH_AXIS_VERTICAL_BOUNDS
        ),
        dppi_axis_bounds=PANEL_E_DPPI_AXIS_BOUNDS,
    )
    panel_f_axis = axes[2].child_axes[-1]
    panel_f_axis.set_title(PANEL_F_TITLE, fontsize=7.2, pad=2)
    panel_f_axis.set_xlabel(PANEL_F_XLABEL, fontsize=6, labelpad=1.0)
    for axis, title in zip(axes, PANEL_TITLES, strict=True):
        axis.set_title(title, fontsize=7.2, pad=2)

    fig.canvas.draw()
    fig.set_layout_engine(None)
    _figure_3.add_aligned_panel_headers(
        fig,
        axes,
        labels=PANEL_HEADER_LABELS,
        titles=PANEL_TITLES,
        label_x_offsets=PANEL_HEADER_LABEL_X_OFFSETS,
        fontsize=7.2,
    )
    _add_aligned_panel_e_f_headers(fig, panel_f_axis)
    _align_panel_f_histogram_to_panel_d_bottom(axes[1], panel_f_axis)

    _figure_3.save_figure(fig, output_path, dpi=dpi)
    plt.close(fig)
    for missing in panel_data["panel_e_behavior_payload"]["missing_artifacts"]:
        print(
            "Panel E dark-activity missing "
            f"{missing['artifact']} for {missing['animal_name']} "
            f"{missing['date']} {missing['epoch']}: {missing['path']}"
        )
    print(f"Saved Figure 4 to {output_path}")
    return output_path


def _argv_has_output_name(argv: Sequence[str]) -> bool:
    """Return whether command-line arguments explicitly set the output name."""
    return any(
        argument == "--output-name" or argument.startswith("--output-name=")
        for argument in argv
    )


def parse_arguments(argv: Sequence[str] | None = None) -> Any:
    """Parse Figure 4 arguments using the Figure 3 data-selection interface."""
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = _figure_3.parse_arguments(raw_argv)
    if not _argv_has_output_name(raw_argv):
        args.output_name = DEFAULT_OUTPUT_NAME
    return args


def main(argv: Sequence[str] | None = None) -> None:
    """Run Figure 4 generation."""
    args = parse_arguments(argv)
    datasets = args.dataset if args.dataset is not None else get_processed_datasets()
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
        light_epoch=args.light_epoch,
        dark_epoch=args.dark_epoch,
        sleep_epoch=args.sleep_epoch,
        ripple_threshold_zscore=args.ripple_threshold_zscore,
        ripple_window_s=args.ripple_window_s,
        ripple_window_offset_s=args.ripple_window_offset_s,
        ripple_selection=args.ripple_selection,
        ridge_strength=args.ridge_strength,
        dark_movement_fr_cache_dir=args.dark_movement_fr_cache_dir,
        refresh_dark_movement_fr_cache=args.refresh_dark_movement_fr_cache,
        refresh_panel_b_schematic_cache=args.refresh_panel_b_schematic_cache,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
