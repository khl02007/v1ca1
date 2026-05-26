from __future__ import annotations

"""Generate Figure 1 panels for pooled dark-epoch place-field heatmaps."""

import argparse
import hashlib
import html
import io
import json
import os
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from v1ca1.helper.session import (
    DEFAULT_DATA_ROOT,
    DEFAULT_PLACE_BIN_SIZE_CM,
    DEFAULT_POSITION_OFFSET,
    DEFAULT_SPEED_THRESHOLD_CM_S,
    REGIONS,
    TRAJECTORY_TYPES,
)
from v1ca1.helper.plot_wtrack_schematic import get_w_track_geometry
from v1ca1.helper.wtrack import get_wtrack_total_length
from v1ca1.paper_figures.datasets import (
    DEFAULT_DARK_EPOCH,
    DatasetId,
    FigureEpochDatasetId,
    get_dataset_dark_epoch,
    get_processed_datasets,
    make_dataset_id,
    normalize_dataset_id,
)
from v1ca1.paper_figures.style import (
    ANIMAL_COLORS,
    COMPACT_HISTOGRAM_KWARGS,
    EMPHASIS_HISTOGRAM_KWARGS,
    ENCODING_COMPARISON_COLORS,
    HISTOGRAM_KWARGS,
    NEUTRAL_COLORS,
    RASTER_TICK_KWARGS,
    REGION_COLORS,
    SCHEMATIC_COLORS,
    TRAJECTORY_COLORS,
    apply_paper_style,
    figure_size,
    label_axis,
    save_figure,
)
from v1ca1.paper_figures.w_track_schematic import draw_w_track_schematic
from v1ca1.raster.plot_place_field_heatmap import (
    DEFAULT_SIGMA_BINS,
    align_and_normalize_panel_values,
    build_linear_position_by_trajectory,
    compute_place_tuning_curve,
    compute_odd_even_place_tuning_curves,
    compute_unit_order,
    prepare_heatmap_session,
)
from v1ca1.raster.plot_1d_place_field_trajectory import (
    compute_trial_spike_positions,
    make_linear_position_interpolator,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.image import AxesImage
    from matplotlib.text import Text


DEFAULT_OUTPUT_DIR = Path("paper_figures") / "output"
DEFAULT_OUTPUT_NAME = "figure_1"
DEFAULT_OUTPUT_FORMAT = "pdf"
DEFAULT_ASSET_DIR = Path("paper_figures") / "assets" / "figure_1"
DEFAULT_PROBE_ASSET_NAME = "probe.jpg"
DEFAULT_HISTOLOGY_ASSET_NAME = "histology.svg"
DEFAULT_BEHAVIOR_ASSET_NAME = "behavior.png"
DEFAULT_POSITION_BIN_COUNT = 50
DEFAULT_REGIONS = ("v1",)
DEFAULT_FIGURE_WIDTH_MM = 165.0
DEFAULT_TOP_ROW_HEIGHT_MM = 40.0
DEFAULT_HEATMAP_HEIGHT_MM = 84.0
DEFAULT_MIDDLE_TO_FINAL_ROW_SPACER_MM = 0.5
DEFAULT_BOTTOM_ROW_HEIGHT_MM = 30.0
DEFAULT_HEATMAP_PANEL_WIDTH_FRACTION = 0.7
DEFAULT_PANEL_E_WIDTH_FRACTION = 0.3
DEFAULT_PANEL_F_WIDTH_FRACTION = 1.0 / 3.0
DEFAULT_PANEL_G_WIDTH_FRACTION = 1.0 / 3.0
DEFAULT_PANEL_H_WIDTH_FRACTION = 1.0 / 3.0
PANEL_H_DECODING_ANIMALS = ("L12", "L14", "L15", "L19")
BOTTOM_ROW_PANEL_WSPACE = 0.05
HEATMAP_COLORBAR_PAD = 0.001
HEATMAP_COLORBAR_LABEL_FONTSIZE = 4.9
HEATMAP_COLORBAR_LABELPAD = 0
HEATMAP_TUNING_LABEL_OFFSET = -0.004
HEATMAP_ORDER_LABEL_OFFSET = 0.007
PANEL_D_CACHE_PREFIX = "figure_1_panel_d"
PANEL_D_CACHE_VERSION = 1
PANEL_D_CACHE_METADATA_KEY = "__metadata__"
PANEL_D_CACHE_DATASET_TOKEN_LIMIT = 96
PANEL_E_CACHE_PREFIX = "figure_1_panel_e"
PANEL_E_CACHE_VERSION = 1
PANEL_E_CACHE_METADATA_KEY = "__metadata__"
FIGURE_FORMATS = ("pdf", "svg", "png", "tiff")
RASTER_ASSET_EXTENSIONS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")
NEURON_SCALE_BAR_COUNT = 100
PANEL_E_EXAMPLES = (
    ("L14", "20240611", "08_r4", "v1", 34),
    ("L15", "20241121", "10_r5", "v1", 473),
)
PANEL_E_RASTER_TRAJECTORY_LAYOUT = (
    ("center_to_left", "center_to_right"),
    ("right_to_center", "left_to_center"),
)
PANEL_E_FR_TRAJECTORY_PAIRS = (
    ("center_to_left", "right_to_center"),
    ("center_to_right", "left_to_center"),
)
PANEL_E_TRAJECTORY_LABELS = {
    "center_to_left": "C→L",
    "center_to_right": "C→R",
    "right_to_center": "R→C",
    "left_to_center": "L→C",
}
PANEL_E_TRAJECTORY_COLORS = TRAJECTORY_COLORS
PANEL_E_RASTER_TICK_MARKERSIZE = RASTER_TICK_KWARGS["markersize"]
PANEL_E_RASTER_TICK_MARKEREDGEWIDTH = RASTER_TICK_KWARGS["markeredgewidth"]
PANEL_E_AXIS_LABEL_FONTSIZE = 5.4
PANEL_E_TICK_LABEL_FONTSIZE = 5.0
TASK_PROGRESSION_SEGMENT_BOUNDARIES = (0.4, 0.6)
TASK_PROGRESSION_SEGMENT_BOUNDARY_COLOR = NEUTRAL_COLORS["segment_boundary"]
TASK_PROGRESSION_SEGMENT_BOUNDARY_LINEWIDTH = 0.45
STABILITY_TRAJECTORY_LAYOUT = PANEL_E_RASTER_TRAJECTORY_LAYOUT
STABILITY_LEGEND_FONTSIZE = 5.4
STABILITY_AXIS_LABEL_FONTSIZE = 5.7
STABILITY_TICK_LABEL_FONTSIZE = 5.5
STABILITY_TABLE_RELATIVE_PATH = (
    Path("task_progression") / "stability" / "odd_even_task_progression_stability.parquet"
)
MOTOR_NESTED_CV_RELATIVE_DIR = Path("task_progression") / "motor" / "nested_lap_cv"
MOTOR_DELTA_METRIC = "dll_motor_tp_vs_motor_bits_per_spike"
MOTOR_DELTA_REGION = "v1"
MOTOR_PREFERRED_FILENAME_TOKEN = "_zscore_"
ENCODING_COMPARISON_RELATIVE_DIR = Path("task_progression") / "encoding_comparison"
ENCODING_COMPARISON_REGION = "v1"
ENCODING_COMPARISON_N_FOLDS = 5
ENCODING_COMPARISON_PLACE_BIN_SIZE_CM = DEFAULT_PLACE_BIN_SIZE_CM
ENCODING_COMPARISON_MIN_SPIKES = 0
DECODING_COMPARISON_RELATIVE_DIR = Path("task_progression") / "decoding_comparison"
DECODING_COMPARISON_REGION = "v1"
STABILITY_REGIONS = ("v1", "ca1")
STABILITY_REGION_COLORS = REGION_COLORS
ENCODING_DPP_COMPARISONS = (
    (
        "dpp_vs_absolute_place",
        "DGP - absolute place",
        "delta_bits_generalized_place_vs_tp",
    ),
    (
        "dpp_vs_absolute_task_progression",
        "DGP - distance-to-reward",
        "delta_bits_gtp_vs_tp",
    ),
)
ENCODING_DPP_COMPARISON_COLORS = ENCODING_COMPARISON_COLORS
DECODING_CROSS_TRAJECTORY_COMPARISONS = (
    (
        "same_turn_cross_arm",
        "Same turn\ncross arm",
        "same_turn_cross_arm",
        (
            ("center_to_left", "right_to_center"),
            ("right_to_center", "center_to_left"),
            ("center_to_right", "left_to_center"),
            ("left_to_center", "center_to_right"),
        ),
    ),
    (
        "opposite_turn_same_arm",
        "Opposite turn\nsame arm",
        "opposite_turn_same_arm",
        (
            ("center_to_left", "left_to_center"),
            ("left_to_center", "center_to_left"),
            ("center_to_right", "right_to_center"),
            ("right_to_center", "center_to_right"),
        ),
    ),
    (
        "same_inbound_outbound_cross_arm",
        "Same in/out\ncross arm",
        "same_inbound_outbound_cross_arm",
        (
            ("center_to_left", "center_to_right"),
            ("center_to_right", "center_to_left"),
            ("left_to_center", "right_to_center"),
            ("right_to_center", "left_to_center"),
        ),
    ),
)
DECODING_ANIMAL_COLORS = ANIMAL_COLORS
DECODING_EXAMPLE_TRAIN_TRAJECTORY = "center_to_left"
DECODING_EXAMPLE_TEST_TRAJECTORIES = {
    "same_turn_cross_arm": "right_to_center",
    "opposite_turn_same_arm": "left_to_center",
    "same_inbound_outbound_cross_arm": "center_to_right",
}
DECODING_TRAIN_SCHEMATIC_CENTER_X = -0.075
DECODING_SCHEMATIC_Y = -0.55
DECODING_SCHEMATIC_WIDTH = 0.132
DECODING_SCHEMATIC_HEIGHT = 0.198
DECODING_TRAIN_LABEL_Y = -0.32
DECODING_YLABEL_FONTSIZE = 7.6
DECODING_XTICK_LABEL_FONTSIZE = 5.6
DECODING_MEDIAN_LABEL_FONTSIZE = 4.8
DECODING_MEDIAN_LABEL_X_OFFSET = 0.09
DELTA_LOG_LIKELIHOOD_AXIS_LABEL = "Δ log likelihood\n(bits/spike)"
STABILITY_TABLE_COLUMNS = (
    "animal_name",
    "date",
    "unit",
    "region",
    "epoch",
    "trajectory_type",
    "firing_rate_hz",
    "stability_correlation",
    "n_odd_trials",
    "n_even_trials",
    "odd_duration_s",
    "even_duration_s",
)
MOTOR_DELTA_TABLE_COLUMNS = (
    "animal_name",
    "date",
    "epoch",
    "region",
    "unit",
    "delta_log_likelihood_bits_per_spike",
    "source_path",
)
ENCODING_DELTA_TABLE_COLUMNS = (
    "animal_name",
    "date",
    "epoch",
    "region",
    "unit",
    "n_spikes",
    "comparison",
    "comparison_label",
    "delta_log_likelihood_bits_per_spike",
    "source_path",
)
DECODING_ABSOLUTE_ERROR_TABLE_COLUMNS = (
    "animal_name",
    "date",
    "epoch",
    "region",
    "comparison",
    "comparison_label",
    "transfer_family",
    "encoding_trajectory",
    "decoding_trajectory",
    "absolute_error",
    "true_path",
    "decoded_path",
)
CYCLE_TRAJECTORY_LAYOUT = (
    ("left_to_center", (0.03, 0.45, 0.26, 0.27)),
    ("center_to_right", (0.37, 0.70, 0.26, 0.27)),
    ("right_to_center", (0.71, 0.45, 0.26, 0.27)),
    ("center_to_left", (0.37, 0.27, 0.26, 0.27)),
)
CYCLE_ARROW_SPECS = (
    ((0.24, 0.68), (0.37, 0.80), -0.25),
    ((0.64, 0.80), (0.77, 0.68), -0.25),
    ((0.77, 0.46), (0.64, 0.34), -0.25),
    ((0.36, 0.34), (0.23, 0.46), -0.25),
)
CYCLE_ARROW_LINEWIDTH = 1.08
CYCLE_ARROW_MUTATION_SCALE = 12.6
MOVEMENT_AXIS_Y = -0.13
MOVEMENT_AXIS_ARROW_MARGIN = 0.12


def get_trajectory_endpoint_labels(trajectory_type: str) -> tuple[str, str]:
    """Return normalized-axis endpoint labels for one trajectory type."""
    if "left" in trajectory_type:
        return "C", "L"
    if "right" in trajectory_type:
        return "C", "R"
    raise ValueError(f"Unknown trajectory type {trajectory_type!r}.")


def get_movement_arrow_points(trajectory_type: str) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return axes-fraction start/end points for the movement-direction arrow."""
    if trajectory_type.startswith("center_to_"):
        return (
            (MOVEMENT_AXIS_ARROW_MARGIN, MOVEMENT_AXIS_Y),
            (1.0 - MOVEMENT_AXIS_ARROW_MARGIN, MOVEMENT_AXIS_Y),
        )
    if trajectory_type.endswith("_to_center"):
        return (
            (1.0 - MOVEMENT_AXIS_ARROW_MARGIN, MOVEMENT_AXIS_Y),
            (MOVEMENT_AXIS_ARROW_MARGIN, MOVEMENT_AXIS_Y),
        )
    raise ValueError(f"Unknown trajectory type {trajectory_type!r}.")


def add_movement_axis_arrow(ax: "Axes", trajectory_type: str) -> None:
    """Draw a bottom movement-direction arrow for one trajectory column."""
    start, end = get_movement_arrow_points(trajectory_type)
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops={
            "arrowstyle": "->",
            "color": "black",
            "linewidth": 0.7,
            "shrinkA": 0,
            "shrinkB": 0,
        },
        annotation_clip=False,
    )


def add_movement_axis_labels(ax: "Axes", trajectory_type: str) -> None:
    """Draw endpoint labels aligned to the movement-direction arrow."""
    left_label, right_label = get_trajectory_endpoint_labels(trajectory_type)
    for x, label in ((0.0, left_label), (1.0, right_label)):
        ax.text(
            x,
            MOVEMENT_AXIS_Y,
            label,
            ha="center",
            va="center",
            transform=ax.transAxes,
            clip_on=False,
        )


def add_movement_axis_annotations(ax: "Axes", trajectory_type: str) -> None:
    """Draw aligned endpoint labels and movement-direction arrow."""
    add_movement_axis_labels(ax, trajectory_type)
    add_movement_axis_arrow(ax, trajectory_type)


def add_centered_axis_text(
    fig: Any,
    axes: Sequence["Axes"],
    text: str,
    *,
    y_offset: float = 0.01,
    rotation: float = 0.0,
    fontsize: float = 9.0,
) -> "Text":
    """Add text centered over or beside a group of axes."""
    boxes = [ax.get_position() for ax in axes]
    x0 = min(box.x0 for box in boxes)
    x1 = max(box.x1 for box in boxes)
    y0 = min(box.y0 for box in boxes)
    y1 = max(box.y1 for box in boxes)
    if rotation:
        return fig.text(
            x0 - y_offset,
            (y0 + y1) / 2,
            text,
            ha="center",
            va="center",
            rotation=rotation,
            fontsize=fontsize,
        )
    return fig.text(
        (x0 + x1) / 2,
        y1 + y_offset,
        text,
        ha="center",
        va="bottom",
        fontsize=fontsize,
    )


def get_dark_epoch(animal_name: str, date: str, dark_epoch: str | None = None) -> str:
    """Return the dark run epoch label for one session."""
    del date
    if dark_epoch is not None:
        return str(dark_epoch)
    return get_dataset_dark_epoch(animal_name)


def parse_dataset_id(value: str) -> DatasetId:
    """Parse one `animal:date[:dark_epoch]` data-set identifier."""
    parts = value.split(":")
    if len(parts) not in (2, 3) or not all(parts):
        raise argparse.ArgumentTypeError(
            "Data sets must be specified as animal:date or animal:date:dark_epoch, "
            "for example L14:20240611 or L15:20241121:10_r5."
        )
    return make_dataset_id(*parts)


def build_output_path(
    output_dir: Path,
    output_name: str,
    output_format: str,
) -> Path:
    """Return the figure output path for one requested format."""
    if output_format not in FIGURE_FORMATS:
        raise ValueError(
            f"Unknown output format {output_format!r}. Expected one of {FIGURE_FORMATS!r}."
        )
    return Path(output_dir) / f"{output_name}.{output_format}"


def get_figure_1_asset_path(asset_dir: Path, asset_name: str) -> Path:
    """Return the path to one Figure 1 external asset."""
    return Path(asset_dir) / asset_name


def _parse_svg_aspect_ratio(svg_path: Path) -> float:
    """Return width divided by height for one SVG file."""
    root = ET.parse(svg_path).getroot()
    viewbox = root.attrib.get("viewBox")
    if viewbox is not None:
        values = [float(value) for value in viewbox.replace(",", " ").split()]
        if len(values) == 4 and values[2] > 0 and values[3] > 0:
            return values[2] / values[3]

    width = _parse_svg_length(root.attrib.get("width"))
    height = _parse_svg_length(root.attrib.get("height"))
    if width is None or height is None or height <= 0:
        return 1.0
    return width / height


def _parse_svg_length(value: str | None) -> float | None:
    """Parse one SVG length string and ignore its unit."""
    if value is None:
        return None
    stripped = value.strip()
    if not stripped:
        return None
    number = []
    for character in stripped:
        if character.isdigit() or character in ".+-eE":
            number.append(character)
        else:
            break
    if not number:
        return None
    return float("".join(number))


def _find_svg_sidecar_raster(svg_path: Path) -> Path | None:
    """Return a same-stem raster export for an SVG when one exists."""
    for extension in RASTER_ASSET_EXTENSIONS:
        sidecar_path = svg_path.with_suffix(extension)
        if sidecar_path.exists():
            return sidecar_path
    return None


def _render_svg_with_cairosvg(svg_path: Path, *, output_width_px: int) -> np.ndarray | None:
    """Rasterize an SVG with cairosvg when that optional dependency is installed."""
    try:
        import cairosvg
        import matplotlib.image as mpimg
    except ImportError:
        return None

    png_bytes = cairosvg.svg2png(
        url=str(svg_path),
        output_width=int(output_width_px),
    )
    return np.asarray(mpimg.imread(io.BytesIO(png_bytes), format="png"))


def _find_chrome_executable() -> str | None:
    """Return an available Chrome executable for SVG rasterization."""
    for executable in ("google-chrome", "chromium", "chromium-browser"):
        path = shutil.which(executable)
        if path is not None:
            return path
    return None


def _render_svg_with_chrome(svg_path: Path, *, output_width_px: int) -> np.ndarray | None:
    """Rasterize an SVG with headless Chrome when available."""
    import matplotlib.image as mpimg

    chrome_path = _find_chrome_executable()
    if chrome_path is None:
        return None

    aspect_ratio = _parse_svg_aspect_ratio(svg_path)
    output_height_px = max(1, int(round(output_width_px / aspect_ratio)))
    with tempfile.TemporaryDirectory(prefix="v1ca1-figure-asset-") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        html_path = temp_dir / "asset.html"
        png_path = temp_dir / "asset.png"
        html_path.write_text(
            "\n".join(
                [
                    "<!doctype html>",
                    "<html>",
                    "<head>",
                    "<meta charset='utf-8'>",
                    "<style>",
                    "html, body { margin: 0; width: 100%; height: 100%; background: white; }",
                    "img { width: 100vw; height: 100vh; object-fit: contain; display: block; }",
                    "</style>",
                    "</head>",
                    "<body>",
                    f"<img src='{html.escape(svg_path.resolve().as_uri(), quote=True)}'>",
                    "</body>",
                    "</html>",
                ]
            ),
            encoding="utf-8",
        )
        command = [
            chrome_path,
            "--headless=new",
            "--disable-gpu",
            "--no-sandbox",
            "--disable-dev-shm-usage",
            "--disable-crash-reporter",
            "--disable-breakpad",
            "--disable-features=Crashpad",
            f"--user-data-dir={temp_dir / 'chrome-profile'}",
            f"--screenshot={png_path}",
            f"--window-size={int(output_width_px)},{output_height_px}",
            html_path.as_uri(),
        ]
        env = {
            **os.environ,
            "HOME": str(temp_dir),
            "XDG_CONFIG_HOME": str(temp_dir),
            "XDG_CACHE_HOME": str(temp_dir),
        }
        try:
            subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                env=env,
            )
        except subprocess.CalledProcessError as error:
            message = (error.stderr or error.stdout or "").strip()
            raise RuntimeError(
                "Could not rasterize SVG asset with headless Chrome. "
                "Install cairosvg or place a same-stem raster export next to the SVG. "
                f"Chrome output: {message}"
            ) from error

        return np.asarray(mpimg.imread(png_path))


def load_panel_asset_image(asset_path: Path, *, svg_output_width_px: int = 1200) -> np.ndarray:
    """Load one raster image asset, rasterizing SVG assets when needed."""
    import matplotlib.image as mpimg

    asset_path = Path(asset_path)
    if not asset_path.exists():
        raise FileNotFoundError(f"Missing Figure 1 asset: {asset_path}")

    suffix = asset_path.suffix.lower()
    if suffix in RASTER_ASSET_EXTENSIONS:
        return np.asarray(mpimg.imread(asset_path))
    if suffix != ".svg":
        raise ValueError(f"Unsupported Figure 1 asset format: {asset_path}")

    sidecar_path = _find_svg_sidecar_raster(asset_path)
    if sidecar_path is not None:
        return np.asarray(mpimg.imread(sidecar_path))

    image = _render_svg_with_cairosvg(asset_path, output_width_px=svg_output_width_px)
    if image is not None:
        return image

    image = _render_svg_with_chrome(asset_path, output_width_px=svg_output_width_px)
    if image is not None:
        return image

    raise RuntimeError(
        "Could not rasterize SVG asset. Install cairosvg or place a same-stem "
        f"raster export next to {asset_path}."
    )


def build_normalized_position_bins(position_bin_count: int) -> np.ndarray:
    """Return normalized trajectory-position bin edges from 0 to 1."""
    if position_bin_count <= 0:
        raise ValueError("--position-bin-count must be positive.")
    return np.linspace(0.0, 1.0, int(position_bin_count) + 1)


def get_stability_table_path(data_root: Path, animal_name: str, date: str) -> Path:
    """Return the saved task-progression stability table path for one session."""
    return Path(data_root) / animal_name / date / STABILITY_TABLE_RELATIVE_PATH


def get_motor_nested_cv_dir(data_root: Path, animal_name: str, date: str) -> Path:
    """Return the saved task-progression motor nested-CV directory."""
    return Path(data_root) / animal_name / date / MOTOR_NESTED_CV_RELATIVE_DIR


def _format_float_token(value: float) -> str:
    """Return a path-safe compact token for one numeric value."""
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def format_place_bin_size_token(place_bin_size_cm: float) -> str:
    """Return the filename token for one place-bin-size setting."""
    return f"placebin{_format_float_token(place_bin_size_cm)}cm"


def get_encoding_comparison_dir(data_root: Path, animal_name: str, date: str) -> Path:
    """Return the saved task-progression encoding-comparison directory."""
    return Path(data_root) / animal_name / date / ENCODING_COMPARISON_RELATIVE_DIR


def get_decoding_comparison_dir(data_root: Path, animal_name: str, date: str) -> Path:
    """Return the saved task-progression decoding-comparison directory."""
    return Path(data_root) / animal_name / date / DECODING_COMPARISON_RELATIVE_DIR


def find_motor_nested_cv_path(
    *,
    data_root: Path,
    animal_name: str,
    date: str,
    region: str,
    epoch: str,
) -> Path:
    """Return the preferred motor nested-CV output path for one session epoch."""
    data_dir = get_motor_nested_cv_dir(data_root, animal_name, date)
    candidates = sorted(data_dir.glob(f"{region}_{epoch}_nested_lapcv_*.nc"))
    if not candidates:
        raise FileNotFoundError(
            "Missing task-progression motor nested-CV output. Expected a file "
            f"matching {data_dir / f'{region}_{epoch}_nested_lapcv_*.nc'}. "
            "Run `python -m v1ca1.task_progression.motor "
            f"--animal-name {animal_name} --date {date} --region {region} "
            f"--epoch {epoch}` first."
        )

    preferred = [
        path for path in candidates if MOTOR_PREFERRED_FILENAME_TOKEN in path.name
    ]
    candidates = preferred if preferred else candidates
    return max(candidates, key=lambda path: (path.stat().st_mtime, path.name))


def find_encoding_summary_path(
    *,
    data_root: Path,
    animal_name: str,
    date: str,
    region: str,
    epoch: str,
    n_folds: int = ENCODING_COMPARISON_N_FOLDS,
    place_bin_size_cm: float = ENCODING_COMPARISON_PLACE_BIN_SIZE_CM,
) -> Path:
    """Return the preferred encoding-comparison summary table path."""
    data_dir = get_encoding_comparison_dir(data_root, animal_name, date)
    place_bin_token = format_place_bin_size_token(place_bin_size_cm)
    candidates = (
        data_dir / f"{region}_{epoch}_cv{n_folds}_{place_bin_token}_encoding_summary.parquet",
        data_dir / f"{region}_{epoch}_cv{n_folds}_encoding_summary.parquet",
    )
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Missing task-progression encoding-comparison summary. Expected "
        f"{candidates[0]} or legacy file {candidates[1]}. Run "
        "`python -m v1ca1.task_progression.encoding_comparison "
        f"--animal-name {animal_name} --date {date} --dark-epoch {epoch} "
        f"--regions {region} --place-bin-size-cm {place_bin_size_cm}` first."
    )


def get_cross_trajectory_decoding_tsd_paths(
    *,
    data_root: Path,
    animal_name: str,
    date: str,
    region: str,
    epoch: str,
    transfer_family: str,
    encoding_trajectory: str,
    decoding_trajectory: str,
) -> tuple[Path, Path]:
    """Return true and decoded cross-trajectory task-progression `.npz` paths."""
    data_dir = get_decoding_comparison_dir(data_root, animal_name, date)
    suffix = f"{transfer_family}_{encoding_trajectory}_to_{decoding_trajectory}"
    true_path = data_dir / f"{region}_{epoch}_{suffix}_true_tp_cross_traj.npz"
    decoded_path = data_dir / f"{region}_{epoch}_{suffix}_decoded_tp_cross_traj.npz"
    return true_path, decoded_path


def _load_tsd_npz(path: Path) -> Any:
    """Load one pynapple-backed `.npz` time-series artifact."""
    import pynapple as nap

    return nap.load_file(path)


def load_decoding_absolute_error_table(
    *,
    data_root: Path,
    datasets: Sequence[DatasetId],
    region: str = DECODING_COMPARISON_REGION,
    comparisons: Sequence[
        tuple[str, str, str, Sequence[tuple[str, str]]]
    ] = DECODING_CROSS_TRAJECTORY_COMPARISONS,
) -> Any:
    """Load pooled sample-level cross-trajectory decoding absolute errors."""
    import pandas as pd

    from v1ca1.task_progression.decoding_comparison import align_true_to_decoded

    tables = []
    for dataset in datasets:
        animal_name, date, epoch = normalize_dataset_id(dataset)
        for comparison, label, transfer_family, trajectory_pairs in comparisons:
            for encoding_trajectory, decoding_trajectory in trajectory_pairs:
                true_path, decoded_path = get_cross_trajectory_decoding_tsd_paths(
                    data_root=data_root,
                    animal_name=animal_name,
                    date=date,
                    region=region,
                    epoch=epoch,
                    transfer_family=transfer_family,
                    encoding_trajectory=encoding_trajectory,
                    decoding_trajectory=decoding_trajectory,
                )
                missing_paths = [
                    path for path in (true_path, decoded_path) if not path.exists()
                ]
                if missing_paths:
                    raise FileNotFoundError(
                        "Missing task-progression decoding-comparison output. "
                        f"Expected {missing_paths[0]}. Run "
                        "`python -m v1ca1.task_progression.decoding_comparison "
                        f"--animal-name {animal_name} --date {date} "
                        f"--dark-epoch {epoch} --regions {region}` first."
                    )

                true_tsd = _load_tsd_npz(true_path)
                decoded_tsd = _load_tsd_npz(decoded_path)
                true_values, decoded_values = align_true_to_decoded(true_tsd, decoded_tsd)
                absolute_error = np.abs(decoded_values - true_values)
                absolute_error = absolute_error[np.isfinite(absolute_error)]
                if absolute_error.size == 0:
                    continue
                tables.append(
                    pd.DataFrame(
                        {
                            "animal_name": animal_name,
                            "date": date,
                            "epoch": epoch,
                            "region": region,
                            "comparison": comparison,
                            "comparison_label": label,
                            "transfer_family": transfer_family,
                            "encoding_trajectory": encoding_trajectory,
                            "decoding_trajectory": decoding_trajectory,
                            "absolute_error": absolute_error,
                            "true_path": str(true_path),
                            "decoded_path": str(decoded_path),
                        }
                    )
                )

    if not tables:
        return pd.DataFrame(columns=DECODING_ABSOLUTE_ERROR_TABLE_COLUMNS)
    return pd.concat(tables, axis=0, ignore_index=True)


def filter_datasets_by_animals(
    datasets: Sequence[DatasetId | FigureEpochDatasetId | tuple[str, str]],
    animal_names: Sequence[str],
) -> list[DatasetId]:
    """Return normalized data-set IDs for selected animals."""
    selected_animals = {str(animal_name) for animal_name in animal_names}
    return [
        normalized_dataset
        for dataset in datasets
        if (normalized_dataset := normalize_dataset_id(dataset))[0] in selected_animals
    ]


def load_dark_epoch_stability_table(
    *,
    data_root: Path,
    datasets: Sequence[DatasetId],
    regions: Sequence[str] = STABILITY_REGIONS,
) -> Any:
    """Load pooled dark-epoch stability rows for the requested data sets."""
    import pandas as pd

    tables = []
    selected_regions = set(regions)
    for dataset in datasets:
        animal_name, date, dark_epoch = normalize_dataset_id(dataset)
        table_path = get_stability_table_path(data_root, animal_name, date)
        if not table_path.exists():
            raise FileNotFoundError(
                "Missing task-progression stability table. Expected "
                f"{table_path}. Run `python -m v1ca1.task_progression.stability` "
                "for this session first."
            )
        table = pd.read_parquet(table_path)
        table = table[
            (table["epoch"].astype(str) == dark_epoch)
            & (table["region"].astype(str).isin(selected_regions))
            & (table["trajectory_type"].astype(str).isin(TRAJECTORY_TYPES))
        ].copy()
        tables.append(table)

    if not tables:
        return pd.DataFrame(columns=STABILITY_TABLE_COLUMNS)
    return pd.concat(tables, axis=0, ignore_index=True)


def load_motor_delta_table(
    *,
    data_root: Path,
    datasets: Sequence[DatasetId],
    region: str = MOTOR_DELTA_REGION,
    delta_metric: str = MOTOR_DELTA_METRIC,
) -> Any:
    """Load pooled V1 motor+DPP versus motor delta log-likelihood values."""
    import pandas as pd
    import xarray as xr

    rows: list[dict[str, Any]] = []
    for dataset in datasets:
        animal_name, date, epoch = normalize_dataset_id(dataset)
        nested_cv_path = find_motor_nested_cv_path(
            data_root=data_root,
            animal_name=animal_name,
            date=date,
            region=region,
            epoch=epoch,
        )
        fit_dataset = xr.open_dataset(nested_cv_path)
        try:
            values = np.asarray(
                fit_dataset["pooled_delta_bits_per_spike"]
                .sel(delta_metric=delta_metric)
                .values,
                dtype=float,
            ).reshape(-1)
            units = np.asarray(fit_dataset.coords["unit"].values)
        finally:
            fit_dataset.close()

        rows.extend(
            {
                "animal_name": animal_name,
                "date": date,
                "epoch": epoch,
                "region": region,
                "unit": int(unit),
                "delta_log_likelihood_bits_per_spike": float(value),
                "source_path": str(nested_cv_path),
            }
            for unit, value in zip(units, values, strict=True)
        )

    if not rows:
        return pd.DataFrame(columns=MOTOR_DELTA_TABLE_COLUMNS)
    return pd.DataFrame(rows, columns=MOTOR_DELTA_TABLE_COLUMNS)


def _to_scalar(value: Any) -> Any:
    """Return a Python scalar for NumPy scalar values."""
    return value.item() if hasattr(value, "item") else value


def load_encoding_delta_table(
    *,
    data_root: Path,
    datasets: Sequence[DatasetId],
    region: str = ENCODING_COMPARISON_REGION,
    n_folds: int = ENCODING_COMPARISON_N_FOLDS,
    place_bin_size_cm: float = ENCODING_COMPARISON_PLACE_BIN_SIZE_CM,
    min_spikes: int = ENCODING_COMPARISON_MIN_SPIKES,
) -> Any:
    """Load pooled V1 DPP-versus-absolute-model delta log-likelihood values."""
    import pandas as pd

    rows: list[dict[str, Any]] = []
    for dataset in datasets:
        animal_name, date, epoch = normalize_dataset_id(dataset)
        summary_path = find_encoding_summary_path(
            data_root=data_root,
            animal_name=animal_name,
            date=date,
            region=region,
            epoch=epoch,
            n_folds=n_folds,
            place_bin_size_cm=place_bin_size_cm,
        )
        table = pd.read_parquet(summary_path)
        missing_columns = [
            source_column
            for _comparison, _label, source_column in ENCODING_DPP_COMPARISONS
            if source_column not in table.columns
        ]
        if "n_spikes" not in table.columns:
            missing_columns.append("n_spikes")
        if missing_columns:
            raise ValueError(
                f"Encoding summary {summary_path} is missing columns "
                f"{missing_columns!r}."
            )

        if int(min_spikes) > 0:
            table = table[np.asarray(table["n_spikes"], dtype=float) >= int(min_spikes)]
        units = np.asarray(table.index)
        n_spikes = np.asarray(table["n_spikes"], dtype=int)
        for comparison, label, source_column in ENCODING_DPP_COMPARISONS:
            values = -np.asarray(table[source_column], dtype=float).reshape(-1)
            rows.extend(
                {
                    "animal_name": animal_name,
                    "date": date,
                    "epoch": epoch,
                    "region": region,
                    "unit": _to_scalar(unit),
                    "n_spikes": int(spike_count),
                    "comparison": comparison,
                    "comparison_label": label,
                    "delta_log_likelihood_bits_per_spike": float(value),
                    "source_path": str(summary_path),
                }
                for unit, spike_count, value in zip(units, n_spikes, values, strict=True)
            )

    if not rows:
        return pd.DataFrame(columns=ENCODING_DELTA_TABLE_COLUMNS)
    return pd.DataFrame(rows, columns=ENCODING_DELTA_TABLE_COLUMNS)


def build_unit_keys(
    animal_name: str,
    date: str,
    region: str,
    units: np.ndarray,
) -> np.ndarray:
    """Return globally unique unit keys for pooled heatmap alignment."""
    return np.asarray(
        [f"{animal_name}:{date}:{region}:{unit}" for unit in units.tolist()],
        dtype=object,
    )


def has_plottable_values(values: np.ndarray) -> bool:
    """Return whether one matrix contains positive finite values."""
    values = np.asarray(values, dtype=float)
    return bool(values.size and np.isfinite(values).any() and np.nanmax(values) > 0)


def extract_tuning_curve_arrays(tuning_curve: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return unit labels and a unit-by-position value matrix from one tuning curve."""
    if len(tuning_curve.dims) != 2:
        raise ValueError(
            "Expected a 2D tuning curve with unit and position dimensions. "
            f"Got dims {tuning_curve.dims!r}."
        )
    unit_dim, pos_dim = tuning_curve.dims
    values = np.asarray(
        tuning_curve.transpose(unit_dim, pos_dim).values,
        dtype=float,
    )
    units = np.asarray(tuning_curve.coords[unit_dim].values)
    return units, values


def normalize_linear_position_by_trajectory(
    animal_name: str,
    linear_position_by_trajectory: dict[str, Any],
) -> dict[str, Any]:
    """Scale linear trajectory coordinates to normalized 0-1 position."""
    import pynapple as nap

    total_length_cm = float(get_wtrack_total_length(animal_name))
    if total_length_cm <= 0:
        raise ValueError(f"W-track total length must be positive for {animal_name!r}.")

    normalized: dict[str, Any] = {}
    for trajectory_type, linear_position in linear_position_by_trajectory.items():
        normalized[trajectory_type] = nap.Tsd(
            t=np.asarray(linear_position.t, dtype=float),
            d=np.asarray(linear_position.d, dtype=float) / total_length_cm,
            time_support=linear_position.time_support,
            time_units="s",
        )
    return normalized


def compute_dark_epoch_tuning_curves(
    *,
    animal_name: str,
    date: str,
    data_root: Path,
    region: str,
    epoch: str,
    position_bin_count: int,
    position_offset: int,
    speed_threshold_cm_s: float,
    sigma_bins: float,
) -> dict[str, Any]:
    """Compute odd/even normalized-position tuning curves for one dark epoch."""
    session = prepare_heatmap_session(
        animal_name=animal_name,
        date=date,
        data_root=data_root,
        regions=(region,),
        position_offset=position_offset,
        speed_threshold_cm_s=speed_threshold_cm_s,
        requested_epoch=epoch,
    )

    selected_epoch = session["run_epochs"][0]
    linear_position_by_trajectory = build_linear_position_by_trajectory(
        animal_name,
        session["position_by_epoch"][selected_epoch],
        session["timestamps_position"][selected_epoch],
        session["trajectory_intervals"][selected_epoch],
        position_offset=position_offset,
    )
    normalized_position_by_trajectory = normalize_linear_position_by_trajectory(
        animal_name,
        linear_position_by_trajectory,
    )
    odd_curves, even_curves = compute_odd_even_place_tuning_curves(
        session["spikes_by_region"][region],
        normalized_position_by_trajectory,
        session["trajectory_intervals"][selected_epoch],
        session["movement_by_run"][selected_epoch],
        bin_edges=build_normalized_position_bins(position_bin_count),
        sigma_bins=sigma_bins,
    )
    return {
        "animal_name": animal_name,
        "date": date,
        "region": region,
        "epoch": selected_epoch,
        "odd_curves": odd_curves,
        "even_curves": even_curves,
    }


def get_unit_spike_times(spikes: Any, unit_id: int) -> np.ndarray:
    """Return spike times for one unit from a pynapple TsGroup-like object."""
    if unit_id not in list(spikes.keys()):
        raise ValueError(f"Unit {unit_id!r} was not found in the requested spikes.")
    return np.asarray(spikes[unit_id].t, dtype=float)


def extract_unit_rate_curve(
    tuning_curve: Any | None,
    unit_id: int,
    fallback_position: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return normalized position and firing-rate values for one unit."""
    fallback_position = np.asarray(fallback_position, dtype=float)
    if tuning_curve is None:
        return fallback_position, np.full(fallback_position.shape, np.nan, dtype=float)

    if len(tuning_curve.dims) != 2:
        raise ValueError(
            "Expected a 2D tuning curve with unit and position dimensions. "
            f"Got dims {tuning_curve.dims!r}."
        )
    unit_dim, position_dim = tuning_curve.dims
    units = np.asarray(tuning_curve.coords[unit_dim].values)
    matching_indices = np.flatnonzero(units == unit_id)
    if matching_indices.size == 0:
        return fallback_position, np.full(fallback_position.shape, np.nan, dtype=float)

    values = np.asarray(
        tuning_curve.transpose(unit_dim, position_dim).values,
        dtype=float,
    )
    position = np.asarray(tuning_curve.coords[position_dim].values, dtype=float)
    return position, values[int(matching_indices[0])]


def orient_panel_e_task_progression(linear_position: Any, trajectory_type: str) -> Any:
    """Orient normalized position so 0 is the trajectory start for panel E."""
    import pynapple as nap

    values = np.asarray(linear_position.d, dtype=float)
    if trajectory_type.endswith("_to_center"):
        values = 1.0 - values
    return nap.Tsd(
        t=np.asarray(linear_position.t, dtype=float),
        d=values,
        time_support=linear_position.time_support,
        time_units="s",
    )


def load_panel_e_example_data(
    *,
    data_root: Path,
    animal_name: str,
    date: str,
    epoch: str,
    region: str,
    unit_id: int,
    position_bin_count: int,
    position_offset: int,
    speed_threshold_cm_s: float,
    sigma_bins: float,
) -> dict[str, Any]:
    """Load one panel E example unit with normalized-position rasters and rate curves."""
    session = prepare_heatmap_session(
        animal_name=animal_name,
        date=date,
        data_root=data_root,
        regions=(region,),
        position_offset=position_offset,
        speed_threshold_cm_s=speed_threshold_cm_s,
        requested_epoch=epoch,
    )
    selected_epoch = session["run_epochs"][0]
    linear_position_by_trajectory = build_linear_position_by_trajectory(
        animal_name,
        session["position_by_epoch"][selected_epoch],
        session["timestamps_position"][selected_epoch],
        session["trajectory_intervals"][selected_epoch],
        position_offset=position_offset,
    )
    normalized_position_by_trajectory = normalize_linear_position_by_trajectory(
        animal_name,
        linear_position_by_trajectory,
    )
    task_progression_by_trajectory = {
        trajectory_type: orient_panel_e_task_progression(
            normalized_position,
            trajectory_type,
        )
        for trajectory_type, normalized_position in normalized_position_by_trajectory.items()
    }

    spikes = session["spikes_by_region"][region]
    spike_times_s = get_unit_spike_times(spikes, unit_id)
    bin_edges = build_normalized_position_bins(position_bin_count)
    fallback_position = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    raster_positions: dict[str, list[np.ndarray]] = {}
    firing_rates: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for trajectory_type in TRAJECTORY_TYPES:
        task_progression = task_progression_by_trajectory[trajectory_type]
        task_progression_interpolator = make_linear_position_interpolator(task_progression)
        trial_positions = compute_trial_spike_positions(
            spike_times_s,
            session["trajectory_intervals"][selected_epoch][trajectory_type],
            task_progression_interpolator,
        )
        raster_positions[trajectory_type] = [
            positions[(positions >= 0.0) & (positions <= 1.0)]
            for positions in trial_positions
        ]

        movement_epochs = session["trajectory_intervals"][selected_epoch][
            trajectory_type
        ].intersect(session["movement_by_run"][selected_epoch])
        tuning_curve = compute_place_tuning_curve(
            spikes,
            task_progression,
            movement_epochs,
            bin_edges=bin_edges,
            sigma_bins=sigma_bins,
        )
        firing_rates[trajectory_type] = extract_unit_rate_curve(
            tuning_curve,
            unit_id,
            fallback_position,
        )

    return {
        "animal_name": animal_name,
        "date": date,
        "epoch": selected_epoch,
        "region": region,
        "unit_id": unit_id,
        "raster_positions": raster_positions,
        "firing_rates": firing_rates,
    }


def build_panel_e_cache_metadata(
    *,
    data_root: Path,
    animal_name: str,
    date: str,
    epoch: str,
    region: str,
    unit_id: int,
    position_bin_count: int,
    position_offset: int,
    speed_threshold_cm_s: float,
    sigma_bins: float,
) -> dict[str, Any]:
    """Return metadata that identifies one Panel E example-cell cache."""
    return {
        "cache_version": PANEL_E_CACHE_VERSION,
        "data_root": str(Path(data_root)),
        "animal_name": str(animal_name),
        "date": str(date),
        "epoch": str(epoch),
        "region": str(region),
        "unit_id": int(unit_id),
        "trajectory_types": list(TRAJECTORY_TYPES),
        "position_bin_count": int(position_bin_count),
        "position_offset": int(position_offset),
        "speed_threshold_cm_s": float(speed_threshold_cm_s),
        "sigma_bins": float(sigma_bins),
    }


def build_panel_e_cache_path(cache_dir: Path, metadata: dict[str, Any]) -> Path:
    """Return the descriptive cache path for one Panel E example-cell payload."""
    dataset_token = "-".join(
        _format_panel_d_cache_token(value)
        for value in (
            metadata["animal_name"],
            metadata["date"],
            metadata["epoch"],
            metadata["region"],
            f"unit{metadata['unit_id']}",
        )
    )
    filename = (
        f"{PANEL_E_CACHE_PREFIX}_{dataset_token}"
        f"_posbins{int(metadata['position_bin_count'])}"
        f"_offset{int(metadata['position_offset'])}"
        f"_speed{_format_panel_d_cache_number(metadata['speed_threshold_cm_s'])}"
        f"_sigma{_format_panel_d_cache_number(metadata['sigma_bins'])}"
        f"_cachev{int(metadata['cache_version'])}.npz"
    )
    return Path(cache_dir) / filename


def _panel_e_cache_trajectory_token(trajectory_type: str) -> str:
    """Return a compact trajectory token for Panel E cache array names."""
    return _format_panel_d_cache_token(trajectory_type)


def save_panel_e_cache(
    cache_path: Path,
    example_data: dict[str, Any],
    metadata: dict[str, Any],
) -> None:
    """Write one Panel E example-cell cache as compressed NumPy arrays."""
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        PANEL_E_CACHE_METADATA_KEY: np.asarray(json.dumps(metadata, sort_keys=True))
    }
    for trajectory_type in TRAJECTORY_TYPES:
        token = _panel_e_cache_trajectory_token(trajectory_type)
        raster_trials = [
            np.asarray(trial_positions, dtype=float)
            for trial_positions in example_data["raster_positions"][trajectory_type]
        ]
        if raster_trials:
            payload[f"raster_{token}_values"] = np.concatenate(raster_trials)
            payload[f"raster_{token}_lengths"] = np.asarray(
                [trial_positions.size for trial_positions in raster_trials],
                dtype=int,
            )
        else:
            payload[f"raster_{token}_values"] = np.asarray([], dtype=float)
            payload[f"raster_{token}_lengths"] = np.asarray([], dtype=int)

        rate_position, rate_values = example_data["firing_rates"][trajectory_type]
        payload[f"rate_{token}_position"] = np.asarray(rate_position, dtype=float)
        payload[f"rate_{token}_values"] = np.asarray(rate_values, dtype=float)

    np.savez_compressed(cache_path, **payload)


def load_panel_e_cache(
    cache_path: Path,
    expected_metadata: dict[str, Any],
) -> dict[str, Any] | None:
    """Return cached Panel E example-cell data when metadata still matches."""
    cache_path = Path(cache_path)
    if not cache_path.exists():
        return None

    try:
        with np.load(cache_path, allow_pickle=False) as data:
            cached_metadata = json.loads(str(data[PANEL_E_CACHE_METADATA_KEY].item()))
            if cached_metadata != expected_metadata:
                print(f"Ignoring stale Panel E cache at {cache_path}.")
                return None

            raster_positions: dict[str, list[np.ndarray]] = {}
            firing_rates: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            for trajectory_type in TRAJECTORY_TYPES:
                token = _panel_e_cache_trajectory_token(trajectory_type)
                values = np.asarray(data[f"raster_{token}_values"], dtype=float)
                lengths = np.asarray(data[f"raster_{token}_lengths"], dtype=int)
                split_points = np.cumsum(lengths)[:-1]
                raster_positions[trajectory_type] = (
                    [
                        np.asarray(trial_positions, dtype=float)
                        for trial_positions in np.split(values, split_points)
                    ]
                    if lengths.size
                    else []
                )
                firing_rates[trajectory_type] = (
                    np.asarray(data[f"rate_{token}_position"], dtype=float),
                    np.asarray(data[f"rate_{token}_values"], dtype=float),
                )
    except (KeyError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Ignoring unreadable Panel E cache at {cache_path}: {exc}")
        return None

    return {
        "animal_name": str(expected_metadata["animal_name"]),
        "date": str(expected_metadata["date"]),
        "epoch": str(expected_metadata["epoch"]),
        "region": str(expected_metadata["region"]),
        "unit_id": int(expected_metadata["unit_id"]),
        "raster_positions": raster_positions,
        "firing_rates": firing_rates,
    }


def load_or_compute_panel_e_example_data(
    *,
    data_root: Path,
    animal_name: str,
    date: str,
    epoch: str,
    region: str,
    unit_id: int,
    position_bin_count: int,
    position_offset: int,
    speed_threshold_cm_s: float,
    sigma_bins: float,
    panel_e_cache_dir: Path | None,
    refresh_panel_e_cache: bool,
) -> dict[str, Any]:
    """Load cached Panel E example-cell data or compute and cache it."""
    metadata = build_panel_e_cache_metadata(
        data_root=data_root,
        animal_name=animal_name,
        date=date,
        epoch=epoch,
        region=region,
        unit_id=unit_id,
        position_bin_count=position_bin_count,
        position_offset=position_offset,
        speed_threshold_cm_s=speed_threshold_cm_s,
        sigma_bins=sigma_bins,
    )
    cache_path = (
        build_panel_e_cache_path(panel_e_cache_dir, metadata)
        if panel_e_cache_dir is not None
        else None
    )
    if cache_path is not None and not refresh_panel_e_cache:
        cached_example = load_panel_e_cache(cache_path, metadata)
        if cached_example is not None:
            print(f"Loaded Panel E cache from {cache_path}.")
            return cached_example

    example_data = load_panel_e_example_data(
        data_root=data_root,
        animal_name=animal_name,
        date=date,
        epoch=epoch,
        region=region,
        unit_id=unit_id,
        position_bin_count=position_bin_count,
        position_offset=position_offset,
        speed_threshold_cm_s=speed_threshold_cm_s,
        sigma_bins=sigma_bins,
    )
    if cache_path is not None:
        save_panel_e_cache(cache_path, example_data, metadata)
        print(f"Saved Panel E cache to {cache_path}.")
    return example_data


def _concatenate_unit_parts(parts: list[np.ndarray]) -> np.ndarray:
    """Concatenate pooled unit-key chunks."""
    if not parts:
        return np.asarray([], dtype=object)
    return np.concatenate(parts).astype(object, copy=False)


def _concatenate_value_parts(parts: list[np.ndarray], position_bin_count: int) -> np.ndarray:
    """Concatenate pooled tuning-matrix chunks."""
    if not parts:
        return np.empty((0, position_bin_count), dtype=float)
    return np.vstack(parts)


def build_pooled_panel_values(
    curve_sets: Sequence[dict[str, Any]],
    *,
    position_bin_count: int,
) -> dict[tuple[str, str], np.ndarray]:
    """Return normalized heatmap panels pooled across data sets."""
    odd_units_by_trajectory: dict[str, list[np.ndarray]] = {
        trajectory_type: [] for trajectory_type in TRAJECTORY_TYPES
    }
    odd_values_by_trajectory: dict[str, list[np.ndarray]] = {
        trajectory_type: [] for trajectory_type in TRAJECTORY_TYPES
    }
    even_units_by_trajectory: dict[str, list[np.ndarray]] = {
        trajectory_type: [] for trajectory_type in TRAJECTORY_TYPES
    }
    even_values_by_trajectory: dict[str, list[np.ndarray]] = {
        trajectory_type: [] for trajectory_type in TRAJECTORY_TYPES
    }

    for curve_set in curve_sets:
        animal_name = str(curve_set["animal_name"])
        date = str(curve_set["date"])
        region = str(curve_set["region"])
        for trajectory_type in TRAJECTORY_TYPES:
            odd_curve = curve_set["odd_curves"].get(trajectory_type)
            if odd_curve is not None:
                units, values = extract_tuning_curve_arrays(odd_curve)
                odd_units_by_trajectory[trajectory_type].append(
                    build_unit_keys(animal_name, date, region, units)
                )
                odd_values_by_trajectory[trajectory_type].append(values)

            even_curve = curve_set["even_curves"].get(trajectory_type)
            if even_curve is not None:
                units, values = extract_tuning_curve_arrays(even_curve)
                even_units_by_trajectory[trajectory_type].append(
                    build_unit_keys(animal_name, date, region, units)
                )
                even_values_by_trajectory[trajectory_type].append(values)

    panels: dict[tuple[str, str], np.ndarray] = {}
    for order_trajectory in TRAJECTORY_TYPES:
        reference_units = _concatenate_unit_parts(odd_units_by_trajectory[order_trajectory])
        order_values = _concatenate_value_parts(
            odd_values_by_trajectory[order_trajectory],
            position_bin_count,
        )
        if reference_units.size and has_plottable_values(order_values):
            unit_order = compute_unit_order(order_values)
        else:
            unit_order = np.asarray([], dtype=int)

        for plot_trajectory in TRAJECTORY_TYPES:
            display_units = _concatenate_unit_parts(even_units_by_trajectory[plot_trajectory])
            display_values = _concatenate_value_parts(
                even_values_by_trajectory[plot_trajectory],
                position_bin_count,
            )
            if unit_order.size == 0 or display_units.size == 0:
                panels[(order_trajectory, plot_trajectory)] = np.full(
                    (reference_units.size, position_bin_count),
                    np.nan,
                    dtype=float,
                )
                continue
            panels[(order_trajectory, plot_trajectory)] = align_and_normalize_panel_values(
                display_values,
                display_units,
                reference_units,
                unit_order,
            )
    return panels


def _format_panel_d_cache_token(value: object) -> str:
    """Return a filesystem-safe token for Panel D cache file names."""
    text = str(value).strip()
    cleaned = []
    for character in text:
        if character.isalnum() or character in {"-", "_"}:
            cleaned.append(character)
        elif character == ".":
            cleaned.append("p")
        else:
            cleaned.append("-")
    token = "".join(cleaned).strip("-")
    while "--" in token:
        token = token.replace("--", "-")
    return token or "none"


def _format_panel_d_cache_number(value: float | int) -> str:
    """Return a compact numeric token for Panel D cache file names."""
    return _format_panel_d_cache_token(f"{float(value):g}")


def _build_panel_d_dataset_cache_token(
    dataset_metadata: Sequence[dict[str, str]],
) -> str:
    """Return a descriptive cache token for the Panel D data-set list."""
    dataset_tokens = [
        _format_panel_d_cache_token(
            f"{dataset['animal_name']}-{dataset['date']}-{dataset['dark_epoch']}"
        )
        for dataset in dataset_metadata
    ]
    token = "_".join(dataset_tokens) or "none"
    if len(token) <= PANEL_D_CACHE_DATASET_TOKEN_LIMIT:
        return token

    digest = hashlib.sha1(token.encode("utf-8")).hexdigest()[:12]
    prefix = "_".join(dataset_tokens[:2])
    return _format_panel_d_cache_token(
        f"{prefix}_{len(dataset_tokens)}datasets_{digest}"
    )


def build_panel_d_cache_metadata(
    *,
    data_root: Path,
    datasets: Sequence[DatasetId],
    region: str,
    position_bin_count: int,
    position_offset: int,
    speed_threshold_cm_s: float,
    sigma_bins: float,
) -> dict[str, Any]:
    """Return metadata that identifies one Panel D heatmap cache."""
    dataset_metadata = []
    for dataset in datasets:
        animal_name, date, dark_epoch = normalize_dataset_id(dataset)
        dataset_metadata.append(
            {
                "animal_name": animal_name,
                "date": date,
                "dark_epoch": dark_epoch,
            }
        )

    return {
        "cache_version": PANEL_D_CACHE_VERSION,
        "figure": DEFAULT_OUTPUT_NAME,
        "panel": "D",
        "data_root": str(Path(data_root)),
        "region": str(region),
        "datasets": dataset_metadata,
        "trajectory_types": list(TRAJECTORY_TYPES),
        "position_bin_count": int(position_bin_count),
        "position_offset": int(position_offset),
        "speed_threshold_cm_s": float(speed_threshold_cm_s),
        "sigma_bins": float(sigma_bins),
        "pooled_builder": "build_pooled_panel_values",
    }


def build_panel_d_cache_path(cache_dir: Path, metadata: dict[str, Any]) -> Path:
    """Return the descriptive cache path for one Panel D heatmap payload."""
    region_token = _format_panel_d_cache_token(metadata["region"])
    dataset_metadata = metadata["datasets"]
    dark_epochs = [
        _format_panel_d_cache_token(dataset["dark_epoch"])
        for dataset in dataset_metadata
    ]
    unique_dark_epochs = list(dict.fromkeys(dark_epochs))
    dark_epoch_token = (
        unique_dark_epochs[0]
        if len(unique_dark_epochs) == 1
        else "mixed-" + "_".join(unique_dark_epochs)
    )
    dataset_token = _build_panel_d_dataset_cache_token(dataset_metadata)
    filename = (
        f"{PANEL_D_CACHE_PREFIX}_{region_token}_dark{dark_epoch_token}"
        f"_datasets-{dataset_token}"
        f"_posbins{int(metadata['position_bin_count'])}"
        f"_offset{int(metadata['position_offset'])}"
        f"_speed{_format_panel_d_cache_number(metadata['speed_threshold_cm_s'])}"
        f"_sigma{_format_panel_d_cache_number(metadata['sigma_bins'])}"
        f"_cachev{int(metadata['cache_version'])}.npz"
    )
    return Path(cache_dir) / filename


def _panel_d_cache_array_name(order_trajectory: str, plot_trajectory: str) -> str:
    """Return the array name for one Panel D heatmap matrix."""
    return f"{order_trajectory}__{plot_trajectory}"


def save_panel_d_cache(
    cache_path: Path,
    panels: dict[tuple[str, str], np.ndarray],
    metadata: dict[str, Any],
) -> None:
    """Write one Panel D heatmap cache as compressed NumPy arrays."""
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        PANEL_D_CACHE_METADATA_KEY: np.asarray(json.dumps(metadata, sort_keys=True)),
    }
    for order_trajectory in TRAJECTORY_TYPES:
        for plot_trajectory in TRAJECTORY_TYPES:
            payload[_panel_d_cache_array_name(order_trajectory, plot_trajectory)] = (
                np.asarray(panels[(order_trajectory, plot_trajectory)], dtype=float)
            )
    np.savez_compressed(cache_path, **payload)


def load_panel_d_cache(
    cache_path: Path,
    expected_metadata: dict[str, Any],
) -> dict[tuple[str, str], np.ndarray] | None:
    """Return cached Panel D heatmap matrices when metadata still matches."""
    cache_path = Path(cache_path)
    if not cache_path.exists():
        return None

    try:
        with np.load(cache_path, allow_pickle=False) as data:
            cached_metadata = json.loads(str(data[PANEL_D_CACHE_METADATA_KEY].item()))
            if cached_metadata != expected_metadata:
                print(f"Ignoring stale Panel D cache at {cache_path}.")
                return None

            panels: dict[tuple[str, str], np.ndarray] = {}
            for order_trajectory in TRAJECTORY_TYPES:
                for plot_trajectory in TRAJECTORY_TYPES:
                    array_name = _panel_d_cache_array_name(
                        order_trajectory,
                        plot_trajectory,
                    )
                    panels[(order_trajectory, plot_trajectory)] = np.asarray(
                        data[array_name],
                        dtype=float,
                    )
            return panels
    except Exception as exc:
        print(f"Ignoring unreadable Panel D cache at {cache_path}: {exc}")
        return None


def load_or_compute_panel_d_heatmap_panels(
    *,
    data_root: Path,
    datasets: Sequence[DatasetId],
    region: str,
    position_bin_count: int,
    position_offset: int,
    speed_threshold_cm_s: float,
    sigma_bins: float,
    panel_d_cache_dir: Path | None,
    refresh_panel_d_cache: bool,
) -> dict[tuple[str, str], np.ndarray]:
    """Load cached Panel D panels or compute and cache them."""
    metadata = build_panel_d_cache_metadata(
        data_root=data_root,
        datasets=datasets,
        region=region,
        position_bin_count=position_bin_count,
        position_offset=position_offset,
        speed_threshold_cm_s=speed_threshold_cm_s,
        sigma_bins=sigma_bins,
    )
    cache_path = (
        build_panel_d_cache_path(panel_d_cache_dir, metadata)
        if panel_d_cache_dir is not None
        else None
    )
    if cache_path is not None and not refresh_panel_d_cache:
        cached_panels = load_panel_d_cache(cache_path, metadata)
        if cached_panels is not None:
            print(f"Loaded Panel D cache from {cache_path}.")
            return cached_panels

    print(f"Building pooled dark-epoch heatmap for region {region}.")
    curve_sets = []
    for dataset in datasets:
        animal_name, date, epoch = normalize_dataset_id(dataset)
        print(f"  Loading {animal_name} {date} epoch {epoch}.")
        curve_sets.append(
            compute_dark_epoch_tuning_curves(
                animal_name=animal_name,
                date=date,
                data_root=data_root,
                region=region,
                epoch=epoch,
                position_bin_count=position_bin_count,
                position_offset=position_offset,
                speed_threshold_cm_s=speed_threshold_cm_s,
                sigma_bins=sigma_bins,
            )
        )

    panels = build_pooled_panel_values(
        curve_sets,
        position_bin_count=position_bin_count,
    )
    if cache_path is not None:
        save_panel_d_cache(cache_path, panels, metadata)
        print(f"Saved Panel D cache to {cache_path}.")
    return panels


def plot_dark_heatmap_regions(
    heatmap_axes: np.ndarray,
    *,
    data_root: Path,
    datasets: Sequence[DatasetId],
    regions: Sequence[str],
    position_bin_count: int,
    position_offset: int,
    speed_threshold_cm_s: float,
    sigma_bins: float,
    panel_d_cache_dir: Path | None = None,
    refresh_panel_d_cache: bool = False,
) -> "AxesImage | None":
    """Plot pooled dark-epoch heatmaps for all requested regions."""
    color_image = None
    for region_index, region in enumerate(regions):
        panels = load_or_compute_panel_d_heatmap_panels(
            data_root=data_root,
            datasets=datasets,
            region=region,
            position_bin_count=position_bin_count,
            position_offset=position_offset,
            speed_threshold_cm_s=speed_threshold_cm_s,
            sigma_bins=sigma_bins,
            panel_d_cache_dir=panel_d_cache_dir,
            refresh_panel_d_cache=refresh_panel_d_cache,
        )
        start_row = region_index * len(TRAJECTORY_TYPES)
        stop_row = start_row + len(TRAJECTORY_TYPES)
        image = plot_pooled_heatmap_grid(
            heatmap_axes[start_row:stop_row, :],
            panels,
        )
        if color_image is None and image is not None:
            color_image = image
    return color_image


def plot_pooled_heatmap_grid(
    axes: np.ndarray,
    panels: dict[tuple[str, str], np.ndarray],
) -> "AxesImage | None":
    """Plot one pooled 4x4 odd/even trajectory heatmap grid."""
    color_image = None
    for row_index, order_trajectory in enumerate(TRAJECTORY_TYPES):
        row_size = max(
            panels[(order_trajectory, plot_trajectory)].shape[0]
            for plot_trajectory in TRAJECTORY_TYPES
        )
        y_limit = max(row_size, 1)
        for col_index, plot_trajectory in enumerate(TRAJECTORY_TYPES):
            ax: Axes = axes[row_index, col_index]
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(y_limit, 0)
            ax.set_xticks([])
            ax.set_yticks([])

            if row_index == len(TRAJECTORY_TYPES) - 1:
                add_movement_axis_annotations(ax, plot_trajectory)

            panel_values = panels[(order_trajectory, plot_trajectory)]
            if not has_plottable_values(panel_values):
                add_task_progression_segment_boundary_lines(ax)
                continue
            image = ax.imshow(
                panel_values,
                origin="upper",
                aspect="auto",
                interpolation="nearest",
                extent=[0.0, 1.0, panel_values.shape[0], 0],
                vmin=0.0,
                vmax=1.0,
                cmap="viridis",
            )
            add_task_progression_segment_boundary_lines(ax)
            if color_image is None:
                color_image = image
    return color_image


def add_task_progression_segment_boundary_lines(ax: "Axes") -> None:
    """Draw normalized task-progression segment boundaries."""
    for boundary in TASK_PROGRESSION_SEGMENT_BOUNDARIES:
        ax.axvline(
            boundary,
            color=TASK_PROGRESSION_SEGMENT_BOUNDARY_COLOR,
            linewidth=TASK_PROGRESSION_SEGMENT_BOUNDARY_LINEWIDTH,
            zorder=1,
        )


def draw_neuron_scale_bar(
    ax: "Axes",
    *,
    neuron_count: int = NEURON_SCALE_BAR_COUNT,
    x: float = 1.08,
) -> None:
    """Draw a vertical data-scaled neuron count bar beside one heatmap axis."""
    from matplotlib.transforms import blended_transform_factory

    if neuron_count <= 0:
        raise ValueError("neuron_count must be positive.")

    y_limits = [float(value) for value in ax.get_ylim()]
    y_min = min(y_limits)
    y_max = max(y_limits)
    y_span = y_max - y_min
    margin = max(8.0, 0.28 * y_span)
    if y_span >= neuron_count + margin:
        y_bottom = y_max - margin
        y_top = y_bottom - float(neuron_count)
    else:
        y_top = y_min
        y_bottom = min(y_max, y_min + float(neuron_count))

    transform = blended_transform_factory(ax.transAxes, ax.transData)
    ax.plot(
        [x, x],
        [y_bottom, y_top],
        color="black",
        linewidth=1.0,
        solid_capstyle="butt",
        transform=transform,
        clip_on=False,
    )
    ax.text(
        x + 0.05,
        (y_bottom + y_top) / 2,
        f"{neuron_count} neurons",
        ha="left",
        va="center",
        rotation=90,
        transform=transform,
        clip_on=False,
    )


def draw_order_schematic(
    ax: "Axes",
    trajectory_type: str,
    *,
    arrow_color: str = "red",
    fill_track: bool = True,
) -> "Axes":
    """Draw one compact order schematic centered in a heatmap-height row."""
    ax.axis("off")
    inset = ax.inset_axes([0.0, 0.29, 1.0, 0.42])
    draw_w_track_schematic(
        inset,
        trajectory_name=trajectory_type,
        arrow_color=arrow_color,
        fill_track=fill_track,
    )
    return inset


def draw_image_asset(ax: "Axes", image: np.ndarray, *, aspect: str = "equal") -> None:
    """Draw one external raster asset in an axis."""
    ax.imshow(image, aspect=aspect)
    ax.axis("off")


def draw_panel_a_assets(
    ax: "Axes",
    *,
    asset_dir: Path,
    probe_asset_name: str = DEFAULT_PROBE_ASSET_NAME,
    histology_asset_name: str = DEFAULT_HISTOLOGY_ASSET_NAME,
    behavior_asset_name: str = DEFAULT_BEHAVIOR_ASSET_NAME,
) -> None:
    """Draw panel A external assets: probe, behavior, and histology."""
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    probe_path = get_figure_1_asset_path(asset_dir, probe_asset_name)
    histology_path = get_figure_1_asset_path(asset_dir, histology_asset_name)
    behavior_path = get_figure_1_asset_path(asset_dir, behavior_asset_name)
    probe_image = np.rot90(load_panel_asset_image(probe_path))
    histology_image = load_panel_asset_image(histology_path)
    behavior_image = load_panel_asset_image(behavior_path)

    probe_ax = ax.inset_axes([0.12, 0.508, 0.24, 0.624])
    behavior_ax = ax.inset_axes([0.00, -0.18, 0.48, 0.76])
    histology_ax = ax.inset_axes([0.47, -0.18, 0.86, 1.36])
    draw_image_asset(probe_ax, probe_image)
    draw_image_asset(behavior_ax, behavior_image)
    draw_image_asset(histology_ax, histology_image)


def draw_panel_a_anatomy_assets(
    ax: "Axes",
    *,
    asset_dir: Path,
    probe_asset_name: str = DEFAULT_PROBE_ASSET_NAME,
    histology_asset_name: str = DEFAULT_HISTOLOGY_ASSET_NAME,
) -> None:
    """Draw the Figure 1A probe and histology assets without behavior."""
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    probe_path = get_figure_1_asset_path(asset_dir, probe_asset_name)
    histology_path = get_figure_1_asset_path(asset_dir, histology_asset_name)
    probe_image = np.rot90(load_panel_asset_image(probe_path))
    histology_image = load_panel_asset_image(histology_path)

    probe_ax = ax.inset_axes([0.02, 0.21, 0.23, 0.60])
    histology_ax = ax.inset_axes([0.30, -0.08, 0.72, 1.16])
    draw_image_asset(probe_ax, probe_image)
    draw_image_asset(histology_ax, histology_image)


def draw_behavior_task_design_panel(
    ax: "Axes",
    *,
    asset_dir: Path,
    behavior_asset_name: str = DEFAULT_BEHAVIOR_ASSET_NAME,
) -> None:
    """Draw behavior and task-design schematics in one Figure 1B panel."""
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    behavior_path = get_figure_1_asset_path(asset_dir, behavior_asset_name)
    behavior_image = load_panel_asset_image(behavior_path)

    behavior_ax = ax.inset_axes([0.00, 0.11, 0.34, 0.76])
    task_ax = ax.inset_axes([0.32, 0.00, 0.68, 1.00])
    draw_image_asset(behavior_ax, behavior_image)
    draw_w_track_cycle_panel(task_ax)


def _scale_points_to_axes(
    points: Sequence[tuple[float, float]],
    *,
    bounds: tuple[float, float, float, float],
    source_xlim: tuple[float, float],
    source_ylim: tuple[float, float],
) -> list[tuple[float, float]]:
    """Scale points from one source coordinate system into axes coordinates."""
    x0, y0, width, height = bounds
    xmin, xmax = source_xlim
    ymin, ymax = source_ylim
    return [
        (
            x0 + (float(x) - xmin) / (xmax - xmin) * width,
            y0 + (float(y) - ymin) / (ymax - ymin) * height,
        )
        for x, y in points
    ]


def _scale_rect_to_axes(
    rect: tuple[float, float, float, float],
    *,
    bounds: tuple[float, float, float, float],
    source_xlim: tuple[float, float],
    source_ylim: tuple[float, float],
) -> tuple[float, float, float, float]:
    """Scale one rectangle from source coordinates into axes coordinates."""
    (x, y, width, height) = rect
    (x0, y0), (x1, y1) = _scale_points_to_axes(
        [(x, y), (x + width, y + height)],
        bounds=bounds,
        source_xlim=source_xlim,
        source_ylim=source_ylim,
    )
    return x0, y0, x1 - x0, y1 - y0


def _get_axes_display_aspect(ax: "Axes") -> float:
    """Return physical width/height for one axes."""
    fig_width, fig_height = ax.figure.get_size_inches()
    box = ax.get_position()
    height = box.height * fig_height
    if height <= 0.0:
        return 1.0
    return (box.width * fig_width) / height


def draw_visual_stimuli_schematic(ax: "Axes") -> None:
    """Draw a compact W-track and visual-stimulus sequence in panel B."""
    from matplotlib.patches import Ellipse, Polygon, Rectangle

    transform = ax.transAxes
    outline, points, dims = get_w_track_geometry()
    source_xlim = (dims["x0"] - 0.35, dims["x5"] + 0.35)
    source_ylim = (dims["y0"] - 0.55, dims["y2"] + 0.45)
    axes_aspect = _get_axes_display_aspect(ax)
    track_height = 0.235
    track_width = track_height * (
        (source_xlim[1] - source_xlim[0])
        / (source_ylim[1] - source_ylim[0])
        / axes_aspect
    )
    track_bounds = (0.045, 0.020, track_width, track_height)
    scaled_outline = _scale_points_to_axes(
        outline,
        bounds=track_bounds,
        source_xlim=source_xlim,
        source_ylim=source_ylim,
    )

    monitor_color = SCHEMATIC_COLORS["visual_stimulus"]
    monitor_bar_w = 0.16
    monitor_y = dims["y1"] + 0.12
    monitor_h = dims["y2"] - dims["y1"] - 0.32
    monitor_source_rects = (
        (dims["x0"] - 0.24, monitor_y, monitor_bar_w, monitor_h),
        (dims["x1"] + 0.10, monitor_y, monitor_bar_w, monitor_h),
        (dims["x4"] - 0.26, monitor_y, monitor_bar_w, monitor_h),
        (dims["x5"] + 0.10, monitor_y, monitor_bar_w, monitor_h),
        (
            dims["x0"],
            dims["y0"] - 0.42,
            dims["x5"] - dims["x0"],
            monitor_bar_w,
        ),
    )
    for source_rect in monitor_source_rects:
        rect = _scale_rect_to_axes(
            source_rect,
            bounds=track_bounds,
            source_xlim=source_xlim,
            source_ylim=source_ylim,
        )
        ax.add_patch(
            Rectangle(
                rect[:2],
                rect[2],
                rect[3],
                facecolor=monitor_color,
                edgecolor="none",
                alpha=0.90,
                transform=transform,
                zorder=1,
            )
        )
    wall_top_y = dims["y2"] + 0.20
    wall_bottom_y = dims["y1"] + 0.06
    wall_color = "0.25"
    wall_linewidth = 0.8
    horizontal_wall = _scale_points_to_axes(
        [
            (dims["x0"] - 0.18, wall_top_y),
            (dims["x5"] + 0.18, wall_top_y),
        ],
        bounds=track_bounds,
        source_xlim=source_xlim,
        source_ylim=source_ylim,
    )
    ax.plot(
        [horizontal_wall[0][0], horizontal_wall[1][0]],
        [horizontal_wall[0][1], horizontal_wall[1][1]],
        color=wall_color,
        linewidth=wall_linewidth,
        transform=transform,
        clip_on=False,
        zorder=4,
    )
    wall_x_positions = (
        (dims["x1"] + dims["x2"]) / 2,
        (dims["x3"] + dims["x4"]) / 2,
    )
    for wall_x in wall_x_positions:
        (x, y0), (_x, y1) = _scale_points_to_axes(
            [(wall_x, wall_bottom_y), (wall_x, wall_top_y)],
            bounds=track_bounds,
            source_xlim=source_xlim,
            source_ylim=source_ylim,
        )
        ax.plot(
            [x, x],
            [y0, y1],
            color=wall_color,
            linewidth=wall_linewidth,
            transform=transform,
            clip_on=False,
            zorder=4,
        )
    ax.add_patch(
        Polygon(
            scaled_outline,
            closed=True,
            facecolor="white",
            edgecolor="black",
            linewidth=0.8,
            transform=transform,
            zorder=2,
        )
    )
    label_points = _scale_points_to_axes(
        [
            (points["left"][0], wall_top_y + 0.10),
            (points["center"][0], wall_top_y + 0.10),
            (points["right"][0], wall_top_y + 0.10),
        ],
        bounds=track_bounds,
        source_xlim=source_xlim,
        source_ylim=source_ylim,
    )
    for (x, y), label in zip(label_points, ("L", "C", "R"), strict=True):
        ax.text(
            x,
            y,
            label,
            ha="center",
            va="bottom",
            fontsize=5.2,
            color="black",
            transform=transform,
            zorder=4,
        )

    screen_y = 0.015
    screen_h = 0.105
    screen_w = screen_h / axes_aspect * 1.10
    screen_gap = 0.055
    screen_specs = (
        (0.405, "black"),
        (0.405 + screen_w + screen_gap, "grating"),
        (0.405 + 2 * (screen_w + screen_gap), "dots"),
    )
    stimulus_center = (screen_specs[0][0] + screen_specs[-1][0] + screen_w) / 2

    def add_display_circle(
        x: float,
        y: float,
        radius: float,
        color: str,
        *,
        zorder: int = 3,
    ) -> None:
        ax.add_patch(
            Ellipse(
                (x, y),
                width=2 * radius / axes_aspect,
                height=2 * radius,
                facecolor=color,
                edgecolor="none",
                transform=transform,
                zorder=zorder,
            )
        )

    ax.text(
        stimulus_center,
        0.135,
        "Visual stimuli",
        ha="center",
        va="bottom",
        fontsize=8.5,
        transform=transform,
        zorder=4,
    )

    ax.add_patch(
        Rectangle(
            (0.315, 0.025),
            0.014,
            0.100,
            facecolor=monitor_color,
            edgecolor="none",
            alpha=0.90,
            transform=transform,
            zorder=2,
        )
    )
    for y in (0.060, 0.095):
        ax.add_patch(
            Rectangle(
                (0.350, y),
                0.012,
                0.012,
                facecolor="black",
                edgecolor="black",
                transform=transform,
                zorder=3,
            )
        )

    for x0, screen_type in screen_specs:
        if screen_type == "black":
            facecolor = "black"
        elif screen_type == "dots":
            facecolor = "0.65"
        else:
            facecolor = "white"
        ax.add_patch(
            Rectangle(
                (x0, screen_y),
                screen_w,
                screen_h,
                facecolor=facecolor,
                edgecolor="black",
                linewidth=0.6,
                transform=transform,
                zorder=2,
            )
        )
        if screen_type == "grating":
            stripe_w = screen_w / 8.0
            for stripe_index in range(0, 8, 2):
                ax.add_patch(
                    Rectangle(
                        (x0 + stripe_index * stripe_w, screen_y),
                        stripe_w,
                        screen_h,
                        facecolor="black",
                        edgecolor="none",
                        transform=transform,
                        zorder=3,
                    )
                )
        elif screen_type == "dots":
            dot_specs = (
                (0.187, 0.696, 0.168, "white"),
                (0.432, 0.320, 0.104, "white"),
                (0.665, 0.688, 0.080, "black"),
                (0.852, 0.624, 0.120, "white"),
                (0.148, 0.192, 0.112, "black"),
                (0.652, 0.184, 0.104, "white"),
                (0.813, 0.240, 0.160, "black"),
                (0.587, 0.432, 0.064, "black"),
            )
            for x_frac, y_frac, radius_frac, color in dot_specs:
                add_display_circle(
                    x0 + x_frac * screen_w,
                    screen_y + y_frac * screen_h,
                    radius_frac * screen_h,
                    color,
                )

    ellipsis_start = screen_specs[-1][0] + screen_w + 0.050
    for x in (ellipsis_start, ellipsis_start + 0.030, ellipsis_start + 0.060):
        add_display_circle(x, screen_y + 0.52 * screen_h, 0.006, "black")


def draw_w_track_cycle_panel(ax: "Axes") -> None:
    """Draw the four-trajectory W-track task cycle schematic."""
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    for trajectory_type, bounds in CYCLE_TRAJECTORY_LAYOUT:
        inset = ax.inset_axes(bounds)
        draw_w_track_schematic(
            inset,
            trajectory_name=trajectory_type,
            arrow_color=PANEL_E_TRAJECTORY_COLORS[trajectory_type],
            track_linewidth=1.2,
            trajectory_linewidth=1.4,
            arrow_mutation_scale=13.0,
            fill_track=False,
        )

    for start, end, rad in CYCLE_ARROW_SPECS:
        ax.annotate(
            "",
            xy=end,
            xytext=start,
            xycoords="axes fraction",
            textcoords="axes fraction",
            arrowprops={
                "arrowstyle": "-|>",
                "color": "black",
                "linewidth": CYCLE_ARROW_LINEWIDTH,
                "mutation_scale": CYCLE_ARROW_MUTATION_SCALE,
                "shrinkA": 0,
                "shrinkB": 0,
                "connectionstyle": f"arc3,rad={rad}",
            },
            annotation_clip=False,
        )
    draw_visual_stimuli_schematic(ax)


def add_w_track_arm_endpoint_labels(
    ax: "Axes",
    *,
    fontsize: float = 6.0,
) -> None:
    """Add center, left, and right arm endpoint labels to one W-track axis."""
    _outline, points, dims = get_w_track_geometry()
    y = dims["y2"] + 0.04
    for arm_name, label in (("left", "L"), ("center", "C"), ("right", "R")):
        x, _arm_y = points[arm_name]
        ax.text(
            x,
            y,
            label,
            ha="center",
            va="bottom",
            fontsize=fontsize,
            color="black",
            zorder=10,
        )


def _fraction_histogram_weights(values: np.ndarray) -> np.ndarray:
    """Return weights that normalize one histogram to a fraction of units."""
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size == 0:
        return np.asarray([], dtype=float)
    return np.full(values.shape, 1.0 / float(values.size), dtype=float)


def _format_delta_summary(values: np.ndarray, *, label: str | None = None) -> str:
    """Return fraction-positive and median text for delta log-likelihood values."""
    values = np.asarray(values, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    prefix = "" if label is None else f"{label}\n"
    if values.size == 0:
        return f"{prefix}frac >0: n/a\nmedian: n/a"
    fraction_positive = float(np.mean(values > 0.0))
    median = float(np.median(values))
    return f"{prefix}frac >0: {fraction_positive:.2f}\nmedian: {median:.2f}"


def _format_delta_advantage_summary(
    values: np.ndarray,
    *,
    label: str | None = None,
) -> str:
    """Return compact DPP-side summary text for delta log-likelihood values."""
    values = np.asarray(values, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    prefix = "" if label is None else f"{label}\n"
    if values.size == 0:
        return f"{prefix}n/a >0, med. n/a"
    fraction_positive = float(np.mean(values > 0.0))
    median = float(np.median(values))
    return f"{prefix}{fraction_positive:.0%} >0, med. {median:.2f}"


def _format_cell_animal_count(
    table: Any,
    *,
    value_column: str | None = None,
) -> str:
    """Return a compact unique cell and animal count label for panel annotations."""
    columns = set(getattr(table, "columns", []))
    count_table = table
    if value_column is not None and value_column in columns:
        count_table = table[np.isfinite(np.asarray(table[value_column], dtype=float))]

    if {"animal_name", "date", "unit"}.issubset(columns):
        cell_columns = ["animal_name", "date", "unit"]
        n_cells = int(count_table.loc[:, cell_columns].drop_duplicates().shape[0])
    elif "unit" in columns:
        n_cells = int(count_table["unit"].nunique())
    else:
        n_cells = int(len(count_table))

    n_animals = (
        int(count_table["animal_name"].nunique()) if "animal_name" in columns else 0
    )
    cell_word = "cell" if n_cells == 1 else "cells"
    animal_word = "animal" if n_animals == 1 else "animals"
    return f"n = {n_cells} {cell_word}\n{n_animals} {animal_word}"


def build_zero_including_histogram_bins(
    values: np.ndarray,
    *,
    n_bins: int | None = None,
) -> np.ndarray:
    """Return histogram bin edges with zero included as a bin edge."""
    values = np.asarray(values, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.linspace(-0.1, 0.1, 17)
    if n_bins is not None:
        span = max(abs(float(np.nanmin(values))), abs(float(np.nanmax(values))), 0.1)
        bin_width = (2.0 * span) / max(int(n_bins), 1)
        n_left = int(np.ceil(abs(float(np.nanmin(values))) / bin_width))
        n_right = int(np.ceil(abs(float(np.nanmax(values))) / bin_width))
        bin_edges = bin_width * np.arange(-n_left, n_right + 1, dtype=float)
        if bin_edges.size < 2:
            return np.asarray([-bin_width, 0.0, bin_width], dtype=float)
        return bin_edges
    if np.allclose(values, values[0]):
        half_width = max(0.1, abs(float(values[0])) * 0.1 + 0.1)
        bin_edges = np.linspace(
            float(values[0]) - half_width,
            float(values[0]) + half_width,
            17,
        )
    else:
        bin_edges = np.histogram_bin_edges(values, bins="auto")

    if not np.any(np.isclose(bin_edges, 0.0)):
        bin_edges = np.sort(np.unique(np.concatenate([bin_edges, np.array([0.0])])))
    return np.asarray(bin_edges, dtype=float)


def plot_stability_panel(
    ax: "Axes",
    stability_table: Any,
    *,
    regions: Sequence[str] = STABILITY_REGIONS,
) -> None:
    """Plot pooled odd/even stability histograms with trajectory schematics."""
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    bins = np.linspace(-1.0, 1.0, 25)
    cell_width = 0.46
    cell_height = 0.43
    x_positions = (0.02, 0.52)
    y_positions = (0.53, 0.04)
    legend_handles = []
    legend_labels = []

    for row, trajectory_row in enumerate(STABILITY_TRAJECTORY_LAYOUT):
        for col, trajectory_type in enumerate(trajectory_row):
            x0 = x_positions[col]
            y0 = y_positions[row]
            schematic_ax = ax.inset_axes(
                [x0 + 0.16, y0 + cell_height * 0.67, 0.14, 0.17]
            )
            draw_w_track_schematic(
                schematic_ax,
                trajectory_name=trajectory_type,
                arrow_color=PANEL_E_TRAJECTORY_COLORS[trajectory_type],
                track_linewidth=0.5,
                trajectory_linewidth=0.75,
                arrow_mutation_scale=6.5,
                fill_track=True,
            )

            hist_ax = ax.inset_axes([x0, y0, cell_width, cell_height * 0.62])
            trajectory_rows = stability_table[
                stability_table["trajectory_type"].astype(str) == trajectory_type
            ]
            for region in regions:
                values = np.asarray(
                    trajectory_rows.loc[
                        trajectory_rows["region"].astype(str) == region,
                        "stability_correlation",
                    ],
                    dtype=float,
                )
                values = values[np.isfinite(values)]
                if values.size == 0:
                    continue
                counts, _edges, patches = hist_ax.hist(
                    values,
                    bins=bins,
                    weights=_fraction_histogram_weights(values),
                    color=STABILITY_REGION_COLORS.get(region),
                    **HISTOGRAM_KWARGS,
                )
                del counts
                if row == 0 and col == 0 and len(patches) > 0:
                    legend_handles.append(patches[0])
                    legend_labels.append(region.upper())

            hist_ax.set_xlim(-1.0, 1.0)
            hist_ax.set_ylim(bottom=0.0)
            hist_ax.spines["top"].set_visible(False)
            hist_ax.spines["right"].set_visible(False)
            hist_ax.tick_params(
                labelsize=STABILITY_TICK_LABEL_FONTSIZE,
                length=2,
                pad=1,
            )
            if row == len(STABILITY_TRAJECTORY_LAYOUT) - 1:
                hist_ax.set_xlabel(
                    "Odd/even corr.",
                    fontsize=STABILITY_AXIS_LABEL_FONTSIZE,
                    labelpad=1,
                )
            else:
                hist_ax.set_xticklabels([])
            if col == 0:
                hist_ax.set_ylabel(
                    "Frac.",
                    fontsize=STABILITY_AXIS_LABEL_FONTSIZE,
                    labelpad=1,
                )
            else:
                hist_ax.set_yticklabels([])

    if legend_handles:
        ax.legend(
            legend_handles,
            legend_labels,
            loc="upper left",
            bbox_to_anchor=(0.0, 1.0),
            frameon=False,
            fontsize=STABILITY_LEGEND_FONTSIZE,
            handlelength=1.0,
            borderpad=0.1,
            labelspacing=0.2,
        )


def plot_motor_delta_panel(ax: "Axes", motor_delta_table: Any) -> None:
    """Plot pooled V1 motor+DPP versus motor delta log-likelihood values."""
    x_limits = (-1.0, 1.0)
    bin_edges = np.round(np.arange(x_limits[0], x_limits[1] + 0.05, 0.1), 10)
    values = np.asarray(
        motor_delta_table["delta_log_likelihood_bits_per_spike"],
        dtype=float,
    )
    values = values[np.isfinite(values)]

    ax.axvspan(
        x_limits[0],
        0.0,
        color=NEUTRAL_COLORS["dark_epoch_background"],
        alpha=0.65,
        linewidth=0,
        zorder=0,
    )
    ax.axvline(0.0, color="black", linestyle="--", linewidth=0.8, zorder=1)
    if values.size == 0:
        ax.text(
            0.5,
            0.5,
            "No finite values",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
    else:
        ax.hist(
            values,
            bins=bin_edges,
            weights=_fraction_histogram_weights(values),
            color=REGION_COLORS.get(MOTOR_DELTA_REGION, REGION_COLORS["v1"]),
            **EMPHASIS_HISTOGRAM_KWARGS,
            zorder=2,
        )

    ax.text(
        0.03,
        0.97,
        "Motor only better",
        ha="left",
        va="top",
        fontsize=5.5,
        transform=ax.transAxes,
    )
    ax.text(
        0.68,
        0.97,
        "Motor+DGP better",
        ha="left",
        va="top",
        fontsize=5.5,
        transform=ax.transAxes,
    )
    ax.text(
        0.68,
        0.74,
        _format_delta_advantage_summary(values),
        ha="left",
        va="top",
        fontsize=4.8,
        color=REGION_COLORS.get(MOTOR_DELTA_REGION, REGION_COLORS["v1"]),
        transform=ax.transAxes,
    )
    ax.text(
        0.03,
        0.06,
        _format_cell_animal_count(
            motor_delta_table,
            value_column="delta_log_likelihood_bits_per_spike",
        ),
        ha="left",
        va="bottom",
        fontsize=4.8,
        color="0.25",
        transform=ax.transAxes,
    )
    ax.set_xlim(*x_limits)
    ax.set_xlabel(DELTA_LOG_LIKELIHOOD_AXIS_LABEL, fontsize=7, labelpad=2)
    ax.set_ylabel("Frac.", fontsize=8, labelpad=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=7, length=2, pad=1)


def plot_encoding_delta_panel(ax: "Axes", encoding_delta_table: Any) -> None:
    """Plot pooled V1 DPP-versus-absolute-model delta log-likelihood values."""
    x_limits = (-1.0, 1.0)
    bin_edges = np.round(np.arange(x_limits[0], x_limits[1] + 0.05, 0.1), 10)
    all_values = np.asarray(
        encoding_delta_table["delta_log_likelihood_bits_per_spike"],
        dtype=float,
    )
    all_values = all_values[np.isfinite(all_values)]

    ax.axvspan(
        x_limits[0],
        0.0,
        color=NEUTRAL_COLORS["dark_epoch_background"],
        alpha=0.65,
        linewidth=0,
        zorder=0,
    )
    ax.axvline(0.0, color="black", linestyle="--", linewidth=0.8, zorder=1)
    summary_rows = []
    if all_values.size == 0:
        ax.text(
            0.5,
            0.5,
            "No finite values",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
    else:
        for comparison, label, _source_column in ENCODING_DPP_COMPARISONS:
            values = np.asarray(
                encoding_delta_table.loc[
                    encoding_delta_table["comparison"].astype(str) == comparison,
                    "delta_log_likelihood_bits_per_spike",
                ],
                dtype=float,
            )
            values = values[np.isfinite(values)]
            if values.size == 0:
                continue
            color = ENCODING_DPP_COMPARISON_COLORS.get(comparison, "0.4")
            ax.hist(
                values,
                bins=bin_edges,
                weights=_fraction_histogram_weights(values),
                color=color,
                label=label,
                **COMPACT_HISTOGRAM_KWARGS,
                zorder=2,
            )
            median = float(np.median(values))
            summary_rows.append(
                (
                    comparison,
                    f"{np.mean(values > 0.0):.0%} >0, med. {median:.2f}",
                    color,
                )
            )

    ax.text(
        0.03,
        0.97,
        "Abs place better",
        ha="left",
        va="top",
        fontsize=4.8,
        color=ENCODING_DPP_COMPARISON_COLORS["dpp_vs_absolute_place"],
        transform=ax.transAxes,
    )
    ax.text(
        0.03,
        0.82,
        "Distance-to-reward\nbetter",
        ha="left",
        va="top",
        fontsize=4.8,
        color=ENCODING_DPP_COMPARISON_COLORS["dpp_vs_absolute_task_progression"],
        transform=ax.transAxes,
    )
    ax.text(
        0.67,
        0.97,
        "DGP better",
        ha="left",
        va="top",
        fontsize=4.8,
        color="black",
        transform=ax.transAxes,
    )
    summary_label_by_comparison = {
        "dpp_vs_absolute_place": "DGP > abs place",
        "dpp_vs_absolute_task_progression": "DGP > dist.-to-reward",
    }
    for row_index, (comparison, summary, color) in enumerate(summary_rows):
        ax.text(
            0.67,
            0.74 - 0.28 * row_index,
            f"{summary_label_by_comparison[comparison]}\n{summary}",
            ha="left",
            va="top",
            fontsize=4.2,
            color=color,
            transform=ax.transAxes,
        )
    ax.text(
        0.03,
        0.06,
        _format_cell_animal_count(
            encoding_delta_table,
            value_column="delta_log_likelihood_bits_per_spike",
        ),
        ha="left",
        va="bottom",
        fontsize=4.8,
        color="0.25",
        transform=ax.transAxes,
    )
    ax.set_xlim(*x_limits)
    ax.set_xlabel(DELTA_LOG_LIKELIHOOD_AXIS_LABEL, fontsize=7, labelpad=2)
    ax.set_ylabel("Frac.", fontsize=8, labelpad=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=7, length=2, pad=1)


def plot_decoding_error_panel(
    ax: "Axes",
    decoding_error_table: Any,
    *,
    comparisons: Sequence[
        tuple[str, str, str, Sequence[tuple[str, str]]]
    ] = DECODING_CROSS_TRAJECTORY_COMPARISONS,
) -> None:
    """Plot pooled median and IQR cross-trajectory decoding errors."""
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    plot_ax = ax.inset_axes([0.0, 0.0, 1.0, 1.0])
    positions = np.arange(1, len(comparisons) + 1, dtype=float)
    labels = [label for _comparison, label, _family, _pairs in comparisons]
    plot_table = decoding_error_table.copy()
    medians = []
    q25_values = []
    q75_values = []
    plot_positions = []
    for position, (comparison, _label, _family, _pairs) in zip(
        positions,
        comparisons,
        strict=True,
    ):
        values = np.asarray(
            plot_table.loc[
                plot_table["comparison"].astype(str) == comparison,
                "absolute_error",
            ],
            dtype=float,
        )
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        medians.append(float(np.median(values)))
        q25_values.append(float(np.quantile(values, 0.25)))
        q75_values.append(float(np.quantile(values, 0.75)))
        plot_positions.append(float(position))

    plotted_any = bool(medians)
    if plotted_any:
        color = REGION_COLORS.get(DECODING_COMPARISON_REGION, REGION_COLORS["v1"])
        plot_ax.vlines(
            plot_positions,
            q25_values,
            q75_values,
            colors=color,
            linewidth=1.0,
            alpha=0.75,
            zorder=3,
        )
        plot_ax.scatter(
            plot_positions,
            medians,
            c=color,
            s=14,
            edgecolors="black",
            linewidths=0.3,
            zorder=4,
        )
        for position, median in zip(
            plot_positions,
            medians,
            strict=True,
        ):
            plot_ax.text(
                position + DECODING_MEDIAN_LABEL_X_OFFSET,
                median,
                f"med. {median:.2f}",
                ha="left",
                va="center",
                fontsize=DECODING_MEDIAN_LABEL_FONTSIZE,
                color="0.20",
                zorder=5,
            )

    if not plotted_any:
        plot_ax.text(
            0.5,
            0.5,
            "No finite values",
            ha="center",
            va="center",
            transform=plot_ax.transAxes,
        )

    plot_ax.set_xticks(positions)
    plot_ax.set_xticklabels(labels, fontsize=DECODING_XTICK_LABEL_FONTSIZE)
    plot_ax.set_xlim(0.5, len(comparisons) + 0.5)
    plot_ax.set_ylim(0.0, 0.5)
    plot_ax.set_ylabel(
        "Abs. norm. error",
        fontsize=DECODING_YLABEL_FONTSIZE,
        labelpad=2,
    )
    plot_ax.spines["top"].set_visible(False)
    plot_ax.spines["right"].set_visible(False)
    plot_ax.tick_params(axis="y", labelsize=7, length=2, pad=1)
    plot_ax.tick_params(axis="x", length=0, pad=1)

    train_center = DECODING_TRAIN_SCHEMATIC_CENTER_X
    ax.text(
        train_center,
        DECODING_TRAIN_LABEL_Y,
        "Train",
        ha="center",
        va="bottom",
        fontsize=5.2,
        transform=ax.transAxes,
    )
    train_ax = ax.inset_axes(
        [
            train_center - DECODING_SCHEMATIC_WIDTH / 2,
            DECODING_SCHEMATIC_Y,
            DECODING_SCHEMATIC_WIDTH,
            DECODING_SCHEMATIC_HEIGHT,
        ]
    )
    draw_w_track_schematic(
        train_ax,
        trajectory_name=DECODING_EXAMPLE_TRAIN_TRAJECTORY,
        arrow_color=PANEL_E_TRAJECTORY_COLORS[DECODING_EXAMPLE_TRAIN_TRAJECTORY],
        track_linewidth=0.45,
        trajectory_linewidth=0.65,
        arrow_mutation_scale=5.8,
        fill_track=True,
    )

    plot_left = 0.0
    plot_width = 1.0
    icon_width = DECODING_SCHEMATIC_WIDTH
    icon_height = DECODING_SCHEMATIC_HEIGHT
    for position, (comparison, _label, _family, _pairs) in zip(
        positions,
        comparisons,
        strict=True,
    ):
        test_trajectory = DECODING_EXAMPLE_TEST_TRAJECTORIES.get(comparison)
        if test_trajectory is None:
            continue
        x_center = plot_left + plot_width * (position - 0.5) / len(comparisons)
        icon_ax = ax.inset_axes(
            [
                x_center - icon_width / 2,
                DECODING_SCHEMATIC_Y,
                icon_width,
                icon_height,
            ]
        )
        draw_w_track_schematic(
            icon_ax,
            trajectory_name=test_trajectory,
            arrow_color=PANEL_E_TRAJECTORY_COLORS[test_trajectory],
            track_linewidth=0.45,
            trajectory_linewidth=0.65,
            arrow_mutation_scale=5.8,
            fill_track=True,
        )


def plot_position_aligned_raster_axis(
    ax: "Axes",
    trial_positions: Sequence[np.ndarray],
    trajectory_type: str,
    *,
    show_ylabel: bool = False,
) -> None:
    """Plot spikes by normalized task progression across trajectory trials."""
    for trial_index, positions in enumerate(trial_positions, start=1):
        positions = np.asarray(positions, dtype=float)
        if positions.size == 0:
            continue
        ax.plot(
            positions,
            np.full(positions.shape, trial_index, dtype=float),
            "|",
            color=PANEL_E_TRAJECTORY_COLORS[trajectory_type],
            **RASTER_TICK_KWARGS,
        )

    add_task_progression_segment_boundary_lines(ax)

    n_trials = len(trial_positions)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, max(1, n_trials) + 1.0)
    ax.set_xticks([0.0, 1.0])
    ax.set_xticklabels([])
    ax.set_yticks([])
    if show_ylabel:
        ax.set_ylabel("Trials", fontsize=PANEL_E_AXIS_LABEL_FONTSIZE, labelpad=1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=0.9, width=0.35, pad=1)


def draw_panel_e_raster_schematic(ax: "Axes", trajectory_type: str) -> None:
    """Draw one compact dark-epoch trajectory icon for a panel E raster."""
    draw_w_track_schematic(
        ax,
        trajectory_name=trajectory_type,
        arrow_color=PANEL_E_TRAJECTORY_COLORS[trajectory_type],
        track_linewidth=0.45,
        trajectory_linewidth=0.65,
        arrow_mutation_scale=5.8,
        fill_track=True,
    )


def plot_panel_e_rate_axis(
    ax: "Axes",
    firing_rates: dict[str, tuple[np.ndarray, np.ndarray]],
    trajectory_pair: Sequence[str],
    *,
    y_max: float,
    show_ylabel: bool = False,
) -> None:
    """Plot occupancy-normalized firing rates for a pair of trajectories."""
    for trajectory_type in trajectory_pair:
        position, rate = firing_rates[trajectory_type]
        ax.plot(
            position,
            rate,
            color=PANEL_E_TRAJECTORY_COLORS[trajectory_type],
            linewidth=0.8,
            label=PANEL_E_TRAJECTORY_LABELS[trajectory_type],
        )

    add_task_progression_segment_boundary_lines(ax)

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, y_max)
    ax.set_xticks([0.0, 1.0])
    ax.set_yticks([0.0, y_max])
    ax.set_yticklabels(["0", f"{y_max:g}"])
    ax.set_xlabel(
        "Nom. goal progression",
        fontsize=PANEL_E_AXIS_LABEL_FONTSIZE,
        labelpad=1,
    )
    if show_ylabel:
        ax.set_ylabel("FR (Hz)", fontsize=PANEL_E_AXIS_LABEL_FONTSIZE, labelpad=1)
    ax.legend(frameon=False, fontsize=3.8, handlelength=0.9, borderpad=0.1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(
        labelsize=PANEL_E_TICK_LABEL_FONTSIZE,
        length=0.9,
        width=0.35,
        pad=1,
    )


def plot_panel_e_example(
    ax: "Axes",
    example: dict[str, Any],
    *,
    title: str | None = None,
) -> None:
    """Plot one panel E example with trajectory rasters and firing-rate curves."""
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    if title is not None:
        ax.text(
            0.50,
            0.995,
            title,
            ha="center",
            va="top",
            fontsize=5.8,
            transform=ax.transAxes,
        )

    raster_positions = example["raster_positions"]
    firing_rates = example["firing_rates"]
    schematic_bounds = {
        "center_to_left": (0.210, 0.880, 0.09, 0.070),
        "center_to_right": (0.710, 0.880, 0.09, 0.070),
        "right_to_center": (0.210, 0.565, 0.09, 0.070),
        "left_to_center": (0.710, 0.565, 0.09, 0.070),
    }
    raster_bounds = {
        "center_to_left": (0.05, 0.65, 0.41, 0.19),
        "center_to_right": (0.55, 0.65, 0.41, 0.19),
        "right_to_center": (0.05, 0.35, 0.41, 0.19),
        "left_to_center": (0.55, 0.35, 0.41, 0.19),
    }
    for row in PANEL_E_RASTER_TRAJECTORY_LAYOUT:
        for trajectory_type in row:
            schematic_ax = ax.inset_axes(schematic_bounds[trajectory_type])
            draw_panel_e_raster_schematic(schematic_ax, trajectory_type)
            raster_ax = ax.inset_axes(raster_bounds[trajectory_type])
            plot_position_aligned_raster_axis(
                raster_ax,
                raster_positions[trajectory_type],
                trajectory_type,
                show_ylabel=trajectory_type in {"center_to_left", "right_to_center"},
            )

    finite_rate_maxima = [
        float(np.nanmax(rate))
        for _position, rate in firing_rates.values()
        if np.isfinite(rate).any()
    ]
    y_max = 1.0 if not finite_rate_maxima else max(1.0, np.ceil(max(finite_rate_maxima)))
    rate_bounds = ((0.05, 0.08, 0.41, 0.25), (0.55, 0.08, 0.41, 0.25))
    for pair_index, trajectory_pair in enumerate(PANEL_E_FR_TRAJECTORY_PAIRS):
        rate_ax = ax.inset_axes(rate_bounds[pair_index])
        plot_panel_e_rate_axis(
            rate_ax,
            firing_rates,
            trajectory_pair,
            y_max=y_max,
            show_ylabel=pair_index == 0,
        )


def plot_panel_e_examples(
    ax: "Axes",
    examples: Sequence[dict[str, Any]],
) -> None:
    """Plot all panel E example units stacked in one panel axis."""
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    if not examples:
        ax.text(0.5, 0.5, "No examples", ha="center", va="center", transform=ax.transAxes)
        return

    block_height = 0.47
    y_positions = np.linspace(1.0 - block_height, 0.02, len(examples))
    for example_index, (y0, example) in enumerate(
        zip(y_positions, examples, strict=False),
        start=1,
    ):
        example_ax = ax.inset_axes([0.0, float(y0), 1.0, block_height])
        plot_panel_e_example(
            example_ax,
            example,
            title=f"Example cell {example_index}",
        )


def make_figure_1(
    *,
    data_root: Path,
    asset_dir: Path,
    output_path: Path,
    datasets: Sequence[DatasetId],
    regions: Sequence[str],
    position_bin_count: int,
    position_offset: int,
    speed_threshold_cm_s: float,
    sigma_bins: float,
    encoding_place_bin_size_cm: float,
    dpi: int,
    panel_d_cache_dir: Path | None = None,
    refresh_panel_d_cache: bool = False,
    panel_e_cache_dir: Path | None = None,
    refresh_panel_e_cache: bool = False,
) -> Path:
    """Build and save Figure 1."""
    import matplotlib.pyplot as plt

    panel_d_cache_dir = (
        Path(output_path).parent / "cache"
        if panel_d_cache_dir is None
        else Path(panel_d_cache_dir)
    )
    panel_e_cache_dir = (
        Path(output_path).parent / "cache"
        if panel_e_cache_dir is None
        else Path(panel_e_cache_dir)
    )
    apply_paper_style()
    n_region_rows = len(regions) * len(TRAJECTORY_TYPES)
    heatmap_height_mm = DEFAULT_HEATMAP_HEIGHT_MM * max(len(regions), 1)
    fig_height_mm = (
        DEFAULT_TOP_ROW_HEIGHT_MM
        + heatmap_height_mm
        + DEFAULT_MIDDLE_TO_FINAL_ROW_SPACER_MM
        + DEFAULT_BOTTOM_ROW_HEIGHT_MM
    )
    fig = plt.figure(
        figsize=figure_size(DEFAULT_FIGURE_WIDTH_MM, fig_height_mm),
        constrained_layout=True,
    )
    outer_grid = fig.add_gridspec(
        nrows=4,
        ncols=1,
        height_ratios=[
            DEFAULT_TOP_ROW_HEIGHT_MM,
            heatmap_height_mm,
            DEFAULT_MIDDLE_TO_FINAL_ROW_SPACER_MM,
            DEFAULT_BOTTOM_ROW_HEIGHT_MM,
        ],
    )

    main_grid = outer_grid[:2].subgridspec(
        nrows=2,
        ncols=2,
        wspace=BOTTOM_ROW_PANEL_WSPACE,
        hspace=0.08,
        height_ratios=[
            DEFAULT_TOP_ROW_HEIGHT_MM,
            heatmap_height_mm,
        ],
        width_ratios=[
            DEFAULT_PANEL_E_WIDTH_FRACTION,
            DEFAULT_HEATMAP_PANEL_WIDTH_FRACTION,
        ],
    )
    panel_b_axis = fig.add_subplot(main_grid[0, 0])
    draw_behavior_task_design_panel(panel_b_axis, asset_dir=asset_dir)
    panel_b_axis.set_title("Task design", fontsize=9, pad=2)
    label_axis(panel_b_axis, "A", x=-0.04, y=1.02)

    heatmap_grid = main_grid[:, 1].subgridspec(
        nrows=n_region_rows + 1,
        ncols=len(TRAJECTORY_TYPES) + 1,
        height_ratios=[0.42, *([1.0] * n_region_rows)],
        width_ratios=[0.48, *([1.0] * len(TRAJECTORY_TYPES))],
    )
    panel_d_axis = fig.add_subplot(main_grid[1, 0])
    panel_e_examples = [
        load_or_compute_panel_e_example_data(
            data_root=data_root,
            animal_name=animal_name,
            date=date,
            epoch=epoch,
            region=region,
            unit_id=unit_id,
            position_bin_count=position_bin_count,
            position_offset=position_offset,
            speed_threshold_cm_s=speed_threshold_cm_s,
            sigma_bins=sigma_bins,
            panel_e_cache_dir=panel_e_cache_dir,
            refresh_panel_e_cache=refresh_panel_e_cache,
        )
        for animal_name, date, epoch, region, unit_id in PANEL_E_EXAMPLES
    ]
    plot_panel_e_examples(panel_d_axis, panel_e_examples)
    panel_d_axis.set_title("Example dark DGP coding cells", fontsize=8, pad=2)
    label_axis(panel_d_axis, "B", x=-0.04, y=1.02)

    spacer_axis = fig.add_subplot(outer_grid[2])
    spacer_axis.axis("off")

    final_row_grid = outer_grid[3].subgridspec(
        nrows=1,
        ncols=3,
        width_ratios=[
            DEFAULT_PANEL_F_WIDTH_FRACTION,
            DEFAULT_PANEL_G_WIDTH_FRACTION,
            DEFAULT_PANEL_H_WIDTH_FRACTION,
        ],
    )
    panel_f_axis = fig.add_subplot(final_row_grid[0, 0])
    panel_g_axis = fig.add_subplot(final_row_grid[0, 1])
    panel_h_axis = fig.add_subplot(final_row_grid[0, 2])
    motor_delta_table = load_motor_delta_table(
        data_root=data_root,
        datasets=datasets,
        region=MOTOR_DELTA_REGION,
    )
    plot_motor_delta_panel(panel_f_axis, motor_delta_table)
    panel_f_axis.set_title("Comparison to motor", fontsize=8, pad=2)
    label_axis(panel_f_axis, "D", x=-0.04, y=1.02)
    encoding_delta_table = load_encoding_delta_table(
        data_root=data_root,
        datasets=datasets,
        region=ENCODING_COMPARISON_REGION,
        place_bin_size_cm=encoding_place_bin_size_cm,
    )
    plot_encoding_delta_panel(panel_g_axis, encoding_delta_table)
    panel_g_axis.set_title("Comparison to alternative codes", fontsize=8, pad=2)
    label_axis(panel_g_axis, "E", x=-0.08, y=1.02)
    decoding_error_table = load_decoding_absolute_error_table(
        data_root=data_root,
        datasets=filter_datasets_by_animals(datasets, PANEL_H_DECODING_ANIMALS),
        region=DECODING_COMPARISON_REGION,
    )
    plot_decoding_error_panel(panel_h_axis, decoding_error_table)
    panel_h_axis.set_title("Cross trajectory decoding", fontsize=8, pad=2)
    label_axis(panel_h_axis, "F", x=-0.04, y=1.02)

    axes = np.asarray(
        [
            [fig.add_subplot(heatmap_grid[row, col]) for col in range(len(TRAJECTORY_TYPES) + 1)]
            for row in range(n_region_rows + 1)
        ],
        dtype=object,
    )

    corner_axis = axes[0, 0]
    corner_axis.axis("off")
    tuning_schematic_axes = axes[0, 1:]
    order_schematic_axes = axes[1:, 0]
    heatmap_axes = axes[1:, 1:]
    for ax, trajectory_type in zip(tuning_schematic_axes, TRAJECTORY_TYPES, strict=True):
        draw_w_track_schematic(
            ax,
            trajectory_name=trajectory_type,
            arrow_color=PANEL_E_TRAJECTORY_COLORS[trajectory_type],
            fill_track=True,
        )
    for row_index, ax in enumerate(order_schematic_axes):
        draw_order_schematic(
            ax,
            TRAJECTORY_TYPES[row_index % len(TRAJECTORY_TYPES)],
            arrow_color=PANEL_E_TRAJECTORY_COLORS[
                TRAJECTORY_TYPES[row_index % len(TRAJECTORY_TYPES)]
            ],
        )

    color_image = plot_dark_heatmap_regions(
        heatmap_axes,
        data_root=data_root,
        datasets=datasets,
        regions=regions,
        position_bin_count=position_bin_count,
        position_offset=position_offset,
        speed_threshold_cm_s=speed_threshold_cm_s,
        sigma_bins=sigma_bins,
        panel_d_cache_dir=panel_d_cache_dir,
        refresh_panel_d_cache=refresh_panel_d_cache,
    )

    if color_image is not None:
        colorbar = fig.colorbar(
            color_image,
            ax=heatmap_axes.ravel().tolist(),
            shrink=0.24,
            pad=HEATMAP_COLORBAR_PAD,
            aspect=7,
            ticks=[0.0, 1.0],
        )
        colorbar.ax.set_yticklabels(["0", "1"])
        colorbar.ax.tick_params(length=2)
        colorbar.set_label(
            "Norm. FR",
            rotation=90,
            labelpad=HEATMAP_COLORBAR_LABELPAD,
            fontsize=HEATMAP_COLORBAR_LABEL_FONTSIZE,
        )

    draw_neuron_scale_bar(heatmap_axes[-1, -1])

    fig.canvas.draw()
    add_centered_axis_text(
        fig,
        tuning_schematic_axes,
        "Tuning",
        y_offset=HEATMAP_TUNING_LABEL_OFFSET,
    )
    add_centered_axis_text(
        fig,
        order_schematic_axes,
        "Order",
        y_offset=HEATMAP_ORDER_LABEL_OFFSET,
        rotation=90,
    )
    label_axis(corner_axis, "C", x=-0.12, y=1.04)
    save_figure(fig, output_path, dpi=dpi)
    plt.close(fig)
    print(f"Saved Figure 1 to {output_path}")
    return output_path


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for Figure 1 generation."""
    parser = argparse.ArgumentParser(
        description="Generate Figure 1 pooled dark-epoch place-field heatmaps."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=f"Base directory containing analysis outputs. Default: {DEFAULT_DATA_ROOT}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for figure output. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--asset-dir",
        type=Path,
        default=DEFAULT_ASSET_DIR,
        help=f"Directory containing Figure 1 external assets. Default: {DEFAULT_ASSET_DIR}",
    )
    parser.add_argument(
        "--output-name",
        default=DEFAULT_OUTPUT_NAME,
        help=f"Output basename without extension. Default: {DEFAULT_OUTPUT_NAME}",
    )
    parser.add_argument(
        "--panel-d-cache-dir",
        type=Path,
        default=None,
        help=(
            "Directory for cached Panel D heatmap matrices. "
            "Default: <output-dir>/cache."
        ),
    )
    parser.add_argument(
        "--refresh-panel-d-cache",
        action="store_true",
        help="Recompute Panel D and overwrite its cache even when a matching cache exists.",
    )
    parser.add_argument(
        "--panel-e-cache-dir",
        type=Path,
        default=None,
        help=(
            "Directory for cached Panel E example-cell data. "
            "Default: <output-dir>/cache."
        ),
    )
    parser.add_argument(
        "--refresh-panel-e-cache",
        action="store_true",
        help="Recompute Panel E and overwrite its cache even when a matching cache exists.",
    )
    parser.add_argument(
        "--format",
        dest="output_format",
        choices=FIGURE_FORMATS,
        default=DEFAULT_OUTPUT_FORMAT,
        help=f"Output format. Default: {DEFAULT_OUTPUT_FORMAT}",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        type=parse_dataset_id,
        help=(
            "Animal/date data set to include as animal:date. May be repeated. "
            "Default: use v1ca1.paper_figures.datasets."
        ),
    )
    parser.add_argument(
        "--region",
        action="append",
        choices=REGIONS,
        help=(
            "Region to include. May be repeated. "
            f"Default: {', '.join(DEFAULT_REGIONS)}."
        ),
    )
    parser.add_argument(
        "--position-bin-count",
        type=int,
        default=DEFAULT_POSITION_BIN_COUNT,
        help=(
            "Number of bins from normalized trajectory position 0 to 1. "
            f"Default: {DEFAULT_POSITION_BIN_COUNT}"
        ),
    )
    parser.add_argument(
        "--position-offset",
        type=int,
        default=DEFAULT_POSITION_OFFSET,
        help=f"Number of leading position samples to ignore. Default: {DEFAULT_POSITION_OFFSET}",
    )
    parser.add_argument(
        "--speed-threshold-cm-s",
        type=float,
        default=DEFAULT_SPEED_THRESHOLD_CM_S,
        help=(
            "Speed threshold in cm/s used to define movement intervals. "
            f"Default: {DEFAULT_SPEED_THRESHOLD_CM_S}"
        ),
    )
    parser.add_argument(
        "--sigma-bins",
        type=float,
        default=DEFAULT_SIGMA_BINS,
        help=f"Gaussian smoothing width in bins. Default: {DEFAULT_SIGMA_BINS}",
    )
    parser.add_argument(
        "--encoding-place-bin-size-cm",
        type=float,
        default=ENCODING_COMPARISON_PLACE_BIN_SIZE_CM,
        help=(
            "Place-bin size used to find encoding-comparison summary files. "
            f"Default: {ENCODING_COMPARISON_PLACE_BIN_SIZE_CM}"
        ),
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Rasterization dpi for saved output. Default: 300",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Run Figure 1 generation."""
    args = parse_arguments(argv)
    datasets = args.dataset if args.dataset is not None else get_processed_datasets()
    regions = tuple(args.region) if args.region is not None else DEFAULT_REGIONS
    output_path = build_output_path(
        args.output_dir,
        args.output_name,
        args.output_format,
    )
    panel_d_cache_dir = (
        args.panel_d_cache_dir
        if args.panel_d_cache_dir is not None
        else args.output_dir / "cache"
    )
    panel_e_cache_dir = (
        args.panel_e_cache_dir
        if args.panel_e_cache_dir is not None
        else args.output_dir / "cache"
    )
    make_figure_1(
        data_root=args.data_root,
        asset_dir=args.asset_dir,
        output_path=output_path,
        datasets=datasets,
        regions=regions,
        position_bin_count=args.position_bin_count,
        position_offset=args.position_offset,
        speed_threshold_cm_s=args.speed_threshold_cm_s,
        sigma_bins=args.sigma_bins,
        encoding_place_bin_size_cm=args.encoding_place_bin_size_cm,
        dpi=args.dpi,
        panel_d_cache_dir=panel_d_cache_dir,
        refresh_panel_d_cache=args.refresh_panel_d_cache,
        panel_e_cache_dir=panel_e_cache_dir,
        refresh_panel_e_cache=args.refresh_panel_e_cache,
    )


if __name__ == "__main__":
    main()
