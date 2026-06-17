from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import v1ca1.paper_figures.heatmaps as heatmaps_module
from v1ca1.paper_figures.heatmaps import (
    DEFAULT_OUTPUT_NAME,
    HEATMAP_ROW_SPECS,
    build_output_path,
    make_heatmaps_figure,
    parse_arguments,
)


def test_default_cli_writes_heatmaps_svg() -> None:
    args = parse_arguments([])

    assert DEFAULT_OUTPUT_NAME == "heatmaps"
    assert args.output_name == "heatmaps"
    assert args.output_format == "svg"
    assert args.panel_d_cache_dir is None
    assert args.panel_b_cache_dir is None
    assert args.refresh_panel_cache is False
    assert build_output_path(args.output_dir, args.output_name, args.output_format) == (
        Path("paper_figures") / "output" / "heatmaps.svg"
    )


def test_make_heatmaps_figure_uses_both_normalizations(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")

    calls: dict[str, object] = {}

    def _dummy_panels(
        trajectory_types: tuple[str, ...],
        position_bin_count: int,
    ) -> dict[tuple[str, str], np.ndarray]:
        return {
            (order_trajectory, plot_trajectory): np.ones((1, position_bin_count))
            for order_trajectory in trajectory_types
            for plot_trajectory in trajectory_types
        }

    def fake_load_or_compute_panel_d_heatmap_panels(**kwargs: object):
        calls.setdefault("panel_d_calls", []).append(kwargs)
        return _dummy_panels(
            heatmaps_module.PANEL_D_TRAJECTORY_TYPES,
            int(kwargs["position_bin_count"]),
        )

    def fake_load_or_compute_panel_b_heatmap_panels(**kwargs: object):
        calls.setdefault("panel_b_calls", []).append(kwargs)
        return _dummy_panels(
            heatmaps_module.PANEL_B_TRAJECTORY_TYPES,
            int(kwargs["position_bin_count"]),
        )

    def fake_load_or_compute_panel_d_heatmap_payload(**kwargs: object):
        calls.setdefault("panel_d_payload_calls", []).append(kwargs)
        position_bin_count = int(kwargs["position_bin_count"])
        ordered_unit_keys_by_trajectory = {
            trajectory_type: np.asarray(["unit-0"], dtype=object)
            for trajectory_type in heatmaps_module.PANEL_D_TRAJECTORY_TYPES
        }
        return (
            _dummy_panels(heatmaps_module.PANEL_D_TRAJECTORY_TYPES, position_bin_count),
            ordered_unit_keys_by_trajectory,
        )

    def fake_build_light_curve_sets(**kwargs: object):
        calls.setdefault("light_curve_set_calls", []).append(kwargs)
        return [{"source": "light"}]

    def fake_build_light_panels_in_dark_order(*args: object, **kwargs: object):
        calls.setdefault("dark_order_panel_calls", []).append(
            {
                "args": args,
                **kwargs,
            }
        )
        return _dummy_panels(
            heatmaps_module.PANEL_B_TRAJECTORY_TYPES,
            int(kwargs["position_bin_count"]),
        )

    def fake_save_figure(figure: object, output_path: Path, dpi: int) -> Path:
        calls["output_path"] = output_path
        calls["dpi"] = dpi
        calls["axis_count"] = len(figure.axes)
        return output_path

    monkeypatch.setattr(
        heatmaps_module,
        "load_or_compute_panel_d_heatmap_panels",
        fake_load_or_compute_panel_d_heatmap_panels,
    )
    monkeypatch.setattr(
        heatmaps_module,
        "load_or_compute_panel_b_heatmap_panels",
        fake_load_or_compute_panel_b_heatmap_panels,
    )
    monkeypatch.setattr(
        heatmaps_module,
        "load_or_compute_panel_d_heatmap_payload",
        fake_load_or_compute_panel_d_heatmap_payload,
    )
    monkeypatch.setattr(
        heatmaps_module,
        "build_light_curve_sets",
        fake_build_light_curve_sets,
    )
    monkeypatch.setattr(
        heatmaps_module,
        "build_light_panels_in_dark_order",
        fake_build_light_panels_in_dark_order,
    )
    monkeypatch.setattr(heatmaps_module, "save_figure", fake_save_figure)

    output_path = tmp_path / "heatmaps.svg"
    saved_path = make_heatmaps_figure(
        data_root=Path("/analysis"),
        output_path=output_path,
        datasets=[("L14", "20240611", "08_r4")],
        regions=("v1",),
        light_epoch=None,
        position_bin_count=2,
        position_offset=10,
        speed_threshold_cm_s=4.0,
        sigma_bins=1.5,
        panel_d_cache_dir=None,
        panel_b_cache_dir=None,
        refresh_panel_cache=False,
        dpi=300,
    )

    assert saved_path == output_path
    assert calls["output_path"] == output_path
    assert calls["dpi"] == 300
    assert calls["axis_count"] > 0
    assert [call["firing_rate_normalization"] for call in calls["panel_d_calls"]] == [
        row_spec[1] for row_spec in HEATMAP_ROW_SPECS
    ]
    assert [call["firing_rate_normalization"] for call in calls["panel_b_calls"]] == [
        HEATMAP_ROW_SPECS[0][1],
        HEATMAP_ROW_SPECS[1][1],
    ]
    assert [
        call["firing_rate_normalization"] for call in calls["dark_order_panel_calls"]
    ] == [
        HEATMAP_ROW_SPECS[2][1],
        HEATMAP_ROW_SPECS[3][1],
    ]
    assert calls["panel_d_calls"][0]["panel_d_cache_dir"] == output_path.parent / "cache"
    assert calls["panel_b_calls"][0]["panel_b_cache_dir"] == output_path.parent / "cache"
    assert len(calls["panel_d_payload_calls"]) == 1
    assert calls["panel_d_payload_calls"][0]["require_ordered_unit_keys"] is True
    assert len(calls["light_curve_set_calls"]) == 1
