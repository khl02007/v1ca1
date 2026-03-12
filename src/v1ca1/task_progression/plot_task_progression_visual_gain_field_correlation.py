from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np


TRAJECTORIES = {
    "outbound": ("center_to_left", "center_to_right"),
    "inbound": ("left_to_center", "right_to_center"),
}

MODEL_SUFFIXES = {
    "mult": "mult_speed.pkl",
    "mult_per_segment": "mult_per_segment_speed.pkl",
    "mult_per_segment_scalar": "mult_per_segment_scalar_speed.pkl",
    "mult_per_segment_overlap": "mult_per_segment_overlap_speed.pkl",
    "sep": "sep_speed.pkl",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot swap-consistency of trajectory-specific visual gain fields across "
            "task configurations from saved task-progression GLM models."
        )
    )
    home = Path(__file__).resolve().parent
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=home / "tp_glm_updated2",
        help="Directory containing saved epoch-level model pickles.",
    )
    parser.add_argument(
        "--fig-dir",
        type=Path,
        default=home / "figs" / "visual_gain_correlation",
        help="Output directory for saved figures.",
    )
    parser.add_argument(
        "--model-kind",
        choices=(
            "mult",
            "mult_per_segment",
            "mult_per_segment_scalar",
            "mult_per_segment_overlap",
            "mult_fewer_gain_splines",
            "sep",
        ),
        default="mult_per_segment",
        help="Saved model family to load.",
    )
    parser.add_argument(
        "--n-splines-gain",
        type=int,
        default=6,
        help=(
            "Gain spline count used only when --model-kind=mult_fewer_gain_splines."
        ),
    )
    parser.add_argument(
        "--epoch1",
        type=str,
        default=None,
        help="First epoch to compare. Defaults to the first available saved epoch.",
    )
    parser.add_argument(
        "--epoch2",
        type=str,
        default=None,
        help="Second epoch to compare. Defaults to the third available saved epoch.",
    )
    parser.add_argument(
        "--baseline-epoch",
        type=str,
        default=None,
        help="Optional baseline epoch to subtract before comparing epochs.",
    )
    parser.add_argument(
        "--ridge",
        type=float,
        default=1e-3,
        help="Preferred ridge value. The nearest saved ridge is used if needed.",
    )
    parser.add_argument(
        "--n-boot",
        type=int,
        default=10000,
        help="Number of bootstrap resamples across neurons for each segment.",
    )
    parser.add_argument(
        "--ci",
        type=float,
        default=95.0,
        help="Bootstrap confidence interval width in percent.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base random seed for bootstrapping.",
    )
    parser.add_argument(
        "--exclude-global-light-offset",
        action="store_true",
        help=(
            "Use only the segment-specific interaction term, matching the original "
            "notebook logic more closely. By default the full per-segment light "
            "effect is used."
        ),
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figure interactively after saving.",
    )
    return parser.parse_args()


def _model_suffix(model_kind: str, n_splines_gain: int) -> str:
    if model_kind == "mult_fewer_gain_splines":
        return f"mult_fewer_gain_splines_{n_splines_gain}_speed.pkl"
    return MODEL_SUFFIXES[model_kind]


def _load_saved_models(model_dir: Path, model_kind: str, n_splines_gain: int) -> dict:
    suffix = _model_suffix(model_kind, n_splines_gain)
    models = {}
    for path in sorted(model_dir.glob(f"*_{suffix}")):
        epoch = path.name[: -len(f"_{suffix}")]
        with path.open("rb") as f:
            models[epoch] = pickle.load(f)
    if not models:
        raise FileNotFoundError(
            f"No files matching '*_{suffix}' were found in {model_dir}."
        )
    return models


def _choose_epochs(models: dict, epoch1: str | None, epoch2: str | None) -> tuple[str, str]:
    epochs = sorted(models)
    if epoch1 is None:
        epoch1 = epochs[0]
    if epoch2 is None:
        epoch2 = epochs[2] if len(epochs) >= 3 else epochs[-1]
    missing = [ep for ep in (epoch1, epoch2) if ep not in models]
    if missing:
        raise KeyError(f"Requested epoch(s) not found in saved models: {missing}")
    return epoch1, epoch2


def _get_res(models: dict, epoch: str, traj: str, ridge: float) -> tuple[dict, float]:
    by_traj = models[epoch][traj]
    if ridge in by_traj:
        return by_traj[ridge], ridge
    keys = np.asarray(list(by_traj.keys()), dtype=float)
    ridge_used = float(keys[np.argmin(np.abs(np.log10(keys) - np.log10(ridge)))])
    return by_traj[ridge_used], ridge_used


def _align_by_unit_ids(ids_a, x_a, ids_b, x_b):
    ids_a = np.asarray(ids_a)
    ids_b = np.asarray(ids_b)
    common, idx_a, idx_b = np.intersect1d(ids_a, ids_b, return_indices=True)
    if common.size == 0:
        raise ValueError("No overlapping unit_ids.")
    return common, np.asarray(x_a)[idx_a], np.asarray(x_b)[idx_b]


def _corr_fast(x, y) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x - x.mean()
    y = y - y.mean()
    denom = np.sqrt(np.sum(x * x) * np.sum(y * y))
    if denom <= 0:
        return np.nan
    return float(np.sum(x * y) / denom)


def _bootstrap_r(x, y, *, n_boot: int, ci: float, seed: int):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = x.size

    r_obs = _corr_fast(x, y)
    boot = np.full(n_boot, np.nan, dtype=float)
    for idx in range(n_boot):
        sample = rng.integers(0, n, size=n)
        boot[idx] = _corr_fast(x[sample], y[sample])
    boot = boot[np.isfinite(boot)]

    alpha = (100.0 - float(ci)) / 2.0
    ci_low, ci_high = np.nan, np.nan
    if boot.size:
        ci_low, ci_high = np.percentile(boot, [alpha, 100.0 - alpha])
    return float(r_obs), float(ci_low), float(ci_high), boot


def _segment_effect(
    res: dict,
    seg_idx: int,
    *,
    include_global_light: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    unit_ids = np.asarray(res["unit_ids"])
    edges = (
        np.asarray(res["segment_edges"], dtype=float)
        if "segment_edges" in res
        else None
    )
    gain_basis = res.get("gain_basis", "")
    global_light = np.asarray(res.get("coef_light_full_all", []), dtype=float)
    gamma = np.asarray(res["coef_place_x_light_full_all"], dtype=float)

    if gain_basis == "segment_scalar":
        if edges is None:
            raise ValueError("segment_scalar model is missing segment_edges.")
        n_seg = edges.size - 1
        if seg_idx < 0 or seg_idx >= n_seg:
            raise ValueError(f"seg_idx={seg_idx} out of range for n_seg={n_seg}")
        if include_global_light:
            if seg_idx == 0:
                effect = global_light
            else:
                effect = global_light + gamma[seg_idx - 1, :]
        else:
            effect = np.zeros_like(global_light) if seg_idx == 0 else gamma[seg_idx - 1, :]
        return unit_ids, np.asarray(effect, dtype=float), edges

    if seg_idx < 0 or seg_idx >= gamma.shape[0]:
        raise ValueError(
            f"seg_idx={seg_idx} out of range for n_seg={gamma.shape[0]}"
        )
    effect = gamma[seg_idx, :]
    if include_global_light and global_light.shape == effect.shape:
        effect = global_light + effect
    return unit_ids, np.asarray(effect, dtype=float), edges


def _effect_seg(
    models: dict,
    epoch: str,
    traj: str,
    ridge: float,
    seg_idx: int,
    *,
    baseline_epoch: str | None,
    include_global_light: bool,
):
    res, ridge_used = _get_res(models, epoch, traj, ridge)
    ids, effect, edges = _segment_effect(
        res,
        seg_idx,
        include_global_light=include_global_light,
    )
    if baseline_epoch is None:
        return ids, effect, ridge_used, edges

    res0, _ = _get_res(models, baseline_epoch, traj, ridge_used)
    ids0, effect0, _ = _segment_effect(
        res0,
        seg_idx,
        include_global_light=include_global_light,
    )
    common, effect_aligned, effect0_aligned = _align_by_unit_ids(
        ids, effect, ids0, effect0
    )
    return common, (effect_aligned - effect0_aligned), ridge_used, edges


def _left_right_contrast(
    models: dict,
    epoch: str,
    direction: str,
    ridge: float,
    seg_idx: int,
    *,
    baseline_epoch: str | None,
    include_global_light: bool,
):
    try:
        traj_left, traj_right = TRAJECTORIES[direction]
    except KeyError as exc:
        raise ValueError("direction must be 'outbound' or 'inbound'.") from exc

    ids_left, effect_left, ridge_used, edges = _effect_seg(
        models,
        epoch,
        traj_left,
        ridge,
        seg_idx,
        baseline_epoch=baseline_epoch,
        include_global_light=include_global_light,
    )
    ids_right, effect_right, _, edges_right = _effect_seg(
        models,
        epoch,
        traj_right,
        ridge_used,
        seg_idx,
        baseline_epoch=baseline_epoch,
        include_global_light=include_global_light,
    )
    common, left_aligned, right_aligned = _align_by_unit_ids(
        ids_left, effect_left, ids_right, effect_right
    )
    edges_use = edges if edges is not None else edges_right
    return common, left_aligned - right_aligned, ridge_used, edges_use


def _n_segments_for_epoch(
    models: dict,
    epoch: str,
    direction: str,
    ridge: float,
    *,
    include_global_light: bool,
) -> tuple[int, float]:
    traj = TRAJECTORIES[direction][0]
    res, ridge_used = _get_res(models, epoch, traj, ridge)
    edges = res.get("segment_edges")
    gain_basis = res.get("gain_basis", "")
    gamma = np.asarray(res["coef_place_x_light_full_all"], dtype=float)

    if gain_basis == "segment_scalar":
        if edges is None:
            raise ValueError("segment_scalar model is missing segment_edges.")
        return int(len(edges) - 1), ridge_used

    if edges is not None and include_global_light:
        n_seg_edges = int(len(edges) - 1)
        if n_seg_edges != gamma.shape[0]:
            raise ValueError(
                "segment_edges and coef_place_x_light_full_all disagree on segment count. "
                "Use a segment model or pass --exclude-global-light-offset if you want "
                "the raw interaction coefficients only."
            )
        return n_seg_edges, ridge_used

    return int(gamma.shape[0]), ridge_used


def swap_consistency_all_segments(
    models: dict,
    epoch1: str,
    epoch2: str,
    direction: str,
    ridge: float,
    *,
    baseline_epoch: str | None,
    include_global_light: bool,
    n_boot: int,
    ci: float,
    seed: int,
) -> list[dict]:
    n_seg, ridge_used = _n_segments_for_epoch(
        models,
        epoch1,
        direction,
        ridge,
        include_global_light=include_global_light,
    )
    results = []
    for seg_idx in range(n_seg):
        ids1, d1, _, edges1 = _left_right_contrast(
            models,
            epoch1,
            direction,
            ridge_used,
            seg_idx,
            baseline_epoch=baseline_epoch,
            include_global_light=include_global_light,
        )
        ids2, d2, _, edges2 = _left_right_contrast(
            models,
            epoch2,
            direction,
            ridge_used,
            seg_idx,
            baseline_epoch=baseline_epoch,
            include_global_light=include_global_light,
        )

        common, x, y = _align_by_unit_ids(ids1, d1, ids2, d2)
        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]
        n_neurons = int(x.size)
        edges = edges1 if edges1 is not None else edges2

        if n_neurons < 3:
            results.append(
                {
                    "seg_idx": seg_idx,
                    "r_swap": np.nan,
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                    "n_neurons": n_neurons,
                    "ridge_used": float(ridge_used),
                    "segment_edges": edges,
                }
            )
            continue

        r_swap, ci_low, ci_high, _ = _bootstrap_r(
            x,
            -y,
            n_boot=n_boot,
            ci=ci,
            seed=seed + 1000 * seg_idx,
        )
        results.append(
            {
                "seg_idx": seg_idx,
                "r_swap": r_swap,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "n_neurons": n_neurons,
                "ridge_used": float(ridge_used),
                "segment_edges": edges,
            }
        )
    return results


def _segment_axis(reslist: list[dict]) -> tuple[np.ndarray, list[str]]:
    edges = reslist[0].get("segment_edges")
    if edges is None:
        return np.arange(len(reslist), dtype=float), [f"seg{k}" for k in range(len(reslist))]

    edges = np.asarray(edges, dtype=float)
    labels = [f"{edges[idx]:.2f}-{edges[idx + 1]:.2f}" for idx in range(len(edges) - 1)]
    centers = 0.5 * (edges[:-1] + edges[1:])
    if len(labels) != len(reslist):
        raise ValueError(
            "Segment label count does not match the number of computed results. "
            "This usually means the loaded model does not encode one visual-gain "
            "coefficient vector per segment."
        )
    return centers, labels


def _plot_panel(ax, reslist: list[dict], title: str) -> None:
    x, labels = _segment_axis(reslist)
    r = np.asarray([item["r_swap"] for item in reslist], dtype=float)
    ci_low = np.asarray([item["ci_low"] for item in reslist], dtype=float)
    ci_high = np.asarray([item["ci_high"] for item in reslist], dtype=float)

    finite_ci = np.isfinite(ci_low) & np.isfinite(ci_high)
    ax.vlines(x[finite_ci], ci_low[finite_ci], ci_high[finite_ci], color="0.25", linewidth=1.5)
    ax.scatter(x, r, s=28, color="black", zorder=3)
    ax.axhline(0.0, linewidth=1.0, color="0.35", alpha=0.5)
    ax.set_title(title)
    ax.set_ylabel(r"$r_{\mathrm{swap}} = \mathrm{corr}(d_1,\,-d_2)$")
    ax.set_xlabel("Segment")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylim(-1.05, 1.05)
    ax.grid(True, alpha=0.2)


def plot_rswap_by_segment(
    results_outbound: list[dict],
    results_inbound: list[dict],
    *,
    title: str,
):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    _plot_panel(axes[0], results_outbound, "Outbound")
    _plot_panel(axes[1], results_inbound, "Inbound")
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def main() -> None:
    args = _parse_args()
    include_global_light = not args.exclude_global_light_offset

    try:
        models = _load_saved_models(
            args.model_dir,
            args.model_kind,
            args.n_splines_gain,
        )
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Failed to unpickle a saved model because this Python environment is "
            "missing a dependency or uses an incompatible NumPy build."
        ) from exc

    epoch1, epoch2 = _choose_epochs(models, args.epoch1, args.epoch2)
    if args.baseline_epoch is not None and args.baseline_epoch not in models:
        raise KeyError(
            f"Requested baseline epoch {args.baseline_epoch!r} not found in saved models."
        )

    results_outbound = swap_consistency_all_segments(
        models,
        epoch1,
        epoch2,
        "outbound",
        args.ridge,
        baseline_epoch=args.baseline_epoch,
        include_global_light=include_global_light,
        n_boot=args.n_boot,
        ci=args.ci,
        seed=args.seed,
    )
    results_inbound = swap_consistency_all_segments(
        models,
        epoch1,
        epoch2,
        "inbound",
        args.ridge,
        baseline_epoch=args.baseline_epoch,
        include_global_light=include_global_light,
        n_boot=args.n_boot,
        ci=args.ci,
        seed=args.seed + 1,
    )

    stat_label = "full visual gain field" if include_global_light else "segment interaction term"
    title = (
        f"Swap-consistency across segments\n"
        f"{epoch1} vs {epoch2} | {args.model_kind} | {stat_label}"
    )
    if args.baseline_epoch is not None:
        title += f" | baseline-subtracted: {args.baseline_epoch}"

    import matplotlib.pyplot as plt

    fig = plot_rswap_by_segment(
        results_outbound,
        results_inbound,
        title=title,
    )

    args.fig_dir.mkdir(parents=True, exist_ok=True)
    baseline_tag = f"_minus_{args.baseline_epoch}" if args.baseline_epoch else ""
    mode_tag = "full_light" if include_global_light else "interaction_only"
    fig_path = args.fig_dir / (
        f"{args.model_kind}_{epoch1}_vs_{epoch2}{baseline_tag}_{mode_tag}_rswap.png"
    )
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    print(f"Saved figure to {fig_path}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
