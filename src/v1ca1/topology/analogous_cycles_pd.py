"""analogous_cycles_pd.py

A Python implementation of the *within-population* part of Yoon et al. (PNAS 2024)
"Tracking the topology of neural manifolds across populations":

1) bin spike trains (e.g., from a pynapple TsGroup)
2) compute the windowed cross-correlation similarity Sim_ℓ
3) convert to dissimilarity Dis_ℓ (a symmetric dissimilarity matrix)
4) compute persistent homology / persistence diagrams (typically H1)

This file intentionally focuses on producing PD(P) for one or more conditions.
The full "analogous cycles" pipeline additionally needs witness/Dowker persistence
(WPD(P,Q)) and induced maps to match cycles across systems.

Author: ChatGPT (GPT-5.2 Pro)
License: MIT-like; copy/modify as you like.

Notes
-----
- This module tries to depend only on numpy by default.
- If you want to pass pynapple TsGroups, you need `pynapple` installed.
- If you want persistence diagrams, you need `ripser` (or substitute Gudhi).

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np


ArrayLike = Union[np.ndarray]


@dataclass
class DissimilarityResult:
    """Result of computing dissimilarity."""

    D: np.ndarray  # (n,n) or (n,m) dissimilarity
    S: np.ndarray  # corresponding similarity matrix
    M: float  # scale factor used in Dis = 1 - S/M


def _as_2d_float_array(x: np.ndarray, *, name: str) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim != 2:
        raise ValueError(
            f"{name} must be a 2D array of shape (n_units, n_bins). Got shape {x.shape}."
        )
    # float for BLAS matmul and for ripser
    return x.astype(np.float64, copy=False)


def bin_tsgroup(
    ts_group: Any,
    *,
    bin_size: float,
    ep: Optional[Any] = None,
    binarize: bool = False,
) -> Tuple[np.ndarray, List[Any]]:
    """Bin a pynapple TsGroup into a (n_units, n_bins) count matrix.

    Parameters
    ----------
    ts_group:
        A pynapple TsGroup.
    bin_size:
        Bin size in seconds.
    ep:
        Optional epoch / IntervalSet to restrict before binning.
        In pynapple, this is typically an `nap.IntervalSet`.
    binarize:
        If True, convert counts to {0,1}.

    Returns
    -------
    X:
        np.ndarray of shape (n_units, n_bins)
    unit_ids:
        The column labels from the returned TsdFrame when available,
        otherwise integer indices.

    Notes
    -----
    This function *requires* pynapple at runtime.
    """
    try:
        import pynapple as nap  # type: ignore
    except Exception as e:
        raise ImportError(
            "bin_tsgroup requires pynapple. Install pynapple or pass a pre-binned numpy array instead."
        ) from e

    if ep is None:
        counts = ts_group.count(bin_size)
    else:
        # Most pynapple versions support `ep=`; if yours doesn't, restrict ts_group first.
        counts = ts_group.count(bin_size, ep=ep)

    # TsdFrame values are (n_bins, n_units) -> transpose
    X = np.asarray(counts.values).T
    if binarize:
        X = (X > 0).astype(np.float64)

    unit_ids: List[Any]
    if hasattr(counts, "columns"):
        unit_ids = list(counts.columns)
    else:
        unit_ids = list(range(X.shape[0]))

    return X.astype(np.float64, copy=False), unit_ids


def windowed_xcorr_similarity(
    X: np.ndarray,
    Y: Optional[np.ndarray] = None,
    *,
    max_shift_bins: int,
) -> np.ndarray:
    """Compute windowed cross-correlation similarity Sim_ℓ between rows of X and Y.

    This matches the paper's definition:

        Sim_ℓ(x,y) = ( Σ_{n=-ℓ}^{ℓ} Σ_m x_m y_{m+n} ) / (||x||_2 ||y||_2)

    where out-of-range indices are ignored (i.e., no circular wrap).

    Parameters
    ----------
    X:
        (n_x, T) array
    Y:
        (n_y, T) array, or None meaning Y=X
    max_shift_bins:
        ℓ in *bins*. ℓ=0 reduces to cosine similarity of the binned vectors.

    Returns
    -------
    S:
        (n_x, n_y) similarity matrix.

    Implementation detail
    ---------------------
    We avoid calling correlate() per pair. Instead, for each lag we do
    a matrix multiplication between aligned segments, which is typically faster.
    """
    X = _as_2d_float_array(X, name="X")
    Y = X if Y is None else _as_2d_float_array(Y, name="Y")

    n_x, T = X.shape
    n_y, T2 = Y.shape
    if T != T2:
        raise ValueError(
            f"X and Y must have the same number of time bins. Got {T} and {T2}."
        )

    if max_shift_bins < 0:
        raise ValueError("max_shift_bins must be >= 0")
    if T == 0:
        return np.zeros((n_x, n_y), dtype=np.float64)

    L = int(max_shift_bins)
    if L >= T:
        # Can't shift more than length-1 without losing all overlap.
        L = T - 1

    # Accumulate unnormalized windowed cross-correlation sums.
    C = np.zeros((n_x, n_y), dtype=np.float64)

    # lag >= 0 means y is shifted forward relative to x: sum_m x[m] y[m+lag]
    for lag in range(0, L + 1):
        if lag == 0:
            X_seg = X
            Y_seg = Y
        else:
            X_seg = X[:, : T - lag]
            Y_seg = Y[:, lag:]
        # (n_x, T-lag) @ (T-lag, n_y) -> (n_x, n_y)
        C += X_seg @ Y_seg.T

    # lag < 0, i.e. -lag: sum_m x[m] y[m-lag] = sum_m x[m+(-lag)] y[m]
    for lag in range(1, L + 1):
        X_seg = X[:, lag:]
        Y_seg = Y[:, : T - lag]
        C += X_seg @ Y_seg.T

    # Normalize by ||x|| ||y||, with safe divide.
    norm_x = np.linalg.norm(X, axis=1)
    norm_y = np.linalg.norm(Y, axis=1)
    denom = norm_x[:, None] * norm_y[None, :]

    S = np.divide(C, denom, out=np.zeros_like(C), where=(denom > 0))
    return S


def dissimilarity_matrix(
    X: np.ndarray,
    Y: Optional[np.ndarray] = None,
    *,
    max_shift_bins: int,
    M: Optional[float] = None,
    M_mode: Literal["auto", "ceil"] = "auto",
    include_self_in_M: bool = False,
    clip: bool = True,
) -> DissimilarityResult:
    """Compute the windowed cross-correlation dissimilarity matrix.

    Dis(x,y) = 1 - Sim(x,y) / M  (and Dis(x,x)=0 for within-system)

    Parameters
    ----------
    X, Y:
        (n,T) and (m,T) binned spike count matrices. If Y is None => within-system.
    max_shift_bins:
        Window half-width ℓ (in bins).
    M:
        Scale factor. If None, computed from the similarity matrix.
        For within-system, a common choice is max_{i<j} Sim(x_i,x_j).
        For cross-system, max_{i,j} Sim(x_i,y_j).
    M_mode:
        "auto": use M as-is.
        "ceil": use ceil(M). The authors' reference code often does this.
    include_self_in_M:
        Only relevant for within-system. If False (recommended), exclude diagonal.
    clip:
        If True, clip dissimilarities into [0,1] after floating point ops.

    Returns
    -------
    DissimilarityResult

    Notes
    -----
    - This dissimilarity is generally *not* a metric (triangle inequality may fail).
      Vietoris–Rips persistent homology still works on a symmetric dissimilarity.
    """
    X = _as_2d_float_array(X, name="X")
    Y_is_none = Y is None
    Y = X if Y is None else _as_2d_float_array(Y, name="Y")

    S = windowed_xcorr_similarity(X, Y, max_shift_bins=max_shift_bins)

    if M is None:
        if Y_is_none:
            n = S.shape[0]
            if n <= 1:
                M = 1.0
            else:
                if include_self_in_M:
                    M = float(np.max(S))
                else:
                    iu = np.triu_indices(n, k=1)
                    M = float(np.max(S[iu]))
        else:
            M = float(np.max(S))

    if not np.isfinite(M) or M <= 0:
        # Degenerate case: everything silent => S all zeros.
        M = 1.0

    if M_mode == "ceil":
        M_eff = float(np.ceil(M))
        if M_eff <= 0:
            M_eff = 1.0
    else:
        M_eff = float(M)

    D = 1.0 - (S / M_eff)

    if Y_is_none:
        np.fill_diagonal(D, 0.0)
        # enforce symmetry (numerical)
        D = 0.5 * (D + D.T)

    if clip:
        D = np.clip(D, 0.0, 1.0)

    return DissimilarityResult(D=D, S=S, M=M_eff)


# -----------------------------------------------------------------------------
# Optional: simple preprocessing helpers
# -----------------------------------------------------------------------------


def filter_units_basic(
    X: np.ndarray,
    unit_ids: Optional[Sequence[Any]] = None,
    *,
    min_total_spikes: int = 1,
) -> Tuple[np.ndarray, List[Any]]:
    """Drop units with total spike count < min_total_spikes."""
    X = _as_2d_float_array(X, name="X")
    totals = X.sum(axis=1)
    keep = totals >= float(min_total_spikes)
    X2 = X[keep]

    if unit_ids is None:
        ids2 = [i for i, k in enumerate(keep) if k]
    else:
        ids2 = [uid for uid, k in zip(unit_ids, keep) if k]

    return X2, ids2


def avg_cluster_std_1d(
    counts: np.ndarray,
    *,
    eps_bins: int = 40,
    min_cluster_spikes: int = 20,
) -> float:
    """Approximate the SI's 'average std of spike clusters' in 1D without DBSCAN.

    The SI clusters spike times using DBSCAN in 1D and averages the within-cluster
    standard deviation of spike times. fileciteturn1file1

    Here we implement an equivalent 1D clustering rule:
    - Expand spikes into event times at bin indices (with multiplicity counts[t])
    - Split into clusters whenever the gap between consecutive spike times > eps_bins
    - Ignore clusters with fewer than min_cluster_spikes events
    - Return the average of per-cluster std

    If there are no clusters meeting min_cluster_spikes, returns 0.0.
    """
    counts = np.asarray(counts)
    if counts.ndim != 1:
        raise ValueError("counts must be 1D")

    # Expand bin indices by count; safe for moderate spike counts.
    idx = np.nonzero(counts > 0)[0]
    if idx.size == 0:
        return 0.0

    # Multiplicity expansion
    reps = counts[idx].astype(int)
    times = np.repeat(idx, reps)
    if times.size == 0:
        return 0.0

    times.sort()

    # Find split points based on eps gap
    gaps = np.diff(times)
    split_locs = np.where(gaps > eps_bins)[0]

    # cluster boundaries in the 'times' array
    starts = np.r_[0, split_locs + 1]
    ends = np.r_[split_locs + 1, times.size]

    stds: List[float] = []
    for a, b in zip(starts, ends):
        cluster = times[a:b]
        if cluster.size >= min_cluster_spikes:
            stds.append(float(np.std(cluster)))

    if len(stds) == 0:
        return 0.0
    return float(np.mean(stds))


def filter_units_by_cluster_std(
    X: np.ndarray,
    unit_ids: Optional[Sequence[Any]] = None,
    *,
    eps_bins: int = 40,
    min_cluster_spikes: int = 20,
    max_avg_cluster_std: float = 55.0,
) -> Tuple[np.ndarray, List[Any], np.ndarray]:
    """Drop units whose avg_cluster_std_1d exceeds max_avg_cluster_std.

    This targets 'uniformly firing' units, following the SI preprocessing idea.
    Thresholds (eps, min_samples, 55) are dataset-dependent.

    Returns
    -------
    X_filt, ids_filt, scores
        scores are the avg_cluster_std values *for the original ordering*.
    """
    X = _as_2d_float_array(X, name="X")

    scores = np.array(
        [
            avg_cluster_std_1d(
                X[i], eps_bins=eps_bins, min_cluster_spikes=min_cluster_spikes
            )
            for i in range(X.shape[0])
        ]
    )

    keep = scores <= float(max_avg_cluster_std)
    X2 = X[keep]

    if unit_ids is None:
        ids2 = [i for i, k in enumerate(keep) if k]
    else:
        ids2 = [uid for uid, k in zip(unit_ids, keep) if k]

    return X2, ids2, scores


# -----------------------------------------------------------------------------
# Persistent homology wrappers
# -----------------------------------------------------------------------------


def compute_persistence_diagrams(
    D: np.ndarray,
    *,
    maxdim: int = 1,
    thresh: Optional[float] = None,
) -> List[np.ndarray]:
    """Compute persistence diagrams using ripser.

    Parameters
    ----------
    D:
        (n,n) dissimilarity matrix.
    maxdim:
        Maximum homology dimension. For loops, use 1.
    thresh:
        Optional maximum filtration value (edge length) for speed.

    Returns
    -------
    dgms:
        A list where dgms[d] is the persistence diagram for H_d.

    Notes
    -----
    ripser computes homology with Z/2Z coefficients by default, matching
    the paper's "Z2 coefficients" setting. fileciteturn1file6
    """
    try:
        from ripser import ripser  # type: ignore
    except Exception as e:
        raise ImportError(
            "compute_persistence_diagrams requires ripser. pip install ripser"
        ) from e

    D = np.asarray(D, dtype=np.float64)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError(f"D must be square. Got {D.shape}.")

    out = ripser(D, distance_matrix=True, maxdim=maxdim, thresh=thresh)
    return out["dgms"]


def iqr_significance_threshold(
    dgm: np.ndarray,
    *,
    k: float = 3.0,
    min_points: int = 10,
) -> Tuple[float, np.ndarray]:
    """Compute an IQR-based lifetime threshold and a boolean mask of 'significant' points.

    This matches the paper's rule: threshold lifetimes at Q3 + 3(Q3-Q1).
    fileciteturn1file6

    Parameters
    ----------
    dgm:
        Persistence diagram for some dimension, shape (n_points, 2).
    k:
        Multiplier on IQR (paper uses 3).
    min_points:
        If there are fewer than this many finite-lifetime points, return
        threshold=+inf and mask=all False (because the IQR rule is unstable).

    Returns
    -------
    threshold, mask
        threshold is in *lifetime* units (death-birth), not in birth/death units.
    """
    dgm = np.asarray(dgm, dtype=np.float64)
    if dgm.size == 0:
        return float("inf"), np.zeros((0,), dtype=bool)

    births = dgm[:, 0]
    deaths = dgm[:, 1]

    finite = np.isfinite(deaths)
    lifetimes = deaths[finite] - births[finite]

    if lifetimes.size < min_points:
        # Too few points to robustly estimate quartiles.
        return float("inf"), np.zeros((dgm.shape[0],), dtype=bool)

    q1 = float(np.percentile(lifetimes, 25))
    q3 = float(np.percentile(lifetimes, 75))
    thr = q3 + float(k) * (q3 - q1)

    all_lifetimes = deaths - births
    mask = np.isinf(deaths) | (all_lifetimes > thr)
    return thr, mask


import numpy as np


def significant_h1_iqr(dgm_h1, k=3.0, min_finite_points=10, small_diagram_policy="all"):
    """
    Returns:
      lifetimes: array
      thr: float (IQR threshold in lifetime units) or np.nan if not computed
      sig_mask: boolean mask of significant points
    """
    dgm_h1 = np.asarray(dgm_h1, dtype=float)
    if dgm_h1.size == 0:
        return np.array([]), np.nan, np.array([], dtype=bool)

    births = dgm_h1[:, 0]
    deaths = dgm_h1[:, 1]
    lifetimes = deaths - births

    finite = np.isfinite(deaths)
    finite_lifetimes = lifetimes[finite]

    # "Not enough points" handling (SI: take all points as significant)
    if finite_lifetimes.size < min_finite_points:
        if small_diagram_policy == "all":
            sig_mask = np.ones(dgm_h1.shape[0], dtype=bool)
            return lifetimes, np.nan, sig_mask
        elif small_diagram_policy == "none":
            sig_mask = np.zeros(dgm_h1.shape[0], dtype=bool)
            return lifetimes, np.nan, sig_mask
        else:
            raise ValueError("small_diagram_policy must be 'all' or 'none'.")

    q1 = np.percentile(finite_lifetimes, 25)
    q3 = np.percentile(finite_lifetimes, 75)
    thr = q3 + k * (q3 - q1)

    sig_mask = np.isinf(deaths) | (lifetimes >= thr)
    return lifetimes, thr, sig_mask


def summarize_condition(cond, name="P", k=3.0, min_finite_points=10):
    dgm_h1 = cond.dgms[1]
    lifetimes, thr, sig = significant_h1_iqr(
        dgm_h1, k=k, min_finite_points=min_finite_points, small_diagram_policy="all"
    )

    births = dgm_h1[:, 0] if len(dgm_h1) else np.array([])
    deaths = dgm_h1[:, 1] if len(dgm_h1) else np.array([])

    print(f"== {name} ==")
    print(f"n_units: {len(cond.unit_ids)}")
    print(f"D shape: {cond.dissimilarity.D.shape}, M used: {cond.dissimilarity.M:.4g}")
    print(f"H1 points (total loops appearing across scales): {len(dgm_h1)}")

    if len(dgm_h1):
        print(
            f"  finite deaths: {np.isfinite(deaths).sum()}, inf deaths: {np.isinf(deaths).sum()}"
        )
        print(
            f"  lifetimes: min={np.min(lifetimes):.4g}, median={np.median(lifetimes):.4g}, max={np.max(lifetimes):.4g}"
        )
        if np.isfinite(thr):
            print(f"  IQR threshold (k={k}): {thr:.4g}")
        else:
            print(
                f"  IQR threshold: not computed (too few finite points) → treating all points as significant"
            )
        print(f"  significant H1 points: {sig.sum()}")
    print()


# -----------------------------------------------------------------------------
# Convenience: end-to-end for two conditions
# -----------------------------------------------------------------------------


@dataclass
class ConditionPH:
    unit_ids: List[Any]
    X_binned: np.ndarray
    dissimilarity: DissimilarityResult
    dgms: List[np.ndarray]


def compute_condition_ph(
    data: Union[Any, np.ndarray],
    *,
    bin_size: float,
    max_shift_bins: int,
    ep: Optional[Any] = None,
    binarize: bool = False,
    min_total_spikes: int = 1,
    filter_uniform: bool = False,
    uniform_eps_bins: int = 40,
    uniform_min_cluster_spikes: int = 20,
    uniform_max_avg_std: float = 55.0,
    M: Optional[float] = None,
    M_mode: Literal["auto", "ceil"] = "ceil",
    ripser_maxdim: int = 1,
    ripser_thresh: Optional[float] = None,
) -> ConditionPH:
    """Compute PDs for one condition.

    data can be:
    - a pynapple TsGroup (recommended), or
    - an already-binned array of shape (n_units, n_bins)

    Preprocessing implemented here is intentionally light.
    """
    if isinstance(data, np.ndarray):
        X = data
        unit_ids = list(range(X.shape[0]))
    else:
        X, unit_ids = bin_tsgroup(data, bin_size=bin_size, ep=ep, binarize=binarize)

    # Basic filtering
    X, unit_ids = filter_units_basic(X, unit_ids, min_total_spikes=min_total_spikes)

    if filter_uniform and X.shape[0] > 0:
        X, unit_ids, _scores = filter_units_by_cluster_std(
            X,
            unit_ids,
            eps_bins=uniform_eps_bins,
            min_cluster_spikes=uniform_min_cluster_spikes,
            max_avg_cluster_std=uniform_max_avg_std,
        )

    dis = dissimilarity_matrix(
        X,
        None,
        max_shift_bins=max_shift_bins,
        M=M,
        M_mode=M_mode,
        include_self_in_M=False,
        clip=True,
    )

    dgms = compute_persistence_diagrams(
        dis.D, maxdim=ripser_maxdim, thresh=ripser_thresh
    )

    return ConditionPH(unit_ids=unit_ids, X_binned=X, dissimilarity=dis, dgms=dgms)


def compute_two_condition_pds(
    cond_P: Union[Any, np.ndarray],
    cond_Q: Union[Any, np.ndarray],
    *,
    bin_size: float,
    max_shift_bins: int,
    ep_P: Optional[Any] = None,
    ep_Q: Optional[Any] = None,
    binarize: bool = False,
    # preprocessing
    min_total_spikes: int = 1,
    filter_uniform: bool = False,
    uniform_eps_bins: int = 40,
    uniform_min_cluster_spikes: int = 20,
    uniform_max_avg_std: float = 55.0,
    # scaling
    M_strategy: Literal[
        "separate", "shared_from_P", "shared_from_union"
    ] = "shared_from_union",
    M_mode: Literal["auto", "ceil"] = "ceil",
    # PH
    ripser_maxdim: int = 1,
    ripser_thresh: Optional[float] = None,
) -> Dict[str, Any]:
    """Compute PDs for two conditions and (optionally) a cross-dissimilarity matrix.

    Returns dict with keys:
        - 'P', 'Q': ConditionPH
        - 'M_shared': float
        - 'D_PQ': cross-system dissimilarity (or None if lengths mismatch)

    Scaling strategy
    ---------------
    The paper sometimes uses a *shared* M so that dissimilarities live on the
    same numeric scale across systems/conditions. fileciteturn1file8

    - 'separate': compute M separately for P and Q
    - 'shared_from_P': compute M from P only, reuse for Q
    - 'shared_from_union': compute M from max similarity over P∪Q

    In all cases, we exclude the diagonal when computing M within-system.
    """

    # Bin first (so we can compute a shared M if needed)
    if isinstance(cond_P, np.ndarray):
        Xp, idsP = cond_P, list(range(cond_P.shape[0]))
    else:
        Xp, idsP = bin_tsgroup(cond_P, bin_size=bin_size, ep=ep_P, binarize=binarize)

    if isinstance(cond_Q, np.ndarray):
        Xq, idsQ = cond_Q, list(range(cond_Q.shape[0]))
    else:
        Xq, idsQ = bin_tsgroup(cond_Q, bin_size=bin_size, ep=ep_Q, binarize=binarize)

    # Basic filtering
    Xp, idsP = filter_units_basic(Xp, idsP, min_total_spikes=min_total_spikes)
    Xq, idsQ = filter_units_basic(Xq, idsQ, min_total_spikes=min_total_spikes)

    if filter_uniform:
        Xp, idsP, _ = filter_units_by_cluster_std(
            Xp,
            idsP,
            eps_bins=uniform_eps_bins,
            min_cluster_spikes=uniform_min_cluster_spikes,
            max_avg_cluster_std=uniform_max_avg_std,
        )
        Xq, idsQ, _ = filter_units_by_cluster_std(
            Xq,
            idsQ,
            eps_bins=uniform_eps_bins,
            min_cluster_spikes=uniform_min_cluster_spikes,
            max_avg_cluster_std=uniform_max_avg_std,
        )

    # Choose M
    M_shared: Optional[float] = None
    if M_strategy == "separate":
        pass
    elif M_strategy == "shared_from_P":
        # Compute similarity within P and take its max (excluding diag)
        Sp = windowed_xcorr_similarity(Xp, None, max_shift_bins=max_shift_bins)
        if Sp.shape[0] > 1:
            iu = np.triu_indices(Sp.shape[0], k=1)
            M_shared = float(np.max(Sp[iu]))
        else:
            M_shared = 1.0
        if M_mode == "ceil":
            M_shared = float(np.ceil(M_shared))
    elif M_strategy == "shared_from_union":
        # Stack and compute within-union similarities
        if Xp.shape[1] != Xq.shape[1]:
            # Can't compute union similarity meaningfully if lengths differ.
            # Fall back to separate.
            M_shared = None
        else:
            Xu = np.vstack([Xp, Xq])
            Su = windowed_xcorr_similarity(Xu, None, max_shift_bins=max_shift_bins)
            if Su.shape[0] > 1:
                iu = np.triu_indices(Su.shape[0], k=1)
                M_shared = float(np.max(Su[iu]))
            else:
                M_shared = 1.0
            if M_mode == "ceil":
                M_shared = float(np.ceil(M_shared))
    else:
        raise ValueError(f"Unknown M_strategy: {M_strategy}")

    # Compute condition PH
    disP = dissimilarity_matrix(
        Xp,
        None,
        max_shift_bins=max_shift_bins,
        M=M_shared,
        M_mode=M_mode,
        include_self_in_M=False,
        clip=True,
    )
    dgmsP = compute_persistence_diagrams(
        disP.D, maxdim=ripser_maxdim, thresh=ripser_thresh
    )

    disQ = dissimilarity_matrix(
        Xq,
        None,
        max_shift_bins=max_shift_bins,
        M=M_shared if (M_strategy != "separate") else None,
        M_mode=M_mode,
        include_self_in_M=False,
        clip=True,
    )
    dgmsQ = compute_persistence_diagrams(
        disQ.D, maxdim=ripser_maxdim, thresh=ripser_thresh
    )

    P = ConditionPH(unit_ids=idsP, X_binned=Xp, dissimilarity=disP, dgms=dgmsP)
    Q = ConditionPH(unit_ids=idsQ, X_binned=Xq, dissimilarity=disQ, dgms=dgmsQ)

    # Optional cross-system dissimilarity (only if bin counts match)
    D_PQ: Optional[DissimilarityResult]
    if Xp.shape[1] != Xq.shape[1] or Xp.shape[0] == 0 or Xq.shape[0] == 0:
        D_PQ = None
    else:
        D_PQ = dissimilarity_matrix(
            Xp,
            Xq,
            max_shift_bins=max_shift_bins,
            M=M_shared,
            M_mode=M_mode,
            include_self_in_M=False,
            clip=True,
        )

    return {
        "P": P,
        "Q": Q,
        "M_shared": M_shared,
        "D_PQ": D_PQ,
    }


def make_pickle_safe(out):
    P = out["P"]
    Q = out["Q"]
    D_PQ = out["D_PQ"]

    safe = {
        "M_shared": out["M_shared"],
        "P": {
            "unit_ids": P.unit_ids,
            "X_binned": P.X_binned,
            "D": P.dissimilarity.D,
            "S": P.dissimilarity.S,
            "M": P.dissimilarity.M,
            "dgms": P.dgms,
        },
        "Q": {
            "unit_ids": Q.unit_ids,
            "X_binned": Q.X_binned,
            "D": Q.dissimilarity.D,
            "S": Q.dissimilarity.S,
            "M": Q.dissimilarity.M,
            "dgms": Q.dgms,
        },
        "D_PQ": (
            None
            if D_PQ is None
            else {
                "D": D_PQ.D,
                "S": D_PQ.S,
                "M": D_PQ.M,
            }
        ),
    }
    return safe


import numpy as np


def summarize_offdiag(D):
    D = np.asarray(D)
    n = D.shape[0]
    off = D[np.triu_indices(n, 1)]
    return {
        "min": float(off.min()),
        "p01": float(np.percentile(off, 1)),
        "p05": float(np.percentile(off, 5)),
        "median": float(np.percentile(off, 50)),
        "p95": float(np.percentile(off, 95)),
        "p99": float(np.percentile(off, 99)),
        "max": float(off.max()),
    }


import numpy as np


def h1_lifetimes(dgm_h1: np.ndarray) -> np.ndarray:
    dgm_h1 = np.asarray(dgm_h1, dtype=float)
    if dgm_h1.size == 0:
        return np.array([], dtype=float)
    births = dgm_h1[:, 0]
    deaths = dgm_h1[:, 1]
    return deaths - births


def shuffle_bins_per_unit(X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    X = np.asarray(X)
    Xs = X.copy()
    # independently permute time bins within each neuron
    for i in range(Xs.shape[0]):
        rng.shuffle(Xs[i])
    return Xs


def poisson_surrogate(X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    X = np.asarray(X)
    lam = X.mean(axis=1, keepdims=True)  # mean count per bin per unit
    return rng.poisson(lam=lam, size=X.shape).astype(float)


def circular_shift_surrogate(X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    X = np.asarray(X)
    T = X.shape[1]
    shifts = rng.integers(0, T, size=X.shape[0])
    return np.vstack([np.roll(X[i], shifts[i]) for i in range(X.shape[0])]).astype(
        float
    )


def surrogate_h1_threshold(
    X_data: np.ndarray,
    *,
    max_shift_bins: int,
    M_data: float,
    n_surrogates: int = 200,
    method: str = "shuffle",  # "shuffle" | "poisson" | "shift"
    seed: int = 0,
    ripser_thresh: float = np.inf,
):
    """
    Returns:
      threshold = max over surrogates of (max H1 lifetime in that surrogate PD)
      max_lifetimes = array of length n_surrogates
    """
    rng = np.random.default_rng(seed)
    max_lifetimes = np.zeros(n_surrogates, dtype=float)

    for k in range(n_surrogates):
        if method == "shuffle":
            Xs = shuffle_bins_per_unit(X_data, rng)
        elif method == "poisson":
            Xs = poisson_surrogate(X_data, rng)
        elif method == "shift":
            Xs = circular_shift_surrogate(X_data, rng)
        else:
            raise ValueError("method must be 'shuffle', 'poisson', or 'shift'")

        dis = dissimilarity_matrix(
            Xs,
            None,
            max_shift_bins=max_shift_bins,
            M=M_data,  # IMPORTANT: use original-data M for scale consistency
            M_mode="auto",  # use the M you pass; don't re-ceil it again
            include_self_in_M=False,
            clip=True,
        )
        dgms = compute_persistence_diagrams(dis.D, maxdim=1, thresh=ripser_thresh)
        lt = h1_lifetimes(dgms[1])
        lt = lt[np.isfinite(lt)]
        max_lifetimes[k] = float(lt.max()) if lt.size else 0.0

    threshold = float(max_lifetimes.max())
    return threshold, max_lifetimes


def significant_h1_points_smallPD(
    condition: ConditionPH,
    *,
    max_shift_bins: int,
    n_surrogates: int = 200,
    surrogate_method: str = "shuffle",
    seed: int = 0,
):
    dgm = condition.dgms[1]
    lt = h1_lifetimes(dgm)

    thr, max_lts = surrogate_h1_threshold(
        condition.X_binned,
        max_shift_bins=max_shift_bins,
        M_data=condition.dissimilarity.M,
        n_surrogates=n_surrogates,
        method=surrogate_method,
        seed=seed,
        ripser_thresh=np.inf,
    )
    sig = np.isinf(lt) | (lt > thr)
    return {
        "threshold": thr,
        "lifetimes": lt,
        "significant_mask": sig,
        "significant_points": dgm[sig],
        "null_max_lifetimes": max_lts,
    }


def main():
    import pynapple as nap

    # from analogous_cycles_pd import compute_two_condition_pds

    import numpy as np
    import matplotlib.pyplot as plt
    import pickle
    from pathlib import Path
    import scipy
    import position_tools as pt
    import spikeinterface.full as si
    import kyutils
    import pandas as pd
    import track_linearization as tl

    animal_name = "L14"
    date = "20240611"
    analysis_path = Path("/stelmo/kyu/analysis") / animal_name / date
    out_dir = analysis_path / "analogous_cycles"
    out_dir.mkdir(parents=True, exist_ok=True)

    num_sleep_epochs = 5
    num_run_epochs = 4

    epoch_list, run_epoch_list = kyutils.get_epoch_list(
        num_sleep_epochs=num_sleep_epochs, num_run_epochs=num_run_epochs
    )
    sleep_epoch_list = [epoch for epoch in epoch_list if epoch not in run_epoch_list]

    regions = ["v1", "ca1"]

    time_bin_size = 2e-3
    sampling_rate = int(1 / time_bin_size)
    speed_threshold = 4  # cm/s
    position_offset = 10

    trajectory_types = [
        "center_to_left",
        "left_to_center",
        "center_to_right",
        "right_to_center",
    ]

    with open(analysis_path / "timestamps_ephys.pkl", "rb") as f:
        timestamps_ephys = pickle.load(f)

    with open(analysis_path / "timestamps_position.pkl", "rb") as f:
        timestamps_position_dict = pickle.load(f)

    with open(analysis_path / "timestamps_ephys_all.pkl", "rb") as f:
        timestamps_ephys_all_ptp = pickle.load(f)

    with open(analysis_path / "position.pkl", "rb") as f:
        position_dict = pickle.load(f)

    with open(analysis_path / "body_position.pkl", "rb") as f:
        body_position_dict = pickle.load(f)

    with open(analysis_path / "trajectory_times.pkl", "rb") as f:
        trajectory_times = pickle.load(f)

    sorting = {}
    for region in regions:
        sorting[region] = si.load(analysis_path / f"sorting_{region}")

    def _sampling_rate(t_position: np.ndarray) -> float:
        return (len(t_position) - 1) / (t_position[-1] - t_position[0])

    def get_tsgroup(sorting):
        data = {}
        for unit_id in sorting.get_unit_ids():
            data[unit_id] = nap.Ts(
                t=timestamps_ephys_all_ptp[sorting.get_unit_spike_train(unit_id)]
            )
        tsgroup = nap.TsGroup(data, time_units="s")
        return tsgroup

    spikes = {}
    for region in regions:
        spikes[region] = get_tsgroup(sorting[region])

    all_ep = {}
    trajectory_ep = {}
    for epoch in run_epoch_list:
        all_ep[epoch] = nap.IntervalSet(
            start=timestamps_position_dict[epoch][position_offset],
            end=timestamps_position_dict[epoch][-1],
        )
        trajectory_ep[epoch] = {}
        for trajectory_type in trajectory_types:
            trajectory_ep[epoch][trajectory_type] = nap.IntervalSet(
                start=trajectory_times[epoch][trajectory_type][:, 0],
                end=trajectory_times[epoch][trajectory_type][:, -1],
            )
    speed = {}
    movement = {}
    for epoch in epoch_list:
        speed[epoch] = nap.Tsd(
            t=timestamps_position_dict[epoch][position_offset:],
            d=pt.get_speed(
                position=position_dict[epoch][position_offset:],
                time=timestamps_position_dict[epoch][position_offset:],
                sampling_frequency=(
                    len(timestamps_position_dict[epoch][position_offset:]) - 1
                )
                / (
                    timestamps_position_dict[epoch][position_offset:][-1]
                    - timestamps_position_dict[epoch][position_offset:][0]
                ),
                sigma=0.1,
            ),
        )
        movement[epoch] = (
            speed[epoch].threshold(speed_threshold, method="above").time_support
        )

    # Suppose you already have two TsGroups:
    # tsP = ...  # condition P spikes
    # tsQ = ...  # condition Q spikes
    epoch1 = run_epoch_list[0]
    epoch2 = run_epoch_list[3]
    tsP = spikes["v1"].restrict(movement[epoch1])
    tsQ = spikes["v1"].restrict(movement[epoch2])

    # tsP = tsP[(tsP.rate > 0.5)]
    # tsQ = tsQ[(tsQ.rate > 0.5)]

    # Optional: define epochs if your TsGroups include longer recordings
    # epP = nap.IntervalSet(start=[t0], end=[t1])
    # epQ = nap.IntervalSet(start=[t2], end=[t3])

    bin_size = 0.075  # seconds
    max_shift_s = 1.0
    max_shift_bins = int(round(max_shift_s / bin_size))
    min_total_spikes = 100
    filter_uniform = True

    out = compute_two_condition_pds(
        tsP,
        tsQ,
        bin_size=bin_size,
        max_shift_bins=max_shift_bins,
        # If you want comparable scales across conditions:
        M_strategy="shared_from_union",  # or "shared_from_P" or "separate"
        M_mode="ceil",  # matches authors' common practice
        # light preprocessing:
        min_total_spikes=min_total_spikes,
        filter_uniform=filter_uniform,  # can turn on if you want
        ripser_maxdim=1,
        ripser_thresh=float("inf"),
    )

    P = out["P"]
    Q = out["Q"]

    summarize_condition(P, "P")
    summarize_condition(Q, "Q")

    print("P D off-diag stats:", summarize_offdiag(P.dissimilarity.D))
    print("Q D off-diag stats:", summarize_offdiag(Q.dissimilarity.D))

    print("P H1 points:", P.dgms[1])
    print("Q H1 points:", Q.dgms[1])

    sigP = significant_h1_points_smallPD(
        P, max_shift_bins=max_shift_bins, n_surrogates=200, surrogate_method="shuffle"
    )
    sigQ = significant_h1_points_smallPD(
        Q, max_shift_bins=max_shift_bins, n_surrogates=200, surrogate_method="shuffle"
    )

    print("P surrogate threshold:", sigP["threshold"])
    print("P lifetimes:", sigP["lifetimes"], "significant:", sigP["significant_mask"])

    print("Q surrogate threshold:", sigQ["threshold"])
    print("Q lifetimes:", sigQ["lifetimes"], "significant:", sigQ["significant_mask"])
    # with open(out_dir / "out.pkl", "wb") as f:
    #     pickle.dump(out, f)
    safe_out = make_pickle_safe(out)
    with open(
        out_dir
        / f"{epoch1}_{epoch2}_bin_size_{bin_size}_max_shift_bins_{max_shift_bins}_min_total_spikes_{min_total_spikes}_filter_uniform_{filter_uniform}.pkl",
        "wb",
    ) as f:
        pickle.dump(safe_out, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
