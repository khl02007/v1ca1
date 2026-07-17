# Repository scientific and correctness audit

**Review date:** 2026-07-16  
**Repository:** `v1ca1`  
**Baseline commit:** `ed65b7d1cca64335c680e47d650cf15a55218096`  
**Review target:** the current working tree, including its uncommitted paper-figure and test changes

**Implementation update (2026-07-16):** F1 has been resolved by replacing shuffled lap-wise 1D decoder folds with deterministic contiguous-time folds. The remaining findings describe the audit baseline unless separately marked resolved.

## Executive summary

This is a substantial, scientifically serious analysis repository for simultaneous V1 and CA1 recordings in freely moving rats during W-track behavior and sleep. The newer code has many good properties: explicit session paths, modern tabular and array formats, centralized task-coordinate construction, extensive argument and shape checks, train/test separation in several major models, unique run logs, and a large synthetic test suite. The current suite passes in full: **687 tests passed, 0 skipped, with 2 warnings**.

I would nevertheless avoid treating every current output as publication-final until the highest-priority findings below are resolved or empirically shown not to affect the reported conclusions. The most consequential issues are:

1. **Resolved:** the full-epoch 1D decoder formerly ran its state-space recursion across disjoint held-out time segments as though they were adjacent.
2. Ripple-event resume logic decides reuse from the presence of an epoch alone, not from detector parameters or input provenance; one output file can also mix epochs produced with different settings.
3. Missing position support can be classified as immobility instead of unknown.
4. Position video streams are assigned to epochs by container order rather than verified identity.
5. Some cross-condition and cross-trajectory analyses select neurons using the held-out target data before cross-validation.
6. Several figure-level tests treat neurons as independent observations despite only four registered animals; ripple-unit significance is also used without a defined multiple-testing family.
7. A few additional current paths are nondeterministic, use target-derived fallback rates, or construct invalid null pairings.

These findings do **not** imply that the main biological conclusions are false. A read-only check of the representative L14/20240611 session found coherent absolute clocks, matching video order, and fully finite cleaned DLC. Rather, the code does not yet enforce those properties generally, and several analysis paths contain direct implementation or inference problems that should be checked against saved results.

### How to read the priorities

- **P0:** fix or explicitly invalidate with a result-level sensitivity check before relying on the affected result.
- **P1:** material risk to correctness, inference, or reproducibility; address before final publication/archive.
- **P2:** important robustness, scalability, or maintainability improvement.
- **Confirmed defect:** follows directly from the current execution path.
- **Latent integrity risk:** the representative data were coherent, but the code can silently accept a bad future or legacy input.
- **Methodological risk:** the computation runs as written, but the inferential claim is stronger than the design supports.

## What the repository is doing

The package implements a multi-stage systems-neuroscience workflow:

```text
NWB acquisition and epoch metadata       DLC / video tracking       curated sorting
                 |                              |                         |
                 +---------- timestamps -------+                         |
                                |                                        |
                     cleaned head/body position                          |
                                |                                        |
                movement, immobility, W-track geometry, laps             |
                                |                                        |
                                +------------- spike times --------------+
                                                     |
             +----------------------+----------------+-------------------+
             |                      |                |                   |
      task progression        ripple analyses   1D decoding      sleep/theta/xcorr
   tuning, MI, GLMs, CV       detection, GLMs,   and error          state analyses
      and transfer              decoding
             |                      |                |
             +----------------------+----------------+
                                    |
                         signal-dimensionality and
                          manuscript figure assembly
```

The key scientific ideas are:

- build a direction-aware task-progression coordinate over W-track trajectories;
- compare spatial and task-progression tuning and encoding across light and dark epochs;
- quantify generalization between trajectories and conditions with GLMs and Bayesian decoders;
- detect CA1 sharp-wave ripples and ask whether CA1 population activity predicts or agrees with V1 activity;
- estimate signal dimensionality with cross-validated PCA and a MEME estimator adapted to grouped laps;
- assemble a four-animal manuscript figure set from saved session artifacts.

The modern artifact conventions are sensible: parquet for summary tables, pynapple-backed NPZ for temporal objects, and xarray/NetCDF for large model results. The main problem is not the high-level organization; it is that provenance and statistical boundaries are not enforced uniformly from one stage to the next.

## Review scope and validation

I traced the shared NWB, timestamp, position, trajectory, sorting, and task-progression loaders, then reviewed representative implementations from each main scientific branch. I also inspected figure-level aggregation and statistics because that is where otherwise sound per-session computations become manuscript claims.

Validation performed:

- `pytest -q -p no:cacheprovider` in the existing `v1ca1` conda environment: **687 passed, 2 warnings in 76.60 s**, with no skipped tests. The warnings come from an intentionally unsorted IntervalSet fixture in `test_export_augmented_nwb.py`.
- `python -m pip check`: **no broken requirements found**.
- A read-only representative-data check of `/stelmo/nwb/raw/L1420240611.nwb` and its L14 analysis artifacts.
- A direct check of the README timestamp command, which currently fails because `--ephys-format` is no longer a valid flag.
- Static inspection of the installed `replay_trajectory_classification` prediction behavior for the 1D decoding issue.

Important limits:

- I did not rerun every data-dependent scientific analysis across every animal. Many required NWB and intermediate artifacts are intentionally outside git, and several stored arrays are tens of GiB.
- This is a deep path-oriented audit, not a formal proof of every one of roughly 146,000 tracked Python lines.
- The working tree already contained large uncommitted changes, particularly in manuscript figures. This report describes the code as observed and does not attempt to distinguish committed from pre-existing uncommitted logic.

## Prioritized findings

| ID | Priority | Type | Finding | Main consequence |
|---|---:|---|---|---|
| F1 | Resolved | Confirmed defect, fixed 2026-07-16 | 1D state recursion formerly crossed disjoint prediction segments | Replaced with independent contiguous-time fold predictions |
| F2 | P0 | Confirmed defect | Ripple outputs are resumed by epoch presence, not configuration/provenance | Parameter changes can silently reuse old events; one parquet can mix detector settings |
| F3 | P0 | Confirmed defect / data semantics | Missing position support becomes immobility | State labels and every speed-gated downstream analysis can be biased |
| F4 | P0 | Latent integrity risk | Video streams are mapped to epochs by order | A reordered NWB container can silently relabel all position timestamps |
| F5 | P0/P1 | Confirmed selection leakage | Target-condition data select neurons before transfer CV | Held-out performance and the evaluated cell cohort are optimistically conditioned |
| F6 | P0/P1 | Methodological risk | Figure inference pools neurons across four animals | P-values can be much too small because within-animal dependence is ignored |
| F7 | P1 | Confirmed defect | Ripple-decoding null concatenates unequal-length response blocks | Null rows no longer align with design rows by ripple |
| F8 | P1 | Confirmed defect | 1D spike binning always leaves the final output row empty | Systematic endpoint misalignment and a slightly incorrect requested bin width |
| F9 | P1 | Confirmed defect / reproducibility | Mutual-information shuffles have no seed | Corrected MI changes across runs and cannot be reconstructed from logs |
| F10 | P1 | Confirmed analysis mismatch | Cross-trajectory encoding fills source-curve gaps with a target-derived rate | The nominal source model partly uses target training data in unsupported bins |
| F11 | P1 | Integrity / multiplicity | Broad NPZ fallback and uncorrected ripple-unit significance | Stale data can be selected silently; false-positive families are undefined |
| F12 | P1/P2 | Provenance / operations | Failed batch fits can exit successfully and logs lack input fingerprints | Incomplete result sets can look successful and fixed-name outputs are hard to reproduce |
| F13 | P2 | Scientific robustness | Lap and tuning support QC is permissive | False pokes, unsupported bins, and best-side selection can bias tuning comparisons |
| F14 | P2 | Performance / archive | Exact clocks and uncompressed NetCDF outputs are extremely large | Repeated multi-GiB loads slow analyses and make archival reproduction fragile |

## Detailed correctness findings

### F1 — 1D prediction formerly joined separated segments into one state sequence

**Status: resolved on 2026-07-16.** The original implementation assigned shuffled laps to folds, compacted separated held-out intervals, and passed each compacted fold through one state-space recursion. That allowed causal and acausal inference to bridge real temporal gaps.

The active workflow now uses [`build_contiguous_time_folds`](src/v1ca1/decoding/_1d.py) to partition every decoder time bin into deterministic, exhaustive, non-overlapping contiguous chunks. [`build_fold_training_mask`](src/v1ca1/decoding/fit_1d.py) trains on eligible trajectory/movement bins outside the held-out chunk, while [`predict_region`](src/v1ca1/decoding/predict_1d.py) predicts every bin inside that chunk in one independent call. Each fold therefore begins from an explicitly configured uniform initial condition, and stationary inter-lap bins remain in the prediction.

The shuffled lap-wise helpers and random-state CLI controls were removed. New artifacts carry a `cv_contiguous_time` path token and record `cv_scheme="contiguous_time"`, so historical lap-wise fits cannot be loaded silently. Regression tests verify contiguous exhaustive folds, movement-only default training masks, inclusion of stationary prediction bins, and rejection of noncontiguous fold assignments ([`test_decoding_1d.py`](tests/test_decoding_1d.py), [`test_predict_1d.py`](tests/test_predict_1d.py)).

### F2 — Ripple resume and overwrite semantics can mix scientific configurations

**Evidence.** [`plan_ripple_epoch_execution`](src/v1ca1/ripple/detect_ripples.py#L973-L991) skips an epoch solely because it is already present in the flattened output. It does not compare detector threshold, speed gating, position offset, channel IDs, notch setting, LFP cache identity, or input hashes. This decision is made before detection in [`get_ripple_times`](src/v1ca1/ripple/detect_ripples.py#L1152-L1200). When overwriting selected epochs, other epochs are preserved and merged into the same parquet at [`detect_ripples.py`](src/v1ca1/ripple/detect_ripples.py#L1204-L1209).

The `--overwrite` help text says only that it recomputes the cached LFP, although it also determines event replacement ([`detect_ripples.py`](src/v1ca1/ripple/detect_ripples.py#L1301-L1305)). Tests verify missing-epoch resume and overwrite, but do not change a detector parameter between runs.

**Impact.** Running the same command with a different `--zscore-threshold` can silently retain old events. A partially overwritten table can contain epochs generated with different thresholds or filters without row-level provenance.

**Recommendation.** Give the event artifact a versioned configuration fingerprint. Reuse only if the full fingerprint and input fingerprints match. Otherwise fail with an actionable message or recompute all selected epochs. Store the configuration/run ID in the parquet metadata or columns, not only in a detached log. Make `--overwrite-events` and `--overwrite-lfp-cache` separate if their semantics differ.

### F3 — Unknown position time can be called immobility

**Evidence.** [`has_any_finite_position`](src/v1ca1/helper/get_immobility_times.py#L70-L75) accepts an epoch when any single coordinate is finite; the test explicitly accepts a row with finite X and NaN Y ([`test_get_immobility_times.py`](tests/test_get_immobility_times.py#L82-L85)). Speed is produced from the whole position array in [`build_speed_tsd`](src/v1ca1/helper/session.py#L734-L769). Movement is the above-threshold support, and immobility is computed as the full epoch minus movement ([`get_immobility_times.py`](src/v1ca1/helper/get_immobility_times.py#L45-L67)).

NaN speed is not evidence of zero speed. Under this complement construction, unobserved or smoothing-contaminated support can enter the immobility set.

**Impact.** This affects exported immobility, speed-gated ripple detection, firing-rate filters, sleep/state comparisons, and any analysis restricted to movement. A read-only L14 check found finite cleaned DLC, so this is principally a missing-data semantics problem for other sessions or future cleaning changes.

**Recommendation.** Define valid observation support from rows where both X and Y are finite, impose coverage and maximum-gap criteria, and compute movement and immobility only within that support. Export a third `unknown_position_times` state and coverage QC. Never define immobility as the complement over time that lacks a usable speed estimate.

### F4 — Position video-to-epoch identity is not validated

**Evidence.** [`get_timestamps_position`](src/v1ca1/helper/get_timestamps.py#L172-L193) converts `video.time_series.keys()` to a list, checks only that its length matches the epoch list, and zips the two orders. It does not validate filename tokens, timestamp monotonicity/finiteness, or overlap with the assigned NWB epoch. The later [`validate_epoch_alignment`](src/v1ca1/helper/get_timestamps.py#L142-L169) checks ephys segments only, and only requires overlap.

**Impact.** Reordering the NWB container can silently assign each video clock to the wrong epoch. Exact downstream DLC-to-saved-timestamp validation would then preserve, not discover, the wrong labels.

**Recommendation.** Match video streams by an epoch token in the filename or by unique overlap with validated epoch bounds; reject ambiguous matches. Validate each position clock as finite, strictly increasing, and appropriately contained. Save the explicit `video_stream -> epoch` mapping in the run record.

The current L14/20240611 NWB happens to have matching names and order, so no mismatch was observed in that representative session.

### F5 — Target data select cells before cross-validation

There are two related paths.

**Dark-to-light transfer.** The default tuning-stability filter is enabled in [`dark_light_transfer.py`](src/v1ca1/task_progression/dark_light_transfer.py#L1750-L1766). Full light and dark movement firing rates and saved whole-epoch odd/even stability select cells at [`dark_light_transfer.py`](src/v1ca1/task_progression/dark_light_transfer.py#L1976-L2017). Outer light folds are built only afterward ([`dark_light_transfer.py`](src/v1ca1/task_progression/dark_light_transfer.py#L2046-L2050)), and held-out light laps are scored at [`dark_light_transfer.py`](src/v1ca1/task_progression/dark_light_transfer.py#L2190-L2251). The stability table itself uses all odd and even laps ([`stability.py`](src/v1ca1/task_progression/stability.py#L176-L214)).

**Cross-trajectory decoding.** Whole-epoch movement firing rate and optional maximum stability across trajectories are calculated before source-to-target decoding ([`decoding_comparison.py`](src/v1ca1/task_progression/decoding_comparison.py#L364-L455), [`decoding_comparison.py`](src/v1ca1/task_progression/decoding_comparison.py#L1211-L1297)). The target trajectory therefore helps determine which cells enter the decoder.

**Impact.** The prediction code can still be train/test separated conditional on the chosen cells, but the reported cohort and performance are selected using the test outcome. This can preferentially retain target-responsive or stable cells and overstate transfer.

**Recommendation.** Perform unit inclusion inside each outer fold using target-training laps only, or define an independent/prespecified cohort from a different epoch or quality metric. Report an all-curated-unit sensitivity analysis and the number/identity of cells per fold. When comparing light and dark, also show results on the same fixed cell set.

The nested hyperparameter selection inside the dark/light GLM is otherwise a strong implementation and should be preserved.

### F6 — Figure-level inference needs animal/session clustering

**Evidence.** The manuscript registry currently contains four animals—L12, L14, L15, and L19—while L16 is commented out ([`datasets.py`](src/v1ca1/paper_figures/datasets.py#L21-L27)). One Figure 3 analysis pools cell-level similarity values, selects cells using ripple significance and dark firing rate ([`figure_3.py`](src/v1ca1/paper_figures/figure_3.py#L6093-L6153)), runs a Mann–Whitney test on cells, and samples arbitrary cells for its null ([`figure_3.py`](src/v1ca1/paper_figures/figure_3.py#L6166-L6206)).

Ripple GLM creates per-unit empirical p-values ([`ripple_glm.py`](src/v1ca1/ripple/ripple_glm.py#L1663-L1726)), and Figure 3 directly calls `p < 0.05` units significant across units, epochs, and offsets ([`figure_3.py`](src/v1ca1/paper_figures/figure_3.py#L3319-L3345)). No multiple-testing family or FDR-adjusted field is carried into that figure path.

Figure 2 additionally draws `"*"` significance labels from a constant, without computing a test from the table supplied to the panel ([`figure_2.py`](src/v1ca1/paper_figures/figure_2.py#L57-L60), [`figure_2.py`](src/v1ca1/paper_figures/figure_2.py#L249-L295), [`figure_2.py`](src/v1ca1/paper_figures/figure_2.py#L1090-L1104)). This is a presentation integrity risk because the mark can become stale while tests still pass.

**Impact.** Neurons from one animal are correlated and are not independent experimental replicates. Treating hundreds of cells as the sample size can make uncertainty and p-values much too optimistic. Uncorrected per-cell thresholding also creates an undefined false-positive family.

**Recommendation.** Make animal, or animal/session where appropriate, the resampling/permutation unit. Plot animal-level effects, use a clustered bootstrap/permutation or hierarchical model, and report leave-one-animal-out sensitivity. Define each multiple-testing family in advance and carry BH/FDR q-values into saved outputs. Generate every significance mark from the plotted analysis artifact, with test name, unit of replication, effect size, interval, exact p/q, and sample counts available in panel metadata. Record an eligibility rationale for every included or excluded animal.

### F7 — The ripple-decoding shuffle can destroy row alignment

**Evidence.** Ripple folds correctly keep all bins from a ripple together ([`ripple_decoding_glm.py`](src/v1ca1/ripple/ripple_decoding_glm.py#L389-L410)). The null, however, extracts variable-length response blocks, permutes them, and concatenates them in the new order ([`ripple_decoding_glm.py`](src/v1ca1/ripple/ripple_decoding_glm.py#L413-L429)). The design matrix remains in original row order.

If ripple A has 3 bins and ripple B has 7, placing B's response first assigns the first three B bins to A's design rows and the remaining four B bins across subsequent destination boundaries. The null changes duration and boundary structure in addition to breaking the intended ripple association.

**Recommendation.** Use fixed-length windows, shuffle only among equal-length blocks, or map a source ripple into a destination block with a documented length-preserving resampling rule. Add an unequal-length regression test that verifies every destination ripple receives exactly one complete, aligned null block.

### F8 — The 1D spike indicator has an endpoint/bin-count defect

**Evidence.** [`build_time_grid`](src/v1ca1/decoding/_1d.py#L730-L751) uses `ceil` and then `linspace`, so the actual interval is not exactly the requested `time_bin_size_s`. [`get_spike_indicator`](src/v1ca1/decoding/_1d.py#L754-L779) digitizes against only `time_array[1:-1]` but requests `N` output rows. `np.digitize` can therefore produce only indices `0` through `N-2`; the final row is always zero.

**Impact.** At minimum there is a systematic empty endpoint row and an inaccurate stored bin size. This also complicates alignment among spike counts, interpolated position, and posterior time coordinates.

**Recommendation.** Construct explicit `N+1` bin edges and `N` centers, then use the same edges for spike counts and coordinates. Store the actual bin width and add tests with spikes in the first, interior, and final bins.

### F9 — Corrected mutual information is not reproducible

**Evidence.** [`compute_shuffled_si`](src/v1ca1/task_progression/mutual_info.py#L121-L161) creates `np.random.default_rng()` without a seed. The CLI exposes shuffle count and minimum shift but no seed ([`mutual_info.py`](src/v1ca1/task_progression/mutual_info.py#L346-L407)), and the run log records none ([`mutual_info.py`](src/v1ca1/task_progression/mutual_info.py#L487-L507)). The default is only 50 shuffles, and `n_shuffles <= 0` is not rejected before division.

The movement-axis circular-shift idea is reasonable, but place and task-progression representations also differ in bin count and occupancy. Subtracting mean null plug-in MI does not make them automatically equal-complexity model comparisons.

**Recommendation.** Add a logged base seed and stable substreams keyed by animal/date/region/epoch/representation; validate a positive shuffle count; save the null distribution or at least its quantiles. Treat MI as descriptive unless bin/occupancy sensitivity is shown. Use the repository's held-out likelihood comparisons for stronger model evidence.

### F10 — Source transfer tuning can use a target-derived fill rate

**Evidence.** In cross-trajectory encoding, the source and target tuning curves are trained separately, which is good. But for each unit, `train_rate` is computed from the **target** training fold and then used as `fill_rate` for both the target and source curves ([`encoding_comparison.py`](src/v1ca1/task_progression/encoding_comparison.py#L746-L804)). Unsupported bins in the nominal source model therefore use target-training activity.

**Impact.** This is not leakage from the held-out target fold, but it weakens the stated interpretation that the source model is trained only on source laps and can favor transfer when source occupancy is incomplete.

**Recommendation.** Fill source gaps from source training data only, or mark/drop target samples whose corresponding source bins are unsupported. If target rate calibration is intentional, name the model “source shape with target-rate calibration” and compare it with a strictly source-only transfer model.

### F11 — Canonical artifact fallback and ripple multiplicity are too permissive

**Artifact selection.** Shared timestamp loaders broadly catch any failure reading an existing modern NPZ and silently use a legacy pickle if one exists: [`load_epoch_tags`](src/v1ca1/helper/session.py#L133-L164), [`load_ephys_timestamps_all`](src/v1ca1/helper/session.py#L178-L206), [`load_ephys_timestamps_by_epoch`](src/v1ca1/helper/session.py#L209-L269), and [`load_position_timestamps`](src/v1ca1/helper/session.py#L272-L317). A corrupt NPZ, incompatible pynapple version, or missing dependency can therefore change the dataset without a warning.

**Model selection.** Ripple GLM filters V1 responses and CA1 predictors over all ripples before constructing folds ([`ripple_glm.py`](src/v1ca1/ripple/ripple_glm.py#L1121-L1170)). Fold fitting and preprocessing are otherwise correctly train-only ([`ripple_glm.py`](src/v1ca1/ripple/ripple_glm.py#L1334-L1413)). The ridge selector chooses from the same held-out CV summaries ([`ripple_glm_select_ridge.py`](src/v1ca1/ripple/ripple_glm_select_ridge.py#L3-L18), [`ripple_glm_select_ridge.py`](src/v1ca1/ripple/ripple_glm_select_ridge.py#L213-L245)); reporting those scores after choosing the maximum is optimistic.

**Recommendation.** If a canonical artifact exists but cannot be read or validated, fail with the original cause. Permit legacy fallback only when the canonical file is absent and an explicit option is set. For ripple models, use independent or fold-local unit inclusion, nested CV for ridge selection, and BH/hierarchical population inference.

### F12 — Some batch failures and logs can make incomplete analyses look complete

Several batch loops catch broad exceptions, print a skip, and can exit successfully even if every requested fit failed. Representative paths include [`cv_pca.py`](src/v1ca1/signal_dim/cv_pca.py#L2406-L2442), [`dark_light_glm.py`](src/v1ca1/task_progression/dark_light_glm.py#L2235-L2259), [`motor.py`](src/v1ca1/task_progression/motor.py#L4010-L4043), and [`ripple_trajectory_identity.py`](src/v1ca1/ripple/ripple_trajectory_identity.py#L1722-L1749). By contrast, ripple GLM has an explicit all-failed check.

The shared logger is a good start, but it records only package version, commit, dirty boolean, parameters, and outputs ([`run_logging.py`](src/v1ca1/helper/run_logging.py#L60-L92)). It does not fingerprint inputs, dependencies, backend, schema, or the dirty diff; git is queried relative to process CWD; and output JSON is written non-atomically.

**Recommendation.** Catch only expected data-insufficiency exceptions. Preserve traceback and structured failure status, and exit nonzero if no requested result succeeded. Extend the run record with argv, CWD, Python/platform, dependency lock hash, numerical backend, input fingerprints, output schema version, status/error/timing, and a diff hash when dirty. Embed the producing run ID in each artifact and use atomic writes.

## Scientific interpretation and analysis-specific review

### Task-progression tuning and encoding

The direction-aware construction in [`task_progression/_session.py`](src/v1ca1/task_progression/_session.py#L137-L360) is explicit and internally coherent: left turns map through one half of the progression coordinate, right turns through the other, and travel direction is normalized. Session preparation performs unusually strong epoch/source checks.

Two interpretation cautions remain:

- [`interpolate_nans`](src/v1ca1/task_progression/tuning_analysis.py#L124-L142) linearly fills internal gaps and constant-extrapolates edges before similarity is calculated. Similarity should be evaluated on jointly occupied/supportable bins, with coverage reported.
- Pooled same-turn and same-arm similarity takes the maximum side per cell ([`tuning_analysis.py`](src/v1ca1/task_progression/tuning_analysis.py#L559-L597)). This is a valid “best side” statistic but is upward-selected and should be labeled as such, not interpreted as a neutral pooled mean.

The tuning significance workflow is a positive example: it uses deterministic unit-specific circular shifts, a +1 empirical-p correction, and Benjamini-Hochberg FDR ([`tuning_analysis.py`](src/v1ca1/task_progression/tuning_analysis.py#L216-L315), [`tuning_analysis.py`](src/v1ca1/task_progression/tuning_analysis.py#L619-L706)). Ripple analyses should adopt the same explicit family-level approach.

### Motor and dark/light GLMs

The task-progression motor GLM is the best internal template. It combines lap-level outer CV, inner hyperparameter selection, training-only transforms, training-only unit inclusion, refit logic, and shuffled nulls. Preserve that design when repairing transfer and ripple paths.

For dark/light analyses, separate three claims clearly:

1. within-condition held-out prediction;
2. transfer after rate calibration;
3. strict source-only transfer.

They answer different biological questions. Current naming sometimes obscures that distinction.

### Ripple detection and CA1-to-V1 models

The detector has strong LFP cache schema and parameter validation, explicit modern-artifact requirements, speed-gating checks, and useful per-epoch logs. Additional issues to address:

- Any exception while reading acquisition sampling rate silently falls back to 30 kHz ([`detect_ripples.py`](src/v1ca1/ripple/detect_ripples.py#L629-L665)). A wrong rate changes filters and decimation. Fail loudly or require an explicit, logged override.
- The cropped epoch is zero-phase filtered without padding or an explicit transient exclusion margin ([`detect_ripples.py`](src/v1ca1/ripple/detect_ripples.py#L574-L665)). Pad before filtering, crop afterward, and report event distances from boundaries.
- Native-rate bandpass followed by stride decimation should be replaced with an explicitly anti-aliased resampler unless the filter's stopband attenuation is documented as sufficient.
- The cache validates settings but not the raw NWB identity or timestamp content; add source fingerprints.

The ripple GLM's held-out prediction is technically careful after folds are constructed. However, the default same-window CA1 and V1 spike counts support **association/prediction**, not directional communication. Shared ripple amplitude/duration, arousal/state, elapsed time, and population rate can drive both regions. Directional language needs prespecified source-before-target windows plus negative offsets and relevant nuisance covariates. The repository already has mean-activity and temporal-offset controls, which are valuable foundations.

### Signal dimensionality

The cvPCA implementation uses held-out groups and training-side normalization, and it saves both signed and nonnegative spectra. Those are strengths. The main scientific assumptions to expose are:

- Cell selection uses firing rate and condition modulation estimated from the full light/dark tensors ([`cv_pca.py`](src/v1ca1/signal_dim/cv_pca.py#L197-L260)). This can condition dimensionality on the same repeat data being evaluated. Show fixed independent/all-cell sensitivities.
- Negative cross-validated spectral components are clipped before participation ratio ([`cv_pca.py`](src/v1ca1/signal_dim/cv_pca.py#L273-L325)). Retain and plot signed spectra, and show how clipping affects low-SNR estimates.
- MEME is adapted from independent repeated responses to disjoint groups of sequential laps, an assumption the code documents honestly ([`meme.py`](src/v1ca1/signal_dim/meme.py#L463-L485)). Serial drift and behavioral nonstationarity can violate repeat independence. Use blocked/temporally separated sensitivity analyses or simulations matched to lap autocorrelation.
- MEME z-scoring estimates center and scale from all repeats ([`meme.py`](src/v1ca1/signal_dim/meme.py#L370-L391)), coupling repeat noise. Estimate scaling independently or validate bias with simulations.

Random-seed repeat intervals quantify estimator Monte Carlo sensitivity, not animal-level biological uncertainty; label them accordingly.

### Laps and behavioral support

Trajectory intervals are built from adjacent allowed poke transitions ([`get_trajectory_times.py`](src/v1ca1/helper/get_trajectory_times.py#L290-L329)). There is no debounce, plausible-duration check, endpoint/path concordance, or rejected-event QC. A duplicate or false poke can truncate or relabel a lap and then propagate into every lap-wise CV analysis.

The parquet saver checks shape but not finite values or `end > start` ([`get_trajectory_times.py`](src/v1ca1/helper/get_trajectory_times.py#L245-L273)). The shared loader filters to expected labels and reconstructs absent combinations as empty intervals without rejecting unknown epoch/type rows ([`session.py`](src/v1ca1/helper/session.py#L582-L630)). Add strict schema/domain checks, lap-duration and path-concordance QC, and an explicit rejected-lap table.

### Figure assembly and example selection

Manuscript examples are manually registered in code. That is workable, but their selection rule and whether selection occurred before looking at the plotted outcome should be stored next to the IDs. Quantitative claims should never depend on hard-coded stars or manually copied p-values. Figure tests should recompute and assert the statistic from a small input table, not only assert the placement of an annotation.

## Data integrity, reproducibility, and performance

### What is already strong

- Ephys timestamp splitting checks dimensionality, finiteness, monotonicity, segment count, and empty segments ([`get_timestamps.py`](src/v1ca1/helper/get_timestamps.py#L53-L139)).
- Combined cleaned DLC uses unit-bearing columns and exact row-count/timestamp checks ([`session.py`](src/v1ca1/helper/session.py#L382-L452)).
- Major task-progression loaders compare epoch order across sources and require current artifacts.
- Sorting frame indices are converted through acquired timestamps rather than assuming a constant clock.
- NWB export stages and replaces output atomically.
- Run logs are unique per process/run, avoiding concurrent filename collisions.
- The full local test environment exercised all 687 collected tests with no optional-dependency skips.

### Environment and documentation drift

The active `v1ca1` environment that passes the tests is Python **3.10.20**, while [`pyproject.toml`](pyproject.toml#L10-L15), [`environment.yml`](environment.yml#L1-L17), and the README require or recommend 3.11. Neither version is currently enforced in CI because no CI configuration was found.

Scientific dependencies such as `replay_trajectory_classification`, `ripple_detection`, `position_tools`, and `track_linearization` are only lower-bounded or unpinned ([`pyproject.toml`](pyproject.toml#L17-L38)). Decoder recurrence, filter behavior, serialization, and numerical results can change across versions. The general `analysis` extra also requires CUDA-specific CuPy even for CPU workflows, while several direct imports are not declared explicitly.

The README's primary timestamp example passes `--ephys-format both` ([`README.md`](README.md#L60-L67)), but that flag has been removed from the parser ([`get_timestamps.py`](src/v1ca1/helper/get_timestamps.py#L325-L359)). The command fails before doing work.

**Recommendation.** Add CI for a locked Python 3.11 full-analysis environment and assert no unexpected skips. Add a smaller CPU profile, split GPU extras, declare direct dependencies, add lint/type/import-smoke checks, and generate a lock file or archived explicit environment. Test every README command.

### Large artifacts and avoidable eager loads

Representative L14/20240611 sizes are substantial:

- `timestamps_ephys_all.npz`: about **3.41 GB** for 426,845,408 samples;
- sixteen 1D prediction NetCDF files: about **52.69 GiB** total;
- cvPCA outputs: about **5.2 GiB**;
- task-progression dark/light GLM outputs: about **1.2 GiB**;
- ripple LFP cache: about **504 MiB**.

The full ephys clock is repeatedly materialized simply to map sorting frame indices to times ([`session.py`](src/v1ca1/helper/session.py#L663-L690)). Most `to_netcdf` calls omit an explicit engine, chunking, compression, and dtype policy, making storage backend-dependent and often uncompressed. Some figure code uses `xr.load_dataset` before selecting a short snippet.

**Recommendation.** Verify within-epoch regularity and store a compact piecewise clock `(epoch, first_frame, sample_count, start_time, rate, max residual)`; retain exact irregular clocks in lazy/chunked storage only when necessary. Pin `h5netcdf` or `netCDF4`, define versioned encodings with compression/chunking/dtypes, use `open_dataset` and slice before `.load()`, and consider Zarr for repeatedly accessed multi-GiB posterior arrays.

### Legacy boundary

Two top-level xcorr scripts still execute hard-coded L14 file I/O and create directories at import time ([`compute_xcorr.py`](src/v1ca1/xcorr/compute_xcorr.py#L13-L73), [`compute_autocorrelograms.py`](src/v1ca1/xcorr/compute_autocorrelograms.py#L13-L76)). They require removed pickle layouts and internal `kyutils`, and some validations instantiate `ValueError` without raising it. They are not covered by tests.

Move these scripts under an explicit `legacy` namespace or refactor them onto shared loaders and guarded CLIs. Production modules should be safe to import without touching data or creating outputs.

## Recommended implementation sequence

### Phase 1 — Protect result validity

1. F1 is resolved. Fix the remaining 1D spike-bin edge construction, then regenerate the new contiguous-CV classifier, posterior, and error outputs.
2. Make ripple event reuse configuration- and input-aware; prohibit mixed-provenance parquet files.
3. Introduce valid/unknown position support and rerun movement, immobility, and speed-gated outputs where coverage is imperfect.
4. Replace target-informed unit selection with fold-local or independent cohorts in dark/light and cross-trajectory transfer.
5. Repair the unequal-length ripple-block null and rerun its significance outputs.

Acceptance criteria should include focused regression tests for each issue and a before/after result-difference summary for every manuscript panel that consumes the affected artifact.

### Phase 2 — Make inference match the experiment

1. Define the experimental unit and estimand for every figure-level comparison.
2. Recompute population statistics using animal/session-clustered inference and leave-one-animal-out sensitivity.
3. Define multiple-testing families and save q-values.
4. Generate panel annotations from result artifacts; remove fixed significance constants.
5. Document dataset eligibility and example selection before final figure generation.

### Phase 3 — Make artifacts self-describing

1. Create a validated session manifest for epoch roles, meters-per-pixel, ripple/theta channels, dataset eligibility, and rationale.
2. Add schema versions, run IDs, input/config fingerprints, units, clock convention, and source identities to every canonical artifact.
3. Fail on unreadable canonical files; require an explicit legacy conversion path.
4. Make batch failure status structured and nonzero when nothing succeeds.

### Phase 4 — Lock and scale the workflow

1. Add a full Python 3.11 CI profile with zero unexpected skips, a CPU smoke profile, formatting/linting, and import safety checks.
2. Archive exact dependency versions/commits and numerical backend information.
3. Use explicit compressed/chunked array encodings and lazy reads.
4. Replace the multi-GiB exact timestamp vector with a validated compact clock representation where scientifically permissible.
5. Archive or modernize import-time legacy scripts.

## Suggested result-level sensitivity checks

Code fixes alone are not enough because saved outputs may already contain affected results. For manuscript evaluation, I would request these compact checks:

- **1D decoder:** per animal/epoch, compare decoding error and panel summaries before and after segment resets; count reset boundaries per fold.
- **Ripple events:** regenerate all epochs under one fingerprint and report event overlap, count, duration, and panel changes versus current tables.
- **Position coverage:** report valid/unknown fraction and maximum gap per epoch; recompute all speed-gated analyses after excluding unknown support.
- **Transfer:** compare current selected cohort, fold-local cohort, independent cohort, and all curated units.
- **Ripple GLM:** show nested-ridge results, animal-clustered population effects, FDR q-values, and nuisance-covariate/offset controls.
- **Figure statistics:** show one point per animal alongside cell-level distributions and leave-one-animal-out effects.
- **MI/tuning:** rerun with fixed seeds, more shuffles, occupancy-matched binning, and jointly supported bins.
- **Dimensionality:** compare current unit selection/z-scoring with independent/all-cell and temporally blocked repeat definitions.

## Questions for the analysis owners

1. Which figure panels are intended as animal-level inferential claims, and which are explicitly descriptive cell-population summaries?
2. Were the fixed Figure 2 significance stars computed elsewhere, and if so, where is the test definition and result artifact?
3. Is target-rate calibration in cross-trajectory encoding intentional, or should transfer be strictly source-only?
4. Were light/dark stability thresholds and example neurons prespecified before examining target outcomes?
5. Is the CA1-to-V1 ripple claim predictive association or directional communication?
6. Why is L16 excluded from the registered manuscript set, and are eligibility rules outcome-independent?
7. Which saved panels consume the full-epoch 1D posterior, ripple-decoding null, or parameter-resumed ripple tables and therefore require regeneration?

## Bottom line

The repository has a strong modern core and considerably better validation than many active lab codebases. The central task-progression representation is clear, the best nested-CV paths are careful, and the test suite is broad and currently green. The remaining work is concentrated rather than diffuse: repair a handful of concrete temporal/null/CV defects, make unknown data and artifact provenance explicit, and move figure inference from pooled neurons to the animal/session hierarchy. Doing those things—and recording result-level sensitivity rather than only code changes—would materially improve confidence in both the analyses and the eventual archive.
