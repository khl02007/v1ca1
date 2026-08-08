# Lee V1-CA1 Project

Python analysis code for the Lee V1-CA1 project: simultaneous V1 and CA1 recordings in freely moving rats during W-track behavior and sleep-box epochs.

This is an active lab analysis repo, not a polished end-user package. Most workflows expect local NWB files plus session-specific intermediate artifacts that are not tracked in git.

## Repo Layout

All main code lives under `src/v1ca1`.

- `helper`: shared session loading, timestamp extraction, interval builders, W-track utilities
- `position`: DLC cleaning, geometry fitting, IMU fusion, legacy-position conversion
- `spikesorting`: Mountainsort4 sorting, figurl generation, curated sorting consolidation
- `ripple`: ripple detection, GLMs, decoding, modulation plots
- `task_progression`: tuning, mutual information, encoding/decoding comparisons, GLMs
- `oscillation`, `sleep`, `behavior`, `motor`, `raster`, `xcorr`, `nwb`: analysis-specific scripts and plots
- `communication_subspace`, `signal_dim`, `topology`: more specialized analyses

Several subpackages still contain `legacy` scripts alongside newer CLI-oriented workflows.

## Setup

Use Python 3.11. The active package metadata and conda environment both assume
Python 3.11; older environments may import stale editable installs but fail on a
fresh reinstall.

Preferred:

```bash
conda env create -f environment.yml
conda activate v1ca1
```

Or install into an existing environment:

```bash
pip install -e ".[analysis,test]"
```

Optional extras:

```bash
pip install -e ".[glm]"
pip install -e ".[figurl]"
```

- `analysis`: main scientific Python and neuroscience stack
- `test`: pytest dependencies
- `glm`: JAX-based dependencies used by GLM-heavy workflows
- `figurl`: figurl export support

## Running Scripts

Scripts are typically run directly from the checkout:

```bash
PYTHONPATH=src python src/v1ca1/<module>.py ...
```

Examples:

```bash
PYTHONPATH=src python src/v1ca1/helper/get_timestamps.py \
  --animal-name L14 \
  --date 20240611 \
  --ephys-format both
```

```bash
PYTHONPATH=src python src/v1ca1/ripple/detect_ripples.py \
  --animal-name L14 \
  --date 20240611
```

```bash
PYTHONPATH=src python src/v1ca1/task_progression/motor.py \
  --animal-name L14 \
  --date 20240611 \
  --regions v1 ca1
```

Good current entry points include:

- `src/v1ca1/helper/get_timestamps.py`
- `src/v1ca1/helper/get_trajectory_times.py`
- `src/v1ca1/helper/get_immobility_times.py`
- `src/v1ca1/spikesorting/sort.py`
- `src/v1ca1/spikesorting/generate_figurl.py`
- `src/v1ca1/spikesorting/consolidate_sorting.py`
- `src/v1ca1/ripple/detect_ripples.py`
- `src/v1ca1/ripple/ripple_glm.py`
- `src/v1ca1/task_progression/tuning_analysis.py`
- `src/v1ca1/task_progression/mutual_info.py`
- `src/v1ca1/task_progression/motor.py`

Many other scripts still rely on file-local defaults or experiment-specific assumptions.

## Data Assumptions

The current helpers and CLIs mostly assume:

- NWB root: `/stelmo/nwb/raw`
- analysis root: `/stelmo/kyu/analysis`
- session layout: `analysis_root / animal_name / date`

Common required intermediates include timestamp files, position outputs, trajectory tables, run or sleep intervals, sorting outputs, and other session-level artifacts produced by earlier steps.

If that local data layout is missing, many scripts will need small path or workflow adjustments before they can run.

## Outputs

Recent workflows tend to write:

- parquet tables for summaries
- pynapple-backed `.npz` files for timestamps and interval-style artifacts
- NetCDF-backed `xarray.Dataset` outputs for larger model-fit results
- compatibility pickles where downstream code still expects them
- JSON run logs under `analysis_root / animal_name / date / v1ca1_log`

## In Practice

A typical workflow is:

1. Generate or validate session timestamps and intervals with the helper scripts.
2. Run sorting, ripple detection, or position preprocessing as needed.
3. Run downstream analyses from the relevant subpackage once the expected session artifacts exist.

The repo is most useful if you already have the local Frank Lab-style data layout and intermediate files in place.

## Project Spyglass Pipeline

`v1ca1.spyglass` contains the project-owned `kyuv1ca1` tables. Importing the
package is passive: it does not import DataJoint or Spyglass, connect to a
database, activate schemas, insert rows, or modify NWB files. Run it from a
separately configured Spyglass process. The intended order is:

```python
import datajoint as dj

from v1ca1.spyglass import activate, ingest_v1ca1_nwb

custom = dict(dj.config.get("custom", {}))
custom["database.prefix"] = "kyuv1ca1"
dj.config["custom"] = custom
tables = activate()  # Explicit schema/table declaration; no data insertion.
tables["ripple_modulation_parameters"].insert_default()
tables["movement_parameters"].insert_default()
tables["tuning_curve_parameters"].insert_presets()
tables["tuning_similarity_parameters"].insert_presets()
tables["dpp_encoding_comparison_parameters"].insert_presets()
tables["path_progression_decoding_parameters"].insert_presets()
tables["motor_encoding_comparison_parameters"].insert_presets()
ingest_v1ca1_nwb("L1420240611_augmented.nwb", tables=tables)
```

The custom prefix must be set before activation so Spyglass associates the
project `AnalysisNwbfile` table with the `kyuv1ca1_nwbfile` schema.
`activate()` may create the schemas and tables when requested, but it does not
insert data or parameters, populate computations, register artifacts, alter
Spyglass's analysis-table registry, or write NWB files. If `AnalysisNwbfile` is
later used for an NWB-native result, register it once, explicitly, with
`tables["analysis_nwbfile"]().register_with_spyglass()`.

Standard Spyglass ingestion, including `Session`, `Nwbfile`, and
the relevant spike-sorting tables, must already be complete. The explicit
custom ingestion indexes NWB object pointers and small metadata only; pass
`dry_run=True` to validate without inserting. Arrays remain in the registered
NWB and are read on demand:

```python
ripples = tables["ripples"].load_intervals(ripple_key)
position = tables["position"].load_position(position_key)
graph_inputs = tables["wtrack_graph"].load_graph(graph_key)
track_graph = make_track_graph(**graph_inputs["track_graph_kwargs"])
linearized = get_linearized_position(
    position.values,
    track_graph,
    **graph_inputs["linearization_kwargs"],
)
```

`Position` selections use the actual NWB `position_series_name`, with a
descriptive `position_role`; current files expose `head_position` and
`body_position`. Loading applies the NWB-recorded leading-sample analysis
offset by default without truncating or rewriting the source series.
`TrajectoryIntervals`, `Ripples`, and `Position` are children of
`EpochIntervals`, so every indexed trajectory, ripple set, and position series
has an audited epoch parent. A provenance-selected ripple epoch is cataloged
even when it contains no events, with `ripple_count=0`.

The analyses share an adapter for standard `SortedSpikesGroup`,
`UnitSelectionParams`, and `SpikeSortingOutput`. It supports imported or
curated sorting parents, applies include/exclude label filters and region
selection, and combines canonical ephys-referenced spike times. Persistent
unit identity is `(spikesorting_merge_id, unit_id)`; the Pynapple group keys
used during computation are temporary. Custom tables do not store one row per
unit.

`RegionSortedSpikesGroup` is an explicitly registered logical view of a
standard sorting group for one normalized region. Its UUIDv5 freezes source
membership, unit-label filters, region, and the selected-unit identity digest;
the row stores only hashes, snapshots, and a unit count. It creates neither
per-unit database rows nor artifact files. `register_regions()` performs the
insertion, while `load_spikes()` reloads and verifies the standard sources.
Importing `v1ca1.spyglass` or calling `activate()` does neither operation.

The implemented table flows are:

```text
Ripples + EpochIntervals + RippleModulationParameters
    + SortedSpikesGroup / UnitSelectionParams
    -> RippleModulationSelection
    -> RippleModulation

Ripples + EpochIntervals + CA1/V1 RegionSortedSpikesGroup rows
    + RippleGLMParameters
    -> RippleGLMSelection
    -> RippleGLM

Position + MovementParameters
    + SortedSpikesGroup / UnitSelectionParams
    -> MovementFiringRateSelection
    -> MovementFiringRate

TrajectoryIntervals + WTrackGraph + MovementFiringRate
    + TuningCurveParameters + trial_subset
    -> PathSpecificPlaceTuningCurveSelection
    -> PathSpecificPlaceTuningCurve

two same-turn TrajectoryIntervals + their WTrackGraph rows
    + MovementFiringRate + TuningCurveParameters + trial_subset
    -> DPPTuningCurveSelection
    -> DPPTuningCurve

odd PathSpecificPlaceTuningCurve + even PathSpecificPlaceTuningCurve
    -> PathSpecificPlaceStabilitySelection
    -> PathSpecificPlaceStability

four matching all-trial PathSpecificPlaceTuningCurve rows
    + TuningSimilarityParameters
    -> PathSpecificPlaceTuningSimilaritySelection
    -> PathSpecificPlaceTuningSimilarity

RegionSortedSpikesGroup + MovementFiringRate
    + four TrajectoryIntervals + four trajectory WTrackGraph rows
    + full_w WTrackGraph
    + four PathSpecificPlaceStability rows backed by
      legacy_4cm_unsmoothed odd/even curves
    + DPPEncodingComparisonParameters
    -> DPPEncodingComparisonSelection
    -> DPPEncodingComparison

RegionSortedSpikesGroup
    + target/cohort MovementFiringRate and eight matching stability rows
    + target four TrajectoryIntervals and four WTrackGraph rows
    + PathProgressionDecodingParameters
    -> PathProgressionDecodingComparisonSelection
    -> PathProgressionDecodingComparison

RegionSortedSpikesGroup + MovementFiringRate
    + four TrajectoryIntervals + four WTrackGraph rows
    + PathSpecificPlaceDecodingParameters
    -> PathSpecificPlaceDecodingSelection
    -> PathSpecificPlaceDecoding

RegionSortedSpikesGroup + MovementFiringRate
    + primary Position + orientation-reference Position
    + four TrajectoryIntervals + four trajectory WTrackGraph rows
    + full_w WTrackGraph + MotorEncodingComparisonParameters
    -> MotorEncodingComparisonSelection
    -> MotorEncodingComparison

RegionSortedSpikesGroup
    + dark/light MovementFiringRate rows
    + eight epoch-specific TrajectoryIntervals
    + four shared WTrackGraph rows + DarkLightGLMParameters
    -> DarkLightGLMSelection
    -> DarkLightGLM

DarkLightGLM + RegionSortedSpikesGroup
    + held-out light MovementFiringRate
    + four held-out light TrajectoryIntervals
    + four shared WTrackGraph rows + SwapGLMParameters
    -> SwapGLMSelection
    -> SwapGLM

RegionSortedSpikesGroup
    + dark/train-light/test-light MovementFiringRate rows
    + twelve all-trial, unsmoothed 4-cm PathSpecificPlaceTuningCurve rows
    + three EpochIntervals + SwapTuningCurveComparisonParameters
    -> SwapTuningCurveComparisonSelection
    -> SwapTuningCurveComparison
```

`SwapGLM` reuses one exact selected `DarkLightGLM` fit and scores it without
refitting on a distinct held-out light run. The selection freezes the upstream
model-file checksums, the same regional sorting and movement definition, all
four held-out lap sources, and the shared centimeter graphs. Its unit audit
keeps the upstream selected order and marks a score valid only when the
upstream fit and all expected model-by-trajectory scores are valid.

`SwapTuningCurveComparison` is the empirical counterpart to `SwapGLM`. One
result row covers one dark-training run, one light-training run, one distinct
light-test run, one region, and one parameter preset; unit, trajectory, and
model results remain inside its artifacts. It builds six fixed empirical
predictions from full-trajectory training curves and scores only the configured
swapped segment of each held-out trajectory. The V1 and CA1 manuscript presets
both use 50-ms evaluation bins and one-bin Gaussian smoothing. Their strict
dark/light training-rate thresholds are respectively `> 0.5`/`> 0.5` Hz and
`> 0.0`/`> 0.0` Hz; held-out firing rate is not a filter.
The standalone `task_progression.swap_tuning_curve_comparison` default remains
20 ms, so the manuscript-compatible 50-ms choice is explicit in these presets.

`RippleGLM` fits the fixed CA1-to-V1 ripple population model once per epoch.
Its selection may use different standard sorted-spikes groups for CA1 and V1,
but both groups and the ripple row must belong to the same NWB file. It freezes
the raw ripple start/end digest, detector/NWB-object provenance, the exact
single-ripple windows retained after epoch clipping, both regional unit
snapshots, and the parameter/output hashes. The two manuscript presets use a
0.2-s zero-offset window, five folds, 100 shuffles, ridge 0.1, seed 45, and
separate `unit_vector` and `mean_activity` CA1 predictors; both require
speed-gated detector-threshold-2.0 events.

`DPPEncodingComparison`'s manuscript preset is five lap-wise folds, 50-ms
evaluation bins, 4-cm spatial bins, one-bin Gaussian smoothing, random seed 47,
movement firing rate
at least 0.5 Hz, and stability at least 0.5 on at least one trajectory. The
standalone `task_progression.encoding_comparison` default remains 20 ms; the
Spyglass preset makes the manuscript-specific 50-ms choice explicit.
Each trajectory receives a deterministic independent lap shuffle. Bins and
smoothing are restarted at every concatenated path or DPP block boundary.

The computed tables are named `RippleModulation`, `MovementFiringRate`,
`PathSpecificPlaceTuningCurve`, `PathSpecificPlaceTuningSimilarity`,
`DPPTuningCurve`, `PathSpecificPlaceStability`,
`DPPEncodingComparison`, `PathProgressionDecodingComparison`, and
`PathSpecificPlaceDecoding`, `MotorEncodingComparison`, `DarkLightGLM`, and
`SwapGLM`, `SwapTuningCurveComparison`, and `RippleGLM`, without a `Computed`
suffix. Each
explicit selection freezes its upstream membership, filters, and parameter values. That
snapshot determines a table-specific UUIDv5. Computation rejects later edits
to those Manual parameter values and requires a new selection. Results use
session-first, UUID-keyed paths rooted at `/stelmo/nwb/analysis/kyu/v1ca1`:

```text
<animal>/<date>/ripple_modulation/<epoch>/<region>/<uuid>/
    summary.parquet
    peri_ripple_firing_rate.parquet
<animal>/<date>/movement_firing_rate/<epoch>/<region>/<uuid>/
    movement_firing_rate.parquet
    movement_intervals.npz
<animal>/<date>/path_specific_place_tuning_curve/<epoch>/<trajectory>/<subset>/<region>/<uuid>/
    tuning_curve.nc
<animal>/<date>/dpp_tuning_curve/<epoch>/<turn>/<subset>/<region>/<uuid>/
    tuning_curve.nc
<animal>/<date>/path_specific_place_tuning_similarity/<epoch>/<region>/<metric>/<uuid>/
    similarity.parquet
<animal>/<date>/path_specific_place_stability/<epoch>/<trajectory>/<region>/<uuid>/
    stability.parquet
<animal>/<date>/dpp_encoding_comparison/<epoch>/<region>/<uuid>/
    encoding_comparison.parquet
<animal>/<date>/path_progression_decoding_comparison/<epoch>/<region>/<uuid>/
    manifest.parquet
    unit_eligibility.parquet
    decoding_summary.parquet
    cross_path_error_by_position.parquet
    cross_<family>_<source>_to_<target>_{true,decoded}.npz
<animal>/<date>/path_specific_place_decoding/<epoch>/<region>/<uuid>/
    manifest.parquet
    selected_units.parquet
    fold_qc.parquet
    decoding_summary.parquet
    decoding_error_by_position.parquet
    {true,decoded}_place.npz
<animal>/<date>/motor_encoding_comparison/<epoch>/<region>/<uuid>/
    manifest.parquet
    selected_units.parquet
    nested_cv.nc
    full_refit.nc
<animal>/<date>/dark_light_glm/<light>_vs_<dark>/<region>/<uuid>/
    manifest.parquet
    selected_units.parquet
    selection_summary.nc
    candidates/*.nc
    selected/{model}.nc
<animal>/<date>/swap_glm/<light-train>_train_to_<light-test>_test/dark_<dark>/<region>/<uuid>/
    manifest.parquet
    selected_units.parquet
    swap_glm.nc
<animal>/<date>/swap_tuning_curve_comparison/<light-train>_train_to_<light-test>_test/dark_<dark>/<region>/<uuid>/
    manifest.parquet
    selected_units.parquet
    summary.parquet
    swap_tuning.nc
<animal>/<date>/ripple_glm/<epoch>/<uuid>/
    manifest.parquet
    selected_units.parquet
    summary.parquet
    ripple_glm.nc
```

Any explicit `artifact_root` must remain within the stage configured for the
DataJoint `analysis` store so its paths are valid for `filepath@analysis`.

The canonical `Ripples` source contains speed-gated events detected at the
2.0 z-score threshold. The default `RippleModulationParameters` validates that
upstream provenance and uses every event in the selected source row; it does
not apply a downstream ripple-mean-z-score threshold. Selected zero-event
epochs remain explicit `no_ripples` results. Stability retains all selected
units and explicit QC for undefined correlations. Firing-rate and stability
thresholds are downstream scientific selections, not hidden producer filters.

`MovementFiringRate` saves one Parquet row per selected unit plus the exact
Pynapple movement `IntervalSet` in ephys-referenced seconds. There is no
firing-rate prefilter: a zero-spike unit has a valid 0 Hz rate whenever valid
movement support exists. Sleep epochs are allowed when the selected Position
series exists. Its result statuses are `valid`, `no_units`,
`no_valid_position`, and `no_movement`; the latter two retain one
undefined-rate QC row per selected unit.
`DPPTuningCurve` uses fixed same-turn pairs: `center_to_left` plus
`right_to_center` for left turns and `center_to_right` plus `left_to_center`
for right turns. It selects odd/even trials independently within each source
trajectory and pools normalized, direction-aligned samples and movement
support before estimating one curve; it does not average two separately
estimated curves. Both source interval and graph rows remain explicit in the
selection, and the paired graph paths must have one common physical length for
the shared centimeter-bin interpretation.
`PathSpecificPlaceStability.make()` loads the matching persisted odd/even
tuning curves and their shared upstream `MovementFiringRate` row; it does not
recompute position linearization, speed, movement support, firing rates, or
tuning curves.
`PathSpecificPlaceTuningSimilarity` has one result row per session, epoch,
region, tuning configuration, metric, and matching set of four all-trial path
curves. Its metrics are `correlation`, `absolute_overlap`, and
`shape_overlap`; the Parquet contains four unit/path-pair rows per unit. It
copies upstream `movement_firing_rate_hz` without filtering. DPPI,
significance, and cross-epoch aggregation remain downstream figure or analysis
logic.

`DPPEncodingComparison` refits training-fold tuning curves for
four fixed Poisson encoding models: path-specific physical place,
direction-independent absolute place on the full W-track, same-turn DPP, and
start-to-goal distance-to-reward. Its eligible-unit Parquet stores persistent
identity, movement rate and four stability values, total held-out log
likelihoods in nats, null-relative information in bits per spike,
DPP-minus-alternative contrasts, and explicit model/unit QC. Units remain rows
in the artifact, not rows in DataJoint.

`PathProgressionDecodingComparison` computes the fixed 16 directed
cross-path Bayesian transfers on normalized start-to-goal path progression.
Each result is one target epoch and region, while an explicit cohort epoch
defines a symmetric target/cohort intersection of eligible persistent units.
The default preset uses 20-ms decode bins, a four-bin sliding window, 4-cm
path bins, movement firing rate at least 0.5 Hz in both epochs, and no
stability threshold. Supplying a stability threshold additionally requires at
least one passing trajectory in each epoch. This auditable shared-population
policy intentionally differs from the legacy manuscript workflow, which
selected units from the dark epoch alone. Path-specific-place decoding is not
bundled because the legacy workflow used a different unit population for it;
that all-unit within-epoch decoder is represented by the separate
`PathSpecificPlaceDecoding` table.
The selected stability rows remain explicit dependencies when filtering is
disabled so their values can be retained for audit without affecting the
cohort. Persistent sorting-output/unit identities—not ephemeral Pynapple
group keys—link units across epochs and reloads. Eligibility, transfer, and
fixed output-summary rules are independently hashed into the selection.
The Pynapple NPZ timestamps are seconds on the augmented NWB's ephys timestamp
reference; true and decoded time grids are allowed to differ.

Calling `make()` computes from its selected upstream rows and artifacts; the
source-loading stages ultimately derive their data from NWB and the selected
sorting group. Calling
`register_existing()` on `RippleModulation`, `PathSpecificPlaceTuningCurve`,
`PathSpecificPlaceTuningSimilarity`, `DPPTuningCurve`,
`PathSpecificPlaceStability`, `DPPEncodingComparison`,
`PathSpecificPlaceDecoding`, `MotorEncodingComparison`, `DarkLightGLM`, or
`SwapGLM`, `SwapTuningCurveComparison`, or `RippleGLM` validates a
compatible legacy artifact, copies
the selected content into the canonical output layout, and inserts the result
without rerunning the analysis. Legacy tuning-curve
registration is limited to the all-trial, 4 cm unsmoothed preset; odd/even
curves are computed from NWB. Registration additionally requires the legacy
cleaned-DLC `head_position`, its 10-sample offset, and the 4.0 cm/s,
0.1 s-sigma movement defaults; selections with other inputs are recomputed.
Similarity registration accepts a complete `*_all_units.parquet` source only
for the legacy 4 cm unsmoothed, all-trial tuning configuration and a matching
imported sorting selection.
`DPPEncodingComparison.register_existing()` validates one legacy
epoch summary against its regional group, movement rates, four stability
inputs, parameters, and exact eligible-unit set, converts per-spike legacy
likelihoods to total nats, and copies the canonical Parquet into the path above.
The legacy filename must verify the fold count and temporal/spatial bins;
smoothing, random seed, and fold-level QC are not encoded, so registration
records those limitations explicitly rather than claiming they were
reconstructed from the file.
`SwapGLM.register_existing()` additionally requires the exact selected
DarkLightGLM artifact frozen by the selection. It normalizes the complete
legacy schema, verifies the historical position offset and movement-speed
threshold, and re-scores the selected NWB inputs without refitting. Every
available scientific value must match before the recomputed canonical result
is written with persistent group-unit IDs.
`SwapTuningCurveComparison.register_existing()` is limited to the available
V1 legacy artifacts and matching imported sorting. It requires the historical
10-sample position offset and 4.0 cm/s movement threshold, reconstructs and
re-scores the selected NWB inputs, and compares the complete scientific
content rather than trusting the legacy file's coordinates or summary alone.
It does not register legacy cross-epoch joins or cross-trajectory-transfer
outputs.
`RippleGLM.register_existing()` accepts one legacy NetCDF only when both
regional views resolve uniquely to `ImportedSpikeSorting` unit IDs. It
reconstructs the selected ripple windows, source/target counts, folds,
metrics, and coefficients from the selected NWB-backed inputs before writing
the canonical UUID bundle.
`MovementFiringRate` is compute-only and writes its Parquet/NPZ bundle
atomically. `PathProgressionDecodingComparison` is also compute-only: legacy
decoding NPZs omit selected-unit identities, sorting and graph snapshots,
parameters, and output hashes, so they cannot support scientifically exact
`register_existing()` validation. Legacy registration elsewhere is limited to matching imported sorting
outputs. It requires the complete canonical Parquet schemas and, for
peri-ripple firing rates, one complete common time grid per unit. Canonical
empty artifacts are retained as `no_units` or `no_ripples` terminal results.
UUID-keyed result rows and destinations are immutable: registration rejects
`overwrite=True`, checks for an existing result before invoking its artifact
hook, and directly inserts into the computed table only after that preflight.
With `skip_duplicates=True`, an existing row is returned without touching its
files. Tables with both routes retain artifact origin, a selected-unit digest,
and actual runtime V1–CA1 and Spyglass commits; registration also retains source
paths, hashes, and optional source commits. `MovementFiringRate` records its
selected-unit digest and runtime commits directly. Current file-backed results
use `filepath@analysis`; the project `AnalysisNwbfile` remains available for
future NWB-natural results.

Use a new `v1ca1-spyglass` environment for this pipeline. The old local
`spyglass` environment (Python 3.9/PyNWB 2.3) cannot read the augmented files.
The new environment needs the local Spyglass checkout, PyNWB 3.1.3 or newer,
Pynapple, PyArrow, position-tools, and track-linearization. Do not install the
full `v1ca1[analysis]` extra there because its SpikeInterface requirement
conflicts with the Spyglass dependency set; install this package without
dependencies and add only the pipeline runtime dependencies. The documented
Spyglass target is commit
`d5fa7fe1d07c5a349a6d5e0f15d821e5cfe08d38`, but runtime code records the
actual commit as provenance rather than rejecting a different checkout.

See [the package-level pipeline guide](src/v1ca1/spyglass/README.md) for table
semantics, exact artifact names, selection insertion, and provenance details.
