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
tables["task_progression_stability_parameters"].insert_default()
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

The implemented table flows are:

```text
Ripples + EpochIntervals + RippleModulationParameters
    + SortedSpikesGroup / UnitSelectionParams
    -> RippleModulationSelection
    -> RippleModulation

Position + MovementParameters
    + SortedSpikesGroup / UnitSelectionParams
    -> MovementFiringRateSelection
    -> MovementFiringRate

TrajectoryIntervals + WTrackGraph + MovementFiringRate
    + TaskProgressionStabilityParameters
    -> TaskProgressionStabilitySelection
    -> TaskProgressionStability
```

The computed tables are named `RippleModulation`, `MovementFiringRate`, and
`TaskProgressionStability`, without a `Computed` suffix. Each explicit
selection freezes its upstream membership, filters, and parameter values. That
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
<animal>/<date>/task_progression_stability/<epoch>/<trajectory>/<region>/<uuid>/
    stability.parquet
```

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
undefined-rate QC row per selected unit. `TaskProgressionStability.make()` loads
these saved artifacts through its upstream `MovementFiringRate` row and does
not recompute speed, movement support, or firing rates.

Calling `make()` computes from NWB and the selected sorting group. Calling
`register_existing()` on `RippleModulation` or `TaskProgressionStability`
validates and partitions a compatible legacy artifact, copies it into the same
canonical output layout, and inserts the result without rerunning the analysis.
`MovementFiringRate` is compute-only and writes its Parquet/NPZ bundle
atomically. Legacy registration is limited to matching imported sorting
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
