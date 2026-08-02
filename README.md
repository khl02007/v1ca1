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
package is passive: it does not connect to DataJoint, activate schemas, insert
rows, or populate computations. In a separately configured Spyglass process,
the intended order is:

```python
import datajoint as dj

from v1ca1.spyglass import activate, ingest_v1ca1_nwb

custom = dict(dj.config.get("custom", {}))
custom["database.prefix"] = "kyuv1ca1"
dj.config["custom"] = custom
tables = activate()  # Explicit schema activation/DDL.
tables["ripple_modulation_parameters"].insert_default()
ingest_v1ca1_nwb("L1420240611_augmented.nwb", tables=tables)
```

The custom prefix must be set before activation so Spyglass associates the
project AnalysisNwbfile table with the `kyuv1ca1_nwbfile` schema. Activation
does not modify Spyglass's analysis-table registry. If that table is later used
for NWB-natural results, register it once, explicitly, with
`tables["analysis_nwbfile"]().register_with_spyglass()`.

Standard Spyglass ingestion, including `Session`, `Nwbfile`, and
`ImportedSpikeSorting`, must already be complete. The custom ingestion indexes
NWB object pointers and small metadata only. Arrays remain in NWB and are read
on demand:

```python
ripples = tables["ripples"].load_intervals(ripple_key)
head_position = tables["position"].load_position(position_key)
graph_inputs = tables["wtrack_graph"].load_graph(graph_key)
track_graph = make_track_graph(**graph_inputs["track_graph_kwargs"])
linearized = get_linearized_position(
    head_position.values,
    track_graph,
    **graph_inputs["linearization_kwargs"],
)
```

The initial computed pipeline is `RippleModulationComputed`. Its selection is
downstream of `Ripples`, `EpochIntervals`, scalar parameters, and the standard
Spyglass `SortedSpikesGroup`; the current implementation deliberately requires
the `all_units` filter. Results are keyed Parquets under
`/stelmo/nwb/analysis/kyu/v1ca1`, while a project-owned `AnalysisNwbfile` table
is available for future NWB-natural outputs. Existing legacy Parquets can be
registered with `RippleModulationComputed.register_existing()`; source paths,
SHA-256 hashes, optional source commits, and runtime commits are retained.

Use a new `v1ca1-spyglass` environment for this pipeline. The old local
`spyglass` environment (Python 3.9/PyNWB 2.3) cannot read the augmented files.
The new environment needs the pinned local Spyglass checkout, PyNWB 3.1.3 or
newer, Pynapple, and PyArrow. Do not install the full `v1ca1[analysis]` extra
there because its SpikeInterface requirement conflicts with the pinned
Spyglass checkout; install this package without dependencies and add only the
pipeline runtime dependencies.
