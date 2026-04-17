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
