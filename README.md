# Lee V1-CA1 Project

This repository contains Python analysis code for the V1-CA1 project: simultaneous recordings from primary visual cortex (V1) and hippocampal CA1 in freely moving rats performing a W-track spatial alternation task, along with sleep-box epochs collected before, between, and after run sessions.

The repository is currently best understood as a working analysis codebase rather than a polished Python package. Most of the code lives in `src/v1ca1` and is organized by analysis area. Some scripts expose command-line interfaces, while many others still encode experiment-specific defaults such as `animal_name`, `date`, and absolute data paths at the top of the file.

## What Is In This Repo

- Analysis scripts for spike sorting, decoding, ripple/theta detection, place-field and raster analyses, motor-variable analyses, cross-correlations, GLM/task-progression analyses, communication subspace analyses, and topology/signal-dimension work.
- A lightweight importable package namespace at `src/v1ca1`.
- No raw data, processed data, or figures tracked in git. Generated artifacts such as `*.pkl`, `*.npy`, `*.npz`, and figure files are ignored by the repo.

## Data Assumptions

The scripts in this repo operate on simultaneous electrophysiology and behavior data from the V1-CA1 project. In the typical experiment, four Livermore polymer probes are implanted in a single rat, targeting bilateral V1 and CA1 while the animal performs a spatial alternation task on a W-track.

The data include both run and sleep epochs:

- During run epochs, the animal performs spatial alternation on the W-track for milk reward.
- During sleep epochs, the animal is placed in a sleep box.
- Position is tracked with an overhead camera in both epoch types.
- Some run epochs include visual manipulations presented on specific W-track segments, including the left arm, right arm, and middle segment.
- At least one run epoch (typically `08_r4`) is performed in darkness (`<10^2 R*/s/rod`).

The code assumes access to Frank Lab-style data and analysis directories that are not included here. Across the current scripts, common hard-coded locations include:

- `/stelmo/nwb/raw`
- `/stelmo/kyu/analysis`
- `/nimbus/kyu`

Many scripts expect precomputed intermediate files such as:

- `timestamps_ephys.pkl`
- `timestamps_position.pkl`
- `timestamps_ephys_all.pkl`
- `position.pkl`
- `trajectory_times.pkl`
- region-specific sorting outputs such as `sorting_v1` and `sorting_ca1`

If you do not have that directory layout and those intermediate files available, most scripts will need either local path edits or small refactors before they can run.

## Repository Layout

All analysis code currently lives under `src/v1ca1`:

- `src/v1ca1/helper`: timestamp extraction and helper routines for sleep, immobility, and trajectory intervals.
- `src/v1ca1/spikesorting`: Mountainsort4-based spike sorting, consolidation, and figurl generation.
- `src/v1ca1/decoding`: conditional intensity estimation, likelihood computation, and 1D/2D position decoding.
- `src/v1ca1/raster`: place rasters, place fields, spike-triggered averages, and related plotting utilities.
- `src/v1ca1/motor`: decoding and visualization for speed, head direction, angular velocity, and related motor variables.
- `src/v1ca1/ripple`: ripple detection and ripple-centered GLM analyses.
- `src/v1ca1/oscillation`: theta phase extraction and oscillatory event detection.
- `src/v1ca1/xcorr`: auto/cross-correlogram analyses.
- `src/v1ca1/task_progression`: task progression encoding/decoding and several GLM variants.
- `src/v1ca1/communication_subspace`: cross-area latent subspace analyses.
- `src/v1ca1/signal_dim`: signal-dimensionality analyses.
- `src/v1ca1/topology`: topology-oriented analyses.

## Running Code From This Checkout

The repo now includes a minimal `pyproject.toml` and `environment.yml`, so the preferred setup is an editable install:

```bash
conda env create -f environment.yml
conda activate v1ca1
```

If you already have a Python environment, you can also install the package directly:

```bash
pip install -e .
```

You can still run scripts directly by pointing Python at `src`:

```bash
PYTHONPATH=src python src/v1ca1/<module>.py ...
```

Examples:

```bash
PYTHONPATH=src python src/v1ca1/spikesorting/sort.py \
  --animal-name L14 \
  --date 20240611 \
  --probe-idx 0 \
  --shank-idx 0
```

```bash
PYTHONPATH=src python src/v1ca1/decoding/fit_2d_decode_trajectory.py \
  --epoch_idx 0 \
  --use_half all \
  --movement True \
  --position_std 4.0 \
  --discrete_var switching \
  --place_bin_size 1.0 \
  --movement_var 4.0 \
  --overwrite False
```

```bash
PYTHONPATH=src python src/v1ca1/ripple/detect_ripples.py \
  --use_speed True \
  --overwrite False
```

Important: the first example is relatively configurable. The decoding and ripple examples still rely on hard-coded defaults inside the source file for items like `animal_name`, `date`, and analysis root paths.

## Dependencies

The codebase uses a mix of standard scientific Python packages and domain-specific neuroscience tooling. Imports currently present in the repo include:

- `numpy`, `scipy`, `pandas`, `matplotlib`, `xarray`, `scikit-learn`
- `spikeinterface`
- `pynwb`
- `pynapple`
- `position_tools`
- `track_linearization`
- `replay_trajectory_classification`
- `ripple_detection`
- `trajectory_analysis_tools`
- `non_local_detector`
- `kyutils`

`kyutils` appears throughout the repo and looks like a project- or lab-specific dependency, so a usable local environment will likely need both public packages and internal analysis utilities.

## Current State And Caveats

- This repo is script-heavy. Many files execute with experiment-specific globals rather than reusable config objects.
- Packaging is now minimal but intentionally light: editable installs work, but the project is not yet set up as a polished, fully reproducible distribution.
- There are no tests, notebooks, or environment files checked into the repo right now.
- Several modules mix reusable functions with one-off analysis code and plotting side effects.
- Absolute paths are common, so portability is limited until those are parameterized.

## Suggested Workflow

If you are picking this codebase up fresh, a practical order is:

1. Verify that your NWB files and analysis directories mirror the paths expected by the scripts you want to run.
2. Start with the helper and spikesorting scripts to generate or validate timestamps and sorting outputs.
3. Run downstream analyses from the relevant subpackage once the expected `pkl` intermediates exist.
4. Refactor high-value scripts to accept dataset identifiers and root paths via CLI arguments before scaling them up.

## Summary

This repo already contains a broad set of V1-CA1 analysis routines, but it is currently structured as an active lab analysis workspace rather than a turnkey package. The code is most usable if you already have the expected NWB files, intermediate pickles, and Frank Lab directory layout in place.
