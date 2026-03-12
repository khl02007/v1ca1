# Lee V1-CA1 Project

This repository contains Python analysis code for the Lee V1-CA1 project: simultaneous recordings from primary visual cortex (V1) and hippocampal CA1 in freely moving rats performing a W-track spatial alternation task, along with sleep-box epochs collected before, between, and after run sessions.

## At A Glance

- This is an active analysis workspace, not a polished end-user package.
- Most code lives under `src/v1ca1`, organized by analysis type.
- The repository covers spike sorting, helper preprocessing, decoding, raster/place-field analyses, motor analyses, ripple/theta analyses, cross-correlations, task-progression GLMs, and more specialized communication-subspace, topology, and signal-dimension work.
- Some scripts now expose command-line interfaces, but many still carry experiment-specific defaults such as `animal_name`, `date`, and absolute filesystem roots.

If you are new to the repo, the shortest accurate description is: this is a lab analysis codebase for working with V1-CA1 recordings once the local data layout and required intermediate files are already in place.

## How To Navigate The Repo

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

Good entry points if you want something runnable and relatively self-contained:

- `src/v1ca1/spikesorting/sort.py`
- `src/v1ca1/spikesorting/generate_figurl.py`
- `src/v1ca1/spikesorting/consolidate_sorting.py`
- `src/v1ca1/helper/get_timestamps.py`

The spike sorting curation flow is:

1. Run `sort.py` to produce sorter output.
2. Run `generate_figurl.py` to create a figurl URL for a probe/shank.
3. Open that URL in the browser and assign curation labels such as accept or reject.
4. Those labels are stored in `LorenFrankLab/sorting-curations` as `curation.json`.
5. Run `consolidate_sorting.py` to load that JSON locally and apply it to the SpikeInterface sorting object.

## What Is In This Repo

- Analysis scripts for spike sorting, decoding, ripple/theta detection, place-field and raster analyses, motor-variable analyses, cross-correlations, GLM/task-progression analyses, communication subspace analyses, and topology/signal-dimension work.
- A lightweight importable package namespace at `src/v1ca1`.
- Minimal packaging metadata via `pyproject.toml` and a starter environment via `environment.yml`.
- No raw data, processed data, or figures tracked in git. Generated artifacts such as `*.pkl`, `*.npy`, `*.npz`, and figure files are ignored by the repo.

## Quick Start

The preferred setup is:

```bash
conda env create -f environment.yml
conda activate v1ca1
```

Or, if you already have a compatible Python environment:

```bash
pip install -e .
```

You can run scripts directly from the checkout with:

```bash
PYTHONPATH=src python src/v1ca1/<module>.py ...
```

Examples:

```bash
PYTHONPATH=src python src/v1ca1/helper/get_timestamps.py \
  --animal-name L14 \
  --date 20240611
```

```bash
PYTHONPATH=src python src/v1ca1/spikesorting/sort.py \
  --animal-name L14 \
  --date 20240611 \
  --probe-idx 0 \
  --shank-idx 0
```

```bash
PYTHONPATH=src python src/v1ca1/spikesorting/generate_figurl.py \
  --animal-name L14 \
  --date 20240611 \
  --probe-idx 0 \
  --shank-idx 0
```

That command writes a text file containing a figurl URL. Opening the URL gives
you an interactive curation interface backed by a `curation.json` entry in the
`LorenFrankLab/sorting-curations` GitHub repository.

```bash
PYTHONPATH=src python src/v1ca1/spikesorting/consolidate_sorting.py \
  --animal-name L14 \
  --date 20240611
```

That command reads local `curation.json` files from a `sorting-curations`
checkout, saves curated per-shank sortings, and then writes aggregated
`sorting_v1` and `sorting_ca1` outputs.

Important: some scripts are now parameterized, but many downstream analyses still rely on hard-coded defaults inside the source file for items like `animal_name`, `date`, and root paths.

## Data And Experimental Context

The scripts in this repo operate on simultaneous electrophysiology and behavior data from the V1-CA1 project. In the typical experiment, four Livermore polymer probes are implanted in a single rat, targeting bilateral V1 and CA1 while the animal performs a spatial alternation task on a W-track.

The data include both run and sleep epochs:

- During run epochs, the animal performs spatial alternation on the W-track for milk reward.
- During sleep epochs, the animal is placed in a sleep box.
- Position is tracked with an overhead camera in both epoch types.
- Some run epochs include visual manipulations presented on specific W-track segments, including the left arm, right arm, and middle segment.
- At least one run epoch is typically performed in darkness (`<10^2 R*/s/rod`).

## Data Assumptions

The code assumes access to Frank Lab-style data and analysis directories that are not included here. Common filesystem roots referenced across the current scripts include:

- `/stelmo/nwb/raw`
- `/stelmo/kyu/analysis`
- `/nimbus/kyu`

Many scripts expect precomputed intermediate files such as:

- `timestamps_ephys.pkl`
- `timestamps_position.pkl`
- `timestamps_ephys_all.pkl`
- `position.pkl`
- `trajectory_times.pkl`
- sorting outputs and sorting analyzer folders under the analysis directory

If you do not have that directory layout and those intermediate files available, many scripts will need either local path edits or small refactors before they can run.

## Suggested Workflow

If you are picking this codebase up fresh, a practical order is:

1. Verify that your NWB files, extracted data, and analysis directories match the expectations of the scripts you want to run.
2. Start with the helper and spikesorting scripts to generate or validate timestamps and sorting outputs.
3. Run downstream analyses from the relevant subpackage once the expected intermediate `pkl` files and sorting outputs exist.
4. Parameterize high-value scripts further before trying to scale them across many sessions.

## Dependencies

The codebase uses a mix of standard scientific Python packages and domain-specific neuroscience tooling. Imports currently present in the repo include:

- `numpy`, `scipy`, `pandas`, `matplotlib`, `xarray`, `scikit-learn`, `seaborn`
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

`environment.yml` provides a good starting environment, but some workflows still require manual installation of additional packages depending on which analyses you run.

Figurl support is optional. If you want to run `generate_figurl.py`, install:

```bash
pip install -e ".[figurl]"
```

## Current State And Caveats

- This repo is still script-heavy, and many files mix reusable functions with one-off analysis code.
- Packaging is intentionally light: editable installs work, but the project is not yet a fully reproducible software distribution.
- There are still few guardrails around path assumptions and intermediate-file dependencies.
- Portability remains limited until more scripts are parameterized and absolute paths are reduced.

## Summary

This repo already contains a broad set of V1-CA1 analysis routines, but it is best understood as an active lab analysis workspace. It is most useful to people who already have the expected NWB files, extracted data, intermediate pickles, and analysis directory layout available locally.
