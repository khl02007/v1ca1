# AGENTS.md

This file is for LLM agents and coding assistants working in this repository.

## Repo Summary

- This is a script-heavy neuroscience analysis repository for the Lee V1-CA1 project.
- Most code lives under `src/v1ca1`.
- The project is installable via `pyproject.toml`, but many modules still behave like lab scripts rather than polished package APIs.
- The code assumes access to local NWB files, extracted Trodes data, and previously generated intermediate artifacts that are not stored in git.

## Important Path Assumptions

- Current script conventions in the refactored files use:
  - analysis root: `/stelmo/kyu/analysis`
  - NWB root: `/stelmo/nwb/raw`
  - extracted data root: `/nimbus/kyu`
- The current analysis path convention is:
  - `analysis_root / animal_name / date`
- Do not reintroduce `singleday_sort` into analysis paths unless the user explicitly asks for it.

## Current Script Style

When editing scripts, prefer the lighter style now used in:

- `src/v1ca1/spikesorting/sort.py`
- `src/v1ca1/spikesorting/generate_figurl.py`
- `src/v1ca1/spikesorting/consolidate_sorting.py`
- `src/v1ca1/helper/get_timestamps.py`

That style means:

- Prefer plain helper functions over small dataclasses when the dataclass only wraps path composition.
- Keep path-building logic explicit with small functions like `get_analysis_path(...)` or `get_sorting_path(...)`.
- Put script-specific configuration close to `main()` when that makes the workflow easier to read.
- Use CLI flags for dataset identifiers and root paths when practical, especially `--animal-name` and `--date`.
- Add short docstrings to top-level helper functions.

## Working Assumptions

- Many scripts still contain hard-coded `animal_name`, `date`, and absolute paths.
- A change in one workflow often needs matching path/config updates in downstream scripts.
- Intermediate files commonly expected by downstream analyses include:
  - `timestamps_ephys.pkl`
  - `timestamps_position.pkl`
  - `timestamps_ephys_all.pkl`
  - `position.pkl`
  - `trajectory_times.pkl`
- Spike sorting workflows also expect sorter outputs and sorting analyzer folders under the analysis directory.
- Manual curation currently follows this flow:
  - `generate_figurl.py` writes a figurl URL tied to a remote `curation.json`
  - a human assigns labels in the browser
  - `consolidate_sorting.py` loads the local `sorting-curations` checkout and applies those labels with SpikeInterface
- Probe-to-region assignment is not universal. `consolidate_sorting.py` defaults to one known layout, but callers should override `--v1-probes` and `--ca1-probes` when a session uses a different implant mapping.
- Refactored spikesorting scripts now write one JSON run record per execution under `analysis_root / animal_name / date / v1ca1_log/`.

## Safe Editing Guidance

- Preserve the user’s current path conventions and repo-specific choices; do not “restore” older layouts from historical code unless asked.
- Assume the repo may contain user edits in progress. Check `git diff` before “cleanups” that touch the same files.
- When modernizing a script, avoid broad architectural rewrites unless the user asks for them.
- For quick verification, `python -m py_compile <file>` is usually the least disruptive check.

## Navigation Guide

- `src/v1ca1/helper`: timestamps and epoch helper scripts.
- `src/v1ca1/spikesorting`: sorting, analyzer generation, and figurl export.
- `src/v1ca1/decoding`: trajectory and sleep-box decoding models.
- `src/v1ca1/raster`: place rasters, place fields, and STA-related plots.
- `src/v1ca1/motor`: motor-variable decoding and tuning analyses.
- `src/v1ca1/ripple`: ripple detection and ripple GLM analyses.
- `src/v1ca1/oscillation`: theta phase and oscillatory event work.
- `src/v1ca1/xcorr`: auto- and cross-correlation analyses.
- `src/v1ca1/task_progression`: GLM, tuning, and task progression analyses.
- `src/v1ca1/communication_subspace`, `signal_dim`, `topology`: more specialized downstream analyses.

## Environment Notes

- `environment.yml` gives a useful base environment, but several workflows also depend on additional packages not guaranteed to be installed.
- Figurl generation uses the optional `.[figurl]` extra rather than the base install.
- `kyutils` is used throughout the repo and appears to be lab-specific.
- Some subpackages depend on heavier neuroscience tooling such as `spikeinterface`, `pynwb`, `pynapple`, `position_tools`, `track_linearization`, and replay/ripple-related libraries.
