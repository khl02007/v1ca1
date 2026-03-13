# AGENTS.md

Concise guidance for LLM agents working in this repository.

## Core Context

- Script-heavy neuroscience analysis repo for the Lee V1-CA1 project; most code lives in `src/v1ca1`.
- `pyproject.toml` makes it installable, but many modules still behave like lab scripts.
- Workflows assume local NWB data, extracted Trodes data, and intermediate artifacts that are not in git.

## Paths

- Default roots:
  - analysis: `/stelmo/kyu/analysis`
  - NWB: `/stelmo/nwb/raw`
  - extracted data: `/nimbus/kyu`
- Analysis path convention: `analysis_root / animal_name / date`
- Do not reintroduce `singleday_sort` unless the user explicitly asks.

## Editing Style

- Match the style in `src/v1ca1/spikesorting/sort.py`, `src/v1ca1/spikesorting/generate_figurl.py`, `src/v1ca1/spikesorting/consolidate_sorting.py`, and `src/v1ca1/helper/get_timestamps.py`.
- Prefer small helper functions over thin dataclasses; keep path building explicit.
- Keep script-specific config near `main()`.
- Prefer CLI flags for dataset IDs and roots, especially `--animal-name` and `--date`.
- Add short module and top-level helper docstrings.
- Prefer parquet for tabular summary outputs and pynapple-backed `.npz` for time-domain artifacts.

## Workflow Notes

- Many scripts still hard-code `animal_name`, `date`, and absolute paths, so downstream scripts may need matching path/config updates.
- Common intermediate files: `timestamps_ephys.pkl`, `timestamps_position.pkl`, `timestamps_ephys_all.pkl`, `position.pkl`, `trajectory_times.pkl`.
- Spike sorting expects sorter outputs and sorting analyzer folders under the analysis directory.
- Manual curation flow: `generate_figurl.py` writes a figurl URL for a remote `curation.json`, a human labels in the browser, then `consolidate_sorting.py` applies labels from the local `sorting-curations` checkout with SpikeInterface.
- Probe-to-region mappings are session-specific; override `--v1-probes` and `--ca1-probes` when needed.
- Refactored helper and spikesorting scripts write one JSON run log per execution to `analysis_root / animal_name / date / v1ca1_log/`, including script name, package version, git state, parameters, and key outputs. `get_timestamps.py` also records ephys gap segmentation.

## Guardrails

- Preserve current repo conventions; do not restore older layouts unless asked.
- Check `git diff` before cleanup in files you touch.
- Avoid broad rewrites unless requested.
- For quick verification, prefer `python -m py_compile <file>`.

## Navigation and Environment

- Key areas: `helper`, `spikesorting`, `decoding`, `raster`, `motor`, `ripple`, `oscillation`, `sleep`, `nwb`, `xcorr`, `task_progression`, plus `communication_subspace`, `signal_dim`, and `topology`.
- `environment.yml` is a base environment only; some workflows need extra packages.
- Figurl uses the optional `.[figurl]` extra.
- Package version is in `src/v1ca1/__init__.py`.
- `kyutils` is lab-specific and widely used.
- Common heavy dependencies include `spikeinterface`, `pynwb`, `pynapple`, `position_tools`, and `track_linearization`.
