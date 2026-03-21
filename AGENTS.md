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
- Also follow the newer shared-helper patterns in `src/v1ca1/helper/session.py`, `src/v1ca1/task_progression/_session.py`, `src/v1ca1/ripple/detect_ripples.py`, `src/v1ca1/ripple/ripple_glm.py`, and `src/v1ca1/signal_dim/meme.py`.
- Prefer small helper functions over thin dataclasses; keep path building explicit.
- Keep script-specific config near `main()`.
- Prefer CLI flags for dataset IDs and roots, especially `--animal-name` and `--date`.
- Centralize generic session/path/timestamp/position loading in `v1ca1.helper.session`.
- Keep task-progression-specific derived representations in `v1ca1.task_progression._session`.
- Add short module and top-level helper docstrings.
- Prefer parquet for tabular summary outputs, pynapple-backed `.npz` for time-domain artifacts, and NetCDF-backed `xarray.Dataset` outputs for larger model-fit results.

## Workflow Notes

- Many scripts still hard-code `animal_name`, `date`, and absolute paths, so downstream scripts may need matching path/config updates.
- Common session artifacts now include `timestamps_ephys.pkl`, `timestamps_position.pkl`, `timestamps_ephys_all.pkl`, `timestamps_ephys.npz`, `timestamps_ephys_all.npz`, `timestamps_position.npz`, `position.pkl`, `trajectory_times.pkl`, `trajectory_times.npz`, `run_times.pkl`, `run_times.npz`, `immobility_times.pkl`, `immobility_times.npz`, and `body_position.pkl`.
- Spike sorting expects sorter outputs and sorting analyzer folders under the analysis directory.
- Ripple workflows expect outputs under `analysis_root / animal_name / date / ripple/`, including detector pickles, interval `.npz` files, and cached ripple-channel LFP data.
- Several modernized workflows still write compatibility pickles alongside canonical `.npz`, parquet, or NetCDF outputs.
- Manual curation flow: `generate_figurl.py` writes a figurl URL for a remote `curation.json`, a human labels in the browser, then `consolidate_sorting.py` applies labels from the local `sorting-curations` checkout with SpikeInterface.
- Probe-to-region mappings are session-specific; override `--v1-probes` and `--ca1-probes` when needed.
- Ripple channel assignments are session-specific too; register new sessions in `v1ca1.ripple._channels` instead of hard-coding channels inside scripts.
- Refactored helper, spikesorting, ripple, task-progression, and signal-dimension scripts write one JSON run log per execution to `analysis_root / animal_name / date / v1ca1_log/`, including script name, package version, git state, parameters, and key outputs. `get_timestamps.py` also records ephys gap segmentation.
- When a workflow writes both tables and figures, keep tables in the workflow data folder and save figures under `analysis_root / animal_name / date / figs / <workflow_name>/` rather than mixing image files into the data directory.

## Legacy Modules

- New feature work belongs in the active `ripple` and `task_progression` modules.
- Treat `src/v1ca1/ripple/legacy` and `src/v1ca1/task_progression/legacy` as reproducibility and inspection code paths.
- Only make minimal fixes in legacy scripts unless the user explicitly asks for legacy workflow changes.

## Guardrails

- Preserve current repo conventions; do not restore older layouts unless asked.
- Check `git diff` before cleanup in files you touch.
- Avoid broad rewrites unless requested.
- For quick verification, prefer `python -m py_compile <file>`.

## Navigation and Environment

- Key areas: `helper`, `spikesorting`, `decoding`, `raster`, `motor`, `position`, `ripple`, `oscillation`, `sleep`, `nwb`, `xcorr`, `task_progression`, plus `communication_subspace`, `signal_dim`, and `topology`.
- `docs` currently exists as a lightweight stub or notes area rather than a full documentation site.
- `environment.yml` is a base environment only; some workflows need extra packages.
- `pyproject.toml` extras are meaningful here: `.[analysis]` for most modern workflows, `.[glm]` for JAX or `nemos`-based GLMs, `.[figurl]` for figurl, and `.[dev]` for tests.
- Package version is in `src/v1ca1/__init__.py`.
- `kyutils` is lab-specific and widely used.
- Common heavy dependencies include `spikeinterface`, `pynwb`, `pynapple`, `position_tools`, `track_linearization`, `jax`, and `nemos`.

## Testing

- There is now targeted pytest coverage for refactored helper and ripple workflows under `tests/`.
- Changes to helper timestamp or interval exports, ripple loading, or ripple summary logic should usually come with test updates.
- For quick verification during small edits, `python -m py_compile <file>` is still the least disruptive check for Python modules; for behavior changes, prefer the relevant pytest targets.
