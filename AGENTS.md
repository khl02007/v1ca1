# AGENTS.md

## Ground rules

- Briefly plan before editing.
- Make the smallest change that solves the task.
- Follow existing code style and dependencies.
- Do not silently change data formats, paths, or analysis defaults.
- Validate with the smallest relevant test/script, or say why you could not.
- If assumptions about data are unclear, inspect the code and existing examples first.
- Workflows assume local NWB data and intermediate artifacts that are not in git.

## Paths

- Default roots:
  - analysis: `/stelmo/kyu/analysis`
  - NWB: `/stelmo/nwb/raw`
- Analysis path convention: `analysis_root / animal_name / date`

## Editing Style

- Match the style in `src/v1ca1/helper/get_timestamps.py`.
- Follow the shared-helper patterns in `src/v1ca1/helper/session.py`.
- Prefer small helper functions over thin dataclasses; keep path building explicit.
- Keep script-specific config near `main()`.
- Prefer CLI flags for dataset IDs and roots, especially `--animal-name` and `--date`.
- Add short module and top-level helper docstrings.
- Prefer parquet for tabular summary outputs, pynapple-backed `.npz` for time-domain artifacts, and NetCDF-backed `xarray.Dataset` outputs for larger model-fit results.