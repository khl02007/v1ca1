# Lee V1-CA1 Project

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22155664.svg)](https://doi.org/10.5281/zenodo.22155664)

Analysis code related to Lee et al. 2026.

## Repository layout

Within `src/v1ca1/`:

- `helper/`: session paths and loaders, timestamps, intervals,
  logging, and W-track utilities.
- `nwb/`: augmented-NWB export and NWB-based visualization.
- `position/`, `spikesorting/`, `ripple/`, `sleep/`, and
  `oscillation/`: preprocessing, spike sorting, event detection, and core
  physiological workflows.
- `task_progression/`, `motor/`, `decoding/`, `signal_dim/`, and
  `xcorr/`: tuning, encoding and decoding, dimensionality, and correlation
  analyses.
- `behavior/` and `raster/`: behavioral summaries and
  trajectory/place-field visualizations.
- `communication_subspace/` and `topology/`: retained specialized and legacy
  analyses.
- `paper_figures/`: manuscript figure builders.
- `spyglass/`: project-owned `kyuv1ca1` Spyglass tables and
  database-free computation adapters.

At the repository root:

- `paper_figures/assets/` and `paper_figures/output/`: external figure assets
  and generated manuscript figures.
- `figurl/`: per-session, per-shank spike-sorting FigURL records.
- `tests/`: unit and workflow tests, using synthetic or mocked inputs where
  practical.

Some analysis packages retain `legacy/` scripts for provenance. Prefer the
newer CLI-oriented modules and shared helpers for new work.

## Setup

Python 3.11 is the pinned and recommended environment; the package metadata
supports Python 3.11 and newer.

```bash
conda env create -f environment.yml
conda activate v1ca1
```

For an existing environment:

```bash
pip install -e ".[analysis,test]"
```

The optional `glm` extra installs JAX-based dependencies. The custom Spyglass
pipeline uses a separate environment; see the
[Spyglass pipeline guide](src/v1ca1/spyglass/README.md).

## Data and output conventions

Default local paths are:

```text
NWB files:          /stelmo/nwb/raw
analysis artifacts: /stelmo/kyu/analysis/<animal_name>/<date>
Spyglass artifacts: /stelmo/nwb/analysis/kyu/v1ca1
```

Many current CLIs accept `--nwb-root` or `--data-root` when a different layout is
needed. Standalone analyses still consume intermediate session artifacts;
the custom Spyglass pipeline is the NWB-first route for tracked analyses.

Common output formats are Parquet for tabular summaries, Pynapple-compatible
NPZ for time-domain data, and NetCDF-backed xarray datasets for larger model
results. Run metadata is generally written below the session's `v1ca1_log`
directory.

## Common commands

After the editable install, run modules with `python -m`. Use `--help` on an
individual module for its current options.

Create session timestamps:

```bash
python -m v1ca1.helper.get_timestamps \
  --animal-name L14 \
  --date 20240611
```

Detect speed-gated ripples:

```bash
python -m v1ca1.ripple.detect_ripples \
  --animal-name L14 \
  --date 20240611
```

Other primary workflows include position cleaning in `v1ca1.position`, spike
sorting and curation in `v1ca1.spikesorting`, and downstream analyses in
`v1ca1.task_progression`, `v1ca1.ripple`, `v1ca1.motor`, and
`v1ca1.decoding`.

## Augmented NWB files

`v1ca1.nwb.export_augmented_nwb` creates a sibling
`<animal><date>_augmented.nwb` file; it does not overwrite the source NWB.
The exporter always adds the ephys recording intervals. Optional flags add
trajectory and ripple intervals, position, W-track inputs, curated units, and
spike-sorting FigURLs needed by NWB-first workflows:

```bash
python -m v1ca1.nwb.export_augmented_nwb \
  --animal-name L14 \
  --date 20240611 \
  --add-trajectory-times \
  --add-ripples \
  --add-position \
  --add-wtrack-linearization \
  --add-spike-sorting-figurls \
  --add-sorting
```

The exporter reads the existing session artifacts and records their timing and
provenance in the new file. It does not recompute the underlying analyses.

## Manuscript figures

Figure builders live in `v1ca1.paper_figures`; generated PDF, PNG, and SVG
files live in `paper_figures/output`. The configured processed data sets can
be inspected before rendering:

```bash
python -m v1ca1.paper_figures.datasets --include-light-sleep
python -m v1ca1.paper_figures.figure_1 --format svg
```

Figures depend on the expected artifacts under the analysis root. Run a
builder with `--help` to see its data, cache, asset, and output options.

## Spyglass pipeline

`v1ca1.spyglass` defines the project-owned `kyuv1ca1` source, parameter,
selection, and computed tables. It indexes data in augmented NWBs and standard
Spyglass spike-sorting tables, while computed artifacts remain separate from
the source NWB files. Standard Spyglass ingestion must be completed first.

Use the separate `v1ca1-spyglass` environment. Importing `v1ca1.spyglass` is
passive; explicit calls such as `activate()` and `ingest_v1ca1_nwb()` declare
tables or insert the custom NWB catalog. Use `dry_run=True` to validate an
ingestion before insertion. See the
[Spyglass pipeline guide](src/v1ca1/spyglass/README.md) for setup, dependencies,
table relationships, artifact registration, and provenance rules.

## Tests

Run tests with pytest, selecting the files relevant to the workflow:

```bash
pytest -q tests/test_export_augmented_nwb.py
```

The test suite spans both environments. Run `test_spyglass_*.py` in the
separate `v1ca1-spyglass` environment.
