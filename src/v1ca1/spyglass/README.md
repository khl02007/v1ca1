# V1–CA1 Spyglass pipeline

This package defines the project-owned `kyuv1ca1` tables. Importing it is
passive: database activation, source indexing, parameter insertion, and
computation happen only through explicit calls.

## Organization

- `table_specs.py` contains schema names, DataJoint definitions, and scalar
  parameter defaults without importing DataJoint or Spyglass.
- `nwb.py` catalogs and reads augmented-NWB intervals, position, W-track
  inputs, and spike-sorting FigURLs without using the database.
- `ingest.py` indexes NWB object pointers and small metadata into the custom
  source tables after standard Spyglass session ingestion.
- `spikes.py` loads canonical spike times in ephys-referenced seconds and
  adapts them for Pynapple or SpikeInterface.
- `ripple_modulation.py` wraps the existing ripple-modulation analysis and its
  Parquet artifact validation and writing.
- `tables.py` lazily constructs the DataJoint tables and connects the readers,
  spike adapters, computation, and `register_existing()` workflow.
- `__init__.py` exposes the lazy `activate()` and `ingest_v1ca1_nwb()` entry
  points.

The source tables are `EpochIntervals`, `TrajectoryIntervals`, `Ripples`,
`Position`, `WTrackGraph`, and `SpikeSortingFigurl`. They retain pointers into
the augmented NWB rather than copying its arrays. The initial analysis flow is:

```text
source tables + RippleModulationParameters + SortedSpikesGroup
    -> RippleModulationSelection
    -> RippleModulationComputed
    -> keyed Parquet artifacts under /stelmo/nwb/analysis/kyu/v1ca1
```

See the repository [Spyglass setup and usage](../../../README.md#project-spyglass-pipeline)
for activation order, environment requirements, and the explicit
AnalysisNwbfile registration step.
