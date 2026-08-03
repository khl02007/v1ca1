# V1–CA1 Spyglass pipeline

This package defines the project-owned `kyuv1ca1` tables. Importing it is
passive: it does not import DataJoint or Spyglass, connect to a database,
activate a schema, insert rows, or modify an NWB file. `activate()` explicitly
declares the schemas and may perform schema/table DDL when requested, but it
does not ingest data, insert parameters or selections, populate computations,
register artifacts, or write to NWB.

## Organization

- `table_specs.py` contains schema names, DataJoint definitions, and scalar
  defaults without importing DataJoint or Spyglass.
- `nwb.py` catalogs augmented-NWB intervals, named position series, W-track
  inputs, and spike-sorting FigURLs. Its loaders open selected arrays only on
  demand.
- `ingest.py` explicitly inserts NWB object pointers and small metadata into
  the custom source tables after standard Spyglass session ingestion.
- `spikes.py` is the shared adapter for `SortedSpikesGroup`,
  `UnitSelectionParams`, and `SpikeSortingOutput` parents. It loads canonical
  ephys-referenced seconds and provides Pynapple and SpikeInterface adapters.
- `selection.py` builds deterministic table-specific UUIDv5 identifiers and
  provenance digests.
- `ripple_modulation.py` and `stability.py` provide database-free computation
  and atomic Parquet writing.
- `tables.py` lazily constructs the DataJoint tables and connects source
  readers, selections, computation, and `register_existing()`.
- `__init__.py` exposes the lazy `activate()` and `ingest_v1ca1_nwb()` entry
  points.

## NWB source catalog

The source tables are `EpochIntervals`, `TrajectoryIntervals`, `Ripples`,
`Position`, `WTrackGraph`, and `SpikeSortingFigurl`. They store object paths,
object IDs, row selectors, and small metadata rather than duplicating NWB
arrays. Source loaders reopen the registered NWB read-only.
`TrajectoryIntervals`, `Ripples`, and `Position` depend directly on
`EpochIntervals`, which enforces an audited parent for every epoch-specific
source row. A provenance-selected ripple epoch is inserted even when its NWB
interval table has no events; that row has `ripple_count=0`.

`Position` is keyed by `epoch` and the actual `position_series_name`, with a
descriptive `position_role`. The current augmented files catalog both
`head_position` (`head`) and `body_position` (`body`). This avoids baking a
head/body enum into analysis selections and makes the selected NWB series
explicit. `Position.load_position()` applies the NWB-recorded leading-sample
analysis offset by default; it does not rewrite or truncate the source series.

Standard Spyglass ingestion must already have created `Session`, `Nwbfile`,
and the relevant spike-sorting rows. `ingest_v1ca1_nwb()` is the separate,
explicit project ingestion call. Its `dry_run=True` mode reads and validates
the catalog without inserting it.

## Units and immutable selections

Both analyses use one shared `SortedSpikesGroup` adapter. It resolves every
group member through `SpikeSortingOutput`, supports imported and curated merge
parents, applies the associated `UnitSelectionParams` include/exclude labels,
checks session and region provenance, and combines canonical spike times in
ephys-referenced seconds. Persistent unit identity is
`(spikesorting_merge_id, unit_id)`; consecutive Pynapple `TsGroup` keys are
temporary computation keys only. There is no project table with one database
row per unit.

An explicit `insert_selection()` snapshots the sorting-group membership,
unit-label filters, and a SHA-256 digest of every selected parameter value. The
natural source/parameter key plus that snapshot is canonicalized into a
table-specific UUIDv5. Repeating an identical selection therefore produces the
same ID, while changing membership, filters, or parameter values produces a
new one. Computation revalidates both snapshots before loading data, so editing
a Manual parameter row after selection is rejected rather than silently
changing an existing UUID's meaning.

## Analysis flows

Ripple modulation uses:

```text
Ripples + EpochIntervals + RippleModulationParameters
    + SortedSpikesGroup / UnitSelectionParams snapshot
    -> RippleModulationSelection (ripple_modulation_id)
    -> RippleModulation
    -> summary.parquet + peri_ripple_firing_rate.parquet
```

The canonical `Ripples` rows contain the speed-gated events that passed the
detector. The default parameters require the source detector threshold to be
2.0 and require `speed_gated=True`. `RippleModulation` uses every event in that
selected source row; it has no downstream ripple-mean-z-score threshold. A
selected row with `ripple_count=0` remains an explicit `no_ripples` result.

Task-progression stability uses:

```text
EpochIntervals + TrajectoryIntervals + Position + WTrackGraph
    + TaskProgressionStabilityParameters
    + SortedSpikesGroup / UnitSelectionParams snapshot
    -> TaskProgressionStabilitySelection (task_progression_stability_id)
    -> TaskProgressionStability
    -> stability.parquet
```

Each result covers one epoch, trajectory, graph configuration, named position
series, region, sorting group, and parameter set. The graph configuration must
match the trajectory. The Parquet retains every selected unit, including
undefined correlations, with explicit QC/status columns; firing-rate and
stability thresholds remain downstream selection choices.

The computed table names are `RippleModulation` and
`TaskProgressionStability`—there is no `Computed` suffix. Empty but valid
selections are recorded through terminal statuses such as `no_units`,
`no_ripples`, or `no_valid_units` rather than being silently omitted.

## Artifacts and provenance

New results are written under the configured `filepath@analysis` store,
defaulting to `/stelmo/nwb/analysis/kyu/v1ca1`, with session-first paths:

```text
<root>/<animal>/<date>/ripple_modulation/<epoch>/<region>/<uuid>/
    summary.parquet
    peri_ripple_firing_rate.parquet

<root>/<animal>/<date>/task_progression_stability/<epoch>/<trajectory>/<region>/<uuid>/
    stability.parquet
```

`make()` computes from the selected NWB and Spyglass sorting sources, writes a
new artifact, and inserts the result row. `register_existing()` instead
validates and partitions matching legacy Parquets, copies the selected content
into the same canonical path, and inserts a result row without rerunning the
analysis. Legacy registration is restricted to matching
`ImportedSpikeSorting` selections. Registration requires complete canonical
schemas; ripple peri-event data must also contain one complete, common time
grid for every unit. Canonical empty artifacts are accepted and recorded with
the applicable `no_units` or `no_ripples` terminal status.

UUID-keyed destinations and result rows are immutable. `register_existing()`
rejects `overwrite=True` and checks for an existing row before calling the
artifact hook. With `skip_duplicates=True`, it returns that row without
touching files; otherwise a duplicate is an error. A new result is inserted
directly into the computed table only after this preflight and artifact
validation. Both routes record `artifact_origin`, a selected-unit digest, and
the actual runtime commits. Registration additionally retains source paths,
source hashes, and any supplied source commits. Neither route writes derived
results into the raw or augmented NWB file.

The project `AnalysisNwbfile` table remains available for future outputs that
fit an NWB-native container. The current Parquet-native analyses use
`filepath@analysis`; activation does not register `AnalysisNwbfile` globally.
Call `register_with_spyglass()` explicitly if it is needed later.

## Environment and version provenance

Run these tables from the separate `v1ca1-spyglass` environment. This keeps
the current V1–CA1 SpikeInterface environment used by sorting and curation
separate from Spyglass's dependency set. Install this repository there without
the full `v1ca1[analysis]` extra, then add the pipeline runtime dependencies,
including the local Spyglass checkout, PyNWB 3.1.3 or newer, Pynapple,
PyArrow, position-tools, and track-linearization.

The documented local Spyglass target is commit
`d5fa7fe1d07c5a349a6d5e0f15d821e5cfe08d38`. Result rows record the actual
runtime V1–CA1 and Spyglass commits. A different commit is retained as
provenance rather than rejected at runtime.

See the repository [Spyglass setup and usage](../../../README.md#project-spyglass-pipeline)
for the activation and ingestion sequence.
