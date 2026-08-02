"""Passive DataJoint definitions for the project-owned Spyglass tables.

This module intentionally contains only strings and Python scalar defaults.  It
is safe to import in processes that do not have DataJoint or Spyglass installed.
"""

from __future__ import annotations

from types import MappingProxyType


EXPECTED_SPYGLASS_GIT_COMMIT = "d5fa7fe1d07c5a349a6d5e0f15d821e5cfe08d38"
# Backward-compatible name for callers that inspect the documented pin. Runtime
# provenance is resolved from the imported Spyglass checkout, not this value.
SPYGLASS_GIT_COMMIT = EXPECTED_SPYGLASS_GIT_COMMIT
DEFAULT_SCHEMA_NAME = "kyuv1ca1"
DEFAULT_ANALYSIS_NWBFILE_SCHEMA_NAME = "kyuv1ca1_nwbfile"


EPOCH_INTERVALS_DEFINITION = """
# One augmented-NWB ephys interval and its audited task metadata.
-> Session
epoch: varchar(64)
---
start_time: double
stop_time: double
tags: longblob
nwb_epoch_start_time: double
nwb_epoch_stop_time: double
task_name = NULL: varchar(255)
task_description = NULL: varchar(2000)
task_environment = NULL: varchar(255)
task_source_path = NULL: varchar(1024)
task_object_id = NULL: varchar(64)
associated_file_names: longblob
associated_file_descriptions: longblob
associated_file_source_paths: longblob
epoch_type = NULL: varchar(32)
epoch_type_source = NULL: varchar(1024)
condition = NULL: enum('AB', 'gray', 'BA', 'dark', 'bright', 'sleep')
condition_source = NULL: varchar(2048)
is_light = NULL: bool
source_table_path: varchar(1024)
source_table_object_id = NULL: varchar(64)
source_object_path: varchar(1024)
source_object_id = NULL: varchar(64)
metadata_table_path: varchar(1024)
metadata_table_object_id = NULL: varchar(64)
"""


TRAJECTORY_INTERVALS_DEFINITION = """
# One augmented-NWB trajectory selector; individual laps remain in NWB.
-> Session
epoch: varchar(64)
trajectory_type: varchar(64)
---
interval_count: int unsigned
source_table_path: varchar(1024)
source_table_object_id = NULL: varchar(64)
source_object_path: varchar(1024)
source_object_id = NULL: varchar(64)
"""


RIPPLES_DEFINITION = """
# One augmented-NWB ripple selector; individual events remain in NWB.
-> Session
epoch: varchar(64)
---
ripple_count: int unsigned
detector_zscore_threshold = NULL: double
speed_gated = NULL: bool
detection_parameters = NULL: longblob
provenance_path = NULL: varchar(1024)
provenance_object_id = NULL: varchar(64)
source_table_path: varchar(1024)
source_table_object_id = NULL: varchar(64)
source_object_path: varchar(1024)
source_object_id = NULL: varchar(64)
"""


POSITION_DEFINITION = """
# One half-open augmented-NWB position slice.
-> Session
epoch: varchar(64)
position_type: enum('head', 'body')
---
start_index: bigint unsigned
stop_index_exclusive: bigint unsigned
sample_count: bigint unsigned
analysis_start_offset_samples: bigint unsigned
start_time: double
stop_time: double
first_frame: bigint
last_frame: bigint
video_series_name: varchar(255)
spatial_unit: enum('cm')
source_row_index: int unsigned
source_table_path: varchar(1024)
source_table_object_id = NULL: varchar(64)
source_object_path: varchar(1024)
source_object_id = NULL: varchar(64)
"""


WTRACK_GRAPH_DEFINITION = """
# One augmented-NWB W-track graph configuration.
-> Session
configuration_name: varchar(64)
---
use_hmm: bool
coordinate_unit: enum('cm')
source_row_index: int unsigned
source_table_path: varchar(1024)
source_table_object_id = NULL: varchar(64)
source_object_path: varchar(1024)
source_object_id = NULL: varchar(64)
"""


SPIKE_SORTING_FIGURL_DEFINITION = """
# One augmented-NWB spike-sorting FigURL per probe and shank.
-> Session
probe_idx: int unsigned
shank_idx: int unsigned
---
sorter: varchar(64)
figurl_url: varchar(4095)
data_uri: varchar(1024)
curation_uri: varchar(1024)
source_file: varchar(1024)
source_row_index: int unsigned
source_table_path: varchar(1024)
source_table_object_id = NULL: varchar(64)
source_object_path: varchar(1024)
source_object_id = NULL: varchar(64)
"""


RIPPLE_MODULATION_PARAMETERS_DEFINITION = """
# Named scalar parameters for ripple-triggered firing-rate analysis.
ripple_modulation_param_name: varchar(64)
---
bin_size_s: double
time_before_s: double
time_after_s: double
response_window_start_s: double
response_window_end_s: double
baseline_window_start_s: double
baseline_window_end_s: double
expected_detector_zscore_threshold: double
require_speed_gated: bool
minimum_ripple_mean_zscore = NULL: double
heatmap_normalize: enum('max', 'zscore')
"""


RIPPLE_MODULATION_SELECTION_DEFINITION = """
# One ripple epoch, standard multi-output sorting group, and region.
-> Ripples
-> EpochIntervals
-> RippleModulationParameters
-> SortedSpikesGroup
region: enum('v1', 'ca1')
"""


RIPPLE_MODULATION_COMPUTED_DEFINITION = """
# Keyed Parquet artifacts for one ripple-modulation selection.
-> RippleModulationSelection
---
summary_path: filepath@analysis
peri_ripple_firing_rate_path: filepath@analysis
n_ripples: int unsigned
n_units: int unsigned
sorting_group_members: longblob
sorting_group_members_sha256: char(64)
artifact_origin: enum('computed', 'registered_existing')
runtime_v1ca1_git_commit = NULL: varchar(64)
runtime_spyglass_git_commit = NULL: varchar(64)
legacy_artifact_provenance = NULL: longblob
"""


# SpyglassAnalysis replaces this declaration with its enforced definition.  It
# remains useful for injectable fakes and documents the intended registry.
ANALYSIS_NWBFILE_DEFINITION = """
# Project-owned registry for future analysis-NWB outputs.
analysis_file_name: varchar(64)
---
-> Nwbfile
analysis_file_abs_path: filepath@analysis
analysis_file_description = "": varchar(2000)
analysis_parameters = NULL: blob
INDEX (analysis_file_abs_path)
"""


DEFAULT_RIPPLE_MODULATION_PARAMETERS = MappingProxyType(
    {
        "ripple_modulation_param_name": "default",
        "bin_size_s": 0.02,
        "time_before_s": 0.5,
        "time_after_s": 0.5,
        "response_window_start_s": 0.0,
        "response_window_end_s": 0.1,
        "baseline_window_start_s": -0.5,
        "baseline_window_end_s": -0.3,
        "expected_detector_zscore_threshold": 2.0,
        "require_speed_gated": True,
        "minimum_ripple_mean_zscore": None,
        "heatmap_normalize": "max",
    }
)


TABLE_DEFINITIONS = MappingProxyType(
    {
        "epoch_intervals": EPOCH_INTERVALS_DEFINITION,
        "trajectory_intervals": TRAJECTORY_INTERVALS_DEFINITION,
        "ripples": RIPPLES_DEFINITION,
        "position": POSITION_DEFINITION,
        "wtrack_graph": WTRACK_GRAPH_DEFINITION,
        "spike_sorting_figurl": SPIKE_SORTING_FIGURL_DEFINITION,
        "ripple_modulation_parameters": RIPPLE_MODULATION_PARAMETERS_DEFINITION,
        "ripple_modulation_selection": RIPPLE_MODULATION_SELECTION_DEFINITION,
        "ripple_modulation_computed": RIPPLE_MODULATION_COMPUTED_DEFINITION,
        "analysis_nwbfile": ANALYSIS_NWBFILE_DEFINITION,
    }
)


__all__ = [
    "DEFAULT_ANALYSIS_NWBFILE_SCHEMA_NAME",
    "DEFAULT_RIPPLE_MODULATION_PARAMETERS",
    "DEFAULT_SCHEMA_NAME",
    "EXPECTED_SPYGLASS_GIT_COMMIT",
    "SPYGLASS_GIT_COMMIT",
    "TABLE_DEFINITIONS",
]
