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
-> EpochIntervals
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
-> EpochIntervals
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
-> EpochIntervals
position_series_name: varchar(255)
---
position_role: varchar(64)
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


MOVEMENT_PARAMETERS_DEFINITION = """
# Named parameters defining movement from one position series.
movement_param_name: varchar(64)
---
speed_threshold_cm_s: double
speed_smoothing_sigma_s: double
"""


MOVEMENT_FIRING_RATE_SELECTION_DEFINITION = """
# One immutable position, sorting-group snapshot, region, and movement definition.
movement_firing_rate_id: uuid
---
-> Position
-> MovementParameters
-> SortedSpikesGroup
region: enum('v1', 'ca1')
sorting_group_members: longblob
sorting_group_members_sha256: char(64)
unit_filter_include_labels: longblob
unit_filter_exclude_labels: longblob
unit_filter_params_sha256: char(64)
movement_parameters_sha256: char(64)
"""


MOVEMENT_FIRING_RATE_DEFINITION = """
# Keyed movement support and all-unit movement firing-rate artifacts.
-> MovementFiringRateSelection
---
movement_firing_rate_path: filepath@analysis
movement_intervals_path: filepath@analysis
n_units: int unsigned
n_valid_units: int unsigned
n_units_with_spikes: int unsigned
movement_interval_count: int unsigned
movement_duration_s: double
analysis_status: enum('valid', 'no_units', 'no_valid_position', 'no_movement')
selected_units_sha256: char(64)
runtime_v1ca1_git_commit = NULL: varchar(64)
runtime_spyglass_git_commit = NULL: varchar(64)
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
heatmap_normalize: enum('max', 'zscore')
"""


RIPPLE_MODULATION_SELECTION_DEFINITION = """
# One immutable ripple epoch, sorting-group snapshot, and region selection.
ripple_modulation_id: uuid
---
-> Ripples
-> RippleModulationParameters
-> SortedSpikesGroup
region: enum('v1', 'ca1')
sorting_group_members: longblob
sorting_group_members_sha256: char(64)
unit_filter_include_labels: longblob
unit_filter_exclude_labels: longblob
unit_filter_params_sha256: char(64)
ripple_modulation_parameters_sha256: char(64)
"""


RIPPLE_MODULATION_DEFINITION = """
# Keyed Parquet artifacts for one ripple-modulation selection.
-> RippleModulationSelection
---
summary_path: filepath@analysis
peri_ripple_firing_rate_path: filepath@analysis
n_ripples: int unsigned
n_units: int unsigned
n_valid_units: int unsigned
analysis_status: enum('valid', 'no_units', 'no_ripples', 'no_valid_units')
selected_units_sha256: char(64)
artifact_origin: enum('computed', 'registered_existing')
runtime_v1ca1_git_commit = NULL: varchar(64)
runtime_spyglass_git_commit = NULL: varchar(64)
legacy_artifact_provenance = NULL: longblob
"""


TASK_PROGRESSION_STABILITY_PARAMETERS_DEFINITION = """
# Named numerical parameters for odd/even task-progression stability.
task_progression_stability_param_name: varchar(64)
---
place_bin_size_cm: double
"""


TASK_PROGRESSION_STABILITY_SELECTION_DEFINITION = """
# One immutable trajectory/graph selection downstream of saved movement support.
task_progression_stability_id: uuid
---
-> TrajectoryIntervals
-> WTrackGraph
-> MovementFiringRate
-> TaskProgressionStabilityParameters
task_progression_stability_parameters_sha256: char(64)
"""


TASK_PROGRESSION_STABILITY_DEFINITION = """
# One all-unit QC Parquet for a trajectory-level stability selection.
-> TaskProgressionStabilitySelection
---
stability_path: filepath@analysis
n_units: int unsigned
n_valid_units: int unsigned
analysis_status: enum('valid', 'no_units', 'no_valid_position', 'no_movement', 'no_valid_units')
selected_units_sha256: char(64)
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
        "heatmap_normalize": "max",
    }
)


DEFAULT_MOVEMENT_PARAMETERS = MappingProxyType(
    {
        "movement_param_name": "default",
        "speed_threshold_cm_s": 4.0,
        "speed_smoothing_sigma_s": 0.1,
    }
)


DEFAULT_TASK_PROGRESSION_STABILITY_PARAMETERS = MappingProxyType(
    {
        "task_progression_stability_param_name": "default",
        "place_bin_size_cm": 4.0,
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
        "movement_parameters": MOVEMENT_PARAMETERS_DEFINITION,
        "movement_firing_rate_selection": (
            MOVEMENT_FIRING_RATE_SELECTION_DEFINITION
        ),
        "movement_firing_rate": MOVEMENT_FIRING_RATE_DEFINITION,
        "ripple_modulation_parameters": RIPPLE_MODULATION_PARAMETERS_DEFINITION,
        "ripple_modulation_selection": RIPPLE_MODULATION_SELECTION_DEFINITION,
        "ripple_modulation": RIPPLE_MODULATION_DEFINITION,
        "task_progression_stability_parameters": (
            TASK_PROGRESSION_STABILITY_PARAMETERS_DEFINITION
        ),
        "task_progression_stability_selection": (
            TASK_PROGRESSION_STABILITY_SELECTION_DEFINITION
        ),
        "task_progression_stability": TASK_PROGRESSION_STABILITY_DEFINITION,
        "analysis_nwbfile": ANALYSIS_NWBFILE_DEFINITION,
    }
)


__all__ = [
    "DEFAULT_ANALYSIS_NWBFILE_SCHEMA_NAME",
    "DEFAULT_MOVEMENT_PARAMETERS",
    "DEFAULT_RIPPLE_MODULATION_PARAMETERS",
    "DEFAULT_TASK_PROGRESSION_STABILITY_PARAMETERS",
    "DEFAULT_SCHEMA_NAME",
    "EXPECTED_SPYGLASS_GIT_COMMIT",
    "SPYGLASS_GIT_COMMIT",
    "TABLE_DEFINITIONS",
]
