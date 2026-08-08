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


REGION_SORTED_SPIKES_GROUP_DEFINITION = """
# One immutable region-specific logical view of a standard sorted-spikes group.
region_sorted_spikes_group_id: uuid
---
-> SortedSpikesGroup
region_name: varchar(64)
sorting_group_members: longblob
sorting_group_members_sha256: char(64)
unit_filter_include_labels: longblob
unit_filter_exclude_labels: longblob
unit_filter_params_sha256: char(64)
n_units: int unsigned
selected_units_sha256: char(64)
"""


MOVEMENT_PARAMETERS_DEFINITION = """
# Named parameters defining movement from one position series.
movement_param_name: varchar(64)
---
speed_threshold_cm_s: double
speed_smoothing_sigma_s: double
"""


EPOCH_MOTOR_BEHAVIOR_PARAMETERS_DEFINITION = """
# Named progression-bin parameters for epoch-level motor summaries.
epoch_motor_behavior_param_name: varchar(64)
---
progression_bin_size_cm: double
"""


EPOCH_MOTOR_BEHAVIOR_SELECTION_DEFINITION = """
# One immutable run-epoch motor-behavior source snapshot.
epoch_motor_behavior_id: uuid
---
-> EpochIntervals
-> Position.proj(primary_position_series_name='position_series_name')
-> Position.proj(orientation_reference_position_series_name='position_series_name')
-> MovementParameters
-> TrajectoryIntervals.proj(center_to_left_trajectory_type='trajectory_type')
-> TrajectoryIntervals.proj(center_to_right_trajectory_type='trajectory_type')
-> TrajectoryIntervals.proj(left_to_center_trajectory_type='trajectory_type')
-> TrajectoryIntervals.proj(right_to_center_trajectory_type='trajectory_type')
-> WTrackGraph.proj(center_to_left_configuration_name='configuration_name')
-> WTrackGraph.proj(center_to_right_configuration_name='configuration_name')
-> WTrackGraph.proj(left_to_center_configuration_name='configuration_name')
-> WTrackGraph.proj(right_to_center_configuration_name='configuration_name')
-> EpochMotorBehaviorParameters
primary_position_role: varchar(64)
orientation_reference_position_role: varchar(64)
position_offset_samples: bigint unsigned
epoch_interval_row_sha256: char(64)
primary_position_row_sha256: char(64)
orientation_reference_position_row_sha256: char(64)
aligned_position_timestamps_sha256: char(64)
primary_position_source_sha256: char(64)
orientation_reference_position_source_sha256: char(64)
trajectory_rows_sha256_by_type: longblob
trajectory_intervals_sha256_by_type: longblob
graph_rows_sha256_by_trajectory: longblob
graph_inputs_sha256_by_trajectory: longblob
movement_parameters_sha256: char(64)
epoch_motor_behavior_parameters_sha256: char(64)
epoch_motor_behavior_output_rule_sha256: char(64)
"""


EPOCH_MOTOR_BEHAVIOR_DEFINITION = """
# One epoch-level motor-distribution, progression, and trajectory-QC bundle.
-> EpochMotorBehaviorSelection
---
artifact_manifest_path: filepath@analysis
distribution_summary_path: filepath@analysis
progression_summary_path: filepath@analysis
trajectory_qc_path: filepath@analysis
schema_version: varchar(8)
bundle_schema_version: varchar(8)
n_position_samples_input: bigint unsigned
n_finite_position_samples: bigint unsigned
n_dropped_nonfinite_samples: bigint unsigned
n_movement_samples: bigint unsigned
movement_duration_s: double
n_supported_trajectories: tinyint unsigned
sampling_rate_hz: double
median_sample_interval_s: double
maximum_sample_gap_s: double
analysis_status: enum('valid', 'partial_valid', 'no_valid_position', 'no_movement', 'no_trials')
artifact_origin: enum('computed', 'registered_existing')
runtime_v1ca1_git_commit = NULL: varchar(64)
runtime_spyglass_git_commit = NULL: varchar(64)
legacy_artifact_provenance = NULL: longblob
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


TUNING_CURVE_PARAMETERS_DEFINITION = """
# Named binning and smoothing parameters shared by tuning-curve pipelines.
tuning_curve_param_name: varchar(64)
---
binning_mode: enum('bin_size_cm', 'bin_count')
place_bin_size_cm = NULL: double
position_bin_count = NULL: smallint unsigned
gaussian_smoothing_sigma_bins: double
"""


TUNING_SIMILARITY_PARAMETERS_DEFINITION = """
# Named fixed metric for comparing matching path-specific tuning curves.
tuning_similarity_param_name: varchar(64)
---
similarity_metric: enum('correlation', 'absolute_overlap', 'shape_overlap')
"""


PATH_SPECIFIC_PLACE_TUNING_CURVE_SELECTION_DEFINITION = """
# One immutable path, movement, unit, parameter, and trial-subset selection.
path_specific_place_tuning_curve_id: uuid
---
-> TrajectoryIntervals
-> WTrackGraph
-> MovementFiringRate
-> TuningCurveParameters
trial_subset: enum('all', 'odd', 'even')
tuning_curve_parameters_sha256: char(64)
"""


PATH_SPECIFIC_PLACE_TUNING_CURVE_DEFINITION = """
# One all-unit path-specific tuning-curve DataArray artifact.
-> PathSpecificPlaceTuningCurveSelection
---
tuning_curve_path: filepath@analysis
n_units: int unsigned
n_valid_units: int unsigned
n_trials: int unsigned
support_duration_s: double
n_feature_samples: int unsigned
n_position_bins: int unsigned
analysis_status: enum('valid', 'no_units', 'no_trials', 'no_valid_position', 'no_movement', 'no_valid_units')
selected_units_sha256: char(64)
artifact_origin: enum('computed', 'registered_existing')
runtime_v1ca1_git_commit = NULL: varchar(64)
runtime_spyglass_git_commit = NULL: varchar(64)
legacy_artifact_provenance = NULL: longblob
"""


PATH_SPECIFIC_PLACE_TUNING_SIMILARITY_SELECTION_DEFINITION = """
# One immutable four-path, all-trial tuning-similarity selection.
path_specific_place_tuning_similarity_id: uuid
---
-> PathSpecificPlaceTuningCurve.proj(center_to_left_tuning_curve_id='path_specific_place_tuning_curve_id')
-> PathSpecificPlaceTuningCurve.proj(center_to_right_tuning_curve_id='path_specific_place_tuning_curve_id')
-> PathSpecificPlaceTuningCurve.proj(left_to_center_tuning_curve_id='path_specific_place_tuning_curve_id')
-> PathSpecificPlaceTuningCurve.proj(right_to_center_tuning_curve_id='path_specific_place_tuning_curve_id')
-> TuningSimilarityParameters
tuning_similarity_parameters_sha256: char(64)
"""


PATH_SPECIFIC_PLACE_TUNING_SIMILARITY_DEFINITION = """
# One all-unit, four-comparison tuning-similarity Parquet artifact.
-> PathSpecificPlaceTuningSimilaritySelection
---
similarity_path: filepath@analysis
n_units: int unsigned
n_valid_comparisons: int unsigned
n_units_with_valid_comparison: int unsigned
analysis_status: enum('valid', 'no_units', 'no_valid_comparisons')
selected_units_sha256: char(64)
artifact_origin: enum('computed', 'registered_existing')
runtime_v1ca1_git_commit = NULL: varchar(64)
runtime_spyglass_git_commit = NULL: varchar(64)
legacy_artifact_provenance = NULL: longblob
"""


DPP_TUNING_CURVE_SELECTION_DEFINITION = """
# One immutable same-turn DPP, movement, unit, parameter, and trial selection.
dpp_tuning_curve_id: uuid
---
-> TrajectoryIntervals.proj(outbound_trajectory_type='trajectory_type')
-> TrajectoryIntervals.proj(inbound_trajectory_type='trajectory_type')
-> WTrackGraph.proj(outbound_configuration_name='configuration_name')
-> WTrackGraph.proj(inbound_configuration_name='configuration_name')
-> MovementFiringRate
-> TuningCurveParameters
turn_type: enum('left', 'right')
trial_subset: enum('all', 'odd', 'even')
tuning_curve_parameters_sha256: char(64)
"""


DPP_TUNING_CURVE_DEFINITION = """
# One all-unit directional path-progression tuning-curve DataArray artifact.
-> DPPTuningCurveSelection
---
tuning_curve_path: filepath@analysis
n_units: int unsigned
n_valid_units: int unsigned
n_trials: int unsigned
n_outbound_trials: int unsigned
n_inbound_trials: int unsigned
support_duration_s: double
n_feature_samples: int unsigned
n_position_bins: int unsigned
analysis_status: enum('valid', 'no_units', 'no_trials', 'no_valid_position', 'no_movement', 'no_valid_units')
selected_units_sha256: char(64)
artifact_origin: enum('computed', 'registered_existing')
runtime_v1ca1_git_commit = NULL: varchar(64)
runtime_spyglass_git_commit = NULL: varchar(64)
legacy_artifact_provenance = NULL: longblob
"""


PATH_SPECIFIC_PLACE_STABILITY_SELECTION_DEFINITION = """
# One immutable odd/even path-specific tuning-curve pair.
path_specific_place_stability_id: uuid
---
-> PathSpecificPlaceTuningCurve.proj(odd_path_specific_place_tuning_curve_id='path_specific_place_tuning_curve_id')
-> PathSpecificPlaceTuningCurve.proj(even_path_specific_place_tuning_curve_id='path_specific_place_tuning_curve_id')
"""


PATH_SPECIFIC_PLACE_STABILITY_DEFINITION = """
# One all-unit QC Parquet for a trajectory-level stability selection.
-> PathSpecificPlaceStabilitySelection
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


DPP_ENCODING_COMPARISON_PARAMETERS_DEFINITION = """
# Named cross-validation, binning, smoothing, and unit-filter parameters.
dpp_encoding_comparison_param_name: varchar(64)
---
n_folds: smallint unsigned
evaluation_bin_size_s: double
spatial_bin_size_cm: double
gaussian_smoothing_sigma_bins: double
random_seed: int
minimum_movement_firing_rate_hz: double
minimum_stability_correlation: double
"""


DPP_ENCODING_COMPARISON_SELECTION_DEFINITION = """
# One immutable epoch-level four-model encoding-comparison selection.
dpp_encoding_comparison_id: uuid
---
-> RegionSortedSpikesGroup
-> MovementFiringRate
-> TrajectoryIntervals.proj(center_to_left_trajectory_type='trajectory_type')
-> TrajectoryIntervals.proj(center_to_right_trajectory_type='trajectory_type')
-> TrajectoryIntervals.proj(left_to_center_trajectory_type='trajectory_type')
-> TrajectoryIntervals.proj(right_to_center_trajectory_type='trajectory_type')
-> WTrackGraph.proj(center_to_left_configuration_name='configuration_name')
-> WTrackGraph.proj(center_to_right_configuration_name='configuration_name')
-> WTrackGraph.proj(left_to_center_configuration_name='configuration_name')
-> WTrackGraph.proj(right_to_center_configuration_name='configuration_name')
-> WTrackGraph.proj(full_w_configuration_name='configuration_name')
-> PathSpecificPlaceStability.proj(center_to_left_stability_id='path_specific_place_stability_id')
-> PathSpecificPlaceStability.proj(center_to_right_stability_id='path_specific_place_stability_id')
-> PathSpecificPlaceStability.proj(left_to_center_stability_id='path_specific_place_stability_id')
-> PathSpecificPlaceStability.proj(right_to_center_stability_id='path_specific_place_stability_id')
-> DPPEncodingComparisonParameters
dpp_encoding_comparison_parameters_sha256: char(64)
"""


DPP_ENCODING_COMPARISON_DEFINITION = """
# One eligible-unit four-model cross-validated encoding Parquet artifact.
-> DPPEncodingComparisonSelection
---
encoding_comparison_path: filepath@analysis
n_units_input: int unsigned
n_units_eligible: int unsigned
n_units_valid: int unsigned
analysis_status: enum('valid', 'no_eligible_units', 'no_valid_units')
eligible_units_sha256: char(64)
artifact_origin: enum('computed', 'registered_existing')
runtime_v1ca1_git_commit = NULL: varchar(64)
runtime_spyglass_git_commit = NULL: varchar(64)
legacy_artifact_provenance = NULL: longblob
"""


PATH_PROGRESSION_DECODING_PARAMETERS_DEFINITION = """
# Named Bayesian-decoding, spatial-binning, and unit-cohort parameters.
path_progression_decoding_param_name: varchar(64)
---
decoding_bin_size_s: double
sliding_window_size_bins: smallint unsigned
spatial_bin_size_cm: double
minimum_movement_firing_rate_hz: double
minimum_stability_correlation = NULL: double
"""


PATH_PROGRESSION_DECODING_SELECTION_DEFINITION = """
# One immutable epoch-level decoding selection with an explicit cohort epoch.
path_progression_decoding_comparison_id: uuid
---
-> RegionSortedSpikesGroup
-> MovementFiringRate
-> MovementFiringRate.proj(cohort_movement_firing_rate_id='movement_firing_rate_id')
-> TrajectoryIntervals.proj(center_to_left_trajectory_type='trajectory_type')
-> TrajectoryIntervals.proj(center_to_right_trajectory_type='trajectory_type')
-> TrajectoryIntervals.proj(left_to_center_trajectory_type='trajectory_type')
-> TrajectoryIntervals.proj(right_to_center_trajectory_type='trajectory_type')
-> WTrackGraph.proj(center_to_left_configuration_name='configuration_name')
-> WTrackGraph.proj(center_to_right_configuration_name='configuration_name')
-> WTrackGraph.proj(left_to_center_configuration_name='configuration_name')
-> WTrackGraph.proj(right_to_center_configuration_name='configuration_name')
-> PathSpecificPlaceStability.proj(center_to_left_stability_id='path_specific_place_stability_id')
-> PathSpecificPlaceStability.proj(center_to_right_stability_id='path_specific_place_stability_id')
-> PathSpecificPlaceStability.proj(left_to_center_stability_id='path_specific_place_stability_id')
-> PathSpecificPlaceStability.proj(right_to_center_stability_id='path_specific_place_stability_id')
-> PathSpecificPlaceStability.proj(cohort_center_to_left_stability_id='path_specific_place_stability_id')
-> PathSpecificPlaceStability.proj(cohort_center_to_right_stability_id='path_specific_place_stability_id')
-> PathSpecificPlaceStability.proj(cohort_left_to_center_stability_id='path_specific_place_stability_id')
-> PathSpecificPlaceStability.proj(cohort_right_to_center_stability_id='path_specific_place_stability_id')
-> PathProgressionDecodingParameters
cohort_epoch: varchar(64)
path_progression_decoding_parameters_sha256: char(64)
eligibility_rule_sha256: char(64)
transfer_spec_sha256: char(64)
decoding_output_rule_sha256: char(64)
"""


PATH_PROGRESSION_DECODING_DEFINITION = """
# One shared-cohort path-progression decoding artifact bundle.
-> PathProgressionDecodingComparisonSelection
---
artifact_manifest_path: filepath@analysis
decoding_summary_path: filepath@analysis
unit_eligibility_path: filepath@analysis
n_units_input: int unsigned
n_units_eligible: int unsigned
n_transfer_pairs_expected: smallint unsigned
n_transfer_pairs_valid: smallint unsigned
n_decoded_samples: bigint unsigned
analysis_status: enum('valid', 'partial_valid', 'no_units', 'no_eligible_units', 'no_valid_decodes')
eligible_units_sha256: char(64)
runtime_v1ca1_git_commit = NULL: varchar(64)
runtime_spyglass_git_commit = NULL: varchar(64)
"""


PATH_SPECIFIC_PLACE_DECODING_PARAMETERS_DEFINITION = """
# Named within-epoch path-specific physical-place decoding parameters.
path_specific_place_decoding_param_name: varchar(64)
---
n_folds: smallint unsigned
decoding_bin_size_s: double
sliding_window_size_bins: smallint unsigned
spatial_bin_size_cm: double
random_seed: int unsigned
"""


PATH_SPECIFIC_PLACE_DECODING_SELECTION_DEFINITION = """
# One immutable all-unit within-epoch physical-place decoding selection.
path_specific_place_decoding_id: uuid
---
-> RegionSortedSpikesGroup
-> MovementFiringRate
-> TrajectoryIntervals.proj(center_to_left_trajectory_type='trajectory_type')
-> TrajectoryIntervals.proj(center_to_right_trajectory_type='trajectory_type')
-> TrajectoryIntervals.proj(left_to_center_trajectory_type='trajectory_type')
-> TrajectoryIntervals.proj(right_to_center_trajectory_type='trajectory_type')
-> WTrackGraph.proj(center_to_left_configuration_name='configuration_name')
-> WTrackGraph.proj(center_to_right_configuration_name='configuration_name')
-> WTrackGraph.proj(left_to_center_configuration_name='configuration_name')
-> WTrackGraph.proj(right_to_center_configuration_name='configuration_name')
-> PathSpecificPlaceDecodingParameters
path_specific_place_decoding_parameters_sha256: char(64)
path_specific_place_decoding_output_rule_sha256: char(64)
"""


PATH_SPECIFIC_PLACE_DECODING_DEFINITION = """
# One within-epoch path-specific physical-place decoding artifact bundle.
-> PathSpecificPlaceDecodingSelection
---
artifact_manifest_path: filepath@analysis
selected_units_path: filepath@analysis
fold_qc_path: filepath@analysis
decoding_summary_path: filepath@analysis
decoding_error_by_position_path: filepath@analysis
n_units: int unsigned
n_folds_expected: smallint unsigned
n_folds_valid: smallint unsigned
n_decoded_samples: bigint unsigned
analysis_status: enum('valid', 'partial_valid', 'no_units', 'no_valid_decodes')
selected_units_sha256: char(64)
artifact_origin: enum('computed', 'registered_existing')
runtime_v1ca1_git_commit = NULL: varchar(64)
runtime_spyglass_git_commit = NULL: varchar(64)
legacy_artifact_provenance = NULL: longblob
"""


MOTOR_ENCODING_COMPARISON_PARAMETERS_DEFINITION = """
# Named nested-CV, basis, motor-feature, and unit-filter parameters.
motor_encoding_comparison_param_name: varchar(64)
---
evaluation_bin_size_s: double
outer_n_folds: smallint unsigned
inner_n_folds: smallint unsigned
random_seed: int
ridge_values: longblob
spatial_bin_sizes_cm: longblob
minimum_movement_firing_rate_hz: double
motor_feature_mode: enum('zscore', 'spline')
motor_zscore_eps: double
motor_spline_n_basis: smallint unsigned
motor_spline_order: smallint unsigned
position_spline_order: smallint unsigned
speed_smoothing_sigma_s: double
generalized_place_branch_gap_cm: double
"""


MOTOR_ENCODING_COMPARISON_SELECTION_DEFINITION = """
# One immutable epoch-level nine-model motor-encoding selection.
motor_encoding_comparison_id: uuid
---
-> RegionSortedSpikesGroup
-> MovementFiringRate
-> Position.proj(primary_position_series_name='position_series_name')
-> Position.proj(orientation_reference_position_series_name='position_series_name')
-> TrajectoryIntervals.proj(center_to_left_trajectory_type='trajectory_type')
-> TrajectoryIntervals.proj(center_to_right_trajectory_type='trajectory_type')
-> TrajectoryIntervals.proj(left_to_center_trajectory_type='trajectory_type')
-> TrajectoryIntervals.proj(right_to_center_trajectory_type='trajectory_type')
-> WTrackGraph.proj(center_to_left_configuration_name='configuration_name')
-> WTrackGraph.proj(center_to_right_configuration_name='configuration_name')
-> WTrackGraph.proj(left_to_center_configuration_name='configuration_name')
-> WTrackGraph.proj(right_to_center_configuration_name='configuration_name')
-> WTrackGraph.proj(full_w_configuration_name='configuration_name')
-> MotorEncodingComparisonParameters
motor_encoding_comparison_parameters_sha256: char(64)
motor_encoding_comparison_model_spec_sha256: char(64)
motor_encoding_comparison_output_rule_sha256: char(64)
"""


MOTOR_ENCODING_COMPARISON_DEFINITION = """
# One nested-CV and full-refit motor-encoding artifact bundle.
-> MotorEncodingComparisonSelection
---
artifact_manifest_path: filepath@analysis
nested_cv_path: filepath@analysis
full_refit_path: filepath@analysis
selected_units_path: filepath@analysis
n_units_input: int unsigned
n_units_eligible: int unsigned
n_units_valid: int unsigned
n_outer_folds_expected: smallint unsigned
n_outer_folds_valid: smallint unsigned
analysis_status: enum('valid', 'partial_valid', 'no_units', 'no_eligible_units', 'no_trials', 'no_valid_position', 'no_movement', 'no_valid_units')
selected_units_sha256: char(64)
artifact_origin: enum('computed', 'registered_existing')
runtime_v1ca1_git_commit = NULL: varchar(64)
runtime_spyglass_git_commit = NULL: varchar(64)
legacy_artifact_provenance = NULL: longblob
"""


DARK_LIGHT_GLM_PARAMETERS_DEFINITION = """
# Named dark/light GLM candidate-grid and unit-filter parameters.
dark_light_glm_param_name: varchar(64)
---
basis_candidate_mode: enum('spatial_bin_size_cm', 'n_splines')
basis_candidates: longblob
bin_sizes_s: longblob
ridges: longblob
n_folds: smallint unsigned
random_seed: int
spline_order: smallint unsigned
min_dark_firing_rate_hz: double
min_light_firing_rate_hz: double
use_speed: bool
speed_feature_mode: enum('linear', 'bspline')
n_splines_speed: smallint unsigned
spline_order_speed: smallint unsigned
speed_bounds = NULL: longblob
speed_smoothing_sigma_s = 0.1: double
"""


DARK_LIGHT_GLM_SELECTION_DEFINITION = """
# One immutable same-session light/dark four-model GLM selection.
dark_light_glm_id: uuid
---
-> RegionSortedSpikesGroup
-> MovementFiringRate.proj(dark_movement_firing_rate_id='movement_firing_rate_id')
-> MovementFiringRate.proj(light_movement_firing_rate_id='movement_firing_rate_id')
-> TrajectoryIntervals.proj(dark_epoch='epoch', dark_center_to_left_trajectory_type='trajectory_type')
-> TrajectoryIntervals.proj(dark_epoch='epoch', dark_center_to_right_trajectory_type='trajectory_type')
-> TrajectoryIntervals.proj(dark_epoch='epoch', dark_left_to_center_trajectory_type='trajectory_type')
-> TrajectoryIntervals.proj(dark_epoch='epoch', dark_right_to_center_trajectory_type='trajectory_type')
-> TrajectoryIntervals.proj(light_epoch='epoch', light_center_to_left_trajectory_type='trajectory_type')
-> TrajectoryIntervals.proj(light_epoch='epoch', light_center_to_right_trajectory_type='trajectory_type')
-> TrajectoryIntervals.proj(light_epoch='epoch', light_left_to_center_trajectory_type='trajectory_type')
-> TrajectoryIntervals.proj(light_epoch='epoch', light_right_to_center_trajectory_type='trajectory_type')
-> WTrackGraph.proj(center_to_left_configuration_name='configuration_name')
-> WTrackGraph.proj(center_to_right_configuration_name='configuration_name')
-> WTrackGraph.proj(left_to_center_configuration_name='configuration_name')
-> WTrackGraph.proj(right_to_center_configuration_name='configuration_name')
-> DarkLightGLMParameters
dark_light_glm_parameters_sha256: char(64)
dark_light_glm_output_rule_sha256: char(64)
"""


DARK_LIGHT_GLM_DEFINITION = """
# One coupled dark/light candidate-selection and selected-model artifact bundle.
-> DarkLightGLMSelection
---
artifact_manifest_path: filepath@analysis
selected_units_path: filepath@analysis
selection_summary_path: filepath@analysis
visual_model_path: filepath@analysis
task_segment_bump_model_path: filepath@analysis
task_segment_scalar_model_path: filepath@analysis
task_dense_gain_model_path: filepath@analysis
schema_version: varchar(8)
n_units: int unsigned
n_candidates: int unsigned
n_selected_models: smallint unsigned
analysis_status: enum('valid', 'partial_valid', 'no_units', 'no_eligible_units', 'no_valid_position', 'no_movement', 'no_valid_units')
selected_units_sha256: char(64)
artifact_origin: enum('computed', 'registered_existing')
runtime_v1ca1_git_commit = NULL: varchar(64)
runtime_spyglass_git_commit = NULL: varchar(64)
legacy_artifact_provenance = NULL: longblob
"""


SWAP_GLM_PARAMETERS_DEFINITION = """
# Named held-out swapped-light scoring parameters.
swap_glm_param_name: varchar(64)
---
swap_light_offset: bool
observed_spatial_bin_size_cm: double
"""


SWAP_GLM_SELECTION_DEFINITION = """
# One immutable held-out light-epoch swapped-segment GLM selection.
swap_glm_id: uuid
---
-> DarkLightGLM
-> RegionSortedSpikesGroup
-> MovementFiringRate.proj(light_test_movement_firing_rate_id='movement_firing_rate_id')
-> EpochIntervals.proj(dark_epoch='epoch')
-> EpochIntervals.proj(light_train_epoch='epoch')
-> TrajectoryIntervals.proj(light_test_epoch='epoch', light_test_center_to_left_trajectory_type='trajectory_type')
-> TrajectoryIntervals.proj(light_test_epoch='epoch', light_test_center_to_right_trajectory_type='trajectory_type')
-> TrajectoryIntervals.proj(light_test_epoch='epoch', light_test_left_to_center_trajectory_type='trajectory_type')
-> TrajectoryIntervals.proj(light_test_epoch='epoch', light_test_right_to_center_trajectory_type='trajectory_type')
-> WTrackGraph.proj(center_to_left_configuration_name='configuration_name')
-> WTrackGraph.proj(center_to_right_configuration_name='configuration_name')
-> WTrackGraph.proj(left_to_center_configuration_name='configuration_name')
-> WTrackGraph.proj(right_to_center_configuration_name='configuration_name')
-> SwapGLMParameters
dark_condition: enum('dark')
light_train_condition: enum('AB', 'gray', 'BA', 'bright')
light_test_condition: enum('AB', 'gray', 'BA', 'bright')
dark_light_manifest_sha256: char(64)
dark_light_selected_sha256_by_model: longblob
dark_light_parameter_sha256: char(64)
dark_light_output_rule_sha256: char(64)
upstream_analysis_status: enum('valid', 'partial_valid', 'no_units', 'no_eligible_units', 'no_valid_position', 'no_movement', 'no_valid_units')
swap_glm_parameters_sha256: char(64)
swap_glm_output_rule_sha256: char(64)
"""


SWAP_GLM_DEFINITION = """
# One held-out swapped-light unit audit and consolidated model-score bundle.
-> SwapGLMSelection
---
artifact_manifest_path: filepath@analysis
selected_units_path: filepath@analysis
swap_glm_path: filepath@analysis
schema_version: varchar(8)
bundle_schema_version: varchar(8)
n_units: int unsigned
n_valid_units: int unsigned
analysis_status: enum('valid', 'partial_valid', 'upstream_terminal', 'no_units', 'no_valid_position', 'no_movement', 'no_trajectory_samples', 'no_valid_units')
selected_units_sha256: char(64)
dark_light_manifest_sha256: char(64)
dark_light_selected_sha256_by_model: longblob
dark_light_parameter_sha256: char(64)
dark_light_output_rule_sha256: char(64)
upstream_analysis_status: enum('valid', 'partial_valid', 'no_units', 'no_eligible_units', 'no_valid_position', 'no_movement', 'no_valid_units')
artifact_origin: enum('computed', 'registered_existing')
runtime_v1ca1_git_commit = NULL: varchar(64)
runtime_spyglass_git_commit = NULL: varchar(64)
legacy_artifact_provenance = NULL: longblob
"""


SWAP_TUNING_CURVE_COMPARISON_PARAMETERS_DEFINITION = """
# Named empirical swapped-light tuning-comparison parameters.
swap_tuning_curve_comparison_param_name: varchar(64)
---
evaluation_bin_size_s: double
gaussian_smoothing_sigma_bins: double
min_dark_firing_rate_hz: double
min_light_firing_rate_hz: double
"""


SWAP_TUNING_CURVE_COMPARISON_SELECTION_DEFINITION = """
# One immutable three-epoch empirical swapped-light tuning selection.
swap_tuning_curve_comparison_id: uuid
---
-> RegionSortedSpikesGroup
-> MovementFiringRate.proj(dark_movement_firing_rate_id='movement_firing_rate_id')
-> MovementFiringRate.proj(light_train_movement_firing_rate_id='movement_firing_rate_id')
-> MovementFiringRate.proj(light_test_movement_firing_rate_id='movement_firing_rate_id')
-> PathSpecificPlaceTuningCurve.proj(dark_center_to_left_tuning_curve_id='path_specific_place_tuning_curve_id')
-> PathSpecificPlaceTuningCurve.proj(dark_center_to_right_tuning_curve_id='path_specific_place_tuning_curve_id')
-> PathSpecificPlaceTuningCurve.proj(dark_left_to_center_tuning_curve_id='path_specific_place_tuning_curve_id')
-> PathSpecificPlaceTuningCurve.proj(dark_right_to_center_tuning_curve_id='path_specific_place_tuning_curve_id')
-> PathSpecificPlaceTuningCurve.proj(light_train_center_to_left_tuning_curve_id='path_specific_place_tuning_curve_id')
-> PathSpecificPlaceTuningCurve.proj(light_train_center_to_right_tuning_curve_id='path_specific_place_tuning_curve_id')
-> PathSpecificPlaceTuningCurve.proj(light_train_left_to_center_tuning_curve_id='path_specific_place_tuning_curve_id')
-> PathSpecificPlaceTuningCurve.proj(light_train_right_to_center_tuning_curve_id='path_specific_place_tuning_curve_id')
-> PathSpecificPlaceTuningCurve.proj(light_test_center_to_left_tuning_curve_id='path_specific_place_tuning_curve_id')
-> PathSpecificPlaceTuningCurve.proj(light_test_center_to_right_tuning_curve_id='path_specific_place_tuning_curve_id')
-> PathSpecificPlaceTuningCurve.proj(light_test_left_to_center_tuning_curve_id='path_specific_place_tuning_curve_id')
-> PathSpecificPlaceTuningCurve.proj(light_test_right_to_center_tuning_curve_id='path_specific_place_tuning_curve_id')
-> EpochIntervals.proj(dark_epoch='epoch')
-> EpochIntervals.proj(light_train_epoch='epoch')
-> EpochIntervals.proj(light_test_epoch='epoch')
-> SwapTuningCurveComparisonParameters
dark_condition: enum('dark')
light_train_condition: enum('AB', 'gray', 'BA', 'bright')
light_test_condition: enum('AB', 'gray', 'BA', 'bright')
selected_units_sha256: char(64)
position_offset_samples: bigint unsigned
speed_threshold_cm_s: double
source_tuning_curve_sha256_by_role_trajectory: longblob
source_tuning_parameters_sha256_by_role_trajectory: longblob
movement_firing_rate_table_sha256_by_role: longblob
movement_intervals_sha256_by_role: longblob
swap_tuning_curve_comparison_parameters_sha256: char(64)
swap_tuning_curve_comparison_output_rule_sha256: char(64)
"""


SWAP_TUNING_CURVE_COMPARISON_DEFINITION = """
# One empirical swapped-light summary and model-score artifact bundle.
-> SwapTuningCurveComparisonSelection
---
artifact_manifest_path: filepath@analysis
selected_units_path: filepath@analysis
summary_path: filepath@analysis
swap_tuning_curve_comparison_path: filepath@analysis
schema_version: varchar(8)
bundle_schema_version: varchar(8)
n_source_units: int unsigned
n_units: int unsigned
n_valid_units: int unsigned
analysis_status: enum('valid', 'partial_valid', 'no_units', 'no_eligible_units', 'upstream_terminal', 'no_valid_position', 'no_movement', 'no_trajectory_samples', 'no_valid_units')
selected_units_sha256: char(64)
artifact_origin: enum('computed', 'registered_existing')
runtime_v1ca1_git_commit = NULL: varchar(64)
runtime_spyglass_git_commit = NULL: varchar(64)
legacy_artifact_provenance = NULL: longblob
"""


RIPPLE_GLM_PARAMETERS_DEFINITION = """
# Named CA1-to-V1 ripple population-GLM parameters.
ripple_glm_param_name: varchar(64)
---
ripple_window_s: double
ripple_window_offset_s: double
source_window_s: double
source_window_offset_s: double
target_window_s: double
target_window_offset_s: double
source_target_windows_differ: bool
ripple_selection_mode: enum('allripples', 'deduped', 'single')
source_predictor_mode: enum('unit_vector', 'mean_activity')
min_spikes_per_ripple: double
min_ca1_spikes_per_ripple: double
n_splits: smallint unsigned
n_shuffles_ripple: int unsigned
ridge_strength: double
shuffle_seed: int
maxiter: int unsigned
tol: double
expected_detector_zscore_threshold: double
require_speed_gated: bool
"""


RIPPLE_GLM_SELECTION_DEFINITION = """
# One immutable epoch-level CA1-to-V1 ripple population-GLM selection.
ripple_glm_id: uuid
---
-> Ripples
-> RegionSortedSpikesGroup.proj(source_region_sorted_spikes_group_id='region_sorted_spikes_group_id')
-> RegionSortedSpikesGroup.proj(target_region_sorted_spikes_group_id='region_sorted_spikes_group_id')
-> RippleGLMParameters
source_region: enum('ca1')
target_region: enum('v1')
source_ripple_count: int unsigned
detector_zscore_threshold: double
speed_gated: bool
source_ripple_intervals_sha256: char(64)
ripple_provenance_sha256: char(64)
n_selected_ripples: int unsigned
selected_ripple_events_sha256: char(64)
source_sorting_group_members_sha256: char(64)
source_unit_filter_params_sha256: char(64)
source_selected_units_sha256: char(64)
source_n_units: int unsigned
target_sorting_group_members_sha256: char(64)
target_unit_filter_params_sha256: char(64)
target_selected_units_sha256: char(64)
target_n_units: int unsigned
ripple_glm_parameters_sha256: char(64)
ripple_glm_output_rule_sha256: char(64)
"""


RIPPLE_GLM_DEFINITION = """
# One CA1-to-V1 ripple population-GLM audit, summary, and NetCDF bundle.
-> RippleGLMSelection
---
artifact_manifest_path: filepath@analysis
selected_units_path: filepath@analysis
summary_path: filepath@analysis
ripple_glm_path: filepath@analysis
schema_version: varchar(8)
bundle_schema_version: varchar(8)
n_source_units: int unsigned
n_target_units: int unsigned
n_source_units_in_fit: int unsigned
n_target_units_in_fit: int unsigned
n_valid_target_units: int unsigned
n_ripples: int unsigned
selected_ripple_events_sha256: char(64)
selected_units_sha256: char(64)
analysis_status: enum('valid', 'partial_valid', 'no_source_units', 'no_target_units', 'no_ripples', 'insufficient_ripples', 'no_eligible_source_units', 'no_eligible_target_units', 'no_valid_target_units')
artifact_origin: enum('computed', 'registered_existing')
runtime_v1ca1_git_commit = NULL: varchar(64)
runtime_spyglass_git_commit = NULL: varchar(64)
legacy_artifact_provenance = NULL: longblob
"""


CROSS_REGION_XCORR_PARAMETERS_DEFINITION = """
# Named fixed parameters for ripple-restricted CA1-to-V1 cross-correlation.
cross_region_xcorr_param_name: varchar(64)
---
bin_size_s: double
max_lag_s: double
min_ripple_spikes: int unsigned
extremum_half_width_bins: smallint unsigned
norm: bool
expected_detector_zscore_threshold: double
require_speed_gated: bool
"""


CROSS_REGION_XCORR_SELECTION_DEFINITION = """
# One immutable epoch-level exact-ripple CA1-to-V1 xcorr selection.
cross_region_xcorr_id: uuid
---
-> Ripples
-> RegionSortedSpikesGroup.proj(source_region_sorted_spikes_group_id='region_sorted_spikes_group_id')
-> RegionSortedSpikesGroup.proj(target_region_sorted_spikes_group_id='region_sorted_spikes_group_id')
-> CrossRegionXCorrParameters
source_region: enum('ca1')
target_region: enum('v1')
source_ripple_count: int unsigned
detector_zscore_threshold: double
speed_gated: bool
selected_ripple_intervals_sha256: char(64)
ripple_provenance_sha256: char(64)
source_sorting_group_members_sha256: char(64)
source_unit_filter_params_sha256: char(64)
source_selected_units_sha256: char(64)
source_n_units: int unsigned
target_sorting_group_members_sha256: char(64)
target_unit_filter_params_sha256: char(64)
target_selected_units_sha256: char(64)
target_n_units: int unsigned
cross_region_xcorr_parameters_sha256: char(64)
cross_region_xcorr_output_rule_sha256: char(64)
"""


CROSS_REGION_XCORR_DEFINITION = """
# One exact-ripple CA1-to-V1 xcorr audit, pair summary, and NetCDF bundle.
-> CrossRegionXCorrSelection
---
artifact_manifest_path: filepath@analysis
ca1_units_path: filepath@analysis
v1_units_path: filepath@analysis
summary_path: filepath@analysis
cross_region_xcorr_path: filepath@analysis
schema_version: varchar(8)
bundle_schema_version: varchar(8)
n_ripples: int unsigned
ripple_duration_s: double
n_ca1_units: int unsigned
n_v1_units: int unsigned
n_ca1_units_in_xcorr: int unsigned
n_v1_units_in_xcorr: int unsigned
n_pairs: int unsigned
n_valid_pairs: int unsigned
selected_ripple_intervals_sha256: char(64)
ca1_units_sha256: char(64)
v1_units_sha256: char(64)
summary_sha256: char(64)
analysis_status: enum('valid', 'partial_valid', 'no_valid_pairs', 'no_ripples', 'no_ca1_units', 'no_v1_units', 'no_eligible_ca1_units', 'no_eligible_v1_units')
artifact_origin: enum('computed', 'registered_existing')
runtime_v1ca1_git_commit = NULL: varchar(64)
runtime_spyglass_git_commit = NULL: varchar(64)
legacy_artifact_provenance = NULL: longblob
"""


RIPPLE_DECODING_COMPARISON_PARAMETERS_DEFINITION = """
# Named fixed parameters for CA1/V1 ripple decoding-state comparison.
ripple_decoding_comparison_param_name: varchar(64)
---
decode_bin_size_s: double
spatial_bin_size_cm: double
tuning_smoothing_sigma_bins: double
ca1_min_movement_rate_hz: double
v1_min_movement_rate_hz: double
n_shuffles: int unsigned
shuffle_seed: int
expected_detector_zscore_threshold: double
require_speed_gated: bool
"""


RIPPLE_DECODING_COMPARISON_SELECTION_DEFINITION = """
# One immutable session/train/decode/representation ripple-decoding selection.
ripple_decoding_comparison_id: uuid
---
-> Ripples.proj(decode_epoch='epoch')
-> Position.proj(train_epoch='epoch', train_position_series_name='position_series_name')
-> RegionSortedSpikesGroup.proj(ca1_region_sorted_spikes_group_id='region_sorted_spikes_group_id')
-> RegionSortedSpikesGroup.proj(v1_region_sorted_spikes_group_id='region_sorted_spikes_group_id')
-> MovementFiringRate.proj(ca1_movement_firing_rate_id='movement_firing_rate_id')
-> MovementFiringRate.proj(v1_movement_firing_rate_id='movement_firing_rate_id')
-> TrajectoryIntervals.proj(train_epoch='epoch', center_to_left_trajectory_type='trajectory_type')
-> TrajectoryIntervals.proj(train_epoch='epoch', center_to_right_trajectory_type='trajectory_type')
-> TrajectoryIntervals.proj(train_epoch='epoch', left_to_center_trajectory_type='trajectory_type')
-> TrajectoryIntervals.proj(train_epoch='epoch', right_to_center_trajectory_type='trajectory_type')
-> WTrackGraph.proj(center_to_left_configuration_name='configuration_name')
-> WTrackGraph.proj(center_to_right_configuration_name='configuration_name')
-> WTrackGraph.proj(left_to_center_configuration_name='configuration_name')
-> WTrackGraph.proj(right_to_center_configuration_name='configuration_name')
-> RippleDecodingComparisonParameters
representation: enum('path_specific_place', 'dpp')
source_region: enum('ca1')
target_region: enum('v1')
source_ripple_count: int unsigned
detector_zscore_threshold: double
speed_gated: bool
selected_ripple_intervals_sha256: char(64)
ripple_provenance_sha256: char(64)
ca1_sorting_group_members_sha256: char(64)
ca1_unit_filter_params_sha256: char(64)
ca1_selected_units_sha256: char(64)
ca1_n_units: int unsigned
v1_sorting_group_members_sha256: char(64)
v1_unit_filter_params_sha256: char(64)
v1_selected_units_sha256: char(64)
v1_n_units: int unsigned
movement_param_name: varchar(64)
movement_parameters_sha256: char(64)
movement_speed_threshold_cm_s: double
movement_speed_sigma_s: double
ca1_movement_firing_rate_sha256: char(64)
ca1_movement_intervals_sha256: char(64)
ca1_movement_rates_sha256: char(64)
ca1_movement_support_sha256: char(64)
ca1_movement_analysis_status: enum('valid', 'no_movement', 'no_valid_position')
v1_movement_firing_rate_sha256: char(64)
v1_movement_intervals_sha256: char(64)
v1_movement_rates_sha256: char(64)
v1_movement_support_sha256: char(64)
v1_movement_analysis_status: enum('valid', 'no_movement', 'no_valid_position')
position_snapshot_sha256: char(64)
trajectory_intervals_sha256_by_type: longblob
graph_rows_sha256_by_trajectory: longblob
graph_policy_sha256: char(64)
ripple_decoding_comparison_parameters_sha256: char(64)
ripple_decoding_comparison_output_rule_sha256: char(64)
"""


RIPPLE_DECODING_COMPARISON_DEFINITION = """
# One categorical CA1/V1 ripple decoding comparison artifact bundle.
-> RippleDecodingComparisonSelection
---
artifact_manifest_path: filepath@analysis
selected_units_path: filepath@analysis
ripple_qc_path: filepath@analysis
ripple_metrics_path: filepath@analysis
epoch_summary_path: filepath@analysis
ripple_decoding_comparison_path: filepath@analysis
ca1_decoded_path: filepath@analysis
v1_decoded_path: filepath@analysis
schema_version: varchar(8)
bundle_schema_version: varchar(8)
n_ripple_events_input: int unsigned
n_ripples: int unsigned
n_ripple_bins: bigint unsigned
n_ca1_units: int unsigned
n_v1_units: int unsigned
n_ca1_units_in_decoder: int unsigned
n_v1_units_in_decoder: int unsigned
selected_ripple_intervals_sha256: char(64)
graph_policy_sha256: char(64)
selected_units_sha256: char(64)
ripple_qc_sha256: char(64)
ripple_metrics_sha256: char(64)
epoch_summary_sha256: char(64)
analysis_status: enum('valid', 'partial_valid', 'no_ripples', 'no_ca1_units', 'no_v1_units', 'no_eligible_ca1_units', 'no_eligible_v1_units', 'no_train_movement', 'no_train_trajectory_samples', 'no_common_decoded_bins', 'no_valid_metrics')
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


MANUSCRIPT_EPOCH_MOTOR_BEHAVIOR_PARAMETERS = MappingProxyType(
    {
        "epoch_motor_behavior_param_name": "manuscript_4cm",
        "progression_bin_size_cm": 4.0,
    }
)


LEGACY_TUNING_CURVE_PARAMETERS = MappingProxyType(
    {
        "tuning_curve_param_name": "legacy_4cm_unsmoothed",
        "binning_mode": "bin_size_cm",
        "place_bin_size_cm": 4.0,
        "position_bin_count": None,
        "gaussian_smoothing_sigma_bins": 0.0,
    }
)


FIGURE_1D_TUNING_CURVE_PARAMETERS = MappingProxyType(
    {
        "tuning_curve_param_name": "figure1d_50bin_sigma1p5",
        "binning_mode": "bin_count",
        "place_bin_size_cm": None,
        "position_bin_count": 50,
        "gaussian_smoothing_sigma_bins": 1.5,
    }
)


TUNING_CURVE_PARAMETER_PRESETS = (
    LEGACY_TUNING_CURVE_PARAMETERS,
    FIGURE_1D_TUNING_CURVE_PARAMETERS,
)


CORRELATION_TUNING_SIMILARITY_PARAMETERS = MappingProxyType(
    {
        "tuning_similarity_param_name": "correlation",
        "similarity_metric": "correlation",
    }
)


ABSOLUTE_OVERLAP_TUNING_SIMILARITY_PARAMETERS = MappingProxyType(
    {
        "tuning_similarity_param_name": "absolute_overlap",
        "similarity_metric": "absolute_overlap",
    }
)


SHAPE_OVERLAP_TUNING_SIMILARITY_PARAMETERS = MappingProxyType(
    {
        "tuning_similarity_param_name": "shape_overlap",
        "similarity_metric": "shape_overlap",
    }
)


TUNING_SIMILARITY_PARAMETER_PRESETS = (
    CORRELATION_TUNING_SIMILARITY_PARAMETERS,
    ABSOLUTE_OVERLAP_TUNING_SIMILARITY_PARAMETERS,
    SHAPE_OVERLAP_TUNING_SIMILARITY_PARAMETERS,
)


MANUSCRIPT_DPP_ENCODING_COMPARISON_PARAMETERS = MappingProxyType(
    {
        "dpp_encoding_comparison_param_name": "manuscript_5fold_50ms_4cm_sigma1",
        "n_folds": 5,
        "evaluation_bin_size_s": 0.05,
        "spatial_bin_size_cm": 4.0,
        "gaussian_smoothing_sigma_bins": 1.0,
        "random_seed": 47,
        "minimum_movement_firing_rate_hz": 0.5,
        "minimum_stability_correlation": 0.5,
    }
)


DPP_ENCODING_COMPARISON_PARAMETER_PRESETS = (
    MANUSCRIPT_DPP_ENCODING_COMPARISON_PARAMETERS,
)


PATH_PROGRESSION_DECODING_ELIGIBILITY_RULE = MappingProxyType(
    {
        "version": 1,
        "cohort_policy": "target_and_cohort_intersection",
        "movement_operator": "greater_than_or_equal",
        "stability_aggregation": "at_least_one_trajectory",
        "stability_operator": "greater_than_or_equal",
        "null_stability_threshold": "disabled",
    }
)


PATH_PROGRESSION_DECODING_OUTPUT_RULE = MappingProxyType(
    {
        "version": 1,
        "coordinate_unit": "normalized_path_progression",
        "time_unit": "s",
        "time_reference": "augmented_nwb_ephys_timestamps",
        "error_mode": "signed",
        "error_summary": "median_iqr",
        "min_bin_count": 5,
    }
)


MANUSCRIPT_PATH_PROGRESSION_DECODING_PARAMETERS = MappingProxyType(
    {
        "path_progression_decoding_param_name": (
            "manuscript_20ms_window4_4cm_mfr0p5"
        ),
        "decoding_bin_size_s": 0.02,
        "sliding_window_size_bins": 4,
        "spatial_bin_size_cm": 4.0,
        "minimum_movement_firing_rate_hz": 0.5,
        # The manuscript decoding artifacts did not apply a stability filter.
        "minimum_stability_correlation": None,
    }
)


PATH_PROGRESSION_DECODING_PARAMETER_PRESETS = (
    MANUSCRIPT_PATH_PROGRESSION_DECODING_PARAMETERS,
)


PATH_SPECIFIC_PLACE_DECODING_OUTPUT_RULE = MappingProxyType(
    {
        "version": 1,
        "coordinate": "concatenated_path_specific_linear_position",
        "coordinate_unit": "cm",
        "trajectory_order": (
            "center_to_left",
            "left_to_center",
            "center_to_right",
            "right_to_center",
        ),
        "path_orientation": "from_center",
        "unit_policy": "all_region_sorted_spikes_group_units",
        "cross_validation": "lap_wise_kfold_per_trajectory_then_pooled",
        "time_unit": "s",
        "time_reference": "augmented_nwb_ephys_timestamps",
        "error_mode": "signed",
        "error_summary": "median_iqr",
        "min_bin_count": 5,
    }
)


MANUSCRIPT_PATH_SPECIFIC_PLACE_DECODING_PARAMETERS = MappingProxyType(
    {
        "path_specific_place_decoding_param_name": (
            "manuscript_5fold_20ms_window4_4cm_all_units"
        ),
        "n_folds": 5,
        "decoding_bin_size_s": 0.02,
        "sliding_window_size_bins": 4,
        "spatial_bin_size_cm": 4.0,
        "random_seed": 47,
    }
)


PATH_SPECIFIC_PLACE_DECODING_PARAMETER_PRESETS = (
    MANUSCRIPT_PATH_SPECIFIC_PLACE_DECODING_PARAMETERS,
)


MOTOR_ENCODING_COMPARISON_MODEL_SPEC = MappingProxyType(
    {
        "motor": "strict motor covariates only",
        "motor_tp": (
            "motor covariates plus TP group offset and TP group-specific "
            "spline fields"
        ),
        "tp_only": "TP group offset plus TP group-specific spline fields",
        "motor_place": (
            "motor covariates plus trajectory offset and "
            "trajectory-specific place fields"
        ),
        "place_only": "trajectory offset plus trajectory-specific place fields",
        "motor_generalized_place": (
            "motor covariates plus one generalized full-W place spline field"
        ),
        "generalized_place_only": (
            "one generalized full-W place spline field"
        ),
        "motor_generalized_task_progression": (
            "motor covariates plus one generalized task-progression spline field"
        ),
        "generalized_task_progression_only": (
            "one generalized task-progression spline field"
        ),
    }
)


MOTOR_ENCODING_COMPARISON_OUTPUT_RULE = MappingProxyType(
    {
        "version": 1,
        "model_names": tuple(MOTOR_ENCODING_COMPARISON_MODEL_SPEC),
        "cross_validation": "nested_lap_level_by_trajectory_movement_only",
        "hyperparameter_selection": (
            "per_model_population_median_unit_information_bits_per_spike"
        ),
        "unit_fit_failure_policy": (
            "retry_nonfinite_or_failed_population_units_independently"
        ),
        "unit_selection": (
            "strict_greater_than_movement_firing_rate_threshold"
        ),
        "full_refit_role": "visualization_not_heldout_inference",
        "primary_position_role": (
            "speed_acceleration_and_track_linearization"
        ),
        "orientation_reference_role": (
            "primary_minus_reference_head_direction"
        ),
        "time_unit": "s",
        "time_reference": "augmented_nwb_ephys_timestamps",
        "position_unit": "cm",
    }
)


MANUSCRIPT_V1_MOTOR_ENCODING_COMPARISON_PARAMETERS = MappingProxyType(
    {
        "motor_encoding_comparison_param_name": (
            "manuscript_v1_nested5x3_50ms_zscore"
        ),
        "evaluation_bin_size_s": 0.05,
        "outer_n_folds": 5,
        "inner_n_folds": 3,
        "random_seed": 0,
        "ridge_values": (1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6),
        "spatial_bin_sizes_cm": (2.0, 4.0, 8.0),
        "minimum_movement_firing_rate_hz": 0.5,
        "motor_feature_mode": "zscore",
        "motor_zscore_eps": 1e-12,
        "motor_spline_n_basis": 5,
        "motor_spline_order": 4,
        "position_spline_order": 4,
        "speed_smoothing_sigma_s": 0.1,
        "generalized_place_branch_gap_cm": 15.0,
    }
)


MANUSCRIPT_CA1_MOTOR_ENCODING_COMPARISON_PARAMETERS = MappingProxyType(
    {
        **MANUSCRIPT_V1_MOTOR_ENCODING_COMPARISON_PARAMETERS,
        "motor_encoding_comparison_param_name": (
            "manuscript_ca1_nested5x3_50ms_zscore"
        ),
        "minimum_movement_firing_rate_hz": 0.0,
    }
)


MOTOR_ENCODING_COMPARISON_PARAMETER_PRESETS = (
    MANUSCRIPT_V1_MOTOR_ENCODING_COMPARISON_PARAMETERS,
    MANUSCRIPT_CA1_MOTOR_ENCODING_COMPARISON_PARAMETERS,
)


DARK_LIGHT_GLM_OUTPUT_RULE = MappingProxyType(
    {
        "version": 1,
        "model_names": (
            "visual",
            "task_segment_bump",
            "task_segment_scalar",
            "task_dense_gain",
        ),
        "schema_by_basis_candidate_mode": (
            ("spatial_bin_size_cm", "5"),
            ("n_splines", "4"),
        ),
        "cross_validation": "lap_level_by_trajectory_movement_only",
        "shared_hyperparameter_selection": (
            "visual_model_median_trajectory_unit_information_bits_per_spike"
        ),
        "comparison_model_candidate_policy": (
            "selected_visual_bin_and_basis_only"
        ),
        "ridge_selection": "per_model_median_trajectory_unit_score",
        "unit_selection": (
            "strict_greater_than_dark_and_light_movement_firing_rate_thresholds"
        ),
        "unit_fit_failure_policy": (
            "retry_nonfinite_or_failed_population_units_independently"
        ),
        "unit_fit_qc": (
            "finite_intercept_for_all_four_models_and_four_trajectories"
        ),
        "speed_smoothing": "gaussian_sigma_seconds_from_frozen_parameter",
        "time_unit": "s",
        "time_reference": "augmented_nwb_ephys_timestamps",
        "position_unit": "cm",
        "selected_model_storage": "one_netcdf_per_model",
    }
)


CURRENT_V5_V1_DARK_LIGHT_GLM_PARAMETERS = MappingProxyType(
    {
        "dark_light_glm_param_name": "current_v5_v1",
        "basis_candidate_mode": "spatial_bin_size_cm",
        "basis_candidates": (2.0, 4.0, 8.0),
        "bin_sizes_s": (0.02, 0.05),
        "ridges": (1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6),
        "n_folds": 5,
        "random_seed": 47,
        "spline_order": 4,
        "min_dark_firing_rate_hz": 0.5,
        "min_light_firing_rate_hz": 0.5,
        "use_speed": True,
        "speed_feature_mode": "linear",
        "n_splines_speed": 5,
        "spline_order_speed": 4,
        "speed_bounds": None,
        "speed_smoothing_sigma_s": 0.1,
    }
)


CURRENT_V5_CA1_DARK_LIGHT_GLM_PARAMETERS = MappingProxyType(
    {
        **CURRENT_V5_V1_DARK_LIGHT_GLM_PARAMETERS,
        "dark_light_glm_param_name": "current_v5_ca1",
        "min_dark_firing_rate_hz": 0.0,
        "min_light_firing_rate_hz": 0.0,
    }
)


LEGACY_V4_V1_DARK_LIGHT_GLM_PARAMETERS = MappingProxyType(
    {
        **CURRENT_V5_V1_DARK_LIGHT_GLM_PARAMETERS,
        "dark_light_glm_param_name": "legacy_v4_v1",
        "basis_candidate_mode": "n_splines",
        "basis_candidates": (25, 40, 60),
    }
)


LEGACY_V4_CA1_DARK_LIGHT_GLM_PARAMETERS = MappingProxyType(
    {
        **CURRENT_V5_CA1_DARK_LIGHT_GLM_PARAMETERS,
        "dark_light_glm_param_name": "legacy_v4_ca1",
        "basis_candidate_mode": "n_splines",
        "basis_candidates": (25, 40, 60),
    }
)


DARK_LIGHT_GLM_PARAMETER_PRESETS = (
    CURRENT_V5_V1_DARK_LIGHT_GLM_PARAMETERS,
    CURRENT_V5_CA1_DARK_LIGHT_GLM_PARAMETERS,
    LEGACY_V4_V1_DARK_LIGHT_GLM_PARAMETERS,
    LEGACY_V4_CA1_DARK_LIGHT_GLM_PARAMETERS,
)


SWAP_GLM_OUTPUT_RULE = MappingProxyType(
    {
        "version": 2,
        "models": (
            "visual",
            "task_segment_bump",
            "task_segment_scalar",
            "task_dense_gain",
            "dark",
        ),
        "derived_model_sources": {"dark": "task_segment_bump"},
        "swap_configuration": {
            "center_to_left": {
                "source_trajectory": "center_to_right",
                "segment_index": 2,
            },
            "center_to_right": {
                "source_trajectory": "center_to_left",
                "segment_index": 2,
            },
            "left_to_center": {
                "source_trajectory": "right_to_center",
                "segment_index": 0,
            },
            "right_to_center": {
                "source_trajectory": "left_to_center",
                "segment_index": 0,
            },
        },
        "fit_source": "exact_selected_dark_light_glm_artifact",
        "evaluation_epoch": "held_out_light_movement_laps",
        "primary_metric": (
            "test_light_swapped_segment_swapped_delta_model_minus_visual_"
            "raw_ll_bits_per_spike"
        ),
        "unit_policy": (
            "all_upstream_dark_light_selected_units_in_saved_order"
        ),
        "unit_failure_policy": (
            "retain_all_units_and_isolate_nonfinite_scores_by_unit_model_trajectory"
        ),
        "unit_validity_policy": (
            "upstream_valid_glm_fit_and_all_expected_primary_scores_finite"
        ),
        "runtime_unit_key_policy": (
            "persistent_identity_aligned_native_tsgroup_keys_with_canonical_output_ids"
        ),
        "trajectory_support_policy": (
            "all_or_none_terminal_if_any_path_has_no_movement_bins"
        ),
        "movement_terminal_status_policy": (
            "selected_movement_firing_rate_status_precedes_interval_fallback"
        ),
        "legacy_registration_policy": (
            "normalize_verified_schema4_or_schema6_then_exact_nwb_rescore_without_refit"
        ),
        "legacy_comparison_policy": (
            "all_scientific_coordinates_and_variables_tight_equal"
        ),
        "legacy_missing_derived_model_policy": (
            "schema4_four_source_models_compare_all_available_then_rescore_dark_"
            "from_verified_task_segment_bump"
        ),
        "legacy_preprocessing_provenance_policy": (
            "historical_schema4_or_schema6_position_offset_and_speed_threshold_"
            "must_match_selected_nwb_inputs"
        ),
        "time_unit": "s",
        "time_reference": "augmented_nwb_ephys_timestamps",
        "position_unit": "cm",
    }
)


DEFAULT_SWAP_GLM_PARAMETERS = MappingProxyType(
    {
        "swap_glm_param_name": "default",
        "swap_light_offset": False,
        "observed_spatial_bin_size_cm": 4.0,
    }
)


# Swap scoring itself has one legacy-compatible rule. Current-v5 versus
# normalized-legacy-v4 provenance is frozen by the selected DarkLightGLM row.
SWAP_GLM_PARAMETER_PRESETS = (DEFAULT_SWAP_GLM_PARAMETERS,)


SWAP_TUNING_CURVE_COMPARISON_OUTPUT_RULE = MappingProxyType(
    {
        "version": 1,
        "models": (
            "empirical_visual",
            "empirical_dark",
            "empirical_pointwise_multiplicative_ratio",
            "empirical_segment_multiplicative_ratio",
            "empirical_pointwise_additive_delta",
            "empirical_segment_additive_delta",
        ),
        "swap_configuration": {
            "center_to_left": {
                "source_trajectory": "center_to_right",
                "segment_index": 2,
            },
            "center_to_right": {
                "source_trajectory": "center_to_left",
                "segment_index": 2,
            },
            "left_to_center": {
                "source_trajectory": "right_to_center",
                "segment_index": 0,
            },
            "right_to_center": {
                "source_trajectory": "left_to_center",
                "segment_index": 0,
            },
        },
        "empirical_model_formulas": {
            "empirical_visual": "other_light",
            "empirical_dark": "same_dark",
            "empirical_pointwise_multiplicative_ratio": (
                "same_dark * other_light / max(other_dark, epsilon)"
            ),
            "empirical_segment_multiplicative_ratio": (
                "same_dark * sum(other_light_in_swap_segment) / "
                "max(sum(other_dark_in_swap_segment), epsilon)"
            ),
            "empirical_pointwise_additive_delta": (
                "same_dark + other_light - other_dark"
            ),
            "empirical_segment_additive_delta": (
                "same_dark + mean(other_light_in_swap_segment - "
                "other_dark_in_swap_segment)"
            ),
        },
        "training_tuning_source": (
            "twelve_all_trial_path_specific_place_tuning_curves"
        ),
        "training_tuning_input_policy": (
            "four_cm_unsmoothed_curves_interpolate_nans_then_gaussian_smooth"
        ),
        "all_nan_tuning_fallback": (
            "trajectory_spike_count_divided_by_movement_support_duration"
        ),
        "evaluation_scope": "heldout_light_swapped_segment_movement_laps",
        "evaluation_bin_size_unit": "s",
        "eligibility_policy": (
            "strict_dark_and_light_train_epoch_wide_movement_firing_rate_thresholds"
        ),
        "movement_interval_provenance_policy": (
            "exact_artifact_sha256_frozen_for_dark_light_train_and_light_test"
        ),
        "heldout_firing_rate_filter": False,
        "trajectory_support_policy": (
            "all_or_none_terminal_if_any_heldout_path_has_no_scoring_bins"
        ),
        "unit_failure_policy": (
            "retain_and_isolate_nonfinite_scores_per_unit"
        ),
        "unit_audit_policy": (
            "retain_all_upstream_units_with_explicit_eligibility"
        ),
        "runtime_unit_key_policy": (
            "persistent_identity_aligned_native_tsgroup_keys_with_stable_output_ids"
        ),
        "legacy_registration_policy": (
            "imported_spike_sorting_only_exact_nwb_reconstruction_and_rescore"
        ),
        "legacy_preprocessing_policy": (
            "position_offset_10_and_speed_threshold_4_cm_s_required"
        ),
        "legacy_comparison_policy": "all_scientific_values_tight_equal",
        "time_unit": "s",
        "time_reference": "augmented_nwb_ephys_timestamps",
        "position_unit": "cm",
    }
)


MANUSCRIPT_V1_SWAP_TUNING_CURVE_COMPARISON_PARAMETERS = MappingProxyType(
    {
        "swap_tuning_curve_comparison_param_name": "manuscript_v1",
        "evaluation_bin_size_s": 0.05,
        "gaussian_smoothing_sigma_bins": 1.0,
        "min_dark_firing_rate_hz": 0.5,
        "min_light_firing_rate_hz": 0.5,
    }
)


MANUSCRIPT_CA1_SWAP_TUNING_CURVE_COMPARISON_PARAMETERS = MappingProxyType(
    {
        **MANUSCRIPT_V1_SWAP_TUNING_CURVE_COMPARISON_PARAMETERS,
        "swap_tuning_curve_comparison_param_name": "manuscript_ca1",
        "min_dark_firing_rate_hz": 0.0,
        "min_light_firing_rate_hz": 0.0,
    }
)


SWAP_TUNING_CURVE_COMPARISON_PARAMETER_PRESETS = (
    MANUSCRIPT_V1_SWAP_TUNING_CURVE_COMPARISON_PARAMETERS,
    MANUSCRIPT_CA1_SWAP_TUNING_CURVE_COMPARISON_PARAMETERS,
)


RIPPLE_GLM_OUTPUT_RULE = MappingProxyType(
    {
        "version": 1,
        "model_direction": "ca1_to_v1",
        "model_family": "ridge_regularized_poisson_population_glm",
        "cv_policy": "contiguous_unshuffled_kfold_over_selected_ripples",
        "shuffle_policy": (
            "independently_permute_each_target_unit_training_response_and_refit"
        ),
        "ripple_selection_modes": ("allripples", "deduped", "single"),
        "source_predictor_modes": ("unit_vector", "mean_activity"),
        "ripple_input_policy": (
            "detector_zscore_threshold_2_and_speed_gated_events_required"
        ),
        "source_preprocessing": (
            "drop_near_constant_train_features_then_zscore_divide_by_sqrt_n_and_clip_10"
        ),
        "source_coefficient_space": "full_data_preprocessed_predictor",
        "unit_filter_policy": (
            "inclusive_mean_spike_count_per_selected_ripple_threshold"
        ),
        "unit_audit_policy": "retain_all_source_and_target_input_units",
        "unit_failure_policy": (
            "shared_population_fit_preserved_and_nonfinite_target_metrics_isolated_in_audit"
        ),
        "terminal_artifact_policy": (
            "explicit_for_no_units_no_or_insufficient_ripples_and_no_eligible_units"
        ),
        "legacy_registration_policy": (
            "imported_sorting_identity_resolved_then_verify_nwb_event_windows_target_counts_fold_layout_metric_self_consistency_and_coefficient_axes_shape_finiteness"
        ),
        "time_unit": "s",
        "time_reference": "augmented_nwb_ephys_timestamps",
    }
)


MANUSCRIPT_UNIT_VECTOR_RIPPLE_GLM_PARAMETERS = MappingProxyType(
    {
        "ripple_glm_param_name": "manuscript_single_unit_vector",
        "ripple_window_s": 0.2,
        "ripple_window_offset_s": 0.0,
        "source_window_s": 0.2,
        "source_window_offset_s": 0.0,
        "target_window_s": 0.2,
        "target_window_offset_s": 0.0,
        "source_target_windows_differ": False,
        "ripple_selection_mode": "single",
        "source_predictor_mode": "unit_vector",
        "min_spikes_per_ripple": 0.1,
        "min_ca1_spikes_per_ripple": 0.0,
        "n_splits": 5,
        "n_shuffles_ripple": 100,
        "ridge_strength": 0.1,
        "shuffle_seed": 45,
        "maxiter": 6000,
        "tol": 1e-7,
        "expected_detector_zscore_threshold": 2.0,
        "require_speed_gated": True,
    }
)


MANUSCRIPT_MEAN_ACTIVITY_RIPPLE_GLM_PARAMETERS = MappingProxyType(
    {
        **MANUSCRIPT_UNIT_VECTOR_RIPPLE_GLM_PARAMETERS,
        "ripple_glm_param_name": "manuscript_single_mean_activity",
        "source_predictor_mode": "mean_activity",
    }
)


RIPPLE_GLM_PARAMETER_PRESETS = (
    MANUSCRIPT_UNIT_VECTOR_RIPPLE_GLM_PARAMETERS,
    MANUSCRIPT_MEAN_ACTIVITY_RIPPLE_GLM_PARAMETERS,
)


CROSS_REGION_XCORR_OUTPUT_RULE = MappingProxyType(
    {
        "version": 1,
        "direction": "ca1_reference_to_v1_target",
        "interval_scope": (
            "exact_selected_detected_ripple_intervals_for_one_epoch"
        ),
        "epoch_pooling": False,
        "fixed_ripple_windows": False,
        "state_intervals": False,
        "normalization": "pynapple_target_rate_norm_true",
        "bin_size_s": 0.005,
        "max_lag_s": 0.5,
        "minimum_ripple_spikes_per_unit": 30,
        "extremum_half_width_bins": 1,
        "detector_event_policy": "zscore_threshold_2_and_speed_gated",
        "unit_audit_policy": "retain_all_ca1_and_v1_input_units",
        "unit_identity_policy": (
            "stable_sorting_identity_with_runtime_group_key_audit"
        ),
        "terminal_artifact_policy": "explicit_empty_and_partial_statuses",
        "legacy_registration_policy": (
            "imported_spike_sorting_identity_resolution_and_exact_nwb_recomputation"
        ),
        "legacy_comparison_policy": (
            "all_four_scientific_artifacts_tight_equal"
        ),
        "time_unit": "s",
        "time_reference": "augmented_nwb_ephys_timestamps",
    }
)


MANUSCRIPT_CROSS_REGION_XCORR_PARAMETERS = MappingProxyType(
    {
        "cross_region_xcorr_param_name": (
            "manuscript_ripple_5ms_lag500ms_min30"
        ),
        "bin_size_s": 0.005,
        "max_lag_s": 0.5,
        "min_ripple_spikes": 30,
        "extremum_half_width_bins": 1,
        "norm": True,
        "expected_detector_zscore_threshold": 2.0,
        "require_speed_gated": True,
    }
)


CROSS_REGION_XCORR_PARAMETER_PRESETS = (
    MANUSCRIPT_CROSS_REGION_XCORR_PARAMETERS,
)


RIPPLE_DECODING_COMPARISON_OUTPUT_RULE = MappingProxyType(
    {
        "version": 1,
        "comparison_direction": "ca1_vs_v1_decoded_state",
        "result_granularity": (
            "session_x_train_epoch_x_decode_epoch_x_representation"
        ),
        "representations": ("path_specific_place", "dpp"),
        "path_specific_place_coordinate": (
            "four_concatenated_trajectory_specific_center_to_arm_blocks_cm"
        ),
        "dpp_coordinate": (
            "two_same_turn_blocks_left_then_right_natural_direction_progression"
        ),
        "trajectory_order": (
            "center_to_left",
            "left_to_center",
            "center_to_right",
            "right_to_center",
        ),
        "turn_group_by_trajectory": {
            "center_to_left": "left",
            "right_to_center": "left",
            "center_to_right": "right",
            "left_to_center": "right",
        },
        "physical_arm_by_trajectory": {
            "center_to_left": "left",
            "left_to_center": "left",
            "center_to_right": "right",
            "right_to_center": "right",
        },
        "arm_boundary_policy": (
            "graph_derived_midpoint_of_penultimate_center_to_arm_edge"
        ),
        "categorical_scoring_schemes": (
            "trajectory",
            "turn_group",
            "arm_identity",
        ),
        "path_specific_place_schemes": (
            "trajectory",
            "turn_group",
            "arm_identity",
        ),
        "dpp_schemes": ("turn_group",),
        "continuous_scoring": False,
        "unit_filter_policy": (
            "inclusive_train_epoch_movement_firing_rate_threshold_by_region"
        ),
        "training_policy": "movement_supported_train_epoch_tuning_curves",
        "decoding_policy": (
            "independent_ca1_and_v1_bayesian_decoders_per_exact_detected_ripple"
        ),
        "alignment_policy": (
            "common_ripple_source_index_and_rounded_bin_time"
        ),
        "shuffle_policy": (
            "deranged_v1_ripple_blocks_within_equal_bin_count_groups"
        ),
        "ripple_input_policy": (
            "one_decode_epoch_detector_zscore_threshold_2_speed_gated_exact_intervals"
        ),
        "graph_policy": (
            "four_directional_path_wtrack_graphs_in_centimeters"
        ),
        "terminal_policy": "explicit_expected_terminal_and_partial_artifacts",
        "unexpected_error_policy": "raise",
        "legacy_registration_policy": (
            "imported_sorting_identity_resolution_exact_nwb_redecode_rescore_and_"
            "complete_five_file_comparison_under_corrected_graph_scoring"
        ),
        "legacy_arm_bug_policy": (
            "reject_turn_group_labeled_inbound_arm_outputs"
        ),
        "time_unit": "s",
        "time_reference": "augmented_nwb_ephys_timestamps",
    }
)


MANUSCRIPT_RIPPLE_DECODING_COMPARISON_PARAMETERS = MappingProxyType(
    {
        "ripple_decoding_comparison_param_name": "manuscript_categorical",
        "decode_bin_size_s": 0.002,
        "spatial_bin_size_cm": 4.0,
        "tuning_smoothing_sigma_bins": 0.0,
        "ca1_min_movement_rate_hz": 0.0,
        "v1_min_movement_rate_hz": 0.5,
        "n_shuffles": 100,
        "shuffle_seed": 45,
        "expected_detector_zscore_threshold": 2.0,
        "require_speed_gated": True,
    }
)


RIPPLE_DECODING_COMPARISON_PARAMETER_PRESETS = (
    MANUSCRIPT_RIPPLE_DECODING_COMPARISON_PARAMETERS,
)


TABLE_DEFINITIONS = MappingProxyType(
    {
        "epoch_intervals": EPOCH_INTERVALS_DEFINITION,
        "trajectory_intervals": TRAJECTORY_INTERVALS_DEFINITION,
        "ripples": RIPPLES_DEFINITION,
        "position": POSITION_DEFINITION,
        "wtrack_graph": WTRACK_GRAPH_DEFINITION,
        "spike_sorting_figurl": SPIKE_SORTING_FIGURL_DEFINITION,
        "region_sorted_spikes_group": REGION_SORTED_SPIKES_GROUP_DEFINITION,
        "movement_parameters": MOVEMENT_PARAMETERS_DEFINITION,
        "epoch_motor_behavior_parameters": (
            EPOCH_MOTOR_BEHAVIOR_PARAMETERS_DEFINITION
        ),
        "epoch_motor_behavior_selection": (
            EPOCH_MOTOR_BEHAVIOR_SELECTION_DEFINITION
        ),
        "epoch_motor_behavior": EPOCH_MOTOR_BEHAVIOR_DEFINITION,
        "movement_firing_rate_selection": (
            MOVEMENT_FIRING_RATE_SELECTION_DEFINITION
        ),
        "movement_firing_rate": MOVEMENT_FIRING_RATE_DEFINITION,
        "ripple_modulation_parameters": RIPPLE_MODULATION_PARAMETERS_DEFINITION,
        "ripple_modulation_selection": RIPPLE_MODULATION_SELECTION_DEFINITION,
        "ripple_modulation": RIPPLE_MODULATION_DEFINITION,
        "tuning_curve_parameters": TUNING_CURVE_PARAMETERS_DEFINITION,
        "tuning_similarity_parameters": TUNING_SIMILARITY_PARAMETERS_DEFINITION,
        "path_specific_place_tuning_curve_selection": (
            PATH_SPECIFIC_PLACE_TUNING_CURVE_SELECTION_DEFINITION
        ),
        "path_specific_place_tuning_curve": (
            PATH_SPECIFIC_PLACE_TUNING_CURVE_DEFINITION
        ),
        "path_specific_place_tuning_similarity_selection": (
            PATH_SPECIFIC_PLACE_TUNING_SIMILARITY_SELECTION_DEFINITION
        ),
        "path_specific_place_tuning_similarity": (
            PATH_SPECIFIC_PLACE_TUNING_SIMILARITY_DEFINITION
        ),
        "dpp_tuning_curve_selection": DPP_TUNING_CURVE_SELECTION_DEFINITION,
        "dpp_tuning_curve": DPP_TUNING_CURVE_DEFINITION,
        "path_specific_place_stability_selection": (
            PATH_SPECIFIC_PLACE_STABILITY_SELECTION_DEFINITION
        ),
        "path_specific_place_stability": PATH_SPECIFIC_PLACE_STABILITY_DEFINITION,
        "dpp_encoding_comparison_parameters": (
            DPP_ENCODING_COMPARISON_PARAMETERS_DEFINITION
        ),
        "dpp_encoding_comparison_selection": (
            DPP_ENCODING_COMPARISON_SELECTION_DEFINITION
        ),
        "dpp_encoding_comparison": DPP_ENCODING_COMPARISON_DEFINITION,
        "path_progression_decoding_parameters": (
            PATH_PROGRESSION_DECODING_PARAMETERS_DEFINITION
        ),
        "path_progression_decoding_comparison_selection": (
            PATH_PROGRESSION_DECODING_SELECTION_DEFINITION
        ),
        "path_progression_decoding_comparison": (
            PATH_PROGRESSION_DECODING_DEFINITION
        ),
        "path_specific_place_decoding_parameters": (
            PATH_SPECIFIC_PLACE_DECODING_PARAMETERS_DEFINITION
        ),
        "path_specific_place_decoding_selection": (
            PATH_SPECIFIC_PLACE_DECODING_SELECTION_DEFINITION
        ),
        "path_specific_place_decoding": (
            PATH_SPECIFIC_PLACE_DECODING_DEFINITION
        ),
        "motor_encoding_comparison_parameters": (
            MOTOR_ENCODING_COMPARISON_PARAMETERS_DEFINITION
        ),
        "motor_encoding_comparison_selection": (
            MOTOR_ENCODING_COMPARISON_SELECTION_DEFINITION
        ),
        "motor_encoding_comparison": MOTOR_ENCODING_COMPARISON_DEFINITION,
        "dark_light_glm_parameters": DARK_LIGHT_GLM_PARAMETERS_DEFINITION,
        "dark_light_glm_selection": DARK_LIGHT_GLM_SELECTION_DEFINITION,
        "dark_light_glm": DARK_LIGHT_GLM_DEFINITION,
        "swap_glm_parameters": SWAP_GLM_PARAMETERS_DEFINITION,
        "swap_glm_selection": SWAP_GLM_SELECTION_DEFINITION,
        "swap_glm": SWAP_GLM_DEFINITION,
        "swap_tuning_curve_comparison_parameters": (
            SWAP_TUNING_CURVE_COMPARISON_PARAMETERS_DEFINITION
        ),
        "swap_tuning_curve_comparison_selection": (
            SWAP_TUNING_CURVE_COMPARISON_SELECTION_DEFINITION
        ),
        "swap_tuning_curve_comparison": (
            SWAP_TUNING_CURVE_COMPARISON_DEFINITION
        ),
        "ripple_glm_parameters": RIPPLE_GLM_PARAMETERS_DEFINITION,
        "ripple_glm_selection": RIPPLE_GLM_SELECTION_DEFINITION,
        "ripple_glm": RIPPLE_GLM_DEFINITION,
        "cross_region_xcorr_parameters": (
            CROSS_REGION_XCORR_PARAMETERS_DEFINITION
        ),
        "cross_region_xcorr_selection": (
            CROSS_REGION_XCORR_SELECTION_DEFINITION
        ),
        "cross_region_xcorr": CROSS_REGION_XCORR_DEFINITION,
        "ripple_decoding_comparison_parameters": (
            RIPPLE_DECODING_COMPARISON_PARAMETERS_DEFINITION
        ),
        "ripple_decoding_comparison_selection": (
            RIPPLE_DECODING_COMPARISON_SELECTION_DEFINITION
        ),
        "ripple_decoding_comparison": (
            RIPPLE_DECODING_COMPARISON_DEFINITION
        ),
        "analysis_nwbfile": ANALYSIS_NWBFILE_DEFINITION,
    }
)


__all__ = [
    "ABSOLUTE_OVERLAP_TUNING_SIMILARITY_PARAMETERS",
    "CORRELATION_TUNING_SIMILARITY_PARAMETERS",
    "CURRENT_V5_CA1_DARK_LIGHT_GLM_PARAMETERS",
    "CURRENT_V5_V1_DARK_LIGHT_GLM_PARAMETERS",
    "DARK_LIGHT_GLM_DEFINITION",
    "DARK_LIGHT_GLM_OUTPUT_RULE",
    "DARK_LIGHT_GLM_PARAMETERS_DEFINITION",
    "DARK_LIGHT_GLM_PARAMETER_PRESETS",
    "DARK_LIGHT_GLM_SELECTION_DEFINITION",
    "DEFAULT_SWAP_GLM_PARAMETERS",
    "DEFAULT_ANALYSIS_NWBFILE_SCHEMA_NAME",
    "FIGURE_1D_TUNING_CURVE_PARAMETERS",
    "LEGACY_TUNING_CURVE_PARAMETERS",
    "LEGACY_V4_CA1_DARK_LIGHT_GLM_PARAMETERS",
    "LEGACY_V4_V1_DARK_LIGHT_GLM_PARAMETERS",
    "DEFAULT_MOVEMENT_PARAMETERS",
    "EPOCH_MOTOR_BEHAVIOR_DEFINITION",
    "EPOCH_MOTOR_BEHAVIOR_PARAMETERS_DEFINITION",
    "EPOCH_MOTOR_BEHAVIOR_SELECTION_DEFINITION",
    "DEFAULT_RIPPLE_MODULATION_PARAMETERS",
    "DEFAULT_SCHEMA_NAME",
    "DPP_ENCODING_COMPARISON_PARAMETER_PRESETS",
    "EXPECTED_SPYGLASS_GIT_COMMIT",
    "MANUSCRIPT_DPP_ENCODING_COMPARISON_PARAMETERS",
    "MANUSCRIPT_EPOCH_MOTOR_BEHAVIOR_PARAMETERS",
    "MANUSCRIPT_CA1_MOTOR_ENCODING_COMPARISON_PARAMETERS",
    "MANUSCRIPT_PATH_PROGRESSION_DECODING_PARAMETERS",
    "MANUSCRIPT_PATH_SPECIFIC_PLACE_DECODING_PARAMETERS",
    "MANUSCRIPT_V1_MOTOR_ENCODING_COMPARISON_PARAMETERS",
    "MOTOR_ENCODING_COMPARISON_MODEL_SPEC",
    "MOTOR_ENCODING_COMPARISON_OUTPUT_RULE",
    "MOTOR_ENCODING_COMPARISON_PARAMETER_PRESETS",
    "PATH_SPECIFIC_PLACE_DECODING_OUTPUT_RULE",
    "PATH_SPECIFIC_PLACE_DECODING_PARAMETER_PRESETS",
    "PATH_PROGRESSION_DECODING_ELIGIBILITY_RULE",
    "PATH_PROGRESSION_DECODING_OUTPUT_RULE",
    "PATH_PROGRESSION_DECODING_PARAMETER_PRESETS",
    "SPYGLASS_GIT_COMMIT",
    "SHAPE_OVERLAP_TUNING_SIMILARITY_PARAMETERS",
    "SWAP_GLM_DEFINITION",
    "SWAP_GLM_OUTPUT_RULE",
    "SWAP_GLM_PARAMETERS_DEFINITION",
    "SWAP_GLM_PARAMETER_PRESETS",
    "SWAP_GLM_SELECTION_DEFINITION",
    "MANUSCRIPT_CA1_SWAP_TUNING_CURVE_COMPARISON_PARAMETERS",
    "MANUSCRIPT_V1_SWAP_TUNING_CURVE_COMPARISON_PARAMETERS",
    "MANUSCRIPT_MEAN_ACTIVITY_RIPPLE_GLM_PARAMETERS",
    "MANUSCRIPT_UNIT_VECTOR_RIPPLE_GLM_PARAMETERS",
    "RIPPLE_GLM_DEFINITION",
    "RIPPLE_GLM_OUTPUT_RULE",
    "RIPPLE_GLM_PARAMETERS_DEFINITION",
    "RIPPLE_GLM_PARAMETER_PRESETS",
    "RIPPLE_GLM_SELECTION_DEFINITION",
    "CROSS_REGION_XCORR_DEFINITION",
    "CROSS_REGION_XCORR_OUTPUT_RULE",
    "CROSS_REGION_XCORR_PARAMETERS_DEFINITION",
    "CROSS_REGION_XCORR_PARAMETER_PRESETS",
    "CROSS_REGION_XCORR_SELECTION_DEFINITION",
    "MANUSCRIPT_CROSS_REGION_XCORR_PARAMETERS",
    "MANUSCRIPT_RIPPLE_DECODING_COMPARISON_PARAMETERS",
    "RIPPLE_DECODING_COMPARISON_DEFINITION",
    "RIPPLE_DECODING_COMPARISON_OUTPUT_RULE",
    "RIPPLE_DECODING_COMPARISON_PARAMETERS_DEFINITION",
    "RIPPLE_DECODING_COMPARISON_PARAMETER_PRESETS",
    "RIPPLE_DECODING_COMPARISON_SELECTION_DEFINITION",
    "SWAP_TUNING_CURVE_COMPARISON_DEFINITION",
    "SWAP_TUNING_CURVE_COMPARISON_PARAMETERS_DEFINITION",
    "SWAP_TUNING_CURVE_COMPARISON_PARAMETER_PRESETS",
    "SWAP_TUNING_CURVE_COMPARISON_OUTPUT_RULE",
    "SWAP_TUNING_CURVE_COMPARISON_SELECTION_DEFINITION",
    "TABLE_DEFINITIONS",
    "TUNING_CURVE_PARAMETER_PRESETS",
    "TUNING_SIMILARITY_PARAMETER_PRESETS",
]
