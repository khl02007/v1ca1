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
        "analysis_nwbfile": ANALYSIS_NWBFILE_DEFINITION,
    }
)


__all__ = [
    "ABSOLUTE_OVERLAP_TUNING_SIMILARITY_PARAMETERS",
    "CORRELATION_TUNING_SIMILARITY_PARAMETERS",
    "DEFAULT_ANALYSIS_NWBFILE_SCHEMA_NAME",
    "FIGURE_1D_TUNING_CURVE_PARAMETERS",
    "LEGACY_TUNING_CURVE_PARAMETERS",
    "DEFAULT_MOVEMENT_PARAMETERS",
    "DEFAULT_RIPPLE_MODULATION_PARAMETERS",
    "DEFAULT_SCHEMA_NAME",
    "DPP_ENCODING_COMPARISON_PARAMETER_PRESETS",
    "EXPECTED_SPYGLASS_GIT_COMMIT",
    "MANUSCRIPT_DPP_ENCODING_COMPARISON_PARAMETERS",
    "MANUSCRIPT_PATH_PROGRESSION_DECODING_PARAMETERS",
    "MANUSCRIPT_PATH_SPECIFIC_PLACE_DECODING_PARAMETERS",
    "PATH_SPECIFIC_PLACE_DECODING_OUTPUT_RULE",
    "PATH_SPECIFIC_PLACE_DECODING_PARAMETER_PRESETS",
    "PATH_PROGRESSION_DECODING_ELIGIBILITY_RULE",
    "PATH_PROGRESSION_DECODING_OUTPUT_RULE",
    "PATH_PROGRESSION_DECODING_PARAMETER_PRESETS",
    "SPYGLASS_GIT_COMMIT",
    "SHAPE_OVERLAP_TUNING_SIMILARITY_PARAMETERS",
    "TABLE_DEFINITIONS",
    "TUNING_CURVE_PARAMETER_PRESETS",
    "TUNING_SIMILARITY_PARAMETER_PRESETS",
]
