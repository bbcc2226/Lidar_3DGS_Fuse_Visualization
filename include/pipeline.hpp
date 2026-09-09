#pragma once

// Pipeline configuration and orchestration. The public API also covers the
// frontend, backend, and rematcher implemented in their separate source files.

#include "status.hpp"
#include "types.hpp"
#include "feature_processor.hpp"
#include "dataset_io.hpp"
#include "mapper.hpp"
#include "optimizer.hpp"
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <vector>

namespace lio_visual_ba
{

struct PipelineConfig : DatasetIOConfig
{
    std::filesystem::path output_directory;
    double camera_time_offset_seconds = 0.0;
    std::uint64_t random_seed = 0;
    int maximum_cpu_threads = 4;
    FeatureExtractionOptions feature_extraction;
    FeatureMatchingOptions feature_matching;
    VisualGeometryOptions visual_geometry;
    bool require_visual_geometry = false;
    PairSelectionOptions pair_selection;
    bool enable_weak_frame_recovery = true;
    std::size_t recovery_target_track_observations = 50;
    std::size_t recovery_target_occupied_grid_cells = 16;
    std::size_t recovery_minimum_track_length = 4;
    std::size_t recovery_temporal_gap = 20;
    double recovery_spatial_distance = 1.0;
    std::size_t recovery_spatial_pairs_per_frame = 8;
    double recovery_maximum_view_angle_degrees = 60.0;
    bool enable_track_extension = false;
    double track_extension_search_radius_pixels = 18.0;
    double track_extension_weak_search_radius_pixels = 60.0;
    double track_extension_descriptor_ratio = 0.82;
    double track_extension_descriptor_maximum = 240.0;
    double track_extension_epipolar_error_pixels = 1.25;
    double track_extension_final_reprojection_error_pixels = 2.0;
    double track_extension_maximum_depth = 30.0;
    std::size_t track_extension_weak_landmarks_per_camera = 50;
    bool enable_pnp_recovery = false;
    bool enable_optimized_pose_rematching = false;
    std::size_t pnp_recovery_neighbor_gap = 10;
    double pnp_recovery_reprojection_error_pixels = 3.0;
    std::size_t pnp_recovery_minimum_inliers = 6;
    TrackBuilderOptions track_builder;
    LandmarkBuilderOptions landmark_builder;
    IncrementalOptimizationOptions incremental_optimization;
    OutlierRefinementOptions final_refinement;
};

struct PipelineFrontendResult
{
    Reconstruction reconstruction;
    std::vector<FeatureSet> feature_sets;
    PairSelectionResult candidate_pairs;
    std::vector<FeatureMatch> verified_matches;
    std::vector<FeatureMatch> heldout_matches;
    std::size_t accepted_candidate_pairs = 0;
    std::size_t rejected_candidate_pairs = 0;
    std::size_t weak_cameras_before_recovery = 0;
    std::size_t recovery_candidate_pairs = 0;
    std::size_t recovery_accepted_pairs = 0;
    std::size_t recovery_added_matches = 0;
};

struct PipelineBackendResult
{
    Reconstruction reconstruction;
    std::vector<FeatureSet> feature_sets;
    TrackBuildResult track_build;
    LandmarkBuildResult landmark_build;
    IncrementalOptimizationReport incremental_optimization;
    OutlierRefinementReport final_refinement;
    std::size_t track_extension_proposed_observations = 0;
    std::size_t track_extension_retained_observations = 0;
    std::size_t track_extension_rejected_observations = 0;
    std::size_t track_extension_assignment_conflicts = 0;
    std::size_t pnp_recovery_target_cameras = 0;
    std::size_t pnp_recovery_recovered_cameras = 0;
    std::size_t pnp_recovery_added_observations = 0;
};

struct HeldoutMatchRecord
{
    CameraId first_camera_id;
    CameraId second_camera_id;
    Eigen::Vector2d first_pixel = Eigen::Vector2d::Zero();
    Eigen::Vector2d second_pixel = Eigen::Vector2d::Zero();
};

struct PipelineRunResult
{
    PipelineBackendResult backend;
    std::size_t extracted_features = 0;
    std::size_t candidate_pairs = 0;
    std::size_t accepted_pairs = 0;
    std::size_t rejected_pairs = 0;
    std::size_t verified_matches = 0;
    std::size_t heldout_matches = 0;
    std::size_t weak_cameras_before_recovery = 0;
    std::size_t recovery_candidate_pairs = 0;
    std::size_t recovery_accepted_pairs = 0;
    std::size_t recovery_added_matches = 0;
    std::vector<HeldoutMatchRecord> heldout_match_data;
};

// Stateless component API. Options and data are supplied explicitly per call.
class Pipeline
{
  public:
    // Load a YAML pipeline configuration. Relative paths are resolved against the
    // configuration file's directory, making configurations location-independent.
    static Result<PipelineConfig>
    LoadPipelineConfig(const std::filesystem::path& configuration_file);

    // Execute the input-to-matches half of the reconstruction pipeline. Geometric
    // pair rejection is reported and skipped; malformed inputs remain fatal.
    static Result<PipelineFrontendResult> RunPipelineFrontend(const PipelineConfig& config);

    // Reuse extracted features but rebuild all matches from optimized seed poses.
    // The immutable initial poses remain the BA priors.
    static Result<PipelineFrontendResult>
    RunPipelineRematching(Reconstruction seed, std::vector<FeatureSet> feature_sets,
                          const PipelineConfig& config);

    // Consume frontend data and return the refined map plus per-stage reports.
    // Input is passed by value so the backend can take ownership of large vectors.
    static Result<PipelineBackendResult> RunPipelineBackend(PipelineFrontendResult frontend,
                                                            const PipelineConfig& config);

    // Write and validate a complete staging directory before publishing it.
    // The destination must be absent or empty; existing results are not overwritten.
    static Status PublishPipelineOutputs(const std::filesystem::path& output_directory,
                                         const PipelineRunResult& result);

    // Full entry point: frontend -> optional rematching -> backend -> publication.
    static Result<PipelineRunResult> RunPipeline(const PipelineConfig& config);
};

} // namespace lio_visual_ba
