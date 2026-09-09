#pragma once

// Initialize camera poses, select candidate pairs, and triangulate tracks.
// The pipeline passes explicit options and retains initial LIO poses as BA priors.

#include "status.hpp"
#include "types.hpp"
#include "geometry.hpp"
#include "dataset_io.hpp"
#include <cstddef>
#include <vector>

namespace lio_visual_ba
{

struct CameraPair
{
    CameraId first_camera_id;
    CameraId second_camera_id;
    bool is_loop = false;
    double center_distance = 0.0;
    double view_angle_degrees = 0.0;
};

struct PairSelectionOptions
{
    std::size_t maximum_temporal_gap = 6;
    std::size_t minimum_loop_separation = 20;
    double maximum_loop_distance = 0.5;
    std::size_t maximum_loops_per_frame = 3;
    double maximum_loop_view_angle_degrees = 60.0;
    bool prefer_optimized_camera_poses = true;
};

struct PairSelectionResult
{
    std::vector<CameraPair> temporal_pairs;
    std::vector<CameraPair> loop_pairs;
    std::vector<CameraPair> all_pairs;
};

struct LandmarkBuilderOptions
{
    std::size_t minimum_track_length = 2;
    std::size_t minimum_supplemental_track_length = 4;
    // Zero keeps every valid landmark. A positive value deterministically
    // caps the BA/3DGS set in stable track order for reference parity.
    std::size_t maximum_landmarks = 0;
    std::size_t minimum_landmarks_per_camera = 20;
    // Python applies the stricter 0.75-degree gate only to exactly-three-view
    // tracks made entirely from core SIFT features.
    double minimum_three_view_core_parallax_degrees = 0.75;
    bool prefer_optimized_camera_poses = true;
    bool prune_inconsistent_observations = false;
    double observation_pruning_multiplier = 2.0;
    TriangulationOptions triangulation;
};
struct LandmarkBuildResult
{
    std::vector<Landmark> landmarks;
    std::size_t short_tracks_rejected = 0;
    std::size_t geometric_quality_rejected = 0;
    std::size_t numerical_rejected = 0;
    std::size_t pruned_observations = 0;
};

// Stateless component API. Options and data are supplied explicitly per call.
class Mapper
{
  public:
    // Initialize camera poses at image_time + time_offset_seconds. Queries beyond
    // the trajectory endpoints use the nearest endpoint, matching the legacy tool.
    static Status InitializeCameraPoses(std::vector<CameraFrame>& cameras,
                                        const std::vector<TimedPose>& trajectory,
                                        const Pose3d& lidar_from_camera,
                                        double time_offset_seconds);

    // Propose temporal and spatial-loop pairs; FeatureProcessor verifies their matches.
    static Result<PairSelectionResult>
    SelectCandidatePairs(const std::vector<CameraFrame>& cameras,
                         const PairSelectionOptions& options = {});

    // Convert tracks into 3D points, applying support, depth, parallax, and pixel-error gates.
    static Result<LandmarkBuildResult> BuildSparseLandmarks(
        const CameraIntrinsics& intrinsics, const std::vector<CameraFrame>& cameras,
        const std::vector<FeatureTrack>& tracks, const LandmarkBuilderOptions& options = {});
};

} // namespace lio_visual_ba
