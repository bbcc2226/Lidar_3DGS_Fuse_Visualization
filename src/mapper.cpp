// Initialize camera poses, select candidate pairs, and triangulate tracks.
// The pipeline passes explicit options and retains initial LIO poses as BA priors.

#include "mapper.hpp"
#include "geometry.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <set>
#include <tuple>
#include <utility>

// ---- Pose Initializer ----

namespace lio_visual_ba
{

Status Mapper::InitializeCameraPoses(std::vector<CameraFrame>& cameras,
                                     const std::vector<TimedPose>& trajectory,
                                     const Pose3d& lidar_from_camera, double time_offset_seconds)
{
    if (cameras.empty())
    {
        return Status::Error(ErrorCode::kInvalidArgument, "camera list is empty");
    }
    if (trajectory.size() < 2)
    {
        return Status::Error(ErrorCode::kFailedPrecondition,
                             "trajectory requires at least two poses");
    }
    if (!std::isfinite(time_offset_seconds))
    {
        return Status::Error(ErrorCode::kInvalidArgument, "time offset must be finite");
    }
    const Status extrinsic_status = geometry::ValidatePose(lidar_from_camera);
    if (!extrinsic_status.ok())
    {
        return extrinsic_status.WithContext("invalid lidar_from_camera extrinsic");
    }
    for (std::size_t index = 0; index < trajectory.size(); ++index)
    {
        if (!std::isfinite(trajectory[index].timestamp_seconds))
        {
            return Status::Error(ErrorCode::kDataLoss,
                                 "trajectory contains a non-finite timestamp");
        }
        const Status pose_status = geometry::ValidatePose(trajectory[index].world_from_lidar);
        if (!pose_status.ok())
        {
            return pose_status.WithContext("invalid trajectory pose at index " +
                                           std::to_string(index));
        }
    }
    for (std::size_t index = 1; index < trajectory.size(); ++index)
    {
        if (trajectory[index].timestamp_seconds <= trajectory[index - 1].timestamp_seconds)
        {
            return Status::Error(ErrorCode::kDataLoss,
                                 "trajectory timestamps must be strictly increasing");
        }
    }
    for (const CameraFrame& camera : cameras)
    {
        if (!std::isfinite(camera.timestamp_seconds + time_offset_seconds))
        {
            return Status::Error(ErrorCode::kInvalidArgument, "camera timestamp must be finite");
        }
    }

    for (CameraFrame& camera : cameras)
    {
        const double query = camera.timestamp_seconds + time_offset_seconds;
        auto upper = std::lower_bound(trajectory.begin(), trajectory.end(), query,
                                      [](const TimedPose& sample, double timestamp)
                                      { return sample.timestamp_seconds < timestamp; });
        Pose3d world_from_lidar;
        if (upper == trajectory.begin())
        {
            world_from_lidar = upper->world_from_lidar;
        }
        else if (upper == trajectory.end())
        {
            world_from_lidar = trajectory.back().world_from_lidar;
        }
        else
        {
            auto interpolated = geometry::InterpolatePose(
                (upper - 1)->world_from_lidar, (upper - 1)->timestamp_seconds,
                upper->world_from_lidar, upper->timestamp_seconds, query);
            if (!interpolated.ok())
            {
                return interpolated.status().WithContext("failed to initialize camera " +
                                                         std::to_string(camera.id.value()));
            }
            world_from_lidar = std::move(interpolated).value();
        }
        camera.initial_world_from_camera =
            geometry::ComposePoses(world_from_lidar, lidar_from_camera);
        camera.optimized_world_from_camera = camera.initial_world_from_camera;
        camera.has_optimized_pose = false;
    }
    return Status::Ok();
}

} // namespace lio_visual_ba

// ---- Pair Selector ----

namespace lio_visual_ba
{
namespace
{

const Pose3d& SelectedPose(const CameraFrame& camera, bool prefer_optimized)
{
    return prefer_optimized && camera.has_optimized_pose ? camera.optimized_world_from_camera
                                                         : camera.initial_world_from_camera;
}

double ViewAngleDegrees(const Pose3d& first, const Pose3d& second)
{
    const Eigen::Vector3d first_forward =
        first.rotation_world_from_local * Eigen::Vector3d::UnitZ();
    const Eigen::Vector3d second_forward =
        second.rotation_world_from_local * Eigen::Vector3d::UnitZ();
    const double cosine = std::clamp(first_forward.dot(second_forward), -1.0, 1.0);
    constexpr double kRadiansToDegrees = 180.0 / 3.14159265358979323846;
    return std::acos(cosine) * kRadiansToDegrees;
}

} // namespace

Result<PairSelectionResult> Mapper::SelectCandidatePairs(const std::vector<CameraFrame>& cameras,
                                                         const PairSelectionOptions& options)
{
    if (options.maximum_temporal_gap == 0 || options.minimum_loop_separation == 0 ||
        !std::isfinite(options.maximum_loop_distance) || options.maximum_loop_distance < 0.0 ||
        options.maximum_loops_per_frame == 0 ||
        !std::isfinite(options.maximum_loop_view_angle_degrees) ||
        options.maximum_loop_view_angle_degrees < 0.0 ||
        options.maximum_loop_view_angle_degrees > 180.0)
    {
        return Result<PairSelectionResult>::Failure(
            Status::Error(ErrorCode::kInvalidArgument, "invalid candidate-pair selection options"));
    }

    std::set<int> camera_ids;
    for (const CameraFrame& camera : cameras)
    {
        if (!camera.id.valid() || !camera_ids.insert(camera.id.value()).second)
        {
            return Result<PairSelectionResult>::Failure(
                Status::Error(ErrorCode::kInvalidArgument, "invalid or duplicate camera ID"));
        }
        const Status status =
            geometry::ValidatePose(SelectedPose(camera, options.prefer_optimized_camera_poses));
        if (!status.ok())
        {
            return Result<PairSelectionResult>::Failure(
                status.WithContext("invalid candidate-pair camera pose"));
        }
    }

    PairSelectionResult result;
    std::set<std::pair<int, int>> temporal_keys;
    for (std::size_t first = 0; first < cameras.size(); ++first)
    {
        const std::size_t remaining = cameras.size() - first - 1;
        const std::size_t gap = std::min(options.maximum_temporal_gap, remaining);
        const std::size_t end = first + gap + 1;
        for (std::size_t second = first + 1; second < end; ++second)
        {
            const Pose3d& first_pose =
                SelectedPose(cameras[first], options.prefer_optimized_camera_poses);
            const Pose3d& second_pose =
                SelectedPose(cameras[second], options.prefer_optimized_camera_poses);
            CameraPair pair;
            pair.first_camera_id = cameras[first].id;
            pair.second_camera_id = cameras[second].id;
            pair.center_distance =
                (first_pose.translation_world_from_local - second_pose.translation_world_from_local)
                    .norm();
            pair.view_angle_degrees = ViewAngleDegrees(first_pose, second_pose);
            result.temporal_pairs.push_back(pair);
            temporal_keys.emplace(pair.first_camera_id.value(), pair.second_camera_id.value());
        }
    }

    for (std::size_t first = 0; first < cameras.size(); ++first)
    {
        std::vector<CameraPair> candidates;
        if (options.minimum_loop_separation > cameras.size() - first)
        {
            continue;
        }
        for (std::size_t second = first + options.minimum_loop_separation; second < cameras.size();
             ++second)
        {
            const Pose3d& first_pose =
                SelectedPose(cameras[first], options.prefer_optimized_camera_poses);
            const Pose3d& second_pose =
                SelectedPose(cameras[second], options.prefer_optimized_camera_poses);
            const double distance =
                (first_pose.translation_world_from_local - second_pose.translation_world_from_local)
                    .norm();
            const double angle = ViewAngleDegrees(first_pose, second_pose);
            if (distance <= options.maximum_loop_distance &&
                angle <= options.maximum_loop_view_angle_degrees)
            {
                candidates.push_back(
                    {cameras[first].id, cameras[second].id, true, distance, angle});
            }
        }
        std::sort(candidates.begin(), candidates.end(),
                  [](const CameraPair& lhs, const CameraPair& rhs)
                  {
                      return std::tie(lhs.center_distance, lhs.second_camera_id) <
                             std::tie(rhs.center_distance, rhs.second_camera_id);
                  });
        if (candidates.size() > options.maximum_loops_per_frame)
        {
            candidates.resize(options.maximum_loops_per_frame);
        }
        for (const CameraPair& candidate : candidates)
        {
            const auto key = std::make_pair(candidate.first_camera_id.value(),
                                            candidate.second_camera_id.value());
            if (temporal_keys.count(key) == 0)
            {
                result.loop_pairs.push_back(candidate);
            }
        }
    }

    const auto pair_order = [](const CameraPair& lhs, const CameraPair& rhs)
    {
        return std::tie(lhs.first_camera_id, lhs.second_camera_id) <
               std::tie(rhs.first_camera_id, rhs.second_camera_id);
    };
    std::sort(result.loop_pairs.begin(), result.loop_pairs.end(), pair_order);
    result.all_pairs = result.temporal_pairs;
    result.all_pairs.insert(result.all_pairs.end(), result.loop_pairs.begin(),
                            result.loop_pairs.end());
    std::sort(result.all_pairs.begin(), result.all_pairs.end(), pair_order);
    return Result<PairSelectionResult>::Success(std::move(result));
}

} // namespace lio_visual_ba

// ---- Landmark Builder ----

namespace lio_visual_ba
{
Result<LandmarkBuildResult> Mapper::BuildSparseLandmarks(const CameraIntrinsics& intrinsics,
                                                         const std::vector<CameraFrame>& cameras,
                                                         const std::vector<FeatureTrack>& tracks,
                                                         const LandmarkBuilderOptions& options)
{
    const Status intrinsics_status = geometry::ValidateCameraIntrinsics(intrinsics);
    if (!intrinsics_status.ok())
        return Result<LandmarkBuildResult>::Failure(intrinsics_status);
    if (options.minimum_track_length < 2 ||
        options.minimum_supplemental_track_length < options.minimum_track_length ||
        !std::isfinite(options.minimum_three_view_core_parallax_degrees) ||
        options.minimum_three_view_core_parallax_degrees < 0.0 ||
        !std::isfinite(options.observation_pruning_multiplier) ||
        options.observation_pruning_multiplier < 1.0)
        return Result<LandmarkBuildResult>::Failure(Status::Error(
            ErrorCode::kInvalidArgument, "minimum landmark track length must be at least two"));
    std::map<CameraId, const CameraFrame*> camera_by_id;
    for (const auto& camera : cameras)
    {
        if (!camera.id.valid() || !camera_by_id.emplace(camera.id, &camera).second)
            return Result<LandmarkBuildResult>::Failure(
                Status::Error(ErrorCode::kInvalidArgument, "invalid or duplicate camera ID"));
        const Pose3d& pose = options.prefer_optimized_camera_poses && camera.has_optimized_pose
                                 ? camera.optimized_world_from_camera
                                 : camera.initial_world_from_camera;
        const Status status = geometry::ValidatePose(pose);
        if (!status.ok())
            return Result<LandmarkBuildResult>::Failure(status.WithContext("invalid camera pose"));
    }
    LandmarkBuildResult result;
    std::set<int> track_ids;
    for (const auto& track : tracks)
    {
        if (!track.id.valid() || !track_ids.insert(track.id.value()).second)
            return Result<LandmarkBuildResult>::Failure(
                Status::Error(ErrorCode::kInvalidArgument, "invalid or duplicate track ID"));
        if (track.observations.size() < options.minimum_track_length)
        {
            ++result.short_tracks_rejected;
            continue;
        }
        if (track.is_supplemental &&
            track.observations.size() < options.minimum_supplemental_track_length)
        {
            ++result.short_tracks_rejected;
            continue;
        }
        std::set<int> observed_cameras;
        std::vector<TriangulationObservation> observations;
        observations.reserve(track.observations.size());
        for (const auto& observation : track.observations)
        {
            auto camera = camera_by_id.find(observation.camera_id);
            if (camera == camera_by_id.end() || !observation.feature_id.valid() ||
                !observation.pixel.array().isFinite().all() ||
                !observed_cameras.insert(observation.camera_id.value()).second)
                return Result<LandmarkBuildResult>::Failure(
                    Status::Error(ErrorCode::kInvalidArgument,
                                  "track contains invalid or duplicate-camera observation"));
            const CameraFrame& frame = *camera->second;
            const Pose3d& pose = options.prefer_optimized_camera_poses && frame.has_optimized_pose
                                     ? frame.optimized_world_from_camera
                                     : frame.initial_world_from_camera;
            observations.push_back({pose, observation.pixel});
        }
        std::vector<FeatureObservation> retained_observations = track.observations;
        TriangulationOptions triangulation_options = options.triangulation;
        if (options.prune_inconsistent_observations)
        {
            triangulation_options.maximum_reprojection_error_pixels =
                std::numeric_limits<double>::infinity();
        }
        auto triangulated =
            geometry::TriangulateMultiView(intrinsics, observations, triangulation_options);
        if (triangulated.ok() && options.prune_inconsistent_observations)
        {
            const double pruning_threshold =
                options.observation_pruning_multiplier *
                options.triangulation.maximum_reprojection_error_pixels;
            std::vector<TriangulationObservation> retained_geometry;
            std::vector<FeatureObservation> retained_features;
            for (std::size_t index = 0; index < observations.size(); ++index)
            {
                if (triangulated.value().reprojection_errors_pixels[index] <= pruning_threshold)
                {
                    retained_geometry.push_back(observations[index]);
                    retained_features.push_back(track.observations[index]);
                }
                else
                {
                    ++result.pruned_observations;
                }
            }
            if (retained_geometry.size() < options.minimum_track_length)
            {
                ++result.geometric_quality_rejected;
                continue;
            }
            observations = std::move(retained_geometry);
            retained_observations = std::move(retained_features);
            triangulated =
                geometry::TriangulateMultiView(intrinsics, observations, triangulation_options);
            if (triangulated.ok())
            {
                std::vector<double> sorted = triangulated.value().reprojection_errors_pixels;
                std::sort(sorted.begin(), sorted.end());
                const std::size_t p90_index =
                    static_cast<std::size_t>(std::ceil(0.9 * static_cast<double>(sorted.size()))) -
                    1;
                if (triangulated.value().median_reprojection_error_pixels >
                        options.triangulation.maximum_reprojection_error_pixels ||
                    sorted[p90_index] > pruning_threshold)
                {
                    triangulated = Result<TriangulationResult>::Failure(
                        Status::Error(ErrorCode::kFailedPrecondition,
                                      "triangulated point fails robust reprojection quality"));
                }
            }
        }
        if (!triangulated.ok())
        {
            if (triangulated.status().code() == ErrorCode::kFailedPrecondition)
                ++result.geometric_quality_rejected;
            else if (triangulated.status().code() == ErrorCode::kNumericalFailure)
                ++result.numerical_rejected;
            else
                return Result<LandmarkBuildResult>::Failure(
                    triangulated.status().WithContext("track " + std::to_string(track.id.value())));
            continue;
        }
        if (!track.is_supplemental && retained_observations.size() == 3 &&
            triangulated.value().maximum_parallax_degrees <
                options.minimum_three_view_core_parallax_degrees)
        {
            ++result.geometric_quality_rejected;
            continue;
        }
        Landmark landmark;
        landmark.id = LandmarkId(static_cast<LandmarkId::ValueType>(result.landmarks.size()));
        landmark.source_track_id = track.id;
        landmark.position_world = triangulated.value().point_world;
        landmark.initial_position_world = landmark.position_world;
        landmark.has_initial_position = true;
        landmark.observations = std::move(retained_observations);
        landmark.quality.maximum_parallax_degrees = triangulated.value().maximum_parallax_degrees;
        landmark.quality.median_reprojection_error_pixels =
            triangulated.value().median_reprojection_error_pixels;
        landmark.quality.maximum_reprojection_error_pixels =
            triangulated.value().maximum_reprojection_error_pixels;
        result.landmarks.push_back(std::move(landmark));
    }
    return Result<LandmarkBuildResult>::Success(std::move(result));
}
} // namespace lio_visual_ba
