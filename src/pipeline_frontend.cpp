// Input-to-matches stage: load data, initialize poses, extract features, and
// verify pairs. Held-out matches are kept separate from reconstruction tracks.

#include "pipeline.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <tuple>
#include <utility>

#include "feature_processor.hpp"
#include "dataset_io.hpp"
#include "mapper.hpp"

namespace lio_visual_ba
{
namespace
{

double ViewAngleDegrees(const Pose3d& first, const Pose3d& second)
{
    const Eigen::Vector3d a = first.rotation_world_from_local * Eigen::Vector3d::UnitZ();
    const Eigen::Vector3d b = second.rotation_world_from_local * Eigen::Vector3d::UnitZ();
    constexpr double kRadiansToDegrees = 180.0 / 3.14159265358979323846;
    return std::acos(std::clamp(a.dot(b), -1.0, 1.0)) * kRadiansToDegrees;
}

std::vector<CameraPair> SelectRecoveryPairs(const std::vector<CameraFrame>& cameras,
                                            const std::vector<bool>& weak,
                                            const PipelineConfig& config,
                                            const std::set<std::pair<int, int>>& processed)
{
    std::set<std::pair<int, int>> keys;
    const std::size_t normal_gap = config.pair_selection.maximum_temporal_gap;
    for (std::size_t first = 0; first < cameras.size(); ++first)
    {
        if (!weak[first])
        {
            continue;
        }
        const std::size_t begin =
            first > config.recovery_temporal_gap ? first - config.recovery_temporal_gap : 0;
        const std::size_t end = std::min(cameras.size(), first + config.recovery_temporal_gap + 1);
        for (std::size_t second = begin; second < end; ++second)
        {
            const std::size_t gap = first > second ? first - second : second - first;
            if (gap > normal_gap)
            {
                keys.emplace(std::min(cameras[first].id.value(), cameras[second].id.value()),
                             std::max(cameras[first].id.value(), cameras[second].id.value()));
            }
        }
        std::vector<std::pair<double, std::size_t>> spatial;
        for (std::size_t second = 0; second < cameras.size(); ++second)
        {
            const std::size_t gap = first > second ? first - second : second - first;
            if (first == second || gap <= normal_gap)
            {
                continue;
            }
            const double distance =
                (cameras[first].initial_world_from_camera.translation_world_from_local -
                 cameras[second].initial_world_from_camera.translation_world_from_local)
                    .norm();
            const double angle = ViewAngleDegrees(cameras[first].initial_world_from_camera,
                                                  cameras[second].initial_world_from_camera);
            if (distance <= config.recovery_spatial_distance &&
                angle <= config.recovery_maximum_view_angle_degrees)
            {
                spatial.emplace_back(distance, second);
            }
        }
        std::sort(spatial.begin(), spatial.end());
        if (spatial.size() > config.recovery_spatial_pairs_per_frame)
        {
            spatial.resize(config.recovery_spatial_pairs_per_frame);
        }
        for (const auto& candidate : spatial)
        {
            const std::size_t second = candidate.second;
            keys.emplace(std::min(cameras[first].id.value(), cameras[second].id.value()),
                         std::max(cameras[first].id.value(), cameras[second].id.value()));
        }
    }
    std::vector<CameraPair> pairs;
    for (const auto& key : keys)
    {
        if (processed.count(key) == 0)
        {
            pairs.push_back({CameraId(key.first), CameraId(key.second), true});
        }
    }
    return pairs;
}

} // namespace

Result<PipelineFrontendResult> Pipeline::RunPipelineFrontend(const PipelineConfig& config)
{
    auto dataset = DatasetIO::LoadDataset(config);
    if (!dataset.ok())
    {
        return Result<PipelineFrontendResult>::Failure(
            dataset.status().WithContext("pipeline dataset loading"));
    }
    const Status pose_status = Mapper::InitializeCameraPoses(
        dataset.value().cameras, dataset.value().trajectory, dataset.value().lidar_from_camera,
        config.camera_time_offset_seconds);
    if (!pose_status.ok())
    {
        return Result<PipelineFrontendResult>::Failure(
            pose_status.WithContext("pipeline pose initialization"));
    }

    PipelineFrontendResult result;
    result.reconstruction.intrinsics = dataset.value().intrinsics;
    result.reconstruction.cameras = std::move(dataset.value().cameras);
    result.reconstruction.provenance.random_seed = config.random_seed;
    result.feature_sets.reserve(result.reconstruction.cameras.size());
    for (const CameraFrame& camera : result.reconstruction.cameras)
    {
        auto features = FeatureProcessor::ExtractFeatures(camera.id, camera.image_path,
                                                          config.feature_extraction);
        if (!features.ok())
        {
            return Result<PipelineFrontendResult>::Failure(features.status().WithContext(
                "pipeline feature extraction for " + camera.image_name));
        }
        result.feature_sets.push_back(std::move(features.value()));
    }

    // Candidate selection limits the search using poses; descriptor and geometric
    // verification below decide which pairs actually contribute observations.
    auto pairs = Mapper::SelectCandidatePairs(result.reconstruction.cameras, config.pair_selection);
    if (!pairs.ok())
    {
        return Result<PipelineFrontendResult>::Failure(
            pairs.status().WithContext("pipeline candidate-pair selection"));
    }
    result.candidate_pairs = pairs.value();
    std::map<CameraId, const FeatureSet*> feature_sets;
    std::map<CameraId, const CameraFrame*> camera_frames;
    for (const FeatureSet& features : result.feature_sets)
    {
        feature_sets.emplace(features.camera_id, &features);
    }
    for (const CameraFrame& camera : result.reconstruction.cameras)
    {
        camera_frames.emplace(camera.id, &camera);
    }

    for (const CameraPair& pair : result.candidate_pairs.all_pairs)
    {
        const FeatureSet& first = *feature_sets.at(pair.first_camera_id);
        const FeatureSet& second = *feature_sets.at(pair.second_camera_id);
        FeatureMatchingOptions matching_options = config.feature_matching;
        if (pair.is_loop)
        {
            matching_options.require_mutual_match = true;
        }
        auto matches = FeatureProcessor::MatchFeaturesWithPosePrior(
            first, camera_frames.at(pair.first_camera_id)->initial_world_from_camera, second,
            camera_frames.at(pair.second_camera_id)->initial_world_from_camera,
            result.reconstruction.intrinsics, matching_options);
        if (!matches.ok())
        {
            if (matches.status().code() == ErrorCode::kFailedPrecondition ||
                matches.status().code() == ErrorCode::kNumericalFailure)
            {
                ++result.rejected_candidate_pairs;
                continue;
            }
            return Result<PipelineFrontendResult>::Failure(
                matches.status().WithContext("pipeline pose-prior matching"));
        }
        if (!config.require_visual_geometry)
        {
            for (const FeatureMatch& match : matches.value())
            {
                const bool core_match = static_cast<std::size_t>(match.first_feature_id.value()) <
                                            first.core_feature_count &&
                                        static_cast<std::size_t>(match.second_feature_id.value()) <
                                            second.core_feature_count;
                const std::int64_t holdout_hash =
                    static_cast<std::int64_t>(pair.first_camera_id.value()) * 73856093LL +
                    static_cast<std::int64_t>(pair.second_camera_id.value()) * 19349663LL +
                    match.first_feature_id.value();
                if (!pair.is_loop && core_match && holdout_hash % 10 == 0)
                {
                    result.heldout_matches.push_back(match);
                }
                else
                {
                    result.verified_matches.push_back(match);
                }
            }
            ++result.accepted_candidate_pairs;
            continue;
        }
        auto verified = FeatureProcessor::VerifyMatchesWithVisualGeometry(
            first, second, matches.value(), config.visual_geometry);
        if (!verified.ok())
        {
            if (verified.status().code() == ErrorCode::kFailedPrecondition)
            {
                ++result.rejected_candidate_pairs;
                continue;
            }
            return Result<PipelineFrontendResult>::Failure(
                verified.status().WithContext("pipeline visual-geometry verification"));
        }
        result.verified_matches.insert(result.verified_matches.end(),
                                       verified.value().inlier_matches.begin(),
                                       verified.value().inlier_matches.end());
        ++result.accepted_candidate_pairs;
    }

    if (config.enable_weak_frame_recovery && result.reconstruction.cameras.size() >= 2)
    {
        auto provisional = FeatureProcessor::BuildFeatureTracks(
            result.feature_sets, result.verified_matches, config.track_builder);
        if (!provisional.ok())
        {
            return Result<PipelineFrontendResult>::Failure(
                provisional.status().WithContext("weak-frame support estimation"));
        }
        std::map<CameraId, std::size_t> support;
        std::map<CameraId, std::set<std::pair<int, int>>> occupied_cells;
        for (const FeatureTrack& track : provisional.value().tracks)
        {
            if (track.observations.size() < config.recovery_minimum_track_length)
            {
                continue;
            }
            for (const FeatureObservation& observation : track.observations)
            {
                ++support[observation.camera_id];
                const FeatureSet& set = *feature_sets.at(observation.camera_id);
                const Eigen::Vector2d& pixel =
                    set.features[static_cast<std::size_t>(observation.feature_id.value())].pixel;
                const int cell_x =
                    std::clamp(static_cast<int>(pixel.x() * config.feature_extraction.grid_columns /
                                                static_cast<double>(config.image_width)),
                               0, config.feature_extraction.grid_columns - 1);
                const int cell_y =
                    std::clamp(static_cast<int>(pixel.y() * config.feature_extraction.grid_rows /
                                                static_cast<double>(config.image_height)),
                               0, config.feature_extraction.grid_rows - 1);
                occupied_cells[observation.camera_id].emplace(cell_x, cell_y);
            }
        }
        std::vector<bool> weak(result.reconstruction.cameras.size(), false);
        for (std::size_t index = 0; index < result.reconstruction.cameras.size(); ++index)
        {
            const CameraId camera_id = result.reconstruction.cameras[index].id;
            weak[index] =
                support[camera_id] < config.recovery_target_track_observations ||
                occupied_cells[camera_id].size() < config.recovery_target_occupied_grid_cells;
            result.weak_cameras_before_recovery += weak[index] ? 1U : 0U;
        }
        std::set<std::pair<int, int>> processed;
        for (const CameraPair& pair : result.candidate_pairs.all_pairs)
        {
            processed.emplace(pair.first_camera_id.value(), pair.second_camera_id.value());
        }
        const std::vector<CameraPair> recovery_pairs =
            SelectRecoveryPairs(result.reconstruction.cameras, weak, config, processed);
        result.recovery_candidate_pairs = recovery_pairs.size();
        for (const CameraPair& pair : recovery_pairs)
        {
            const FeatureSet& first = *feature_sets.at(pair.first_camera_id);
            const FeatureSet& second = *feature_sets.at(pair.second_camera_id);
            FeatureMatchingOptions mutual_options = config.feature_matching;
            mutual_options.require_mutual_match = true;
            auto descriptor_matches =
                FeatureProcessor::MatchFeatureDescriptors(first, second, mutual_options);
            if (!descriptor_matches.ok())
            {
                return Result<PipelineFrontendResult>::Failure(
                    descriptor_matches.status().WithContext("weak-frame descriptor matching"));
            }
            auto prior_matches = FeatureProcessor::MatchFeaturesWithPosePrior(
                first, camera_frames.at(pair.first_camera_id)->initial_world_from_camera, second,
                camera_frames.at(pair.second_camera_id)->initial_world_from_camera,
                result.reconstruction.intrinsics, mutual_options);
            if (!prior_matches.ok() &&
                prior_matches.status().code() != ErrorCode::kFailedPrecondition &&
                prior_matches.status().code() != ErrorCode::kNumericalFailure)
            {
                return Result<PipelineFrontendResult>::Failure(
                    prior_matches.status().WithContext("weak-frame pose-prior matching"));
            }
            std::map<std::pair<int, int>, FeatureMatch> recovered;
            if (prior_matches.ok())
            {
                for (const FeatureMatch& match : prior_matches.value())
                {
                    recovered.emplace(std::make_pair(match.first_feature_id.value(),
                                                     match.second_feature_id.value()),
                                      match);
                }
            }
            auto visual = FeatureProcessor::VerifyMatchesWithVisualGeometry(
                first, second, descriptor_matches.value(), config.visual_geometry);
            if (visual.ok())
            {
                for (const FeatureMatch& match : visual.value().inlier_matches)
                {
                    recovered.emplace(std::make_pair(match.first_feature_id.value(),
                                                     match.second_feature_id.value()),
                                      match);
                }
            }
            if (!recovered.empty())
            {
                ++result.recovery_accepted_pairs;
                result.recovery_added_matches += recovered.size();
                for (const auto& entry : recovered)
                {
                    result.verified_matches.push_back(entry.second);
                }
            }
        }
    }
    return Result<PipelineFrontendResult>::Success(std::move(result));
}

} // namespace lio_visual_ba
