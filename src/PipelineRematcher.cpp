// Rebuild matches using optimized camera poses while reusing image features.
// Initial LIO poses remain unchanged so later BA retains the original priors.

#include "Pipeline.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <utility>

#include "FeatureProcessor.hpp"

namespace lio_visual_ba
{
namespace
{

const Pose3d& SeedPose(const CameraFrame& camera)
{
    return camera.has_optimized_pose ? camera.optimized_world_from_camera
                                     : camera.initial_world_from_camera;
}

double ViewAngleDegrees(const Pose3d& first, const Pose3d& second)
{
    const Eigen::Vector3d a = first.rotation_world_from_local * Eigen::Vector3d::UnitZ();
    const Eigen::Vector3d b = second.rotation_world_from_local * Eigen::Vector3d::UnitZ();
    constexpr double kRadiansToDegrees = 180.0 / 3.14159265358979323846;
    return std::acos(std::clamp(a.dot(b), -1.0, 1.0)) * kRadiansToDegrees;
}

std::vector<CameraPair> RecoveryPairs(const std::vector<CameraFrame>& cameras,
                                      const std::vector<bool>& weak, const PipelineConfig& config,
                                      const std::set<std::pair<int, int>>& processed)
{
    std::set<std::pair<int, int>> keys;
    const std::size_t normal_gap = config.pair_selection.maximum_temporal_gap;
    for (std::size_t first = 0; first < cameras.size(); ++first)
    {
        if (!weak[first])
            continue;
        const std::size_t begin =
            first > config.recovery_temporal_gap ? first - config.recovery_temporal_gap : 0;
        const std::size_t end = std::min(cameras.size(), first + config.recovery_temporal_gap + 1);
        for (std::size_t second = begin; second < end; ++second)
        {
            const std::size_t gap = first > second ? first - second : second - first;
            if (gap > normal_gap)
                keys.emplace(std::min(cameras[first].id.value(), cameras[second].id.value()),
                             std::max(cameras[first].id.value(), cameras[second].id.value()));
        }
        std::vector<std::pair<double, std::size_t>> spatial;
        for (std::size_t second = 0; second < cameras.size(); ++second)
        {
            const std::size_t gap = first > second ? first - second : second - first;
            if (first == second || gap <= normal_gap)
                continue;
            const double distance = (SeedPose(cameras[first]).translation_world_from_local -
                                     SeedPose(cameras[second]).translation_world_from_local)
                                        .norm();
            const double angle =
                ViewAngleDegrees(SeedPose(cameras[first]), SeedPose(cameras[second]));
            if (distance <= config.recovery_spatial_distance &&
                angle <= config.recovery_maximum_view_angle_degrees)
                spatial.emplace_back(distance, second);
        }
        std::sort(spatial.begin(), spatial.end());
        if (spatial.size() > config.recovery_spatial_pairs_per_frame)
            spatial.resize(config.recovery_spatial_pairs_per_frame);
        for (const auto& candidate : spatial)
            keys.emplace(std::min(cameras[first].id.value(), cameras[candidate.second].id.value()),
                         std::max(cameras[first].id.value(), cameras[candidate.second].id.value()));
    }
    std::vector<CameraPair> result;
    for (const auto& key : keys)
        if (processed.count(key) == 0)
            result.push_back({CameraId(key.first), CameraId(key.second), true});
    return result;
}

} // namespace

Result<PipelineFrontendResult> Pipeline::RunPipelineRematching(Reconstruction seed,
                                                               std::vector<FeatureSet> feature_sets,
                                                               const PipelineConfig& config)
{
    seed.tracks.clear();
    seed.landmarks.clear();
    PipelineFrontendResult result;
    result.reconstruction = std::move(seed);
    result.feature_sets = std::move(feature_sets);

    auto pairs = Mapper::SelectCandidatePairs(result.reconstruction.cameras, config.pair_selection);
    if (!pairs.ok())
        return Result<PipelineFrontendResult>::Failure(
            pairs.status().WithContext("optimized-pose rematch pair selection"));
    result.candidate_pairs = pairs.value();
    std::map<CameraId, const FeatureSet*> sets;
    std::map<CameraId, const CameraFrame*> cameras;
    for (const FeatureSet& features : result.feature_sets)
        sets.emplace(features.camera_id, &features);
    for (const CameraFrame& camera : result.reconstruction.cameras)
        cameras.emplace(camera.id, &camera);

    for (const CameraPair& pair : result.candidate_pairs.all_pairs)
    {
        const FeatureSet& first = *sets.at(pair.first_camera_id);
        const FeatureSet& second = *sets.at(pair.second_camera_id);
        FeatureMatchingOptions options = config.feature_matching;
        if (pair.is_loop)
            options.require_mutual_match = true;
        auto matches = FeatureProcessor::MatchFeaturesWithPosePrior(
            first, SeedPose(*cameras.at(pair.first_camera_id)), second,
            SeedPose(*cameras.at(pair.second_camera_id)), result.reconstruction.intrinsics,
            options);
        if (!matches.ok())
        {
            if (matches.status().code() == ErrorCode::kFailedPrecondition ||
                matches.status().code() == ErrorCode::kNumericalFailure)
            {
                ++result.rejected_candidate_pairs;
                continue;
            }
            return Result<PipelineFrontendResult>::Failure(
                matches.status().WithContext("optimized-pose rematching"));
        }
        for (const FeatureMatch& match : matches.value())
        {
            const bool core = static_cast<std::size_t>(match.first_feature_id.value()) <
                                  first.core_feature_count &&
                              static_cast<std::size_t>(match.second_feature_id.value()) <
                                  second.core_feature_count;
            const std::int64_t hash =
                static_cast<std::int64_t>(pair.first_camera_id.value()) * 73856093LL +
                static_cast<std::int64_t>(pair.second_camera_id.value()) * 19349663LL +
                match.first_feature_id.value();
            if (!pair.is_loop && core && hash % 10 == 0)
                result.heldout_matches.push_back(match);
            else
                result.verified_matches.push_back(match);
        }
        ++result.accepted_candidate_pairs;
    }

    if (!config.enable_weak_frame_recovery)
        return Result<PipelineFrontendResult>::Success(std::move(result));
    auto provisional = FeatureProcessor::BuildFeatureTracks(
        result.feature_sets, result.verified_matches, config.track_builder);
    if (!provisional.ok())
        return Result<PipelineFrontendResult>::Failure(
            provisional.status().WithContext("optimized-pose weak support"));
    std::map<CameraId, std::size_t> support;
    std::map<CameraId, std::set<std::pair<int, int>>> occupied;
    for (const FeatureTrack& track : provisional.value().tracks)
    {
        if (track.observations.size() < config.recovery_minimum_track_length)
            continue;
        for (const FeatureObservation& observation : track.observations)
        {
            ++support[observation.camera_id];
            const FeatureSet& set = *sets.at(observation.camera_id);
            const Eigen::Vector2d& pixel =
                set.features[static_cast<std::size_t>(observation.feature_id.value())].pixel;
            const int x =
                std::clamp(static_cast<int>(pixel.x() * config.feature_extraction.grid_columns /
                                            static_cast<double>(config.image_width)),
                           0, config.feature_extraction.grid_columns - 1);
            const int y =
                std::clamp(static_cast<int>(pixel.y() * config.feature_extraction.grid_rows /
                                            static_cast<double>(config.image_height)),
                           0, config.feature_extraction.grid_rows - 1);
            occupied[observation.camera_id].emplace(x, y);
        }
    }
    std::vector<bool> weak(result.reconstruction.cameras.size(), false);
    for (std::size_t index = 0; index < result.reconstruction.cameras.size(); ++index)
    {
        const CameraId id = result.reconstruction.cameras[index].id;
        weak[index] = support[id] < config.recovery_target_track_observations ||
                      occupied[id].size() < config.recovery_target_occupied_grid_cells;
        result.weak_cameras_before_recovery += weak[index] ? 1U : 0U;
    }
    std::set<std::pair<int, int>> processed;
    for (const CameraPair& pair : result.candidate_pairs.all_pairs)
        processed.emplace(pair.first_camera_id.value(), pair.second_camera_id.value());
    const std::vector<CameraPair> recovery =
        RecoveryPairs(result.reconstruction.cameras, weak, config, processed);
    result.recovery_candidate_pairs = recovery.size();
    for (const CameraPair& pair : recovery)
    {
        const FeatureSet& first = *sets.at(pair.first_camera_id);
        const FeatureSet& second = *sets.at(pair.second_camera_id);
        FeatureMatchingOptions options = config.feature_matching;
        options.require_mutual_match = true;
        auto descriptors = FeatureProcessor::MatchFeatureDescriptors(first, second, options);
        if (!descriptors.ok())
            return Result<PipelineFrontendResult>::Failure(descriptors.status());
        auto prior = FeatureProcessor::MatchFeaturesWithPosePrior(
            first, SeedPose(*cameras.at(pair.first_camera_id)), second,
            SeedPose(*cameras.at(pair.second_camera_id)), result.reconstruction.intrinsics,
            options);
        std::map<std::pair<int, int>, FeatureMatch> accepted;
        if (prior.ok())
            for (const FeatureMatch& match : prior.value())
                accepted.emplace(
                    std::make_pair(match.first_feature_id.value(), match.second_feature_id.value()),
                    match);
        auto visual = FeatureProcessor::VerifyMatchesWithVisualGeometry(
            first, second, descriptors.value(), config.visual_geometry);
        if (visual.ok())
            for (const FeatureMatch& match : visual.value().inlier_matches)
                accepted.emplace(
                    std::make_pair(match.first_feature_id.value(), match.second_feature_id.value()),
                    match);
        if (!accepted.empty())
        {
            ++result.recovery_accepted_pairs;
            result.recovery_added_matches += accepted.size();
            for (const auto& entry : accepted)
                result.verified_matches.push_back(entry.second);
        }
    }
    return Result<PipelineFrontendResult>::Success(std::move(result));
}

} // namespace lio_visual_ba
