// Matches-to-reconstruction stage: build tracks and landmarks, optimize, then
// optionally recover weak cameras and extend tracks before refreshing export data.

#include "pipeline.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <set>
#include <tuple>
#include <utility>

#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>

#include "geometry.hpp"

namespace lio_visual_ba
{
namespace
{

const Pose3d& ActivePose(const CameraFrame& camera)
{
    return camera.has_optimized_pose ? camera.optimized_world_from_camera
                                     : camera.initial_world_from_camera;
}

struct TrackExtensionReport
{
    std::size_t proposed = 0;
    std::size_t retained = 0;
    std::size_t rejected = 0;
    std::size_t conflicts = 0;
};

struct PnpRecoveryReport
{
    std::size_t targets = 0;
    std::size_t recovered = 0;
    std::size_t added_observations = 0;
};

Result<PnpRecoveryReport> RecoverWeakCamerasWithPnp(Reconstruction& reconstruction,
                                                    const std::vector<FeatureSet>& feature_sets,
                                                    const std::vector<FeatureMatch>& matches,
                                                    const PipelineConfig& config)
{
    PnpRecoveryReport report;
    if (!config.enable_pnp_recovery)
    {
        return Result<PnpRecoveryReport>::Success(report);
    }
    std::map<CameraId, std::size_t> camera_indices;
    std::map<CameraId, const FeatureSet*> features_by_camera;
    for (std::size_t index = 0; index < reconstruction.cameras.size(); ++index)
    {
        camera_indices.emplace(reconstruction.cameras[index].id, index);
    }
    for (const FeatureSet& features : feature_sets)
    {
        features_by_camera.emplace(features.camera_id, &features);
    }
    std::map<std::pair<int, int>, std::size_t> observation_owner;
    std::vector<std::size_t> support(reconstruction.cameras.size(), 0);
    for (std::size_t landmark_index = 0; landmark_index < reconstruction.landmarks.size();
         ++landmark_index)
    {
        for (const FeatureObservation& observation :
             reconstruction.landmarks[landmark_index].observations)
        {
            observation_owner[{observation.camera_id.value(), observation.feature_id.value()}] =
                landmark_index;
            ++support[camera_indices.at(observation.camera_id)];
        }
    }
    const cv::Mat camera_matrix =
        (cv::Mat_<double>(3, 3) << reconstruction.intrinsics.fx, 0.0, reconstruction.intrinsics.cx,
         0.0, reconstruction.intrinsics.fy, reconstruction.intrinsics.cy, 0.0, 0.0, 1.0);
    for (std::size_t target_index = 0; target_index < reconstruction.cameras.size(); ++target_index)
    {
        if (support[target_index] >= config.track_extension_weak_landmarks_per_camera)
        {
            continue;
        }
        ++report.targets;
        const CameraId target_id = reconstruction.cameras[target_index].id;
        std::map<std::pair<std::size_t, std::size_t>, std::size_t> votes;
        for (const FeatureMatch& match : matches)
        {
            CameraId neighbor_id = CameraId::Invalid();
            FeatureId target_feature = FeatureId::Invalid();
            FeatureId neighbor_feature = FeatureId::Invalid();
            if (match.first_camera_id == target_id)
            {
                target_feature = match.first_feature_id;
                neighbor_id = match.second_camera_id;
                neighbor_feature = match.second_feature_id;
            }
            else if (match.second_camera_id == target_id)
            {
                target_feature = match.second_feature_id;
                neighbor_id = match.first_camera_id;
                neighbor_feature = match.first_feature_id;
            }
            else
            {
                continue;
            }
            const std::size_t neighbor_index = camera_indices.at(neighbor_id);
            const std::size_t gap = target_index > neighbor_index ? target_index - neighbor_index
                                                                  : neighbor_index - target_index;
            if (gap > config.pnp_recovery_neighbor_gap)
            {
                continue;
            }
            const auto owner =
                observation_owner.find({neighbor_id.value(), neighbor_feature.value()});
            if (owner == observation_owner.end() ||
                observation_owner.count({target_id.value(), target_feature.value()}) != 0)
            {
                continue;
            }
            const std::size_t landmark_index = owner->second;
            const Landmark& landmark = reconstruction.landmarks[landmark_index];
            if (std::any_of(landmark.observations.begin(), landmark.observations.end(),
                            [&](const FeatureObservation& observation)
                            { return observation.camera_id == target_id; }))
            {
                continue;
            }
            ++votes[{landmark_index, static_cast<std::size_t>(target_feature.value())}];
        }
        std::map<std::size_t, std::pair<std::size_t, std::size_t>> best_by_landmark;
        for (const auto& entry : votes)
        {
            const std::size_t landmark_index = entry.first.first;
            const std::size_t feature_index = entry.first.second;
            auto found = best_by_landmark.find(landmark_index);
            if (found == best_by_landmark.end() || entry.second > found->second.second)
            {
                best_by_landmark[landmark_index] = {feature_index, entry.second};
            }
        }
        struct PnpCandidate
        {
            std::size_t landmark_index;
            std::size_t feature_index;
            std::size_t votes;
        };
        std::vector<PnpCandidate> candidates;
        candidates.reserve(best_by_landmark.size());
        for (const auto& entry : best_by_landmark)
            candidates.push_back({entry.first, entry.second.first, entry.second.second});
        std::sort(candidates.begin(), candidates.end(),
                  [](const PnpCandidate& first, const PnpCandidate& second)
                  {
                      return first.votes != second.votes
                                 ? first.votes > second.votes
                                 : std::tie(first.landmark_index, first.feature_index) <
                                       std::tie(second.landmark_index, second.feature_index);
                  });
        std::vector<cv::Point3d> object_points;
        std::vector<cv::Point2d> image_points;
        std::vector<std::pair<std::size_t, std::size_t>> correspondences;
        std::set<std::size_t> assigned_target_features;
        const FeatureSet& target_features = *features_by_camera.at(target_id);
        for (const PnpCandidate& candidate : candidates)
        {
            const std::size_t feature_index = candidate.feature_index;
            if (feature_index >= target_features.features.size())
            {
                continue;
            }
            if (!assigned_target_features.insert(feature_index).second)
            {
                continue;
            }
            const Eigen::Vector3d& point =
                reconstruction.landmarks[candidate.landmark_index].position_world;
            const Eigen::Vector2d& pixel = target_features.features[feature_index].pixel;
            object_points.emplace_back(point.x(), point.y(), point.z());
            image_points.emplace_back(pixel.x(), pixel.y());
            correspondences.emplace_back(candidate.landmark_index, feature_index);
        }
        if (correspondences.size() < config.pnp_recovery_minimum_inliers)
        {
            continue;
        }
        const Pose3d& initial = ActivePose(reconstruction.cameras[target_index]);
        const Eigen::Matrix3d rotation_camera_from_world =
            initial.rotation_world_from_local.conjugate().toRotationMatrix();
        cv::Mat rotation_matrix(3, 3, CV_64F);
        for (int row = 0; row < 3; ++row)
            for (int column = 0; column < 3; ++column)
                rotation_matrix.at<double>(row, column) = rotation_camera_from_world(row, column);
        cv::Mat rotation_vector;
        cv::Rodrigues(rotation_matrix, rotation_vector);
        const Eigen::Vector3d translation_camera_from_world =
            -rotation_camera_from_world * initial.translation_world_from_local;
        cv::Mat translation_vector =
            (cv::Mat_<double>(3, 1) << translation_camera_from_world.x(),
             translation_camera_from_world.y(), translation_camera_from_world.z());
        cv::Mat inliers;
        const bool solved =
            cv::solvePnPRansac(object_points, image_points, camera_matrix, cv::noArray(),
                               rotation_vector, translation_vector, true, 2000,
                               static_cast<float>(config.pnp_recovery_reprojection_error_pixels),
                               0.999, inliers, cv::SOLVEPNP_ITERATIVE);
        if (!solved || static_cast<std::size_t>(inliers.rows) < config.pnp_recovery_minimum_inliers)
        {
            continue;
        }
        cv::Rodrigues(rotation_vector, rotation_matrix);
        Eigen::Matrix3d solved_camera_from_world;
        for (int row = 0; row < 3; ++row)
            for (int column = 0; column < 3; ++column)
                solved_camera_from_world(row, column) = rotation_matrix.at<double>(row, column);
        const Eigen::Matrix3d solved_world_from_camera = solved_camera_from_world.transpose();
        const Eigen::Vector3d solved_translation_camera(translation_vector.at<double>(0),
                                                        translation_vector.at<double>(1),
                                                        translation_vector.at<double>(2));
        CameraFrame& camera = reconstruction.cameras[target_index];
        camera.optimized_world_from_camera.rotation_world_from_local =
            Eigen::Quaterniond(solved_world_from_camera).normalized();
        camera.optimized_world_from_camera.translation_world_from_local =
            -solved_world_from_camera * solved_translation_camera;
        camera.has_optimized_pose = true;
        for (int row = 0; row < inliers.rows; ++row)
        {
            const int correspondence_index = inliers.at<int>(row, 0);
            const auto [landmark_index, feature_index] =
                correspondences[static_cast<std::size_t>(correspondence_index)];
            const Feature& feature = target_features.features[feature_index];
            reconstruction.landmarks[landmark_index].observations.push_back(
                {target_id, feature.id, feature.pixel});
            observation_owner[{target_id.value(), feature.id.value()}] = landmark_index;
            ++report.added_observations;
        }
        ++report.recovered;
    }
    return Result<PnpRecoveryReport>::Success(report);
}

double SymmetricEpipolarDistance(const Eigen::Vector2d& first, const Eigen::Vector2d& second,
                                 const Eigen::Matrix3d& fundamental)
{
    const Eigen::Vector3d x(first.x(), first.y(), 1.0);
    const Eigen::Vector3d y(second.x(), second.y(), 1.0);
    const Eigen::Vector3d line_second = fundamental * x;
    const Eigen::Vector3d line_first = fundamental.transpose() * y;
    const double numerator = std::abs(y.dot(line_second));
    const double first_denominator = std::max(line_first.head<2>().norm(), 1e-12);
    const double second_denominator = std::max(line_second.head<2>().norm(), 1e-12);
    return std::max(numerator / first_denominator, numerator / second_denominator);
}

Result<TrackExtensionReport> ExtendLandmarkTracks(Reconstruction& reconstruction,
                                                  const std::vector<FeatureSet>& feature_sets,
                                                  const PipelineConfig& config)
{
    TrackExtensionReport report;
    if (!config.enable_track_extension || reconstruction.landmarks.empty())
    {
        return Result<TrackExtensionReport>::Success(report);
    }
    std::map<CameraId, std::size_t> camera_indices;
    std::map<CameraId, const FeatureSet*> features_by_camera;
    for (std::size_t index = 0; index < reconstruction.cameras.size(); ++index)
    {
        camera_indices.emplace(reconstruction.cameras[index].id, index);
    }
    for (const FeatureSet& features : feature_sets)
    {
        features_by_camera.emplace(features.camera_id, &features);
    }

    cv::Mat representatives(static_cast<int>(reconstruction.landmarks.size()), 128, CV_32F,
                            cv::Scalar(0));
    std::vector<bool> valid_representative(reconstruction.landmarks.size(), false);
    std::vector<std::vector<const float*>> descriptor_rows(reconstruction.landmarks.size());
    std::vector<std::size_t> camera_support(reconstruction.cameras.size(), 0);
    for (std::size_t landmark_index = 0; landmark_index < reconstruction.landmarks.size();
         ++landmark_index)
    {
        const Landmark& landmark = reconstruction.landmarks[landmark_index];
        for (const FeatureObservation& observation : landmark.observations)
        {
            ++camera_support[camera_indices.at(observation.camera_id)];
            const FeatureSet& features = *features_by_camera.at(observation.camera_id);
            const std::size_t feature_index =
                static_cast<std::size_t>(observation.feature_id.value());
            if (feature_index < features.core_feature_count &&
                feature_index < static_cast<std::size_t>(features.descriptors.rows))
            {
                descriptor_rows[landmark_index].push_back(
                    features.descriptors.ptr<float>(static_cast<int>(feature_index)));
            }
        }
        if (descriptor_rows[landmark_index].empty())
        {
            continue;
        }
        float* output = representatives.ptr<float>(static_cast<int>(landmark_index));
        std::vector<float> values(descriptor_rows[landmark_index].size());
        for (int dimension = 0; dimension < 128; ++dimension)
        {
            for (std::size_t row = 0; row < descriptor_rows[landmark_index].size(); ++row)
            {
                values[row] = descriptor_rows[landmark_index][row][dimension];
            }
            const std::size_t middle = values.size() / 2;
            std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(middle),
                             values.end());
            output[dimension] = values[middle];
        }
        valid_representative[landmark_index] = true;
    }

    struct Proposal
    {
        double descriptor_distance;
        double projection_offset;
        std::size_t landmark_index;
        std::size_t feature_index;
    };
    std::set<std::pair<std::size_t, CameraId>> newly_added;
    for (std::size_t camera_index = 0; camera_index < reconstruction.cameras.size(); ++camera_index)
    {
        const CameraFrame& camera = reconstruction.cameras[camera_index];
        const FeatureSet& features = *features_by_camera.at(camera.id);
        if (features.core_feature_count == 0 || features.descriptors.empty())
        {
            continue;
        }
        const double radius =
            camera_support[camera_index] < config.track_extension_weak_landmarks_per_camera
                ? config.track_extension_weak_search_radius_pixels
                : config.track_extension_search_radius_pixels;
        std::map<std::pair<int, int>, std::vector<std::size_t>> grid;
        std::set<std::size_t> reserved_features;
        for (const Landmark& landmark : reconstruction.landmarks)
        {
            for (const FeatureObservation& observation : landmark.observations)
            {
                if (observation.camera_id == camera.id)
                {
                    reserved_features.insert(
                        static_cast<std::size_t>(observation.feature_id.value()));
                }
            }
        }
        for (std::size_t feature_index = 0; feature_index < features.core_feature_count;
             ++feature_index)
        {
            if (reserved_features.count(feature_index) != 0)
            {
                continue;
            }
            const Eigen::Vector2d& pixel = features.features[feature_index].pixel;
            grid[{static_cast<int>(std::floor(pixel.x() / radius)),
                  static_cast<int>(std::floor(pixel.y() / radius))}]
                .push_back(feature_index);
        }
        std::vector<Proposal> proposals;
        for (std::size_t landmark_index = 0; landmark_index < reconstruction.landmarks.size();
             ++landmark_index)
        {
            if (!valid_representative[landmark_index])
            {
                continue;
            }
            Landmark& landmark = reconstruction.landmarks[landmark_index];
            if (std::any_of(landmark.observations.begin(), landmark.observations.end(),
                            [&](const FeatureObservation& observation)
                            { return observation.camera_id == camera.id; }))
            {
                continue;
            }
            const Eigen::Vector3d point_camera =
                geometry::WorldToCamera(ActivePose(camera), landmark.position_world);
            if (point_camera.z() <= 0.1 || point_camera.z() >= config.track_extension_maximum_depth)
            {
                continue;
            }
            auto projected =
                geometry::ProjectCameraPoint(reconstruction.intrinsics, point_camera, 0.1);
            if (!projected.ok() || projected.value().x() < 0.0 || projected.value().y() < 0.0 ||
                projected.value().x() >= reconstruction.intrinsics.width ||
                projected.value().y() >= reconstruction.intrinsics.height)
            {
                continue;
            }
            const int grid_x = static_cast<int>(std::floor(projected.value().x() / radius));
            const int grid_y = static_cast<int>(std::floor(projected.value().y() / radius));
            std::vector<std::size_t> candidates;
            for (int dx = -1; dx <= 1; ++dx)
            {
                for (int dy = -1; dy <= 1; ++dy)
                {
                    const auto found = grid.find({grid_x + dx, grid_y + dy});
                    if (found != grid.end())
                    {
                        for (const std::size_t feature_index : found->second)
                        {
                            if ((features.features[feature_index].pixel - projected.value())
                                    .norm() <= radius)
                            {
                                candidates.push_back(feature_index);
                            }
                        }
                    }
                }
            }
            if (candidates.empty())
            {
                continue;
            }
            const float* representative =
                representatives.ptr<float>(static_cast<int>(landmark_index));
            double best = std::numeric_limits<double>::infinity();
            double second = std::numeric_limits<double>::infinity();
            std::size_t best_feature = 0;
            for (const std::size_t feature_index : candidates)
            {
                const float* descriptor =
                    features.descriptors.ptr<float>(static_cast<int>(feature_index));
                double squared = 0.0;
                for (int dimension = 0; dimension < 128; ++dimension)
                {
                    const double difference = descriptor[dimension] - representative[dimension];
                    squared += difference * difference;
                }
                const double distance = std::sqrt(squared);
                if (distance < best)
                {
                    second = best;
                    best = distance;
                    best_feature = feature_index;
                }
                else if (distance < second)
                {
                    second = distance;
                }
            }
            const double ratio = std::isfinite(second) ? best / std::max(second, 1e-9) : 0.0;
            if (best > config.track_extension_descriptor_maximum ||
                ratio > config.track_extension_descriptor_ratio)
            {
                continue;
            }
            const FeatureObservation* anchor = &landmark.observations.front();
            for (const FeatureObservation& observation : landmark.observations)
            {
                const auto distance =
                    std::abs(static_cast<long long>(camera_indices.at(observation.camera_id)) -
                             static_cast<long long>(camera_index));
                const auto anchor_distance =
                    std::abs(static_cast<long long>(camera_indices.at(anchor->camera_id)) -
                             static_cast<long long>(camera_index));
                if (distance < anchor_distance)
                {
                    anchor = &observation;
                }
            }
            auto fundamental = geometry::ComputeFundamentalMatrix(
                reconstruction.intrinsics,
                ActivePose(reconstruction.cameras[camera_indices.at(anchor->camera_id)]),
                ActivePose(camera));
            if (!fundamental.ok() ||
                SymmetricEpipolarDistance(anchor->pixel, features.features[best_feature].pixel,
                                          fundamental.value()) >
                    config.track_extension_epipolar_error_pixels)
            {
                continue;
            }
            proposals.push_back({best,
                                 (features.features[best_feature].pixel - projected.value()).norm(),
                                 landmark_index, best_feature});
        }
        std::sort(proposals.begin(), proposals.end(),
                  [](const Proposal& first, const Proposal& second)
                  {
                      return first.descriptor_distance != second.descriptor_distance
                                 ? first.descriptor_distance < second.descriptor_distance
                                 : first.projection_offset < second.projection_offset;
                  });
        std::set<std::size_t> used_features = reserved_features;
        for (const Proposal& proposal : proposals)
        {
            if (!used_features.insert(proposal.feature_index).second)
            {
                ++report.conflicts;
                continue;
            }
            const Feature& feature = features.features[proposal.feature_index];
            reconstruction.landmarks[proposal.landmark_index].observations.push_back(
                {camera.id, feature.id, feature.pixel});
            newly_added.emplace(proposal.landmark_index, camera.id);
            ++report.proposed;
        }
    }

    for (std::size_t landmark_index = 0; landmark_index < reconstruction.landmarks.size();
         ++landmark_index)
    {
        Landmark& landmark = reconstruction.landmarks[landmark_index];
        const auto original_observations = landmark.observations;
        std::vector<TriangulationObservation> triangulation_observations;
        triangulation_observations.reserve(landmark.observations.size());
        for (const FeatureObservation& observation : landmark.observations)
        {
            triangulation_observations.push_back(
                {ActivePose(reconstruction.cameras[camera_indices.at(observation.camera_id)]),
                 observation.pixel});
        }
        TriangulationOptions options;
        options.minimum_depth = 0.1;
        options.maximum_reprojection_error_pixels = std::numeric_limits<double>::infinity();
        auto triangulated = geometry::TriangulateMultiView(reconstruction.intrinsics,
                                                           triangulation_observations, options);
        if (!triangulated.ok())
        {
            continue;
        }
        std::vector<FeatureObservation> clean;
        for (const FeatureObservation& observation : landmark.observations)
        {
            const bool is_new = newly_added.count({landmark_index, observation.camera_id}) != 0;
            auto projection = geometry::ProjectWorldPoint(
                reconstruction.intrinsics,
                ActivePose(reconstruction.cameras[camera_indices.at(observation.camera_id)]),
                triangulated.value().point_world, 0.1);
            const double error = projection.ok()
                                     ? (projection.value().pixel - observation.pixel).norm()
                                     : std::numeric_limits<double>::infinity();
            if (!is_new || error <= config.track_extension_final_reprojection_error_pixels)
            {
                clean.push_back(observation);
                report.retained += is_new ? 1U : 0U;
            }
            else
            {
                ++report.rejected;
            }
        }
        if (clean.size() >= 3)
        {
            landmark.observations = std::move(clean);
            landmark.position_world = triangulated.value().point_world;
            landmark.initial_position_world = landmark.position_world;
            landmark.has_initial_position = true;
        }
        else
        {
            landmark.observations = original_observations;
        }
    }
    return Result<TrackExtensionReport>::Success(report);
}

Status RefreshLandmarkExportAttributes(Reconstruction& reconstruction)
{
    std::map<CameraId, const CameraFrame*> cameras;
    std::map<CameraId, cv::Mat> images;
    for (const CameraFrame& camera : reconstruction.cameras)
    {
        cameras.emplace(camera.id, &camera);
        if (!camera.image_path.empty())
        {
            cv::Mat image = cv::imread(camera.image_path.string(), cv::IMREAD_COLOR);
            if (image.empty())
            {
                return Status::Error(ErrorCode::kNotFound,
                                     "cannot decode image for landmark colors: " +
                                         camera.image_path.string());
            }
            images.emplace(camera.id, std::move(image));
        }
    }

    for (Landmark& landmark : reconstruction.landmarks)
    {
        std::vector<double> errors;
        Eigen::Vector3d color_sum = Eigen::Vector3d::Zero();
        std::size_t color_samples = 0;
        for (const FeatureObservation& observation : landmark.observations)
        {
            const CameraFrame& camera = *cameras.at(observation.camera_id);
            const Pose3d& pose = camera.has_optimized_pose ? camera.optimized_world_from_camera
                                                           : camera.initial_world_from_camera;
            auto projection = geometry::ProjectWorldPoint(reconstruction.intrinsics, pose,
                                                          landmark.position_world);
            if (projection.ok())
            {
                errors.push_back((projection.value().pixel - observation.pixel).norm());
            }
            const auto image = images.find(observation.camera_id);
            if (image != images.end())
            {
                const int x = std::clamp(static_cast<int>(std::lround(observation.pixel.x())), 0,
                                         image->second.cols - 1);
                const int y = std::clamp(static_cast<int>(std::lround(observation.pixel.y())), 0,
                                         image->second.rows - 1);
                const cv::Vec3b bgr = image->second.at<cv::Vec3b>(y, x);
                color_sum += Eigen::Vector3d(bgr[2], bgr[1], bgr[0]);
                ++color_samples;
            }
        }
        if (!errors.empty())
        {
            std::sort(errors.begin(), errors.end());
            const std::size_t middle = errors.size() / 2;
            landmark.quality.median_reprojection_error_pixels =
                errors.size() % 2 == 0 ? 0.5 * (errors[middle - 1] + errors[middle])
                                       : errors[middle];
            landmark.quality.maximum_reprojection_error_pixels = errors.back();
        }
        if (color_samples != 0)
        {
            const Eigen::Vector3d mean = color_sum / static_cast<double>(color_samples);
            for (int channel = 0; channel < 3; ++channel)
            {
                landmark.color_rgb[channel] =
                    static_cast<std::uint8_t>(std::clamp(std::lround(mean[channel]), 0L, 255L));
            }
        }
    }
    return Status::Ok();
}

void SelectLandmarksForCoverage(Reconstruction& reconstruction, std::size_t maximum_landmarks,
                                std::size_t minimum_per_camera)
{
    if (maximum_landmarks == 0 || reconstruction.landmarks.size() <= maximum_landmarks)
    {
        return;
    }
    struct Candidate
    {
        std::size_t index;
        double quality;
    };
    std::vector<Candidate> ranked;
    std::map<CameraId, std::vector<Candidate>> by_camera;
    ranked.reserve(reconstruction.landmarks.size());
    for (std::size_t index = 0; index < reconstruction.landmarks.size(); ++index)
    {
        const Landmark& landmark = reconstruction.landmarks[index];
        int first = std::numeric_limits<int>::max();
        int last = -1;
        for (const FeatureObservation& observation : landmark.observations)
        {
            first = std::min(first, observation.camera_id.value());
            last = std::max(last, observation.camera_id.value());
        }
        constexpr double kDegreesToRadians = 3.14159265358979323846 / 180.0;
        const double quality =
            2.0 * std::log1p(static_cast<double>(landmark.observations.size())) +
            0.75 * std::log1p(static_cast<double>(std::max(0, last - first))) +
            2.0 * std::sin(landmark.quality.maximum_parallax_degrees * kDegreesToRadians) -
            std::log1p(landmark.quality.median_reprojection_error_pixels);
        const Candidate candidate{index, quality};
        ranked.push_back(candidate);
        for (const FeatureObservation& observation : landmark.observations)
        {
            by_camera[observation.camera_id].push_back(candidate);
        }
    }
    const auto better = [](const Candidate& lhs, const Candidate& rhs)
    { return lhs.quality != rhs.quality ? lhs.quality > rhs.quality : lhs.index < rhs.index; };
    std::sort(ranked.begin(), ranked.end(), better);
    for (auto& entry : by_camera)
    {
        std::sort(entry.second.begin(), entry.second.end(), better);
    }
    std::vector<bool> selected(reconstruction.landmarks.size(), false);
    std::map<CameraId, std::size_t> camera_counts;
    std::size_t selected_count = 0;
    const auto add = [&](std::size_t index)
    {
        if (selected[index] || selected_count >= maximum_landmarks)
        {
            return false;
        }
        selected[index] = true;
        ++selected_count;
        for (const FeatureObservation& observation : reconstruction.landmarks[index].observations)
        {
            ++camera_counts[observation.camera_id];
        }
        return true;
    };
    std::map<CameraId, std::size_t> cursors;
    while (selected_count < maximum_landmarks)
    {
        CameraId weakest = CameraId::Invalid();
        for (auto& entry : by_camera)
        {
            std::size_t& cursor = cursors[entry.first];
            while (cursor < entry.second.size() && selected[entry.second[cursor].index])
            {
                ++cursor;
            }
            if (camera_counts[entry.first] < minimum_per_camera && cursor < entry.second.size() &&
                (!weakest.valid() || camera_counts[entry.first] < camera_counts[weakest]))
            {
                weakest = entry.first;
            }
        }
        if (!weakest.valid())
        {
            break;
        }
        add(by_camera[weakest][cursors[weakest]++].index);
    }
    for (const Candidate& candidate : ranked)
    {
        if (selected_count >= maximum_landmarks)
        {
            break;
        }
        add(candidate.index);
    }
    std::vector<Landmark> compact;
    compact.reserve(selected_count);
    for (std::size_t index = 0; index < reconstruction.landmarks.size(); ++index)
    {
        if (selected[index])
        {
            Landmark landmark = std::move(reconstruction.landmarks[index]);
            landmark.id = LandmarkId(static_cast<LandmarkId::ValueType>(compact.size()));
            compact.push_back(std::move(landmark));
        }
    }
    reconstruction.landmarks = std::move(compact);
}

} // namespace

Result<PipelineBackendResult> Pipeline::RunPipelineBackend(PipelineFrontendResult frontend,
                                                           const PipelineConfig& config)
{
    auto tracks = FeatureProcessor::BuildFeatureTracks(
        frontend.feature_sets, frontend.verified_matches, config.track_builder);
    if (!tracks.ok())
    {
        return Result<PipelineBackendResult>::Failure(
            tracks.status().WithContext("pipeline track construction"));
    }
    frontend.reconstruction.tracks = tracks.value().tracks;

    auto landmarks = Mapper::BuildSparseLandmarks(
        frontend.reconstruction.intrinsics, frontend.reconstruction.cameras,
        frontend.reconstruction.tracks, config.landmark_builder);
    if (!landmarks.ok())
    {
        return Result<PipelineBackendResult>::Failure(
            landmarks.status().WithContext("pipeline landmark construction"));
    }
    frontend.reconstruction.landmarks = landmarks.value().landmarks;
    SelectLandmarksForCoverage(frontend.reconstruction, config.landmark_builder.maximum_landmarks,
                               config.landmark_builder.minimum_landmarks_per_camera);

    // Optimize the initial sparse map before recovery: refined poses make projected
    // landmark searches more reliable. Newly added observations trigger another cleanup.
    auto incremental = Optimizer::RunIncrementalOptimization(frontend.reconstruction,
                                                             config.incremental_optimization);
    if (!incremental.ok())
    {
        return Result<PipelineBackendResult>::Failure(
            incremental.status().WithContext("pipeline incremental optimization"));
    }
    auto refinement =
        Optimizer::RefineWithOutlierCleanup(frontend.reconstruction, config.final_refinement);
    if (!refinement.ok())
    {
        return Result<PipelineBackendResult>::Failure(
            refinement.status().WithContext("pipeline final refinement"));
    }
    auto pnp = RecoverWeakCamerasWithPnp(frontend.reconstruction, frontend.feature_sets,
                                         frontend.verified_matches, config);
    if (!pnp.ok())
    {
        return Result<PipelineBackendResult>::Failure(
            pnp.status().WithContext("pipeline PnP recovery"));
    }
    if (pnp.value().added_observations != 0)
    {
        refinement =
            Optimizer::RefineWithOutlierCleanup(frontend.reconstruction, config.final_refinement);
        if (!refinement.ok())
        {
            return Result<PipelineBackendResult>::Failure(
                refinement.status().WithContext("pipeline post-PnP refinement"));
        }
    }
    auto extension = ExtendLandmarkTracks(frontend.reconstruction, frontend.feature_sets, config);
    if (!extension.ok())
    {
        return Result<PipelineBackendResult>::Failure(
            extension.status().WithContext("pipeline track extension"));
    }
    if (extension.value().retained != 0)
    {
        refinement =
            Optimizer::RefineWithOutlierCleanup(frontend.reconstruction, config.final_refinement);
        if (!refinement.ok())
        {
            return Result<PipelineBackendResult>::Failure(
                refinement.status().WithContext("pipeline post-extension refinement"));
        }
    }
    const Status attributes = RefreshLandmarkExportAttributes(frontend.reconstruction);
    if (!attributes.ok())
    {
        return Result<PipelineBackendResult>::Failure(
            attributes.WithContext("pipeline landmark export attributes"));
    }

    PipelineBackendResult result;
    result.reconstruction = std::move(frontend.reconstruction);
    result.feature_sets = std::move(frontend.feature_sets);
    result.track_build = std::move(tracks.value());
    result.landmark_build = std::move(landmarks.value());
    result.incremental_optimization = std::move(incremental.value());
    result.final_refinement = std::move(refinement.value());
    result.track_extension_proposed_observations = extension.value().proposed;
    result.track_extension_retained_observations = extension.value().retained;
    result.track_extension_rejected_observations = extension.value().rejected;
    result.track_extension_assignment_conflicts = extension.value().conflicts;
    result.pnp_recovery_target_cameras = pnp.value().targets;
    result.pnp_recovery_recovered_cameras = pnp.value().recovered;
    result.pnp_recovery_added_observations = pnp.value().added_observations;
    return Result<PipelineBackendResult>::Success(std::move(result));
}

} // namespace lio_visual_ba
