// Ceres bundle adjustment and local/global scheduling. Keep initial pose priors
// immutable and merge candidate results only after the corresponding solve succeeds.

#include "optimizer.hpp"
#include "geometry.hpp"
#include <algorithm>
#include <ceres/ceres.h>
#include <cmath>
#include <map>
#include <memory>
#include <set>
#include <utility>
#include <vector>

// ---- Bundle Adjuster ----

namespace lio_visual_ba
{
namespace
{

struct FixedCameraReprojectionResidual
{
    FixedCameraReprojectionResidual(const Pose3d& world_from_camera,
                                    const CameraIntrinsics& intrinsics,
                                    const Eigen::Vector2d& observed_pixel)
        : observed_x(observed_pixel.x()), observed_y(observed_pixel.y()), fx(intrinsics.fx),
          fy(intrinsics.fy), cx(intrinsics.cx), cy(intrinsics.cy)
    {
        const Eigen::Matrix3d camera_from_world =
            world_from_camera.rotation_world_from_local.conjugate().toRotationMatrix();
        const Eigen::Vector3d translation =
            -camera_from_world * world_from_camera.translation_world_from_local;
        for (int row = 0; row < 3; ++row)
        {
            for (int column = 0; column < 3; ++column)
            {
                rotation[3 * row + column] = camera_from_world(row, column);
            }
            camera_translation[row] = translation(row);
        }
    }

    template <typename T> bool operator()(const T* const point_world, T* residuals) const
    {
        T point_camera[3];
        for (int row = 0; row < 3; ++row)
        {
            point_camera[row] = T(camera_translation[row]);
            for (int column = 0; column < 3; ++column)
            {
                point_camera[row] += T(rotation[3 * row + column]) * point_world[column];
            }
        }
        residuals[0] = T(fx) * point_camera[0] / point_camera[2] + T(cx) - T(observed_x);
        residuals[1] = T(fy) * point_camera[1] / point_camera[2] + T(cy) - T(observed_y);
        return true;
    }

    double rotation[9]{};
    double camera_translation[3]{};
    double observed_x = 0.0;
    double observed_y = 0.0;
    double fx = 0.0;
    double fy = 0.0;
    double cx = 0.0;
    double cy = 0.0;
};

struct JointReprojectionResidual
{
    JointReprojectionResidual(const CameraIntrinsics& intrinsics,
                              const Eigen::Vector2d& observed_pixel)
        : observed_x(observed_pixel.x()), observed_y(observed_pixel.y()), fx(intrinsics.fx),
          fy(intrinsics.fy), cx(intrinsics.cx), cy(intrinsics.cy)
    {
    }

    template <typename T>
    bool operator()(const T* const quaternion_world_from_camera,
                    const T* const translation_world_from_camera, const T* const point_world,
                    T* residuals) const
    {
        const Eigen::Map<const Eigen::Quaternion<T>> rotation(quaternion_world_from_camera);
        const Eigen::Map<const Eigen::Matrix<T, 3, 1>> center(translation_world_from_camera);
        const Eigen::Map<const Eigen::Matrix<T, 3, 1>> point(point_world);
        const Eigen::Matrix<T, 3, 1> camera_point = rotation.conjugate() * (point - center);
        residuals[0] = T(fx) * camera_point.x() / camera_point.z() + T(cx) - T(observed_x);
        residuals[1] = T(fy) * camera_point.y() / camera_point.z() + T(cy) - T(observed_y);
        return true;
    }

    double observed_x;
    double observed_y;
    double fx;
    double fy;
    double cx;
    double cy;
};

struct PosePriorResidual
{
    PosePriorResidual(const Pose3d& prior, double translation_sigma, double rotation_sigma)
        : prior_translation(prior.translation_world_from_local),
          prior_rotation(prior.rotation_world_from_local),
          inverse_translation_sigma(1.0 / translation_sigma),
          inverse_rotation_sigma(1.0 / rotation_sigma)
    {
    }

    template <typename T>
    bool operator()(const T* const quaternion, const T* const translation, T* residuals) const
    {
        for (int index = 0; index < 3; ++index)
        {
            residuals[index] =
                (translation[index] - T(prior_translation[index])) * T(inverse_translation_sigma);
        }
        const Eigen::Map<const Eigen::Quaternion<T>> current(quaternion);
        const Eigen::Quaternion<T> delta = prior_rotation.cast<T>().conjugate() * current;
        residuals[3] = T(2.0 * inverse_rotation_sigma) * delta.x();
        residuals[4] = T(2.0 * inverse_rotation_sigma) * delta.y();
        residuals[5] = T(2.0 * inverse_rotation_sigma) * delta.z();
        return true;
    }

    Eigen::Vector3d prior_translation;
    Eigen::Quaterniond prior_rotation;
    double inverse_translation_sigma;
    double inverse_rotation_sigma;
};

struct LandmarkPriorResidual
{
    LandmarkPriorResidual(const Eigen::Vector3d& initial, double sigma)
        : initial_position(initial), inverse_sigma(1.0 / sigma)
    {
    }

    template <typename T> bool operator()(const T* const point, T* residuals) const
    {
        for (int index = 0; index < 3; ++index)
        {
            residuals[index] = (point[index] - T(initial_position[index])) * T(inverse_sigma);
        }
        return true;
    }

    Eigen::Vector3d initial_position;
    double inverse_sigma;
};

struct CameraParameters
{
    CameraId id;
    Pose3d prior;
    double quaternion[4]{}; // Eigen coefficient order: x, y, z, w.
    double translation[3]{};
};

const Pose3d& ActivePose(const CameraFrame& camera)
{
    return camera.has_optimized_pose ? camera.optimized_world_from_camera
                                     : camera.initial_world_from_camera;
}

Result<double> ReprojectionRmse(const CameraIntrinsics& intrinsics,
                                const std::map<CameraId, const CameraFrame*>& cameras,
                                const std::vector<Landmark>& landmarks,
                                const std::set<LandmarkId>& selected, double minimum_depth,
                                bool skip_invalid_geometry = false)
{
    double squared_error = 0.0;
    std::size_t residual_count = 0;
    for (const Landmark& landmark : landmarks)
    {
        if (selected.count(landmark.id) == 0)
        {
            continue;
        }
        for (const FeatureObservation& observation : landmark.observations)
        {
            auto projection = geometry::ProjectWorldPoint(
                intrinsics, ActivePose(*cameras.at(observation.camera_id)), landmark.position_world,
                minimum_depth);
            if (!projection.ok())
            {
                if (skip_invalid_geometry)
                {
                    continue;
                }
                return Result<double>::Failure(projection.status());
            }
            squared_error += (projection.value().pixel - observation.pixel).squaredNorm();
            ++residual_count;
        }
    }
    if (residual_count == 0)
    {
        return Result<double>::Failure(
            Status::Error(ErrorCode::kFailedPrecondition, "bundle adjustment has no residuals"));
    }
    return Result<double>::Success(std::sqrt(squared_error / static_cast<double>(residual_count)));
}

} // namespace

Result<BundleAdjustmentReport>
Optimizer::OptimizeLandmarks(Reconstruction& reconstruction,
                             const LandmarkBundleAdjustmentOptions& options)
{
    const Status intrinsics_status = geometry::ValidateCameraIntrinsics(reconstruction.intrinsics);
    if (!intrinsics_status.ok())
    {
        return Result<BundleAdjustmentReport>::Failure(intrinsics_status);
    }
    if (options.minimum_observations < 2 || options.maximum_iterations <= 0 ||
        !std::isfinite(options.huber_loss_pixels) || options.huber_loss_pixels <= 0.0 ||
        !std::isfinite(options.minimum_depth) || options.minimum_depth < 0.0 ||
        options.number_of_threads <= 0)
    {
        return Result<BundleAdjustmentReport>::Failure(Status::Error(
            ErrorCode::kInvalidArgument, "invalid landmark bundle-adjustment options"));
    }

    std::map<CameraId, const CameraFrame*> cameras;
    for (const CameraFrame& camera : reconstruction.cameras)
    {
        if (!camera.id.valid() || !cameras.emplace(camera.id, &camera).second)
        {
            return Result<BundleAdjustmentReport>::Failure(
                Status::Error(ErrorCode::kInvalidArgument, "invalid or duplicate BA camera ID"));
        }
        const Status pose_status = geometry::ValidatePose(ActivePose(camera));
        if (!pose_status.ok())
        {
            return Result<BundleAdjustmentReport>::Failure(pose_status);
        }
    }

    std::vector<Landmark> optimized = reconstruction.landmarks;
    std::map<LandmarkId, std::size_t> landmark_indices;
    std::set<LandmarkId> selected;
    BundleAdjustmentReport report;
    for (std::size_t index = 0; index < optimized.size(); ++index)
    {
        Landmark& landmark = optimized[index];
        if (!landmark.id.valid() || !landmark_indices.emplace(landmark.id, index).second ||
            !landmark.position_world.array().isFinite().all())
        {
            return Result<BundleAdjustmentReport>::Failure(
                Status::Error(ErrorCode::kInvalidArgument, "invalid or duplicate BA landmark"));
        }
        std::set<int> observed_cameras;
        for (const FeatureObservation& observation : landmark.observations)
        {
            if (cameras.count(observation.camera_id) == 0 ||
                !observation.pixel.array().isFinite().all() ||
                !observed_cameras.insert(observation.camera_id.value()).second)
            {
                return Result<BundleAdjustmentReport>::Failure(
                    Status::Error(ErrorCode::kInvalidArgument, "invalid BA landmark observation"));
            }
        }
        if (landmark.observations.size() < options.minimum_observations)
        {
            ++report.skipped_landmarks;
        }
        else
        {
            selected.insert(landmark.id);
        }
    }
    if (selected.empty())
    {
        return Result<BundleAdjustmentReport>::Failure(Status::Error(
            ErrorCode::kFailedPrecondition, "no landmarks are eligible for optimization"));
    }

    auto initial_rmse = ReprojectionRmse(reconstruction.intrinsics, cameras, optimized, selected,
                                         options.minimum_depth);
    if (!initial_rmse.ok())
    {
        return Result<BundleAdjustmentReport>::Failure(
            initial_rmse.status().WithContext("invalid initial BA geometry"));
    }

    ceres::Problem problem;
    for (Landmark& landmark : optimized)
    {
        if (selected.count(landmark.id) == 0)
        {
            continue;
        }
        double* point = landmark.position_world.data();
        problem.AddParameterBlock(point, 3);
        for (const FeatureObservation& observation : landmark.observations)
        {
            const CameraFrame& camera = *cameras.at(observation.camera_id);
            auto* residual = new ceres::AutoDiffCostFunction<FixedCameraReprojectionResidual, 2, 3>(
                new FixedCameraReprojectionResidual(ActivePose(camera), reconstruction.intrinsics,
                                                    observation.pixel));
            problem.AddResidualBlock(residual, new ceres::HuberLoss(options.huber_loss_pixels),
                                     point);
            ++report.residual_observations;
        }
        ++report.optimized_landmarks;
    }

    ceres::Solver::Options solver_options;
    solver_options.max_num_iterations = options.maximum_iterations;
    solver_options.num_threads = options.number_of_threads;
    solver_options.linear_solver_type = ceres::DENSE_QR;
    solver_options.minimizer_progress_to_stdout = false;
    ceres::Solver::Summary summary;
    ceres::Solve(solver_options, &problem, &summary);
    if (!summary.IsSolutionUsable())
    {
        return Result<BundleAdjustmentReport>::Failure(
            Status::Error(ErrorCode::kNumericalFailure,
                          "Ceres landmark optimization failed: " + summary.BriefReport()));
    }

    auto final_rmse = ReprojectionRmse(reconstruction.intrinsics, cameras, optimized, selected,
                                       options.minimum_depth);
    if (!final_rmse.ok())
    {
        return Result<BundleAdjustmentReport>::Failure(
            final_rmse.status().WithContext("invalid optimized BA geometry"));
    }
    report.iterations = static_cast<int>(summary.iterations.size());
    report.initial_reprojection_rmse_pixels = initial_rmse.value();
    report.final_reprojection_rmse_pixels = final_rmse.value();
    reconstruction.landmarks = std::move(optimized);
    return Result<BundleAdjustmentReport>::Success(report);
}

Result<BundleAdjustmentReport>
Optimizer::OptimizeCamerasAndLandmarks(Reconstruction& reconstruction,
                                       const JointBundleAdjustmentOptions& options)
{
    const Status intrinsics_status = geometry::ValidateCameraIntrinsics(reconstruction.intrinsics);
    if (!intrinsics_status.ok())
    {
        return Result<BundleAdjustmentReport>::Failure(intrinsics_status);
    }
    if (options.minimum_landmark_observations < 2 || options.maximum_iterations <= 0 ||
        !std::isfinite(options.huber_loss_pixels) || options.huber_loss_pixels <= 0.0 ||
        !std::isfinite(options.translation_prior_sigma) || options.translation_prior_sigma <= 0.0 ||
        !std::isfinite(options.rotation_prior_sigma) || options.rotation_prior_sigma <= 0.0 ||
        !std::isfinite(options.minimum_depth) || options.minimum_depth < 0.0 ||
        options.number_of_threads <= 0)
    {
        return Result<BundleAdjustmentReport>::Failure(
            Status::Error(ErrorCode::kInvalidArgument, "invalid joint bundle-adjustment options"));
    }
    if (reconstruction.cameras.empty())
    {
        return Result<BundleAdjustmentReport>::Failure(
            Status::Error(ErrorCode::kFailedPrecondition, "joint BA has no cameras"));
    }

    Reconstruction candidate = reconstruction;
    std::map<CameraId, std::size_t> camera_indices;
    std::vector<CameraParameters> camera_parameters;
    camera_parameters.reserve(candidate.cameras.size());
    for (std::size_t index = 0; index < candidate.cameras.size(); ++index)
    {
        CameraFrame& camera = candidate.cameras[index];
        if (!camera.id.valid() || !camera_indices.emplace(camera.id, index).second)
        {
            return Result<BundleAdjustmentReport>::Failure(
                Status::Error(ErrorCode::kInvalidArgument, "invalid or duplicate joint BA camera"));
        }
        const Pose3d& pose = ActivePose(camera);
        const Status pose_status = geometry::ValidatePose(pose);
        if (!pose_status.ok())
        {
            return Result<BundleAdjustmentReport>::Failure(pose_status);
        }
        CameraParameters parameters;
        parameters.id = camera.id;
        // Keep every repeated solve anchored to the immutable LIO estimate.
        // If the latest optimized pose becomes the next prior, overlapping
        // local solves accumulate drift instead of remaining LIO-regularized.
        parameters.prior = camera.initial_world_from_camera;
        Eigen::Map<Eigen::Quaterniond>(parameters.quaternion) = pose.rotation_world_from_local;
        Eigen::Map<Eigen::Vector3d>(parameters.translation) = pose.translation_world_from_local;
        camera_parameters.push_back(parameters);
    }
    const CameraId anchor = options.anchor_camera_id;
    if (anchor.valid() && camera_indices.count(anchor) == 0)
    {
        return Result<BundleAdjustmentReport>::Failure(
            Status::Error(ErrorCode::kInvalidArgument, "joint BA anchor camera is unknown"));
    }
    const std::set<CameraId> constant_cameras(options.constant_camera_ids.begin(),
                                              options.constant_camera_ids.end());
    for (CameraId camera_id : constant_cameras)
    {
        if (!camera_id.valid() || camera_indices.count(camera_id) == 0)
        {
            return Result<BundleAdjustmentReport>::Failure(
                Status::Error(ErrorCode::kInvalidArgument, "joint BA constant camera is unknown"));
        }
    }

    std::set<LandmarkId> selected;
    std::set<int> landmark_ids;
    BundleAdjustmentReport report;
    for (const Landmark& landmark : candidate.landmarks)
    {
        if (!landmark.id.valid() || !landmark_ids.insert(landmark.id.value()).second ||
            !landmark.position_world.array().isFinite().all())
        {
            return Result<BundleAdjustmentReport>::Failure(Status::Error(
                ErrorCode::kInvalidArgument, "invalid or duplicate joint BA landmark"));
        }
        std::set<int> observed_cameras;
        std::size_t valid_geometry_observations = 0;
        for (const FeatureObservation& observation : landmark.observations)
        {
            if (camera_indices.count(observation.camera_id) == 0 ||
                !observation.pixel.array().isFinite().all() ||
                !observed_cameras.insert(observation.camera_id.value()).second)
            {
                return Result<BundleAdjustmentReport>::Failure(
                    Status::Error(ErrorCode::kInvalidArgument, "invalid joint BA observation"));
            }
            const CameraFrame& camera = candidate.cameras[camera_indices.at(observation.camera_id)];
            if (geometry::ProjectWorldPoint(candidate.intrinsics, ActivePose(camera),
                                            landmark.position_world, options.minimum_depth)
                    .ok())
            {
                ++valid_geometry_observations;
            }
        }
        if (valid_geometry_observations >= options.minimum_landmark_observations)
        {
            selected.insert(landmark.id);
        }
        else
        {
            ++report.skipped_landmarks;
        }
    }
    if (selected.empty())
    {
        return Result<BundleAdjustmentReport>::Failure(
            Status::Error(ErrorCode::kFailedPrecondition, "joint BA has no eligible landmarks"));
    }

    std::map<CameraId, const CameraFrame*> initial_cameras;
    for (const CameraFrame& camera : candidate.cameras)
    {
        initial_cameras.emplace(camera.id, &camera);
    }
    auto initial_rmse = ReprojectionRmse(candidate.intrinsics, initial_cameras, candidate.landmarks,
                                         selected, options.minimum_depth, true);
    if (!initial_rmse.ok())
    {
        return Result<BundleAdjustmentReport>::Failure(initial_rmse.status());
    }

    ceres::Problem problem;
    for (CameraParameters& camera : camera_parameters)
    {
        problem.AddParameterBlock(camera.quaternion, 4, new ceres::EigenQuaternionManifold());
        problem.AddParameterBlock(camera.translation, 3);
        if ((anchor.valid() && camera.id == anchor) || constant_cameras.count(camera.id) != 0)
        {
            problem.SetParameterBlockConstant(camera.quaternion);
            problem.SetParameterBlockConstant(camera.translation);
        }
        else
        {
            auto* prior =
                new ceres::AutoDiffCostFunction<PosePriorResidual, 6, 4, 3>(new PosePriorResidual(
                    camera.prior, options.translation_prior_sigma, options.rotation_prior_sigma));
            problem.AddResidualBlock(prior, new ceres::HuberLoss(1.0), camera.quaternion,
                                     camera.translation);
            ++report.optimized_cameras;
        }
    }

    for (Landmark& landmark : candidate.landmarks)
    {
        if (selected.count(landmark.id) == 0)
        {
            continue;
        }
        problem.AddParameterBlock(landmark.position_world.data(), 3);
        for (const FeatureObservation& observation : landmark.observations)
        {
            CameraParameters& camera = camera_parameters[camera_indices.at(observation.camera_id)];
            const CameraFrame& camera_frame =
                candidate.cameras[camera_indices.at(observation.camera_id)];
            if (!geometry::ProjectWorldPoint(candidate.intrinsics, ActivePose(camera_frame),
                                             landmark.position_world, options.minimum_depth)
                     .ok())
            {
                continue;
            }
            auto* residual = new ceres::AutoDiffCostFunction<JointReprojectionResidual, 2, 4, 3, 3>(
                new JointReprojectionResidual(candidate.intrinsics, observation.pixel));
            problem.AddResidualBlock(residual, new ceres::CauchyLoss(options.huber_loss_pixels),
                                     camera.quaternion, camera.translation,
                                     landmark.position_world.data());
            ++report.residual_observations;
        }
        if (landmark.has_initial_position)
        {
            constexpr double kDegreesToRadians = 3.14159265358979323846 / 180.0;
            const double parallax = landmark.quality.maximum_parallax_degrees * kDegreesToRadians;
            const double track_factor =
                std::min(2.0, std::sqrt(static_cast<double>(std::max<std::size_t>(
                                            3, landmark.observations.size())) /
                                        3.0));
            const double sigma =
                std::clamp((0.10 + 2.0 * std::sin(parallax)) * track_factor, 0.10, 1.0);
            auto* landmark_prior = new ceres::AutoDiffCostFunction<LandmarkPriorResidual, 3, 3>(
                new LandmarkPriorResidual(landmark.initial_position_world, sigma));
            problem.AddResidualBlock(landmark_prior, new ceres::HuberLoss(1.0),
                                     landmark.position_world.data());
        }
        ++report.optimized_landmarks;
    }

    ceres::Solver::Options solver_options;
    solver_options.max_num_iterations = options.maximum_iterations;
    solver_options.num_threads = options.number_of_threads;
    solver_options.linear_solver_type = ceres::SPARSE_SCHUR;
    solver_options.minimizer_progress_to_stdout = false;
    ceres::Solver::Summary summary;
    ceres::Solve(solver_options, &problem, &summary);
    if (!summary.IsSolutionUsable())
    {
        return Result<BundleAdjustmentReport>::Failure(Status::Error(
            ErrorCode::kNumericalFailure, "joint Ceres BA failed: " + summary.BriefReport()));
    }

    for (std::size_t index = 0; index < candidate.cameras.size(); ++index)
    {
        CameraFrame& camera = candidate.cameras[index];
        CameraParameters& parameters = camera_parameters[index];
        Pose3d pose;
        pose.rotation_world_from_local =
            Eigen::Map<const Eigen::Quaterniond>(parameters.quaternion).normalized();
        pose.translation_world_from_local =
            Eigen::Map<const Eigen::Vector3d>(parameters.translation);
        const Status pose_status = geometry::ValidatePose(pose);
        if (!pose_status.ok())
        {
            return Result<BundleAdjustmentReport>::Failure(pose_status);
        }
        camera.optimized_world_from_camera = pose;
        camera.has_optimized_pose = true;
    }
    std::map<CameraId, const CameraFrame*> final_cameras;
    for (const CameraFrame& camera : candidate.cameras)
    {
        final_cameras.emplace(camera.id, &camera);
    }
    auto final_rmse = ReprojectionRmse(candidate.intrinsics, final_cameras, candidate.landmarks,
                                       selected, options.minimum_depth, true);
    if (!final_rmse.ok())
    {
        return Result<BundleAdjustmentReport>::Failure(
            final_rmse.status().WithContext("invalid joint BA result"));
    }
    report.iterations = static_cast<int>(summary.iterations.size());
    report.initial_reprojection_rmse_pixels = initial_rmse.value();
    report.final_reprojection_rmse_pixels = final_rmse.value();
    reconstruction = std::move(candidate);
    return Result<BundleAdjustmentReport>::Success(report);
}

Result<OutlierRefinementReport>
Optimizer::RefineWithOutlierCleanup(Reconstruction& reconstruction,
                                    const OutlierRefinementOptions& options)
{
    if (!std::isfinite(options.maximum_reprojection_error_pixels) ||
        options.maximum_reprojection_error_pixels <= 0.0 || options.maximum_cleanup_cycles <= 0)
    {
        return Result<OutlierRefinementReport>::Failure(
            Status::Error(ErrorCode::kInvalidArgument, "invalid reprojection cleanup options"));
    }

    Reconstruction candidate = reconstruction;
    OutlierRefinementReport report;
    bool needs_final_solve = false;
    for (int cycle = 0; cycle < options.maximum_cleanup_cycles; ++cycle)
    {
        auto bundle_adjustment =
            Optimizer::OptimizeCamerasAndLandmarks(candidate, options.bundle_adjustment);
        if (!bundle_adjustment.ok())
        {
            return Result<OutlierRefinementReport>::Failure(
                bundle_adjustment.status().WithContext("outlier cleanup bundle adjustment"));
        }
        report.final_bundle_adjustment = bundle_adjustment.value();
        ++report.cleanup_cycles;

        std::map<CameraId, const CameraFrame*> cameras;
        for (const CameraFrame& camera : candidate.cameras)
        {
            cameras.emplace(camera.id, &camera);
        }
        std::size_t removed_this_cycle = 0;
        std::vector<Landmark> retained_landmarks;
        retained_landmarks.reserve(candidate.landmarks.size());
        for (Landmark& landmark : candidate.landmarks)
        {
            std::vector<FeatureObservation> retained_observations;
            retained_observations.reserve(landmark.observations.size());
            for (const FeatureObservation& observation : landmark.observations)
            {
                auto projection = geometry::ProjectWorldPoint(
                    candidate.intrinsics, ActivePose(*cameras.at(observation.camera_id)),
                    landmark.position_world, options.bundle_adjustment.minimum_depth);
                if (!projection.ok())
                {
                    ++removed_this_cycle;
                    ++report.removed_observations;
                    continue;
                }
                const double error = (projection.value().pixel - observation.pixel).norm();
                if (error > options.maximum_reprojection_error_pixels)
                {
                    ++removed_this_cycle;
                    ++report.removed_observations;
                }
                else
                {
                    retained_observations.push_back(observation);
                }
            }
            if (retained_observations.size() <
                options.bundle_adjustment.minimum_landmark_observations)
            {
                ++report.removed_landmarks;
            }
            else
            {
                landmark.observations = std::move(retained_observations);
                retained_landmarks.push_back(std::move(landmark));
            }
        }
        candidate.landmarks = std::move(retained_landmarks);
        if (candidate.landmarks.empty())
        {
            return Result<OutlierRefinementReport>::Failure(Status::Error(
                ErrorCode::kFailedPrecondition, "outlier cleanup removed every eligible landmark"));
        }
        if (removed_this_cycle == 0)
        {
            needs_final_solve = false;
            reconstruction = std::move(candidate);
            return Result<OutlierRefinementReport>::Success(report);
        }
        needs_final_solve = true;
    }

    if (needs_final_solve)
    {
        auto final_bundle_adjustment =
            Optimizer::OptimizeCamerasAndLandmarks(candidate, options.bundle_adjustment);
        if (!final_bundle_adjustment.ok())
        {
            return Result<OutlierRefinementReport>::Failure(
                final_bundle_adjustment.status().WithContext("final post-cleanup solve"));
        }
        report.final_bundle_adjustment = final_bundle_adjustment.value();

        // The final solve can move poses/points enough to reintroduce residuals
        // above the export threshold. Python performs a terminal prune without
        // another pose update, guaranteeing the published maximum error gate.
        std::map<CameraId, const CameraFrame*> cameras;
        for (const CameraFrame& camera : candidate.cameras)
        {
            cameras.emplace(camera.id, &camera);
        }
        std::vector<Landmark> retained_landmarks;
        retained_landmarks.reserve(candidate.landmarks.size());
        for (Landmark& landmark : candidate.landmarks)
        {
            std::vector<FeatureObservation> retained_observations;
            retained_observations.reserve(landmark.observations.size());
            for (const FeatureObservation& observation : landmark.observations)
            {
                auto projection = geometry::ProjectWorldPoint(
                    candidate.intrinsics, ActivePose(*cameras.at(observation.camera_id)),
                    landmark.position_world, options.bundle_adjustment.minimum_depth);
                const double error = projection.ok()
                                         ? (projection.value().pixel - observation.pixel).norm()
                                         : std::numeric_limits<double>::infinity();
                if (error <= options.maximum_reprojection_error_pixels)
                {
                    retained_observations.push_back(observation);
                }
                else
                {
                    ++report.removed_observations;
                }
            }
            if (retained_observations.size() <
                options.bundle_adjustment.minimum_landmark_observations)
            {
                ++report.removed_landmarks;
            }
            else
            {
                landmark.observations = std::move(retained_observations);
                retained_landmarks.push_back(std::move(landmark));
            }
        }
        candidate.landmarks = std::move(retained_landmarks);
        if (candidate.landmarks.empty())
        {
            return Result<OutlierRefinementReport>::Failure(
                Status::Error(ErrorCode::kFailedPrecondition,
                              "terminal reprojection cleanup removed every eligible landmark"));
        }
    }
    reconstruction = std::move(candidate);
    return Result<OutlierRefinementReport>::Success(report);
}

} // namespace lio_visual_ba

// ---- Local Window Selector ----

namespace lio_visual_ba
{

Result<LocalBundleAdjustmentWindow>
Optimizer::SelectLocalBundleAdjustmentWindow(const Reconstruction& reconstruction,
                                             CameraId seed_camera_id,
                                             const LocalWindowSelectionOptions& options)
{
    if (!seed_camera_id.valid() || options.maximum_cameras == 0 ||
        options.minimum_shared_landmarks == 0 || options.minimum_selected_observations < 2)
    {
        return Result<LocalBundleAdjustmentWindow>::Failure(
            Status::Error(ErrorCode::kInvalidArgument, "invalid local-window selection options"));
    }

    std::set<CameraId> known_cameras;
    for (const CameraFrame& camera : reconstruction.cameras)
    {
        if (!camera.id.valid() || !known_cameras.insert(camera.id).second)
        {
            return Result<LocalBundleAdjustmentWindow>::Failure(Status::Error(
                ErrorCode::kInvalidArgument, "invalid or duplicate local-window camera"));
        }
    }
    if (known_cameras.count(seed_camera_id) == 0)
    {
        return Result<LocalBundleAdjustmentWindow>::Failure(
            Status::Error(ErrorCode::kInvalidArgument, "local-window seed camera is unknown"));
    }

    std::map<CameraId, std::set<LandmarkId>> visible_landmarks;
    std::map<LandmarkId, std::set<CameraId>> landmark_cameras;
    std::set<LandmarkId> known_landmarks;
    for (const Landmark& landmark : reconstruction.landmarks)
    {
        if (!landmark.id.valid() || !known_landmarks.insert(landmark.id).second)
        {
            return Result<LocalBundleAdjustmentWindow>::Failure(Status::Error(
                ErrorCode::kInvalidArgument, "invalid or duplicate local-window landmark"));
        }
        std::set<CameraId>& observers = landmark_cameras[landmark.id];
        for (const FeatureObservation& observation : landmark.observations)
        {
            if (known_cameras.count(observation.camera_id) == 0 ||
                !observers.insert(observation.camera_id).second)
            {
                return Result<LocalBundleAdjustmentWindow>::Failure(Status::Error(
                    ErrorCode::kInvalidArgument, "invalid local-window landmark observation"));
            }
            visible_landmarks[observation.camera_id].insert(landmark.id);
        }
    }

    std::set<CameraId> selected{seed_camera_id};
    std::set<LandmarkId> window_landmarks = visible_landmarks[seed_camera_id];
    while (selected.size() < options.maximum_cameras)
    {
        CameraId best_camera = CameraId::Invalid();
        std::size_t best_shared_count = 0;
        for (CameraId candidate : known_cameras)
        {
            if (selected.count(candidate) != 0)
            {
                continue;
            }
            std::size_t shared_count = 0;
            for (LandmarkId landmark_id : visible_landmarks[candidate])
            {
                shared_count += window_landmarks.count(landmark_id);
            }
            if (shared_count > best_shared_count ||
                (shared_count == best_shared_count && shared_count != 0 &&
                 (!best_camera.valid() || candidate < best_camera)))
            {
                best_camera = candidate;
                best_shared_count = shared_count;
            }
        }
        if (!best_camera.valid() || best_shared_count < options.minimum_shared_landmarks)
        {
            break;
        }
        selected.insert(best_camera);
        window_landmarks.insert(visible_landmarks[best_camera].begin(),
                                visible_landmarks[best_camera].end());
    }

    LocalBundleAdjustmentWindow window;
    window.camera_ids.assign(selected.begin(), selected.end());
    for (LandmarkId landmark_id : known_landmarks)
    {
        std::size_t selected_observations = 0;
        for (CameraId camera_id : landmark_cameras[landmark_id])
        {
            selected_observations += selected.count(camera_id);
        }
        if (selected_observations >= options.minimum_selected_observations)
        {
            window.landmark_ids.push_back(landmark_id);
        }
    }
    return Result<LocalBundleAdjustmentWindow>::Success(std::move(window));
}

} // namespace lio_visual_ba

// ---- Local Bundle Adjuster ----

namespace lio_visual_ba
{

Result<LocalBundleAdjustmentReport>
Optimizer::OptimizeLocalBundleAdjustment(Reconstruction& reconstruction, CameraId seed_camera_id,
                                         const LocalBundleAdjustmentOptions& options)
{
    LocalWindowSelectionOptions selection_options = options.window_selection;
    selection_options.minimum_selected_observations =
        std::max(selection_options.minimum_selected_observations,
                 options.bundle_adjustment.minimum_landmark_observations);
    if (selection_options.maximum_cameras == 0)
        return Result<LocalBundleAdjustmentReport>::Failure(
            Status::Error(ErrorCode::kInvalidArgument, "local BA window cannot be empty"));

    // Match the Python reference: use the temporal window ending at the seed
    // as variables and retain every observing camera outside it as a fixed
    // boundary constraint.
    auto seed =
        std::find_if(reconstruction.cameras.begin(), reconstruction.cameras.end(),
                     [&](const CameraFrame& camera) { return camera.id == seed_camera_id; });
    if (seed == reconstruction.cameras.end())
        return Result<LocalBundleAdjustmentReport>::Failure(
            Status::Error(ErrorCode::kInvalidArgument, "local BA seed camera is unknown"));
    const std::size_t seed_index =
        static_cast<std::size_t>(std::distance(reconstruction.cameras.begin(), seed));
    const std::size_t begin = seed_index + 1 > options.window_selection.maximum_cameras
                                  ? seed_index + 1 - options.window_selection.maximum_cameras
                                  : 0;
    std::set<CameraId> selected_cameras;
    for (std::size_t index = begin; index <= seed_index; ++index)
        selected_cameras.insert(reconstruction.cameras[index].id);
    std::set<LandmarkId> selected_landmarks;
    for (const Landmark& landmark : reconstruction.landmarks)
        if (std::any_of(landmark.observations.begin(), landmark.observations.end(),
                        [&](const FeatureObservation& observation)
                        { return selected_cameras.count(observation.camera_id) != 0; }) &&
            landmark.observations.size() >= selection_options.minimum_selected_observations)
            selected_landmarks.insert(landmark.id);
    if (selected_cameras.size() < 2 || selected_landmarks.empty())
        return Result<LocalBundleAdjustmentReport>::Failure(Status::Error(
            ErrorCode::kFailedPrecondition, "local bundle-adjustment window is under-supported"));
    if (options.bundle_adjustment.anchor_camera_id.valid() &&
        selected_cameras.count(options.bundle_adjustment.anchor_camera_id) == 0)
        return Result<LocalBundleAdjustmentReport>::Failure(Status::Error(
            ErrorCode::kInvalidArgument, "local BA anchor lies outside the temporal window"));
    Reconstruction local;
    local.intrinsics = reconstruction.intrinsics;
    local.provenance = reconstruction.provenance;
    local.cameras = reconstruction.cameras;
    for (const Landmark& source : reconstruction.landmarks)
    {
        if (selected_landmarks.count(source.id) == 0)
        {
            continue;
        }
        Landmark landmark = source;
        local.landmarks.push_back(std::move(landmark));
    }

    // Absolute pose priors already remove gauge freedom. In particular, do not
    // freeze seed_camera_id: it is the newly introduced/weak camera that local
    // BA is expected to refine. This matches the Python reference pipeline.
    JointBundleAdjustmentOptions bundle_options = options.bundle_adjustment;
    bundle_options.constant_camera_ids.clear();
    for (const CameraFrame& camera : local.cameras)
        if (selected_cameras.count(camera.id) == 0)
            bundle_options.constant_camera_ids.push_back(camera.id);
    auto solved = Optimizer::OptimizeCamerasAndLandmarks(local, bundle_options);
    if (!solved.ok())
    {
        return Result<LocalBundleAdjustmentReport>::Failure(
            solved.status().WithContext("local bundle-adjustment solve"));
    }

    std::map<CameraId, const CameraFrame*> solved_cameras;
    for (const CameraFrame& camera : local.cameras)
    {
        solved_cameras.emplace(camera.id, &camera);
    }
    std::map<LandmarkId, const Landmark*> solved_landmarks;
    for (const Landmark& landmark : local.landmarks)
    {
        solved_landmarks.emplace(landmark.id, &landmark);
    }

    Reconstruction candidate = reconstruction;
    for (CameraFrame& camera : candidate.cameras)
    {
        const auto solved_camera = solved_cameras.find(camera.id);
        if (solved_camera != solved_cameras.end() && selected_cameras.count(camera.id) != 0)
        {
            camera.optimized_world_from_camera = solved_camera->second->optimized_world_from_camera;
            camera.has_optimized_pose = solved_camera->second->has_optimized_pose;
        }
    }
    for (Landmark& landmark : candidate.landmarks)
    {
        const auto solved_landmark = solved_landmarks.find(landmark.id);
        if (solved_landmark != solved_landmarks.end())
        {
            landmark.position_world = solved_landmark->second->position_world;
        }
    }

    LocalBundleAdjustmentReport report;
    report.window.camera_ids.assign(selected_cameras.begin(), selected_cameras.end());
    report.window.landmark_ids.assign(selected_landmarks.begin(), selected_landmarks.end());
    report.bundle_adjustment = solved.value();
    reconstruction = std::move(candidate);
    return Result<LocalBundleAdjustmentReport>::Success(std::move(report));
}

} // namespace lio_visual_ba

// ---- Incremental Optimizer ----

namespace lio_visual_ba
{
namespace
{

Reconstruction BuildPrefix(const Reconstruction& reconstruction, std::size_t camera_count,
                           std::size_t minimum_observations)
{
    Reconstruction prefix;
    prefix.intrinsics = reconstruction.intrinsics;
    prefix.provenance = reconstruction.provenance;
    prefix.cameras.assign(reconstruction.cameras.begin(),
                          reconstruction.cameras.begin() +
                              static_cast<std::ptrdiff_t>(camera_count));
    std::set<CameraId> included_cameras;
    for (const CameraFrame& camera : prefix.cameras)
    {
        included_cameras.insert(camera.id);
    }
    for (const Landmark& source : reconstruction.landmarks)
    {
        Landmark landmark = source;
        landmark.observations.erase(
            std::remove_if(landmark.observations.begin(), landmark.observations.end(),
                           [&included_cameras](const FeatureObservation& observation)
                           { return included_cameras.count(observation.camera_id) == 0; }),
            landmark.observations.end());
        if (landmark.observations.size() >= minimum_observations)
        {
            prefix.landmarks.push_back(std::move(landmark));
        }
    }
    return prefix;
}

void MergeOptimizedState(const Reconstruction& source, Reconstruction& destination)
{
    std::map<CameraId, const CameraFrame*> source_cameras;
    for (const CameraFrame& camera : source.cameras)
    {
        source_cameras.emplace(camera.id, &camera);
    }
    for (CameraFrame& camera : destination.cameras)
    {
        const auto found = source_cameras.find(camera.id);
        if (found != source_cameras.end() && found->second->has_optimized_pose)
        {
            camera.optimized_world_from_camera = found->second->optimized_world_from_camera;
            camera.has_optimized_pose = true;
        }
    }
    std::map<LandmarkId, const Landmark*> source_landmarks;
    for (const Landmark& landmark : source.landmarks)
    {
        source_landmarks.emplace(landmark.id, &landmark);
    }
    for (Landmark& landmark : destination.landmarks)
    {
        const auto found = source_landmarks.find(landmark.id);
        if (found != source_landmarks.end())
        {
            landmark.position_world = found->second->position_world;
        }
    }
}

} // namespace

Result<IncrementalOptimizationReport>
Optimizer::RunIncrementalOptimization(Reconstruction& reconstruction,
                                      const IncrementalOptimizationOptions& options)
{
    if (options.minimum_registered_cameras < 2 ||
        options.minimum_registered_cameras > reconstruction.cameras.size())
    {
        return Result<IncrementalOptimizationReport>::Failure(Status::Error(
            ErrorCode::kInvalidArgument, "invalid incremental optimization camera threshold"));
    }

    Reconstruction candidate = reconstruction;
    IncrementalOptimizationReport report;
    std::size_t last_global_camera_count = 0;
    for (std::size_t camera_count = options.minimum_registered_cameras;
         camera_count <= candidate.cameras.size(); ++camera_count)
    {
        const std::size_t minimum_observations = std::max(
            options.local_bundle_adjustment.window_selection.minimum_selected_observations,
            options.local_bundle_adjustment.bundle_adjustment.minimum_landmark_observations);
        Reconstruction prefix = BuildPrefix(candidate, camera_count, minimum_observations);
        const CameraId seed_camera_id = prefix.cameras.back().id;
        auto local = Optimizer::OptimizeLocalBundleAdjustment(prefix, seed_camera_id,
                                                              options.local_bundle_adjustment);
        if (!local.ok())
        {
            if (local.status().code() == ErrorCode::kFailedPrecondition)
            {
                ++report.skipped_local_solve_count;
            }
            else
            {
                return Result<IncrementalOptimizationReport>::Failure(
                    local.status().WithContext("incremental local solve"));
            }
        }
        else
        {
            MergeOptimizedState(prefix, candidate);
            ++report.local_solve_count;
            report.locally_optimized_seeds.push_back(seed_camera_id);
        }

        if (options.global_bundle_adjustment_interval != 0 &&
            camera_count % options.global_bundle_adjustment_interval == 0)
        {
            prefix = BuildPrefix(candidate, camera_count,
                                 options.global_bundle_adjustment.minimum_landmark_observations);
            auto global =
                Optimizer::OptimizeCamerasAndLandmarks(prefix, options.global_bundle_adjustment);
            if (!global.ok())
            {
                return Result<IncrementalOptimizationReport>::Failure(
                    global.status().WithContext("incremental periodic global solve"));
            }
            MergeOptimizedState(prefix, candidate);
            report.final_bundle_adjustment = global.value();
            ++report.global_solve_count;
            last_global_camera_count = camera_count;
        }
    }

    if (options.run_final_global_bundle_adjustment &&
        last_global_camera_count != candidate.cameras.size())
    {
        auto global =
            Optimizer::OptimizeCamerasAndLandmarks(candidate, options.global_bundle_adjustment);
        if (!global.ok())
        {
            return Result<IncrementalOptimizationReport>::Failure(
                global.status().WithContext("incremental final global solve"));
        }
        report.final_bundle_adjustment = global.value();
        ++report.global_solve_count;
    }
    reconstruction = std::move(candidate);
    return Result<IncrementalOptimizationReport>::Success(std::move(report));
}

} // namespace lio_visual_ba
