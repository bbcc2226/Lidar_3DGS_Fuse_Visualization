// Stateless pose, camera, epipolar, and triangulation math. Transforms use
// outer_from_inner names; projection converts world points into camera coordinates.

#include "geometry.hpp"
#include <Eigen/SVD>
#include <algorithm>
#include <cmath>
#include <string>

// ---- Pose ----

namespace lio_visual_ba
{
namespace
{

bool IsFinite(const Eigen::Vector3d& value) { return value.array().isFinite().all(); }

bool IsFinite(const Eigen::Quaterniond& value)
{
    return std::isfinite(value.w()) && std::isfinite(value.x()) && std::isfinite(value.y()) &&
           std::isfinite(value.z());
}

} // namespace

Status geometry::ValidatePose(const Pose3d& pose, double quaternion_norm_tolerance)
{
    if (!std::isfinite(quaternion_norm_tolerance) || quaternion_norm_tolerance < 0.0)
    {
        return Status::Error(ErrorCode::kInvalidArgument,
                             "quaternion norm tolerance must be finite and non-negative");
    }
    if (!IsFinite(pose.translation_world_from_local))
    {
        return Status::Error(ErrorCode::kInvalidArgument,
                             "pose translation contains a non-finite value");
    }
    if (!IsFinite(pose.rotation_world_from_local))
    {
        return Status::Error(ErrorCode::kInvalidArgument,
                             "pose quaternion contains a non-finite value");
    }

    const double norm = pose.rotation_world_from_local.norm();
    if (norm <= 1e-12)
    {
        return Status::Error(ErrorCode::kInvalidArgument, "pose quaternion has zero norm");
    }
    if (std::abs(norm - 1.0) > quaternion_norm_tolerance)
    {
        return Status::Error(ErrorCode::kInvalidArgument,
                             "pose quaternion is not normalized; norm=" + std::to_string(norm));
    }
    return Status::Ok();
}

Result<Pose3d> geometry::NormalizePose(const Pose3d& pose)
{
    if (!IsFinite(pose.translation_world_from_local))
    {
        return Result<Pose3d>::Failure(Status::Error(
            ErrorCode::kInvalidArgument, "pose translation contains a non-finite value"));
    }
    if (!IsFinite(pose.rotation_world_from_local) || pose.rotation_world_from_local.norm() <= 1e-12)
    {
        return Result<Pose3d>::Failure(Status::Error(
            ErrorCode::kInvalidArgument, "pose quaternion must be finite and have non-zero norm"));
    }

    Pose3d normalized = pose;
    normalized.rotation_world_from_local.normalize();
    return Result<Pose3d>::Success(std::move(normalized));
}

Pose3d geometry::InversePose(const Pose3d& outer_from_inner)
{
    Pose3d inner_from_outer;
    inner_from_outer.rotation_world_from_local =
        outer_from_inner.rotation_world_from_local.conjugate().normalized();
    inner_from_outer.translation_world_from_local = -(
        inner_from_outer.rotation_world_from_local * outer_from_inner.translation_world_from_local);
    return inner_from_outer;
}

Pose3d geometry::ComposePoses(const Pose3d& outer_from_middle, const Pose3d& middle_from_inner)
{
    Pose3d outer_from_inner;
    outer_from_inner.rotation_world_from_local =
        (outer_from_middle.rotation_world_from_local * middle_from_inner.rotation_world_from_local)
            .normalized();
    outer_from_inner.translation_world_from_local =
        outer_from_middle.rotation_world_from_local *
            middle_from_inner.translation_world_from_local +
        outer_from_middle.translation_world_from_local;
    return outer_from_inner;
}

Eigen::Vector3d geometry::TransformPoint(const Pose3d& outer_from_inner,
                                         const Eigen::Vector3d& point_inner)
{
    return outer_from_inner.rotation_world_from_local * point_inner +
           outer_from_inner.translation_world_from_local;
}

Result<Pose3d> geometry::InterpolatePose(const Pose3d& first_pose, double first_time_seconds,
                                         const Pose3d& second_pose, double second_time_seconds,
                                         double query_time_seconds)
{
    if (!std::isfinite(first_time_seconds) || !std::isfinite(second_time_seconds) ||
        !std::isfinite(query_time_seconds))
    {
        return Result<Pose3d>::Failure(Status::Error(
            ErrorCode::kInvalidArgument, "pose interpolation timestamps must be finite"));
    }
    if (first_time_seconds == second_time_seconds)
    {
        return Result<Pose3d>::Failure(Status::Error(
            ErrorCode::kInvalidArgument, "pose interpolation requires distinct sample timestamps"));
    }

    auto normalized_first = geometry::NormalizePose(first_pose);
    if (!normalized_first.ok())
    {
        return Result<Pose3d>::Failure(
            normalized_first.status().WithContext("invalid first interpolation pose"));
    }
    auto normalized_second = geometry::NormalizePose(second_pose);
    if (!normalized_second.ok())
    {
        return Result<Pose3d>::Failure(
            normalized_second.status().WithContext("invalid second interpolation pose"));
    }

    const double minimum_time = std::min(first_time_seconds, second_time_seconds);
    const double maximum_time = std::max(first_time_seconds, second_time_seconds);
    if (query_time_seconds < minimum_time || query_time_seconds > maximum_time)
    {
        return Result<Pose3d>::Failure(
            Status::Error(ErrorCode::kInvalidArgument,
                          "pose interpolation query is outside the sample interval"));
    }

    const double alpha =
        (query_time_seconds - first_time_seconds) / (second_time_seconds - first_time_seconds);
    const Pose3d& first = normalized_first.value();
    const Pose3d& second = normalized_second.value();

    Pose3d interpolated;
    interpolated.rotation_world_from_local =
        first.rotation_world_from_local.slerp(alpha, second.rotation_world_from_local).normalized();
    interpolated.translation_world_from_local = (1.0 - alpha) * first.translation_world_from_local +
                                                alpha * second.translation_world_from_local;
    return Result<Pose3d>::Success(std::move(interpolated));
}

} // namespace lio_visual_ba

// ---- Camera Model ----

namespace lio_visual_ba
{
namespace
{

bool CameraModelIsFinite(const Eigen::Vector2d& value) { return value.array().isFinite().all(); }

bool CameraModelIsFinite(const Eigen::Vector3d& value) { return value.array().isFinite().all(); }

} // namespace

Status geometry::ValidateCameraIntrinsics(const CameraIntrinsics& intrinsics)
{
    if (!std::isfinite(intrinsics.fx) || !std::isfinite(intrinsics.fy) ||
        !std::isfinite(intrinsics.cx) || !std::isfinite(intrinsics.cy))
    {
        return Status::Error(ErrorCode::kInvalidArgument,
                             "camera intrinsics contain a non-finite value");
    }
    if (intrinsics.fx <= 0.0 || intrinsics.fy <= 0.0)
    {
        return Status::Error(ErrorCode::kInvalidArgument, "camera focal lengths must be positive");
    }
    if (intrinsics.width < 0 || intrinsics.height < 0)
    {
        return Status::Error(ErrorCode::kInvalidArgument,
                             "camera image dimensions cannot be negative");
    }
    return Status::Ok();
}

Result<Eigen::Vector2d> geometry::ProjectCameraPoint(const CameraIntrinsics& intrinsics,
                                                     const Eigen::Vector3d& point_camera,
                                                     double minimum_depth)
{
    const Status intrinsics_status = geometry::ValidateCameraIntrinsics(intrinsics);
    if (!intrinsics_status.ok())
    {
        return Result<Eigen::Vector2d>::Failure(intrinsics_status);
    }
    if (!CameraModelIsFinite(point_camera) || !std::isfinite(minimum_depth) || minimum_depth < 0.0)
    {
        return Result<Eigen::Vector2d>::Failure(Status::Error(
            ErrorCode::kInvalidArgument,
            "camera point and minimum depth must be finite; minimum depth cannot be negative"));
    }
    if (point_camera.z() <= minimum_depth)
    {
        return Result<Eigen::Vector2d>::Failure(
            Status::Error(ErrorCode::kFailedPrecondition,
                          "camera point does not satisfy the positive-depth constraint"));
    }
    Eigen::Vector2d pixel;
    pixel.x() = intrinsics.fx * point_camera.x() / point_camera.z() + intrinsics.cx;
    pixel.y() = intrinsics.fy * point_camera.y() / point_camera.z() + intrinsics.cy;
    if (!CameraModelIsFinite(pixel))
    {
        return Result<Eigen::Vector2d>::Failure(
            Status::Error(ErrorCode::kNumericalFailure, "projection produced a non-finite pixel"));
    }
    return Result<Eigen::Vector2d>::Success(pixel);
}

Result<Eigen::Vector3d> geometry::UnprojectPixel(const CameraIntrinsics& intrinsics,
                                                 const Eigen::Vector2d& pixel, double depth)
{
    const Status intrinsics_status = geometry::ValidateCameraIntrinsics(intrinsics);
    if (!intrinsics_status.ok())
    {
        return Result<Eigen::Vector3d>::Failure(intrinsics_status);
    }
    if (!CameraModelIsFinite(pixel) || !std::isfinite(depth) || depth <= 0.0)
    {
        return Result<Eigen::Vector3d>::Failure(
            Status::Error(ErrorCode::kInvalidArgument,
                          "pixel must be finite and depth must be finite and positive"));
    }
    return Result<Eigen::Vector3d>::Success(
        Eigen::Vector3d((pixel.x() - intrinsics.cx) * depth / intrinsics.fx,
                        (pixel.y() - intrinsics.cy) * depth / intrinsics.fy, depth));
}

Eigen::Vector3d geometry::CameraCenterWorld(const Pose3d& world_from_camera)
{
    return world_from_camera.translation_world_from_local;
}

Eigen::Vector3d geometry::WorldToCamera(const Pose3d& world_from_camera,
                                        const Eigen::Vector3d& point_world)
{
    return world_from_camera.rotation_world_from_local.conjugate() *
           (point_world - world_from_camera.translation_world_from_local);
}

Result<ProjectionResult> geometry::ProjectWorldPoint(const CameraIntrinsics& intrinsics,
                                                     const Pose3d& world_from_camera,
                                                     const Eigen::Vector3d& point_world,
                                                     double minimum_depth)
{
    const Status pose_status = geometry::ValidatePose(world_from_camera);
    if (!pose_status.ok())
    {
        return Result<ProjectionResult>::Failure(
            pose_status.WithContext("invalid world_from_camera pose"));
    }
    if (!CameraModelIsFinite(point_world))
    {
        return Result<ProjectionResult>::Failure(
            Status::Error(ErrorCode::kInvalidArgument, "world point contains a non-finite value"));
    }
    const Eigen::Vector3d point_camera = geometry::WorldToCamera(world_from_camera, point_world);
    auto projected = geometry::ProjectCameraPoint(intrinsics, point_camera, minimum_depth);
    if (!projected.ok())
    {
        return Result<ProjectionResult>::Failure(projected.status());
    }
    ProjectionResult result;
    result.pixel = std::move(projected).value();
    result.depth = point_camera.z();
    return Result<ProjectionResult>::Success(result);
}

Result<Eigen::Matrix<double, 3, 4>>
geometry::CameraProjectionMatrix(const CameraIntrinsics& intrinsics,
                                 const Pose3d& world_from_camera)
{
    const Status intrinsics_status = geometry::ValidateCameraIntrinsics(intrinsics);
    if (!intrinsics_status.ok())
    {
        return Result<Eigen::Matrix<double, 3, 4>>::Failure(intrinsics_status);
    }
    const Status pose_status = geometry::ValidatePose(world_from_camera);
    if (!pose_status.ok())
    {
        return Result<Eigen::Matrix<double, 3, 4>>::Failure(pose_status);
    }
    const Eigen::Matrix3d rotation_camera_from_world =
        world_from_camera.rotation_world_from_local.conjugate().toRotationMatrix();
    Eigen::Matrix<double, 3, 4> camera_from_world;
    camera_from_world.leftCols<3>() = rotation_camera_from_world;
    camera_from_world.rightCols<1>() =
        -rotation_camera_from_world * world_from_camera.translation_world_from_local;
    return Result<Eigen::Matrix<double, 3, 4>>::Success(intrinsics.Matrix() * camera_from_world);
}

bool geometry::IsPixelInsideImage(const CameraIntrinsics& intrinsics, const Eigen::Vector2d& pixel,
                                  double border_pixels)
{
    if (!geometry::ValidateCameraIntrinsics(intrinsics).ok() || intrinsics.width <= 0 ||
        intrinsics.height <= 0 || !CameraModelIsFinite(pixel) || !std::isfinite(border_pixels) ||
        border_pixels < 0.0 || 2.0 * border_pixels >= static_cast<double>(intrinsics.width) ||
        2.0 * border_pixels >= static_cast<double>(intrinsics.height))
    {
        return false;
    }
    return pixel.x() >= border_pixels && pixel.y() >= border_pixels &&
           pixel.x() < static_cast<double>(intrinsics.width) - border_pixels &&
           pixel.y() < static_cast<double>(intrinsics.height) - border_pixels;
}

} // namespace lio_visual_ba

// ---- Epipolar ----

namespace lio_visual_ba
{

Result<RelativePose> geometry::ComputeRelativePose(const Pose3d& world_from_first_camera,
                                                   const Pose3d& world_from_second_camera)
{
    const Status first_status = geometry::ValidatePose(world_from_first_camera);
    if (!first_status.ok())
    {
        return Result<RelativePose>::Failure(first_status.WithContext("invalid first camera pose"));
    }
    const Status second_status = geometry::ValidatePose(world_from_second_camera);
    if (!second_status.ok())
    {
        return Result<RelativePose>::Failure(
            second_status.WithContext("invalid second camera pose"));
    }
    const Eigen::Matrix3d world_from_first =
        world_from_first_camera.rotation_world_from_local.toRotationMatrix();
    const Eigen::Matrix3d second_from_world =
        world_from_second_camera.rotation_world_from_local.conjugate().toRotationMatrix();
    RelativePose relative;
    relative.rotation_second_from_first = second_from_world * world_from_first;
    relative.translation_second_from_first =
        second_from_world * (world_from_first_camera.translation_world_from_local -
                             world_from_second_camera.translation_world_from_local);
    return Result<RelativePose>::Success(relative);
}

Eigen::Matrix3d geometry::SkewSymmetric(const Eigen::Vector3d& vector)
{
    Eigen::Matrix3d skew;
    skew << 0.0, -vector.z(), vector.y(), vector.z(), 0.0, -vector.x(), -vector.y(), vector.x(),
        0.0;
    return skew;
}

Result<Eigen::Matrix3d> geometry::ComputeEssentialMatrix(const Pose3d& world_from_first_camera,
                                                         const Pose3d& world_from_second_camera)
{
    auto relative =
        geometry::ComputeRelativePose(world_from_first_camera, world_from_second_camera);
    if (!relative.ok())
    {
        return Result<Eigen::Matrix3d>::Failure(relative.status());
    }
    return Result<Eigen::Matrix3d>::Success(
        geometry::SkewSymmetric(relative.value().translation_second_from_first) *
        relative.value().rotation_second_from_first);
}

Result<Eigen::Matrix3d> geometry::ComputeFundamentalMatrix(
    const CameraIntrinsics& first_intrinsics, const Pose3d& world_from_first_camera,
    const CameraIntrinsics& second_intrinsics, const Pose3d& world_from_second_camera)
{
    const Status first_status = geometry::ValidateCameraIntrinsics(first_intrinsics);
    if (!first_status.ok())
    {
        return Result<Eigen::Matrix3d>::Failure(
            first_status.WithContext("invalid first camera intrinsics"));
    }
    const Status second_status = geometry::ValidateCameraIntrinsics(second_intrinsics);
    if (!second_status.ok())
    {
        return Result<Eigen::Matrix3d>::Failure(
            second_status.WithContext("invalid second camera intrinsics"));
    }
    auto essential =
        geometry::ComputeEssentialMatrix(world_from_first_camera, world_from_second_camera);
    if (!essential.ok())
    {
        return Result<Eigen::Matrix3d>::Failure(essential.status());
    }
    return Result<Eigen::Matrix3d>::Success(second_intrinsics.Matrix().inverse().transpose() *
                                            essential.value() *
                                            first_intrinsics.Matrix().inverse());
}

Result<Eigen::Matrix3d> geometry::ComputeFundamentalMatrix(const CameraIntrinsics& intrinsics,
                                                           const Pose3d& world_from_first_camera,
                                                           const Pose3d& world_from_second_camera)
{
    return geometry::ComputeFundamentalMatrix(intrinsics, world_from_first_camera, intrinsics,
                                              world_from_second_camera);
}

Result<double> geometry::SampsonErrorPixels(const Eigen::Vector2d& first_pixel,
                                            const Eigen::Vector2d& second_pixel,
                                            const Eigen::Matrix3d& fundamental_matrix,
                                            double minimum_denominator)
{
    if (!first_pixel.array().isFinite().all() || !second_pixel.array().isFinite().all() ||
        !fundamental_matrix.array().isFinite().all() || !std::isfinite(minimum_denominator) ||
        minimum_denominator <= 0.0)
    {
        return Result<double>::Failure(Status::Error(
            ErrorCode::kInvalidArgument,
            "Sampson inputs must be finite and minimum denominator must be positive"));
    }
    const Eigen::Vector3d first_homogeneous(first_pixel.x(), first_pixel.y(), 1.0);
    const Eigen::Vector3d second_homogeneous(second_pixel.x(), second_pixel.y(), 1.0);
    const Eigen::Vector3d line_in_second = fundamental_matrix * first_homogeneous;
    const Eigen::Vector3d line_in_first = fundamental_matrix.transpose() * second_homogeneous;
    const double denominator_squared =
        line_in_second.head<2>().squaredNorm() + line_in_first.head<2>().squaredNorm();
    if (denominator_squared < minimum_denominator)
    {
        return Result<double>::Failure(
            Status::Error(ErrorCode::kNumericalFailure,
                          "Sampson error is undefined for a degenerate epipolar constraint"));
    }
    const double numerator = std::abs(second_homogeneous.dot(line_in_second));
    return Result<double>::Success(numerator / std::sqrt(denominator_squared));
}

} // namespace lio_visual_ba

// ---- Triangulation ----

namespace lio_visual_ba
{
namespace
{

double Median(std::vector<double> values)
{
    std::sort(values.begin(), values.end());
    const std::size_t middle = values.size() / 2;
    if (values.size() % 2 == 0)
    {
        return 0.5 * (values[middle - 1] + values[middle]);
    }
    return values[middle];
}

} // namespace

Result<TriangulationResult>
geometry::TriangulateMultiView(const CameraIntrinsics& intrinsics,
                               const std::vector<TriangulationObservation>& observations,
                               const TriangulationOptions& options)
{
    const Status intrinsics_status = geometry::ValidateCameraIntrinsics(intrinsics);
    if (!intrinsics_status.ok())
    {
        return Result<TriangulationResult>::Failure(intrinsics_status);
    }
    if (observations.size() < 2)
    {
        return Result<TriangulationResult>::Failure(Status::Error(
            ErrorCode::kInvalidArgument, "triangulation requires at least two observations"));
    }
    if (!std::isfinite(options.minimum_depth) || options.minimum_depth < 0.0 ||
        !std::isfinite(options.minimum_parallax_degrees) ||
        options.minimum_parallax_degrees < 0.0 ||
        std::isnan(options.maximum_reprojection_error_pixels) ||
        options.maximum_reprojection_error_pixels <= 0.0)
    {
        return Result<TriangulationResult>::Failure(
            Status::Error(ErrorCode::kInvalidArgument, "invalid triangulation thresholds"));
    }

    // Each observation contributes two homogeneous projection constraints (A * X = 0).
    // The smallest singular vector gives X; depth/parallax checks below reject bad geometry.
    Eigen::MatrixXd system(static_cast<Eigen::Index>(2 * observations.size()), 4);
    std::vector<Eigen::Vector3d> camera_centers;
    camera_centers.reserve(observations.size());
    for (std::size_t index = 0; index < observations.size(); ++index)
    {
        const TriangulationObservation& observation = observations[index];
        if (!observation.pixel.array().isFinite().all())
        {
            return Result<TriangulationResult>::Failure(
                Status::Error(ErrorCode::kInvalidArgument,
                              "observation " + std::to_string(index) + " has a non-finite pixel"));
        }
        auto projection =
            geometry::CameraProjectionMatrix(intrinsics, observation.world_from_camera);
        if (!projection.ok())
        {
            return Result<TriangulationResult>::Failure(
                projection.status().WithContext("invalid observation " + std::to_string(index)));
        }
        const Eigen::Matrix<double, 3, 4>& matrix = projection.value();
        const Eigen::Index row = static_cast<Eigen::Index>(2 * index);
        system.row(row) = observation.pixel.x() * matrix.row(2) - matrix.row(0);
        system.row(row + 1) = observation.pixel.y() * matrix.row(2) - matrix.row(1);
        camera_centers.push_back(geometry::CameraCenterWorld(observation.world_from_camera));
    }

    const Eigen::JacobiSVD<Eigen::MatrixXd> decomposition(system, Eigen::ComputeFullV);
    const auto& singular_values = decomposition.singularValues();
    constexpr double kMinimumRelativeRank = 1e-12;
    if (singular_values.size() < 3 || singular_values(0) <= 0.0 ||
        singular_values(2) <= kMinimumRelativeRank * singular_values(0))
    {
        return Result<TriangulationResult>::Failure(Status::Error(
            ErrorCode::kNumericalFailure, "DLT system does not define a unique finite point"));
    }
    const Eigen::Vector4d homogeneous = decomposition.matrixV().col(3);
    // Values below this are effectively points at infinity. This matches the
    // Python pipeline and prevents enormous points from entering 3DGS.
    constexpr double kMinimumHomogeneousScale = 1e-9;
    if (!homogeneous.array().isFinite().all() ||
        std::abs(homogeneous.w()) <= kMinimumHomogeneousScale)
    {
        return Result<TriangulationResult>::Failure(
            Status::Error(ErrorCode::kNumericalFailure,
                          "DLT produced a point at infinity or non-finite solution"));
    }

    TriangulationResult result;
    result.point_world = homogeneous.head<3>() / homogeneous.w();
    if (!result.point_world.array().isFinite().all())
    {
        return Result<TriangulationResult>::Failure(
            Status::Error(ErrorCode::kNumericalFailure, "DLT produced a non-finite world point"));
    }
    result.minimum_depth = std::numeric_limits<double>::infinity();
    result.reprojection_errors_pixels.reserve(observations.size());
    for (const TriangulationObservation& observation : observations)
    {
        const Eigen::Vector3d point_camera =
            geometry::WorldToCamera(observation.world_from_camera, result.point_world);
        result.minimum_depth = std::min(result.minimum_depth, point_camera.z());
        if (point_camera.z() <= options.minimum_depth)
        {
            return Result<TriangulationResult>::Failure(
                Status::Error(ErrorCode::kFailedPrecondition,
                              "triangulated point fails the cheirality/depth constraint"));
        }
        auto projection =
            geometry::ProjectCameraPoint(intrinsics, point_camera, options.minimum_depth);
        if (!projection.ok())
        {
            return Result<TriangulationResult>::Failure(projection.status());
        }
        result.reprojection_errors_pixels.push_back(
            (projection.value() - observation.pixel).norm());
    }

    for (std::size_t first = 0; first < camera_centers.size(); ++first)
    {
        const Eigen::Vector3d first_ray = (result.point_world - camera_centers[first]).normalized();
        for (std::size_t second = first + 1; second < camera_centers.size(); ++second)
        {
            const Eigen::Vector3d second_ray =
                (result.point_world - camera_centers[second]).normalized();
            const double cosine = std::clamp(first_ray.dot(second_ray), -1.0, 1.0);
            constexpr double kRadiansToDegrees = 180.0 / 3.14159265358979323846;
            result.maximum_parallax_degrees =
                std::max(result.maximum_parallax_degrees, std::acos(cosine) * kRadiansToDegrees);
        }
    }
    result.median_reprojection_error_pixels = Median(result.reprojection_errors_pixels);
    result.maximum_reprojection_error_pixels = *std::max_element(
        result.reprojection_errors_pixels.begin(), result.reprojection_errors_pixels.end());

    if (result.maximum_parallax_degrees < options.minimum_parallax_degrees)
    {
        return Result<TriangulationResult>::Failure(Status::Error(
            ErrorCode::kFailedPrecondition, "triangulated point has insufficient parallax"));
    }
    if (result.maximum_reprojection_error_pixels > options.maximum_reprojection_error_pixels)
    {
        return Result<TriangulationResult>::Failure(
            Status::Error(ErrorCode::kFailedPrecondition,
                          "triangulated point exceeds the reprojection-error threshold"));
    }
    return Result<TriangulationResult>::Success(std::move(result));
}

} // namespace lio_visual_ba
