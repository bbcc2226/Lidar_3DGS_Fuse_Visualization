#pragma once

// Stateless pose, camera, epipolar, and triangulation math. Transforms use
// outer_from_inner names; projection converts world points into camera coordinates.

#include "status.hpp"
#include "types.hpp"
#include <Eigen/Core>
#include <limits>
#include <vector>

namespace lio_visual_ba
{

struct ProjectionResult
{
    Eigen::Vector2d pixel = Eigen::Vector2d::Zero();
    double depth = 0.0;
};

struct RelativePose
{
    Eigen::Matrix3d rotation_second_from_first = Eigen::Matrix3d::Identity();
    Eigen::Vector3d translation_second_from_first = Eigen::Vector3d::Zero();
};

struct TriangulationObservation
{
    Pose3d world_from_camera;
    Eigen::Vector2d pixel = Eigen::Vector2d::Zero();
};

struct TriangulationOptions
{
    double minimum_depth = 1e-6;
    double minimum_parallax_degrees = 0.0;
    double maximum_reprojection_error_pixels = std::numeric_limits<double>::infinity();
};

struct TriangulationResult
{
    Eigen::Vector3d point_world = Eigen::Vector3d::Zero();
    double minimum_depth = 0.0;
    double maximum_parallax_degrees = 0.0;
    double median_reprojection_error_pixels = 0.0;
    double maximum_reprojection_error_pixels = 0.0;
    std::vector<double> reprojection_errors_pixels;
};

// Stateless geometry algorithms; common result types remain in lio_visual_ba.
namespace geometry
{

Status ValidatePose(const Pose3d& pose, double quaternion_norm_tolerance = 1e-6);

Result<Pose3d> NormalizePose(const Pose3d& pose);

Pose3d InversePose(const Pose3d& outer_from_inner);

// Compose outer_from_middle with middle_from_inner to produce
// outer_from_inner.
Pose3d ComposePoses(const Pose3d& outer_from_middle, const Pose3d& middle_from_inner);

Eigen::Vector3d TransformPoint(const Pose3d& outer_from_inner, const Eigen::Vector3d& point_inner);

// Interpolate two poses at query_time_seconds. Query time must lie in the
// closed interval formed by the two sample times. Input order may be either
// ascending or descending.
Result<Pose3d> InterpolatePose(const Pose3d& first_pose, double first_time_seconds,
                               const Pose3d& second_pose, double second_time_seconds,
                               double query_time_seconds);

Status ValidateCameraIntrinsics(const CameraIntrinsics& intrinsics);

// Convert between pixel coordinates and camera-frame coordinates. Projection
// rejects points at or behind minimum_depth; unprojection requires depth > 0.
Result<Eigen::Vector2d> ProjectCameraPoint(const CameraIntrinsics& intrinsics,
                                           const Eigen::Vector3d& point_camera,
                                           double minimum_depth = 1e-6);

Result<Eigen::Vector3d> UnprojectPixel(const CameraIntrinsics& intrinsics,
                                       const Eigen::Vector2d& pixel, double depth);

Eigen::Vector3d CameraCenterWorld(const Pose3d& world_from_camera);

Eigen::Vector3d WorldToCamera(const Pose3d& world_from_camera, const Eigen::Vector3d& point_world);

Result<ProjectionResult> ProjectWorldPoint(const CameraIntrinsics& intrinsics,
                                           const Pose3d& world_from_camera,
                                           const Eigen::Vector3d& point_world,
                                           double minimum_depth = 1e-6);

// Return K [R_camera_from_world | t_camera_from_world].
Result<Eigen::Matrix<double, 3, 4>> CameraProjectionMatrix(const CameraIntrinsics& intrinsics,
                                                           const Pose3d& world_from_camera);

bool IsPixelInsideImage(const CameraIntrinsics& intrinsics, const Eigen::Vector2d& pixel,
                        double border_pixels = 0.0);

Result<RelativePose> ComputeRelativePose(const Pose3d& world_from_first_camera,
                                         const Pose3d& world_from_second_camera);

Eigen::Matrix3d SkewSymmetric(const Eigen::Vector3d& vector);

// E = [t_second_from_first]_x R_second_from_first, satisfying
// normalized_second.transpose() * E * normalized_first = 0.
Result<Eigen::Matrix3d> ComputeEssentialMatrix(const Pose3d& world_from_first_camera,
                                               const Pose3d& world_from_second_camera);

// Supports different intrinsics in the two views. The single-intrinsics
// overload is convenient for this pipeline's shared calibrated camera.
Result<Eigen::Matrix3d> ComputeFundamentalMatrix(const CameraIntrinsics& first_intrinsics,
                                                 const Pose3d& world_from_first_camera,
                                                 const CameraIntrinsics& second_intrinsics,
                                                 const Pose3d& world_from_second_camera);

Result<Eigen::Matrix3d> ComputeFundamentalMatrix(const CameraIntrinsics& intrinsics,
                                                 const Pose3d& world_from_first_camera,
                                                 const Pose3d& world_from_second_camera);

// First-order geometric distance in pixels. This is sqrt of the conventional
// squared Sampson distance and matches the legacy Python threshold semantics.
Result<double> SampsonErrorPixels(const Eigen::Vector2d& first_pixel,
                                  const Eigen::Vector2d& second_pixel,
                                  const Eigen::Matrix3d& fundamental_matrix,
                                  double minimum_denominator = 1e-12);

// Linear multi-view DLT with post-triangulation geometric validation.
// At least two observations are required.
Result<TriangulationResult>
TriangulateMultiView(const CameraIntrinsics& intrinsics,
                     const std::vector<TriangulationObservation>& observations,
                     const TriangulationOptions& options = {});

} // namespace geometry

} // namespace lio_visual_ba
