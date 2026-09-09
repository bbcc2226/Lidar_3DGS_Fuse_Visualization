#include "geometry.hpp"

#include <cmath>
#include <limits>

#include <gtest/gtest.h>

namespace lio_visual_ba
{
namespace
{

constexpr double kTolerance = 1e-10;

TEST(PoseTest, InverseRoundTripRestoresPoint)
{
    Pose3d world_from_camera;
    world_from_camera.rotation_world_from_local = Eigen::AngleAxisd(0.7, Eigen::Vector3d::UnitZ());
    world_from_camera.translation_world_from_local = Eigen::Vector3d(1.0, -2.0, 0.5);

    const Eigen::Vector3d point_camera(0.2, 0.4, 3.0);
    const Eigen::Vector3d point_world = geometry::TransformPoint(world_from_camera, point_camera);
    const Eigen::Vector3d restored =
        geometry::TransformPoint(geometry::InversePose(world_from_camera), point_world);

    EXPECT_TRUE(restored.isApprox(point_camera, kTolerance));
}

TEST(PoseTest, CompositionMatchesSequentialTransformation)
{
    Pose3d world_from_lidar;
    world_from_lidar.rotation_world_from_local = Eigen::AngleAxisd(0.3, Eigen::Vector3d::UnitY());
    world_from_lidar.translation_world_from_local = Eigen::Vector3d(3.0, 1.0, -0.2);

    Pose3d lidar_from_camera;
    lidar_from_camera.rotation_world_from_local = Eigen::AngleAxisd(-0.2, Eigen::Vector3d::UnitX());
    lidar_from_camera.translation_world_from_local = Eigen::Vector3d(0.1, 0.0, 0.2);

    const Eigen::Vector3d point_camera(0.5, -0.4, 2.0);
    const Pose3d world_from_camera = geometry::ComposePoses(world_from_lidar, lidar_from_camera);

    const Eigen::Vector3d sequential = geometry::TransformPoint(
        world_from_lidar, geometry::TransformPoint(lidar_from_camera, point_camera));
    const Eigen::Vector3d composed = geometry::TransformPoint(world_from_camera, point_camera);

    EXPECT_TRUE(composed.isApprox(sequential, kTolerance));
}

TEST(PoseTest, InterpolationUsesLinearTranslationAndQuaternionSlerp)
{
    Pose3d first;
    Pose3d second;
    second.rotation_world_from_local =
        Eigen::AngleAxisd(3.14159265358979323846, Eigen::Vector3d::UnitZ());
    second.translation_world_from_local = Eigen::Vector3d(10.0, 4.0, -2.0);

    auto result = geometry::InterpolatePose(first, 2.0, second, 6.0, 4.0);
    ASSERT_TRUE(result.ok()) << result.status().ToString();

    const Pose3d& midpoint = result.value();
    EXPECT_TRUE(midpoint.translation_world_from_local.isApprox(Eigen::Vector3d(5.0, 2.0, -1.0),
                                                               kTolerance));
    const Eigen::Vector3d rotated = midpoint.rotation_world_from_local * Eigen::Vector3d::UnitX();
    EXPECT_NEAR(rotated.x(), 0.0, kTolerance);
    EXPECT_NEAR(std::abs(rotated.y()), 1.0, kTolerance);
}

TEST(PoseTest, InterpolationAcceptsDescendingSampleTimes)
{
    Pose3d first;
    first.translation_world_from_local.x() = 10.0;
    Pose3d second;

    auto result = geometry::InterpolatePose(first, 10.0, second, 0.0, 2.5);
    ASSERT_TRUE(result.ok()) << result.status().ToString();
    EXPECT_NEAR(result.value().translation_world_from_local.x(), 2.5, kTolerance);
}

TEST(PoseTest, InterpolationRejectsQueryOutsideInterval)
{
    auto result = geometry::InterpolatePose(Pose3d{}, 0.0, Pose3d{}, 1.0, 2.0);
    ASSERT_FALSE(result.ok());
    EXPECT_EQ(result.status().code(), ErrorCode::kInvalidArgument);
}

TEST(PoseTest, ValidationRejectsNonFiniteTranslation)
{
    Pose3d pose;
    pose.translation_world_from_local.x() = std::numeric_limits<double>::quiet_NaN();

    const Status status = geometry::ValidatePose(pose);
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), ErrorCode::kInvalidArgument);
}

TEST(PoseTest, NormalizePoseRejectsZeroQuaternion)
{
    Pose3d pose;
    pose.rotation_world_from_local.coeffs().setZero();

    auto result = geometry::NormalizePose(pose);
    EXPECT_FALSE(result.ok());
    EXPECT_EQ(result.status().code(), ErrorCode::kInvalidArgument);
}

} // namespace
} // namespace lio_visual_ba
