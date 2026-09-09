#include "geometry.hpp"

#include <limits>

#include <gtest/gtest.h>

namespace lio_visual_ba
{
namespace
{

CameraIntrinsics Intrinsics(double focal_length = 600.0)
{
    CameraIntrinsics intrinsics;
    intrinsics.fx = focal_length;
    intrinsics.fy = focal_length;
    intrinsics.cx = 400.0;
    intrinsics.cy = 300.0;
    intrinsics.width = 800;
    intrinsics.height = 600;
    return intrinsics;
}

TEST(EpipolarTest, RelativePoseTransformsFirstCameraPointIntoSecondCamera)
{
    Pose3d world_from_first;
    world_from_first.rotation_world_from_local = Eigen::AngleAxisd(0.2, Eigen::Vector3d::UnitY());
    world_from_first.translation_world_from_local = Eigen::Vector3d(1.0, 2.0, 3.0);
    Pose3d world_from_second;
    world_from_second.rotation_world_from_local = Eigen::AngleAxisd(-0.3, Eigen::Vector3d::UnitX());
    world_from_second.translation_world_from_local = Eigen::Vector3d(-1.0, 0.5, 0.0);
    const Eigen::Vector3d point_first(0.4, -0.2, 5.0);
    const Eigen::Vector3d point_world = world_from_first.rotation_world_from_local * point_first +
                                        world_from_first.translation_world_from_local;

    auto relative = geometry::ComputeRelativePose(world_from_first, world_from_second);

    ASSERT_TRUE(relative.ok()) << relative.status().ToString();
    const Eigen::Vector3d transformed = relative.value().rotation_second_from_first * point_first +
                                        relative.value().translation_second_from_first;
    EXPECT_TRUE(
        transformed.isApprox(geometry::WorldToCamera(world_from_second, point_world), 1e-12));
}

TEST(EpipolarTest, FundamentalMatrixSatisfiesSyntheticCorrespondence)
{
    const CameraIntrinsics intrinsics = Intrinsics();
    Pose3d world_from_first;
    Pose3d world_from_second;
    world_from_second.translation_world_from_local.x() = 1.0;
    const Eigen::Vector3d point_world(0.3, 0.4, 5.0);
    auto first = geometry::ProjectWorldPoint(intrinsics, world_from_first, point_world);
    auto second = geometry::ProjectWorldPoint(intrinsics, world_from_second, point_world);
    auto fundamental =
        geometry::ComputeFundamentalMatrix(intrinsics, world_from_first, world_from_second);
    ASSERT_TRUE(first.ok() && second.ok() && fundamental.ok());

    auto error = geometry::SampsonErrorPixels(first.value().pixel, second.value().pixel,
                                              fundamental.value());

    ASSERT_TRUE(error.ok()) << error.status().ToString();
    EXPECT_NEAR(error.value(), 0.0, 1e-12);
    auto perturbed = geometry::SampsonErrorPixels(first.value().pixel,
                                                  second.value().pixel + Eigen::Vector2d(0.0, 10.0),
                                                  fundamental.value());
    ASSERT_TRUE(perturbed.ok());
    EXPECT_GT(perturbed.value(), 1.0);
}

TEST(EpipolarTest, SupportsDifferentIntrinsicsPerView)
{
    const CameraIntrinsics first_intrinsics = Intrinsics(500.0);
    CameraIntrinsics second_intrinsics = Intrinsics(900.0);
    second_intrinsics.cx = 350.0;
    Pose3d first_pose;
    Pose3d second_pose;
    second_pose.translation_world_from_local = Eigen::Vector3d(0.8, 0.1, 0.0);
    second_pose.rotation_world_from_local = Eigen::AngleAxisd(0.1, Eigen::Vector3d::UnitY());
    const Eigen::Vector3d point_world(0.2, -0.3, 6.0);
    auto first = geometry::ProjectWorldPoint(first_intrinsics, first_pose, point_world);
    auto second = geometry::ProjectWorldPoint(second_intrinsics, second_pose, point_world);
    auto fundamental = geometry::ComputeFundamentalMatrix(first_intrinsics, first_pose,
                                                          second_intrinsics, second_pose);
    ASSERT_TRUE(first.ok() && second.ok() && fundamental.ok());

    auto error = geometry::SampsonErrorPixels(first.value().pixel, second.value().pixel,
                                              fundamental.value());
    ASSERT_TRUE(error.ok()) << error.status().ToString();
    EXPECT_NEAR(error.value(), 0.0, 1e-11);
}

TEST(EpipolarTest, DegenerateCoincidentCamerasProduceNoSampsonDistance)
{
    auto fundamental = geometry::ComputeFundamentalMatrix(Intrinsics(), Pose3d{}, Pose3d{});
    ASSERT_TRUE(fundamental.ok());
    EXPECT_TRUE(fundamental.value().isZero(1e-15));

    auto error = geometry::SampsonErrorPixels(Eigen::Vector2d::Zero(), Eigen::Vector2d::Zero(),
                                              fundamental.value());
    ASSERT_FALSE(error.ok());
    EXPECT_EQ(error.status().code(), ErrorCode::kNumericalFailure);
}

TEST(EpipolarTest, RejectsNonFiniteCorrespondence)
{
    Eigen::Vector2d invalid = Eigen::Vector2d::Zero();
    invalid.x() = std::numeric_limits<double>::quiet_NaN();
    auto error =
        geometry::SampsonErrorPixels(invalid, Eigen::Vector2d::Zero(), Eigen::Matrix3d::Identity());
    ASSERT_FALSE(error.ok());
    EXPECT_EQ(error.status().code(), ErrorCode::kInvalidArgument);
}

} // namespace
} // namespace lio_visual_ba
