#include "geometry.hpp"

#include <limits>

#include <gtest/gtest.h>

namespace lio_visual_ba
{
namespace
{

CameraIntrinsics TestIntrinsics()
{
    CameraIntrinsics intrinsics;
    intrinsics.fx = 600.0;
    intrinsics.fy = 620.0;
    intrinsics.cx = 400.0;
    intrinsics.cy = 300.0;
    intrinsics.width = 800;
    intrinsics.height = 600;
    return intrinsics;
}

TEST(CameraModelTest, ProjectAndUnprojectRoundTrip)
{
    const CameraIntrinsics intrinsics = TestIntrinsics();
    const Eigen::Vector3d point_camera(1.0, -0.5, 4.0);

    auto pixel = geometry::ProjectCameraPoint(intrinsics, point_camera);
    ASSERT_TRUE(pixel.ok()) << pixel.status().ToString();
    EXPECT_TRUE(pixel.value().isApprox(Eigen::Vector2d(550.0, 222.5), 1e-12));

    auto restored = geometry::UnprojectPixel(intrinsics, pixel.value(), point_camera.z());
    ASSERT_TRUE(restored.ok()) << restored.status().ToString();
    EXPECT_TRUE(restored.value().isApprox(point_camera, 1e-12));
}

TEST(CameraModelTest, WorldProjectionUsesWorldFromCameraConvention)
{
    const CameraIntrinsics intrinsics = TestIntrinsics();
    Pose3d world_from_camera;
    world_from_camera.rotation_world_from_local =
        Eigen::AngleAxisd(0.5 * 3.14159265358979323846, Eigen::Vector3d::UnitY());
    world_from_camera.translation_world_from_local = Eigen::Vector3d(2.0, 3.0, 4.0);
    const Eigen::Vector3d point_camera(0.5, 0.25, 5.0);
    const Eigen::Vector3d point_world = world_from_camera.rotation_world_from_local * point_camera +
                                        world_from_camera.translation_world_from_local;

    auto projection = geometry::ProjectWorldPoint(intrinsics, world_from_camera, point_world);

    ASSERT_TRUE(projection.ok()) << projection.status().ToString();
    EXPECT_NEAR(projection.value().depth, 5.0, 1e-12);
    EXPECT_TRUE(projection.value().pixel.isApprox(Eigen::Vector2d(460.0, 331.0), 1e-10));
    EXPECT_TRUE(
        geometry::CameraCenterWorld(world_from_camera).isApprox(Eigen::Vector3d(2.0, 3.0, 4.0)));
}

TEST(CameraModelTest, ProjectionMatrixMatchesDirectProjection)
{
    const CameraIntrinsics intrinsics = TestIntrinsics();
    Pose3d world_from_camera;
    world_from_camera.translation_world_from_local = Eigen::Vector3d(1.0, -2.0, 0.5);
    const Eigen::Vector3d point_world(2.0, -1.0, 5.5);

    auto direct = geometry::ProjectWorldPoint(intrinsics, world_from_camera, point_world);
    auto matrix = geometry::CameraProjectionMatrix(intrinsics, world_from_camera);

    ASSERT_TRUE(direct.ok());
    ASSERT_TRUE(matrix.ok());
    const Eigen::Vector3d homogeneous = matrix.value() * point_world.homogeneous();
    EXPECT_TRUE((homogeneous.head<2>() / homogeneous.z()).isApprox(direct.value().pixel, 1e-12));
}

TEST(CameraModelTest, RejectsBehindCameraAndInvalidInputs)
{
    CameraIntrinsics intrinsics = TestIntrinsics();
    auto behind = geometry::ProjectCameraPoint(intrinsics, Eigen::Vector3d(0.0, 0.0, -1.0));
    ASSERT_FALSE(behind.ok());
    EXPECT_EQ(behind.status().code(), ErrorCode::kFailedPrecondition);

    intrinsics.fx = 0.0;
    EXPECT_FALSE(geometry::ProjectCameraPoint(intrinsics, Eigen::Vector3d(0.0, 0.0, 1.0)).ok());
    EXPECT_FALSE(geometry::UnprojectPixel(TestIntrinsics(), Eigen::Vector2d::Zero(), 0.0).ok());
}

TEST(CameraModelTest, ImageBoundsUseHalfOpenPixelDomainAndBorder)
{
    const CameraIntrinsics intrinsics = TestIntrinsics();
    EXPECT_TRUE(geometry::IsPixelInsideImage(intrinsics, Eigen::Vector2d(0.0, 0.0)));
    EXPECT_TRUE(geometry::IsPixelInsideImage(intrinsics, Eigen::Vector2d(799.9, 599.9)));
    EXPECT_FALSE(geometry::IsPixelInsideImage(intrinsics, Eigen::Vector2d(800.0, 300.0)));
    EXPECT_FALSE(geometry::IsPixelInsideImage(intrinsics, Eigen::Vector2d(4.0, 20.0), 5.0));
    EXPECT_FALSE(geometry::IsPixelInsideImage(
        intrinsics, Eigen::Vector2d(std::numeric_limits<double>::quiet_NaN(), 10.0)));
}

} // namespace
} // namespace lio_visual_ba
