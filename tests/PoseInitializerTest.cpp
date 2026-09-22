#include "Mapper.hpp"

#include <gtest/gtest.h>

namespace lio_visual_ba
{
namespace
{

TEST(PoseInitializerTest, AppliesOffsetInterpolationAndExtrinsic)
{
    std::vector<TimedPose> trajectory(2);
    trajectory[0].timestamp_seconds = 10.0;
    trajectory[1].timestamp_seconds = 20.0;
    trajectory[1].world_from_lidar.translation_world_from_local.x() = 10.0;

    Pose3d lidar_from_camera;
    lidar_from_camera.translation_world_from_local.y() = 2.0;
    CameraFrame camera;
    camera.id = CameraId(0);
    camera.timestamp_seconds = 15.4;
    std::vector<CameraFrame> cameras{camera};

    const Status status =
        Mapper::InitializeCameraPoses(cameras, trajectory, lidar_from_camera, -0.4);

    ASSERT_TRUE(status.ok()) << status.ToString();
    EXPECT_TRUE(cameras[0].initial_world_from_camera.translation_world_from_local.isApprox(
        Eigen::Vector3d(5.0, 2.0, 0.0), 1e-12));
    EXPECT_TRUE(cameras[0].optimized_world_from_camera.translation_world_from_local.isApprox(
        Eigen::Vector3d(5.0, 2.0, 0.0), 1e-12));
    EXPECT_FALSE(cameras[0].has_optimized_pose);
}

TEST(PoseInitializerTest, ClampsQueriesAtTrajectoryEndpoints)
{
    std::vector<TimedPose> trajectory(2);
    trajectory[0].timestamp_seconds = 10.0;
    trajectory[0].world_from_lidar.translation_world_from_local.x() = 3.0;
    trajectory[1].timestamp_seconds = 20.0;
    trajectory[1].world_from_lidar.translation_world_from_local.x() = 7.0;
    std::vector<CameraFrame> cameras(2);
    cameras[0].timestamp_seconds = 0.0;
    cameras[1].timestamp_seconds = 30.0;

    const Status status = Mapper::InitializeCameraPoses(cameras, trajectory, Pose3d{}, 0.0);

    ASSERT_TRUE(status.ok()) << status.ToString();
    EXPECT_DOUBLE_EQ(cameras[0].initial_world_from_camera.translation_world_from_local.x(), 3.0);
    EXPECT_DOUBLE_EQ(cameras[1].initial_world_from_camera.translation_world_from_local.x(), 7.0);
}

TEST(PoseInitializerTest, RejectsUnsortedTrajectory)
{
    std::vector<TimedPose> trajectory(2);
    trajectory[0].timestamp_seconds = 2.0;
    trajectory[1].timestamp_seconds = 1.0;
    std::vector<CameraFrame> cameras(1);

    const Status status = Mapper::InitializeCameraPoses(cameras, trajectory, Pose3d{}, 0.0);

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), ErrorCode::kDataLoss);
}

} // namespace
} // namespace lio_visual_ba
