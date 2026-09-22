#include "DatasetIo.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>

#include <gtest/gtest.h>

namespace lio_visual_ba
{
namespace
{

std::filesystem::path WriteTrajectory(const std::string& contents)
{
    const auto path =
        std::filesystem::temp_directory_path() /
        ("lio_trajectory_" +
         std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".jsonl");
    std::ofstream(path) << contents;
    return path;
}

TEST(TrajectoryIoTest, LoadsMultilineObjectsSortsAndNormalizes)
{
    const auto path = WriteTrajectory(
        R"({"timestamp":2,"lio_pose":{"translation":[2,0,0],"quaternion_xyzw":[0,0,0,2]}}
{
 "timestamp": 1,
 "lio_pose": {"translation":[1,0,0], "quaternion_xyzw":[0,0,0,1]}
})");
    auto result = DatasetIO::LoadLioTrajectory(path);
    std::filesystem::remove(path);

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    ASSERT_EQ(result.value().size(), 2U);
    EXPECT_DOUBLE_EQ(result.value()[0].timestamp_seconds, 1.0);
    EXPECT_DOUBLE_EQ(result.value()[1].world_from_lidar.rotation_world_from_local.norm(), 1.0);
}

TEST(TrajectoryIoTest, RejectsDuplicateTimestamps)
{
    const auto path = WriteTrajectory(
        R"({"timestamp":1,"lio_pose":{"translation":[0,0,0],"quaternion_xyzw":[0,0,0,1]}}
{"timestamp":1,"lio_pose":{"translation":[1,0,0],"quaternion_xyzw":[0,0,0,1]}})");
    auto result = DatasetIO::LoadLioTrajectory(path);
    std::filesystem::remove(path);

    ASSERT_FALSE(result.ok());
    EXPECT_EQ(result.status().code(), ErrorCode::kDataLoss);
}

TEST(TrajectoryIoTest, RejectsMalformedPose)
{
    const auto path = WriteTrajectory(
        R"({"timestamp":1,"lio_pose":{"translation":[0,0],"quaternion_xyzw":[0,0,0,1]}}
{"timestamp":2,"lio_pose":{"translation":[0,0,0],"quaternion_xyzw":[0,0,0,1]}})");
    auto result = DatasetIO::LoadLioTrajectory(path);
    std::filesystem::remove(path);

    ASSERT_FALSE(result.ok());
    EXPECT_EQ(result.status().code(), ErrorCode::kParseError);
}

} // namespace
} // namespace lio_visual_ba
