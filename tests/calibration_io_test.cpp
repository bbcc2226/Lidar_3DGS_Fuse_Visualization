#include "dataset_io.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>

#include <gtest/gtest.h>

#include "geometry.hpp"

namespace lio_visual_ba
{
namespace
{

class CalibrationTemporaryDirectory
{
  public:
    CalibrationTemporaryDirectory()
    {
        const auto suffix = std::chrono::steady_clock::now().time_since_epoch().count();
        path_ = std::filesystem::temp_directory_path() /
                ("lio_visual_ba_calibration_test_" + std::to_string(suffix));
        std::filesystem::create_directories(path_);
    }
    ~CalibrationTemporaryDirectory()
    {
        std::error_code ignored;
        std::filesystem::remove_all(path_, ignored);
    }
    std::filesystem::path Write(const std::string& name, const std::string& contents) const
    {
        const auto path = path_ / name;
        std::ofstream(path) << contents;
        return path;
    }

  private:
    std::filesystem::path path_;
};

TEST(CalibrationIoTest, LoadsLegacyCommaDelimitedIntrinsics)
{
    CalibrationTemporaryDirectory directory;
    const auto path =
        directory.Write("intrinsics.txt", "633.8, 0, 797.4\n0, 634.6, 601.7\n0, 0, 1\n");

    auto result = DatasetIO::LoadCameraIntrinsics(path, 1600, 1200);

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    EXPECT_DOUBLE_EQ(result.value().fx, 633.8);
    EXPECT_DOUBLE_EQ(result.value().cy, 601.7);
    EXPECT_EQ(result.value().width, 1600);
    EXPECT_TRUE(result.value().Matrix().isApprox(
        (Eigen::Matrix3d() << 633.8, 0.0, 797.4, 0.0, 634.6, 601.7, 0.0, 0.0, 1.0).finished()));
}

TEST(CalibrationIoTest, RejectsMalformedOrUnsupportedIntrinsics)
{
    CalibrationTemporaryDirectory directory;
    auto short_result =
        DatasetIO::LoadCameraIntrinsics(directory.Write("short.txt", "1 0 2\n0 1 2\n"));
    EXPECT_FALSE(short_result.ok());
    EXPECT_EQ(short_result.status().code(), ErrorCode::kParseError);

    auto skew_result = DatasetIO::LoadCameraIntrinsics(
        directory.Write("skew.txt", "600 2 400\n0 600 300\n0 0 1\n"));
    EXPECT_FALSE(skew_result.ok());
    EXPECT_EQ(skew_result.status().code(), ErrorCode::kFailedPrecondition);
}

TEST(CalibrationIoTest, LoadsAndInvertsCameraFromLidarExtrinsic)
{
    CalibrationTemporaryDirectory directory;
    const auto path = directory.Write(
        "extrinsic.json", R"({"T_camera_lidar":[[0,-1,0,1],[1,0,0,2],[0,0,1,3],[0,0,0,1]]})");

    auto result = DatasetIO::LoadLidarFromCameraExtrinsic(path);

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    const Eigen::Vector3d point_lidar(4.0, 5.0, 6.0);
    const Eigen::Vector3d point_camera(-4.0, 6.0, 9.0);
    EXPECT_TRUE(
        geometry::TransformPoint(result.value(), point_camera).isApprox(point_lidar, 1e-12));
}

TEST(CalibrationIoTest, RejectsInvalidExtrinsicGeometry)
{
    CalibrationTemporaryDirectory directory;
    auto result = DatasetIO::LoadLidarFromCameraExtrinsic(directory.Write(
        "extrinsic.json", R"({"T_camera_lidar":[[2,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]})"));

    ASSERT_FALSE(result.ok());
    EXPECT_EQ(result.status().code(), ErrorCode::kFailedPrecondition);
}

TEST(CalibrationIoTest, ReportsMissingJsonMember)
{
    CalibrationTemporaryDirectory directory;
    auto result = DatasetIO::LoadLidarFromCameraExtrinsic(
        directory.Write("extrinsic.json", R"({"T_lidar_camera":[]})"));

    ASSERT_FALSE(result.ok());
    EXPECT_EQ(result.status().code(), ErrorCode::kParseError);
}

} // namespace
} // namespace lio_visual_ba
