#include "ReconstructionExporter.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>

#include <gtest/gtest.h>

namespace lio_visual_ba
{
namespace
{

class ValidationDirectory
{
  public:
    ValidationDirectory()
    {
        path_ = std::filesystem::temp_directory_path() /
                ("colmap_validation_" +
                 std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    }
    ~ValidationDirectory()
    {
        std::error_code ignored;
        std::filesystem::remove_all(path_, ignored);
    }
    const std::filesystem::path& path() const { return path_; }

  private:
    std::filesystem::path path_;
};

Reconstruction ValidationSample()
{
    Reconstruction reconstruction;
    reconstruction.intrinsics = {600.0, 600.0, 400.0, 300.0, 800, 600};
    CameraFrame first;
    first.id = CameraId(0);
    first.image_name = "a.png";
    CameraFrame second;
    second.id = CameraId(1);
    second.image_name = "b.png";
    second.initial_world_from_camera.translation_world_from_local.x() = 1.0;
    reconstruction.cameras = {first, second};
    Landmark point;
    point.id = LandmarkId(0);
    point.source_track_id = TrackId(0);
    point.position_world = {0.0, 0.0, 5.0};
    point.quality.median_reprojection_error_pixels = 0.2;
    point.observations = {{CameraId(0), FeatureId(0), {400.0, 300.0}},
                          {CameraId(1), FeatureId(0), {280.0, 300.0}}};
    reconstruction.landmarks = {point};
    return reconstruction;
}

TEST(ReconstructionValidatorTest, ReopensAndValidatesGeneratedModel)
{
    ValidationDirectory directory;
    ASSERT_TRUE(
        ReconstructionExporter::WriteColmapTextReconstruction(directory.path(), ValidationSample())
            .ok());

    auto report = ReconstructionExporter::ValidateColmapTextReconstruction(directory.path());

    ASSERT_TRUE(report.ok()) << report.status().ToString();
    EXPECT_EQ(report.value().cameras, 1U);
    EXPECT_EQ(report.value().images, 2U);
    EXPECT_EQ(report.value().points3d, 1U);
    EXPECT_EQ(report.value().observations, 2U);
}

TEST(ReconstructionValidatorTest, DetectsNonReciprocalPointTrack)
{
    ValidationDirectory directory;
    ASSERT_TRUE(
        ReconstructionExporter::WriteColmapTextReconstruction(directory.path(), ValidationSample())
            .ok());
    std::ofstream points(directory.path() / "points3D.txt", std::ios::trunc);
    points << "1 0 0 5 0 0 0 0.2 1 0\n";
    points.close();

    auto report = ReconstructionExporter::ValidateColmapTextReconstruction(directory.path());

    EXPECT_FALSE(report.ok());
    EXPECT_EQ(report.status().code(), ErrorCode::kDataLoss);
}

TEST(ReconstructionValidatorTest, DetectsInvalidPointIndex)
{
    ValidationDirectory directory;
    ASSERT_TRUE(
        ReconstructionExporter::WriteColmapTextReconstruction(directory.path(), ValidationSample())
            .ok());
    std::ofstream points(directory.path() / "points3D.txt", std::ios::trunc);
    points << "1 0 0 5 0 0 0 0.2 1 99 2 0\n";
    points.close();

    auto report = ReconstructionExporter::ValidateColmapTextReconstruction(directory.path());

    EXPECT_FALSE(report.ok());
    EXPECT_EQ(report.status().code(), ErrorCode::kDataLoss);
}

TEST(ReconstructionValidatorTest, ReportsMissingRequiredFiles)
{
    ValidationDirectory directory;
    std::filesystem::create_directories(directory.path());

    auto report = ReconstructionExporter::ValidateColmapTextReconstruction(directory.path());

    EXPECT_FALSE(report.ok());
    EXPECT_EQ(report.status().code(), ErrorCode::kNotFound);
}

} // namespace
} // namespace lio_visual_ba
