#include "ReconstructionExporter.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>

#include <gtest/gtest.h>

namespace lio_visual_ba
{
namespace
{

class ColmapDirectory
{
  public:
    ColmapDirectory()
    {
        path_ = std::filesystem::temp_directory_path() /
                ("colmap_text_" +
                 std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    }
    ~ColmapDirectory()
    {
        std::error_code ignored;
        std::filesystem::remove_all(path_, ignored);
    }
    const std::filesystem::path& path() const { return path_; }
    std::string Read(const std::string& name) const
    {
        std::ifstream input(path_ / name);
        return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
    }

  private:
    std::filesystem::path path_;
};

Reconstruction SampleReconstruction()
{
    Reconstruction reconstruction;
    reconstruction.intrinsics.fx = 600.0;
    reconstruction.intrinsics.fy = 610.0;
    reconstruction.intrinsics.cx = 400.0;
    reconstruction.intrinsics.cy = 300.0;
    reconstruction.intrinsics.width = 800;
    reconstruction.intrinsics.height = 600;
    CameraFrame first;
    first.id = CameraId(0);
    first.image_name = "first.png";
    CameraFrame second;
    second.id = CameraId(1);
    second.image_name = "second.png";
    second.initial_world_from_camera.translation_world_from_local.x() = 1.0;
    reconstruction.cameras = {first, second};

    Landmark landmark;
    landmark.id = LandmarkId(0);
    landmark.source_track_id = TrackId(4);
    landmark.position_world = {0.0, 0.0, 5.0};
    landmark.color_rgb = {12, 34, 56};
    landmark.quality.median_reprojection_error_pixels = 0.25;
    landmark.observations = {{CameraId(0), FeatureId(7), {400.0, 300.0}},
                             {CameraId(1), FeatureId(3), {280.0, 300.0}}};
    reconstruction.landmarks = {landmark};
    return reconstruction;
}

TEST(ColmapTextWriterTest, WritesStrictCrossReferencedTextModel)
{
    ColmapDirectory output;
    const Status status = ReconstructionExporter::WriteColmapTextReconstruction(
        output.path(), SampleReconstruction());

    ASSERT_TRUE(status.ok()) << status.ToString();
    EXPECT_NE(output.Read("cameras.txt").find("1 PINHOLE 800 600 600 610 400 300"),
              std::string::npos);
    const std::string images = output.Read("images.txt");
    EXPECT_NE(images.find("1 first.png\n400 300 1"), std::string::npos);
    EXPECT_NE(images.find("2 1 -0 -0 -0 -1 -0 -0 1 second.png"), std::string::npos);
    const std::string points = output.Read("points3D.txt");
    EXPECT_NE(points.find("1 0 0 5 12 34 56 0.25 1 0 2 0"), std::string::npos);
}

TEST(ColmapTextWriterTest, PointIndicesFollowSortedFeatureIds)
{
    ColmapDirectory output;
    Reconstruction reconstruction = SampleReconstruction();
    Landmark second = reconstruction.landmarks.front();
    second.id = LandmarkId(1);
    second.source_track_id = TrackId(5);
    second.position_world.x() = 0.5;
    second.observations[0].feature_id = FeatureId(2);
    second.observations[0].pixel.x() = 460.0;
    second.observations[1].feature_id = FeatureId(8);
    second.observations[1].pixel.x() = 340.0;
    reconstruction.landmarks.push_back(second);

    ASSERT_TRUE(
        ReconstructionExporter::WriteColmapTextReconstruction(output.path(), reconstruction).ok());
    EXPECT_NE(output.Read("points3D.txt").find("1 0 0 5 12 34 56 0.25 1 1 2 0"), std::string::npos);
    EXPECT_NE(output.Read("points3D.txt").find("2 0.5 0 5 12 34 56 0.25 1 0 2 1"),
              std::string::npos);
}

TEST(ColmapTextWriterTest, UsesOptimizedPoseConvention)
{
    ColmapDirectory output;
    Reconstruction reconstruction = SampleReconstruction();
    reconstruction.cameras[1].optimized_world_from_camera =
        reconstruction.cameras[1].initial_world_from_camera;
    reconstruction.cameras[1].optimized_world_from_camera.translation_world_from_local.x() = 2.0;
    reconstruction.cameras[1].has_optimized_pose = true;

    ASSERT_TRUE(
        ReconstructionExporter::WriteColmapTextReconstruction(output.path(), reconstruction).ok());
    EXPECT_NE(output.Read("images.txt").find("2 1 -0 -0 -0 -2 -0 -0 1 second.png"),
              std::string::npos);
}

TEST(ColmapTextWriterTest, CanExcludeAndCompactUnsupportedCameras)
{
    ColmapDirectory output;
    Reconstruction reconstruction = SampleReconstruction();
    CameraFrame unsupported;
    unsupported.id = CameraId(2);
    unsupported.image_name = "unsupported.png";
    reconstruction.cameras.push_back(unsupported);
    ColmapTextOptions options;
    options.minimum_observations_per_camera = 1;

    ASSERT_TRUE(ReconstructionExporter::WriteColmapTextReconstruction(output.path(), reconstruction,
                                                                      options)
                    .ok());

    const std::string images = output.Read("images.txt");
    EXPECT_NE(images.find("# Number of images: 2"), std::string::npos);
    EXPECT_EQ(images.find("unsupported.png"), std::string::npos);
    EXPECT_NE(output.Read("points3D.txt").find("1 0 2 0"), std::string::npos);
}

TEST(ColmapTextWriterTest, RejectsBrokenCrossReferencesBeforeWriting)
{
    ColmapDirectory output;
    Reconstruction reconstruction = SampleReconstruction();
    reconstruction.landmarks[0].observations[1].camera_id = CameraId(99);

    const Status status =
        ReconstructionExporter::WriteColmapTextReconstruction(output.path(), reconstruction);

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), ErrorCode::kInvalidArgument);
    EXPECT_FALSE(std::filesystem::exists(output.path()));
}

} // namespace
} // namespace lio_visual_ba
