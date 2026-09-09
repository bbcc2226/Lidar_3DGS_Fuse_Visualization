#include "reconstruction_exporter.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

namespace lio_visual_ba
{
namespace
{

class OutputDirectory
{
  public:
    OutputDirectory()
    {
        path_ = std::filesystem::temp_directory_path() /
                ("reconstruction_io_" +
                 std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        std::filesystem::create_directories(path_);
    }
    ~OutputDirectory()
    {
        std::error_code ignored;
        std::filesystem::remove_all(path_, ignored);
    }
    std::filesystem::path File(const std::string& name) const { return path_ / name; }
    std::string Read(const std::string& name) const
    {
        std::ifstream input(File(name));
        return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
    }

  private:
    std::filesystem::path path_;
};

TEST(ReconstructionIoTest, WritesTumPosesWithOptimizedFallback)
{
    OutputDirectory output;
    CameraFrame first;
    first.id = CameraId(0);
    first.timestamp_seconds = 1.25;
    first.initial_world_from_camera.translation_world_from_local.x() = 2.0;
    CameraFrame second;
    second.id = CameraId(1);
    second.timestamp_seconds = 2.5;
    second.initial_world_from_camera.translation_world_from_local.x() = 3.0;
    second.optimized_world_from_camera.translation_world_from_local.x() = 4.0;
    second.has_optimized_pose = true;

    const Status status = ReconstructionExporter::WriteTumCameraPoses(output.File("poses.txt"),
                                                                      {first, second}, true);

    ASSERT_TRUE(status.ok()) << status.ToString();
    std::istringstream lines(output.Read("poses.txt"));
    double timestamp = 0.0;
    double x = 0.0;
    double ignored = 0.0;
    lines >> timestamp >> x;
    EXPECT_DOUBLE_EQ(timestamp, 1.25);
    EXPECT_DOUBLE_EQ(x, 2.0);
    for (int index = 0; index < 6; ++index)
    {
        lines >> ignored;
    }
    lines >> timestamp >> x;
    EXPECT_DOUBLE_EQ(timestamp, 2.5);
    EXPECT_DOUBLE_EQ(x, 4.0);
}

TEST(ReconstructionIoTest, WritesPlyTracksAndObservations)
{
    OutputDirectory output;
    FeatureTrack track;
    track.id = TrackId(7);
    track.is_supplemental = true;
    track.observations = {{CameraId(0), FeatureId(2), {10.5, 20.5}},
                          {CameraId(1), FeatureId(3), {11.5, 21.5}}};
    Landmark landmark;
    landmark.id = LandmarkId(0);
    landmark.source_track_id = track.id;
    landmark.position_world = {1.0, 2.0, 3.0};
    landmark.color_rgb = {10, 20, 30};
    landmark.observations = track.observations;

    ASSERT_TRUE(
        ReconstructionExporter::WriteLandmarksPly(output.File("points.ply"), {landmark}).ok());
    ASSERT_TRUE(ReconstructionExporter::WriteTracksCsv(output.File("tracks.csv"), {track}).ok());
    ASSERT_TRUE(
        ReconstructionExporter::WriteObservationsCsv(output.File("observations.csv"), {landmark})
            .ok());

    EXPECT_NE(output.Read("points.ply").find("element vertex 1"), std::string::npos);
    EXPECT_NE(output.Read("points.ply").find("1 2 3 10 20 30"), std::string::npos);
    EXPECT_NE(output.Read("tracks.csv").find("7,2,1"), std::string::npos);
    EXPECT_NE(output.Read("observations.csv").find("0,7,1,3,11.5,21.5"), std::string::npos);
}

TEST(ReconstructionIoTest, WritesReadableMetricsJson)
{
    OutputDirectory output;
    nlohmann::json metrics = {{"stage", "triangulation"}, {"landmarks", 42}};

    ASSERT_TRUE(
        ReconstructionExporter::WriteMetricsJson(output.File("metrics.json"), metrics).ok());
    std::ifstream input(output.File("metrics.json"));
    nlohmann::json restored;
    input >> restored;
    EXPECT_EQ(restored, metrics);
    EXPECT_NE(output.Read("metrics.json").find("\n  \"landmarks\""), std::string::npos);
}

TEST(ReconstructionIoTest, RejectsInvalidDataBeforeCreatingOutput)
{
    OutputDirectory output;
    Landmark invalid;
    invalid.id = LandmarkId(0);
    invalid.position_world.x() = std::numeric_limits<double>::quiet_NaN();

    const auto path = output.File("invalid.ply");
    const Status status = ReconstructionExporter::WriteLandmarksPly(path, {invalid});

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), ErrorCode::kInvalidArgument);
    EXPECT_FALSE(std::filesystem::exists(path));
    EXPECT_FALSE(
        ReconstructionExporter::WriteMetricsJson(output.File("array.json"), nlohmann::json::array())
            .ok());
}

} // namespace
} // namespace lio_visual_ba
