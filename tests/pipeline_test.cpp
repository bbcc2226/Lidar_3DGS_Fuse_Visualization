#include "pipeline.hpp"

#include <filesystem>
#include <fstream>

#include <gtest/gtest.h>

#include "reconstruction_exporter.hpp"

namespace lio_visual_ba
{
namespace
{

TEST(PipelineTest, PublishesAndValidatesSparseContract)
{
    const std::filesystem::path output =
        std::filesystem::temp_directory_path() / "lio_visual_ba_pipeline_publish_test";
    std::filesystem::remove_all(output);
    PipelineRunResult result;
    result.backend.reconstruction.intrinsics = {500.0, 500.0, 320.0, 240.0, 640, 480};
    for (int index = 0; index < 2; ++index)
    {
        CameraFrame camera;
        camera.id = CameraId(index);
        camera.image_name = std::to_string(index) + ".png";
        camera.timestamp_seconds = index;
        camera.initial_world_from_camera.translation_world_from_local.x() = index;
        result.backend.reconstruction.cameras.push_back(camera);
    }
    FeatureTrack track;
    track.id = TrackId(0);
    track.observations = {{CameraId(0), FeatureId(0), {320.0, 240.0}},
                          {CameraId(1), FeatureId(0), {195.0, 240.0}}};
    result.backend.reconstruction.tracks.push_back(track);
    Landmark landmark;
    landmark.id = LandmarkId(0);
    landmark.source_track_id = TrackId(0);
    landmark.position_world = {0.0, 0.0, 4.0};
    landmark.observations = track.observations;
    result.backend.reconstruction.landmarks.push_back(landmark);

    const Status status = Pipeline::PublishPipelineOutputs(output, result);

    ASSERT_TRUE(status.ok()) << status.ToString();
    EXPECT_TRUE(std::filesystem::exists(output / "poses_optimized_tum.txt"));
    EXPECT_TRUE(std::filesystem::exists(output / "landmarks_optimized.ply"));
    EXPECT_TRUE(std::filesystem::exists(output / "evaluation.json"));
    EXPECT_TRUE(std::filesystem::exists(output / "heldout.csv"));
    EXPECT_TRUE(std::filesystem::exists(output / "track_summary.csv"));
    EXPECT_TRUE(ReconstructionExporter::ValidateColmapTextReconstruction(output / "sparse/0").ok());
    std::filesystem::remove_all(output);
}

TEST(PipelineTest, RefusesToOverwriteNonemptyOutput)
{
    const std::filesystem::path output =
        std::filesystem::temp_directory_path() / "lio_visual_ba_pipeline_nonempty_test";
    std::filesystem::create_directories(output);
    std::ofstream(output / "keep.txt") << "user data";

    const Status status = Pipeline::PublishPipelineOutputs(output, PipelineRunResult{});

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), ErrorCode::kFailedPrecondition);
    EXPECT_TRUE(std::filesystem::exists(output / "keep.txt"));
    std::filesystem::remove_all(output);
}

} // namespace
} // namespace lio_visual_ba
