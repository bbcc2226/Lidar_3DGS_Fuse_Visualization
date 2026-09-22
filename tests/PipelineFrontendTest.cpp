#include "Pipeline.hpp"

#include <filesystem>
#include <fstream>

#include <gtest/gtest.h>
#include <opencv2/imgcodecs.hpp>

namespace lio_visual_ba
{
namespace
{

class PipelineFrontendTest : public ::testing::Test
{
  protected:
    void SetUp() override
    {
        directory_ = std::filesystem::temp_directory_path() / "lio_visual_ba_frontend_test";
        std::filesystem::remove_all(directory_);
        std::filesystem::create_directories(directory_ / "images");
        Write("intrinsics.txt", "500 0 160\n0 500 120\n0 0 1\n");
        Write("extrinsic.json", R"({"T_camera_lidar":[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]})");
        Write("trajectory.jsonl",
              R"({"timestamp":0,"lio_pose":{"translation":[0,0,0],"quaternion_xyzw":[0,0,0,1]}}
{"timestamp":1,"lio_pose":{"translation":[0,0,0],"quaternion_xyzw":[0,0,0,1]}})");
    }

    void TearDown() override { std::filesystem::remove_all(directory_); }

    void Write(const std::string& name, const std::string& contents)
    {
        std::ofstream(directory_ / name) << contents;
    }

    void WriteImage(const std::string& name)
    {
        cv::Mat image(240, 320, CV_8UC1);
        cv::randu(image, 0, 255);
        ASSERT_TRUE(cv::imwrite((directory_ / "images" / name).string(), image));
    }

    PipelineConfig Config() const
    {
        PipelineConfig config;
        config.inputs.timestamp_file = directory_ / "timestamps.txt";
        config.inputs.image_directory = directory_ / "images";
        config.inputs.intrinsics_file = directory_ / "intrinsics.txt";
        config.inputs.extrinsic_file = directory_ / "extrinsic.json";
        config.inputs.trajectory_file = directory_ / "trajectory.jsonl";
        config.image_list.image_directory = config.inputs.image_directory;
        config.image_list.skip_missing_images = false;
        config.image_width = 320;
        config.image_height = 240;
        config.feature_extraction.maximum_sift_features = 200;
        return config;
    }

    std::filesystem::path directory_;
};

TEST_F(PipelineFrontendTest, LoadsPosesAndExtractsFeatures)
{
    WriteImage("frame.png");
    Write("timestamps.txt", "frame.png 0.5\n");

    auto result = Pipeline::RunPipelineFrontend(Config());

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    ASSERT_EQ(result.value().reconstruction.cameras.size(), 1U);
    ASSERT_EQ(result.value().feature_sets.size(), 1U);
    EXPECT_EQ(result.value().feature_sets[0].camera_id, CameraId(0));
    EXPECT_EQ(result.value().candidate_pairs.all_pairs.size(), 0U);
}

TEST_F(PipelineFrontendTest, RecordsGeometricallyRejectedPair)
{
    WriteImage("first.png");
    std::filesystem::copy_file(directory_ / "images/first.png", directory_ / "images/second.png");
    Write("timestamps.txt", "first.png 0\nsecond.png 1\n");

    auto result = Pipeline::RunPipelineFrontend(Config());

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    EXPECT_EQ(result.value().candidate_pairs.all_pairs.size(), 1U);
    EXPECT_EQ(result.value().accepted_candidate_pairs, 0U);
    EXPECT_EQ(result.value().rejected_candidate_pairs, 1U);
    EXPECT_TRUE(result.value().verified_matches.empty());
}

TEST_F(PipelineFrontendTest, PropagatesMissingImageFailureWithContext)
{
    Write("timestamps.txt", "missing.png 0.5\n");

    auto result = Pipeline::RunPipelineFrontend(Config());

    EXPECT_FALSE(result.ok());
    EXPECT_EQ(result.status().code(), ErrorCode::kNotFound);
    EXPECT_NE(result.status().message().find("dataset camera loading"), std::string::npos);
    EXPECT_NE(result.status().message().find("pipeline dataset loading"), std::string::npos);
}

} // namespace
} // namespace lio_visual_ba
