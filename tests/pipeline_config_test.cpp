#include "pipeline.hpp"

#include <filesystem>
#include <fstream>

#include <gtest/gtest.h>

namespace lio_visual_ba
{
namespace
{

class PipelineConfigTest : public ::testing::Test
{
  protected:
    void SetUp() override
    {
        directory_ = std::filesystem::temp_directory_path() / "lio_visual_ba_pipeline_config_test";
        std::filesystem::remove_all(directory_);
        std::filesystem::create_directories(directory_);
    }

    void TearDown() override { std::filesystem::remove_all(directory_); }

    std::filesystem::path Write(const std::string& contents)
    {
        const std::filesystem::path path = directory_ / "pipeline.yaml";
        std::ofstream stream(path);
        stream << contents;
        return path;
    }

    std::string RequiredConfiguration() const
    {
        return "inputs:\n"
               "  timestamp_file: data/timestamps.txt\n"
               "  image_directory: data/images\n"
               "  intrinsics_file: data/intrinsics.txt\n"
               "  extrinsic_file: data/extrinsic.json\n"
               "  trajectory_file: data/trajectory.jsonl\n"
               "output_directory: output\n"
               "image_width: 800\n"
               "image_height: 600\n";
    }

    std::filesystem::path directory_;
};

TEST_F(PipelineConfigTest, LoadsOptionsAndResolvesRelativePaths)
{
    auto config = Pipeline::LoadPipelineConfig(Write(RequiredConfiguration() +
                                                     "random_seed: 42\n"
                                                     "maximum_images: 12\n"
                                                     "features:\n"
                                                     "  maximum_sift_features: 2500\n"
                                                     "matching:\n"
                                                     "  ratio_threshold: 0.7\n"
                                                     "  require_mutual_match: true\n"
                                                     "mapping:\n"
                                                     "  minimum_track_length: 3\n"
                                                     "optimization:\n"
                                                     "  global_interval: 5\n"
                                                     "  local_window_cameras: 6\n"));

    ASSERT_TRUE(config.ok()) << config.status().ToString();
    EXPECT_EQ(config.value().inputs.timestamp_file, directory_ / "data/timestamps.txt");
    EXPECT_EQ(config.value().image_list.image_directory, directory_ / "data/images");
    EXPECT_EQ(config.value().feature_extraction.maximum_sift_features, 2500);
    EXPECT_DOUBLE_EQ(config.value().feature_matching.ratio_threshold, 0.7);
    EXPECT_TRUE(config.value().feature_matching.require_mutual_match);
    EXPECT_EQ(config.value().landmark_builder.minimum_track_length, 3U);
    EXPECT_EQ(config.value().incremental_optimization.global_bundle_adjustment_interval, 5U);
    EXPECT_EQ(config.value().visual_geometry.random_seed, 42U);
    EXPECT_EQ(config.value().image_list.maximum_images, 12);
}

// Exercise YAML -> DatasetIO -> parsed data, independently of feature extraction.
TEST_F(PipelineConfigTest, DatasetIoUsesConfiguredPathsDimensionsAndImageLimit)
{
    std::filesystem::create_directories(directory_ / "data/images");
    std::ofstream(directory_ / "data/images/a.png") << "image placeholder";
    std::ofstream(directory_ / "data/images/b.png") << "image placeholder";
    std::ofstream(directory_ / "data/timestamps.txt") << "a.png 1\nb.png 2\n";
    std::ofstream(directory_ / "data/intrinsics.txt") << "450 0 321\n0 460 239\n0 0 1\n";
    std::ofstream(directory_ / "data/extrinsic.json")
        << R"({"T_camera_lidar":[[1,0,0,2],[0,1,0,0],[0,0,1,0],[0,0,0,1]]})";
    std::ofstream(directory_ / "data/trajectory.jsonl")
        << R"({"timestamp":1,"lio_pose":{"translation":[3,0,0],"quaternion_xyzw":[0,0,0,1]}})"
        << R"({"timestamp":2,"lio_pose":{"translation":[4,0,0],"quaternion_xyzw":[0,0,0,1]}})";

    auto config = DatasetIO::LoadConfig(Write(RequiredConfiguration() + "maximum_images: 1\n"));
    ASSERT_TRUE(config.ok()) << config.status().ToString();
    auto dataset = DatasetIO::LoadDataset(config.value());
    ASSERT_TRUE(dataset.ok()) << dataset.status().ToString();
    ASSERT_EQ(dataset.value().cameras.size(), 1U);
    EXPECT_EQ(dataset.value().cameras[0].image_path, directory_ / "data/images/a.png");
    EXPECT_EQ(dataset.value().intrinsics.width, 800);
    EXPECT_EQ(dataset.value().intrinsics.height, 600);
    EXPECT_DOUBLE_EQ(dataset.value().intrinsics.fx, 450.0);
    EXPECT_DOUBLE_EQ(dataset.value().lidar_from_camera.translation_world_from_local.x(), -2.0);
    ASSERT_EQ(dataset.value().trajectory.size(), 2U);
    EXPECT_DOUBLE_EQ(
        dataset.value().trajectory[0].world_from_lidar.translation_world_from_local.x(), 3.0);

    // Changing YAML alone changes the accepted dataset.
    config = DatasetIO::LoadConfig(Write(RequiredConfiguration() + "maximum_images: 2\n"));
    ASSERT_TRUE(config.ok());
    dataset = DatasetIO::LoadDataset(config.value());
    ASSERT_TRUE(dataset.ok()) << dataset.status().ToString();
    EXPECT_EQ(dataset.value().cameras.size(), 2U);
}

TEST_F(PipelineConfigTest, DatasetIoHonorsYamlMissingImagePolicy)
{
    std::filesystem::create_directories(directory_ / "data/images");
    std::ofstream(directory_ / "data/images/a.png") << "image placeholder";
    std::ofstream(directory_ / "data/timestamps.txt") << "missing.png 1\na.png 2\n";
    auto config =
        DatasetIO::LoadConfig(Write(RequiredConfiguration() + "skip_missing_images: false\n"));
    ASSERT_TRUE(config.ok());
    auto frames = DatasetIO::LoadCameraFrames(config.value().inputs.timestamp_file,
                                              config.value().image_list);
    EXPECT_FALSE(frames.ok());
    config = DatasetIO::LoadConfig(Write(RequiredConfiguration() + "skip_missing_images: true\n"));
    ASSERT_TRUE(config.ok());
    frames = DatasetIO::LoadCameraFrames(config.value().inputs.timestamp_file,
                                         config.value().image_list);
    ASSERT_TRUE(frames.ok()) << frames.status().ToString();
    ASSERT_EQ(frames.value().size(), 1U);
    EXPECT_EQ(frames.value()[0].image_name, "a.png");
}

TEST_F(PipelineConfigTest, DatasetIoHonorsYamlTimestampPolicyWithoutPipelineSettings)
{
    std::filesystem::create_directories(directory_ / "data/images");
    std::ofstream(directory_ / "data/images/a.png") << "image placeholder";
    std::ofstream(directory_ / "data/images/b.png") << "image placeholder";
    std::ofstream(directory_ / "data/timestamps.txt") << "a.png 2\nb.png 1\n";
    std::string contents = RequiredConfiguration();
    contents.erase(contents.find("output_directory: output\n"),
                   std::string("output_directory: output\n").size());
    auto config =
        DatasetIO::LoadConfig(Write(contents + "require_strictly_increasing_timestamps: false\n"));
    ASSERT_TRUE(config.ok());
    auto frames = DatasetIO::LoadCameraFrames(config.value().inputs.timestamp_file,
                                              config.value().image_list);
    ASSERT_TRUE(frames.ok()) << frames.status().ToString();
    EXPECT_EQ(frames.value().size(), 2U);
    auto pipeline = Pipeline::LoadPipelineConfig(
        Write(RequiredConfiguration() +
              "require_strictly_increasing_timestamps: true\nskip_missing_images: false\n"));
    ASSERT_TRUE(pipeline.ok());
    EXPECT_FALSE(pipeline.value().image_list.skip_missing_images);
    frames = DatasetIO::LoadCameraFrames(pipeline.value().inputs.timestamp_file,
                                         pipeline.value().image_list);
    EXPECT_FALSE(frames.ok());
}

TEST_F(PipelineConfigTest, RejectsMissingRequiredPath)
{
    auto config = Pipeline::LoadPipelineConfig(Write("inputs: {}\noutput_directory: output\n"
                                                     "image_width: 800\nimage_height: 600\n"));

    EXPECT_FALSE(config.ok());
    EXPECT_EQ(config.status().code(), ErrorCode::kInvalidArgument);
}

TEST_F(PipelineConfigTest, RejectsInvalidDimensions)
{
    std::string contents = RequiredConfiguration();
    contents.replace(contents.find("image_width: 800"), 16, "image_width: 0");

    auto config = Pipeline::LoadPipelineConfig(Write(contents));

    EXPECT_FALSE(config.ok());
    EXPECT_EQ(config.status().code(), ErrorCode::kInvalidArgument);
}

TEST_F(PipelineConfigTest, ReportsMalformedYaml)
{
    auto config = Pipeline::LoadPipelineConfig(Write("inputs: [unterminated\n"));

    EXPECT_FALSE(config.ok());
    EXPECT_EQ(config.status().code(), ErrorCode::kParseError);
}

} // namespace
} // namespace lio_visual_ba
