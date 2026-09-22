#include "DatasetIo.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>

#include <gtest/gtest.h>

namespace lio_visual_ba
{
namespace
{

class TemporaryDirectory
{
  public:
    TemporaryDirectory()
    {
        const auto suffix = std::chrono::steady_clock::now().time_since_epoch().count();
        path_ = std::filesystem::temp_directory_path() /
                ("lio_visual_ba_image_list_test_" + std::to_string(suffix));
        std::filesystem::create_directories(path_ / "images");
    }

    ~TemporaryDirectory()
    {
        std::error_code ignored;
        std::filesystem::remove_all(path_, ignored);
    }

    const std::filesystem::path& path() const { return path_; }

    void Write(const std::filesystem::path& relative_path, const std::string& contents) const
    {
        std::ofstream(path_ / relative_path) << contents;
    }

  private:
    std::filesystem::path path_;
};

ImageListOptions OptionsFor(const TemporaryDirectory& directory)
{
    ImageListOptions options;
    options.image_directory = directory.path() / "images";
    return options;
}

TEST(ImageListIoTest, LoadsFramesAndAssignsCompactStableIds)
{
    TemporaryDirectory directory;
    directory.Write("images/a.png", "a");
    directory.Write("images/b.png", "b");
    directory.Write("timestamps.txt", "# image timestamps\n"
                                      "a.png 10.25\n"
                                      "\n"
                                      "b.png 11.5 # accepted comment\n");

    auto result =
        DatasetIO::LoadCameraFrames(directory.path() / "timestamps.txt", OptionsFor(directory));

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    ASSERT_EQ(result.value().size(), 2U);
    EXPECT_EQ(result.value()[0].id, CameraId(0));
    EXPECT_EQ(result.value()[1].id, CameraId(1));
    EXPECT_EQ(result.value()[0].image_name, "a.png");
    EXPECT_DOUBLE_EQ(result.value()[1].timestamp_seconds, 11.5);
}

TEST(ImageListIoTest, SkipsMissingImagesAndKeepsIdsCompact)
{
    TemporaryDirectory directory;
    directory.Write("images/b.png", "b");
    directory.Write("timestamps.txt", "missing.png 1.0\n"
                                      "b.png 2.0\n");

    auto result =
        DatasetIO::LoadCameraFrames(directory.path() / "timestamps.txt", OptionsFor(directory));

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    ASSERT_EQ(result.value().size(), 1U);
    EXPECT_EQ(result.value()[0].id, CameraId(0));
    EXPECT_EQ(result.value()[0].image_name, "b.png");
}

TEST(ImageListIoTest, CanFailWhenAnImageIsMissing)
{
    TemporaryDirectory directory;
    directory.Write("timestamps.txt", "missing.png 1.0\n");
    ImageListOptions options = OptionsFor(directory);
    options.skip_missing_images = false;

    auto result = DatasetIO::LoadCameraFrames(directory.path() / "timestamps.txt", options);

    ASSERT_FALSE(result.ok());
    EXPECT_EQ(result.status().code(), ErrorCode::kNotFound);
}

TEST(ImageListIoTest, RejectsDuplicateImageNames)
{
    TemporaryDirectory directory;
    directory.Write("images/a.png", "a");
    directory.Write("timestamps.txt", "a.png 1.0\na.png 2.0\n");

    auto result =
        DatasetIO::LoadCameraFrames(directory.path() / "timestamps.txt", OptionsFor(directory));

    ASSERT_FALSE(result.ok());
    EXPECT_EQ(result.status().code(), ErrorCode::kDataLoss);
}

TEST(ImageListIoTest, RejectsNonIncreasingTimestamps)
{
    TemporaryDirectory directory;
    directory.Write("images/a.png", "a");
    directory.Write("images/b.png", "b");
    directory.Write("timestamps.txt", "a.png 2.0\nb.png 1.0\n");

    auto result =
        DatasetIO::LoadCameraFrames(directory.path() / "timestamps.txt", OptionsFor(directory));

    ASSERT_FALSE(result.ok());
    EXPECT_EQ(result.status().code(), ErrorCode::kDataLoss);
}

TEST(ImageListIoTest, AppliesMaximumToAcceptedImages)
{
    TemporaryDirectory directory;
    directory.Write("images/a.png", "a");
    directory.Write("images/b.png", "b");
    directory.Write("timestamps.txt", "a.png 1.0\nb.png 2.0\n");
    ImageListOptions options = OptionsFor(directory);
    options.maximum_images = 1;

    auto result = DatasetIO::LoadCameraFrames(directory.path() / "timestamps.txt", options);

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    ASSERT_EQ(result.value().size(), 1U);
    EXPECT_EQ(result.value()[0].image_name, "a.png");
}

} // namespace
} // namespace lio_visual_ba
