#include "feature_processor.hpp"

#include <gtest/gtest.h>
#include <opencv2/imgproc.hpp>

namespace lio_visual_ba
{
namespace
{

cv::Mat TexturedImage()
{
    cv::Mat image(240, 320, CV_8UC1, cv::Scalar(20));
    for (int y = 20; y < image.rows; y += 30)
    {
        for (int x = 20; x < image.cols; x += 30)
        {
            const int shade = ((x + y) / 30) % 2 == 0 ? 240 : 80;
            cv::rectangle(image, cv::Rect(x, y, 14, 14), cv::Scalar(shade), cv::FILLED);
            cv::circle(image, cv::Point(x + 7, y + 7), 4, cv::Scalar(255 - shade), 1);
        }
    }
    return image;
}

TEST(FeatureExtractorTest, ProducesStableIdsAndSiftDescriptors)
{
    auto result = FeatureProcessor::ExtractFeatures(CameraId(7), TexturedImage());

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    ASSERT_FALSE(result.value().features.empty());
    EXPECT_EQ(result.value().camera_id, CameraId(7));
    EXPECT_EQ(result.value().descriptors.rows, static_cast<int>(result.value().features.size()));
    EXPECT_EQ(result.value().descriptors.cols, 128);
    EXPECT_EQ(result.value().descriptors.type(), CV_32F);
    for (std::size_t index = 0; index < result.value().features.size(); ++index)
    {
        EXPECT_EQ(result.value().features[index].id,
                  FeatureId(static_cast<FeatureId::ValueType>(index)));
        EXPECT_TRUE(result.value().features[index].pixel.array().isFinite().all());
    }
}

TEST(FeatureExtractorTest, GridBalancingAppendsSupplementalFeatures)
{
    FeatureExtractionOptions options;
    options.maximum_sift_features = 1;
    options.grid_columns = 4;
    options.grid_rows = 3;
    options.minimum_core_features_per_cell = 5;
    options.maximum_supplemental_features_per_cell = 4;

    auto result = FeatureProcessor::ExtractFeatures(CameraId(0), TexturedImage(), options);

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    EXPECT_GT(result.value().core_feature_count, 0U);
    EXPECT_GT(result.value().features.size(), result.value().core_feature_count);
}

TEST(FeatureExtractorTest, HandlesFeaturelessImage)
{
    FeatureExtractionOptions options;
    options.maximum_supplemental_features_per_cell = 0;
    auto result = FeatureProcessor::ExtractFeatures(
        CameraId(0), cv::Mat(100, 100, CV_8UC1, cv::Scalar(50)), options);

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    EXPECT_TRUE(result.value().features.empty());
    EXPECT_TRUE(result.value().descriptors.empty());
    EXPECT_EQ(result.value().core_feature_count, 0U);
}

TEST(FeatureExtractorTest, RejectsInvalidImageAndOptions)
{
    EXPECT_FALSE(FeatureProcessor::ExtractFeatures(CameraId(0), cv::Mat{}).ok());
    EXPECT_FALSE(FeatureProcessor::ExtractFeatures(CameraId::Invalid(), TexturedImage()).ok());
    FeatureExtractionOptions options;
    options.grid_columns = 0;
    EXPECT_FALSE(FeatureProcessor::ExtractFeatures(CameraId(0), TexturedImage(), options).ok());
}

} // namespace
} // namespace lio_visual_ba
