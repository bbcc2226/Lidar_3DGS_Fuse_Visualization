#include "FeatureProcessor.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>

#include <gtest/gtest.h>

namespace lio_visual_ba
{
namespace
{

std::filesystem::path CachePath()
{
    return std::filesystem::temp_directory_path() /
           ("features_" +
            std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".bin");
}

FeatureSet SampleFeatures()
{
    FeatureSet set;
    set.camera_id = CameraId(3);
    set.core_feature_count = 1;
    for (int index = 0; index < 2; ++index)
    {
        Feature feature;
        feature.id = FeatureId(index);
        feature.pixel = Eigen::Vector2d(10.0 + index, 20.0 - index);
        feature.scale = 3.0F;
        feature.angle_degrees = 45.0F;
        feature.response = 0.5F;
        feature.octave = index;
        set.features.push_back(feature);
    }
    set.descriptors = cv::Mat(2, 128, CV_32F);
    for (int row = 0; row < 2; ++row)
        for (int column = 0; column < 128; ++column)
            set.descriptors.at<float>(row, column) = static_cast<float>(row * 128 + column);
    return set;
}

TEST(FeatureCacheTest, RoundTripsFeaturesAndDescriptors)
{
    const auto path = CachePath();
    const FeatureCacheFingerprint fingerprint{11, 22};
    ASSERT_TRUE(FeatureProcessor::SaveFeatureCache(path, SampleFeatures(), fingerprint).ok());
    auto loaded = FeatureProcessor::LoadFeatureCache(path, fingerprint);
    std::filesystem::remove(path);
    ASSERT_TRUE(loaded.ok()) << loaded.status().ToString();
    EXPECT_EQ(loaded.value().camera_id, CameraId(3));
    EXPECT_EQ(loaded.value().core_feature_count, 1U);
    ASSERT_EQ(loaded.value().features.size(), 2U);
    EXPECT_TRUE(loaded.value().features[1].pixel.isApprox(Eigen::Vector2d(11.0, 19.0)));
    EXPECT_EQ(cv::norm(loaded.value().descriptors, SampleFeatures().descriptors), 0.0);
}

TEST(FeatureCacheTest, RejectsFingerprintMismatch)
{
    const auto path = CachePath();
    ASSERT_TRUE(FeatureProcessor::SaveFeatureCache(path, SampleFeatures(), {1, 2}).ok());
    auto loaded = FeatureProcessor::LoadFeatureCache(path, {1, 3});
    std::filesystem::remove(path);
    ASSERT_FALSE(loaded.ok());
    EXPECT_EQ(loaded.status().code(), ErrorCode::kFailedPrecondition);
}

TEST(FeatureCacheTest, DetectsTruncationAndCorruption)
{
    const auto path = CachePath();
    ASSERT_TRUE(FeatureProcessor::SaveFeatureCache(path, SampleFeatures(), {1, 2}).ok());
    std::fstream file(path, std::ios::binary | std::ios::in | std::ios::out);
    file.seekp(-1, std::ios::end);
    char byte = 0;
    file.read(&byte, 1);
    file.clear();
    file.seekp(-1, std::ios::end);
    byte ^= 0x55;
    file.write(&byte, 1);
    file.close();
    auto corrupted = FeatureProcessor::LoadFeatureCache(path, {1, 2});
    ASSERT_FALSE(corrupted.ok());
    EXPECT_EQ(corrupted.status().code(), ErrorCode::kDataLoss);
    std::filesystem::resize_file(path, 12);
    auto truncated = FeatureProcessor::LoadFeatureCache(path, {1, 2});
    std::filesystem::remove(path);
    EXPECT_FALSE(truncated.ok());
}

TEST(FeatureCacheTest, RejectsInconsistentFeatureSetBeforeWriting)
{
    FeatureSet invalid = SampleFeatures();
    invalid.core_feature_count = 3;
    const auto path = CachePath();
    auto status = FeatureProcessor::SaveFeatureCache(path, invalid, {1, 2});
    EXPECT_FALSE(status.ok());
    EXPECT_FALSE(std::filesystem::exists(path));
}

} // namespace
} // namespace lio_visual_ba
