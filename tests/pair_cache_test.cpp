#include "feature_processor.hpp"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
namespace lio_visual_ba
{
namespace
{
std::filesystem::path Path()
{
    return std::filesystem::temp_directory_path() /
           ("pairs_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) +
            ".bin");
}
PairCacheKey Key()
{
    PairCacheKey k;
    k.first_camera_id = CameraId(2);
    k.second_camera_id = CameraId(4);
    k.first_feature_fingerprint = 11;
    k.second_feature_fingerprint = 22;
    k.matching_configuration_fingerprint = 33;
    k.first_feature_count = 10;
    k.second_feature_count = 12;
    return k;
}
std::vector<FeatureMatch> Matches()
{
    std::vector<FeatureMatch> out;
    for (int i = 0; i < 3; ++i)
    {
        FeatureMatch m;
        m.first_camera_id = CameraId(2);
        m.second_camera_id = CameraId(4);
        m.first_feature_id = FeatureId(i);
        m.second_feature_id = FeatureId(i + 2);
        m.descriptor_distance = static_cast<float>(i) + 0.5F;
        m.geometric_error_pixels = static_cast<double>(i) * 0.25;
        out.push_back(m);
    }
    return out;
}
TEST(PairCacheTest, RoundTripsMatches)
{
    auto path = Path();
    ASSERT_TRUE(FeatureProcessor::SavePairCache(path, Key(), Matches()).ok());
    auto loaded = FeatureProcessor::LoadPairCache(path, Key());
    std::filesystem::remove(path);
    ASSERT_TRUE(loaded.ok()) << loaded.status().ToString();
    ASSERT_EQ(loaded.value().size(), 3U);
    EXPECT_EQ(loaded.value()[2].second_feature_id, FeatureId(4));
    EXPECT_FLOAT_EQ(loaded.value()[1].descriptor_distance, 1.5F);
    EXPECT_DOUBLE_EQ(loaded.value()[2].geometric_error_pixels, 0.5);
}
TEST(PairCacheTest, RejectsStaleFeatureOrConfigurationFingerprint)
{
    auto path = Path();
    ASSERT_TRUE(FeatureProcessor::SavePairCache(path, Key(), Matches()).ok());
    auto changed = Key();
    changed.second_feature_fingerprint++;
    EXPECT_EQ(FeatureProcessor::LoadPairCache(path, changed).status().code(),
              ErrorCode::kFailedPrecondition);
    changed = Key();
    changed.matching_configuration_fingerprint++;
    EXPECT_FALSE(FeatureProcessor::LoadPairCache(path, changed).ok());
    std::filesystem::remove(path);
}
TEST(PairCacheTest, DetectsCorruption)
{
    auto path = Path();
    ASSERT_TRUE(FeatureProcessor::SavePairCache(path, Key(), Matches()).ok());
    std::fstream file(path, std::ios::binary | std::ios::in | std::ios::out);
    file.seekg(-1, std::ios::end);
    char byte = 0;
    file.read(&byte, 1);
    file.seekp(-1, std::ios::end);
    byte ^= 0x7f;
    file.write(&byte, 1);
    file.close();
    auto loaded = FeatureProcessor::LoadPairCache(path, Key());
    std::filesystem::remove(path);
    EXPECT_FALSE(loaded.ok());
    EXPECT_EQ(loaded.status().code(), ErrorCode::kDataLoss);
}
TEST(PairCacheTest, RejectsInvalidOrUnorderedMatches)
{
    auto matches = Matches();
    std::swap(matches[0], matches[2]);
    auto path = Path();
    EXPECT_FALSE(FeatureProcessor::SavePairCache(path, Key(), matches).ok());
    EXPECT_FALSE(std::filesystem::exists(path));
    matches = Matches();
    matches[0].first_feature_id = FeatureId(99);
    EXPECT_FALSE(FeatureProcessor::SavePairCache(path, Key(), matches).ok());
}
} // namespace
} // namespace lio_visual_ba
