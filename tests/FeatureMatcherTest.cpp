#include "FeatureProcessor.hpp"
#include "Geometry.hpp"
#include <gtest/gtest.h>
namespace lio_visual_ba
{
namespace
{
FeatureSet Set(CameraId id, const std::vector<float>& values,
               const std::vector<Eigen::Vector2d>& pixels)
{
    FeatureSet set;
    set.camera_id = id;
    set.core_feature_count = values.size();
    set.descriptors = cv::Mat(static_cast<int>(values.size()), 128, CV_32F);
    for (std::size_t i = 0; i < values.size(); ++i)
    {
        Feature f;
        f.id = FeatureId(static_cast<int>(i));
        f.pixel = pixels[i];
        set.features.push_back(f);
        set.descriptors.row(static_cast<int>(i)).setTo(values[i]);
    }
    return set;
}
CameraIntrinsics K()
{
    CameraIntrinsics k;
    k.fx = k.fy = 600;
    k.cx = 400;
    k.cy = 300;
    k.width = 800;
    k.height = 600;
    return k;
}
TEST(FeatureMatcherTest, RatioMatchingIsDeterministic)
{
    auto a = Set(CameraId(0), {0, 10}, {{1, 2}, {3, 4}}),
         b = Set(CameraId(1), {0.1F, 10.1F, 50}, {{1, 2}, {3, 4}, {5, 6}});
    auto result = FeatureProcessor::MatchFeatureDescriptors(a, b);
    ASSERT_TRUE(result.ok()) << result.status().ToString();
    ASSERT_EQ(result.value().size(), 2U);
    EXPECT_EQ(result.value()[0].first_feature_id, FeatureId(0));
    EXPECT_EQ(result.value()[0].second_feature_id, FeatureId(0));
    EXPECT_EQ(result.value()[1].first_feature_id, FeatureId(1));
    EXPECT_EQ(result.value()[1].second_feature_id, FeatureId(1));
}
TEST(FeatureMatcherTest, MutualMatchingRejectsManyToOneAmbiguity)
{
    auto a = Set(CameraId(0), {0, 0.2F}, {{1, 2}, {3, 4}}),
         b = Set(CameraId(1), {0.05F, 50}, {{1, 2}, {3, 4}});
    FeatureMatchingOptions options;
    options.require_mutual_match = true;
    auto result = FeatureProcessor::MatchFeatureDescriptors(a, b, options);
    ASSERT_TRUE(result.ok());
    ASSERT_EQ(result.value().size(), 1U);
    EXPECT_EQ(result.value()[0].first_feature_id, FeatureId(0));
}
TEST(FeatureMatcherTest, PosePriorRejectsEpipolarOutlier)
{
    Pose3d first, second;
    second.translation_world_from_local.x() = 1;
    auto p1 = geometry::ProjectWorldPoint(K(), first, Eigen::Vector3d(0, 0, 5));
    auto p2 = geometry::ProjectWorldPoint(K(), second, Eigen::Vector3d(0, 0, 5));
    ASSERT_TRUE(p1.ok() && p2.ok());
    auto a =
        Set(CameraId(0), {0, 10}, {p1.value().pixel, p1.value().pixel + Eigen::Vector2d(20, 0)});
    auto b = Set(CameraId(1), {0.1F, 10.1F, 50},
                 {p2.value().pixel, p2.value().pixel + Eigen::Vector2d(20, 30), {0, 0}});
    FeatureMatchingOptions options;
    options.maximum_sampson_error_pixels = 1;
    auto result = FeatureProcessor::MatchFeaturesWithPosePrior(a, first, b, second, K(), options);
    ASSERT_TRUE(result.ok()) << result.status().ToString();
    ASSERT_EQ(result.value().size(), 1U);
    EXPECT_NEAR(result.value()[0].geometric_error_pixels, 0, 1e-12);
}
TEST(FeatureMatcherTest, RejectsSelfMatchAndInvalidRatio)
{
    auto a = Set(CameraId(0), {0, 1}, {{0, 0}, {1, 1}});
    EXPECT_FALSE(FeatureProcessor::MatchFeatureDescriptors(a, a).ok());
    FeatureMatchingOptions options;
    options.ratio_threshold = 1.0;
    auto b = Set(CameraId(1), {0, 1}, {{0, 0}, {1, 1}});
    EXPECT_FALSE(FeatureProcessor::MatchFeatureDescriptors(a, b, options).ok());
}
TEST(FeatureMatcherTest, VisualGeometryRecoversDistributedInliers)
{
    Pose3d first, second;
    second.translation_world_from_local.x() = 1;
    std::vector<Eigen::Vector2d> pixels1, pixels2;
    std::vector<float> descriptors;
    for (int y = 0; y < 6; ++y)
        for (int x = 0; x < 6; ++x)
        {
            const Eigen::Vector3d point((x - 2.5) * 0.35, (y - 2.5) * 0.25, 4.0 + 0.08 * x);
            auto a = geometry::ProjectWorldPoint(K(), first, point),
                 b = geometry::ProjectWorldPoint(K(), second, point);
            ASSERT_TRUE(a.ok() && b.ok());
            pixels1.push_back(a.value().pixel);
            pixels2.push_back(b.value().pixel);
            descriptors.push_back(static_cast<float>(x + 6 * y));
        }
    for (std::size_t i = 28; i < pixels2.size(); ++i)
        pixels2[i] = Eigen::Vector2d(40.0 + 73.0 * static_cast<double>(i - 28),
                                     35.0 + 47.0 * static_cast<double>((i - 28) % 5));
    auto a = Set(CameraId(0), descriptors, pixels1), b = Set(CameraId(1), descriptors, pixels2);
    std::vector<FeatureMatch> candidates;
    for (int i = 0; i < 36; ++i)
    {
        FeatureMatch m;
        m.first_camera_id = CameraId(0);
        m.second_camera_id = CameraId(1);
        m.first_feature_id = FeatureId(i);
        m.second_feature_id = FeatureId(i);
        candidates.push_back(m);
    }
    VisualGeometryOptions options;
    options.ransac_threshold_pixels = 1.0;
    options.minimum_inliers = 20;
    options.random_seed = 42;
    auto result = FeatureProcessor::VerifyMatchesWithVisualGeometry(a, b, candidates, options);
    ASSERT_TRUE(result.ok()) << result.status().ToString();
    EXPECT_GE(result.value().inlier_matches.size(), 28U);
    EXPECT_LT(result.value().inlier_matches.size(), 36U);
    EXPECT_GE(result.value().first_occupied_cells, 6);
    EXPECT_GE(result.value().second_occupied_cells, 6);
}
TEST(FeatureMatcherTest, VisualGeometryRejectsInvalidReferences)
{
    auto a = Set(CameraId(0), {0, 1}, {{0, 0}, {1, 1}}),
         b = Set(CameraId(1), {0, 1}, {{0, 0}, {1, 1}});
    std::vector<FeatureMatch> candidates(20);
    for (auto& m : candidates)
    {
        m.first_camera_id = CameraId(0);
        m.second_camera_id = CameraId(1);
        m.first_feature_id = FeatureId(99);
        m.second_feature_id = FeatureId(0);
    }
    auto result = FeatureProcessor::VerifyMatchesWithVisualGeometry(a, b, candidates);
    EXPECT_FALSE(result.ok());
    EXPECT_EQ(result.status().code(), ErrorCode::kInvalidArgument);
}
} // namespace
} // namespace lio_visual_ba
