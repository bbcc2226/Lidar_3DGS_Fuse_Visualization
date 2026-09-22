#include "feature_processor.hpp"

#include <algorithm>
#include <set>

#include <gtest/gtest.h>

namespace lio_visual_ba
{
namespace
{

FeatureSet Features(int camera, int count, std::size_t core)
{
    FeatureSet set;
    set.camera_id = CameraId(camera);
    set.core_feature_count = core;
    for (int i = 0; i < count; ++i)
    {
        Feature f;
        f.id = FeatureId(i);
        f.pixel = Eigen::Vector2d(camera * 10 + i, i);
        set.features.push_back(f);
    }
    return set;
}
FeatureMatch Match(int a, int fa, int b, int fb, float distance = 1)
{
    FeatureMatch m;
    m.first_camera_id = CameraId(a);
    m.first_feature_id = FeatureId(fa);
    m.second_camera_id = CameraId(b);
    m.second_feature_id = FeatureId(fb);
    m.descriptor_distance = distance;
    return m;
}
TEST(TrackBuilderTest, BuildsMultiViewTrackWithStableObservations)
{
    std::vector<FeatureSet> sets{Features(0, 2, 2), Features(1, 2, 2), Features(2, 2, 2)};
    std::vector<FeatureMatch> matches{Match(1, 0, 2, 0), Match(0, 0, 1, 0)};
    auto result = FeatureProcessor::BuildFeatureTracks(sets, matches);
    ASSERT_TRUE(result.ok()) << result.status().ToString();
    ASSERT_EQ(result.value().tracks.size(), 1U);
    const auto& track = result.value().tracks[0];
    EXPECT_EQ(track.id, TrackId(0));
    ASSERT_EQ(track.observations.size(), 3U);
    EXPECT_EQ(track.observations[0].camera_id, CameraId(0));
    EXPECT_EQ(track.observations[2].camera_id, CameraId(2));
    EXPECT_EQ(result.value().accepted_matches, 2U);
}
TEST(TrackBuilderTest, RejectsMergeThatDuplicatesCameraObservation)
{
    std::vector<FeatureSet> sets{Features(0, 2, 2), Features(1, 1, 1), Features(2, 1, 1)};
    std::vector<FeatureMatch> matches{Match(0, 0, 1, 0), Match(1, 0, 2, 0), Match(0, 1, 2, 0)};
    auto result = FeatureProcessor::BuildFeatureTracks(sets, matches);
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result.value().conflict_rejected_matches, 1U);
    // Earlier temporal links win, matching the Python DSU policy. The
    // conflicting camera-0 feature remains a singleton and is discarded.
    ASSERT_EQ(result.value().tracks.size(), 1U);
    for (const auto& track : result.value().tracks)
    {
        std::set<int> cameras;
        for (const auto& observation : track.observations)
            cameras.insert(observation.camera_id.value());
        EXPECT_EQ(cameras.size(), track.observations.size());
    }
}
TEST(TrackBuilderTest, IsDeterministicAcrossMatchOrdering)
{
    std::vector<FeatureSet> sets{Features(0, 2, 2), Features(1, 2, 2), Features(2, 2, 2)};
    std::vector<FeatureMatch> forward{Match(0, 1, 1, 1), Match(1, 0, 2, 0), Match(0, 0, 1, 0)};
    auto reverse = forward;
    std::reverse(reverse.begin(), reverse.end());
    auto a = FeatureProcessor::BuildFeatureTracks(sets, forward),
         b = FeatureProcessor::BuildFeatureTracks(sets, reverse);
    ASSERT_TRUE(a.ok() && b.ok());
    ASSERT_EQ(a.value().tracks.size(), b.value().tracks.size());
    for (std::size_t i = 0; i < a.value().tracks.size(); ++i)
    {
        ASSERT_EQ(a.value().tracks[i].observations.size(), b.value().tracks[i].observations.size());
        for (std::size_t j = 0; j < a.value().tracks[i].observations.size(); ++j)
        {
            EXPECT_EQ(a.value().tracks[i].observations[j].camera_id,
                      b.value().tracks[i].observations[j].camera_id);
            EXPECT_EQ(a.value().tracks[i].observations[j].feature_id,
                      b.value().tracks[i].observations[j].feature_id);
        }
    }
}
TEST(TrackBuilderTest, MarksSupplementalTracksAndAppliesMinimumLength)
{
    std::vector<FeatureSet> sets{Features(0, 2, 1), Features(1, 2, 2), Features(2, 1, 1)};
    std::vector<FeatureMatch> matches{Match(0, 1, 1, 1), Match(0, 0, 1, 0), Match(1, 0, 2, 0)};
    TrackBuilderOptions options;
    options.minimum_track_length = 3;
    auto result = FeatureProcessor::BuildFeatureTracks(sets, matches, options);
    ASSERT_TRUE(result.ok());
    ASSERT_EQ(result.value().tracks.size(), 1U);
    EXPECT_FALSE(result.value().tracks[0].is_supplemental);
    options.minimum_track_length = 2;
    result = FeatureProcessor::BuildFeatureTracks(sets, matches, options);
    ASSERT_EQ(result.value().tracks.size(), 2U);
    EXPECT_TRUE(result.value().tracks[1].is_supplemental);
}
TEST(TrackBuilderTest, RejectsUnknownFeatureReference)
{
    std::vector<FeatureSet> sets{Features(0, 1, 1), Features(1, 1, 1)};
    auto result = FeatureProcessor::BuildFeatureTracks(sets, {Match(0, 9, 1, 0)});
    EXPECT_FALSE(result.ok());
    EXPECT_EQ(result.status().code(), ErrorCode::kInvalidArgument);
}
} // namespace
} // namespace lio_visual_ba
