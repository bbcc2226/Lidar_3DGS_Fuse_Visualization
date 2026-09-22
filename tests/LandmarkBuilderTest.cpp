#include "Geometry.hpp"
#include "Mapper.hpp"
#include <gtest/gtest.h>
namespace lio_visual_ba
{
namespace
{
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
CameraFrame Camera(int id, const Eigen::Vector3d& center)
{
    CameraFrame camera;
    camera.id = CameraId(id);
    camera.initial_world_from_camera.translation_world_from_local = center;
    return camera;
}
FeatureTrack Track(int id, const std::vector<CameraFrame>& cameras, const Eigen::Vector3d& point)
{
    FeatureTrack track;
    track.id = TrackId(id);
    for (const auto& camera : cameras)
    {
        auto projection = geometry::ProjectWorldPoint(K(), camera.initial_world_from_camera, point);
        EXPECT_TRUE(projection.ok());
        track.observations.push_back({camera.id, FeatureId(id), projection.value().pixel});
    }
    return track;
}
TEST(LandmarkBuilderTest, TriangulatesTracksAndCompactsLandmarkIds)
{
    std::vector<CameraFrame> cameras{Camera(0, {0, 0, 0}), Camera(1, {1, 0, 0}),
                                     Camera(2, {0, 1, 0})};
    auto first = Track(4, cameras, {0.2, -0.1, 5});
    auto second = Track(9, cameras, {-0.3, 0.4, 6});
    auto result = Mapper::BuildSparseLandmarks(K(), cameras, {first, second});
    ASSERT_TRUE(result.ok()) << result.status().ToString();
    ASSERT_EQ(result.value().landmarks.size(), 2U);
    EXPECT_EQ(result.value().landmarks[0].id, LandmarkId(0));
    EXPECT_EQ(result.value().landmarks[0].source_track_id, TrackId(4));
    EXPECT_TRUE(
        result.value().landmarks[1].position_world.isApprox(Eigen::Vector3d(-0.3, 0.4, 6), 1e-10));
    EXPECT_GT(result.value().landmarks[0].quality.maximum_parallax_degrees, 0);
    EXPECT_EQ(result.value().landmarks[0].observations.size(), 3U);
}
TEST(LandmarkBuilderTest, UsesOptimizedPosesWhenAvailable)
{
    std::vector<CameraFrame> cameras{Camera(0, {0, 0, 0}), Camera(1, {1, 0, 0})};
    const Eigen::Vector3d point(0.1, 0.2, 4);
    FeatureTrack track = Track(0, cameras, point);
    cameras[1].optimized_world_from_camera = cameras[1].initial_world_from_camera;
    cameras[1].initial_world_from_camera.translation_world_from_local.x() = 2;
    cameras[1].has_optimized_pose = true;
    auto result = Mapper::BuildSparseLandmarks(K(), cameras, {track});
    ASSERT_TRUE(result.ok());
    ASSERT_EQ(result.value().landmarks.size(), 1U);
    EXPECT_TRUE(result.value().landmarks[0].position_world.isApprox(point, 1e-10));
}
TEST(LandmarkBuilderTest, RequiresFourViewsForSupplementalTrack)
{
    std::vector<CameraFrame> cameras{Camera(0, {0, 0, 0}), Camera(1, {1, 0, 0}),
                                     Camera(2, {0, 1, 0})};
    FeatureTrack track = Track(0, cameras, {0.1, 0.2, 4});
    track.is_supplemental = true;

    auto result = Mapper::BuildSparseLandmarks(K(), cameras, {track});

    ASSERT_TRUE(result.ok());
    EXPECT_TRUE(result.value().landmarks.empty());
    EXPECT_EQ(result.value().short_tracks_rejected, 1U);
}
TEST(LandmarkBuilderTest, ReportsShortGeometricAndNumericalRejections)
{
    std::vector<CameraFrame> cameras{Camera(0, {0, 0, 0}), Camera(1, {0.001, 0, 0})};
    FeatureTrack short_track;
    short_track.id = TrackId(0);
    short_track.observations.push_back({CameraId(0), FeatureId(0), {400, 300}});
    auto low = Track(1, cameras, {0, 0, 100});
    FeatureTrack degenerate;
    degenerate.id = TrackId(2);
    degenerate.observations = {{CameraId(0), FeatureId(0), {400, 300}},
                               {CameraId(0), FeatureId(1), {400, 300}}};
    LandmarkBuilderOptions options;
    options.minimum_track_length = 2;
    options.triangulation.minimum_parallax_degrees = 1;
    auto result = Mapper::BuildSparseLandmarks(K(), cameras, {short_track, low}, options);
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result.value().short_tracks_rejected, 1U);
    EXPECT_EQ(result.value().geometric_quality_rejected, 1U);
    EXPECT_TRUE(result.value().landmarks.empty());
    auto invalid = Mapper::BuildSparseLandmarks(K(), cameras, {degenerate}, options);
    EXPECT_FALSE(invalid.ok());
}
TEST(LandmarkBuilderTest, RejectsUnknownCameraReference)
{
    std::vector<CameraFrame> cameras{Camera(0, {0, 0, 0}), Camera(1, {1, 0, 0})};
    FeatureTrack track;
    track.id = TrackId(0);
    track.observations = {{CameraId(0), FeatureId(0), {400, 300}},
                          {CameraId(9), FeatureId(0), {300, 300}}};
    auto result = Mapper::BuildSparseLandmarks(K(), cameras, {track});
    EXPECT_FALSE(result.ok());
    EXPECT_EQ(result.status().code(), ErrorCode::kInvalidArgument);
}
TEST(LandmarkBuilderTest, PrunesInconsistentObservationAndRetriangulates)
{
    std::vector<CameraFrame> cameras{Camera(0, {0, 0, 0}), Camera(1, {1, 0, 0}),
                                     Camera(2, {0, 1, 0}), Camera(3, {-1, 0, 0}),
                                     Camera(4, {0, -1, 0})};
    FeatureTrack track = Track(0, cameras, {0.2, -0.1, 5.0});
    track.observations.back().pixel.x() += 10.0;
    LandmarkBuilderOptions options;
    options.minimum_track_length = 3;
    options.prune_inconsistent_observations = true;
    options.triangulation.maximum_reprojection_error_pixels = 2.0;

    auto result = Mapper::BuildSparseLandmarks(K(), cameras, {track}, options);

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    ASSERT_EQ(result.value().landmarks.size(), 1U);
    EXPECT_EQ(result.value().pruned_observations, 1U);
    EXPECT_EQ(result.value().landmarks[0].observations.size(), 4U);
    EXPECT_TRUE(
        result.value().landmarks[0].position_world.isApprox(Eigen::Vector3d(0.2, -0.1, 5.0), 1e-6));
}
} // namespace
} // namespace lio_visual_ba
