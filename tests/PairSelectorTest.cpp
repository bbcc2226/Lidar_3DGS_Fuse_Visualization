#include "Mapper.hpp"

#include <gtest/gtest.h>

namespace lio_visual_ba
{
namespace
{

CameraFrame Camera(int id, const Eigen::Vector3d& center)
{
    CameraFrame camera;
    camera.id = CameraId(id);
    camera.initial_world_from_camera.translation_world_from_local = center;
    return camera;
}

TEST(PairSelectorTest, CreatesTemporalWindowPairs)
{
    std::vector<CameraFrame> cameras;
    for (int index = 0; index < 5; ++index)
    {
        cameras.push_back(Camera(index, Eigen::Vector3d(index, 0.0, 0.0)));
    }
    PairSelectionOptions options;
    options.maximum_temporal_gap = 2;
    options.minimum_loop_separation = 4;
    options.maximum_loop_distance = 0.0;

    auto result = Mapper::SelectCandidatePairs(cameras, options);

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    ASSERT_EQ(result.value().temporal_pairs.size(), 7U);
    EXPECT_EQ(result.value().temporal_pairs.front().first_camera_id, CameraId(0));
    EXPECT_EQ(result.value().temporal_pairs.front().second_camera_id, CameraId(1));
    EXPECT_EQ(result.value().temporal_pairs.back().first_camera_id, CameraId(3));
    EXPECT_EQ(result.value().temporal_pairs.back().second_camera_id, CameraId(4));
}

TEST(PairSelectorTest, SelectsNearestSpatialLoopsWithViewGate)
{
    std::vector<CameraFrame> cameras{Camera(0, {0.0, 0.0, 0.0}), Camera(1, {5.0, 0.0, 0.0}),
                                     Camera(2, {6.0, 0.0, 0.0}), Camera(3, {0.4, 0.0, 0.0}),
                                     Camera(4, {0.2, 0.0, 0.0}), Camera(5, {0.3, 0.0, 0.0})};
    cameras[3].initial_world_from_camera.rotation_world_from_local =
        Eigen::AngleAxisd(3.14159265358979323846, Eigen::Vector3d::UnitY());
    PairSelectionOptions options;
    options.maximum_temporal_gap = 1;
    options.minimum_loop_separation = 3;
    options.maximum_loop_distance = 0.5;
    options.maximum_loops_per_frame = 2;
    options.maximum_loop_view_angle_degrees = 30.0;

    auto result = Mapper::SelectCandidatePairs(cameras, options);

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    ASSERT_EQ(result.value().loop_pairs.size(), 2U);
    EXPECT_EQ(result.value().loop_pairs[0].first_camera_id, CameraId(0));
    EXPECT_EQ(result.value().loop_pairs[0].second_camera_id, CameraId(4));
    EXPECT_EQ(result.value().loop_pairs[1].second_camera_id, CameraId(5));
    EXPECT_TRUE(result.value().loop_pairs[0].is_loop);
}

TEST(PairSelectorTest, UsesOptimizedPoseWhenRequested)
{
    std::vector<CameraFrame> cameras{Camera(0, {0.0, 0.0, 0.0}), Camera(1, {5.0, 0.0, 0.0}),
                                     Camera(2, {5.0, 0.0, 0.0})};
    cameras[2].optimized_world_from_camera.translation_world_from_local =
        Eigen::Vector3d(0.1, 0.0, 0.0);
    cameras[2].has_optimized_pose = true;
    PairSelectionOptions options;
    options.maximum_temporal_gap = 1;
    options.minimum_loop_separation = 2;
    options.maximum_loop_distance = 0.2;

    auto optimized = Mapper::SelectCandidatePairs(cameras, options);
    ASSERT_TRUE(optimized.ok());
    EXPECT_EQ(optimized.value().loop_pairs.size(), 1U);

    options.prefer_optimized_camera_poses = false;
    auto initial = Mapper::SelectCandidatePairs(cameras, options);
    ASSERT_TRUE(initial.ok());
    EXPECT_TRUE(initial.value().loop_pairs.empty());
}

TEST(PairSelectorTest, RejectsInvalidOptionsAndDuplicateCameraIds)
{
    PairSelectionOptions options;
    options.maximum_temporal_gap = 0;
    EXPECT_FALSE(Mapper::SelectCandidatePairs({Camera(0, Eigen::Vector3d::Zero())}, options).ok());

    options.maximum_temporal_gap = 1;
    auto duplicate = Mapper::SelectCandidatePairs(
        {Camera(0, Eigen::Vector3d::Zero()), Camera(0, Eigen::Vector3d::Zero())}, options);
    EXPECT_FALSE(duplicate.ok());
    EXPECT_EQ(duplicate.status().code(), ErrorCode::kInvalidArgument);
}

} // namespace
} // namespace lio_visual_ba
