#include "Optimizer.hpp"

#include <gtest/gtest.h>

#include "Geometry.hpp"

namespace lio_visual_ba
{
namespace
{

Reconstruction LocalBundleAdjustmentSample()
{
    Reconstruction reconstruction;
    reconstruction.intrinsics = {600.0, 600.0, 400.0, 300.0, 800, 600};
    std::vector<Pose3d> truth(4);
    truth[1].translation_world_from_local.x() = 1.0;
    truth[2].translation_world_from_local.y() = 1.0;
    truth[3].translation_world_from_local = {10.0, 0.0, 0.0};
    for (int index = 0; index < 4; ++index)
    {
        CameraFrame camera;
        camera.id = CameraId(index);
        camera.initial_world_from_camera = truth[static_cast<std::size_t>(index)];
        reconstruction.cameras.push_back(camera);
    }
    reconstruction.cameras[1].initial_world_from_camera.translation_world_from_local.x() = 1.15;

    for (int index = 0; index < 6; ++index)
    {
        const Eigen::Vector3d point(-0.4 + 0.16 * index, -0.2 + 0.08 * (index % 3),
                                    4.0 + 0.15 * (index % 2));
        Landmark landmark;
        landmark.id = LandmarkId(index);
        landmark.position_world = point + Eigen::Vector3d(0.08, -0.05, 0.12);
        for (int camera_index = 0; camera_index < 3; ++camera_index)
        {
            auto projection = geometry::ProjectWorldPoint(
                reconstruction.intrinsics, truth[static_cast<std::size_t>(camera_index)], point);
            EXPECT_TRUE(projection.ok());
            landmark.observations.push_back(
                {CameraId(camera_index), FeatureId(index), projection.value().pixel});
        }
        reconstruction.landmarks.push_back(landmark);
    }
    return reconstruction;
}

TEST(LocalBundleAdjusterTest, OptimizesAndMergesOnlySelectedWindow)
{
    Reconstruction reconstruction = LocalBundleAdjustmentSample();
    const CameraFrame outside_before = reconstruction.cameras[3];
    LocalBundleAdjustmentOptions options;
    options.window_selection.maximum_cameras = 3;

    auto report = Optimizer::OptimizeLocalBundleAdjustment(reconstruction, CameraId(2), options);

    ASSERT_TRUE(report.ok()) << report.status().ToString();
    EXPECT_EQ(report.value().window.camera_ids,
              (std::vector<CameraId>{CameraId(0), CameraId(1), CameraId(2)}));
    EXPECT_EQ(report.value().bundle_adjustment.optimized_cameras, 3U);
    EXPECT_GT(report.value().bundle_adjustment.initial_reprojection_rmse_pixels,
              report.value().bundle_adjustment.final_reprojection_rmse_pixels);
    EXPECT_TRUE(reconstruction.cameras[0].has_optimized_pose);
    EXPECT_FALSE(reconstruction.cameras[3].has_optimized_pose);
    EXPECT_TRUE(
        reconstruction.cameras[3].initial_world_from_camera.translation_world_from_local.isApprox(
            outside_before.initial_world_from_camera.translation_world_from_local));
}

TEST(LocalBundleAdjusterTest, RejectsAnchorOutsideWindowWithoutMutation)
{
    Reconstruction reconstruction = LocalBundleAdjustmentSample();
    const Reconstruction original = reconstruction;
    LocalBundleAdjustmentOptions options;
    options.window_selection.maximum_cameras = 3;
    options.bundle_adjustment.anchor_camera_id = CameraId(3);

    auto report = Optimizer::OptimizeLocalBundleAdjustment(reconstruction, CameraId(0), options);

    EXPECT_FALSE(report.ok());
    EXPECT_FALSE(reconstruction.cameras[0].has_optimized_pose);
    EXPECT_TRUE(
        reconstruction.landmarks[0].position_world.isApprox(original.landmarks[0].position_world));
}

TEST(LocalBundleAdjusterTest, RejectsUnderSupportedWindowWithoutMutation)
{
    Reconstruction reconstruction = LocalBundleAdjustmentSample();
    LocalBundleAdjustmentOptions options;
    options.window_selection.maximum_cameras = 1;

    auto report = Optimizer::OptimizeLocalBundleAdjustment(reconstruction, CameraId(0), options);

    EXPECT_FALSE(report.ok());
    EXPECT_EQ(report.status().code(), ErrorCode::kFailedPrecondition);
    EXPECT_FALSE(reconstruction.cameras[0].has_optimized_pose);
}

} // namespace
} // namespace lio_visual_ba
