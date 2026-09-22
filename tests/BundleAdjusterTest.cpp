#include "Optimizer.hpp"

#include <gtest/gtest.h>

#include "Geometry.hpp"

namespace lio_visual_ba
{
namespace
{

Reconstruction BundleAdjustmentSample()
{
    Reconstruction reconstruction;
    reconstruction.intrinsics = {600.0, 600.0, 400.0, 300.0, 800, 600};
    for (int index = 0; index < 3; ++index)
    {
        CameraFrame camera;
        camera.id = CameraId(index);
        camera.initial_world_from_camera.translation_world_from_local =
            Eigen::Vector3d(index == 1 ? 1.0 : 0.0, index == 2 ? 1.0 : 0.0, 0.0);
        reconstruction.cameras.push_back(camera);
    }
    const Eigen::Vector3d expected(0.2, -0.1, 5.0);
    Landmark landmark;
    landmark.id = LandmarkId(0);
    landmark.source_track_id = TrackId(0);
    landmark.position_world = expected + Eigen::Vector3d(0.3, -0.2, 0.5);
    for (const CameraFrame& camera : reconstruction.cameras)
    {
        auto projection = geometry::ProjectWorldPoint(reconstruction.intrinsics,
                                                      camera.initial_world_from_camera, expected);
        EXPECT_TRUE(projection.ok());
        landmark.observations.push_back({camera.id, FeatureId(0), projection.value().pixel});
    }
    reconstruction.landmarks.push_back(landmark);
    return reconstruction;
}

Reconstruction JointBundleAdjustmentSample()
{
    Reconstruction reconstruction;
    reconstruction.intrinsics = {600.0, 600.0, 400.0, 300.0, 800, 600};
    std::vector<Pose3d> true_poses(3);
    true_poses[1].translation_world_from_local.x() = 1.0;
    true_poses[2].translation_world_from_local.y() = 1.0;
    for (int index = 0; index < 3; ++index)
    {
        CameraFrame camera;
        camera.id = CameraId(index);
        camera.initial_world_from_camera = true_poses[static_cast<std::size_t>(index)];
        reconstruction.cameras.push_back(camera);
    }
    reconstruction.cameras[1].initial_world_from_camera.translation_world_from_local.x() = 1.12;
    reconstruction.cameras[1].initial_world_from_camera.rotation_world_from_local =
        Eigen::AngleAxisd(0.025, Eigen::Vector3d::UnitY());
    reconstruction.cameras[2].initial_world_from_camera.translation_world_from_local.y() = 0.9;

    for (int index = 0; index < 8; ++index)
    {
        const Eigen::Vector3d truth(-0.5 + 0.18 * index, -0.3 + 0.12 * (index % 4),
                                    4.0 + 0.2 * (index % 3));
        Landmark landmark;
        landmark.id = LandmarkId(index);
        landmark.source_track_id = TrackId(index);
        landmark.position_world = truth + Eigen::Vector3d(0.08, -0.06, 0.15);
        for (int camera_index = 0; camera_index < 3; ++camera_index)
        {
            auto projection = geometry::ProjectWorldPoint(
                reconstruction.intrinsics, true_poses[static_cast<std::size_t>(camera_index)],
                truth);
            EXPECT_TRUE(projection.ok());
            landmark.observations.push_back(
                {CameraId(camera_index), FeatureId(index), projection.value().pixel});
        }
        reconstruction.landmarks.push_back(landmark);
    }
    return reconstruction;
}

void AddFourthConsistentView(Reconstruction& reconstruction)
{
    CameraFrame camera;
    camera.id = CameraId(3);
    camera.initial_world_from_camera.translation_world_from_local = {1.0, 1.0, 0.0};
    reconstruction.cameras.push_back(camera);
    for (int index = 0; index < 8; ++index)
    {
        const Eigen::Vector3d truth(-0.5 + 0.18 * index, -0.3 + 0.12 * (index % 4),
                                    4.0 + 0.2 * (index % 3));
        auto projection = geometry::ProjectWorldPoint(reconstruction.intrinsics,
                                                      camera.initial_world_from_camera, truth);
        EXPECT_TRUE(projection.ok());
        reconstruction.landmarks[static_cast<std::size_t>(index)].observations.push_back(
            {camera.id, FeatureId(index), projection.value().pixel});
    }
}

TEST(BundleAdjusterTest, LandmarkOnlyOptimizationReducesReprojectionError)
{
    Reconstruction reconstruction = BundleAdjustmentSample();

    auto report = Optimizer::OptimizeLandmarks(reconstruction);

    ASSERT_TRUE(report.ok()) << report.status().ToString();
    EXPECT_EQ(report.value().optimized_landmarks, 1U);
    EXPECT_EQ(report.value().residual_observations, 3U);
    EXPECT_GT(report.value().initial_reprojection_rmse_pixels,
              report.value().final_reprojection_rmse_pixels);
    EXPECT_LT(report.value().final_reprojection_rmse_pixels, 1e-6);
    EXPECT_TRUE(
        reconstruction.landmarks[0].position_world.isApprox(Eigen::Vector3d(0.2, -0.1, 5.0), 1e-6));
}

TEST(BundleAdjusterTest, UsesOptimizedCameraPoses)
{
    Reconstruction reconstruction = BundleAdjustmentSample();
    CameraFrame& second = reconstruction.cameras[1];
    second.optimized_world_from_camera = second.initial_world_from_camera;
    second.has_optimized_pose = true;
    second.initial_world_from_camera.translation_world_from_local.x() = 2.0;

    auto report = Optimizer::OptimizeLandmarks(reconstruction);

    ASSERT_TRUE(report.ok()) << report.status().ToString();
    EXPECT_LT(report.value().final_reprojection_rmse_pixels, 1e-6);
}

TEST(BundleAdjusterTest, SkipsShortLandmarks)
{
    Reconstruction reconstruction = BundleAdjustmentSample();
    Landmark short_landmark;
    short_landmark.id = LandmarkId(1);
    short_landmark.source_track_id = TrackId(1);
    short_landmark.position_world = {0.0, 0.0, 4.0};
    short_landmark.observations.push_back({CameraId(0), FeatureId(1), {400.0, 300.0}});
    reconstruction.landmarks.push_back(short_landmark);

    auto report = Optimizer::OptimizeLandmarks(reconstruction);

    ASSERT_TRUE(report.ok());
    EXPECT_EQ(report.value().skipped_landmarks, 1U);
    EXPECT_TRUE(
        reconstruction.landmarks[1].position_world.isApprox(Eigen::Vector3d(0.0, 0.0, 4.0)));
}

TEST(BundleAdjusterTest, RejectsInvalidReferencesWithoutMutation)
{
    Reconstruction reconstruction = BundleAdjustmentSample();
    reconstruction.landmarks[0].observations[0].camera_id = CameraId(99);
    const Eigen::Vector3d original = reconstruction.landmarks[0].position_world;

    auto report = Optimizer::OptimizeLandmarks(reconstruction);

    EXPECT_FALSE(report.ok());
    EXPECT_EQ(report.status().code(), ErrorCode::kInvalidArgument);
    EXPECT_TRUE(reconstruction.landmarks[0].position_world.isApprox(original));
}

TEST(BundleAdjusterTest, RejectsInvalidOptions)
{
    Reconstruction reconstruction = BundleAdjustmentSample();
    LandmarkBundleAdjustmentOptions options;
    options.maximum_iterations = 0;

    auto report = Optimizer::OptimizeLandmarks(reconstruction, options);

    EXPECT_FALSE(report.ok());
    EXPECT_EQ(report.status().code(), ErrorCode::kInvalidArgument);
}

TEST(BundleAdjusterTest, JointOptimizationImprovesPosesAndLandmarks)
{
    Reconstruction reconstruction = JointBundleAdjustmentSample();
    JointBundleAdjustmentOptions options;
    options.translation_prior_sigma = 0.5;
    options.rotation_prior_sigma = 0.2;

    auto report = Optimizer::OptimizeCamerasAndLandmarks(reconstruction, options);

    ASSERT_TRUE(report.ok()) << report.status().ToString();
    EXPECT_EQ(report.value().optimized_cameras, 3U);
    EXPECT_EQ(report.value().optimized_landmarks, 8U);
    EXPECT_GT(report.value().initial_reprojection_rmse_pixels,
              report.value().final_reprojection_rmse_pixels);
    EXPECT_LT(report.value().final_reprojection_rmse_pixels, 0.1);
    EXPECT_TRUE(reconstruction.cameras[0].has_optimized_pose);
    EXPECT_NEAR(
        reconstruction.cameras[1].optimized_world_from_camera.translation_world_from_local.x(), 1.0,
        0.05);
}

TEST(BundleAdjusterTest, JointOptimizationSupportsExplicitAnchor)
{
    Reconstruction reconstruction = JointBundleAdjustmentSample();
    const Pose3d anchor_before = reconstruction.cameras[1].initial_world_from_camera;
    JointBundleAdjustmentOptions options;
    options.anchor_camera_id = CameraId(1);

    auto report = Optimizer::OptimizeCamerasAndLandmarks(reconstruction, options);

    ASSERT_TRUE(report.ok()) << report.status().ToString();
    EXPECT_TRUE(
        reconstruction.cameras[1].optimized_world_from_camera.translation_world_from_local.isApprox(
            anchor_before.translation_world_from_local, 1e-12));
}

TEST(BundleAdjusterTest, JointOptimizationRejectsUnknownAnchorWithoutMutation)
{
    Reconstruction reconstruction = JointBundleAdjustmentSample();
    const Reconstruction original = reconstruction;
    JointBundleAdjustmentOptions options;
    options.anchor_camera_id = CameraId(99);

    auto report = Optimizer::OptimizeCamerasAndLandmarks(reconstruction, options);

    EXPECT_FALSE(report.ok());
    EXPECT_EQ(report.status().code(), ErrorCode::kInvalidArgument);
    EXPECT_TRUE(
        reconstruction.landmarks[0].position_world.isApprox(original.landmarks[0].position_world));
    EXPECT_FALSE(reconstruction.cameras[0].has_optimized_pose);
}

TEST(BundleAdjusterTest, OutlierCleanupRemovesGrossObservationAndResolves)
{
    Reconstruction reconstruction = JointBundleAdjustmentSample();
    AddFourthConsistentView(reconstruction);
    reconstruction.landmarks[0].observations[3].pixel.y() += 100.0;
    OutlierRefinementOptions options;
    options.maximum_reprojection_error_pixels = 3.0;

    auto report = Optimizer::RefineWithOutlierCleanup(reconstruction, options);

    ASSERT_TRUE(report.ok()) << report.status().ToString();
    EXPECT_GE(report.value().cleanup_cycles, 2);
    EXPECT_GE(report.value().removed_observations, 1U);
    EXPECT_EQ(report.value().removed_landmarks, 0U);
    ASSERT_EQ(reconstruction.landmarks.size(), 8U);
    EXPECT_EQ(reconstruction.landmarks[0].observations.size(), 3U);
    EXPECT_LT(report.value().final_bundle_adjustment.final_reprojection_rmse_pixels, 0.1);
}

TEST(BundleAdjusterTest, OutlierCleanupDropsLandmarkBelowMinimumSupport)
{
    Reconstruction reconstruction = JointBundleAdjustmentSample();
    AddFourthConsistentView(reconstruction);
    reconstruction.landmarks[0].observations[3].pixel.x() += 100.0;
    OutlierRefinementOptions options;
    options.bundle_adjustment.minimum_landmark_observations = 4;
    options.maximum_reprojection_error_pixels = 3.0;

    auto report = Optimizer::RefineWithOutlierCleanup(reconstruction, options);

    ASSERT_TRUE(report.ok()) << report.status().ToString();
    EXPECT_GE(report.value().removed_observations, 1U);
    EXPECT_EQ(report.value().removed_landmarks, 1U);
    ASSERT_EQ(reconstruction.landmarks.size(), 7U);
    EXPECT_EQ(reconstruction.landmarks.front().id, LandmarkId(1));
}

TEST(BundleAdjusterTest, OutlierCleanupRejectsInvalidOptionsWithoutMutation)
{
    Reconstruction reconstruction = JointBundleAdjustmentSample();
    const Reconstruction original = reconstruction;
    OutlierRefinementOptions options;
    options.maximum_cleanup_cycles = 0;

    auto report = Optimizer::RefineWithOutlierCleanup(reconstruction, options);

    EXPECT_FALSE(report.ok());
    EXPECT_EQ(report.status().code(), ErrorCode::kInvalidArgument);
    EXPECT_TRUE(
        reconstruction.landmarks[0].position_world.isApprox(original.landmarks[0].position_world));
    EXPECT_FALSE(reconstruction.cameras[0].has_optimized_pose);
}

} // namespace
} // namespace lio_visual_ba
