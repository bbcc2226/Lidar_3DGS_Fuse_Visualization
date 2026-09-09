#include "optimizer.hpp"

#include <gtest/gtest.h>

#include "geometry.hpp"

namespace lio_visual_ba
{
namespace
{

Reconstruction IncrementalSample()
{
    Reconstruction reconstruction;
    reconstruction.intrinsics = {500.0, 500.0, 320.0, 240.0, 640, 480};
    std::vector<Pose3d> truth(4);
    for (int index = 0; index < 4; ++index)
    {
        truth[static_cast<std::size_t>(index)].translation_world_from_local.x() = 0.5 * index;
        CameraFrame camera;
        camera.id = CameraId(index);
        camera.initial_world_from_camera = truth[static_cast<std::size_t>(index)];
        if (index > 0)
        {
            camera.initial_world_from_camera.translation_world_from_local.y() = 0.04;
        }
        reconstruction.cameras.push_back(camera);
    }
    for (int index = 0; index < 8; ++index)
    {
        const Eigen::Vector3d point(-0.4 + 0.12 * index, -0.2 + 0.1 * (index % 4),
                                    4.0 + 0.2 * (index % 3));
        Landmark landmark;
        landmark.id = LandmarkId(index);
        landmark.position_world = point + Eigen::Vector3d(0.05, -0.04, 0.1);
        for (int camera_index = 0; camera_index < 4; ++camera_index)
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

TEST(IncrementalOptimizerTest, RunsPrefixLocalAndPeriodicGlobalSolves)
{
    Reconstruction reconstruction = IncrementalSample();
    IncrementalOptimizationOptions options;
    options.global_bundle_adjustment_interval = 2;

    auto report = Optimizer::RunIncrementalOptimization(reconstruction, options);

    ASSERT_TRUE(report.ok()) << report.status().ToString();
    EXPECT_EQ(report.value().local_solve_count, 3U);
    EXPECT_EQ(report.value().global_solve_count, 2U);
    EXPECT_EQ(report.value().locally_optimized_seeds,
              (std::vector<CameraId>{CameraId(1), CameraId(2), CameraId(3)}));
    EXPECT_LT(report.value().final_bundle_adjustment.final_reprojection_rmse_pixels, 0.1);
    EXPECT_TRUE(reconstruction.cameras.back().has_optimized_pose);
}

TEST(IncrementalOptimizerTest, CanDisablePeriodicAndFinalGlobalSolves)
{
    Reconstruction reconstruction = IncrementalSample();
    IncrementalOptimizationOptions options;
    options.global_bundle_adjustment_interval = 0;
    options.run_final_global_bundle_adjustment = false;

    auto report = Optimizer::RunIncrementalOptimization(reconstruction, options);

    ASSERT_TRUE(report.ok()) << report.status().ToString();
    EXPECT_EQ(report.value().local_solve_count, 3U);
    EXPECT_EQ(report.value().global_solve_count, 0U);
}

TEST(IncrementalOptimizerTest, RejectsInvalidThresholdWithoutMutation)
{
    Reconstruction reconstruction = IncrementalSample();
    IncrementalOptimizationOptions options;
    options.minimum_registered_cameras = 1;

    auto report = Optimizer::RunIncrementalOptimization(reconstruction, options);

    EXPECT_FALSE(report.ok());
    EXPECT_EQ(report.status().code(), ErrorCode::kInvalidArgument);
    EXPECT_FALSE(reconstruction.cameras[0].has_optimized_pose);
}

TEST(IncrementalOptimizerTest, SkipsWeakLocalWindowAndContinues)
{
    Reconstruction reconstruction = IncrementalSample();
    reconstruction.landmarks.erase(reconstruction.landmarks.begin(),
                                   reconstruction.landmarks.end() - 1);
    reconstruction.landmarks[0].observations.erase(
        reconstruction.landmarks[0].observations.begin() + 2,
        reconstruction.landmarks[0].observations.end());
    IncrementalOptimizationOptions options;
    options.global_bundle_adjustment_interval = 0;
    options.run_final_global_bundle_adjustment = false;

    auto report = Optimizer::RunIncrementalOptimization(reconstruction, options);

    ASSERT_TRUE(report.ok()) << report.status().ToString();
    // Temporal windows retain fixed boundary observations, so these prefixes
    // remain constrained instead of being rejected as weak covisibility sets.
    EXPECT_EQ(report.value().local_solve_count, 3U);
    EXPECT_EQ(report.value().skipped_local_solve_count, 0U);
}

} // namespace
} // namespace lio_visual_ba
