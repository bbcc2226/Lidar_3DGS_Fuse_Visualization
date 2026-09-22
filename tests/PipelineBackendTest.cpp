#include "Pipeline.hpp"

#include <gtest/gtest.h>

#include "Geometry.hpp"

namespace lio_visual_ba
{
namespace
{

PipelineFrontendResult BackendSample()
{
    PipelineFrontendResult frontend;
    frontend.reconstruction.intrinsics = {500.0, 500.0, 320.0, 240.0, 640, 480};
    std::vector<Pose3d> poses(3);
    poses[1].translation_world_from_local.x() = 0.7;
    poses[2].translation_world_from_local.y() = 0.7;
    for (int camera_index = 0; camera_index < 3; ++camera_index)
    {
        CameraFrame camera;
        camera.id = CameraId(camera_index);
        camera.initial_world_from_camera = poses[static_cast<std::size_t>(camera_index)];
        frontend.reconstruction.cameras.push_back(camera);
        FeatureSet features;
        features.camera_id = camera.id;
        features.core_feature_count = 8;
        frontend.feature_sets.push_back(features);
    }
    for (int feature_index = 0; feature_index < 8; ++feature_index)
    {
        const Eigen::Vector3d point(-0.35 + 0.1 * feature_index, -0.2 + 0.1 * (feature_index % 4),
                                    4.0);
        for (int camera_index = 0; camera_index < 3; ++camera_index)
        {
            auto projection =
                geometry::ProjectWorldPoint(frontend.reconstruction.intrinsics,
                                            poses[static_cast<std::size_t>(camera_index)], point);
            EXPECT_TRUE(projection.ok());
            Feature feature;
            feature.id = FeatureId(feature_index);
            feature.pixel = projection.value().pixel;
            frontend.feature_sets[static_cast<std::size_t>(camera_index)].features.push_back(
                feature);
        }
        frontend.verified_matches.push_back(
            {CameraId(0), FeatureId(feature_index), CameraId(1), FeatureId(feature_index)});
        frontend.verified_matches.push_back(
            {CameraId(1), FeatureId(feature_index), CameraId(2), FeatureId(feature_index)});
    }
    return frontend;
}

PipelineConfig BackendConfig()
{
    PipelineConfig config;
    config.landmark_builder.triangulation.minimum_parallax_degrees = 0.1;
    config.landmark_builder.triangulation.maximum_reprojection_error_pixels = 1.0;
    config.incremental_optimization.global_bundle_adjustment_interval = 0;
    config.final_refinement.maximum_reprojection_error_pixels = 2.0;
    return config;
}

TEST(PipelineBackendTest, BuildsTracksLandmarksAndOptimizes)
{
    auto result = Pipeline::RunPipelineBackend(BackendSample(), BackendConfig());

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    EXPECT_EQ(result.value().reconstruction.tracks.size(), 8U);
    EXPECT_EQ(result.value().reconstruction.landmarks.size(), 8U);
    EXPECT_EQ(result.value().track_build.accepted_matches, 16U);
    EXPECT_EQ(result.value().incremental_optimization.local_solve_count, 2U);
    EXPECT_LT(
        result.value().final_refinement.final_bundle_adjustment.final_reprojection_rmse_pixels,
        1e-5);
}

TEST(PipelineBackendTest, PropagatesTrackConstructionFailure)
{
    PipelineFrontendResult frontend = BackendSample();
    frontend.verified_matches[0].first_camera_id = CameraId(99);

    auto result = Pipeline::RunPipelineBackend(std::move(frontend), BackendConfig());

    EXPECT_FALSE(result.ok());
    EXPECT_EQ(result.status().code(), ErrorCode::kInvalidArgument);
    EXPECT_NE(result.status().message().find("pipeline track construction"), std::string::npos);
}

} // namespace
} // namespace lio_visual_ba
