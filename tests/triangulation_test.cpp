#include "geometry.hpp"

#include <gtest/gtest.h>

namespace lio_visual_ba
{
namespace
{

CameraIntrinsics TestIntrinsics()
{
    CameraIntrinsics intrinsics;
    intrinsics.fx = 600.0;
    intrinsics.fy = 600.0;
    intrinsics.cx = 400.0;
    intrinsics.cy = 300.0;
    intrinsics.width = 800;
    intrinsics.height = 600;
    return intrinsics;
}

TriangulationObservation Observe(const CameraIntrinsics& intrinsics, const Pose3d& pose,
                                 const Eigen::Vector3d& point_world)
{
    auto projection = geometry::ProjectWorldPoint(intrinsics, pose, point_world);
    EXPECT_TRUE(projection.ok()) << projection.status().ToString();
    return {pose, projection.value().pixel};
}

TEST(TriangulationTest, RecoversExactPointFromMultipleViews)
{
    const CameraIntrinsics intrinsics = TestIntrinsics();
    const Eigen::Vector3d expected(0.25, -0.1, 5.0);
    Pose3d first;
    Pose3d second;
    second.translation_world_from_local.x() = 1.0;
    Pose3d third;
    third.translation_world_from_local.y() = 0.8;
    const std::vector<TriangulationObservation> observations{Observe(intrinsics, first, expected),
                                                             Observe(intrinsics, second, expected),
                                                             Observe(intrinsics, third, expected)};

    auto result = geometry::TriangulateMultiView(intrinsics, observations);

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    EXPECT_TRUE(result.value().point_world.isApprox(expected, 1e-10));
    EXPECT_NEAR(result.value().minimum_depth, 5.0, 1e-10);
    EXPECT_LT(result.value().maximum_reprojection_error_pixels, 1e-9);
    EXPECT_GT(result.value().maximum_parallax_degrees, 10.0);
    EXPECT_EQ(result.value().reprojection_errors_pixels.size(), 3U);
}

TEST(TriangulationTest, ReportsNoisyObservationQuality)
{
    const CameraIntrinsics intrinsics = TestIntrinsics();
    const Eigen::Vector3d point(0.2, 0.3, 4.0);
    Pose3d first;
    Pose3d second;
    second.translation_world_from_local.x() = 1.0;
    Pose3d third;
    third.translation_world_from_local.y() = 1.0;
    std::vector<TriangulationObservation> observations{Observe(intrinsics, first, point),
                                                       Observe(intrinsics, second, point),
                                                       Observe(intrinsics, third, point)};
    observations[2].pixel += Eigen::Vector2d(1.0, -0.5);

    auto result = geometry::TriangulateMultiView(intrinsics, observations);

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    EXPECT_GT(result.value().median_reprojection_error_pixels, 0.0);
    EXPECT_GE(result.value().maximum_reprojection_error_pixels,
              result.value().median_reprojection_error_pixels);
    EXPECT_TRUE(result.value().point_world.isApprox(point, 0.02));
}

TEST(TriangulationTest, AppliesParallaxAndReprojectionGates)
{
    const CameraIntrinsics intrinsics = TestIntrinsics();
    const Eigen::Vector3d point(0.0, 0.0, 100.0);
    Pose3d first;
    Pose3d second;
    second.translation_world_from_local.x() = 0.01;
    std::vector<TriangulationObservation> observations{Observe(intrinsics, first, point),
                                                       Observe(intrinsics, second, point)};
    TriangulationOptions options;
    options.minimum_parallax_degrees = 1.0;
    auto low_parallax = geometry::TriangulateMultiView(intrinsics, observations, options);
    ASSERT_FALSE(low_parallax.ok());
    EXPECT_EQ(low_parallax.status().code(), ErrorCode::kFailedPrecondition);

    second.translation_world_from_local.x() = 1.0;
    observations = {Observe(intrinsics, first, Eigen::Vector3d(0.0, 0.0, 5.0)),
                    Observe(intrinsics, second, Eigen::Vector3d(0.0, 0.0, 5.0))};
    observations[1].pixel.y() += 10.0;
    options.minimum_parallax_degrees = 0.0;
    options.maximum_reprojection_error_pixels = 1.0;
    auto high_error = geometry::TriangulateMultiView(intrinsics, observations, options);
    ASSERT_FALSE(high_error.ok());
    EXPECT_EQ(high_error.status().code(), ErrorCode::kFailedPrecondition);
}

TEST(TriangulationTest, RejectsPointBehindCameras)
{
    const CameraIntrinsics intrinsics = TestIntrinsics();
    Pose3d first;
    Pose3d second;
    second.translation_world_from_local.x() = 1.0;
    // These are algebraically consistent projections of (0, 0, -5), but the
    // point is not visible because both depths are negative.
    const std::vector<TriangulationObservation> observations{
        {first, Eigen::Vector2d(400.0, 300.0)}, {second, Eigen::Vector2d(520.0, 300.0)}};

    auto result = geometry::TriangulateMultiView(intrinsics, observations);

    ASSERT_FALSE(result.ok());
    EXPECT_EQ(result.status().code(), ErrorCode::kFailedPrecondition);
}

TEST(TriangulationTest, RejectsInsufficientOrDegenerateObservations)
{
    const CameraIntrinsics intrinsics = TestIntrinsics();
    std::vector<TriangulationObservation> one(1);
    EXPECT_EQ(geometry::TriangulateMultiView(intrinsics, one).status().code(),
              ErrorCode::kInvalidArgument);

    std::vector<TriangulationObservation> coincident(2);
    coincident[0].pixel = Eigen::Vector2d(400.0, 300.0);
    coincident[1].pixel = coincident[0].pixel;
    auto degenerate = geometry::TriangulateMultiView(intrinsics, coincident);
    EXPECT_FALSE(degenerate.ok());
    EXPECT_EQ(degenerate.status().code(), ErrorCode::kNumericalFailure);
}

} // namespace
} // namespace lio_visual_ba
