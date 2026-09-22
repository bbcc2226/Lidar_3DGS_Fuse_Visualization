#include "Optimizer.hpp"

#include <gtest/gtest.h>

namespace lio_visual_ba
{
namespace
{

Reconstruction SelectionSample()
{
    Reconstruction reconstruction;
    for (int index = 0; index < 5; ++index)
    {
        CameraFrame camera;
        camera.id = CameraId(index);
        reconstruction.cameras.push_back(camera);
    }
    const std::vector<std::vector<int>> observers{{0, 1, 2}, {0, 1}, {1, 2}, {2, 3}, {3, 4}};
    for (std::size_t index = 0; index < observers.size(); ++index)
    {
        Landmark landmark;
        landmark.id = LandmarkId(static_cast<int>(index));
        for (int camera_id : observers[index])
        {
            landmark.observations.push_back(
                {CameraId(camera_id), FeatureId(static_cast<int>(index)), {0.0, 0.0}});
        }
        reconstruction.landmarks.push_back(landmark);
    }
    return reconstruction;
}

TEST(LocalWindowSelectorTest, GrowsByCovisibilityWithinCameraLimit)
{
    const Reconstruction reconstruction = SelectionSample();
    LocalWindowSelectionOptions options;
    options.maximum_cameras = 3;

    auto window =
        Optimizer::SelectLocalBundleAdjustmentWindow(reconstruction, CameraId(0), options);

    ASSERT_TRUE(window.ok()) << window.status().ToString();
    EXPECT_EQ(window.value().camera_ids,
              (std::vector<CameraId>{CameraId(0), CameraId(1), CameraId(2)}));
    EXPECT_EQ(window.value().landmark_ids,
              (std::vector<LandmarkId>{LandmarkId(0), LandmarkId(1), LandmarkId(2)}));
}

TEST(LocalWindowSelectorTest, StopsWhenRemainingCamerasLackSharedSupport)
{
    const Reconstruction reconstruction = SelectionSample();
    LocalWindowSelectionOptions options;
    options.maximum_cameras = 5;
    options.minimum_shared_landmarks = 2;

    auto window =
        Optimizer::SelectLocalBundleAdjustmentWindow(reconstruction, CameraId(0), options);

    ASSERT_TRUE(window.ok());
    EXPECT_EQ(window.value().camera_ids,
              (std::vector<CameraId>{CameraId(0), CameraId(1), CameraId(2)}));
}

TEST(LocalWindowSelectorTest, UsesCameraIdAsDeterministicTieBreaker)
{
    const Reconstruction reconstruction = SelectionSample();
    LocalWindowSelectionOptions options;
    options.maximum_cameras = 2;

    auto window =
        Optimizer::SelectLocalBundleAdjustmentWindow(reconstruction, CameraId(2), options);

    ASSERT_TRUE(window.ok());
    EXPECT_EQ(window.value().camera_ids, (std::vector<CameraId>{CameraId(1), CameraId(2)}));
}

TEST(LocalWindowSelectorTest, RejectsUnknownSeedAndBrokenObservation)
{
    Reconstruction reconstruction = SelectionSample();
    EXPECT_FALSE(Optimizer::SelectLocalBundleAdjustmentWindow(reconstruction, CameraId(99)).ok());

    reconstruction.landmarks[0].observations[0].camera_id = CameraId(99);
    auto window = Optimizer::SelectLocalBundleAdjustmentWindow(reconstruction, CameraId(0));

    EXPECT_FALSE(window.ok());
    EXPECT_EQ(window.status().code(), ErrorCode::kInvalidArgument);
}

} // namespace
} // namespace lio_visual_ba
