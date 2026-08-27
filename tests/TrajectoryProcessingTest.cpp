#include "TrajectoryProcessing.h"

#include <gtest/gtest.h>

#include <QTemporaryFile>

namespace
{
bool writeText(QTemporaryFile& file, const QByteArray& text)
{
    return file.open() && file.write(text) == text.size() && file.flush();
}
} // namespace

TEST(TrajectoryProcessingTest, LoadsColmapWorldToCameraPoses)
{
    QTemporaryFile file;
    ASSERT_TRUE(writeText(file,
        "# IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID IMAGE_NAME\n"
        "2 1 0 0 0 -4 -5 -6 1 second.png\n\n"
        "1 1 0 0 0 -1 -2 -3 1 first.png\n"));

    TrajectoryProcessing processing;
    ASSERT_TRUE(processing.loadOptimizedCameraPoses(
        file.fileName().toStdString())) << processing.lastError();
    ASSERT_EQ(processing.poses().size(), 2U);
    EXPECT_EQ(processing.poses()[0].image_id, 1);
    EXPECT_TRUE(processing.poses()[0].position_world.isApprox(
        Vec3d(1.0, 2.0, 3.0)));
    EXPECT_TRUE(processing.poses()[1].position_world.isApprox(
        Vec3d(4.0, 5.0, 6.0)));
}

TEST(TrajectoryProcessingTest, SmoothsXYAndUsesMedianFixedHeight)
{
    QTemporaryFile file;
    ASSERT_TRUE(writeText(file,
        "1 1 0 0 0 0 0 -1 1 a.png\n"
        "2 1 0 0 0 -10 0 -2 1 b.png\n"
        "3 1 0 0 0 0 0 -3 1 c.png\n"));

    TrajectoryProcessing processing;
    ASSERT_TRUE(processing.loadOptimizedCameraPoses(
        file.fileName().toStdString())) << processing.lastError();
    TrajectorySmoothingOptions options;
    options.window_radius = 1;
    options.gaussian_sigma = 1.0;
    ASSERT_TRUE(processing.smoothTrajectory(options))
        << processing.lastError();

    ASSERT_EQ(processing.smoothedTrajectory().size(), 3U);
    EXPECT_DOUBLE_EQ(processing.trajectoryHeight(), 2.0);
    EXPECT_GT(processing.smoothedTrajectory()[0].position_world.x(), 0.0);
    EXPECT_LT(processing.smoothedTrajectory()[1].position_world.x(), 10.0);
    for (const TrajectoryPoint& point : processing.smoothedTrajectory())
        EXPECT_DOUBLE_EQ(point.position_world.z(), 2.0);
}

TEST(TrajectoryProcessingTest, RejectsMalformedPose)
{
    QTemporaryFile file;
    ASSERT_TRUE(writeText(file, "1 1 0 0 0 incomplete\n"));
    TrajectoryProcessing processing;
    EXPECT_FALSE(processing.loadOptimizedCameraPoses(
        file.fileName().toStdString()));
    EXPECT_NE(processing.lastError().find("line 1"), std::string::npos);
}
