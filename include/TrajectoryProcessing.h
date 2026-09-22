#pragma once

#include "DataType.h"

#include <limits>
#include <string>
#include <vector>

// One record from COLMAP's images.txt-style optimized camera output. T_cw
// maps world points into the camera; position_world is the translation of its
// inverse and is therefore the camera center in world coordinates.
struct OptimizedCameraPose
{
    int image_id = -1;
    int camera_id = -1;
    std::string image_name;
    SE3d T_cw;
    Vec3d position_world = Vec3d::Zero();
};

struct TrajectoryPoint
{
    int image_id = -1;
    std::string image_name;
    // Position in the coordinate frame supplied to smoothTrajectory().
    Vec3d position = Vec3d::Zero();
};

struct TrajectorySmoothingOptions
{
    // Number of neighboring poses on either side of each output point.
    int window_radius = 4;
    double gaussian_sigma = 2.0;

    // NaN selects the median camera-center Z. Set an explicit value to use a
    // known floor-relative camera height instead.
    double fixed_height = std::numeric_limits<double>::quiet_NaN();
    // Recorded outdoor trajectories must retain road elevation changes.
    bool preserve_height = false;
};

class TrajectoryProcessing
{
public:
    bool loadOptimizedCameraPoses(const std::string& path);
    bool smoothTrajectory(const TrajectorySmoothingOptions& options = {});
    bool smoothTrajectory(const Mat4d& world_to_output,
                          const TrajectorySmoothingOptions& options = {});
    bool saveSmoothTrajectory(const std::string& path) const;

    const std::vector<OptimizedCameraPose>& poses() const { return poses_; }
    const std::vector<TrajectoryPoint>& smoothedTrajectory() const
    {
        return smooth_trajectory_;
    }
    double trajectoryHeight() const { return trajectory_height_; }
    const std::string& lastError() const { return last_error_; }

private:
    std::vector<OptimizedCameraPose> poses_;
    std::vector<TrajectoryPoint> smooth_trajectory_;
    double trajectory_height_ = 0.0;
    std::string last_error_;
};
