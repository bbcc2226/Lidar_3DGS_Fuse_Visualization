#include "TrajectoryProcessing.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>

namespace
{
bool isFinite(const Vec3d& value)
{
    return value.array().isFinite().all();
}

double medianHeight(const std::vector<OptimizedCameraPose>& poses)
{
    std::vector<double> heights;
    heights.reserve(poses.size());
    for (const OptimizedCameraPose& pose : poses)
        heights.push_back(pose.position_world.z());

    const std::size_t middle = heights.size() / 2;
    std::nth_element(heights.begin(), heights.begin() + middle, heights.end());
    if (heights.size() % 2 != 0) return heights[middle];
    const double upper = heights[middle];
    std::nth_element(
        heights.begin(), heights.begin() + middle - 1, heights.end());
    return 0.5 * (heights[middle - 1] + upper);
}
} // namespace

bool TrajectoryProcessing::loadOptimizedCameraPoses(const std::string& path)
{
    poses_.clear();
    smooth_trajectory_.clear();
    last_error_.clear();

    std::ifstream input(path);
    if (!input.is_open()) {
        last_error_ = "Cannot open optimized camera pose file: " + path;
        return false;
    }

    std::string line;
    int line_number = 0;
    while (std::getline(input, line)) {
        ++line_number;
        const std::size_t first = line.find_first_not_of(" \t\r");
        if (first == std::string::npos || line[first] == '#') continue;

        int image_id = -1;
        int camera_id = -1;
        std::string image_name;
        double qw = 0.0;
        double qx = 0.0;
        double qy = 0.0;
        double qz = 0.0;
        double tx = 0.0;
        double ty = 0.0;
        double tz = 0.0;
        std::istringstream values(line);
        if (!(values >> image_id >> qw >> qx >> qy >> qz
                     >> tx >> ty >> tz >> camera_id >> image_name)) {
            last_error_ = "Malformed optimized camera pose at line " +
                std::to_string(line_number);
            poses_.clear();
            return false;
        }

        Eigen::Quaterniond quaternion(qw, qx, qy, qz);
        const Vec3d translation(tx, ty, tz);
        if (image_id < 0 || camera_id < 0 || !isFinite(translation) ||
            !std::isfinite(quaternion.norm()) || quaternion.norm() < 1.0e-12) {
            last_error_ = "Invalid optimized camera pose at line " +
                std::to_string(line_number);
            poses_.clear();
            return false;
        }
        quaternion.normalize();

        OptimizedCameraPose pose;
        pose.image_id = image_id;
        pose.camera_id = camera_id;
        pose.image_name = image_name;
        // COLMAP stores p_c = R_cw * p_w + t_cw.
        pose.T_cw = SE3d(SO3d(quaternion), translation);
        pose.position_world = pose.T_cw.inverse().translation();
        poses_.push_back(std::move(pose));
    }

    if (poses_.empty()) {
        last_error_ = "Optimized camera pose file contains no poses: " + path;
        return false;
    }

    std::stable_sort(poses_.begin(), poses_.end(),
        [](const OptimizedCameraPose& left, const OptimizedCameraPose& right) {
            return left.image_id < right.image_id;
        });
    for (std::size_t index = 1; index < poses_.size(); ++index) {
        if (poses_[index - 1].image_id == poses_[index].image_id) {
            last_error_ = "Duplicate image ID in optimized camera pose file: " +
                std::to_string(poses_[index].image_id);
            poses_.clear();
            return false;
        }
    }
    return true;
}

bool TrajectoryProcessing::smoothTrajectory(
    const TrajectorySmoothingOptions& options)
{
    smooth_trajectory_.clear();
    last_error_.clear();
    if (poses_.empty()) {
        last_error_ = "Load optimized camera poses before smoothing";
        return false;
    }
    if (options.window_radius < 0 || !std::isfinite(options.gaussian_sigma) ||
        options.gaussian_sigma <= 0.0) {
        last_error_ = "Smoothing radius must be non-negative and sigma positive";
        return false;
    }

    trajectory_height_ = std::isfinite(options.fixed_height)
        ? options.fixed_height : medianHeight(poses_);
    smooth_trajectory_.reserve(poses_.size());
    const int pose_count = static_cast<int>(poses_.size());
    const double sigma_squared =
        options.gaussian_sigma * options.gaussian_sigma;

    for (int index = 0; index < pose_count; ++index) {
        Vec3d weighted_position = Vec3d::Zero();
        double weight_sum = 0.0;
        const int begin = std::max(0, index - options.window_radius);
        const int end = std::min(pose_count - 1, index + options.window_radius);
        for (int neighbor = begin; neighbor <= end; ++neighbor) {
            const double offset = static_cast<double>(neighbor - index);
            const double weight = std::exp(
                -0.5 * offset * offset / sigma_squared);
            weighted_position += weight * poses_[neighbor].position_world;
            weight_sum += weight;
        }

        TrajectoryPoint point;
        point.image_id = poses_[index].image_id;
        point.image_name = poses_[index].image_name;
        point.position_world = weighted_position / weight_sum;
        point.position_world.z() = trajectory_height_;
        smooth_trajectory_.push_back(std::move(point));
    }
    return true;
}

bool TrajectoryProcessing::saveSmoothTrajectory(const std::string& path) const
{
    if (smooth_trajectory_.empty()) return false;
    const std::filesystem::path output_path(path);
    if (output_path.has_parent_path())
        std::filesystem::create_directories(output_path.parent_path());

    std::ofstream output(path);
    if (!output.is_open()) return false;
    output << "# IMAGE_ID X Y Z IMAGE_NAME\n" << std::setprecision(17);
    for (const TrajectoryPoint& point : smooth_trajectory_) {
        output << point.image_id << ' '
               << point.position_world.x() << ' '
               << point.position_world.y() << ' '
               << point.position_world.z() << ' '
               << point.image_name << '\n';
    }
    return output.good();
}
