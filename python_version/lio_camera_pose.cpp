#include <Eigen/Core>
#include <Eigen/Geometry>
#include <nlohmann/json.hpp>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <unordered_map>

namespace fs = std::filesystem;
using json = nlohmann::json;
using Vec3 = Eigen::Vector3d;
using Mat3 = Eigen::Matrix3d;
using Quat = Eigen::Quaterniond;

struct Pose {
    Quat q = Quat::Identity(); // local to world
    Vec3 t = Vec3::Zero();
};
struct TimedPose { double time; Pose pose; };
struct CameraPose { int id; std::string image; double image_time; Pose prior; Pose optimized; };

Pose inverse(const Pose& a) {
    Pose b;
    b.q = a.q.conjugate();
    b.t = -(b.q * a.t);
    return b;
}
Pose compose(const Pose& a, const Pose& b) {
    return {a.q * b.q, a.q * b.t + a.t};
}
std::vector<json> read_jsonl_objects(const fs::path& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Cannot open " + path.string());
    std::vector<json> result;
    std::string line, buffer;
    int braces = 0;
    while (std::getline(in, line)) {
        for (char c : line) {
            if (c == '{') ++braces;
            if (c == '}') --braces;
        }
        buffer += line + "\n";
        if (braces == 0 && !buffer.empty()) {
            result.push_back(json::parse(buffer));
            buffer.clear();
        }
    }
    return result;
}
Pose json_pose(const json& value) {
    const auto& t = value.at("translation");
    const auto& q = value.at("quaternion_xyzw");
    return {Quat(q[3], q[0], q[1], q[2]).normalized(),
            Vec3(t[0], t[1], t[2])};
}
Pose interpolate(const std::vector<TimedPose>& trajectory, double time) {
    auto upper = std::lower_bound(
        trajectory.begin(), trajectory.end(), time,
        [](const TimedPose& p, double t) { return p.time < t; });
    if (upper == trajectory.begin()) return upper->pose;
    if (upper == trajectory.end()) return trajectory.back().pose;
    const auto& a = *(upper - 1);
    const auto& b = *upper;
    const double alpha = (time - a.time) / (b.time - a.time);
    return {a.pose.q.slerp(alpha, b.pose.q).normalized(),
            (1.0 - alpha) * a.pose.t + alpha * b.pose.t};
}
Pose load_lidar_to_camera(const fs::path& path) {
    std::ifstream in(path);
    json root;
    in >> root;
    const auto& a = root.at("T_camera_lidar");
    Mat3 R;
    Vec3 t;
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) R(r,c) = a[r][c];
        t[r] = a[r][3];
    }
    // Input is T_camera_lidar. Camera-to-lidar is its inverse.
    return inverse({Quat(R).normalized(), t});
}
std::vector<TimedPose> load_lio(const fs::path& path) {
    std::vector<TimedPose> poses;
    for (const auto& item : read_jsonl_objects(path))
        poses.push_back({item.at("timestamp"), json_pose(item.at("lio_pose"))});
    std::sort(poses.begin(), poses.end(),
              [](const auto& a, const auto& b) { return a.time < b.time; });
    if (poses.size() < 2) throw std::runtime_error("Need at least two LIO poses");
    return poses;
}
std::vector<CameraPose> initialize_cameras(
    const fs::path& data, double time_offset, int max_images,
    const fs::path& lidar_timestamps_path) {
    const auto lio = load_lio(data / "key_frames.jsonl");
    const Pose T_lidar_camera =
        load_lidar_to_camera(data / "tuned_camera_lidar_extrinsic.json");
    std::ifstream in(data / "timestamps.txt");
    if (!in) throw std::runtime_error("Cannot open timestamps.txt");
    std::unordered_map<std::string, double> lidar_timestamps;
    if (!lidar_timestamps_path.empty()) {
        std::ifstream mapped(lidar_timestamps_path);
        if (!mapped) throw std::runtime_error("Cannot open " + lidar_timestamps_path.string());
        std::string mapped_image;
        double mapped_time;
        while (mapped >> mapped_image >> mapped_time)
            lidar_timestamps[mapped_image] = mapped_time;
        if (lidar_timestamps.empty())
            throw std::runtime_error("No mapped timestamps in " + lidar_timestamps_path.string());
    }
    std::vector<CameraPose> cameras;
    std::string image;
    double image_time;
    while (in >> image >> image_time) {
        if (max_images >= 0 && static_cast<int>(cameras.size()) >= max_images) break;
        if (!fs::exists(data / "undistorted" / image)) continue;
        double lidar_time = image_time + time_offset;
        if (!lidar_timestamps.empty()) {
            const auto mapped = lidar_timestamps.find(image);
            if (mapped == lidar_timestamps.end())
                throw std::runtime_error("No estimated LiDAR timestamp for " + image);
            lidar_time = mapped->second;
        }
        const Pose T_world_lidar = interpolate(lio, lidar_time);
        const Pose T_world_camera = compose(T_world_lidar, T_lidar_camera);
        cameras.push_back({static_cast<int>(cameras.size()), image, image_time,
                           T_world_camera, T_world_camera});
    }
    if (cameras.empty()) throw std::runtime_error("No input images found");
    return cameras;
}
void save_tum(const fs::path& out, const std::vector<CameraPose>& cameras) {
    fs::create_directories(out);
    std::ofstream file(out / "poses_lio_prior_tum.txt");
    file << std::setprecision(16);
    for (const auto& c : cameras)
        file << c.image_time << " " << c.prior.t.transpose() << " "
             << c.prior.q.x() << " " << c.prior.q.y() << " "
             << c.prior.q.z() << " " << c.prior.q.w() << "\n";
}
int main(int argc, char** argv) {
    try {
        fs::path data = "data", out = "output/lio_camera_pose";
        double offset = -0.4;
        int max_images = -1;
        fs::path lidar_timestamps_path;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--data" && i + 1 < argc) data = argv[++i];
            else if (arg == "--output" && i + 1 < argc) out = argv[++i];
            else if (arg == "--time-offset" && i + 1 < argc) offset = std::stod(argv[++i]);
            else if (arg == "--max-images" && i + 1 < argc) max_images = std::stoi(argv[++i]);
            else if (arg == "--lidar-timestamps" && i + 1 < argc) lidar_timestamps_path = argv[++i];
        }
        auto cameras = initialize_cameras(data, offset, max_images, lidar_timestamps_path);
        save_tum(out, cameras);
        const json applied_offset = lidar_timestamps_path.empty() ? json(offset) : json(nullptr);
        json metrics = {
            {"stage", "lio_prior_initialization"},
            {"images", cameras.size()},
            {"time_offset_seconds", applied_offset},
            {"timestamp_source", lidar_timestamps_path.empty() ?
                "t_lidar = t_image + time_offset" : lidar_timestamps_path.string()},
            {"next_stage", "feature tracks, triangulation, local BA, global BA"}
        };
        std::ofstream(out / "metrics.json") << std::setw(2) << metrics << "\n";
        std::cout << metrics.dump(2) << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}
