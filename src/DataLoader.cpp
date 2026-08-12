//
// Created by chuchu on 8/5/26.
//


#include <string>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <unordered_map>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>

#include "DataLoader.h"
#include "General.h"

namespace fs = std::filesystem;
 Status DataLoader::load(){
  //fill the camera map
  std::cout << "Loading intrinsic..." << std::endl;
  load_intrinsic();
  std::cout << "Loading extrinsic..." << std::endl;
  load_extrinsic();
  std::cout << "Loading Lidar..." << std::endl;
  load_lidar();
  std::cout << "Loading camera..." << std::endl;
  load_camera();
  // association
  std::cout << "Associating cameras with lidars..." << std::endl;
  camera_lidar_association();
  std::cout << "Loading depth..." << std::endl;
  update_depth_map_parallel(false);

  return {true, "OK"};
}

void DataLoader::set_camera_map(const std::map<int, Camera>& camera_map){
    camera_map_ = camera_map;
}

void DataLoader::set_lidar_info_map(const std::map<int, LidarPointCloudInfo>& lidar_info_map){
    lidar_info_map_ = lidar_info_map;
}

std::map<int, Camera>& DataLoader::get_camera_map(){
 return camera_map_;
}

std::map<int, LidarPointCloudInfo>& DataLoader::get_lidar_info_map(){
 return lidar_info_map_;
}

void DataLoader::load_intrinsic(){
  std::string intrinsic_path  = config_.camera_intrinsic_path_;
     std::ifstream fin(intrinsic_path);
     if (!fin.is_open()) {
         throw std::runtime_error(
             "Cannot open intrinsic file: " + intrinsic_path);
     }

     std::string line;
     intrinsic_ = CameraIntrinsic();
     for (int row = 0; row < 3; ++row) {

         if (!std::getline(fin, line)) {
             throw std::runtime_error(
                 "Invalid intrinsic file.");
         }

         // Replace ',' with ' '
         for (char& c : line) {
             if (c == ',')
                 c = ' ';
         }

         std::stringstream ss(line);

         for (int col = 0; col < 3; ++col) {
             if (!(ss >> intrinsic_.K(row, col))) {
                 throw std::runtime_error(
                     "Failed to parse intrinsic matrix.");
             }
         }
     }

     std::cout << "Camera intrinsic loaded:\n"
               << intrinsic_.K << std::endl;
}

void DataLoader::load_extrinsic(){
  std::string extrinsic_path = config_.camera_extrinsic_path_;
     std::ifstream input_file(extrinsic_path);

     if (!input_file.is_open()) {
         throw std::runtime_error(
             "Failed to open extrinsic file: " + extrinsic_path
         );
     }

     nlohmann::json root;

     try {
         input_file >> root;
     } catch (const nlohmann::json::parse_error& error) {
         throw std::runtime_error(
             "Failed to parse extrinsic JSON: " +
             std::string(error.what())
         );
     }

     if (!root.contains("T_camera_lidar")) {
         throw std::runtime_error(
             "Missing T_camera_lidar in: " + extrinsic_path
         );
     }

     Mat4d T_CL_matrix =
         load_matrix4d(root.at("T_camera_lidar"));

     const Mat3d R_CL =
         T_CL_matrix.block<3, 3>(0, 0);

     const Vec3d t_CL =
         T_CL_matrix.block<3, 1>(0, 3);

     extrinsic_.T_camera_lidar =
         SE3d(R_CL, t_CL);

     // Compute the inverse instead of trusting a second stored matrix.
     extrinsic_.T_lidar_camera =
         extrinsic_.T_camera_lidar.inverse();

    std::cout << "Camera extrinsic loaded:\n"
              << "T_camera_lidar:\n" << extrinsic_.T_camera_lidar.matrix() << "\n"
              << "T_lidar_camera:\n" << extrinsic_.T_lidar_camera.matrix() << std::endl;
}

void DataLoader::load_depth(){
     int loaded_count = 0;
     int missing_count = 0;
     const fs::path depth_dir = config_.projective_z_buffer_dir_;

     if (!fs::exists(depth_dir)) {
         throw std::runtime_error(
             "Depth directory does not exist: " +
             depth_dir.string()
         );
     }

     if (!fs::is_directory(depth_dir)) {
         throw std::runtime_error(
             "Depth path is not a directory: " +
             depth_dir.string()
         );
     }
    for (auto& [camera_id, camera] : camera_map_) {
        const fs::path image_path(camera.camera_path_);

        // Original:
        // /image_dir/IMG_2129.jpg
        //
        // Depth:
        // /depth_dir/IMG_2129.png
        const fs::path depth_path =
            depth_dir /
            (image_path.stem().string() + ".png");

        if (!fs::exists(depth_path)) {
            std::cerr
                << "[load_depth] Missing depth image: "
                << depth_path << '\n';

            camera.depth_map_.release();
            ++missing_count;
            continue;
        }

        cv::Mat depth_map = cv::imread(
            depth_path.string(),
            cv::IMREAD_UNCHANGED
        );

        if (depth_map.empty()) {
            std::cerr
                << "[load_depth] Failed to read: "
                << depth_path << '\n';

            camera.depth_map_.release();
            ++missing_count;
            continue;
        }

        if (depth_map.channels() != 1) {
            std::cerr
                << "[load_depth] Depth image must be single-channel: "
                << depth_path
                << ", channels = "
                << depth_map.channels()
                << '\n';

            camera.depth_map_.release();
            ++missing_count;
            continue;
        }

        // Optional: verify depth resolution matches original image.
        cv::Mat image = cv::imread(
            camera.camera_path_,
            cv::IMREAD_COLOR
        );

        if (!image.empty() &&
            image.size() != depth_map.size()) {
            std::cerr
                << "[load_depth] Size mismatch for camera "
                << camera_id
                << ". Image: "
                << image.cols << " x " << image.rows
                << ", depth: "
                << depth_map.cols << " x " << depth_map.rows
                << '\n';

            camera.depth_map_.release();
            ++missing_count;
            continue;
        }

        // cv::Mat uses reference-counted memory.
        // This modifies the Camera object inside camera_dict_.
        camera.depth_map_ = depth_map;

        ++loaded_count;
    }

    std::cout
        << "[load_depth] Loaded "
        << loaded_count
        << " depth maps, missing/invalid "
        << missing_count
        << std::endl;
  return;
}

void DataLoader::load_camera(){
  std::string camera_img_dir = config_.input_img_dir_;
  std::string camera_timestamp_path = config_.camera_timestamp_path_;
  std::ifstream file(camera_timestamp_path);
  if (!file.is_open()) {
        throw std::runtime_error(
            "Failed to open camera timestamp file: " +
            config_.camera_timestamp_path_);
    }
    camera_map_.clear();

    std::string image_name;
    double timestamp;
    int camera_id = 0;
    bool image_read = false;
    int img_width = 0;
    int img_height = 0;

    while (file >> image_name >> timestamp) {
        if (config_.max_images_ >= 0 &&
            camera_id >= config_.max_images_) {
            break;
        }

        Camera camera;
        camera.camera_id_ = camera_id;
        camera.camera_name_ = image_name;
        camera.time_stamp_ = timestamp;

       // Full image path
        camera.camera_path_ = camera_img_dir + "/" + image_name;
        if(!fs::exists(camera.camera_path_)){
            //std::cerr << "[load_camera] Image not found, skipping: " << camera.camera_path_ << '\n';
            continue;
        }
        // Initial pose will be filled later
        camera.initial_T_cw_ = SE3d();
        camera.optimized_T_cw_ = SE3d();
        camera.optimized_ = false;
        if(!image_read){
            // only load once, since image are assumed to have the same dimensions
            image_read = true;
            cv::Mat image = cv::imread(camera.camera_path_, cv::IMREAD_COLOR);
            img_width  = image.cols;
            img_height = image.rows;
        }
        camera.img_width_ = img_width;
        camera.img_height_ = img_height;
        // init the depth map
        camera.depth_map_ = cv::Mat(img_height, img_width, CV_32FC1, cv::Scalar(0));
        camera_map_[camera_id] = camera;
        ++camera_id;
    }
    file.close();
    std::cout << "Loaded " << camera_map_.size() << " camera images\n";
  return;
}

void DataLoader::load_lidar(){
    // load keyframes from json file
    load_keyframe_jsonl();

    // load the lidar point for each key frames [temporary disable for performance/memeroy concern]
    // for(auto& [lidar_id, info] : lidar_info_map_){
    //     lidar_info_map_[lidar_id].points_.clear();
    //     read_ply_xyz(lidar_info_map_[lidar_id].lidar_path_, lidar_info_map_[lidar_id].points_);
    // }
    return;
}

Mat4d DataLoader::load_matrix4d(
    const nlohmann::json& json_matrix){
     if (!json_matrix.is_array() || json_matrix.size() != 4) {
         throw std::runtime_error(
             "Expected a 4x4 matrix in the extrinsic JSON."
         );
     }

     Mat4d matrix;

     for (int row = 0; row < 4; ++row) {
         if (!json_matrix[row].is_array() ||
             json_matrix[row].size() != 4) {
             throw std::runtime_error(
                 "Expected a 4x4 matrix in the extrinsic JSON."
             );
             }

         for (int col = 0; col < 4; ++col) {
             matrix(row, col) =
                 json_matrix[row][col].get<double>();
         }
     }
     return matrix;
}

void DataLoader::read_ply_xyz(const std::string& filename , std::vector<Vec3d>& points){

    std::ifstream file(filename);

    if(!file.is_open()){
        throw std::runtime_error("Cannot open ply file: " + filename);
    }

    std::string line;
    size_t vertex_count = 0;
    while(std::getline(file, line)){
        std::stringstream ss(line);
        std::string key;
        ss >> key;
        if(key == "element"){
            std::string type;
            ss >> type;

            if(type == "vertex"){
                ss >> vertex_count;
            }
        }
        if(line == "end_header"){
            break;
        }
    }

    points.clear();
    points.reserve(vertex_count);
    for(size_t i=0; i<vertex_count; i++){
        double x,y,z;
        file >> x >> y >> z;
        points.emplace_back(x,y,z);
    }
    file.close();
}

void DataLoader::load_keyframe_jsonl(){
    // Load the keyframe JSONL file specified in the configuration
    std::ifstream file(config_.lidar_kf_path_);
    if(!file.is_open()){
        throw std::runtime_error("Cannot open keyframe JSONL file: " + config_.lidar_kf_path_);
    }
    std::string line;
    std::string json_buffer;
    int brace_count = 0;
    while(std::getline(file, line)){
        if (line.empty())
        continue;
        // Count opening/closing braces
        for (char c : line) {
            if (c == '{')
                ++brace_count;
            else if (c == '}')
                --brace_count;
        }
        json_buffer += line;
        json_buffer += '\n';
        if (brace_count == 0 && !json_buffer.empty()) {
        // Process the accumulated JSON buffer as a JSON object
            nlohmann::json json_line = nlohmann::json::parse(json_buffer);
            // Handle the JSON object as needed
            LidarPointCloudInfo info;

            info.lidar_id_ = json_line["key_frame_id"].get<int>();
            info.time_stamp_ = json_line["timestamp"].get<double>();
            info.lidar_path_ = config_.lidar_ply_dirs_+ json_line["saved_frame_path"].get<std::string>().erase(0,1);
            // get the LIO se3d
            auto t = json_line["lio_pose"]["translation"];
            Eigen::Vector3d translation(
                t[0].get<double>(),
                t[1].get<double>(),
                t[2].get<double>());
            auto q = json_line["lio_pose"]["quaternion_xyzw"];
            Eigen::Quaterniond rotation(
                q[3].get<double>(), // w
                q[0].get<double>(), // x
                q[1].get<double>(), // y
                q[2].get<double>()  // z
            );
            info.initial_T_wl_ = SE3d(rotation, translation);
            lidar_info_map_[info.lidar_id_] = info;
            json_buffer.clear();
        }
    }

    file.close();
    std::cout<< "Finished loading keyframe JSONL. Total keyframes: " << lidar_info_map_.size() << std::endl;
}

void DataLoader::camera_lidar_association(){
    //Build (timestamp, lidar_id) table
    std::vector<std::pair<double, int>> lidar_time_table;
    lidar_time_table.reserve(lidar_info_map_.size());

    for (const auto& [lidar_id, lidar] : lidar_info_map_){
        lidar_time_table.emplace_back(lidar.time_stamp_, lidar_id);
    }
    // sort the lidar_time_table by timestamp
    std::sort(lidar_time_table.begin(),lidar_time_table.end(),[](const auto& a, const auto& b){
            return a.first < b.first;
        });

    // associate each camera with the closest lidar timestamp
    for (auto& [camera_id, camera] : camera_map_){
        // offset the camera timestamp by 0.4 seconds to account for synchronization delay [TODO: adding in config]
        double t = camera.time_stamp_ - 0.4;
        auto it = std::lower_bound(lidar_time_table.begin(),lidar_time_table.end(),t,
            [](const auto& lhs, double value){
                return lhs.first < value;
            });
        // Camera falls before the first or after the last LiDAR keyframe.
        // Only use the boundary pose if it is within the allowed time gap.
        if(it == lidar_time_table.begin()){
            double gap = lidar_time_table.front().first - t;
            if (gap > config_.max_camera_lidar_time_gap_) {
                std::cerr << "[camera_lidar_association] camera_id=" << camera_id
                          << " adjusted_ts=" << t
                          << " is " << gap << " s before LiDAR start — skipping (no valid pose).\n";
                continue; // matched_lidar_id_ stays -1
            }
            const auto& lidar = lidar_info_map_.at(it->second);
            camera.matched_lidar_id_ = it->second;
            camera.initial_T_cw_ = extrinsic_.T_camera_lidar * lidar.initial_T_wl_.inverse();
            continue;
        }
        if (it == lidar_time_table.end()){
            auto last = std::prev(it);
            double gap = t - last->first;
            if (gap > config_.max_camera_lidar_time_gap_) {
                std::cerr << "[camera_lidar_association] camera_id=" << camera_id
                          << " adjusted_ts=" << t
                          << " is " << gap << " s after LiDAR end — skipping (no valid pose).\n";
                continue;
            }
            const auto& lidar = lidar_info_map_.at(last->second);
            camera.matched_lidar_id_ = last->second;
            camera.initial_T_cw_ = extrinsic_.T_camera_lidar * lidar.initial_T_wl_.inverse();
            continue;
        }

        // Interpolate between two LiDAR poses
        auto next = it;
        auto prev = std::prev(it);

        const auto& lidar0 = lidar_info_map_.at(prev->second);
        const auto& lidar1 =lidar_info_map_.at(next->second);
        camera.matched_lidar_id_ = prev->second;

        double t0 = prev->first;
        double t1 = next->first;

        double alpha =(t - t0) / (t1 - t0);
        alpha = std::clamp(alpha, 0.0, 1.0);

        // Translation interpolation
        Eigen::Vector3d translation = (1.0 - alpha) * lidar0.initial_T_wl_.translation() +
            alpha * lidar1.initial_T_wl_.translation();

        // Rotation interpolation (SLERP)
        Eigen::Quaterniond q0 = lidar0.initial_T_wl_.unit_quaternion();
        Eigen::Quaterniond q1 = lidar1.initial_T_wl_.unit_quaternion();
        Eigen::Quaterniond q =q0.slerp(alpha, q1);

        // Interpolated LiDAR pose
        SE3d T_wl_interp(q, translation);
        // Convert LiDAR pose -> Camera pose
        camera.initial_T_cw_ = extrinsic_.T_camera_lidar * T_wl_interp.inverse();
        std::cout<<std::setprecision(15)<<"matched_lidar_id=" << camera.matched_lidar_id_ << " camera_ts=" << camera.time_stamp_-0.4<< " lidar_ts=" << lidar_info_map_.at(camera.matched_lidar_id_).time_stamp_ << std::endl;
    }
}


void DataLoader::update_depth_map(const bool optimized){
    // use plus minus 30 lidar frame, 
    //project the lidar points to each image frame, generate depth map
    // points under lidar local frame 
    auto project_lidar_to_camera = [](const std::vector<Vec3d>& points, const SE3d& T_cw,  const SE3d& T_wl, 
            const Mat3d& K, const int image_width, const int image_height, cv::Mat& depth_map) {
    
        const double fx = K(0, 0);
        const double fy = K(1, 1);
        const double cx = K(0, 2);
        const double cy = K(1, 2);

        for (const auto& p_l : points){
            // lidar ->World -> Camera
            Vec3d p_c = T_cw * (T_wl * p_l);
            // Behind camera
            if (p_c.z() <= 0.0)
                continue;

            const double inv_z = 1.0 / p_c.z();
            const int u = static_cast<int>(std::round(fx * p_c.x() * inv_z + cx));
            const int v = static_cast<int>(std::round(fy * p_c.y() * inv_z + cy));

            if (u < 0 || u >= image_width ||v < 0 || v >= image_height)
                continue;

            const float depth = static_cast<float>(p_c.z());

            // Keep nearest point (z-buffer)
            float& current = depth_map.at<float>(v, u);
            if (current == 0.0f || depth < current)
                current = depth;
        }
    };

    const int window_size = 30;
    std::unordered_map<int, std::vector<Vec3d>> pointcloud_map;
    for(auto& [camera_id, camera] : camera_map_){
        int associated_lidar_id = camera.matched_lidar_id_;
        if(associated_lidar_id < 0) continue;
        int lower_bound = std::max(0, associated_lidar_id - window_size);
        int upper_bound = associated_lidar_id + window_size;
        //std::cout<<"Processing camera ID: " << camera_id << " with associated lidar ID: " << associated_lidar_id << std::endl;
        for(int lidar_id = lower_bound; lidar_id <= upper_bound; ++lidar_id){
            if(lidar_info_map_.find(lidar_id) == lidar_info_map_.end()) continue;
            const auto& lidar = lidar_info_map_.at(lidar_id);
            if(pointcloud_map.find(lidar_id) == pointcloud_map.end()){
                read_ply_xyz(lidar_info_map_.at(lidar_id).lidar_path_,pointcloud_map[lidar_id]);
            }
            auto& curr_cloud = pointcloud_map[lidar_id];
            if(camera.depth_map_.empty()){
                camera.depth_map_ = cv::Mat(camera.img_height_, camera.img_width_, CV_32FC1, cv::Scalar(0));
            }
            if(camera.optimized_){
                project_lidar_to_camera(curr_cloud, camera.optimized_T_cw_, lidar.initial_T_wl_, intrinsic_.K, camera.img_width_, camera.img_height_, camera.depth_map_);
            }else{
                project_lidar_to_camera(curr_cloud, camera.initial_T_cw_, lidar.initial_T_wl_, intrinsic_.K, camera.img_width_, camera.img_height_, camera.depth_map_);
            }
        }
    }
}


void DataLoader::update_depth_map_parallel(const bool optimized){
    const int window_size = 50;
    // LiDAR -> world -> Camera projection
    auto project_lidar_to_camera = [](const std::vector<Vec3d>& points, const SE3d& T_cw, const SE3d& T_wl,
            const Mat3d& K, const int image_width, const int image_height, cv::Mat& depth_map) {

        const double fx = K(0, 0);
        const double fy = K(1, 1);
        const double cx = K(0, 2);
        const double cy = K(1, 2);

        for (const auto& p_l : points){
            // lidar ->World -> Camera
            Vec3d p_c = T_cw * (T_wl * p_l);
            // Behind camera
            if (p_c.z() <= 0.0)
                continue;

            const double inv_z = 1.0 / p_c.z();
            const int u = static_cast<int>(std::round(fx * p_c.x() * inv_z + cx));
            const int v = static_cast<int>(std::round(fy * p_c.y() * inv_z + cy));

            if (u < 0 || u >= image_width ||v < 0 || v >= image_height)
                continue;

            const float depth = static_cast<float>(p_c.z());

            // Keep nearest point (z-buffer)
            float& current = depth_map.at<float>(v, u);
            if (current == 0.0f || depth < current)
                current = depth;
        }
    };

    // Collect all LiDAR IDs actually needed across all cameras' windows
    std::vector<int> required_lidar_ids;
    for(auto& [camera_id, camera] : camera_map_){
        int associated_lidar_id = camera.matched_lidar_id_;
        if(associated_lidar_id < 0) continue;
        int lower_bound = std::max(0, associated_lidar_id - window_size);
        int upper_bound = associated_lidar_id + window_size;
        for(int lidar_id = lower_bound; lidar_id <= upper_bound; ++lidar_id){
            if(lidar_info_map_.find(lidar_id) == lidar_info_map_.end()) continue;
            required_lidar_ids.push_back(lidar_id);
        }
    }
    std::sort(required_lidar_ids.begin(), required_lidar_ids.end());
    required_lidar_ids.erase(std::unique(required_lidar_ids.begin(), required_lidar_ids.end()), required_lidar_ids.end());

    // Load LiDAR point clouds in parallel
    std::unordered_map<int, std::vector<Vec3d>> pointcloud_map;
    // Pre-allocate entries so parallel threads don't touch the unordered_map's structure concurrently
    for(const int lidar_id : required_lidar_ids){
        pointcloud_map.emplace(lidar_id, std::vector<Vec3d>{});
    }
    tbb::parallel_for(tbb::blocked_range<size_t>(0, required_lidar_ids.size()), [&](const tbb::blocked_range<size_t>& range){
        for(size_t i = range.begin(); i != range.end(); ++i){
            const int lidar_id = required_lidar_ids[i];
            auto& cloud = pointcloud_map.at(lidar_id);
            const auto& lidar = lidar_info_map_.at(lidar_id);
            read_ply_xyz(lidar.lidar_path_, cloud);
        }
    });

    // unordered_map cannot be indexed directly, so collect camera IDs for parallel processing
    std::vector<int> camera_ids;
    camera_ids.reserve(camera_map_.size());
    for(const auto& [camera_id, camera] : camera_map_){
        camera_ids.push_back(camera_id);
    }

    // Generate depth maps in parallel
    tbb::parallel_for(tbb::blocked_range<size_t>(0, camera_ids.size()), [&](const tbb::blocked_range<size_t>& range){
        for(size_t i = range.begin(); i != range.end(); ++i){
            const int camera_id = camera_ids[i];
            auto& camera = camera_map_.at(camera_id);
            const int associated_lidar_id = camera.matched_lidar_id_;
            if(associated_lidar_id < 0) continue;

            if(camera.depth_map_.empty()){
                camera.depth_map_ = cv::Mat(camera.img_height_, camera.img_width_, CV_32FC1, cv::Scalar(0));
            }

            const SE3d& T_cw = optimized ? camera.optimized_T_cw_ : camera.initial_T_cw_;

            const int lower_bound = std::max(0, associated_lidar_id - window_size);
            const int upper_bound = associated_lidar_id + window_size;
            for(int lidar_id = lower_bound; lidar_id <= upper_bound; ++lidar_id){
                auto it = pointcloud_map.find(lidar_id);
                if(it == pointcloud_map.end()) continue;

                const auto& cloud = it->second;
                const auto& lidar = lidar_info_map_.at(lidar_id);
                project_lidar_to_camera(cloud, T_cw, lidar.initial_T_wl_, intrinsic_.K, camera.img_width_, camera.img_height_, camera.depth_map_);
            }
        }
    });

    std::cout<<"Depth map generation completed." << std::endl;
}