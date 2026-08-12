//
// Created by chuchu on 8/5/26.
//

#include "General.h"
#include "SIFT.h"

#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/eigen.hpp>

#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <filesystem>
#include <functional>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <mutex>
#include <set>
#include <sstream>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace {

struct DepthAccumulator
{
    double sum = 0.0;
    int count = 0;

    void add(double depth)
    {
        if (!std::isfinite(depth) || depth <= 0.0)
        {
            return;
        }

        sum += depth;
        ++count;
    }

    double mean() const
    {
        if (count <= 0)
        {
            return 0.0;
        }
        return sum / static_cast<double>(count);
    }
};

FeatureNode resolve_feature_node(
    int camera_id,
    int keypoint_idx,
    const Camera& camera,
    double merge_radius_px,
    std::unordered_map<int, std::vector<FeatureNode>>& camera_feature_nodes)
{
    const cv::Point2f& query_point = camera.keypoints_[keypoint_idx].pt;
    const double max_dist_sq = merge_radius_px * merge_radius_px;

    std::vector<FeatureNode>& nodes = camera_feature_nodes[camera_id];
    for (const FeatureNode& node : nodes)
    {
        const int representative_idx = node.second;
        if (representative_idx < 0 ||
            representative_idx >= static_cast<int>(camera.keypoints_.size()))
        {
            continue;
        }

        const cv::Point2f& representative_point =
            camera.keypoints_[representative_idx].pt;
        const double dx = static_cast<double>(query_point.x - representative_point.x);
        const double dy = static_cast<double>(query_point.y - representative_point.y);
        if (dx * dx + dy * dy <= max_dist_sq)
        {
            return node;
        }
    }

    const FeatureNode node{camera_id, keypoint_idx};
    nodes.push_back(node);
    return node;
}

}


// ============================================================
// Extract SIFT
// ============================================================

Status SIFT::extract_sift(
    Camera& camera,
    Config& config)
{
    //----------------------------------------------------------
    // Load image
    //----------------------------------------------------------
    cv::Mat image = cv::imread(
        camera.camera_path_,
        cv::IMREAD_GRAYSCALE);

    if (image.empty())
    {
        return {
            false,
            "Failed to load image: " + camera.camera_path_
        };
    }

    //----------------------------------------------------------
    // Create SIFT detector
    //----------------------------------------------------------
    cv::Ptr<cv::SIFT> sift = cv::SIFT::create(
        config.sift_nfeatures_,
        config.sift_n_octave_layers_,
        config.sift_contrast_threshold_,
        config.sift_edge_threshold_,
        config.sift_sigma_);

    //----------------------------------------------------------
    // Clear previous result
    //----------------------------------------------------------
    camera.keypoints_.clear();
    camera.descriptors_.release();

    //----------------------------------------------------------
    // Extract
    //----------------------------------------------------------
    sift->detectAndCompute(
        image,
        cv::noArray(),
        camera.keypoints_,
        camera.descriptors_);

    //----------------------------------------------------------
    // Drop keypoints with no valid LiDAR depth in this frame -
    // they can never be triangulated/tracked reliably anyway.
    // //----------------------------------------------------------
    // if (!camera.depth_map_.empty())
    // {
    //     std::vector<cv::KeyPoint> valid_keypoints;
    //     cv::Mat valid_descriptors;
    //     valid_keypoints.reserve(camera.keypoints_.size());

    //     for (size_t i = 0; i < camera.keypoints_.size(); ++i)
    //     {
    //         int u = cvRound(camera.keypoints_[i].pt.x);
    //         int v = cvRound(camera.keypoints_[i].pt.y);

    //         if (u < 0 || u >= camera.depth_map_.cols ||
    //             v < 0 || v >= camera.depth_map_.rows)
    //         {
    //             continue;
    //         }

    //         if (camera.depth_map_.at<float>(v, u) <= 0.0f)
    //         {
    //             continue;
    //         }

    //         valid_keypoints.push_back(camera.keypoints_[i]);
    //         valid_descriptors.push_back(camera.descriptors_.row(static_cast<int>(i)));
    //     }

    //     camera.keypoints_ = std::move(valid_keypoints);
    //     camera.descriptors_ = valid_descriptors;
    // }

    if (camera.keypoints_.empty())
    {
        return {
            true,
            "No SIFT features detected in image: "
                + camera.camera_name_
        };
    }

    if (camera.descriptors_.empty())
    {
        return {
            false,
            "Failed to compute SIFT descriptors for image: "
                + camera.camera_name_
        };
    }

    return {
        true,
        "Extracted "
            + std::to_string(camera.keypoints_.size())
            + " SIFT features from "
            + camera.camera_name_
    };
}


// ============================================================
// KNN matching
// ============================================================

void SIFT::knn_matching(
    const cv::Mat& desc_1,
    const cv::Mat& desc_2,
    std::vector<std::vector<cv::DMatch>>& knn_matches)
{
    knn_matches.clear();

    //----------------------------------------------------------
    // SIFT uses floating-point descriptors, so use L2 distance
    //----------------------------------------------------------
    cv::BFMatcher matcher(cv::NORM_L2);

    matcher.knnMatch(
        desc_1,
        desc_2,
        knn_matches,
        2);
}


// ============================================================
// Lowe ratio test
// ============================================================

void SIFT::lowe_ratio_test(
    const std::vector<std::vector<cv::DMatch>>& knn_matches,
    std::vector<cv::DMatch>& ratio_matches,
    float ratio_threshold,
    float max_match_distance)
{
    ratio_matches.clear();

    for (const auto& matches : knn_matches)
    {
        //------------------------------------------------------
        // Need first and second nearest neighbors
        //------------------------------------------------------
        if (matches.size() < 2)
        {
            continue;
        }

        const cv::DMatch& best = matches[0];
        const cv::DMatch& second = matches[1];

        //------------------------------------------------------
        // Absolute distance cap, in addition to the ratio test.
        //------------------------------------------------------
        if (max_match_distance >= 0.0f &&
            best.distance > max_match_distance)
        {
            continue;
        }

        //------------------------------------------------------
        // Lowe ratio test
        //------------------------------------------------------
        if (best.distance <
            ratio_threshold * second.distance)
        {
            ratio_matches.push_back(best);
        }
    }
}


// ============================================================
// Match -> pixel coordinates
// ============================================================

void SIFT::match_to_pixels(
    const std::vector<cv::DMatch>& ratio_matches,
    const Camera& camera_1,
    const Camera& camera_2,
    std::vector<cv::Point2f>& points_1,
    std::vector<cv::Point2f>& points_2)
{
    points_1.clear();
    points_2.clear();

    points_1.reserve(ratio_matches.size());
    points_2.reserve(ratio_matches.size());

    for (const auto& match : ratio_matches)
    {
        //------------------------------------------------------
        // queryIdx -> camera 1 keypoint
        //------------------------------------------------------
        points_1.push_back(
            camera_1.keypoints_[match.queryIdx].pt);

        //------------------------------------------------------
        // trainIdx -> camera 2 keypoint
        //------------------------------------------------------
        points_2.push_back(
            camera_2.keypoints_[match.trainIdx].pt);
    }
}


// ============================================================
// Geometric verification
// ============================================================

int SIFT::geometric_verification(
    const std::vector<cv::Point2f>& points_1,
    const std::vector<cv::Point2f>& points_2,
    double ransac_threshold,
    std::vector<uchar>& inlier_mask)
{
    inlier_mask.clear();

    //----------------------------------------------------------
    // Fundamental matrix requires at least 8 points for the
    // standard 8-point formulation
    //----------------------------------------------------------
    if (points_1.size() < 8 ||
        points_2.size() < 8)
    {
        return 0;
    }

    //----------------------------------------------------------
    // Estimate F and reject geometrically inconsistent matches
    //----------------------------------------------------------
    cv::Mat F = cv::findFundamentalMat(
        points_1,
        points_2,
        cv::FM_RANSAC,
        ransac_threshold,
        0.99,
        inlier_mask);

    if (F.empty())
    {
        return 0;
    }

    //----------------------------------------------------------
    // Count RANSAC inliers
    //----------------------------------------------------------
    int num_inliers = 0;

    for (uchar value : inlier_mask)
    {
        if (value)
        {
            ++num_inliers;
        }
    }

    return num_inliers;
}


// ============================================================
// Union matches
// ============================================================

void SIFT::union_matches(
    int camera_id_1,
    const Camera& camera_1,
    int camera_id_2,
    const Camera& camera_2,
    const std::vector<cv::DMatch>& ratio_matches,
    const std::vector<uchar>& inlier_mask,
    std::unordered_map<FeatureNode, FeatureNode, FeatureNodeHash>& parent,
    std::unordered_map<FeatureNode, int, FeatureNodeHash>& rank,
    std::unordered_map<int, std::vector<FeatureNode>>& camera_feature_nodes,
    double feature_track_merge_radius_px)
{
    //----------------------------------------------------------
    // Find root
    //----------------------------------------------------------
    std::function<FeatureNode(const FeatureNode&)> find_root;

    find_root =
        [&](const FeatureNode& node) -> FeatureNode
    {
        //------------------------------------------------------
        // First time seeing this feature
        //------------------------------------------------------
        if (parent.find(node) == parent.end())
        {
            parent[node] = node;
            rank.emplace(node, 0);
        }

        //------------------------------------------------------
        // Path compression
        //------------------------------------------------------
        if (parent[node] != node)
        {
            parent[node] = find_root(parent[node]);
        }

        return parent[node];
    };


    //----------------------------------------------------------
    // Union two feature nodes
    //----------------------------------------------------------
    auto unite =
        [&](const FeatureNode& a,
            const FeatureNode& b)
    {
        FeatureNode root_a = find_root(a);
        FeatureNode root_b = find_root(b);

        if (root_a == root_b)
        {
            return;
        }

        //------------------------------------------------------
        // Union by rank
        //------------------------------------------------------
        if (rank[root_a] < rank[root_b])
        {
            std::swap(root_a, root_b);
        }

        parent[root_b] = root_a;

        if (rank[root_a] == rank[root_b])
        {
            ++rank[root_a];
        }
    };


    //----------------------------------------------------------
    // Add only RANSAC inlier matches
    //----------------------------------------------------------
    const size_t count =
        std::min(
            ratio_matches.size(),
            inlier_mask.size());

    for (size_t i = 0; i < count; ++i)
    {
        if (!inlier_mask[i])
        {
            continue;
        }

        const cv::DMatch& match =
            ratio_matches[i];

        //------------------------------------------------------
        // A feature is identified by:
        //
        // (camera id, keypoint index)
        //------------------------------------------------------
        FeatureNode node_1 = resolve_feature_node(
            camera_id_1,
            match.queryIdx,
            camera_1,
            feature_track_merge_radius_px,
            camera_feature_nodes);

        FeatureNode node_2 = resolve_feature_node(
            camera_id_2,
            match.trainIdx,
            camera_2,
            feature_track_merge_radius_px,
            camera_feature_nodes);

        unite(
            node_1,
            node_2);
    }
}


Status SIFT::export_landmark_track_images(
    const std::map<int, Landmark>& landmark_map,
    const std::map<int, Camera>& camera_map,
    const std::string& output_dir,
    int crop_radius)
{
    if (landmark_map.empty())
    {
        return {true, "No landmark tracks to visualize."};
    }

    fs::create_directories(output_dir);
    std::map<int, cv::Mat> image_cache;
    int written = 0;

    for (const auto& [landmark_id, landmark] : landmark_map)
    {
        std::vector<cv::Mat> crops;
        crops.reserve(landmark.observations_.size());

        for (const Observation& observation : landmark.observations_)
        {
            auto camera_it = camera_map.find(observation.camera_id_);
            if (camera_it == camera_map.end())
            {
                continue;
            }

            const Camera& camera = camera_it->second;
            cv::Mat& image = image_cache[observation.camera_id_];
            if (image.empty())
            {
                image = cv::imread(camera.camera_path_, cv::IMREAD_COLOR);
            }
            if (image.empty())
            {
                continue;
            }

            const cv::Point center(
                cvRound(observation.pixel_.x()),
                cvRound(observation.pixel_.y()));

            const int crop_size = 2 * crop_radius + 1;
            const int x0 = center.x - crop_radius;
            const int y0 = center.y - crop_radius;
            const int x1 = center.x + crop_radius;
            const int y1 = center.y + crop_radius;

            const int src_x0 = std::max(0, x0);
            const int src_y0 = std::max(0, y0);
            const int src_x1 = std::min(image.cols - 1, x1);
            const int src_y1 = std::min(image.rows - 1, y1);

            if (src_x0 > src_x1 || src_y0 > src_y1)
            {
                continue;
            }

            cv::Mat crop(
                crop_size,
                crop_size,
                image.type(),
                cv::Scalar::all(0));

            const cv::Rect src_rect(
                src_x0,
                src_y0,
                src_x1 - src_x0 + 1,
                src_y1 - src_y0 + 1);
            const cv::Rect dst_rect(
                src_x0 - x0,
                src_y0 - y0,
                src_rect.width,
                src_rect.height);

            image(src_rect).copyTo(crop(dst_rect));

            const cv::Point local_center(center.x - x0, center.y - y0);
            cv::circle(crop, local_center, 4, cv::Scalar(0, 255, 255), 1, cv::LINE_AA);

            std::ostringstream label;
            label << "cam " << observation.camera_id_
                  << " kp " << observation.keypoint_idx_;
            cv::putText(
                crop,
                label.str(),
                cv::Point(4, crop.rows - 6),
                cv::FONT_HERSHEY_SIMPLEX,
                0.35,
                cv::Scalar(0, 255, 0),
                1,
                cv::LINE_AA);

            crops.push_back(crop);
        }

        if (crops.empty())
        {
            continue;
        }

        cv::Mat strip;
        cv::hconcat(crops, strip);
        cv::putText(
            strip,
            "landmark " + std::to_string(landmark_id),
            cv::Point(6, 16),
            cv::FONT_HERSHEY_SIMPLEX,
            0.5,
            cv::Scalar(0, 255, 255),
            1,
            cv::LINE_AA);

        const fs::path file_path =
            fs::path(output_dir) /
            ("tracker_" + std::to_string(landmark_id) + ".png");

        if (cv::imwrite(file_path.string(), strip))
        {
            ++written;
        }
    }

    return {
        true,
        "Wrote " + std::to_string(written)
            + " landmark tracker images to " + output_dir
    };
}


// ============================================================
// Extract Landmark map
// ============================================================

void SIFT::extract_landmark_map(
    const std::unordered_map<FeatureNode, FeatureNode, FeatureNodeHash>& parent,
    const std::map<int, Camera>& camera_map,
    std::map<int, Landmark>& landmark_map)
{
    landmark_map.clear();

    //----------------------------------------------------------
    // Nothing matched
    //----------------------------------------------------------
    if (parent.empty())
    {
        return;
    }


    //----------------------------------------------------------
    // Helper for finding the final root.
    //
    // parent is const here, so we don't do path compression.
    //----------------------------------------------------------
    auto find_root =
        [&parent](FeatureNode node)
    {
        auto it = parent.find(node);

        while (it != parent.end() &&
               it->second != node)
        {
            node = it->second;
            it = parent.find(node);
        }

        return node;
    };


    //----------------------------------------------------------
    // Group features according to Union-Find root
    //----------------------------------------------------------
    std::unordered_map<
        FeatureNode,
        std::vector<FeatureNode>,
        FeatureNodeHash
    > tracks;

    for (const auto& entry : parent)
    {
        const FeatureNode& node =
            entry.first;

        FeatureNode root =
            find_root(node);

        tracks[root].push_back(node);
    }


    //----------------------------------------------------------
    // Convert tracks to Landmark objects
    //----------------------------------------------------------
    int landmark_id = 0;

    for (const auto& track_entry : tracks)
    {
        const std::vector<FeatureNode>& track =
            track_entry.second;

        //------------------------------------------------------
        // One feature alone is not a multi-view landmark
        //------------------------------------------------------
        if (track.size() < 2)
        {
            continue;
        }

        Landmark landmark;
        landmark.landmark_id_ = landmark_id;

        //------------------------------------------------------
        // One physical point should occur at most once
        // in each camera.
        //------------------------------------------------------
        std::set<int> used_cameras;

        for (const FeatureNode& node : track)
        {
            const int camera_id =
                node.first;

            const int keypoint_index =
                node.second;


            //--------------------------------------------------
            // Avoid duplicate observations from same camera
            //--------------------------------------------------
            if (used_cameras.count(camera_id) > 0)
            {
                continue;
            }


            //--------------------------------------------------
            // Make sure camera exists
            //--------------------------------------------------
            auto camera_it =
                camera_map.find(camera_id);

            if (camera_it == camera_map.end())
            {
                continue;
            }

            const Camera& camera =
                camera_it->second;


            //--------------------------------------------------
            // Make sure keypoint index is valid
            //--------------------------------------------------
            if (keypoint_index < 0 ||
                keypoint_index >=
                    static_cast<int>(
                        camera.keypoints_.size()))
            {
                continue;
            }

            used_cameras.insert(camera_id);


            //--------------------------------------------------
            // Create observation
            //--------------------------------------------------
            Observation observation;

            observation.camera_id_ =
                camera_id;

            observation.keypoint_idx_ =
                keypoint_index;

            const cv::Point2f& pt = camera.keypoints_[keypoint_index].pt;

            observation.pixel_ = Vec2d( static_cast<double>(pt.x), static_cast<double>(pt.y));

            landmark.observations_.push_back(
                observation);
        }


        //------------------------------------------------------
        // Need observations from at least two cameras
        //------------------------------------------------------
        if (landmark.observations_.size() < 2)
        {
            continue;
        }


        //------------------------------------------------------
        // Add landmark
        //------------------------------------------------------
        landmark_map.emplace(
            landmark_id,
            std::move(landmark));

        ++landmark_id;
    }
}


// ============================================================
// Save / load landmark cache
//
// Binary format so a rerun can resume without redoing the
// expensive exhaustive matching:
//
// [int64 landmark_count]
// per landmark:
//   [int landmark_id_][char optimized_]
//   [double x3 initial_position_][double x3 optimized_position_]
//   [int64 observation_count]
//   per observation:
//     [int camera_id_][int keypoint_idx_]
//     [double x2 pixel_][double depth_][double optimized_depth_]
//     [char optimized_]
// ============================================================

Status SIFT::save_landmarks(
    const std::string& file_path,
    const std::map<int, Landmark>& landmarks)
{
    std::ofstream out(file_path, std::ios::binary);

    if (!out.is_open())
    {
        return {
            false,
            "Failed to open landmark cache for writing: " + file_path
        };
    }

    const int64_t landmark_count =
        static_cast<int64_t>(landmarks.size());

    out.write(
        reinterpret_cast<const char*>(&landmark_count),
        sizeof(landmark_count));

    for (const auto& [landmark_id, landmark] : landmarks)
    {
        out.write(
            reinterpret_cast<const char*>(&landmark.landmark_id_),
            sizeof(landmark.landmark_id_));

        const char optimized = landmark.optimized_ ? 1 : 0;
        out.write(&optimized, sizeof(optimized));

        out.write(
            reinterpret_cast<const char*>(landmark.initial_position_.data()),
            sizeof(double) * 3);

        out.write(
            reinterpret_cast<const char*>(landmark.optimized_position_.data()),
            sizeof(double) * 3);

        const int64_t observation_count =
            static_cast<int64_t>(landmark.observations_.size());

        out.write(
            reinterpret_cast<const char*>(&observation_count),
            sizeof(observation_count));

        for (const Observation& observation : landmark.observations_)
        {
            out.write(
                reinterpret_cast<const char*>(&observation.camera_id_),
                sizeof(observation.camera_id_));

            out.write(
                reinterpret_cast<const char*>(&observation.keypoint_idx_),
                sizeof(observation.keypoint_idx_));

            out.write(
                reinterpret_cast<const char*>(observation.pixel_.data()),
                sizeof(double) * 2);

            out.write(
                reinterpret_cast<const char*>(&observation.depth_),
                sizeof(observation.depth_));

            out.write(
                reinterpret_cast<const char*>(&observation.optimized_depth_),
                sizeof(observation.optimized_depth_));

            const char obs_optimized = observation.optimized_ ? 1 : 0;
            out.write(&obs_optimized, sizeof(obs_optimized));
        }
    }

    if (!out)
    {
        return {
            false,
            "Failed while writing landmark cache: " + file_path
        };
    }

    return {
        true,
        "Saved " + std::to_string(landmarks.size())
            + " landmarks to " + file_path
    };
}

Status SIFT::load_landmarks(
    const std::string& file_path,
    std::map<int, Landmark>& landmarks)
{
    landmarks.clear();

    std::ifstream in(file_path, std::ios::binary);

    if (!in.is_open())
    {
        return {
            false,
            "Failed to open landmark cache for reading: " + file_path
        };
    }

    int64_t landmark_count = 0;

    in.read(
        reinterpret_cast<char*>(&landmark_count),
        sizeof(landmark_count));

    for (int64_t i = 0; i < landmark_count && in; ++i)
    {
        Landmark landmark;

        in.read(
            reinterpret_cast<char*>(&landmark.landmark_id_),
            sizeof(landmark.landmark_id_));

        char optimized = 0;
        in.read(&optimized, sizeof(optimized));
        landmark.optimized_ = (optimized != 0);

        in.read(
            reinterpret_cast<char*>(landmark.initial_position_.data()),
            sizeof(double) * 3);

        in.read(
            reinterpret_cast<char*>(landmark.optimized_position_.data()),
            sizeof(double) * 3);

        int64_t observation_count = 0;

        in.read(
            reinterpret_cast<char*>(&observation_count),
            sizeof(observation_count));

        landmark.observations_.reserve(observation_count);

        for (int64_t j = 0; j < observation_count && in; ++j)
        {
            Observation observation;

            in.read(
                reinterpret_cast<char*>(&observation.camera_id_),
                sizeof(observation.camera_id_));

            in.read(
                reinterpret_cast<char*>(&observation.keypoint_idx_),
                sizeof(observation.keypoint_idx_));

            in.read(
                reinterpret_cast<char*>(observation.pixel_.data()),
                sizeof(double) * 2);

            in.read(
                reinterpret_cast<char*>(&observation.depth_),
                sizeof(observation.depth_));

            in.read(
                reinterpret_cast<char*>(&observation.optimized_depth_),
                sizeof(observation.optimized_depth_));

            char obs_optimized = 0;
            in.read(&obs_optimized, sizeof(obs_optimized));
            observation.optimized_ = (obs_optimized != 0);

            landmark.observations_.push_back(observation);
        }

        landmarks.emplace(landmark.landmark_id_, std::move(landmark));
    }

    if (!in && !in.eof())
    {
        landmarks.clear();
        return {
            false,
            "Failed while reading landmark cache: " + file_path
        };
    }

    return {
        true,
        "Loaded " + std::to_string(landmarks.size())
            + " landmarks from " + file_path
    };
}


// ============================================================
// Match + verify a single camera pair
// ============================================================

void SIFT::match_pair(
    int camera_id_1,
    const Camera& camera_1,
    int camera_id_2,
    const Camera& camera_2,
    float ratio_threshold,
    double ransac_threshold,
    int min_inliers,
    std::vector<cv::DMatch>& verified_matches,
    std::vector<uchar>& inlier_mask,
    float max_match_distance)
{
    verified_matches.clear();
    inlier_mask.clear();

    // =================================================
    // 1. KNN descriptor matching
    // =================================================

    std::vector<std::vector<cv::DMatch>> knn_matches;

    knn_matching(
        camera_1.descriptors_,
        camera_2.descriptors_,
        knn_matches);


    // =================================================
    // 2. Lowe ratio test
    // =================================================

    std::vector<cv::DMatch> ratio_matches;

    lowe_ratio_test(
        knn_matches,
        ratio_matches,
        ratio_threshold,
        max_match_distance);

    if (static_cast<int>(ratio_matches.size()) < min_inliers)
    {
        return;
    }


    // =================================================
    // 3. Convert matches to pixels
    // =================================================

    std::vector<cv::Point2f> points_1;
    std::vector<cv::Point2f> points_2;

    match_to_pixels(
        ratio_matches,
        camera_1,
        camera_2,
        points_1,
        points_2);


    // =================================================
    // 4. Fundamental matrix + RANSAC
    // =================================================

    const int num_inliers =
        geometric_verification(
            points_1,
            points_2,
            ransac_threshold,
            inlier_mask);

    if (num_inliers < min_inliers)
    {
        inlier_mask.clear();
        return;
    }

    verified_matches = std::move(ratio_matches);
}


// ============================================================
// Sequential pair matching
// ============================================================

Status SIFT::sequential_pair_matching(
    const std::map<int, Camera>& camera_map,
    Config& config,
    std::map<int, Landmark>& landmarks)
{
    landmarks.clear();

    const int window_size =
        config.sequential_match_window_size_;

    if (window_size <= 0)
    {
        return {
            false,
            "sequential_match_window_size must be greater than zero."
        };
    }

    const float ratio_threshold =
        static_cast<float>(config.ratio_threshold_);
    const double ransac_threshold =
        config.ransac_threshold_;
    const int min_inliers =
        config.min_inliers_;
    const float max_match_distance =
        static_cast<float>(config.sift_max_match_distance_);

    // Preserve map order even when a camera has no descriptors. The window
    // describes neighboring frames, not neighboring non-empty descriptors.
    std::vector<std::pair<int, const Camera*>> cameras;
    cameras.reserve(camera_map.size());

    for (const auto& [camera_id, camera] : camera_map)
    {
        cameras.emplace_back(camera_id, &camera);
    }

    std::unordered_map<FeatureNode, FeatureNode, FeatureNodeHash> parent;
    std::unordered_map<FeatureNode, int, FeatureNodeHash> rank;
    std::unordered_map<int, std::vector<FeatureNode>> camera_feature_nodes;
    size_t evaluated_pairs = 0;
    size_t verified_pairs = 0;

    for (size_t i = 0; i < cameras.size(); ++i)
    {
        if (cameras[i].second->descriptors_.empty())
        {
            continue;
        }

        const size_t window_end = std::min(
            cameras.size(),
            i + static_cast<size_t>(window_size) + 1);

        for (size_t j = i + 1; j < window_end; ++j)
        {
            if (cameras[j].second->descriptors_.empty())
            {
                continue;
            }

            ++evaluated_pairs;

            std::vector<cv::DMatch> verified_matches;
            std::vector<uchar> inlier_mask;

            match_pair(
                cameras[i].first,
                *cameras[i].second,
                cameras[j].first,
                *cameras[j].second,
                ratio_threshold,
                ransac_threshold,
                min_inliers,
                verified_matches,
                inlier_mask,
                max_match_distance);

            if (verified_matches.empty())
            {
                continue;
            }

            ++verified_pairs;
            union_matches(
                cameras[i].first,
                *cameras[i].second,
                cameras[j].first,
                *cameras[j].second,
                verified_matches,
                inlier_mask,
                parent,
                rank,
                camera_feature_nodes,
                config.feature_track_merge_radius_px_);

            std::cout
                << "Camera " << cameras[i].first
                << " <-> " << cameras[j].first
                << " | ratio matches: " << verified_matches.size()
                << " | inliers: "
                << std::count(inlier_mask.begin(), inlier_mask.end(), 1)
                << std::endl;
        }
    }

    extract_landmark_map(parent, camera_map, landmarks);
    const fs::path tracker_dir =
        fs::path(config.output_path_).parent_path() / "landmark_track_images";
    Status tracker_status = export_landmark_track_images(
        landmarks,
        camera_map,
        tracker_dir.string());
    std::cout << "[landmark_track_images] " << tracker_status.message << std::endl;

    std::cout
        << "Sequential matching evaluated " << evaluated_pairs
        << " pairs in a window of " << window_size
        << ", verified " << verified_pairs
        << ", and generated " << landmarks.size()
        << " landmark tracks." << std::endl;

    return {
        true,
        "Sequential feature matching done! Evaluated "
            + std::to_string(evaluated_pairs)
            + " pairs and generated "
            + std::to_string(landmarks.size())
            + " landmarks."
    };
}


Status SIFT::prior_pose_landmark_generation(
    std::map<int, Camera>& camera_map,
    const CameraIntrinsic& intrinsic,
    Config& config,
    std::map<int, Landmark>& landmarks)
{
    landmarks.clear();

    const Mat3d& K = intrinsic.K;
    const double epipolar_threshold_px = config.prior_epipolar_threshold_px_;
    const double min_parallax_deg = config.prior_min_parallax_deg_;
    const double max_reprojection_error_px = config.prior_max_reprojection_error_px_;
    const int max_camera_pair_gap = std::max(1, config.prior_max_camera_pair_gap_);
    const int min_epipolar_matches_per_pair = std::max(1, config.prior_min_epipolar_matches_per_pair_);
    const int min_usable_matches_per_pair = std::max(1, config.prior_min_usable_matches_per_pair_);
    const int max_matches_per_pair_for_geometry = std::max(1, config.prior_max_matches_per_pair_for_geometry_);
    const int min_landmark_observations = std::max(2, config.prior_min_landmark_observations_);

    std::vector<std::map<int, Camera>::iterator> valid_cameras;
    valid_cameras.reserve(camera_map.size());
    for (auto it = camera_map.begin(); it != camera_map.end(); ++it)
    {
        if (it->second.matched_lidar_id_ < 0)
        {
            continue;
        }

        if (it->second.descriptors_.empty())
        {
            Status sift_status = extract_sift(it->second, config);
            if (!sift_status.success)
            {
                return sift_status;
            }
        }

        if (it->second.descriptors_.empty())
        {
            continue;
        }

        valid_cameras.push_back(it);
    }

    if (valid_cameras.size() < 2)
    {
        return {false, "Need at least two valid cameras with descriptors for prior-pose landmark generation."};
    }

    auto make_projection = [&](const SE3d& T_cw) -> cv::Mat {
        Eigen::Matrix<double, 3, 4> Rt;
        Rt.leftCols<3>() = T_cw.rotationMatrix();
        Rt.rightCols<1>() = T_cw.translation();

        Eigen::Matrix<double, 3, 4> P_eigen = K * Rt;
        cv::Mat P(3, 4, CV_64F);
        cv::eigen2cv(P_eigen, P);
        return P;
    };

    auto build_fundamental = [&](const SE3d& T_cw_1, const SE3d& T_cw_2) -> Mat3d {
        const SE3d T_c2_c1 = T_cw_2 * T_cw_1.inverse();
        const Mat3d R = T_c2_c1.rotationMatrix();
        const Vec3d t = T_c2_c1.translation();

        Mat3d t_hat;
        t_hat << 0, -t.z(), t.y(),
                 t.z(), 0, -t.x(),
                -t.y(), t.x(), 0;

        const Mat3d K_inv = K.inverse();
        return K_inv.transpose() * (t_hat * R) * K_inv;
    };

    auto epipolar_filter = [&](const Camera& cam_1,
                               const Camera& cam_2,
                               const Mat3d& F) -> std::vector<cv::DMatch>
    {
        std::vector<cv::DMatch> kept;
        std::vector<std::vector<cv::DMatch>> knn_matches;
        knn_matching(cam_1.descriptors_, cam_2.descriptors_, knn_matches);

        std::vector<cv::DMatch> ratio_matches;
        lowe_ratio_test(
            knn_matches,
            ratio_matches,
            static_cast<float>(config.ratio_threshold_),
            static_cast<float>(config.sift_max_match_distance_));

        for (const cv::DMatch& match : ratio_matches)
        {
            const cv::Point2f& p_1 = cam_1.keypoints_[match.queryIdx].pt;
            const cv::Point2f& p_2 = cam_2.keypoints_[match.trainIdx].pt;
            const Vec3d x_1(p_1.x, p_1.y, 1.0);
            const Vec3d x_2(p_2.x, p_2.y, 1.0);

            const Vec3d line_2 = F * x_1;
            const Vec3d line_1 = F.transpose() * x_2;

            const double denom_2 = std::hypot(line_2.x(), line_2.y());
            const double denom_1 = std::hypot(line_1.x(), line_1.y());
            if (denom_1 < 1e-9 || denom_2 < 1e-9)
            {
                continue;
            }

            const double err_2 = std::abs(x_2.dot(line_2)) / denom_2;
            const double err_1 = std::abs(x_1.dot(line_1)) / denom_1;
            const double sym_err = 0.5 * (err_1 + err_2);

            if (sym_err <= epipolar_threshold_px)
            {
                kept.push_back(match);
            }
        }

        return kept;
    };

    auto collect_usable_matches = [&](const Camera& cam_1,
                                      const Camera& cam_2,
                                      const std::vector<cv::DMatch>& matches)
        -> std::vector<std::tuple<cv::DMatch, double, double>>
    {
        std::vector<std::tuple<cv::DMatch, double, double>> usable_matches;
        if (matches.empty())
        {
            return usable_matches;
        }

        const cv::Mat P1 = make_projection(cam_1.initial_T_cw_);
        const cv::Mat P2 = make_projection(cam_2.initial_T_cw_);
        const Vec3d C1_w = cam_1.initial_T_cw_.inverse().translation();
        const Vec3d C2_w = cam_2.initial_T_cw_.inverse().translation();

        std::vector<cv::Point2f> pts_1;
        std::vector<cv::Point2f> pts_2;
        pts_1.reserve(matches.size());
        pts_2.reserve(matches.size());
        for (const cv::DMatch& match : matches)
        {
            pts_1.push_back(cam_1.keypoints_[match.queryIdx].pt);
            pts_2.push_back(cam_2.keypoints_[match.trainIdx].pt);
        }

        cv::Mat points_4d;
        cv::triangulatePoints(P1, P2, pts_1, pts_2, points_4d);

        auto get_points_4d_value = [&](int row, int col) -> double {
            if (points_4d.type() == CV_32F)
            {
                return static_cast<double>(points_4d.at<float>(row, col));
            }
            return points_4d.at<double>(row, col);
        };

        for (int i = 0; i < points_4d.cols; ++i)
        {
            const double w = get_points_4d_value(3, i);
            if (std::abs(w) < 1e-9)
            {
                continue;
            }

            const Vec3d p_w(
                get_points_4d_value(0, i) / w,
                get_points_4d_value(1, i) / w,
                get_points_4d_value(2, i) / w);

            if (!std::isfinite(p_w.x()) ||
                !std::isfinite(p_w.y()) ||
                !std::isfinite(p_w.z()))
            {
                continue;
            }

            const Vec3d p_c1 = cam_1.initial_T_cw_ * p_w;
            const Vec3d p_c2 = cam_2.initial_T_cw_ * p_w;
            if (p_c1.z() <= 0.0 || p_c2.z() <= 0.0)
            {
                continue;
            }

            const Vec3d r1 = (p_w - C1_w).normalized();
            const Vec3d r2 = (p_w - C2_w).normalized();
            const double parallax_deg =
                std::acos(std::clamp(r1.dot(r2), -1.0, 1.0)) * 180.0 / M_PI;
            if (parallax_deg < min_parallax_deg)
            {
                continue;
            }

            const double u1 = K(0, 0) * p_c1.x() / p_c1.z() + K(0, 2);
            const double v1 = K(1, 1) * p_c1.y() / p_c1.z() + K(1, 2);
            const double u2 = K(0, 0) * p_c2.x() / p_c2.z() + K(0, 2);
            const double v2 = K(1, 1) * p_c2.y() / p_c2.z() + K(1, 2);
            const double reproj_err_1 = std::hypot(u1 - pts_1[i].x, v1 - pts_1[i].y);
            const double reproj_err_2 = std::hypot(u2 - pts_2[i].x, v2 - pts_2[i].y);
            if (!std::isfinite(reproj_err_1) || !std::isfinite(reproj_err_2))
            {
                continue;
            }

            if (reproj_err_1 > max_reprojection_error_px ||
                reproj_err_2 > max_reprojection_error_px)
            {
                continue;
            }

            usable_matches.emplace_back(matches[i], p_c1.z(), p_c2.z());
        }

        return usable_matches;
    };

    std::unordered_map<FeatureNode, FeatureNode, FeatureNodeHash> parent;
    std::unordered_map<FeatureNode, int, FeatureNodeHash> rank;
    std::unordered_map<int, std::vector<FeatureNode>> camera_feature_nodes;
    std::unordered_map<FeatureNode, DepthAccumulator, FeatureNodeHash> triangulated_depths;

    // Reserve up front so the tables grow once instead of repeatedly
    // rehashing (each rehash briefly holds both the old and new bucket
    // arrays, which is what makes RSS climb during this pass).
    size_t estimated_keypoints = 0;
    for (const auto& camera_it : valid_cameras)
    {
        estimated_keypoints += camera_it->second.keypoints_.size();
    }
    parent.reserve(estimated_keypoints);
    rank.reserve(estimated_keypoints);
    triangulated_depths.reserve(estimated_keypoints);
    camera_feature_nodes.reserve(valid_cameras.size());

    int evaluated_pairs = 0;
    int accepted_pairs = 0;
    int total_kept_for_geometry = 0;

    for (size_t i = 0; i + 1 < valid_cameras.size(); ++i)
    {
        const Camera& cam_i = valid_cameras[i]->second;
        std::cout << "[PriorLandmarkGen] processing camera_id=" << valid_cameras[i]->first
                  << " (" << cam_i.camera_name_ << ") "
                  << (i + 1) << "/" << valid_cameras.size() << std::endl;

        for (size_t j = i + 1; j < valid_cameras.size(); ++j)
        {
            if (static_cast<int>(j - i) > max_camera_pair_gap)
            {
                continue;
            }

            ++evaluated_pairs;
            Camera& cam_1 = valid_cameras[i]->second;
            Camera& cam_2 = valid_cameras[j]->second;

            const Mat3d F = build_fundamental(cam_1.initial_T_cw_, cam_2.initial_T_cw_);
            std::vector<cv::DMatch> filtered_matches = epipolar_filter(cam_1, cam_2, F);
            if (static_cast<int>(filtered_matches.size()) < min_epipolar_matches_per_pair)
            {
                continue;
            }

            if (static_cast<int>(filtered_matches.size()) > max_matches_per_pair_for_geometry)
            {
                std::partial_sort(
                    filtered_matches.begin(),
                    filtered_matches.begin() + max_matches_per_pair_for_geometry,
                    filtered_matches.end(),
                    [](const cv::DMatch& a, const cv::DMatch& b) {
                        return a.distance < b.distance;
                    });
                filtered_matches.resize(max_matches_per_pair_for_geometry);
            }

            const std::vector<std::tuple<cv::DMatch, double, double>> usable_samples =
                collect_usable_matches(cam_1, cam_2, filtered_matches);
            if (static_cast<int>(usable_samples.size()) < min_usable_matches_per_pair)
            {
                continue;
            }

            ++accepted_pairs;
            total_kept_for_geometry += static_cast<int>(usable_samples.size());

            std::vector<cv::DMatch> usable_matches;
            usable_matches.reserve(usable_samples.size());
            for (const auto& usable_sample : usable_samples)
            {
                const cv::DMatch& match = std::get<0>(usable_sample);
                const double depth_1 = std::get<1>(usable_sample);
                const double depth_2 = std::get<2>(usable_sample);

                FeatureNode node_1 = resolve_feature_node(
                    valid_cameras[i]->first,
                    match.queryIdx,
                    cam_1,
                    config.feature_track_merge_radius_px_,
                    camera_feature_nodes);
                FeatureNode node_2 = resolve_feature_node(
                    valid_cameras[j]->first,
                    match.trainIdx,
                    cam_2,
                    config.feature_track_merge_radius_px_,
                    camera_feature_nodes);

                triangulated_depths[node_1].add(depth_1);
                triangulated_depths[node_2].add(depth_2);
                usable_matches.push_back(match);
            }

            std::vector<uchar> inlier_mask(usable_matches.size(), 1);
            union_matches(
                valid_cameras[i]->first,
                cam_1,
                valid_cameras[j]->first,
                cam_2,
                usable_matches,
                inlier_mask,
                parent,
                rank,
                camera_feature_nodes,
                config.feature_track_merge_radius_px_);
        }
    }

    extract_landmark_map(parent, camera_map, landmarks);

    for (auto& [landmark_id, landmark] : landmarks)
    {
        (void)landmark_id;
        for (Observation& observation : landmark.observations_)
        {
            const FeatureNode node{observation.camera_id_, observation.keypoint_idx_};
            auto depth_it = triangulated_depths.find(node);
            if (depth_it == triangulated_depths.end())
            {
                continue;
            }

            observation.depth_ = depth_it->second.mean();
        }
    }

    if (min_landmark_observations > 2)
    {
        std::map<int, Landmark> filtered_landmarks;
        for (const auto& [landmark_id, landmark] : landmarks)
        {
            if (static_cast<int>(landmark.observations_.size()) >= min_landmark_observations)
            {
                filtered_landmarks.emplace(landmark_id, landmark);
            }
        }
        landmarks = std::move(filtered_landmarks);
    }

    if (landmarks.empty())
    {
        return {
            false,
            "Prior-pose landmark generation produced no landmarks. "
            "evaluated_pairs=" + std::to_string(evaluated_pairs) +
            " accepted_pairs=" + std::to_string(accepted_pairs)
        };
    }

    std::cout
        << "[PriorLandmarkGen] evaluated_pairs=" << evaluated_pairs
        << " accepted_pairs=" << accepted_pairs
        << " usable_matches=" << total_kept_for_geometry
        << " landmarks=" << landmarks.size()
        << std::endl;

    return {
        true,
        "Prior-pose landmark generation done. landmarks="
            + std::to_string(landmarks.size())
    };
}


// ============================================================
// Exhaustive pair matching
// ============================================================

Status SIFT::exhaust_pair_matching(
    const std::map<int, Camera>& camera_map,
    Config& config,
    std::map<int, Landmark>& landmarks)
{
    //----------------------------------------------------------
    // Config
    //----------------------------------------------------------
    const float ratio_threshold =
        config.ratio_threshold_;

    const double ransac_threshold =
        config.ransac_threshold_;

    const int min_inliers =
        config.min_inliers_;

    const float max_match_distance =
        static_cast<float>(config.sift_max_match_distance_);


    landmarks.clear();


    //----------------------------------------------------------
    // Union-Find must live across ALL image pairs.
    //
    // Very important:
    // Do NOT create this inside the inner loop.
    //----------------------------------------------------------
    std::unordered_map<
        FeatureNode,
        FeatureNode,
        FeatureNodeHash
    > parent;
    std::unordered_map<FeatureNode, int, FeatureNodeHash> rank;
    std::unordered_map<int, std::vector<FeatureNode>> camera_feature_nodes;


    //----------------------------------------------------------
    // Exhaustive image pair matching
    //----------------------------------------------------------
    for (auto it1 = camera_map.begin();
         it1 != camera_map.end();
         ++it1)
    {
        const int camera_id_1 =
            it1->first;

        const Camera& camera_1 =
            it1->second;


        //------------------------------------------------------
        // Skip camera without descriptors
        //------------------------------------------------------
        if (camera_1.descriptors_.empty())
        {
            continue;
        }


        //------------------------------------------------------
        // Start from next camera.
        //
        // This avoids:
        //
        // 0-0
        // 1-1
        //
        // and duplicated pairs:
        //
        // 0-1 and 1-0
        //------------------------------------------------------
        auto it2 = std::next(it1);

        for (; it2 != camera_map.end(); ++it2)
        {
            const int camera_id_2 =
                it2->first;

            const Camera& camera_2 =
                it2->second;


            if (camera_2.descriptors_.empty())
            {
                continue;
            }


            // =================================================
            // 1. KNN descriptor matching
            // =================================================

            std::vector<
                std::vector<cv::DMatch>
            > knn_matches;

            knn_matching(
                camera_1.descriptors_,
                camera_2.descriptors_,
                knn_matches);


            // =================================================
            // 2. Lowe ratio test
            // =================================================

            std::vector<cv::DMatch>
                ratio_matches;

            lowe_ratio_test(
                knn_matches,
                ratio_matches,
                ratio_threshold,
                max_match_distance);


            if (static_cast<int>(
                    ratio_matches.size()) <
                min_inliers)
            {
                continue;
            }


            // =================================================
            // 3. Convert matches to pixels
            // =================================================

            std::vector<cv::Point2f>
                points_1;

            std::vector<cv::Point2f>
                points_2;

            match_to_pixels(
                ratio_matches,
                camera_1,
                camera_2,
                points_1,
                points_2);


            // =================================================
            // 4. Fundamental matrix + RANSAC
            // =================================================

            std::vector<uchar>
                inlier_mask;

            int num_inliers =
                geometric_verification(
                    points_1,
                    points_2,
                    ransac_threshold,
                    inlier_mask);


            if (num_inliers < min_inliers)
            {
                continue;
            }


            // =================================================
            // 5. Add verified matches to Union-Find
            // =================================================

            union_matches(
                camera_id_1,
                camera_1,
                camera_id_2,
                camera_2,
                ratio_matches,
                inlier_mask,
                parent,
                rank,
                camera_feature_nodes,
                config.feature_track_merge_radius_px_);


            //--------------------------------------------------
            // Debug output
            //--------------------------------------------------
            std::cout
                << "Camera "
                << camera_id_1
                << " <-> "
                << camera_id_2
                << " | ratio matches: "
                << ratio_matches.size()
                << " | inliers: "
                << num_inliers
                << std::endl;
        }
    }


    //----------------------------------------------------------
    // Convert tracks to landmark map
    //----------------------------------------------------------
    extract_landmark_map(
        parent,
        camera_map,
        landmarks);
    const fs::path tracker_dir =
        fs::path(config.output_path_).parent_path() / "landmark_track_images";
    Status tracker_status = export_landmark_track_images(
        landmarks,
        camera_map,
        tracker_dir.string());
    std::cout << "[landmark_track_images] " << tracker_status.message << std::endl;


    //----------------------------------------------------------
    // Summary
    //----------------------------------------------------------
    std::cout
        << "Generated "
        << landmarks.size()
        << " landmark tracks."
        << std::endl;


    return {
        true,
        "Feature matching done! Generated "
            + std::to_string(landmarks.size())
            + " landmarks."
    };
}


// ============================================================
// Exhaustive pair matching (parallelized over camera pairs)
// ============================================================

Status SIFT::exhaust_pair_matching_parallel(
    const std::map<int, Camera>& camera_map,
    Config& config,
    std::map<int, Landmark>& landmarks)
{
    //----------------------------------------------------------
    // Config
    //----------------------------------------------------------
    const float ratio_threshold =
        config.ratio_threshold_;

    const double ransac_threshold =
        config.ransac_threshold_;

    const int min_inliers =
        config.min_inliers_;

    const float max_match_distance =
        static_cast<float>(config.sift_max_match_distance_);


    landmarks.clear();


    //----------------------------------------------------------
    // Flatten cameras with descriptors so pairs can be indexed
    // for tbb::parallel_for.
    //----------------------------------------------------------
    std::vector<std::pair<int, const Camera*>> cameras;
    cameras.reserve(camera_map.size());

    for (const auto& [camera_id, camera] : camera_map)
    {
        if (!camera.descriptors_.empty())
        {
            cameras.emplace_back(camera_id, &camera);
        }
    }

    //----------------------------------------------------------
    // Enumerate all (i, j) pairs, i < j, up front so each pair
    // maps to a unique slot -> no data races when writing results.
    //----------------------------------------------------------
    std::vector<std::pair<size_t, size_t>> pair_indices;
    pair_indices.reserve(cameras.size() * (cameras.size() - 1) / 2);

    for (size_t i = 0; i < cameras.size(); ++i)
    {
        for (size_t j = i + 1; j < cameras.size(); ++j)
        {
            pair_indices.emplace_back(i, j);
        }
    }

    struct PairResult
    {
        std::vector<cv::DMatch> verified_matches;
        std::vector<uchar> inlier_mask;
    };

    std::vector<PairResult> pair_results(pair_indices.size());

    const size_t total_pairs = pair_indices.size();
    std::cout
        << "[exhaust_pair_matching_parallel] Matching "
        << total_pairs
        << " camera pairs across "
        << cameras.size()
        << " cameras..."
        << std::endl;

    std::atomic<size_t> completed_pairs{0};
    std::mutex progress_mutex;
    // Print roughly every 5% so progress is visible without flooding stdout.
    const size_t print_interval =
        std::max<size_t>(1, total_pairs / 20);

    tbb::parallel_for(
        tbb::blocked_range<size_t>(0, pair_indices.size()),
        [&](const tbb::blocked_range<size_t>& range)
        {
            for (size_t k = range.begin(); k != range.end(); ++k)
            {
                const auto [i, j] = pair_indices[k];

                match_pair(
                    cameras[i].first,
                    *cameras[i].second,
                    cameras[j].first,
                    *cameras[j].second,
                    ratio_threshold,
                    ransac_threshold,
                    min_inliers,
                    pair_results[k].verified_matches,
                    pair_results[k].inlier_mask,
                    max_match_distance);

                const size_t done = ++completed_pairs;
                if (done % print_interval == 0 || done == total_pairs)
                {
                    std::lock_guard<std::mutex> lock(progress_mutex);
                    std::cout
                        << "[exhaust_pair_matching_parallel] "
                        << done << " / " << total_pairs
                        << " pairs matched ("
                        << (100.0 * done / total_pairs)
                        << "%)"
                        << std::endl;
                }
            }
        });


    //----------------------------------------------------------
    // Union-Find must live across ALL image pairs, merged
    // sequentially since it isn't thread-safe.
    //----------------------------------------------------------
    std::unordered_map<FeatureNode, FeatureNode, FeatureNodeHash> parent;
    std::unordered_map<FeatureNode, int, FeatureNodeHash> rank;
    std::unordered_map<int, std::vector<FeatureNode>> camera_feature_nodes;

    for (size_t k = 0; k < pair_indices.size(); ++k)
    {
        const auto& result = pair_results[k];

        if (result.verified_matches.empty())
        {
            continue;
        }

        const auto [i, j] = pair_indices[k];
        const int camera_id_1 = cameras[i].first;
        const int camera_id_2 = cameras[j].first;

        union_matches(
            camera_id_1,
            *cameras[i].second,
            camera_id_2,
            *cameras[j].second,
            result.verified_matches,
            result.inlier_mask,
            parent,
            rank,
            camera_feature_nodes,
            config.feature_track_merge_radius_px_);

        std::cout
            << "Camera "
            << camera_id_1
            << " <-> "
            << camera_id_2
            << " | ratio matches: "
            << result.verified_matches.size()
            << " | inliers: "
            << std::count(result.inlier_mask.begin(), result.inlier_mask.end(), 1)
            << std::endl;
    }


    //----------------------------------------------------------
    // Convert tracks to landmark map
    //----------------------------------------------------------
    extract_landmark_map(
        parent,
        camera_map,
        landmarks);
    const fs::path tracker_dir =
        fs::path(config.output_path_).parent_path() / "landmark_track_images";
    Status tracker_status = export_landmark_track_images(
        landmarks,
        camera_map,
        tracker_dir.string());
    std::cout << "[landmark_track_images] " << tracker_status.message << std::endl;


    //----------------------------------------------------------
    // Summary
    //----------------------------------------------------------
    std::cout
        << "Generated "
        << landmarks.size()
        << " landmark tracks."
        << std::endl;


    return {
        true,
        "Feature matching done! Generated "
            + std::to_string(landmarks.size())
            + " landmarks."
    };
}
