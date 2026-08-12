//
// Created by chuchu on 8/5/26.
//
#include "General.h"

#include <yaml-cpp/yaml.h>
#include <iostream>
#include <filesystem>


Config::Config(const std::string& yaml_path)
{
    try
    {
        YAML::Node config = YAML::LoadFile(yaml_path);

        auto get_string = [&](const std::string& key)
        {
            try
            {
                return config[key].as<std::string>();
            }
            catch (const YAML::Exception& e)
            {
                throw std::runtime_error(
                    "Failed to load string field '" +
                    key +
                    "': " +
                    e.what());
            }
        };

        auto get_int = [&](const std::string& key)
        {
            try
            {
                return config[key].as<int>();
            }
            catch (const YAML::Exception& e)
            {
                throw std::runtime_error(
                    "Failed to load int field '" +
                    key +
                    "': " +
                    e.what());
            }
        };

        auto get_double = [&](const std::string& key)
        {
            try
            {
                return config[key].as<double>();
            }
            catch (const YAML::Exception& e)
            {
                throw std::runtime_error(
                    "Failed to load double field '" +
                    key +
                    "': " +
                    e.what());
            }
        };

        auto get_bool = [&](const std::string& key)
        {
            try
            {
                return config[key].as<bool>();
            }
            catch (const YAML::Exception& e)
            {
                throw std::runtime_error(
                    "Failed to load bool field '" +
                    key +
                    "': " +
                    e.what());
            }
        };

        // Paths
        input_img_dir_ =
            get_string("input_img_dir");

        camera_extrinsic_path_ =
            get_string("camera_extrinsic_path");

        camera_intrinsic_path_ =
            get_string("camera_intrinsic_path");

        camera_timestamp_path_ =
            get_string("camera_timestamp_path");

        projective_z_buffer_dir_ =
            get_string("projective_z_buffer_dir");

        output_path_ =
            get_string("output_path");

        lidar_ply_dirs_ =
            get_string("lidar_ply_dirs");

        lidar_kf_path_ =
            get_string("lidar_kf_path");

        if(config["landmark_cache_path"]){
            landmark_cache_path_ = get_string("landmark_cache_path");
        }

        // SIFT
        sift_nfeatures_ =
            get_int("sift_nfeatures");

        sift_n_octave_layers_ =
            get_int("sift_n_octave_layers");

        sift_contrast_threshold_ =
            get_double("sift_contrast_threshold");

        sift_edge_threshold_ =
            get_double("sift_edge_threshold");

        sift_sigma_ =
            get_double("sift_sigma");

        // Feature matching
        ratio_threshold_ =
            get_double("ratio_threshold");

        ransac_threshold_ =
            get_double("ransac_threshold");

        min_inliers_ =
            get_int("min_inliers");

        if(config["sift_max_match_distance"]){
            sift_max_match_distance_ = get_double("sift_max_match_distance");
        }

        if(config["feature_track_merge_radius_px"]){
            feature_track_merge_radius_px_ = get_double("feature_track_merge_radius_px");
        }

        if(config["sequential_match_window_size"]){
            sequential_match_window_size_ =
                get_int("sequential_match_window_size");
        }

        if(config["use_prior_pose_landmark_generation"]){
            use_prior_pose_landmark_generation_ =
                get_bool("use_prior_pose_landmark_generation");
        }
        if(config["prior_epipolar_threshold_px"]){
            prior_epipolar_threshold_px_ =
                get_double("prior_epipolar_threshold_px");
        }
        if(config["prior_min_parallax_deg"]){
            prior_min_parallax_deg_ =
                get_double("prior_min_parallax_deg");
        }
        if(config["prior_max_reprojection_error_px"]){
            prior_max_reprojection_error_px_ =
                get_double("prior_max_reprojection_error_px");
        }
        if(config["prior_max_camera_pair_gap"]){
            prior_max_camera_pair_gap_ =
                get_int("prior_max_camera_pair_gap");
        }
        if(config["prior_min_epipolar_matches_per_pair"]){
            prior_min_epipolar_matches_per_pair_ =
                get_int("prior_min_epipolar_matches_per_pair");
        }
        if(config["prior_min_usable_matches_per_pair"]){
            prior_min_usable_matches_per_pair_ =
                get_int("prior_min_usable_matches_per_pair");
        }
        if(config["prior_max_matches_per_pair_for_geometry"]){
            prior_max_matches_per_pair_for_geometry_ =
                get_int("prior_max_matches_per_pair_for_geometry");
        }
        if(config["prior_min_landmark_observations"]){
            prior_min_landmark_observations_ =
                get_int("prior_min_landmark_observations");
        }
        if(config["prior_export_min_observations"]){
            prior_export_min_observations_ =
                get_int("prior_export_min_observations");
        }

        num_iteration_ = get_int("num_iteration");

        if(config["max_images"]){
            max_images_ = get_int("max_images");
        }

        // Optional: fall back to the struct defaults if not present in the yaml.
        if(config["ceres_max_iterations"]){
            ceres_max_iterations_ = get_int("ceres_max_iterations");
        }
        if(config["ba_cost_threshold"]){
            ba_cost_threshold_ = get_double("ba_cost_threshold");
        }
        if(config["landmark_reprojection_error_threshold"]){
            landmark_reprojection_error_threshold_ = get_double("landmark_reprojection_error_threshold");
        }
        if(config["max_camera_lidar_time_gap"]){
            max_camera_lidar_time_gap_ = get_double("max_camera_lidar_time_gap");
        }
    }
    catch (const std::exception& e)
    {
        std::cerr
            << "Failed to load config file: "
            << yaml_path
            << std::endl;

        std::cerr
            << "Reason: "
            << e.what()
            << std::endl;

        throw;
    }
}
