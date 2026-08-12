//
// Standalone test: prior-pose epipolar filtering on the first two cameras.
//
#include <gtest/gtest.h>

#include "SIFT.h"
#include "DataType.h"
#include "General.h"
#include "DataLoader.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <iostream>
#include <map>
#include <filesystem>
#include <algorithm>
#include <chrono>
#include <numeric>
#include <sstream>
#include <vector>
#include <cmath>

namespace fs = std::filesystem;

TEST(SIFTTest, EpipolarFilterUsingPriorPoseTwoCameras)
{
    const std::string config_path =
        std::string(PROJECT_ROOT_DIR) + "/src/config.yaml";
    Config config(config_path);

    DataLoader data_loader(config);
    ASSERT_TRUE(data_loader.load().success);

    std::map<int, Camera>& camera_map = data_loader.get_camera_map();

    // Find the first two cameras that have a valid LiDAR association.
    auto is_valid = [](const std::pair<const int, Camera>& entry) {
        return entry.second.matched_lidar_id_ >= 0;
    };
    auto camera_it_1 = std::find_if(camera_map.begin(), camera_map.end(), is_valid);
    ASSERT_NE(camera_it_1, camera_map.end()) << "No camera has a valid lidar association.";
    auto camera_it_2 = std::find_if(std::next(camera_it_1), camera_map.end(), is_valid);
    ASSERT_NE(camera_it_2, camera_map.end()) << "Only one camera has a valid lidar association.";

    Camera& camera_1 = camera_it_1->second;
    Camera& camera_2 = camera_it_2->second;

    const Status status_1 = SIFT::extract_sift(camera_1, config);
    const Status status_2 = SIFT::extract_sift(camera_2, config);
    ASSERT_TRUE(status_1.success) << status_1.message;
    ASSERT_TRUE(status_2.success) << status_2.message;
    ASSERT_FALSE(camera_1.descriptors_.empty());
    ASSERT_FALSE(camera_2.descriptors_.empty());

    //----------------------------------------------------------
    // Candidate matches: KNN + Lowe ratio test only. Geometric
    // verification below comes from the priori poses instead of
    // RANSAC.
    //----------------------------------------------------------
    std::vector<std::vector<cv::DMatch>> knn_matches;
    SIFT::knn_matching(
        camera_1.descriptors_,
        camera_2.descriptors_,
        knn_matches);

    std::vector<cv::DMatch> ratio_matches;
    SIFT::lowe_ratio_test(
        knn_matches,
        ratio_matches,
        static_cast<float>(config.ratio_threshold_),
        static_cast<float>(config.sift_max_match_distance_));

    ASSERT_FALSE(ratio_matches.empty())
        << "No candidate SIFT matches were found.";

    //----------------------------------------------------------
    // Fundamental matrix from the priori relative pose:
    //
    // T_c2_c1 = T_c2w * T_c1w^-1, p_c2 = R * p_c1 + t
    // E = [t]_x * R
    // F = K^-T * E * K^-1
    //----------------------------------------------------------
    const CameraIntrinsic intrinsic = data_loader.get_intrinsic();
    const Mat3d K = intrinsic.K;
    const Mat3d K_inv = K.inverse();

    // Print the LiDAR frame each camera was matched to and its pose.
    std::cout
        << "[Diag] Camera 1 (id=" << camera_it_1->first
        << ") ts=" << camera_1.time_stamp_
        << " matched lidar_id=" << camera_1.matched_lidar_id_ << std::endl;
    std::cout
        << "[Diag] Camera 2 (id=" << camera_it_2->first
        << ") ts=" << camera_2.time_stamp_
        << " matched lidar_id=" << camera_2.matched_lidar_id_ << std::endl;

    const auto& lidar_map = data_loader.get_lidar_info_map();
    if (lidar_map.count(camera_1.matched_lidar_id_))
    {
        const auto& l1 = lidar_map.at(camera_1.matched_lidar_id_);
        std::cout << "[Diag] Lidar 1 ts=" << l1.time_stamp_
                  << "  cam1_ts-0.4=" << (camera_1.time_stamp_ - 0.4)
                  << "  delta=" << (camera_1.time_stamp_ - 0.4 - l1.time_stamp_)
                  << " s" << std::endl;
        std::cout << "[Diag] T_wl1:\n" << l1.initial_T_wl_.matrix() << std::endl;
    }
    if (lidar_map.count(camera_2.matched_lidar_id_))
    {
        const auto& l2 = lidar_map.at(camera_2.matched_lidar_id_);
        std::cout << "[Diag] Lidar 2 ts=" << l2.time_stamp_
                  << "  cam2_ts-0.4=" << (camera_2.time_stamp_ - 0.4)
                  << "  delta=" << (camera_2.time_stamp_ - 0.4 - l2.time_stamp_)
                  << " s" << std::endl;
        std::cout << "[Diag] T_wl2:\n" << l2.initial_T_wl_.matrix() << std::endl;
    }

    std::cout << "[Diag] T_cw1:\n" << camera_1.initial_T_cw_.matrix() << std::endl;
    std::cout << "[Diag] T_cw2:\n" << camera_2.initial_T_cw_.matrix() << std::endl;

    const SE3d T_c2_c1 =
        camera_2.initial_T_cw_ * camera_1.initial_T_cw_.inverse();
    const Mat3d R = T_c2_c1.rotationMatrix();
    const Vec3d t = T_c2_c1.translation();

    // Diagnose: if |t| is tiny the geometry is degenerate and F is useless.
    const double t_norm = t.norm();
    const double rotation_angle_deg =
        Eigen::AngleAxisd(R).angle() * 180.0 / M_PI;
    std::cout
        << "[Diag] Relative translation |t| = " << t_norm << " m"
        << ", rotation = " << rotation_angle_deg << " deg" << std::endl;

    if (t_norm < 0.01)
    {
        std::cout
            << "[Diag] WARNING: near-zero translation — F matrix is degenerate. "
               "Any match can pass the epipolar filter regardless of correctness."
            << std::endl;
    }

    Mat3d t_hat;
    t_hat << 0, -t.z(), t.y(),
             t.z(), 0, -t.x(),
            -t.y(), t.x(), 0;

    const Mat3d F = K_inv.transpose() * (t_hat * R) * K_inv;

    // Print epipolar error distribution over all ratio_matches so we can see
    // whether the threshold is meaningful or everything is near-zero (degenerate).
    {
        std::vector<double> errors;
        errors.reserve(ratio_matches.size());
        for (const cv::DMatch& m : ratio_matches)
        {
            const cv::Point2f& p1 = camera_1.keypoints_[m.queryIdx].pt;
            const cv::Point2f& p2 = camera_2.keypoints_[m.trainIdx].pt;
            Vec3d vp1(p1.x, p1.y, 1.0), vp2(p2.x, p2.y, 1.0);
            Vec3d l2 = F * vp1, l1 = F.transpose() * vp2;
            double d2 = std::hypot(l2.x(), l2.y());
            double d1 = std::hypot(l1.x(), l1.y());
            if (d2 < 1e-9 || d1 < 1e-9) { errors.push_back(1e9); continue; }
            double num = std::abs(vp2.dot(l2));
            errors.push_back(0.5 * (num / d2 + num / d1));
        }
        std::sort(errors.begin(), errors.end());
        const size_t n = errors.size();
        std::cout
            << "[Diag] Epipolar error over " << n << " ratio matches:"
            << " min=" << errors.front()
            << " p50=" << errors[n / 2]
            << " p90=" << errors[static_cast<size_t>(n * 0.9)]
            << " max=" << errors.back()
            << std::endl;
    }

    // Geometric verification via RANSAC (ground truth for "correct" matches).
    std::vector<cv::Point2f> pts1_all, pts2_all;
    for (const cv::DMatch& m : ratio_matches)
    {
        pts1_all.push_back(camera_1.keypoints_[m.queryIdx].pt);
        pts2_all.push_back(camera_2.keypoints_[m.trainIdx].pt);
    }
    std::vector<uchar> ransac_mask;
    cv::Mat F_ransac = cv::findFundamentalMat(
        pts1_all, pts2_all, cv::FM_RANSAC, 3.0, 0.99, ransac_mask);
    ASSERT_FALSE(F_ransac.empty()) << "RANSAC failed to estimate F.";

    std::vector<cv::DMatch> valid_matches;
    for (size_t i = 0; i < ratio_matches.size(); ++i)
    {
        if (ransac_mask[i])
            valid_matches.push_back(ratio_matches[i]);
    }

    std::cout
        << "[Diag] RANSAC F inliers: " << valid_matches.size()
        << " / " << ratio_matches.size() << std::endl;

    // Verify LIO-derived F against a few RANSAC inliers.
    // Also test F^T (swap cam1/cam2) to catch a pose-inversion bug.
    {
        const int n_check = std::min(5, static_cast<int>(valid_matches.size()));
        std::cout << "[Diag] Epipolar error for first " << n_check
                  << " RANSAC inliers under LIO F / F^T:" << std::endl;
        for (int i = 0; i < n_check; ++i)
        {
            const cv::Point2f& p1 = camera_1.keypoints_[valid_matches[i].queryIdx].pt;
            const cv::Point2f& p2 = camera_2.keypoints_[valid_matches[i].trainIdx].pt;
            Vec3d vp1(p1.x, p1.y, 1.0), vp2(p2.x, p2.y, 1.0);

            // F: p2^T F p1 = 0 (standard direction)
            Vec3d l2 = F * vp1;
            double err_F = std::abs(vp2.dot(l2)) /
                           std::hypot(l2.x(), l2.y());

            // F^T: p1^T F^T p2 = 0 (would be right if T_c2_c1 is inverted)
            Vec3d l1t = F.transpose() * vp2;
            double err_Ft = std::abs(vp1.dot(l1t)) /
                            std::hypot(l1t.x(), l1t.y());

            std::cout << "  match " << i
                      << ": F-err=" << err_F
                      << " px,  F^T-err=" << err_Ft << " px" << std::endl;
        }
    }

    // Print both F matrices and their implied epipoles so we can compare directly.
    {
        // LIO-derived F epipole: null-space of F (right null vector = epipole in img1)
        Eigen::JacobiSVD<Mat3d> svd_lio(F, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Vec3d e1_lio = svd_lio.matrixV().col(2);  // right null: epipole in image 1
        Vec3d e2_lio = svd_lio.matrixU().col(2);  // left null:  epipole in image 2
        e1_lio /= e1_lio.z();
        e2_lio /= e2_lio.z();

        // RANSAC F epipole
        Mat3d F_r;
        cv::cv2eigen(F_ransac, F_r);
        Eigen::JacobiSVD<Mat3d> svd_ransac(F_r, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Vec3d e1_r = svd_ransac.matrixV().col(2);
        Vec3d e2_r = svd_ransac.matrixU().col(2);
        e1_r /= e1_r.z();
        e2_r /= e2_r.z();

        // Singular values of LIO E (should be [s, s, 0] for a valid E).
        Mat3d E_lio = K.transpose() * F * K;
        Eigen::JacobiSVD<Mat3d> svd_e(E_lio);
        std::cout << "[Diag] LIO E singular values: "
                  << svd_e.singularValues().transpose() << std::endl;

        std::cout << "[Diag] LIO  epipole in img1: (" << e1_lio.x() << ", " << e1_lio.y() << ")"
                  << "  in img2: (" << e2_lio.x() << ", " << e2_lio.y() << ")" << std::endl;
        std::cout << "[Diag] RANSAC epipole in img1: (" << e1_r.x() << ", " << e1_r.y() << ")"
                  << "  in img2: (" << e2_r.x() << ", " << e2_r.y() << ")" << std::endl;
        std::cout << "[Diag] LIO F:\n" << F << std::endl;
        std::cout << "[Diag] RANSAC F:\n" << F_r << std::endl;
    }

    // Show how many of the RANSAC inliers also pass the LIO prior F filter.
    {
        int also_pass_lio = 0;
        for (const cv::DMatch& match : valid_matches)
        {
            const cv::Point2f& pt_1 = camera_1.keypoints_[match.queryIdx].pt;
            const cv::Point2f& pt_2 = camera_2.keypoints_[match.trainIdx].pt;
            Vec3d vp1(pt_1.x, pt_1.y, 1.0), vp2(pt_2.x, pt_2.y, 1.0);
            Vec3d l2 = F * vp1, l1 = F.transpose() * vp2;
            double d2 = std::hypot(l2.x(), l2.y());
            double d1 = std::hypot(l1.x(), l1.y());
            if (d2 < 1e-9 || d1 < 1e-9) continue;
            double err = 0.5 * (std::abs(vp2.dot(l2)) / d2 + std::abs(vp1.dot(l1)) / d1);
            if (err <= 3.0) ++also_pass_lio;
        }
        std::cout
            << "[Diag] Of " << valid_matches.size()
            << " RANSAC inliers, " << also_pass_lio
            << " also pass the LIO prior epipolar filter (3 px)." << std::endl;
    }

    ASSERT_FALSE(valid_matches.empty())
        << "No matches survived RANSAC geometric verification.";

    //----------------------------------------------------------
    // Visualize: images side by side, each surviving match drawn
    // with its own random color line/circles.
    //----------------------------------------------------------
    const cv::Mat image_1 = cv::imread(camera_1.camera_path_);
    const cv::Mat image_2 = cv::imread(camera_2.camera_path_);
    ASSERT_FALSE(image_1.empty());
    ASSERT_FALSE(image_2.empty());

    cv::Mat visualization(
        std::max(image_1.rows, image_2.rows),
        image_1.cols + image_2.cols,
        image_1.type(),
        cv::Scalar::all(0));

    image_1.copyTo(visualization(cv::Rect(0, 0, image_1.cols, image_1.rows)));
    image_2.copyTo(visualization(
        cv::Rect(image_1.cols, 0, image_2.cols, image_2.rows)));

    const double x_offset = static_cast<double>(image_1.cols);

    cv::RNG rng(12345);

    for (const cv::DMatch& match : valid_matches)
    {
        const cv::Point2f& pt_1 = camera_1.keypoints_[match.queryIdx].pt;
        const cv::Point2f& pt_2 = camera_2.keypoints_[match.trainIdx].pt;

        const cv::Point2f shifted_pt_2(
            pt_2.x + static_cast<float>(x_offset),
            pt_2.y);

        const cv::Scalar color(
            rng.uniform(0, 256),
            rng.uniform(0, 256),
            rng.uniform(0, 256));

        cv::line(visualization, pt_1, shifted_pt_2, color, 1, cv::LINE_AA);
        cv::circle(visualization, pt_1, 3, color, -1, cv::LINE_AA);
        cv::circle(visualization, shifted_pt_2, 3, color, -1, cv::LINE_AA);
    }

    const fs::path output_dir = fs::path(PROJECT_ROOT_DIR) / "output";
    fs::create_directories(output_dir);

    const fs::path output_path =
        output_dir / "epipolar_prior_pose_filter.png";
    ASSERT_TRUE(cv::imwrite(output_path.string(), visualization));

    std::cout
        << "Priori-pose epipolar filter visualization saved to: "
        << output_path << std::endl;
}

TEST(SIFTTest, LandmarkTrackerUsingPriorPose)
{
    const auto test_start = std::chrono::steady_clock::now();

    const std::string config_path =
        std::string(PROJECT_ROOT_DIR) + "/src/config.yaml";
    Config config(config_path);

    DataLoader data_loader(config);
    ASSERT_TRUE(data_loader.load().success);

    std::map<int, Camera>& camera_map = data_loader.get_camera_map();
    const CameraIntrinsic intrinsic = data_loader.get_intrinsic();
    const Mat3d& K = intrinsic.K;

    auto is_valid = [](const std::pair<const int, Camera>& entry) {
        return entry.second.matched_lidar_id_ >= 0;
    };

    std::vector<std::map<int, Camera>::iterator> valid_cameras;
    for (auto it = camera_map.begin(); it != camera_map.end(); ++it)
    {
        if (is_valid(*it))
        {
            ASSERT_TRUE(SIFT::extract_sift(it->second, config).success);
            ASSERT_FALSE(it->second.descriptors_.empty());
            valid_cameras.push_back(it);
        }
    }

    ASSERT_GE(valid_cameras.size(), 2u)
        << "Need at least two cameras with valid LiDAR associations.";

    const double epipolar_threshold_px = 2.5;
    const double min_parallax_deg = 0.5;
    const double max_reprojection_error_px = 4.0;
    const size_t max_camera_pair_gap = 3;
    const size_t min_epipolar_matches_per_pair = 40;
    const size_t min_usable_matches_per_pair = 30;
    const size_t max_matches_per_pair_for_geometry = 700;

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
                               const Mat3d& F)
        -> std::vector<cv::DMatch>
    {
        std::vector<cv::DMatch> kept;
        std::vector<std::vector<cv::DMatch>> knn_matches;
        SIFT::knn_matching(cam_1.descriptors_, cam_2.descriptors_, knn_matches);

        std::vector<cv::DMatch> ratio_matches;
        SIFT::lowe_ratio_test(
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

    struct DepthSample {
        double triangulated_depth = 0.0;
        double parallax_deg = 0.0;
        double reprojection_error_1_px = 0.0;
        double reprojection_error_2_px = 0.0;
        cv::Point2f pixel_1;
        cv::Point2f pixel_2;
        cv::DMatch match;
    };

    auto collect_depth_samples = [&](const Camera& cam_1,
                                     const Camera& cam_2,
                                     const std::vector<cv::DMatch>& matches) -> std::vector<DepthSample>
    {
        std::vector<DepthSample> samples;
        if (matches.empty())
        {
            return samples;
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

            samples.push_back({
                p_c1.z(),
                parallax_deg,
                reproj_err_1,
                reproj_err_2,
                pts_1[i],
                pts_2[i],
                matches[i]
            });
        }

        return samples;
    };

    struct PairResult {
        std::map<int, Camera>::iterator cam_1;
        std::map<int, Camera>::iterator cam_2;
        std::vector<cv::DMatch> filtered_matches;
        std::vector<cv::DMatch> usable_matches;
        std::vector<DepthSample> samples;
        int usable_samples = 0;
    };

    std::vector<PairResult> accepted_pairs;
    int evaluated_pairs = 0;
    int total_filtered_matches = 0;
    int total_kept_for_geometry = 0;
    int total_usable_matches = 0;

    for (size_t i = 0; i + 1 < valid_cameras.size(); ++i)
    {
        for (size_t j = i + 1; j < valid_cameras.size(); ++j)
        {
            if (j - i > max_camera_pair_gap)
            {
                continue;
            }

            ++evaluated_pairs;
            Camera& cam_1 = valid_cameras[i]->second;
            Camera& cam_2 = valid_cameras[j]->second;
            const Mat3d F = build_fundamental(cam_1.initial_T_cw_, cam_2.initial_T_cw_);
            const std::vector<cv::DMatch> filtered_matches =
                epipolar_filter(cam_1, cam_2, F);

            if (filtered_matches.size() < min_epipolar_matches_per_pair)
            {
                continue;
            }

            std::vector<cv::DMatch> kept_for_geometry = filtered_matches;
            if (kept_for_geometry.size() > max_matches_per_pair_for_geometry)
            {
                std::partial_sort(
                    kept_for_geometry.begin(),
                    kept_for_geometry.begin() + max_matches_per_pair_for_geometry,
                    kept_for_geometry.end(),
                    [](const cv::DMatch& a, const cv::DMatch& b) {
                        return a.distance < b.distance;
                    });
                kept_for_geometry.resize(max_matches_per_pair_for_geometry);
            }

            const std::vector<DepthSample> samples =
                collect_depth_samples(cam_1, cam_2, kept_for_geometry);

            const int usable = static_cast<int>(samples.size());
            if (usable < static_cast<int>(min_usable_matches_per_pair))
            {
                continue;
            }

            PairResult pair_result;
            pair_result.cam_1 = valid_cameras[i];
            pair_result.cam_2 = valid_cameras[j];
            pair_result.filtered_matches = filtered_matches;
            pair_result.samples = samples;
            pair_result.usable_samples = usable;
            pair_result.usable_matches.reserve(samples.size());
            for (const DepthSample& sample : samples)
            {
                pair_result.usable_matches.push_back(sample.match);
            }

            total_filtered_matches += static_cast<int>(filtered_matches.size());
            total_kept_for_geometry += static_cast<int>(kept_for_geometry.size());
            total_usable_matches += usable;
            accepted_pairs.push_back(std::move(pair_result));
        }
    }

    ASSERT_FALSE(accepted_pairs.empty())
        << "No camera pair produced any triangulations that survived the prior-pose geometric filters.";

    std::map<FeatureNode, FeatureNode> parent;
    std::map<FeatureNode, int> rank;
    std::map<int, std::vector<FeatureNode>> camera_feature_nodes;
    for (const PairResult& pair_result : accepted_pairs)
    {
        std::vector<uchar> inlier_mask(pair_result.usable_matches.size(), 1);
        SIFT::union_matches(
            pair_result.cam_1->first,
            pair_result.cam_1->second,
            pair_result.cam_2->first,
            pair_result.cam_2->second,
            pair_result.usable_matches,
            inlier_mask,
            parent,
            rank,
            camera_feature_nodes,
            config.feature_track_merge_radius_px_);
    }

    std::map<int, Landmark> landmarks;
    SIFT::extract_landmark_map(parent, camera_map, landmarks);

    std::map<int, int> obs_hist;
    int landmarks_ge_3_obs = 0;
    for (const auto& [landmark_id, landmark] : landmarks)
    {
        (void)landmark_id;
        const int obs_count = static_cast<int>(landmark.observations_.size());
        obs_hist[obs_count] += 1;
        if (obs_count >= 3)
        {
            ++landmarks_ge_3_obs;
        }
    }

    auto best_pair_it = std::max_element(
        accepted_pairs.begin(),
        accepted_pairs.end(),
        [](const PairResult& a, const PairResult& b) {
            return a.usable_samples < b.usable_samples;
        });

    std::cout
        << "[PriorTracker] evaluated_pairs=" << evaluated_pairs
        << " accepted_pairs=" << accepted_pairs.size()
        << " best_pair=" << best_pair_it->cam_1->first
        << "<->" << best_pair_it->cam_2->first
        << " best_pair_usable_samples=" << best_pair_it->usable_samples
        << " total_filtered_matches=" << total_filtered_matches
        << " total_kept_for_geometry=" << total_kept_for_geometry
        << " total_usable_matches=" << total_usable_matches
        << " landmarks=" << landmarks.size() << std::endl;

    std::cout << "[PriorTracker] observation_histogram:";
    for (const auto& [obs_count, count] : obs_hist)
    {
        std::cout << " " << obs_count << "->" << count;
    }
    std::cout
        << " | landmarks_with_3plus_obs=" << landmarks_ge_3_obs
        << std::endl;

    ASSERT_FALSE(landmarks.empty())
        << "No landmarks were formed from prior-pose epipolar-filtered matches.";
    ASSERT_GT(landmarks_ge_3_obs, 0)
        << "No landmarks reached 3+ observations."
        << " Try relaxing geometry thresholds or increasing overlap.";

    std::map<int, Landmark> landmarks_for_export;
    for (const auto& [landmark_id, landmark] : landmarks)
    {
        if (landmark.observations_.size() > 3)
        {
            landmarks_for_export.emplace(landmark_id, landmark);
        }
    }

    ASSERT_FALSE(landmarks_for_export.empty())
        << "No landmarks have more than 3 observations for export.";

    const fs::path output_dir =
        fs::path(PROJECT_ROOT_DIR) / "output" / "prior_pose_landmark_track_images";
    Status export_status = SIFT::export_landmark_track_images(
        landmarks_for_export,
        camera_map,
        output_dir.string());
    ASSERT_TRUE(export_status.success) << export_status.message;

    std::cout << "[PriorTracker] " << export_status.message << std::endl;

    const double elapsed_s = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - test_start).count();
    std::cout << "[PriorTracker] elapsed_seconds=" << elapsed_s << std::endl;
}

// ============================================================
// Triangulation depth evaluation
//
// Workflow:
// 1. SIFT detect and match
// 2. Filter matches with epipolar error from initial T_cw
// 3. Triangulate surviving matches with the two camera poses
// 4. Compare the triangulated depth against the LiDAR depth map
// ============================================================

TEST(TriangulationTest, DepthVsLidar)
{
    const std::string config_path =
        std::string(PROJECT_ROOT_DIR) + "/src/config.yaml";
    Config config(config_path);

    DataLoader data_loader(config);
    ASSERT_TRUE(data_loader.load().success);

    std::map<int, Camera>& camera_map = data_loader.get_camera_map();
    const CameraIntrinsic intrinsic = data_loader.get_intrinsic();
    const Mat3d& K = intrinsic.K;

    auto is_valid = [](const std::pair<const int, Camera>& entry) {
        return entry.second.matched_lidar_id_ >= 0;
    };

    std::vector<std::map<int, Camera>::iterator> valid_cameras;
    for (auto it = camera_map.begin(); it != camera_map.end(); ++it)
    {
        if (is_valid(*it))
        {
            ASSERT_TRUE(SIFT::extract_sift(it->second, config).success);
            ASSERT_FALSE(it->second.descriptors_.empty());
            valid_cameras.push_back(it);
        }
    }

    ASSERT_GE(valid_cameras.size(), 2u)
        << "Need at least two cameras with valid LiDAR associations.";

    const double epipolar_threshold_px = 2.5;
    const double min_parallax_deg = 0.5;
    const double max_reprojection_error_px = 4.0;

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
                               const Mat3d& F)
        -> std::vector<cv::DMatch>
    {
        std::vector<cv::DMatch> kept;
        std::vector<std::vector<cv::DMatch>> knn_matches;
        SIFT::knn_matching(cam_1.descriptors_, cam_2.descriptors_, knn_matches);

        std::vector<cv::DMatch> ratio_matches;
        SIFT::lowe_ratio_test(
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

    struct DepthSample {
        double triangulated_depth = 0.0;
        double parallax_deg = 0.0;
        double reprojection_error_1_px = 0.0;
        double reprojection_error_2_px = 0.0;
        cv::Point2f pixel_1;
        cv::Point2f pixel_2;
    };

    auto collect_depth_samples = [&](const Camera& cam_1,
                                     const Camera& cam_2,
                                     const std::vector<cv::DMatch>& matches) -> std::vector<DepthSample>
    {
        std::vector<DepthSample> samples;
        if (matches.empty())
        {
            return samples;
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
        samples.reserve(matches.size());

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

            if (!std::isfinite(p_c1.z()) || !std::isfinite(p_c2.z()))
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

            samples.push_back({
                p_c1.z(),
                parallax_deg,
                reproj_err_1,
                reproj_err_2,
                pts_1[i],
                pts_2[i]
            });
        }

        return samples;
    };

    auto render_pair_visualization = [&](const Camera& cam_1,
                                         const Camera& cam_2,
                                         const std::vector<DepthSample>& samples,
                                         const fs::path& output_path,
                                         size_t epipolar_match_count)
    {
        const cv::Mat image_1 = cv::imread(cam_1.camera_path_);
        const cv::Mat image_2 = cv::imread(cam_2.camera_path_);
        if (image_1.empty() || image_2.empty())
        {
            return;
        }

        cv::Mat visualization(
            std::max(image_1.rows, image_2.rows),
            image_1.cols + image_2.cols,
            image_1.type(),
            cv::Scalar::all(0));

        image_1.copyTo(visualization(cv::Rect(0, 0, image_1.cols, image_1.rows)));
        image_2.copyTo(visualization(cv::Rect(image_1.cols, 0, image_2.cols, image_2.rows)));

        const double x_offset = static_cast<double>(image_1.cols);
        std::vector<double> depths;
        depths.reserve(samples.size());
        for (const DepthSample& sample : samples)
        {
            depths.push_back(sample.triangulated_depth);
        }

        const double depth_min = depths.empty() ? 0.0 : *std::min_element(depths.begin(), depths.end());
        const double depth_max = depths.empty() ? 1.0 : *std::max_element(depths.begin(), depths.end());

        for (const DepthSample& sample : samples)
        {
            const double denom = std::max(1e-9, depth_max - depth_min);
            const double norm = (sample.triangulated_depth - depth_min) / denom;
            const int blue = static_cast<int>(255.0 * (1.0 - norm));
            const int red = static_cast<int>(255.0 * norm);
            const cv::Scalar color(blue, 0, red);

            const cv::Point2f pt_2_shifted(
                sample.pixel_2.x + static_cast<float>(x_offset),
                sample.pixel_2.y);

            cv::circle(visualization, sample.pixel_1, 4, color, -1, cv::LINE_AA);
            cv::circle(visualization, pt_2_shifted, 4, color, -1, cv::LINE_AA);
            cv::line(visualization, sample.pixel_1, pt_2_shifted, color, 1, cv::LINE_AA);

            std::ostringstream depth_text;
            depth_text.setf(std::ios::fixed);
            depth_text.precision(2);
            depth_text << sample.triangulated_depth << " m";

            const cv::Point text_pos(
                static_cast<int>(sample.pixel_1.x) + 5,
                static_cast<int>(sample.pixel_1.y) - 5);

            cv::putText(
                visualization,
                depth_text.str(),
                text_pos,
                cv::FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1,
                cv::LINE_AA);
        }

        std::ostringstream summary;
        summary << "epipolar=" << epipolar_match_count
                << " triangulated=" << samples.size();
        cv::putText(
            visualization,
            summary.str(),
            cv::Point(20, 30),
            cv::FONT_HERSHEY_SIMPLEX,
            0.8,
            cv::Scalar(0, 255, 255),
            2,
            cv::LINE_AA);

        cv::imwrite(output_path.string(), visualization);
    };

    struct PairResult {
        std::map<int, Camera>::iterator cam_1;
        std::map<int, Camera>::iterator cam_2;
        Mat3d F;
        std::vector<cv::DMatch> filtered_matches;
        std::vector<DepthSample> samples;
        int usable_samples = 0;
    };

    PairResult best_pair;
    best_pair.usable_samples = -1;
    const fs::path pair_output_dir =
        fs::path(PROJECT_ROOT_DIR) / "output" / "triangulation_pair_images";
    fs::create_directories(pair_output_dir);

    for (size_t i = 0; i + 1 < valid_cameras.size(); ++i)
    {
        for (size_t j = i + 1; j < valid_cameras.size(); ++j)
        {
            if(j - i > 4)
                continue;
            auto cam_1_it = valid_cameras[i];
            auto cam_2_it = valid_cameras[j];
            Camera& cam_1 = cam_1_it->second;
            Camera& cam_2 = cam_2_it->second;

            const Mat3d F = build_fundamental(cam_1.initial_T_cw_, cam_2.initial_T_cw_);
            const std::vector<cv::DMatch> filtered_matches =
                epipolar_filter(cam_1, cam_2, F);

            if (filtered_matches.empty())
            {
                continue;
            }

            const std::vector<DepthSample> samples =
                collect_depth_samples(cam_1, cam_2, filtered_matches);
            const int usable = static_cast<int>(samples.size());

            const fs::path pair_output_path =
                pair_output_dir /
                (std::string("pair_")
                    + std::to_string(cam_1_it->first)
                    + "_"
                    + std::to_string(cam_2_it->first)
                    + ".png");
            render_pair_visualization(
                cam_1,
                cam_2,
                samples,
                pair_output_path,
                filtered_matches.size());

            if (usable <= 0)
            {
                continue;
            }

            if (usable > best_pair.usable_samples)
            {
                best_pair.cam_1 = cam_1_it;
                best_pair.cam_2 = cam_2_it;
                best_pair.F = F;
                best_pair.filtered_matches = filtered_matches;
                best_pair.samples = samples;
                best_pair.usable_samples = usable;
            }
        }
    }

    ASSERT_GT(best_pair.usable_samples, 0)
        << "No camera pair produced any triangulations that survived epipolar, cheirality, parallax, and reprojection filtering.";

    Camera& cam_1 = best_pair.cam_1->second;
    Camera& cam_2 = best_pair.cam_2->second;

    std::cout
        << "Best pair: camera_id=" << best_pair.cam_1->first
        << " (lidar " << cam_1.matched_lidar_id_ << ")"
        << " <-> camera_id=" << best_pair.cam_2->first
        << " (lidar " << cam_2.matched_lidar_id_ << ")"
        << " | epipolar matches=" << best_pair.filtered_matches.size()
        << " | usable triangulated samples=" << best_pair.usable_samples
        << std::endl;

    ASSERT_FALSE(best_pair.filtered_matches.empty())
        << "No matches survived the prior-pose epipolar filter.";

    std::vector<DepthSample> samples = best_pair.samples;
    const int rejected_total = static_cast<int>(best_pair.filtered_matches.size()) -
                               static_cast<int>(samples.size());

    ASSERT_FALSE(samples.empty())
        << "No triangulated points survived cheirality, parallax, and reprojection filtering.";

    const size_t n = samples.size();
    std::vector<double> triangulated_depths;
    std::vector<double> parallax_values;
    std::vector<double> reprojection_errors;
    triangulated_depths.reserve(n);
    parallax_values.reserve(n);
    reprojection_errors.reserve(n);
    for (const DepthSample& sample : samples)
    {
        triangulated_depths.push_back(sample.triangulated_depth);
        parallax_values.push_back(sample.parallax_deg);
        reprojection_errors.push_back(
            0.5 * (sample.reprojection_error_1_px + sample.reprojection_error_2_px));
    }

    std::sort(triangulated_depths.begin(), triangulated_depths.end());
    std::sort(parallax_values.begin(), parallax_values.end());
    std::sort(reprojection_errors.begin(), reprojection_errors.end());

    const double mean_depth = std::accumulate(triangulated_depths.begin(), triangulated_depths.end(), 0.0) /
                              static_cast<double>(n);
    const double mean_parallax = std::accumulate(parallax_values.begin(), parallax_values.end(), 0.0) /
                                 static_cast<double>(n);
    const double mean_reprojection = std::accumulate(reprojection_errors.begin(), reprojection_errors.end(), 0.0) /
                                     static_cast<double>(n);

    std::cout << "=== Triangulation depth evaluation (" << n << " samples) ===" << std::endl;
    std::cout << "  Rejected after triangulation filtering: " << rejected_total << std::endl;
    std::cout << "  Triangulated depth: mean=" << mean_depth
              << "  p50=" << triangulated_depths[n / 2]
              << "  p90=" << triangulated_depths[static_cast<size_t>(n * 0.9)] << " m" << std::endl;
    std::cout << "  Parallax angle:     mean=" << mean_parallax
              << "  p50=" << parallax_values[n / 2]
              << "  p90=" << parallax_values[static_cast<size_t>(n * 0.9)] << " deg" << std::endl;
    std::cout << "  Reprojection err:   mean=" << mean_reprojection
              << "  p50=" << reprojection_errors[n / 2]
              << "  p90=" << reprojection_errors[static_cast<size_t>(n * 0.9)] << " px" << std::endl;

    //----------------------------------------------------------
    // Visualization: side-by-side image pair with surviving
    // matches colored by triangulated depth.
    //----------------------------------------------------------
    const cv::Mat image_1 = cv::imread(cam_1.camera_path_);
    const cv::Mat image_2 = cv::imread(cam_2.camera_path_);
    ASSERT_FALSE(image_1.empty());
    ASSERT_FALSE(image_2.empty());

    cv::Mat visualization(
        std::max(image_1.rows, image_2.rows),
        image_1.cols + image_2.cols,
        image_1.type(),
        cv::Scalar::all(0));

    image_1.copyTo(visualization(cv::Rect(0, 0, image_1.cols, image_1.rows)));
    image_2.copyTo(visualization(cv::Rect(image_1.cols, 0, image_2.cols, image_2.rows)));

    const double x_offset = static_cast<double>(image_1.cols);
    const double depth_max = *std::max_element(triangulated_depths.begin(), triangulated_depths.end());
    const double depth_min = *std::min_element(triangulated_depths.begin(), triangulated_depths.end());
    for (const DepthSample& sample : samples)
    {
        const double denom = std::max(1e-9, depth_max - depth_min);
        const double norm = (sample.triangulated_depth - depth_min) / denom;
        const int blue = static_cast<int>(255.0 * (1.0 - norm));
        const int red = static_cast<int>(255.0 * norm);
        const cv::Scalar color(blue, 0, red);

        const cv::Point2f pt_2_shifted(
            sample.pixel_2.x + static_cast<float>(x_offset),
            sample.pixel_2.y);

        cv::circle(visualization, sample.pixel_1, 4, color, -1, cv::LINE_AA);
        cv::circle(visualization, pt_2_shifted, 4, color, -1, cv::LINE_AA);
        cv::line(visualization, sample.pixel_1, pt_2_shifted, color, 1, cv::LINE_AA);

        std::ostringstream depth_text;
        depth_text.setf(std::ios::fixed);
        depth_text.precision(2);
        depth_text << sample.triangulated_depth << " m";

        const cv::Point text_pos(
            static_cast<int>(sample.pixel_1.x) + 5,
            static_cast<int>(sample.pixel_1.y) - 5);

        cv::putText(
            visualization,
            depth_text.str(),
            text_pos,
            cv::FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1,
            cv::LINE_AA);
    }

    const fs::path output_dir = fs::path(PROJECT_ROOT_DIR) / "output";
    fs::create_directories(output_dir);

    const fs::path output_path = output_dir / "triangulation_depth_eval.png";
    ASSERT_TRUE(cv::imwrite(output_path.string(), visualization));

    std::cout << "Depth evaluation visualization saved to: " << output_path << std::endl;
}
