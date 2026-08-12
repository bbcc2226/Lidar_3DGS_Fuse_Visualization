//
// Created by chuchu on 8/5/26.
//

#include "BaHelper.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <fstream>
#include <filesystem>
using namespace std;

namespace {
// Weiszfeld's algorithm: iteratively reweighted average that converges to the
// point minimizing sum of distances, making it robust to outlier points
// (unlike the arithmetic mean, which minimizes sum of squared distances).
Vec3d geometric_median(const std::vector<Vec3d>& points, int max_iters = 50, double eps = 1e-6){
   if(points.size() == 1){
      return points.front();
   }

   Vec3d median = Vec3d::Zero();
   for(const auto& p : points) median += p;
   median /= static_cast<double>(points.size());

   for(int iter = 0; iter < max_iters; ++iter){
      Vec3d numerator = Vec3d::Zero();
      double denominator = 0.0;
      for(const auto& p : points){
         double dist = (p - median).norm();
         if(dist < 1e-9) continue; // avoid divide-by-zero when a point coincides with the current estimate
         numerator += p / dist;
         denominator += 1.0 / dist;
      }
      if(denominator < 1e-9) break;

      Vec3d next = numerator / denominator;
      bool converged = (next - median).norm() < eps;
      median = next;
      if(converged) break;
   }
   return median;
}
}

Status BaHelper::load_projected_depth(Camera& camera){  
  return {true, "depth load ok!"};
}

Vec3d BaHelper::PixelToCamera(const Vec2d& pixel,double depth,const Mat3d& K){
  double fx = K(0,0);
  double fy = K(1,1);

  double cx = K(0,2);
  double cy = K(1,2);

  double x =
        (pixel.x() - cx) / fx;

  double y =
        (pixel.y() - cy) / fy;

  return Vec3d(
        x * depth,
        y * depth,
        depth);
}

Status BaHelper::extract_landmark_world_pos(const std::map<int ,Camera>& camera_map, const CameraIntrinsic& intrinsic, const Config& config_, std::map<int, Landmark>&landmarks,  bool is_initial){
   //get camera intrinsic
   int removed_observations = 0;
   int removed_landmarks = 0;
   int removed_outlier_observations = 0;

   for(auto lm_it = landmarks.begin(); lm_it != landmarks.end(); ){
      Landmark& lm = lm_it->second;

      // Drop observations with no valid depth association instead of just
      // skipping them here - they're useless for triangulation on every
      // future pass too.
      const size_t before = lm.observations_.size();
      lm.observations_.erase(
         std::remove_if(lm.observations_.begin(), lm.observations_.end(),
            [is_initial](const Observation& ob){
               const double depth = is_initial ? ob.depth_ : ob.optimized_depth_;
               return depth <= 0.0;
            }),
         lm.observations_.end());
      removed_observations += static_cast<int>(before - lm.observations_.size());

      if(lm.observations_.size() < 2){
         lm_it = landmarks.erase(lm_it);
         ++removed_landmarks;
         continue;
      }

      std::vector<Vec3d> world_points;
      world_points.reserve(lm.observations_.size());

      for(auto& ob : lm.observations_){
         //for each pixel, camera<-pixel, world<-camera
         const double depth = is_initial ? ob.depth_ : ob.optimized_depth_;

         Vec3d cam_cor = PixelToCamera(ob.pixel_,depth,intrinsic.K);

         //for orb, get camera cvt to world coordinate
         if(camera_map.find(ob.camera_id_)==camera_map.end()){
            std::cout<<"obj:" << ob.camera_id_<<" "<< "can not find camera: " << ob.camera_id_ << std::endl;
            continue;
         }
         const Camera& cam = camera_map.at(ob.camera_id_);
         const SE3d& T_cw = is_initial ? cam.initial_T_cw_ : cam.optimized_T_cw_;
         SE3d T_wc = T_cw.inverse();
         //comvert it to world coordinate
         world_points.push_back(T_wc * cam_cor);
      }

      if(world_points.empty()){
         ++lm_it;
         continue;
      }

      // Geometric median instead of mean: robust to outlier observations
      // (e.g. LiDAR depth noise near edges) since it minimizes sum of
      // distances rather than squared distances.
      Vec3d pos_w_medium = geometric_median(world_points);

      // Reproject the triangulated point into every observing camera and
      // drop observations whose pixel error is too large - these are
      // likely bad feature matches that would otherwise corrupt the BA.
      const double reproj_threshold = config_.landmark_reprojection_error_threshold_;
      const size_t before_reproj = lm.observations_.size();
      lm.observations_.erase(
         std::remove_if(lm.observations_.begin(), lm.observations_.end(),
            [&](const Observation& ob){
               auto camera_it = camera_map.find(ob.camera_id_);
               if(camera_it == camera_map.end()){
                  return false;
               }
               const SE3d& T_cw = is_initial ? camera_it->second.initial_T_cw_ : camera_it->second.optimized_T_cw_;
               const Vec3d point_c = T_cw * pos_w_medium;
               if(point_c.z() <= 0.0){
                  return true; // behind the camera, can't be a real observation
               }
               const double u = intrinsic.fx() * point_c.x() / point_c.z() + intrinsic.cx();
               const double v = intrinsic.fy() * point_c.y() / point_c.z() + intrinsic.cy();
               const double du = u - ob.pixel_.x();
               const double dv = v - ob.pixel_.y();
               return std::sqrt(du * du + dv * dv) > reproj_threshold;
            }),
         lm.observations_.end());
      removed_outlier_observations += static_cast<int>(before_reproj - lm.observations_.size());

      if(lm.observations_.size() < 2){
         lm_it = landmarks.erase(lm_it);
         ++removed_landmarks;
         continue;
      }

      if(is_initial){
       lm.initial_position_ = pos_w_medium;
      }else{
       lm.optimized_ = true;
       lm.optimized_position_ = pos_w_medium;
      }

      std::cout << "[extract_landmark_world_pos] landmark " << lm_it->first
                 << " has " << lm.observations_.size() << " observations" << std::endl;

      ++lm_it;
   }// loop end, land mark

   if(removed_observations > 0 || removed_landmarks > 0 || removed_outlier_observations > 0){
      std::cout << "[extract_landmark_world_pos] removed " << removed_observations
                 << " depth-less observations, " << removed_outlier_observations
                 << " reprojection outlier observations, dropped " << removed_landmarks
                 << " landmarks left with < 2 observations" << std::endl;
   }

   std::cout << "[extract_landmark_world_pos] " << landmarks.size()
              << " valid landmarks remaining after filtering" << std::endl;

   return {true, "OK"};
}


Status  BaHelper::writeOptimziedCamera(std::map<int ,Camera>& camera_map, Config& config){
   const std::string& output_path = config.output_path_;

   std::filesystem::path path(output_path);
   if(path.has_parent_path()){
      std::filesystem::create_directories(path.parent_path());
   }

   std::ofstream fout(output_path);
   if(!fout.is_open()){
      return {false, "Cannot open output file for writing: " + output_path};
   }

   fout << "# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, IMAGE_NAME\n";

   int written_count = 0;
   for(const auto& [camera_id, camera] : camera_map){
      // COLMAP's images.txt stores the world->camera transform directly
      // (T_cw), unlike save_trajectory()'s T_wc used for TUM-style plotting.
      const SE3d& T_cw = camera.optimized_T_cw_;
      const Vec3d t = T_cw.translation();
      const Eigen::Quaterniond q = T_cw.unit_quaternion();

      // COLMAP IDs are 1-indexed; this project uses a single shared camera
      // model, hence CAMERA_ID is always 1.
      fout << (camera_id + 1) << " "
           << q.w() << " " << q.x() << " " << q.y() << " " << q.z() << " "
           << t.x() << " " << t.y() << " " << t.z() << " "
           << 1 << " "
           << camera.camera_name_ << "\n";
      // COLMAP's images.txt has a second (POINTS2D) line per image; leave it
      // blank since we don't track 2D-3D correspondences here.
      fout << "\n";
      ++written_count;
   }

   fout.close();
   return {true, "Wrote " + std::to_string(written_count) + " optimized camera poses to " + output_path};
}

Status BaHelper::save_trajectory(const std::map<int ,Camera>& camera_map, const std::string& output_path, bool use_optimized){
   std::filesystem::path path(output_path);
   if(path.has_parent_path()){
      std::filesystem::create_directories(path.parent_path());
   }

   std::ofstream fout(output_path);
   if(!fout.is_open()){
      return {false, "Cannot open trajectory file for writing: " + output_path};
   }

   for(const auto& [camera_id, camera] : camera_map){
      const SE3d& T_cw = use_optimized ? camera.optimized_T_cw_ : camera.initial_T_cw_;
      const SE3d T_wc = T_cw.inverse();
      const Vec3d t = T_wc.translation();
      const Eigen::Quaterniond q = T_wc.unit_quaternion();

      fout << camera.time_stamp_ << " "
           << t.x() << " " << t.y() << " " << t.z() << " "
           << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << "\n";
   }

   fout.close();
   return {true, "Trajectory written to " + output_path};
}

Status BaHelper::update_observation_depth(const std::map<int ,Camera>& camera_map, std::map<int, Landmark>& landmarks, bool is_initial){
   for(auto& [lm_id, lm] : landmarks){
      for(auto& ob : lm.observations_){
         if(camera_map.find(ob.camera_id_) == camera_map.end()){
            std::cout<<"obj:" << ob.camera_id_<<" "<< "can not find camera: " << ob.camera_id_ << std::endl;
            continue;
         }
         const Camera& cam = camera_map.at(ob.camera_id_);
         if(cam.depth_map_.empty()){
            continue;
         }

         if(is_initial && ob.depth_ > 0.0){
            continue;
         }

         const int u = static_cast<int>(std::round(ob.pixel_.x()));
         const int v = static_cast<int>(std::round(ob.pixel_.y()));
         if(u < 0 || u >= cam.depth_map_.cols || v < 0 || v >= cam.depth_map_.rows){
            continue;
         }

         const float depth = cam.depth_map_.at<float>(v, u);
         if(depth <= 0.0f){
            continue;
         }

         if(is_initial){
            ob.depth_ = static_cast<double>(depth);
         }else{
            ob.optimized_depth_ = static_cast<double>(depth);
            ob.optimized_ = true;
         }
      }
   }
   return {true, "observation depth updated"};
}








