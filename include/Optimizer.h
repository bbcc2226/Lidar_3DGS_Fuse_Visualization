#pragma once
#include <map>
#include <string>
#include "General.h"

class BundleAdjustmentOptimizer
{
private:

struct ReprojectionError;
struct PosePriorError;
struct DepthError;

std::map<int, Camera> camera_map_;
std::map<int, Landmark> landmark_map_;
Mat3d K_;
// se3 tangent parameter blocks keyed by camera_id / landmark_id
std::map<int, Vec6d> camera_poses_;
std::map<int, Vec3d> landmark_positions_;
// weight for the PosePriorError keeping each pose close to its initial value
double pose_prior_weight_ = 40.0;
// weight for the DepthError anchoring landmark depth to the LiDAR depth map
double depth_prior_weight_ = 1;
// max iterations for each Ceres solve
int max_num_iterations_ = 10;
// final cost of the most recent Optimize() call
double last_final_cost_ = 0.0;

public:
    BundleAdjustmentOptimizer() = default;
    ~BundleAdjustmentOptimizer() = default;

    // Input data
    void SetCameraMap(const std::map<int, Camera>& camera_map);

    void SetLandmarkMap(const std::map<int, Landmark>& landmark_map);

    void SetIntrinsicMatrix(const Mat3d& K);

    void SetPosePriorWeight(double weight);

    void SetDepthPriorWeight(double weight);

    void SetMaxIterations(int max_iterations);

    // Final cost reported by Ceres for the most recent Optimize() call.
    double GetFinalCost() const;

    // Run bundle adjustment
    Status Optimize(const bool initialized);

    // Optional helper if caller only wants optimized poses
    void GetOptimizedPoses(
        std::map<int, SE3d>& optimized_poses) const;

    // Optional helper if caller only wants optimized landmark positions
    void GetOptimizedLandmarks(
        std::map<int, Vec3d>& optimized_landmarks) const;
    
    void GetCameraMap(std::map<int, Camera>& camera_map) const;

    void GetLandmarkMap(std::map<int, Landmark>& landmark_map) const;

};