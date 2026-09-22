#pragma once

// Ceres bundle adjustment and local/global scheduling. Keep initial pose priors
// immutable and merge candidate results only after the corresponding solve succeeds.

#include "Status.hpp"
#include "Types.hpp"
#include <cstddef>
#include <vector>

namespace lio_visual_ba
{

struct LandmarkBundleAdjustmentOptions
{
    std::size_t minimum_observations = 2;
    int maximum_iterations = 50;
    double huber_loss_pixels = 2.0;
    double minimum_depth = 1e-6;
    int number_of_threads = 1;
};

struct BundleAdjustmentReport
{
    std::size_t optimized_cameras = 0;
    std::size_t optimized_landmarks = 0;
    std::size_t residual_observations = 0;
    std::size_t skipped_landmarks = 0;
    int iterations = 0;
    double initial_reprojection_rmse_pixels = 0.0;
    double final_reprojection_rmse_pixels = 0.0;
};

struct JointBundleAdjustmentOptions
{
    std::size_t minimum_landmark_observations = 2;
    int maximum_iterations = 100;
    double huber_loss_pixels = 2.0;
    // Python reference uses a translation residual weight of 2.0 (sigma 0.5 m).
    double translation_prior_sigma = 0.5;
    double rotation_prior_sigma = 0.1;
    double minimum_depth = 1e-6;
    int number_of_threads = 1;
    CameraId anchor_camera_id = CameraId::Invalid();
    std::vector<CameraId> constant_camera_ids;
};

struct OutlierRefinementOptions
{
    JointBundleAdjustmentOptions bundle_adjustment;
    double maximum_reprojection_error_pixels = 4.0;
    int maximum_cleanup_cycles = 3;
};

struct OutlierRefinementReport
{
    int cleanup_cycles = 0;
    std::size_t removed_observations = 0;
    std::size_t removed_landmarks = 0;
    BundleAdjustmentReport final_bundle_adjustment;
};

struct LocalWindowSelectionOptions
{
    std::size_t maximum_cameras = 10;
    std::size_t minimum_shared_landmarks = 1;
    std::size_t minimum_selected_observations = 2;
};

struct LocalBundleAdjustmentWindow
{
    std::vector<CameraId> camera_ids;
    std::vector<LandmarkId> landmark_ids;
};

struct LocalBundleAdjustmentOptions
{
    LocalWindowSelectionOptions window_selection;
    JointBundleAdjustmentOptions bundle_adjustment;
};

struct LocalBundleAdjustmentReport
{
    LocalBundleAdjustmentWindow window;
    BundleAdjustmentReport bundle_adjustment;
};

struct IncrementalOptimizationOptions
{
    std::size_t minimum_registered_cameras = 2;
    std::size_t global_bundle_adjustment_interval = 10;
    bool run_final_global_bundle_adjustment = true;
    LocalBundleAdjustmentOptions local_bundle_adjustment;
    JointBundleAdjustmentOptions global_bundle_adjustment;
};

struct IncrementalOptimizationReport
{
    std::size_t local_solve_count = 0;
    std::size_t skipped_local_solve_count = 0;
    std::size_t global_solve_count = 0;
    std::vector<CameraId> locally_optimized_seeds;
    BundleAdjustmentReport final_bundle_adjustment;
};

// Stateless component API. Options and data are supplied explicitly per call.
class Optimizer
{
  public:
    // Optimize landmark positions while holding camera poses and intrinsics fixed.
    // The reconstruction is changed only after a usable, geometrically valid solve.
    static Result<BundleAdjustmentReport>
    OptimizeLandmarks(Reconstruction& reconstruction,
                      const LandmarkBundleAdjustmentOptions& options = {});

    // Jointly optimize camera poses and landmarks with immutable initial LIO pose priors.
    // An explicitly configured anchor and constant cameras remain fixed.
    static Result<BundleAdjustmentReport>
    OptimizeCamerasAndLandmarks(Reconstruction& reconstruction,
                                const JointBundleAdjustmentOptions& options = {});

    // Repeatedly solve joint BA and remove observations above the pixel threshold.
    // Landmarks falling below the configured minimum support are removed.
    static Result<OutlierRefinementReport>
    RefineWithOutlierCleanup(Reconstruction& reconstruction,
                             const OutlierRefinementOptions& options = {});

    // Grow a deterministic co-visibility window from seed_camera_id. At each step,
    // the camera sharing the most landmarks with the current window is selected;
    // CameraId breaks ties. Returned IDs are sorted for stable downstream use.
    static Result<LocalBundleAdjustmentWindow>
    SelectLocalBundleAdjustmentWindow(const Reconstruction& reconstruction, CameraId seed_camera_id,
                                      const LocalWindowSelectionOptions& options = {});

    // Solve a temporal window ending at the seed, with outside cameras fixed.
    // The seed is variable unless explicitly anchored; merge only after success.
    static Result<LocalBundleAdjustmentReport>
    OptimizeLocalBundleAdjustment(Reconstruction& reconstruction, CameraId seed_camera_id,
                                  const LocalBundleAdjustmentOptions& options = {});

    // Replay optimization in camera-vector order. Each step sees only cameras and
    // observations registered up to that point, preventing future-frame leakage.
    static Result<IncrementalOptimizationReport>
    RunIncrementalOptimization(Reconstruction& reconstruction,
                               const IncrementalOptimizationOptions& options = {});
};

} // namespace lio_visual_ba
