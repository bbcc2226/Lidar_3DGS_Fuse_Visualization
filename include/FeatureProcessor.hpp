#pragma once

// Image features, pair matching, binary caches, and conflict-safe tracks.
// Feature IDs must remain aligned with descriptor rows throughout these stages.

#include "Status.hpp"
#include "Types.hpp"
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <vector>

namespace lio_visual_ba
{

struct FeatureExtractionOptions
{
    std::int32_t maximum_sift_features = 6000;
    std::int32_t grid_columns = 8;
    std::int32_t grid_rows = 6;
    std::int32_t minimum_core_features_per_cell = 20;
    std::int32_t maximum_supplemental_features_per_cell = 40;
    double supplemental_quality_level = 0.003;
    double supplemental_minimum_distance_pixels = 6.0;
};

struct FeatureCacheFingerprint
{
    std::uint64_t source = 0;
    std::uint64_t configuration = 0;
};

struct FeatureMatchingOptions
{
    double ratio_threshold = 0.75;
    bool require_mutual_match = false;
    double maximum_sampson_error_pixels = 4.0;
};

struct VisualGeometryOptions
{
    double ransac_threshold_pixels = 4.0;
    double confidence = 0.999;
    std::int32_t maximum_iterations = 10000;
    std::int32_t minimum_inliers = 20;
    double minimum_inlier_ratio = 0.25;
    std::int32_t coverage_columns = 4;
    std::int32_t coverage_rows = 3;
    std::int32_t minimum_occupied_cells = 6;
    std::uint64_t random_seed = 0;
};

struct VisualGeometryResult
{
    Eigen::Matrix3d fundamental_matrix = Eigen::Matrix3d::Zero();
    std::vector<FeatureMatch> inlier_matches;
    std::int32_t first_occupied_cells = 0;
    std::int32_t second_occupied_cells = 0;
};

struct PairCacheKey
{
    CameraId first_camera_id;
    CameraId second_camera_id;
    std::uint64_t first_feature_fingerprint = 0;
    std::uint64_t second_feature_fingerprint = 0;
    std::uint64_t matching_configuration_fingerprint = 0;
    std::uint64_t first_feature_count = 0;
    std::uint64_t second_feature_count = 0;
};

struct TrackBuilderOptions
{
    std::size_t minimum_track_length = 2;
};
struct TrackBuildResult
{
    std::vector<FeatureTrack> tracks;
    std::size_t accepted_matches = 0;
    std::size_t redundant_matches = 0;
    std::size_t conflict_rejected_matches = 0;
};

// Stateless component API. Options and data are supplied explicitly per call.
class FeatureProcessor
{
  public:
    static Result<FeatureSet> ExtractFeatures(CameraId camera_id,
                                              const std::filesystem::path& image_file,
                                              const FeatureExtractionOptions& options = {});

    static Result<FeatureSet> ExtractFeatures(CameraId camera_id, const cv::Mat& image,
                                              const FeatureExtractionOptions& options = {});

    static Status SaveFeatureCache(const std::filesystem::path& cache_file,
                                   const FeatureSet& features,
                                   const FeatureCacheFingerprint& fingerprint);

    static Result<FeatureSet> LoadFeatureCache(const std::filesystem::path& cache_file,
                                               const FeatureCacheFingerprint& expected_fingerprint);

    static Result<std::vector<FeatureMatch>>
    MatchFeatureDescriptors(const FeatureSet& first, const FeatureSet& second,
                            const FeatureMatchingOptions& options = {});

    static Result<std::vector<FeatureMatch>>
    MatchFeaturesWithPosePrior(const FeatureSet& first, const Pose3d& world_from_first_camera,
                               const FeatureSet& second, const Pose3d& world_from_second_camera,
                               const CameraIntrinsics& intrinsics,
                               const FeatureMatchingOptions& options = {});

    static Result<VisualGeometryResult>
    VerifyMatchesWithVisualGeometry(const FeatureSet& first, const FeatureSet& second,
                                    const std::vector<FeatureMatch>& candidate_matches,
                                    const VisualGeometryOptions& options = {});

    static Status SavePairCache(const std::filesystem::path& cache_file, const PairCacheKey& key,
                                const std::vector<FeatureMatch>& matches);

    static Result<std::vector<FeatureMatch>> LoadPairCache(const std::filesystem::path& cache_file,
                                                           const PairCacheKey& expected_key);

    // Union verified matches while enforcing at most one feature per camera per track.
    // Match order determines which conflicting merge wins; do not reorder casually.
    static Result<TrackBuildResult>
    BuildFeatureTracks(const std::vector<FeatureSet>& feature_sets,
                       const std::vector<FeatureMatch>& verified_matches,
                       const TrackBuilderOptions& options = {});
};

} // namespace lio_visual_ba
