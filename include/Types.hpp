#pragma once

// Shared reconstruction data and typed IDs. Initial and optimized poses are
// separate; camera/feature/track/landmark IDs must not be used interchangeably.

#include <cstdint>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/core.hpp>

namespace lio_visual_ba
{

template <typename Tag> class Id
{
  public:
    using ValueType = std::int32_t;

    constexpr Id() noexcept = default;
    explicit constexpr Id(ValueType value) noexcept : value_(value) {}

    static constexpr Id Invalid() noexcept
    {
        return Id();
    }

    constexpr ValueType value() const noexcept
    {
        return value_;
    }
    constexpr bool valid() const noexcept
    {
        return value_ >= 0;
    }

    friend constexpr bool operator==(Id lhs, Id rhs) noexcept
    {
        return lhs.value_ == rhs.value_;
    }

    friend constexpr bool operator!=(Id lhs, Id rhs) noexcept
    {
        return !(lhs == rhs);
    }

    friend constexpr bool operator<(Id lhs, Id rhs) noexcept
    {
        return lhs.value_ < rhs.value_;
    }

  private:
    ValueType value_ = -1;
};

struct CameraIdTag;
struct FeatureIdTag;
struct TrackIdTag;
struct LandmarkIdTag;

using CameraId = Id<CameraIdTag>;
using FeatureId = Id<FeatureIdTag>;
using TrackId = Id<TrackIdTag>;
using LandmarkId = Id<LandmarkIdTag>;

// Pose convention used by the entire standalone pipeline. A point expressed
// in the local frame is transformed into the world frame as:
//
//   point_world = rotation_world_from_local * point_local
//               + translation_world_from_local
struct Pose3d
{
    Eigen::Quaterniond rotation_world_from_local = Eigen::Quaterniond::Identity();
    Eigen::Vector3d translation_world_from_local = Eigen::Vector3d::Zero();
};

struct CameraIntrinsics
{
    double fx = 0.0;
    double fy = 0.0;
    double cx = 0.0;
    double cy = 0.0;
    std::int32_t width = 0;
    std::int32_t height = 0;

    Eigen::Matrix3d Matrix() const
    {
        Eigen::Matrix3d matrix = Eigen::Matrix3d::Identity();
        matrix(0, 0) = fx;
        matrix(1, 1) = fy;
        matrix(0, 2) = cx;
        matrix(1, 2) = cy;
        return matrix;
    }
};

struct CameraFrame
{
    CameraId id;
    std::string image_name;
    std::filesystem::path image_path;
    double timestamp_seconds = 0.0;
    Pose3d initial_world_from_camera;
    Pose3d optimized_world_from_camera;
    bool has_optimized_pose = false;
};

// Feature metadata and descriptor storage are separated. This keeps all SIFT
// descriptors in one contiguous OpenCV matrix instead of allocating one
// matrix for every feature.
struct Feature
{
    FeatureId id;
    Eigen::Vector2d pixel = Eigen::Vector2d::Zero();
    float scale = 0.0F;
    float angle_degrees = -1.0F;
    float response = 0.0F;
    std::int32_t octave = 0;
};

struct FeatureSet
{
    CameraId camera_id;
    std::vector<Feature> features;
    cv::Mat descriptors;
    // Features [0, core_feature_count) came from the primary SIFT detection;
    // later features are grid-balancing supplements.
    std::size_t core_feature_count = 0;
};

struct FeatureObservation
{
    CameraId camera_id;
    FeatureId feature_id;
    Eigen::Vector2d pixel = Eigen::Vector2d::Zero();
};

struct FeatureMatch
{
    CameraId first_camera_id;
    FeatureId first_feature_id;
    CameraId second_camera_id;
    FeatureId second_feature_id;
    float descriptor_distance = 0.0F;
    double geometric_error_pixels = 0.0;
};

struct FeatureTrack
{
    TrackId id;
    std::vector<FeatureObservation> observations;
    bool is_supplemental = false;
};

struct LandmarkQuality
{
    double maximum_parallax_degrees = 0.0;
    double median_reprojection_error_pixels = 0.0;
    double maximum_reprojection_error_pixels = 0.0;
};

struct Landmark
{
    LandmarkId id;
    TrackId source_track_id;
    Eigen::Vector3d position_world = Eigen::Vector3d::Zero();
    // Immutable triangulation result used by repeated BA landmark priors.
    Eigen::Vector3d initial_position_world = Eigen::Vector3d::Zero();
    bool has_initial_position = false;
    Eigen::Vector3<std::uint8_t> color_rgb = Eigen::Vector3<std::uint8_t>::Zero();
    std::vector<FeatureObservation> observations;
    LandmarkQuality quality;
};

struct ReconstructionProvenance
{
    std::string pipeline_version;
    std::string configuration_fingerprint;
    std::string input_fingerprint;
    std::uint64_t random_seed = 0;
};

struct Reconstruction
{
    CameraIntrinsics intrinsics;
    std::vector<CameraFrame> cameras;
    std::vector<FeatureTrack> tracks;
    std::vector<Landmark> landmarks;
    ReconstructionProvenance provenance;
};

} // namespace lio_visual_ba
