#pragma once

// Dataset configuration and input parsing. All dataset paths and dimensions come
// from DatasetIOConfig; pose initialization is performed later by Mapper.

#include "status.hpp"
#include "types.hpp"
#include <cstdint>
#include <filesystem>
#include <vector>

namespace lio_visual_ba
{

struct ImageListOptions
{
    std::filesystem::path image_directory;
    bool skip_missing_images = true;
    bool require_strictly_increasing_timestamps = true;
    std::int32_t maximum_images = -1;
};

struct TimedPose
{
    double timestamp_seconds = 0.0;
    Pose3d world_from_lidar;
};

struct DatasetInputPaths
{
    std::filesystem::path timestamp_file;
    std::filesystem::path image_directory;
    std::filesystem::path intrinsics_file;
    std::filesystem::path extrinsic_file;
    std::filesystem::path trajectory_file;
};

// Filled by LoadConfig; relative input paths are resolved against the YAML file.
struct DatasetIOConfig
{
    DatasetInputPaths inputs;
    std::int32_t image_width = 0;
    std::int32_t image_height = 0;
    ImageListOptions image_list;
};

struct DatasetInputs
{
    CameraIntrinsics intrinsics;
    Pose3d lidar_from_camera;
    std::vector<CameraFrame> cameras;
    std::vector<TimedPose> trajectory;
};

// Stateless component API. Options and data are supplied explicitly per call.
class DatasetIO
{
  public:
    // Read dataset settings from the same YAML file used by Pipeline.
    // Image dimensions must be positive; no dataset-specific values are embedded.
    static Result<DatasetIOConfig> LoadConfig(const std::filesystem::path& configuration_file);

    // Load all inputs with explicit configuration; input image_directory is authoritative.
    static Result<DatasetInputs> LoadDataset(const DatasetIOConfig& config);

    // Load a comma- or whitespace-delimited 3x3 pinhole calibration matrix.
    // Image dimensions are supplied separately because the legacy intrinsics.txt
    // format does not store them. Zero means unknown.
    static Result<CameraIntrinsics>
    LoadCameraIntrinsics(const std::filesystem::path& intrinsics_file, std::int32_t image_width = 0,
                         std::int32_t image_height = 0);

    // Load T_camera_lidar from the legacy JSON calibration and return its inverse,
    // T_lidar_camera, so it composes directly with T_world_lidar.
    static Result<Pose3d> LoadLidarFromCameraExtrinsic(const std::filesystem::path& extrinsic_file);

    // Read records formatted as:
    //
    //   image_name timestamp_seconds
    //
    // Empty lines and lines beginning with '#' are ignored. Camera IDs are compact
    // and follow the order of accepted records.
    static Result<std::vector<CameraFrame>>
    LoadCameraFrames(const std::filesystem::path& timestamp_file, const ImageListOptions& options);

    // Load consecutive (pretty-printed or one-line) JSON objects containing
    // `timestamp` and `lio_pose` members. Results are sorted by timestamp.
    static Result<std::vector<TimedPose>>
    LoadLioTrajectory(const std::filesystem::path& trajectory_file);
};

} // namespace lio_visual_ba
