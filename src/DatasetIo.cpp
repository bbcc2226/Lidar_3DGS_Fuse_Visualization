// Dataset configuration and input parsing. All dataset paths and dimensions come
// from DatasetIOConfig; pose initialization is performed later by Mapper.

#include "DatasetIo.hpp"
#include "Geometry.hpp"
#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <nlohmann/json.hpp>
#include <sstream>
#include <string>
#include <system_error>
#include <unordered_set>
#include <utility>
#include <yaml-cpp/yaml.h>

// ---- Calibration Io ----

namespace lio_visual_ba
{
namespace
{

Result<Eigen::Matrix4d> ParseTransform(const nlohmann::json& root,
                                       const std::filesystem::path& path)
{
    const auto member = root.find("T_camera_lidar");
    if (member == root.end())
    {
        return Result<Eigen::Matrix4d>::Failure(
            Status::Error(ErrorCode::kParseError, path.string() + ": missing T_camera_lidar"));
    }
    if (!member->is_array() || member->size() != 4U)
    {
        return Result<Eigen::Matrix4d>::Failure(Status::Error(
            ErrorCode::kParseError, path.string() + ": T_camera_lidar must be a 4x4 array"));
    }

    Eigen::Matrix4d transform;
    for (Eigen::Index row = 0; row < 4; ++row)
    {
        const auto& json_row = (*member)[static_cast<std::size_t>(row)];
        if (!json_row.is_array() || json_row.size() != 4U)
        {
            return Result<Eigen::Matrix4d>::Failure(Status::Error(
                ErrorCode::kParseError, path.string() + ": T_camera_lidar must be a 4x4 array"));
        }
        for (Eigen::Index column = 0; column < 4; ++column)
        {
            const auto& value = json_row[static_cast<std::size_t>(column)];
            if (!value.is_number())
            {
                return Result<Eigen::Matrix4d>::Failure(
                    Status::Error(ErrorCode::kParseError,
                                  path.string() + ": T_camera_lidar entries must be numbers"));
            }
            transform(row, column) = value.get<double>();
        }
    }
    if (!transform.array().isFinite().all())
    {
        return Result<Eigen::Matrix4d>::Failure(
            Status::Error(ErrorCode::kParseError,
                          path.string() + ": T_camera_lidar contains a non-finite value"));
    }
    return Result<Eigen::Matrix4d>::Success(std::move(transform));
}

} // namespace

Result<CameraIntrinsics>
DatasetIO::LoadCameraIntrinsics(const std::filesystem::path& intrinsics_file,
                                std::int32_t image_width, std::int32_t image_height)
{
    if (intrinsics_file.empty())
    {
        return Result<CameraIntrinsics>::Failure(
            Status::Error(ErrorCode::kInvalidArgument, "intrinsics file path is empty"));
    }
    if (image_width < 0 || image_height < 0)
    {
        return Result<CameraIntrinsics>::Failure(
            Status::Error(ErrorCode::kInvalidArgument, "image dimensions cannot be negative"));
    }

    std::ifstream input(intrinsics_file);
    if (!input)
    {
        return Result<CameraIntrinsics>::Failure(Status::Error(
            ErrorCode::kNotFound, "cannot open intrinsics file: " + intrinsics_file.string()));
    }
    std::ostringstream contents;
    contents << input.rdbuf();
    if (input.bad())
    {
        return Result<CameraIntrinsics>::Failure(
            Status::Error(ErrorCode::kIoError,
                          "failed while reading intrinsics file: " + intrinsics_file.string()));
    }
    std::string text = contents.str();
    std::replace(text.begin(), text.end(), ',', ' ');
    std::istringstream values(text);
    Eigen::Matrix3d matrix;
    for (Eigen::Index row = 0; row < 3; ++row)
    {
        for (Eigen::Index column = 0; column < 3; ++column)
        {
            if (!(values >> matrix(row, column)))
            {
                return Result<CameraIntrinsics>::Failure(Status::Error(
                    ErrorCode::kParseError,
                    intrinsics_file.string() + ": expected exactly nine matrix values"));
            }
        }
    }
    values >> std::ws;
    if (!values.eof())
    {
        return Result<CameraIntrinsics>::Failure(
            Status::Error(ErrorCode::kParseError,
                          intrinsics_file.string() + ": unexpected data after 3x3 matrix"));
    }
    if (!matrix.array().isFinite().all())
    {
        return Result<CameraIntrinsics>::Failure(
            Status::Error(ErrorCode::kParseError,
                          intrinsics_file.string() + ": matrix contains a non-finite value"));
    }
    constexpr double kTolerance = 1e-9;
    if (matrix(0, 0) <= 0.0 || matrix(1, 1) <= 0.0 || std::abs(matrix(0, 1)) > kTolerance ||
        std::abs(matrix(1, 0)) > kTolerance || std::abs(matrix(2, 0)) > kTolerance ||
        std::abs(matrix(2, 1)) > kTolerance || std::abs(matrix(2, 2) - 1.0) > kTolerance)
    {
        return Result<CameraIntrinsics>::Failure(Status::Error(
            ErrorCode::kFailedPrecondition,
            intrinsics_file.string() + ": matrix is not a supported pinhole calibration"));
    }

    CameraIntrinsics intrinsics;
    intrinsics.fx = matrix(0, 0);
    intrinsics.fy = matrix(1, 1);
    intrinsics.cx = matrix(0, 2);
    intrinsics.cy = matrix(1, 2);
    intrinsics.width = image_width;
    intrinsics.height = image_height;
    return Result<CameraIntrinsics>::Success(std::move(intrinsics));
}

Result<Pose3d> DatasetIO::LoadLidarFromCameraExtrinsic(const std::filesystem::path& extrinsic_file)
{
    if (extrinsic_file.empty())
    {
        return Result<Pose3d>::Failure(
            Status::Error(ErrorCode::kInvalidArgument, "extrinsic file path is empty"));
    }
    std::ifstream input(extrinsic_file);
    if (!input)
    {
        return Result<Pose3d>::Failure(Status::Error(
            ErrorCode::kNotFound, "cannot open extrinsic file: " + extrinsic_file.string()));
    }

    nlohmann::json root;
    try
    {
        input >> root;
    }
    catch (const nlohmann::json::exception& error)
    {
        return Result<Pose3d>::Failure(Status::Error(
            ErrorCode::kParseError, extrinsic_file.string() + ": invalid JSON: " + error.what()));
    }
    auto parsed = ParseTransform(root, extrinsic_file);
    if (!parsed.ok())
    {
        return Result<Pose3d>::Failure(parsed.status());
    }
    const Eigen::Matrix4d& transform = parsed.value();
    constexpr double kTolerance = 1e-6;
    if (!transform.row(3).isApprox(Eigen::RowVector4d(0.0, 0.0, 0.0, 1.0), kTolerance))
    {
        return Result<Pose3d>::Failure(
            Status::Error(ErrorCode::kFailedPrecondition,
                          extrinsic_file.string() + ": invalid homogeneous last row"));
    }
    const Eigen::Matrix3d rotation = transform.topLeftCorner<3, 3>();
    if (!(rotation.transpose() * rotation).isApprox(Eigen::Matrix3d::Identity(), kTolerance) ||
        std::abs(rotation.determinant() - 1.0) > kTolerance)
    {
        return Result<Pose3d>::Failure(Status::Error(
            ErrorCode::kFailedPrecondition,
            extrinsic_file.string() + ": rotation is not a proper orthonormal matrix"));
    }

    Pose3d camera_from_lidar;
    camera_from_lidar.rotation_world_from_local = Eigen::Quaterniond(rotation).normalized();
    camera_from_lidar.translation_world_from_local = transform.topRightCorner<3, 1>();
    return Result<Pose3d>::Success(geometry::InversePose(camera_from_lidar));
}

} // namespace lio_visual_ba

// ---- Image List Io ----

namespace lio_visual_ba
{
namespace
{

std::string Location(const std::filesystem::path& path, std::size_t line_number)
{
    return path.string() + ":" + std::to_string(line_number);
}

} // namespace

Result<std::vector<CameraFrame>>
DatasetIO::LoadCameraFrames(const std::filesystem::path& timestamp_file,
                            const ImageListOptions& options)
{
    if (timestamp_file.empty())
    {
        return Result<std::vector<CameraFrame>>::Failure(
            Status::Error(ErrorCode::kInvalidArgument, "timestamp file path is empty"));
    }
    if (options.image_directory.empty())
    {
        return Result<std::vector<CameraFrame>>::Failure(
            Status::Error(ErrorCode::kInvalidArgument, "image directory path is empty"));
    }
    if (options.maximum_images < -1)
    {
        return Result<std::vector<CameraFrame>>::Failure(Status::Error(
            ErrorCode::kInvalidArgument, "maximum_images must be -1 or non-negative"));
    }

    std::ifstream input(timestamp_file);
    if (!input)
    {
        return Result<std::vector<CameraFrame>>::Failure(Status::Error(
            ErrorCode::kNotFound, "cannot open timestamp file: " + timestamp_file.string()));
    }

    std::vector<CameraFrame> cameras;
    std::unordered_set<std::string> image_names;
    std::string line;
    std::size_t line_number = 0;
    double previous_timestamp = 0.0;
    bool has_previous_timestamp = false;

    while (std::getline(input, line))
    {
        ++line_number;

        std::istringstream line_stream(line);
        line_stream >> std::ws;
        if (line_stream.eof() || line_stream.peek() == '#')
        {
            continue;
        }

        std::string image_name;
        double timestamp_seconds = 0.0;
        if (!(line_stream >> image_name >> timestamp_seconds))
        {
            return Result<std::vector<CameraFrame>>::Failure(Status::Error(
                ErrorCode::kParseError, Location(timestamp_file, line_number) +
                                            ": expected 'image_name timestamp_seconds'"));
        }

        line_stream >> std::ws;
        if (!line_stream.eof() && line_stream.peek() != '#')
        {
            return Result<std::vector<CameraFrame>>::Failure(
                Status::Error(ErrorCode::kParseError, Location(timestamp_file, line_number) +
                                                          ": unexpected data after timestamp"));
        }
        if (!std::isfinite(timestamp_seconds))
        {
            return Result<std::vector<CameraFrame>>::Failure(
                Status::Error(ErrorCode::kParseError, Location(timestamp_file, line_number) +
                                                          ": timestamp must be finite"));
        }
        if (!image_names.insert(image_name).second)
        {
            return Result<std::vector<CameraFrame>>::Failure(
                Status::Error(ErrorCode::kDataLoss, Location(timestamp_file, line_number) +
                                                        ": duplicate image name: " + image_name));
        }
        if (options.require_strictly_increasing_timestamps && has_previous_timestamp &&
            timestamp_seconds <= previous_timestamp)
        {
            return Result<std::vector<CameraFrame>>::Failure(Status::Error(
                ErrorCode::kDataLoss, Location(timestamp_file, line_number) +
                                          ": timestamps are not strictly increasing"));
        }
        previous_timestamp = timestamp_seconds;
        has_previous_timestamp = true;

        const std::filesystem::path image_path =
            options.image_directory / std::filesystem::path(image_name);
        std::error_code filesystem_error;
        const bool image_exists = std::filesystem::exists(image_path, filesystem_error);
        if (filesystem_error)
        {
            return Result<std::vector<CameraFrame>>::Failure(Status::Error(
                ErrorCode::kIoError, "cannot inspect image path " + image_path.string() + ": " +
                                         filesystem_error.message()));
        }
        if (!image_exists)
        {
            if (options.skip_missing_images)
            {
                continue;
            }
            return Result<std::vector<CameraFrame>>::Failure(Status::Error(
                ErrorCode::kNotFound, Location(timestamp_file, line_number) +
                                          ": image does not exist: " + image_path.string()));
        }
        const bool is_regular_file = std::filesystem::is_regular_file(image_path, filesystem_error);
        if (filesystem_error)
        {
            return Result<std::vector<CameraFrame>>::Failure(Status::Error(
                ErrorCode::kIoError, "cannot inspect image path " + image_path.string() + ": " +
                                         filesystem_error.message()));
        }
        if (!is_regular_file)
        {
            return Result<std::vector<CameraFrame>>::Failure(
                Status::Error(ErrorCode::kFailedPrecondition,
                              Location(timestamp_file, line_number) +
                                  ": image path is not a regular file: " + image_path.string()));
        }

        CameraFrame camera;
        camera.id = CameraId(static_cast<CameraId::ValueType>(cameras.size()));
        camera.image_name = std::move(image_name);
        camera.image_path = image_path;
        camera.timestamp_seconds = timestamp_seconds;
        cameras.push_back(std::move(camera));

        if (options.maximum_images >= 0 &&
            cameras.size() >= static_cast<std::size_t>(options.maximum_images))
        {
            break;
        }
    }

    if (!input.eof() && input.fail())
    {
        return Result<std::vector<CameraFrame>>::Failure(
            Status::Error(ErrorCode::kIoError,
                          "failed while reading timestamp file: " + timestamp_file.string()));
    }
    if (cameras.empty())
    {
        return Result<std::vector<CameraFrame>>::Failure(Status::Error(
            ErrorCode::kFailedPrecondition,
            "timestamp file produced no usable camera frames: " + timestamp_file.string()));
    }

    return Result<std::vector<CameraFrame>>::Success(std::move(cameras));
}

} // namespace lio_visual_ba

// ---- Trajectory Io ----

namespace lio_visual_ba
{
namespace
{

Result<Pose3d> ParseLioPose(const nlohmann::json& item, const std::string& location)
{
    try
    {
        const auto& value = item.at("lio_pose");
        const auto& translation = value.at("translation");
        const auto& quaternion = value.at("quaternion_xyzw");
        if (!translation.is_array() || translation.size() != 3 || !quaternion.is_array() ||
            quaternion.size() != 4)
        {
            return Result<Pose3d>::Failure(Status::Error(
                ErrorCode::kParseError,
                location + ": lio_pose requires translation[3] and quaternion_xyzw[4]"));
        }
        Pose3d pose;
        pose.translation_world_from_local =
            Eigen::Vector3d(translation.at(0).get<double>(), translation.at(1).get<double>(),
                            translation.at(2).get<double>());
        pose.rotation_world_from_local =
            Eigen::Quaterniond(quaternion.at(3).get<double>(), quaternion.at(0).get<double>(),
                               quaternion.at(1).get<double>(), quaternion.at(2).get<double>());
        auto normalized = geometry::NormalizePose(pose);
        if (!normalized.ok())
        {
            return Result<Pose3d>::Failure(
                normalized.status().WithContext(location + ": invalid lio_pose"));
        }
        return normalized;
    }
    catch (const nlohmann::json::exception& error)
    {
        return Result<Pose3d>::Failure(Status::Error(
            ErrorCode::kParseError, location + ": invalid lio_pose: " + error.what()));
    }
}

} // namespace

Result<std::vector<TimedPose>>
DatasetIO::LoadLioTrajectory(const std::filesystem::path& trajectory_file)
{
    if (trajectory_file.empty())
    {
        return Result<std::vector<TimedPose>>::Failure(
            Status::Error(ErrorCode::kInvalidArgument, "trajectory file path is empty"));
    }
    std::ifstream input(trajectory_file);
    if (!input)
    {
        return Result<std::vector<TimedPose>>::Failure(Status::Error(
            ErrorCode::kNotFound, "cannot open trajectory file: " + trajectory_file.string()));
    }

    std::vector<TimedPose> trajectory;
    std::size_t record_number = 0;
    while (true)
    {
        input >> std::ws;
        if (input.peek() == std::char_traits<char>::eof())
        {
            break;
        }
        ++record_number;
        nlohmann::json item;
        try
        {
            input >> item;
        }
        catch (const nlohmann::json::exception& error)
        {
            return Result<std::vector<TimedPose>>::Failure(
                Status::Error(ErrorCode::kParseError, trajectory_file.string() + ": record " +
                                                          std::to_string(record_number) +
                                                          ": invalid JSON: " + error.what()));
        }
        const std::string location =
            trajectory_file.string() + ": record " + std::to_string(record_number);
        double timestamp = 0.0;
        try
        {
            timestamp = item.at("timestamp").get<double>();
        }
        catch (const nlohmann::json::exception& error)
        {
            return Result<std::vector<TimedPose>>::Failure(Status::Error(
                ErrorCode::kParseError, location + ": invalid timestamp: " + error.what()));
        }
        if (!std::isfinite(timestamp))
        {
            return Result<std::vector<TimedPose>>::Failure(
                Status::Error(ErrorCode::kParseError, location + ": timestamp must be finite"));
        }
        auto pose = ParseLioPose(item, location);
        if (!pose.ok())
        {
            return Result<std::vector<TimedPose>>::Failure(pose.status());
        }
        trajectory.push_back({timestamp, std::move(pose).value()});
    }
    if (!input.eof() && input.fail())
    {
        return Result<std::vector<TimedPose>>::Failure(
            Status::Error(ErrorCode::kIoError,
                          "failed while reading trajectory file: " + trajectory_file.string()));
    }
    if (trajectory.size() < 2)
    {
        return Result<std::vector<TimedPose>>::Failure(Status::Error(
            ErrorCode::kFailedPrecondition, "trajectory requires at least two poses"));
    }
    std::sort(trajectory.begin(), trajectory.end(), [](const TimedPose& lhs, const TimedPose& rhs)
              { return lhs.timestamp_seconds < rhs.timestamp_seconds; });
    for (std::size_t index = 1; index < trajectory.size(); ++index)
    {
        if (trajectory[index].timestamp_seconds == trajectory[index - 1].timestamp_seconds)
        {
            return Result<std::vector<TimedPose>>::Failure(
                Status::Error(ErrorCode::kDataLoss, "trajectory contains duplicate timestamps"));
        }
    }
    return Result<std::vector<TimedPose>>::Success(std::move(trajectory));
}

} // namespace lio_visual_ba

// ---- Configuration and dataset loading ----
namespace lio_visual_ba
{
namespace
{
Result<std::filesystem::path> RequiredPath(const YAML::Node& node, const char* key,
                                           const std::filesystem::path& base_directory)
{
    if (!node || !node[key] || !node[key].IsScalar())
    {
        return Result<std::filesystem::path>::Failure(Status::Error(
            ErrorCode::kInvalidArgument, std::string("missing required path: ") + key));
    }
    std::filesystem::path path = node[key].as<std::string>();
    if (path.empty())
    {
        return Result<std::filesystem::path>::Failure(
            Status::Error(ErrorCode::kInvalidArgument, std::string("empty path: ") + key));
    }
    if (path.is_relative())
    {
        path = base_directory / path;
    }
    return Result<std::filesystem::path>::Success(path.lexically_normal());
}

template <typename T> void AssignIfPresent(const YAML::Node& node, const char* key, T& value)
{
    if (node && node[key])
    {
        value = node[key].as<T>();
    }
}

} // namespace

Result<DatasetIOConfig> DatasetIO::LoadConfig(const std::filesystem::path& configuration_file)
{
    try
    {
        const YAML::Node root = YAML::LoadFile(configuration_file.string());
        if (!root.IsMap())
            return Result<DatasetIOConfig>::Failure(Status::Error(
                ErrorCode::kInvalidArgument, "dataset configuration root must be a map"));
        const auto base = std::filesystem::absolute(configuration_file).parent_path();
        const YAML::Node inputs = root["inputs"];
        auto timestamps = RequiredPath(inputs, "timestamp_file", base);
        auto images = RequiredPath(inputs, "image_directory", base);
        auto intrinsics = RequiredPath(inputs, "intrinsics_file", base);
        auto extrinsic = RequiredPath(inputs, "extrinsic_file", base);
        auto trajectory = RequiredPath(inputs, "trajectory_file", base);
        for (const auto* path : {&timestamps, &images, &intrinsics, &extrinsic, &trajectory})
            if (!path->ok())
                return Result<DatasetIOConfig>::Failure(path->status());
        DatasetIOConfig config;
        config.inputs = {timestamps.value(), images.value(), intrinsics.value(), extrinsic.value(),
                         trajectory.value()};
        AssignIfPresent(root, "image_width", config.image_width);
        AssignIfPresent(root, "image_height", config.image_height);
        AssignIfPresent(root, "maximum_images", config.image_list.maximum_images);
        AssignIfPresent(root, "skip_missing_images", config.image_list.skip_missing_images);
        AssignIfPresent(root, "require_strictly_increasing_timestamps",
                        config.image_list.require_strictly_increasing_timestamps);
        config.image_list.image_directory = config.inputs.image_directory;
        if (config.image_width <= 0 || config.image_height <= 0)
            return Result<DatasetIOConfig>::Failure(Status::Error(
                ErrorCode::kInvalidArgument, "dataset image dimensions must be positive"));
        return Result<DatasetIOConfig>::Success(std::move(config));
    }
    catch (const YAML::Exception& error)
    {
        return Result<DatasetIOConfig>::Failure(Status::Error(
            ErrorCode::kParseError, configuration_file.string() + ": " + error.what()));
    }
}

Result<DatasetInputs> DatasetIO::LoadDataset(const DatasetIOConfig& config)
{
    if (config.image_width <= 0 || config.image_height <= 0 ||
        config.inputs.image_directory.empty())
        return Result<DatasetInputs>::Failure(
            Status::Error(ErrorCode::kInvalidArgument,
                          "dataset requires positive dimensions and an image directory"));
    // Use the configured dataset directory as the single source of truth, including
    // for callers that construct the configuration directly instead of reading YAML.
    ImageListOptions image_options = config.image_list;
    image_options.image_directory = config.inputs.image_directory;
    auto cameras = LoadCameraFrames(config.inputs.timestamp_file, image_options);
    if (!cameras.ok())
        return Result<DatasetInputs>::Failure(
            cameras.status().WithContext("dataset camera loading"));
    auto intrinsics = LoadCameraIntrinsics(config.inputs.intrinsics_file, config.image_width,
                                           config.image_height);
    if (!intrinsics.ok())
        return Result<DatasetInputs>::Failure(
            intrinsics.status().WithContext("dataset intrinsics loading"));
    auto extrinsic = LoadLidarFromCameraExtrinsic(config.inputs.extrinsic_file);
    if (!extrinsic.ok())
        return Result<DatasetInputs>::Failure(
            extrinsic.status().WithContext("dataset extrinsic loading"));
    auto trajectory = LoadLioTrajectory(config.inputs.trajectory_file);
    if (!trajectory.ok())
        return Result<DatasetInputs>::Failure(
            trajectory.status().WithContext("dataset trajectory loading"));
    DatasetInputs result;
    result.cameras = std::move(cameras.value());
    result.intrinsics = intrinsics.value();
    result.lidar_from_camera = extrinsic.value();
    result.trajectory = std::move(trajectory.value());
    return Result<DatasetInputs>::Success(std::move(result));
}
} // namespace lio_visual_ba
