// Sparse artifact writers and independent COLMAP validation. ID conversion must
// keep image observations and point tracks consistent in both directions.

#include "reconstruction_exporter.hpp"
#include "geometry.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <functional>
#include <iomanip>
#include <map>
#include <nlohmann/json.hpp>
#include <set>
#include <sstream>
#include <string>
#include <system_error>
#include <tuple>
#include <utility>
#include <vector>

// ---- Reconstruction Io ----

namespace lio_visual_ba
{
namespace
{

Status AtomicWrite(const std::filesystem::path& output_file,
                   const std::function<Status(std::ostream&)>& writer)
{
    if (output_file.empty())
    {
        return Status::Error(ErrorCode::kInvalidArgument, "output path is empty");
    }
    std::error_code error;
    if (!output_file.parent_path().empty())
    {
        std::filesystem::create_directories(output_file.parent_path(), error);
    }
    if (error)
    {
        return Status::Error(ErrorCode::kIoError,
                             "cannot create output directory: " + error.message());
    }
    const auto suffix = std::chrono::steady_clock::now().time_since_epoch().count();
    const std::filesystem::path temporary = output_file.string() + ".tmp." + std::to_string(suffix);
    {
        std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
        if (!output)
        {
            return Status::Error(ErrorCode::kIoError,
                                 "cannot open temporary output: " + temporary.string());
        }
        const Status status = writer(output);
        if (!status.ok())
        {
            output.close();
            std::filesystem::remove(temporary, error);
            return status;
        }
        output.flush();
        if (!output)
        {
            output.close();
            std::filesystem::remove(temporary, error);
            return Status::Error(ErrorCode::kIoError,
                                 "failed while writing output: " + output_file.string());
        }
    }
    std::filesystem::rename(temporary, output_file, error);
    if (error)
    {
        std::filesystem::remove(temporary, error);
        return Status::Error(ErrorCode::kIoError, "cannot replace output file: " + error.message());
    }
    return Status::Ok();
}

Status ValidateObservation(const FeatureObservation& observation)
{
    if (!observation.camera_id.valid() || !observation.feature_id.valid() ||
        !observation.pixel.array().isFinite().all())
    {
        return Status::Error(ErrorCode::kInvalidArgument,
                             "observation has invalid IDs or pixel coordinates");
    }
    return Status::Ok();
}

} // namespace

Status ReconstructionExporter::WriteTumCameraPoses(const std::filesystem::path& output_file,
                                                   const std::vector<CameraFrame>& cameras,
                                                   bool write_optimized_poses)
{
    std::set<int> ids;
    for (const CameraFrame& camera : cameras)
    {
        const Pose3d& pose = write_optimized_poses && camera.has_optimized_pose
                                 ? camera.optimized_world_from_camera
                                 : camera.initial_world_from_camera;
        if (!camera.id.valid() || !ids.insert(camera.id.value()).second ||
            !std::isfinite(camera.timestamp_seconds))
        {
            return Status::Error(ErrorCode::kInvalidArgument, "camera has invalid ID or timestamp");
        }
        const Status status = geometry::ValidatePose(pose);
        if (!status.ok())
        {
            return status.WithContext("invalid camera pose");
        }
    }
    return AtomicWrite(
        output_file,
        [&](std::ostream& output)
        {
            output << std::setprecision(16);
            for (const CameraFrame& camera : cameras)
            {
                const Pose3d& pose = write_optimized_poses && camera.has_optimized_pose
                                         ? camera.optimized_world_from_camera
                                         : camera.initial_world_from_camera;
                const Eigen::Quaterniond& rotation = pose.rotation_world_from_local;
                output << camera.timestamp_seconds << ' ' << pose.translation_world_from_local.x()
                       << ' ' << pose.translation_world_from_local.y() << ' '
                       << pose.translation_world_from_local.z() << ' ' << rotation.x() << ' '
                       << rotation.y() << ' ' << rotation.z() << ' ' << rotation.w() << '\n';
            }
            return Status::Ok();
        });
}

Status ReconstructionExporter::WriteLandmarksPly(const std::filesystem::path& output_file,
                                                 const std::vector<Landmark>& landmarks)
{
    std::set<int> ids;
    for (const Landmark& landmark : landmarks)
    {
        if (!landmark.id.valid() || !ids.insert(landmark.id.value()).second ||
            !landmark.position_world.array().isFinite().all())
        {
            return Status::Error(ErrorCode::kInvalidArgument,
                                 "landmark has invalid ID or position");
        }
    }
    return AtomicWrite(
        output_file,
        [&](std::ostream& output)
        {
            output << "ply\nformat ascii 1.0\nelement vertex " << landmarks.size()
                   << "\nproperty double x\nproperty double y\nproperty double z\n"
                      "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
                   << std::setprecision(16);
            for (const Landmark& landmark : landmarks)
            {
                output << landmark.position_world.x() << ' ' << landmark.position_world.y() << ' '
                       << landmark.position_world.z() << ' '
                       << static_cast<int>(landmark.color_rgb.x()) << ' '
                       << static_cast<int>(landmark.color_rgb.y()) << ' '
                       << static_cast<int>(landmark.color_rgb.z()) << '\n';
            }
            return Status::Ok();
        });
}

Status ReconstructionExporter::WriteTracksCsv(const std::filesystem::path& output_file,
                                              const std::vector<FeatureTrack>& tracks)
{
    std::set<int> track_ids;
    for (const FeatureTrack& track : tracks)
    {
        if (!track.id.valid() || !track_ids.insert(track.id.value()).second)
        {
            return Status::Error(ErrorCode::kInvalidArgument, "track has invalid or duplicate ID");
        }
        std::set<int> cameras;
        for (const FeatureObservation& observation : track.observations)
        {
            const Status status = ValidateObservation(observation);
            if (!status.ok() || !cameras.insert(observation.camera_id.value()).second)
            {
                return Status::Error(ErrorCode::kInvalidArgument,
                                     "track has invalid or duplicate-camera observation");
            }
        }
    }
    return AtomicWrite(output_file,
                       [&](std::ostream& output)
                       {
                           output << "track_id,track_length,is_supplemental\n";
                           for (const FeatureTrack& track : tracks)
                           {
                               output << track.id.value() << ',' << track.observations.size() << ','
                                      << (track.is_supplemental ? 1 : 0) << '\n';
                           }
                           return Status::Ok();
                       });
}

Status ReconstructionExporter::WriteObservationsCsv(const std::filesystem::path& output_file,
                                                    const std::vector<Landmark>& landmarks)
{
    for (const Landmark& landmark : landmarks)
    {
        if (!landmark.id.valid() || !landmark.source_track_id.valid())
        {
            return Status::Error(ErrorCode::kInvalidArgument,
                                 "landmark observation owner has invalid ID");
        }
        for (const FeatureObservation& observation : landmark.observations)
        {
            const Status status = ValidateObservation(observation);
            if (!status.ok())
            {
                return status;
            }
        }
    }
    return AtomicWrite(
        output_file,
        [&](std::ostream& output)
        {
            output << "landmark_id,track_id,camera_id,feature_id,u,v\n" << std::setprecision(16);
            for (const Landmark& landmark : landmarks)
            {
                for (const FeatureObservation& observation : landmark.observations)
                {
                    output << landmark.id.value() << ',' << landmark.source_track_id.value() << ','
                           << observation.camera_id.value() << ',' << observation.feature_id.value()
                           << ',' << observation.pixel.x() << ',' << observation.pixel.y() << '\n';
                }
            }
            return Status::Ok();
        });
}

Status ReconstructionExporter::WriteMetricsJson(const std::filesystem::path& output_file,
                                                const nlohmann::json& metrics)
{
    if (!metrics.is_object())
    {
        return Status::Error(ErrorCode::kInvalidArgument, "pipeline metrics must be a JSON object");
    }
    return AtomicWrite(output_file,
                       [&](std::ostream& output)
                       {
                           output << std::setw(2) << metrics << '\n';
                           return Status::Ok();
                       });
}

} // namespace lio_visual_ba

// ---- Colmap Text Writer ----

namespace lio_visual_ba
{
namespace
{

struct ImagePoint
{
    FeatureId feature_id;
    Eigen::Vector2d pixel;
    LandmarkId landmark_id;
};

const Pose3d& SelectedPose(const CameraFrame& camera, bool prefer_optimized)
{
    return prefer_optimized && camera.has_optimized_pose ? camera.optimized_world_from_camera
                                                         : camera.initial_world_from_camera;
}

Status WriteTemporaryThenReplace(const std::filesystem::path& output_file,
                                 const std::string& contents)
{
    const auto suffix = std::chrono::steady_clock::now().time_since_epoch().count();
    const std::filesystem::path temporary = output_file.string() + ".tmp." + std::to_string(suffix);
    {
        std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
        if (!output)
        {
            return Status::Error(ErrorCode::kIoError, "cannot open temporary COLMAP output");
        }
        output << contents;
        if (!output)
        {
            std::error_code ignored;
            output.close();
            std::filesystem::remove(temporary, ignored);
            return Status::Error(ErrorCode::kIoError, "cannot write temporary COLMAP output");
        }
    }
    std::error_code error;
    std::filesystem::rename(temporary, output_file, error);
    if (error)
    {
        std::filesystem::remove(temporary, error);
        return Status::Error(ErrorCode::kIoError,
                             "cannot replace COLMAP output: " + error.message());
    }
    return Status::Ok();
}

} // namespace

Status
ReconstructionExporter::WriteColmapTextReconstruction(const std::filesystem::path& output_directory,
                                                      const Reconstruction& reconstruction,
                                                      const ColmapTextOptions& options)
{
    if (output_directory.empty())
    {
        return Status::Error(ErrorCode::kInvalidArgument, "COLMAP output directory is empty");
    }
    const Status intrinsics_status = geometry::ValidateCameraIntrinsics(reconstruction.intrinsics);
    if (!intrinsics_status.ok() || reconstruction.intrinsics.width <= 0 ||
        reconstruction.intrinsics.height <= 0)
    {
        return Status::Error(ErrorCode::kInvalidArgument,
                             "COLMAP export requires valid image dimensions and intrinsics");
    }

    std::map<CameraId, const CameraFrame*> cameras;
    std::map<CameraId, std::vector<ImagePoint>> image_points;
    for (const CameraFrame& camera : reconstruction.cameras)
    {
        if (!camera.id.valid() || !cameras.emplace(camera.id, &camera).second ||
            camera.image_name.empty() || camera.image_name.find('\n') != std::string::npos ||
            camera.image_name.find('\r') != std::string::npos)
        {
            return Status::Error(ErrorCode::kInvalidArgument,
                                 "COLMAP camera has invalid ID or image name");
        }
        const Status pose_status =
            geometry::ValidatePose(SelectedPose(camera, options.prefer_optimized_camera_poses));
        if (!pose_status.ok())
        {
            return pose_status.WithContext("invalid COLMAP camera pose");
        }
        image_points.emplace(camera.id, std::vector<ImagePoint>{});
    }

    std::set<int> landmark_ids;
    std::map<std::pair<int, int>, LandmarkId> used_features;
    for (const Landmark& landmark : reconstruction.landmarks)
    {
        if (!landmark.id.valid() || !landmark_ids.insert(landmark.id.value()).second ||
            !landmark.position_world.array().isFinite().all() ||
            !std::isfinite(landmark.quality.median_reprojection_error_pixels) ||
            landmark.quality.median_reprojection_error_pixels < 0.0)
        {
            return Status::Error(ErrorCode::kInvalidArgument,
                                 "COLMAP landmark has invalid ID, position, or error");
        }
        std::set<int> observed_cameras;
        for (const FeatureObservation& observation : landmark.observations)
        {
            if (cameras.count(observation.camera_id) == 0 || !observation.feature_id.valid() ||
                !observation.pixel.array().isFinite().all() ||
                !observed_cameras.insert(observation.camera_id.value()).second)
            {
                return Status::Error(ErrorCode::kInvalidArgument,
                                     "COLMAP landmark has an invalid observation");
            }
            const auto feature_key =
                std::make_pair(observation.camera_id.value(), observation.feature_id.value());
            if (!used_features.emplace(feature_key, landmark.id).second)
            {
                return Status::Error(ErrorCode::kInvalidArgument,
                                     "multiple landmarks reference one image feature");
            }
            image_points[observation.camera_id].push_back(
                {observation.feature_id, observation.pixel, landmark.id});
        }
    }

    std::map<std::pair<int, int>, std::size_t> point_indices;
    for (auto& entry : image_points)
    {
        auto& points = entry.second;
        std::sort(points.begin(), points.end(), [](const ImagePoint& lhs, const ImagePoint& rhs)
                  { return lhs.feature_id < rhs.feature_id; });
        for (std::size_t index = 0; index < points.size(); ++index)
        {
            point_indices[{entry.first.value(), points[index].landmark_id.value()}] = index;
        }
    }

    std::map<CameraId, std::size_t> exported_camera_ids;
    for (const auto& entry : cameras)
    {
        if (image_points.at(entry.first).size() >= options.minimum_observations_per_camera)
        {
            exported_camera_ids.emplace(entry.first, exported_camera_ids.size() + 1);
        }
    }
    if (exported_camera_ids.empty())
    {
        return Status::Error(ErrorCode::kFailedPrecondition,
                             "COLMAP export has no visually supported cameras");
    }

    std::ostringstream cameras_text;
    cameras_text << "# Camera list with one line of data per camera:\n"
                 << "# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
                 << "# Number of cameras: 1\n1 PINHOLE " << reconstruction.intrinsics.width << ' '
                 << reconstruction.intrinsics.height << ' ' << std::setprecision(16)
                 << reconstruction.intrinsics.fx << ' ' << reconstruction.intrinsics.fy << ' '
                 << reconstruction.intrinsics.cx << ' ' << reconstruction.intrinsics.cy << '\n';

    std::ostringstream images_text;
    images_text << "# Image list with two lines of data per image:\n"
                << "# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n"
                << "# POINTS2D[] as (X, Y, POINT3D_ID)\n"
                << "# Number of images: " << exported_camera_ids.size() << '\n'
                << std::setprecision(16);
    for (const auto& entry : cameras)
    {
        const auto exported_id = exported_camera_ids.find(entry.first);
        if (exported_id == exported_camera_ids.end())
        {
            continue;
        }
        const CameraFrame& camera = *entry.second;
        const Pose3d camera_from_world =
            geometry::InversePose(SelectedPose(camera, options.prefer_optimized_camera_poses));
        const Eigen::Quaterniond& q = camera_from_world.rotation_world_from_local;
        const Eigen::Vector3d& t = camera_from_world.translation_world_from_local;
        images_text << exported_id->second << ' ' << q.w() << ' ' << q.x() << ' ' << q.y() << ' '
                    << q.z() << ' ' << t.x() << ' ' << t.y() << ' ' << t.z() << " 1 "
                    << camera.image_name << '\n';
        const auto& points = image_points[camera.id];
        for (std::size_t index = 0; index < points.size(); ++index)
        {
            if (index > 0)
            {
                images_text << ' ';
            }
            images_text << points[index].pixel.x() << ' ' << points[index].pixel.y() << ' '
                        << points[index].landmark_id.value() + 1;
        }
        images_text << '\n';
    }

    std::ostringstream points_text;
    points_text << "# 3D point list with one line of data per point:\n"
                << "# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n"
                << "# Number of points: " << reconstruction.landmarks.size() << '\n'
                << std::setprecision(16);
    for (const Landmark& landmark : reconstruction.landmarks)
    {
        points_text << landmark.id.value() + 1 << ' ' << landmark.position_world.x() << ' '
                    << landmark.position_world.y() << ' ' << landmark.position_world.z() << ' '
                    << static_cast<int>(landmark.color_rgb.x()) << ' '
                    << static_cast<int>(landmark.color_rgb.y()) << ' '
                    << static_cast<int>(landmark.color_rgb.z()) << ' '
                    << landmark.quality.median_reprojection_error_pixels;
        for (const FeatureObservation& observation : landmark.observations)
        {
            const auto index =
                point_indices.find({observation.camera_id.value(), landmark.id.value()});
            if (index == point_indices.end())
            {
                return Status::Error(ErrorCode::kInternal,
                                     "missing COLMAP point-to-image cross-reference");
            }
            const auto camera_id = exported_camera_ids.find(observation.camera_id);
            if (camera_id == exported_camera_ids.end())
            {
                return Status::Error(ErrorCode::kInternal,
                                     "landmark references a filtered COLMAP camera");
            }
            points_text << ' ' << camera_id->second << ' ' << index->second;
        }
        points_text << '\n';
    }

    std::error_code error;
    std::filesystem::create_directories(output_directory, error);
    if (error)
    {
        return Status::Error(ErrorCode::kIoError,
                             "cannot create COLMAP output directory: " + error.message());
    }
    Status status = WriteTemporaryThenReplace(output_directory / "cameras.txt", cameras_text.str());
    if (!status.ok())
    {
        return status;
    }
    status = WriteTemporaryThenReplace(output_directory / "images.txt", images_text.str());
    if (!status.ok())
    {
        return status;
    }
    return WriteTemporaryThenReplace(output_directory / "points3D.txt", points_text.str());
}

} // namespace lio_visual_ba

// ---- Reconstruction Validator ----

namespace lio_visual_ba
{
namespace
{

struct ParsedImage
{
    int camera_id = -1;
    std::vector<int> point_ids;
};

struct ParsedPoint
{
    std::vector<std::pair<int, std::size_t>> track;
};

bool ReadDataLine(std::istream& input, std::string& line)
{
    while (std::getline(input, line))
    {
        const auto first = line.find_first_not_of(" \t\r");
        if (first != std::string::npos && line[first] != '#')
        {
            return true;
        }
    }
    return false;
}

Status CheckNoExtraData(std::istringstream& line, const std::string& context)
{
    line >> std::ws;
    if (!line.eof())
    {
        return Status::Error(ErrorCode::kParseError, context + ": unexpected trailing data");
    }
    return Status::Ok();
}

} // namespace

Result<ColmapValidationReport>
ReconstructionExporter::ValidateColmapTextReconstruction(const std::filesystem::path& directory)
{
    if (directory.empty())
    {
        return Result<ColmapValidationReport>::Failure(
            Status::Error(ErrorCode::kInvalidArgument, "COLMAP reconstruction directory is empty"));
    }

    std::ifstream camera_file(directory / "cameras.txt");
    std::ifstream image_file(directory / "images.txt");
    std::ifstream point_file(directory / "points3D.txt");
    if (!camera_file || !image_file || !point_file)
    {
        return Result<ColmapValidationReport>::Failure(Status::Error(
            ErrorCode::kNotFound, "COLMAP reconstruction is missing a required text file"));
    }

    std::set<int> camera_ids;
    std::string line;
    while (ReadDataLine(camera_file, line))
    {
        std::istringstream fields(line);
        int id = -1;
        std::string model;
        int width = 0;
        int height = 0;
        double fx = 0.0;
        double fy = 0.0;
        double cx = 0.0;
        double cy = 0.0;
        if (!(fields >> id >> model >> width >> height >> fx >> fy >> cx >> cy) || id <= 0 ||
            !camera_ids.insert(id).second || model != "PINHOLE" || width <= 0 || height <= 0 ||
            !std::isfinite(fx) || !std::isfinite(fy) || !std::isfinite(cx) || !std::isfinite(cy) ||
            fx <= 0.0 || fy <= 0.0)
        {
            return Result<ColmapValidationReport>::Failure(
                Status::Error(ErrorCode::kDataLoss, "invalid COLMAP camera record"));
        }
        const Status trailing = CheckNoExtraData(fields, "COLMAP camera record");
        if (!trailing.ok())
        {
            return Result<ColmapValidationReport>::Failure(trailing);
        }
    }
    if (camera_ids.empty())
    {
        return Result<ColmapValidationReport>::Failure(
            Status::Error(ErrorCode::kDataLoss, "COLMAP reconstruction has no cameras"));
    }

    std::map<int, ParsedImage> images;
    while (ReadDataLine(image_file, line))
    {
        std::istringstream fields(line);
        int image_id = -1;
        int camera_id = -1;
        double qw = 0.0;
        double qx = 0.0;
        double qy = 0.0;
        double qz = 0.0;
        double tx = 0.0;
        double ty = 0.0;
        double tz = 0.0;
        if (!(fields >> image_id >> qw >> qx >> qy >> qz >> tx >> ty >> tz >> camera_id) ||
            image_id <= 0 || images.count(image_id) != 0 || camera_ids.count(camera_id) == 0 ||
            !std::isfinite(qw) || !std::isfinite(qx) || !std::isfinite(qy) || !std::isfinite(qz) ||
            !std::isfinite(tx) || !std::isfinite(ty) || !std::isfinite(tz))
        {
            return Result<ColmapValidationReport>::Failure(
                Status::Error(ErrorCode::kDataLoss, "invalid COLMAP image record"));
        }
        const double quaternion_norm = std::sqrt(qw * qw + qx * qx + qy * qy + qz * qz);
        fields >> std::ws;
        std::string image_name;
        std::getline(fields, image_name);
        if (std::abs(quaternion_norm - 1.0) > 1e-6 || image_name.empty())
        {
            return Result<ColmapValidationReport>::Failure(
                Status::Error(ErrorCode::kDataLoss, "invalid COLMAP image quaternion or name"));
        }

        std::string observations_line;
        if (!std::getline(image_file, observations_line))
        {
            return Result<ColmapValidationReport>::Failure(
                Status::Error(ErrorCode::kDataLoss, "COLMAP image is missing its POINTS2D line"));
        }
        ParsedImage image;
        image.camera_id = camera_id;
        std::istringstream observations(observations_line);
        while (true)
        {
            observations >> std::ws;
            if (observations.eof())
            {
                break;
            }
            double x = 0.0;
            double y = 0.0;
            int point_id = -1;
            if (!(observations >> x >> y >> point_id) || !std::isfinite(x) || !std::isfinite(y) ||
                point_id == 0 || point_id < -1)
            {
                return Result<ColmapValidationReport>::Failure(
                    Status::Error(ErrorCode::kDataLoss, "invalid COLMAP POINTS2D record"));
            }
            image.point_ids.push_back(point_id);
        }
        images.emplace(image_id, std::move(image));
    }

    std::map<int, ParsedPoint> points;
    while (ReadDataLine(point_file, line))
    {
        std::istringstream fields(line);
        int point_id = -1;
        double x = 0.0;
        double y = 0.0;
        double z = 0.0;
        int red = 0;
        int green = 0;
        int blue = 0;
        double error = 0.0;
        if (!(fields >> point_id >> x >> y >> z >> red >> green >> blue >> error) ||
            point_id <= 0 || points.count(point_id) != 0 || !std::isfinite(x) ||
            !std::isfinite(y) || !std::isfinite(z) || !std::isfinite(error) || error < 0.0 ||
            red < 0 || red > 255 || green < 0 || green > 255 || blue < 0 || blue > 255)
        {
            return Result<ColmapValidationReport>::Failure(
                Status::Error(ErrorCode::kDataLoss, "invalid COLMAP point3D record"));
        }
        ParsedPoint point;
        std::set<std::pair<int, std::size_t>> track_entries;
        while (true)
        {
            fields >> std::ws;
            if (fields.eof())
            {
                break;
            }
            int image_id = -1;
            long long point_index = -1;
            if (!(fields >> image_id >> point_index) || point_index < 0 ||
                images.count(image_id) == 0 ||
                static_cast<unsigned long long>(point_index) >=
                    images.at(image_id).point_ids.size())
            {
                return Result<ColmapValidationReport>::Failure(
                    Status::Error(ErrorCode::kDataLoss, "invalid COLMAP point track reference"));
            }
            const auto entry = std::make_pair(image_id, static_cast<std::size_t>(point_index));
            if (!track_entries.insert(entry).second)
            {
                return Result<ColmapValidationReport>::Failure(
                    Status::Error(ErrorCode::kDataLoss, "duplicate COLMAP point track reference"));
            }
            point.track.push_back(entry);
        }
        points.emplace(point_id, std::move(point));
    }

    std::size_t observations_count = 0;
    for (const auto& image_entry : images)
    {
        for (std::size_t index = 0; index < image_entry.second.point_ids.size(); ++index)
        {
            const int point_id = image_entry.second.point_ids[index];
            if (point_id == -1)
            {
                continue;
            }
            const auto point = points.find(point_id);
            if (point == points.end() ||
                std::find(point->second.track.begin(), point->second.track.end(),
                          std::make_pair(image_entry.first, index)) == point->second.track.end())
            {
                return Result<ColmapValidationReport>::Failure(Status::Error(
                    ErrorCode::kDataLoss, "COLMAP image observation lacks reciprocal point track"));
            }
            ++observations_count;
        }
    }
    for (const auto& point_entry : points)
    {
        for (const auto& track_entry : point_entry.second.track)
        {
            if (images.at(track_entry.first).point_ids[track_entry.second] != point_entry.first)
            {
                return Result<ColmapValidationReport>::Failure(Status::Error(
                    ErrorCode::kDataLoss, "COLMAP point track lacks reciprocal image observation"));
            }
        }
    }

    ColmapValidationReport report;
    report.cameras = camera_ids.size();
    report.images = images.size();
    report.points3d = points.size();
    report.observations = observations_count;
    return Result<ColmapValidationReport>::Success(report);
}

} // namespace lio_visual_ba
