// Pipeline configuration and orchestration. The public API also covers the
// frontend, backend, and rematcher implemented in their separate source files.

#include "Pipeline.hpp"
#include "ReconstructionExporter.hpp"
#include "Geometry.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <map>
#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>
#include <string>
#include <system_error>
#include <utility>
#include <vector>
#include <yaml-cpp/yaml.h>

// ---- Pipeline Config ----

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

Result<PipelineConfig> Pipeline::LoadPipelineConfig(const std::filesystem::path& configuration_file)
{
    try
    {
        const YAML::Node root = YAML::LoadFile(configuration_file.string());
        if (!root.IsMap())
        {
            return Result<PipelineConfig>::Failure(Status::Error(
                ErrorCode::kInvalidArgument, "pipeline configuration root must be a map"));
        }
        const std::filesystem::path base_directory =
            std::filesystem::absolute(configuration_file).parent_path();
        auto dataset_config = DatasetIO::LoadConfig(configuration_file);
        if (!dataset_config.ok())
            return Result<PipelineConfig>::Failure(dataset_config.status());
        PipelineConfig config;
        static_cast<DatasetIOConfig&>(config) = std::move(dataset_config.value());
        auto output = RequiredPath(root, "output_directory", base_directory);
        if (!output.ok())
            return Result<PipelineConfig>::Failure(output.status());
        config.output_directory = output.value();
        AssignIfPresent(root, "camera_time_offset_seconds", config.camera_time_offset_seconds);
        AssignIfPresent(root, "random_seed", config.random_seed);
        AssignIfPresent(root, "maximum_cpu_threads", config.maximum_cpu_threads);

        const YAML::Node features = root["features"];
        AssignIfPresent(features, "maximum_sift_features",
                        config.feature_extraction.maximum_sift_features);
        AssignIfPresent(features, "grid_columns", config.feature_extraction.grid_columns);
        AssignIfPresent(features, "grid_rows", config.feature_extraction.grid_rows);
        AssignIfPresent(features, "minimum_core_features_per_cell",
                        config.feature_extraction.minimum_core_features_per_cell);
        AssignIfPresent(features, "maximum_supplemental_features_per_cell",
                        config.feature_extraction.maximum_supplemental_features_per_cell);

        const YAML::Node matching = root["matching"];
        AssignIfPresent(matching, "ratio_threshold", config.feature_matching.ratio_threshold);
        AssignIfPresent(matching, "require_mutual_match",
                        config.feature_matching.require_mutual_match);
        AssignIfPresent(matching, "maximum_sampson_error_pixels",
                        config.feature_matching.maximum_sampson_error_pixels);
        AssignIfPresent(matching, "ransac_threshold_pixels",
                        config.visual_geometry.ransac_threshold_pixels);
        AssignIfPresent(matching, "minimum_inliers", config.visual_geometry.minimum_inliers);
        AssignIfPresent(matching, "minimum_inlier_ratio",
                        config.visual_geometry.minimum_inlier_ratio);
        AssignIfPresent(matching, "require_visual_geometry", config.require_visual_geometry);
        config.visual_geometry.random_seed = config.random_seed;

        const YAML::Node pairs = root["pairs"];
        AssignIfPresent(pairs, "maximum_temporal_gap", config.pair_selection.maximum_temporal_gap);
        AssignIfPresent(pairs, "minimum_loop_separation",
                        config.pair_selection.minimum_loop_separation);
        AssignIfPresent(pairs, "maximum_loop_distance",
                        config.pair_selection.maximum_loop_distance);
        AssignIfPresent(pairs, "maximum_loops_per_frame",
                        config.pair_selection.maximum_loops_per_frame);

        const YAML::Node recovery = root["recovery"];
        AssignIfPresent(recovery, "enabled", config.enable_weak_frame_recovery);
        AssignIfPresent(recovery, "target_track_observations",
                        config.recovery_target_track_observations);
        AssignIfPresent(recovery, "target_occupied_grid_cells",
                        config.recovery_target_occupied_grid_cells);
        AssignIfPresent(recovery, "minimum_track_length", config.recovery_minimum_track_length);
        AssignIfPresent(recovery, "temporal_gap", config.recovery_temporal_gap);
        AssignIfPresent(recovery, "spatial_distance", config.recovery_spatial_distance);
        AssignIfPresent(recovery, "spatial_pairs_per_frame",
                        config.recovery_spatial_pairs_per_frame);
        AssignIfPresent(recovery, "maximum_view_angle_degrees",
                        config.recovery_maximum_view_angle_degrees);

        const YAML::Node extension = root["track_extension"];
        AssignIfPresent(extension, "enabled", config.enable_track_extension);
        AssignIfPresent(extension, "search_radius_pixels",
                        config.track_extension_search_radius_pixels);
        AssignIfPresent(extension, "weak_search_radius_pixels",
                        config.track_extension_weak_search_radius_pixels);
        AssignIfPresent(extension, "descriptor_ratio", config.track_extension_descriptor_ratio);
        AssignIfPresent(extension, "descriptor_maximum", config.track_extension_descriptor_maximum);
        AssignIfPresent(extension, "epipolar_error_pixels",
                        config.track_extension_epipolar_error_pixels);
        AssignIfPresent(extension, "final_reprojection_error_pixels",
                        config.track_extension_final_reprojection_error_pixels);
        AssignIfPresent(extension, "maximum_depth", config.track_extension_maximum_depth);
        AssignIfPresent(extension, "weak_landmarks_per_camera",
                        config.track_extension_weak_landmarks_per_camera);

        const YAML::Node pnp = root["pnp_recovery"];
        AssignIfPresent(pnp, "enabled", config.enable_pnp_recovery);
        AssignIfPresent(pnp, "neighbor_gap", config.pnp_recovery_neighbor_gap);
        AssignIfPresent(pnp, "reprojection_error_pixels",
                        config.pnp_recovery_reprojection_error_pixels);
        AssignIfPresent(pnp, "minimum_inliers", config.pnp_recovery_minimum_inliers);

        const YAML::Node staged = root["staged_rebuild"];
        AssignIfPresent(staged, "enabled", config.enable_optimized_pose_rematching);

        const YAML::Node mapping = root["mapping"];
        AssignIfPresent(mapping, "minimum_track_length", config.track_builder.minimum_track_length);
        config.landmark_builder.minimum_track_length = config.track_builder.minimum_track_length;
        AssignIfPresent(mapping, "minimum_supplemental_track_length",
                        config.landmark_builder.minimum_supplemental_track_length);
        AssignIfPresent(mapping, "maximum_landmarks", config.landmark_builder.maximum_landmarks);
        AssignIfPresent(mapping, "minimum_landmarks_per_camera",
                        config.landmark_builder.minimum_landmarks_per_camera);
        AssignIfPresent(mapping, "minimum_parallax_degrees",
                        config.landmark_builder.triangulation.minimum_parallax_degrees);
        AssignIfPresent(mapping, "minimum_three_view_core_parallax_degrees",
                        config.landmark_builder.minimum_three_view_core_parallax_degrees);
        AssignIfPresent(mapping, "maximum_reprojection_error_pixels",
                        config.landmark_builder.triangulation.maximum_reprojection_error_pixels);
        AssignIfPresent(mapping, "prune_inconsistent_observations",
                        config.landmark_builder.prune_inconsistent_observations);
        AssignIfPresent(mapping, "observation_pruning_multiplier",
                        config.landmark_builder.observation_pruning_multiplier);

        const YAML::Node optimization = root["optimization"];
        AssignIfPresent(optimization, "minimum_registered_cameras",
                        config.incremental_optimization.minimum_registered_cameras);
        AssignIfPresent(optimization, "global_interval",
                        config.incremental_optimization.global_bundle_adjustment_interval);
        AssignIfPresent(optimization, "local_window_cameras",
                        config.incremental_optimization.local_bundle_adjustment.window_selection
                            .maximum_cameras);
        AssignIfPresent(
            optimization, "maximum_iterations",
            config.incremental_optimization.global_bundle_adjustment.maximum_iterations);
        config.incremental_optimization.local_bundle_adjustment.bundle_adjustment
            .maximum_iterations =
            config.incremental_optimization.global_bundle_adjustment.maximum_iterations;
        AssignIfPresent(optimization, "outlier_threshold_pixels",
                        config.final_refinement.maximum_reprojection_error_pixels);
        AssignIfPresent(optimization, "cleanup_cycles",
                        config.final_refinement.maximum_cleanup_cycles);
        config.final_refinement.bundle_adjustment =
            config.incremental_optimization.global_bundle_adjustment;
        config.incremental_optimization.global_bundle_adjustment.number_of_threads =
            config.maximum_cpu_threads;
        config.incremental_optimization.local_bundle_adjustment.bundle_adjustment
            .number_of_threads = config.maximum_cpu_threads;
        config.final_refinement.bundle_adjustment.number_of_threads = config.maximum_cpu_threads;

        config.image_list.image_directory = config.inputs.image_directory;
        config.visual_geometry.random_seed = config.random_seed;
        if (config.image_width <= 0 || config.image_height <= 0 ||
            config.maximum_cpu_threads <= 0 || !std::isfinite(config.camera_time_offset_seconds) ||
            config.feature_extraction.maximum_sift_features <= 0 ||
            config.feature_extraction.grid_columns <= 0 ||
            config.feature_extraction.grid_rows <= 0 ||
            config.track_builder.minimum_track_length < 2 ||
            config.recovery_target_track_observations == 0 ||
            config.recovery_target_occupied_grid_cells == 0 ||
            config.recovery_minimum_track_length < config.track_builder.minimum_track_length ||
            config.recovery_temporal_gap < config.pair_selection.maximum_temporal_gap ||
            !std::isfinite(config.recovery_spatial_distance) ||
            config.recovery_spatial_distance < 0.0 ||
            !std::isfinite(config.recovery_maximum_view_angle_degrees) ||
            config.recovery_maximum_view_angle_degrees < 0.0 ||
            config.recovery_maximum_view_angle_degrees > 180.0 ||
            !std::isfinite(config.track_extension_search_radius_pixels) ||
            config.track_extension_search_radius_pixels <= 0.0 ||
            !std::isfinite(config.track_extension_weak_search_radius_pixels) ||
            config.track_extension_weak_search_radius_pixels <= 0.0 ||
            !std::isfinite(config.track_extension_descriptor_ratio) ||
            config.track_extension_descriptor_ratio <= 0.0 ||
            config.track_extension_descriptor_ratio >= 1.0 ||
            !std::isfinite(config.track_extension_descriptor_maximum) ||
            config.track_extension_descriptor_maximum <= 0.0 ||
            !std::isfinite(config.track_extension_epipolar_error_pixels) ||
            config.track_extension_epipolar_error_pixels <= 0.0 ||
            !std::isfinite(config.track_extension_final_reprojection_error_pixels) ||
            config.track_extension_final_reprojection_error_pixels <= 0.0 ||
            !std::isfinite(config.track_extension_maximum_depth) ||
            config.track_extension_maximum_depth <= 0.0 ||
            config.track_extension_weak_landmarks_per_camera == 0 ||
            config.pnp_recovery_neighbor_gap == 0 ||
            !std::isfinite(config.pnp_recovery_reprojection_error_pixels) ||
            config.pnp_recovery_reprojection_error_pixels <= 0.0 ||
            config.pnp_recovery_minimum_inliers < 4)
        {
            return Result<PipelineConfig>::Failure(Status::Error(
                ErrorCode::kInvalidArgument, "pipeline configuration contains invalid values"));
        }
        return Result<PipelineConfig>::Success(std::move(config));
    }
    catch (const YAML::Exception& exception)
    {
        return Result<PipelineConfig>::Failure(
            Status::Error(ErrorCode::kParseError,
                          "failed to parse pipeline YAML: " + std::string(exception.what())));
    }
}

} // namespace lio_visual_ba

// ---- Pipeline ----

namespace lio_visual_ba
{
namespace
{

Status WriteIntrinsics(const std::filesystem::path& path, const CameraIntrinsics& intrinsics)
{
    std::ofstream output(path);
    if (!output)
    {
        return Status::Error(ErrorCode::kIoError, "failed to create intrinsics output");
    }
    output.precision(17);
    output << intrinsics.fx << ",0," << intrinsics.cx << '\n'
           << "0," << intrinsics.fy << ',' << intrinsics.cy << "\n0,0,1\n";
    if (!output)
    {
        return Status::Error(ErrorCode::kIoError, "failed to write intrinsics output");
    }
    return Status::Ok();
}

Status WriteLegacyTracks(const std::filesystem::path& path, const std::vector<Landmark>& landmarks)
{
    std::ofstream output(path);
    if (!output)
    {
        return Status::Error(ErrorCode::kIoError, "failed to create legacy tracks output");
    }
    output << "track,frame,u,v\n" << std::setprecision(17);
    for (const Landmark& landmark : landmarks)
    {
        for (const FeatureObservation& observation : landmark.observations)
        {
            output << landmark.id.value() << ',' << observation.camera_id.value() << ','
                   << observation.pixel.x() << ',' << observation.pixel.y() << '\n';
        }
    }
    return output ? Status::Ok()
                  : Status::Error(ErrorCode::kIoError, "failed to write legacy tracks output");
}

Status WriteHeldoutMatches(const std::filesystem::path& path,
                           const std::vector<HeldoutMatchRecord>& matches)
{
    std::ofstream output(path);
    if (!output)
    {
        return Status::Error(ErrorCode::kIoError, "failed to create held-out match output");
    }
    output << "frame1,frame2,u1,v1,u2,v2\n" << std::setprecision(17);
    for (const HeldoutMatchRecord& match : matches)
    {
        output << match.first_camera_id.value() << ',' << match.second_camera_id.value() << ',';
        output << match.first_pixel.x() << ',' << match.first_pixel.y() << ','
               << match.second_pixel.x() << ',' << match.second_pixel.y() << '\n';
    }
    return output ? Status::Ok()
                  : Status::Error(ErrorCode::kIoError, "failed to write held-out match output");
}

std::vector<double> ReprojectionErrors(const Reconstruction& reconstruction)
{
    std::map<CameraId, const CameraFrame*> cameras;
    for (const CameraFrame& camera : reconstruction.cameras)
    {
        cameras.emplace(camera.id, &camera);
    }
    std::vector<double> errors;
    for (const Landmark& landmark : reconstruction.landmarks)
    {
        for (const FeatureObservation& observation : landmark.observations)
        {
            const CameraFrame& camera = *cameras.at(observation.camera_id);
            const Pose3d& pose = camera.has_optimized_pose ? camera.optimized_world_from_camera
                                                           : camera.initial_world_from_camera;
            auto projection = geometry::ProjectWorldPoint(reconstruction.intrinsics, pose,
                                                          landmark.position_world);
            if (projection.ok())
            {
                errors.push_back((projection.value().pixel - observation.pixel).norm());
            }
        }
    }
    std::sort(errors.begin(), errors.end());
    return errors;
}

double Percentile(const std::vector<double>& sorted, double fraction)
{
    if (sorted.empty())
    {
        return 0.0;
    }
    const double position = fraction * static_cast<double>(sorted.size() - 1);
    const std::size_t lower = static_cast<std::size_t>(std::floor(position));
    const std::size_t upper = static_cast<std::size_t>(std::ceil(position));
    const double weight = position - static_cast<double>(lower);
    return sorted[lower] * (1.0 - weight) + sorted[upper] * weight;
}

} // namespace

Status Pipeline::PublishPipelineOutputs(const std::filesystem::path& output_directory,
                                        const PipelineRunResult& result)
{
    if (output_directory.empty())
    {
        return Status::Error(ErrorCode::kInvalidArgument, "pipeline output directory is empty");
    }
    std::error_code error;
    if (std::filesystem::exists(output_directory, error) &&
        !std::filesystem::is_empty(output_directory, error))
    {
        return Status::Error(ErrorCode::kFailedPrecondition,
                             "pipeline output directory must be absent or empty");
    }
    const auto suffix = std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
    const std::filesystem::path staging =
        output_directory.parent_path() / (output_directory.filename().string() + ".tmp-" + suffix);
    std::filesystem::create_directories(staging, error);
    if (error)
    {
        return Status::Error(ErrorCode::kIoError, "failed to create staged output directory");
    }
    const auto cleanup = [&staging]()
    {
        std::error_code ignored;
        std::filesystem::remove_all(staging, ignored);
    };
    const Reconstruction& reconstruction = result.backend.reconstruction;
    const auto require = [&cleanup](const Status& status)
    {
        if (!status.ok())
        {
            cleanup();
        }
        return status;
    };
    Status status = ReconstructionExporter::WriteTumCameraPoses(staging / "poses_lio_prior_tum.txt",
                                                                reconstruction.cameras, false);
    if (!status.ok())
    {
        return require(status);
    }
    status = ReconstructionExporter::WriteTumCameraPoses(staging / "poses_optimized_tum.txt",
                                                         reconstruction.cameras, true);
    if (!status.ok())
    {
        return require(status);
    }
    status = ReconstructionExporter::WriteLandmarksPly(staging / "landmarks_optimized.ply",
                                                       reconstruction.landmarks);
    if (!status.ok())
    {
        return require(status);
    }
    status = ReconstructionExporter::WriteLandmarksPly(staging / "sparse_points.ply",
                                                       reconstruction.landmarks);
    if (!status.ok())
    {
        return require(status);
    }
    status = ReconstructionExporter::WriteTracksCsv(staging / "track_summary.csv",
                                                    reconstruction.tracks);
    if (!status.ok())
    {
        return require(status);
    }
    status = WriteLegacyTracks(staging / "tracks.csv", reconstruction.landmarks);
    if (!status.ok())
    {
        return require(status);
    }
    status = WriteHeldoutMatches(staging / "heldout.csv", result.heldout_match_data);
    if (!status.ok())
    {
        return require(status);
    }
    status = ReconstructionExporter::WriteObservationsCsv(staging / "observations.csv",
                                                          reconstruction.landmarks);
    if (!status.ok())
    {
        return require(status);
    }
    status = WriteIntrinsics(staging / "intrinsics_refined.txt", reconstruction.intrinsics);
    if (!status.ok())
    {
        return require(status);
    }
    ColmapTextOptions colmap_options;
    colmap_options.minimum_observations_per_camera = 1;
    status = ReconstructionExporter::WriteColmapTextReconstruction(staging / "sparse/0",
                                                                   reconstruction, colmap_options);
    if (!status.ok())
    {
        return require(status);
    }
    auto validation =
        ReconstructionExporter::ValidateColmapTextReconstruction(staging / "sparse/0");
    if (!validation.ok())
    {
        cleanup();
        return validation.status().WithContext("published COLMAP validation");
    }
    const std::vector<double> reprojection_errors = ReprojectionErrors(reconstruction);
    double squared_error = 0.0;
    for (double value : reprojection_errors)
    {
        squared_error += value * value;
    }
    const double reprojection_rmse =
        reprojection_errors.empty()
            ? 0.0
            : std::sqrt(squared_error / static_cast<double>(reprojection_errors.size()));
    nlohmann::json metrics = {
        {"pass", true},
        {"cameras", reconstruction.cameras.size()},
        {"images", reconstruction.cameras.size()},
        {"sparse_export_images", validation.value().images},
        {"sparse_excluded_images", reconstruction.cameras.size() - validation.value().images},
        {"landmarks", reconstruction.landmarks.size()},
        {"observations", validation.value().observations},
        {"tracks", reconstruction.tracks.size()},
        {"features", result.extracted_features},
        {"candidate_pairs", result.candidate_pairs},
        {"accepted_pairs", result.accepted_pairs},
        {"rejected_pairs", result.rejected_pairs},
        {"verified_matches", result.verified_matches},
        {"heldout_matches", result.heldout_matches},
        {"weak_cameras_before_recovery", result.weak_cameras_before_recovery},
        {"recovery_candidate_pairs", result.recovery_candidate_pairs},
        {"recovery_accepted_pairs", result.recovery_accepted_pairs},
        {"recovery_added_matches", result.recovery_added_matches},
        {"track_extension_proposed_observations",
         result.backend.track_extension_proposed_observations},
        {"track_extension_retained_observations",
         result.backend.track_extension_retained_observations},
        {"track_extension_rejected_observations",
         result.backend.track_extension_rejected_observations},
        {"track_extension_assignment_conflicts",
         result.backend.track_extension_assignment_conflicts},
        {"pnp_recovery_target_cameras", result.backend.pnp_recovery_target_cameras},
        {"pnp_recovery_recovered_cameras", result.backend.pnp_recovery_recovered_cameras},
        {"pnp_recovery_added_observations", result.backend.pnp_recovery_added_observations},
        {"local_ba_solves", result.backend.incremental_optimization.local_solve_count},
        {"skipped_local_ba_solves",
         result.backend.incremental_optimization.skipped_local_solve_count},
        {"global_ba_solves", result.backend.incremental_optimization.global_solve_count},
        {"cleanup_cycles", result.backend.final_refinement.cleanup_cycles},
        {"removed_observations", result.backend.final_refinement.removed_observations},
        {"removed_landmarks", result.backend.final_refinement.removed_landmarks},
        {"reprojection_rmse_px", reprojection_rmse},
        {"reprojection_median_px", Percentile(reprojection_errors, 0.5)},
        {"reprojection_p90_px", Percentile(reprojection_errors, 0.9)},
        {"reprojection_p95_px", Percentile(reprojection_errors, 0.95)},
        {"reprojection_max_px", reprojection_errors.empty() ? 0.0 : reprojection_errors.back()}};
    status = ReconstructionExporter::WriteMetricsJson(staging / "evaluation.json", metrics);
    if (!status.ok())
    {
        return require(status);
    }

    if (std::filesystem::exists(output_directory, error))
    {
        std::filesystem::remove(output_directory, error);
    }
    std::filesystem::rename(staging, output_directory, error);
    if (error)
    {
        cleanup();
        return Status::Error(ErrorCode::kIoError, "failed to publish staged pipeline outputs");
    }
    return Status::Ok();
}

Result<PipelineRunResult> Pipeline::RunPipeline(const PipelineConfig& config)
{
    cv::setNumThreads(config.maximum_cpu_threads);
    auto frontend = Pipeline::RunPipelineFrontend(config);
    if (!frontend.ok())
    {
        return Result<PipelineRunResult>::Failure(frontend.status());
    }
    if (config.enable_optimized_pose_rematching)
    {
        PipelineConfig preliminary_config = config;
        preliminary_config.enable_optimized_pose_rematching = false;
        preliminary_config.enable_pnp_recovery = false;
        preliminary_config.enable_track_extension = false;
        auto preliminary =
            Pipeline::RunPipelineBackend(std::move(frontend.value()), preliminary_config);
        if (!preliminary.ok())
        {
            return Result<PipelineRunResult>::Failure(
                preliminary.status().WithContext("preliminary staged reconstruction"));
        }
        auto rematched =
            Pipeline::RunPipelineRematching(std::move(preliminary.value().reconstruction),
                                            std::move(preliminary.value().feature_sets), config);
        if (!rematched.ok())
        {
            return Result<PipelineRunResult>::Failure(
                rematched.status().WithContext("optimized-pose staged rebuild"));
        }
        frontend = std::move(rematched);
    }
    PipelineRunResult result;
    for (const FeatureSet& features : frontend.value().feature_sets)
    {
        result.extracted_features += features.features.size();
    }
    result.candidate_pairs = frontend.value().candidate_pairs.all_pairs.size();
    result.accepted_pairs = frontend.value().accepted_candidate_pairs;
    result.rejected_pairs = frontend.value().rejected_candidate_pairs;
    result.verified_matches = frontend.value().verified_matches.size();
    result.heldout_matches = frontend.value().heldout_matches.size();
    result.weak_cameras_before_recovery = frontend.value().weak_cameras_before_recovery;
    result.recovery_candidate_pairs = frontend.value().recovery_candidate_pairs;
    result.recovery_accepted_pairs = frontend.value().recovery_accepted_pairs;
    result.recovery_added_matches = frontend.value().recovery_added_matches;
    std::map<CameraId, const FeatureSet*> feature_sets;
    for (const FeatureSet& features : frontend.value().feature_sets)
    {
        feature_sets.emplace(features.camera_id, &features);
    }
    for (const FeatureMatch& match : frontend.value().heldout_matches)
    {
        const Feature& first =
            feature_sets.at(match.first_camera_id)
                ->features.at(static_cast<std::size_t>(match.first_feature_id.value()));
        const Feature& second =
            feature_sets.at(match.second_camera_id)
                ->features.at(static_cast<std::size_t>(match.second_feature_id.value()));
        result.heldout_match_data.push_back(
            {match.first_camera_id, match.second_camera_id, first.pixel, second.pixel});
    }
    auto backend = Pipeline::RunPipelineBackend(std::move(frontend.value()), config);
    if (!backend.ok())
    {
        return Result<PipelineRunResult>::Failure(backend.status());
    }
    result.backend = std::move(backend.value());
    const Status publish_status = Pipeline::PublishPipelineOutputs(config.output_directory, result);
    if (!publish_status.ok())
    {
        return Result<PipelineRunResult>::Failure(
            publish_status.WithContext("pipeline output publication"));
    }
    return Result<PipelineRunResult>::Success(std::move(result));
}

} // namespace lio_visual_ba
