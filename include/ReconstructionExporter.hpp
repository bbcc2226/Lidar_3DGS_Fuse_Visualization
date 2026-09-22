#pragma once

// Sparse artifact writers and independent COLMAP validation. ID conversion must
// keep image observations and point tracks consistent in both directions.

#include "Status.hpp"
#include "Types.hpp"
#include <cstddef>
#include <filesystem>
#include <nlohmann/json_fwd.hpp>
#include <vector>

namespace lio_visual_ba
{

struct ColmapTextOptions
{
    bool prefer_optimized_camera_poses = true;
    std::size_t minimum_observations_per_camera = 0;
};

struct ColmapValidationReport
{
    std::size_t cameras = 0;
    std::size_t images = 0;
    std::size_t points3d = 0;
    std::size_t observations = 0;
};

// Stateless component API. Options and data are supplied explicitly per call.
class ReconstructionExporter
{
  public:
    static Status WriteTumCameraPoses(const std::filesystem::path& output_file,
                                      const std::vector<CameraFrame>& cameras,
                                      bool write_optimized_poses);

    static Status WriteLandmarksPly(const std::filesystem::path& output_file,
                                    const std::vector<Landmark>& landmarks);

    static Status WriteTracksCsv(const std::filesystem::path& output_file,
                                 const std::vector<FeatureTrack>& tracks);

    static Status WriteObservationsCsv(const std::filesystem::path& output_file,
                                       const std::vector<Landmark>& landmarks);

    static Status WriteMetricsJson(const std::filesystem::path& output_file,
                                   const nlohmann::json& metrics);

    // Write cameras.txt, images.txt, and points3D.txt into output_directory.
    // COLMAP entity IDs are one-based; POINT2D_IDX values are zero-based.
    static Status WriteColmapTextReconstruction(const std::filesystem::path& output_directory,
                                                const Reconstruction& reconstruction,
                                                const ColmapTextOptions& options = {});

    // Independently parse and validate cameras.txt, images.txt, and points3D.txt.
    static Result<ColmapValidationReport>
    ValidateColmapTextReconstruction(const std::filesystem::path& reconstruction_directory);
};

} // namespace lio_visual_ba
