// Image features, pair matching, binary caches, and conflict-safe tracks.
// Feature IDs must remain aligned with descriptor rows throughout these stages.

#include "feature_processor.hpp"
#include "geometry.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <mutex>
#include <numeric>
#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <set>
#include <string>
#include <system_error>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

// ---- Feature Extractor ----

namespace lio_visual_ba
{
namespace
{

Status ValidateOptions(const FeatureExtractionOptions& options)
{
    if (options.maximum_sift_features <= 0 || options.grid_columns <= 0 || options.grid_rows <= 0 ||
        options.minimum_core_features_per_cell < 0 ||
        options.maximum_supplemental_features_per_cell < 0 ||
        !std::isfinite(options.supplemental_quality_level) ||
        options.supplemental_quality_level <= 0.0 || options.supplemental_quality_level > 1.0 ||
        !std::isfinite(options.supplemental_minimum_distance_pixels) ||
        options.supplemental_minimum_distance_pixels < 0.0)
    {
        return Status::Error(ErrorCode::kInvalidArgument, "invalid feature extraction options");
    }
    return Status::Ok();
}

Feature ConvertKeypoint(const cv::KeyPoint& keypoint, std::size_t index)
{
    Feature feature;
    feature.id = FeatureId(static_cast<FeatureId::ValueType>(index));
    feature.pixel = Eigen::Vector2d(keypoint.pt.x, keypoint.pt.y);
    feature.scale = keypoint.size;
    feature.angle_degrees = keypoint.angle;
    feature.response = keypoint.response;
    feature.octave = keypoint.octave;
    return feature;
}

} // namespace

Result<FeatureSet> FeatureProcessor::ExtractFeatures(CameraId camera_id,
                                                     const std::filesystem::path& image_file,
                                                     const FeatureExtractionOptions& options)
{
    if (image_file.empty())
    {
        return Result<FeatureSet>::Failure(
            Status::Error(ErrorCode::kInvalidArgument, "image file path is empty"));
    }
    const cv::Mat image = cv::imread(image_file.string(), cv::IMREAD_UNCHANGED);
    if (image.empty())
    {
        return Result<FeatureSet>::Failure(
            Status::Error(ErrorCode::kNotFound, "cannot decode image: " + image_file.string()));
    }
    auto result = FeatureProcessor::ExtractFeatures(camera_id, image, options);
    if (!result.ok())
    {
        return Result<FeatureSet>::Failure(result.status().WithContext(image_file.string()));
    }
    return result;
}

Result<FeatureSet> FeatureProcessor::ExtractFeatures(CameraId camera_id, const cv::Mat& image,
                                                     const FeatureExtractionOptions& options)
{
    if (!camera_id.valid())
    {
        return Result<FeatureSet>::Failure(
            Status::Error(ErrorCode::kInvalidArgument, "camera ID is invalid"));
    }
    const Status options_status = ValidateOptions(options);
    if (!options_status.ok())
    {
        return Result<FeatureSet>::Failure(options_status);
    }
    if (image.empty() || image.dims != 2 || image.depth() != CV_8U ||
        (image.channels() != 1 && image.channels() != 3 && image.channels() != 4))
    {
        return Result<FeatureSet>::Failure(
            Status::Error(ErrorCode::kInvalidArgument,
                          "image must be a non-empty 8-bit grayscale, BGR, or BGRA matrix"));
    }
    cv::Mat grayscale;
    if (image.channels() == 1)
    {
        grayscale = image;
    }
    else
    {
        cv::cvtColor(image, grayscale,
                     image.channels() == 3 ? cv::COLOR_BGR2GRAY : cv::COLOR_BGRA2GRAY);
    }

    auto sift = cv::SIFT::create(options.maximum_sift_features);
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    sift->detectAndCompute(grayscale, cv::noArray(), keypoints, descriptors);
    const std::size_t core_count = keypoints.size();

    cv::Mat available(grayscale.size(), CV_8U, cv::Scalar(255));
    for (const cv::KeyPoint& keypoint : keypoints)
    {
        cv::circle(available, keypoint.pt, 5, cv::Scalar(0), cv::FILLED);
    }
    std::vector<cv::KeyPoint> supplemental;
    for (std::int32_t row = 0; row < options.grid_rows; ++row)
    {
        for (std::int32_t column = 0; column < options.grid_columns; ++column)
        {
            const int x0 = column * grayscale.cols / options.grid_columns;
            const int x1 = (column + 1) * grayscale.cols / options.grid_columns;
            const int y0 = row * grayscale.rows / options.grid_rows;
            const int y1 = (row + 1) * grayscale.rows / options.grid_rows;
            const float x0_float = static_cast<float>(x0);
            const float x1_float = static_cast<float>(x1);
            const float y0_float = static_cast<float>(y0);
            const float y1_float = static_cast<float>(y1);
            std::int32_t core_in_cell = 0;
            for (const cv::KeyPoint& keypoint : keypoints)
            {
                if (keypoint.pt.x >= x0_float && keypoint.pt.x < x1_float &&
                    keypoint.pt.y >= y0_float && keypoint.pt.y < y1_float)
                {
                    ++core_in_cell;
                }
            }
            if (core_in_cell >= options.minimum_core_features_per_cell ||
                options.maximum_supplemental_features_per_cell == 0)
            {
                continue;
            }
            std::vector<cv::Point2f> corners;
            const cv::Rect region(x0, y0, x1 - x0, y1 - y0);
            cv::goodFeaturesToTrack(
                grayscale(region), corners, options.maximum_supplemental_features_per_cell,
                options.supplemental_quality_level, options.supplemental_minimum_distance_pixels,
                available(region), 5);
            for (const cv::Point2f& corner : corners)
            {
                supplemental.emplace_back(corner.x + static_cast<float>(x0),
                                          corner.y + static_cast<float>(y0), 5.0F);
            }
        }
    }
    if (!supplemental.empty())
    {
        cv::Mat supplemental_descriptors;
        sift->compute(grayscale, supplemental, supplemental_descriptors);
        if (!supplemental_descriptors.empty())
        {
            keypoints.insert(keypoints.end(), supplemental.begin(), supplemental.end());
            if (descriptors.empty())
            {
                descriptors = supplemental_descriptors;
            }
            else
            {
                cv::vconcat(descriptors, supplemental_descriptors, descriptors);
            }
        }
    }

    FeatureSet result;
    result.camera_id = camera_id;
    result.core_feature_count = core_count;
    result.descriptors = descriptors;
    result.features.reserve(keypoints.size());
    for (std::size_t index = 0; index < keypoints.size(); ++index)
    {
        result.features.push_back(ConvertKeypoint(keypoints[index], index));
    }
    if (result.descriptors.rows != static_cast<int>(result.features.size()) ||
        (!result.descriptors.empty() &&
         (result.descriptors.type() != CV_32F || result.descriptors.cols != 128)))
    {
        return Result<FeatureSet>::Failure(
            Status::Error(ErrorCode::kInternal, "SIFT returned inconsistent descriptors"));
    }
    return Result<FeatureSet>::Success(std::move(result));
}

} // namespace lio_visual_ba

// ---- Feature Cache ----

namespace lio_visual_ba
{
namespace
{

constexpr std::array<std::uint8_t, 8> kMagic{{'L', 'V', 'B', 'A', 'F', 'T', 'R', '1'}};
constexpr std::uint32_t kVersion = 1;
constexpr std::uint32_t kEndianMarker = 0x01020304U;
constexpr std::uint64_t kMaximumFeatures = 10'000'000;
constexpr std::uint64_t kMaximumPayloadBytes = 8ULL * 1024ULL * 1024ULL * 1024ULL;

template <typename T> void Append(std::vector<std::uint8_t>& bytes, const T& value)
{
    static_assert(std::is_trivially_copyable<T>::value, "binary field required");
    const auto* begin = reinterpret_cast<const std::uint8_t*>(&value);
    bytes.insert(bytes.end(), begin, begin + sizeof(T));
}

template <typename T>
bool Read(const std::vector<std::uint8_t>& bytes, std::size_t& offset, T& value)
{
    if (offset > bytes.size() || sizeof(T) > bytes.size() - offset)
        return false;
    std::memcpy(&value, bytes.data() + offset, sizeof(T));
    offset += sizeof(T);
    return true;
}

std::uint64_t Checksum(const std::vector<std::uint8_t>& bytes)
{
    std::uint64_t hash = 14695981039346656037ULL;
    for (const std::uint8_t byte : bytes)
    {
        hash ^= byte;
        hash *= 1099511628211ULL;
    }
    return hash;
}

Status ValidateFeatureSet(const FeatureSet& set)
{
    if (!set.camera_id.valid() || set.core_feature_count > set.features.size() ||
        set.features.size() > kMaximumFeatures)
    {
        return Status::Error(ErrorCode::kInvalidArgument, "feature set has invalid IDs or counts");
    }
    if (set.descriptors.rows != static_cast<int>(set.features.size()) ||
        (!set.descriptors.empty() &&
         (set.descriptors.type() != CV_32F || set.descriptors.cols != 128)))
    {
        return Status::Error(ErrorCode::kInvalidArgument,
                             "feature descriptors must be contiguous SIFT rows");
    }
    for (std::size_t index = 0; index < set.features.size(); ++index)
    {
        const Feature& feature = set.features[index];
        if (feature.id != FeatureId(static_cast<FeatureId::ValueType>(index)) ||
            !feature.pixel.array().isFinite().all())
        {
            return Status::Error(ErrorCode::kInvalidArgument,
                                 "feature IDs must be stable and pixels finite");
        }
    }
    return Status::Ok();
}

} // namespace

Status FeatureProcessor::SaveFeatureCache(const std::filesystem::path& cache_file,
                                          const FeatureSet& features,
                                          const FeatureCacheFingerprint& fingerprint)
{
    if (cache_file.empty())
        return Status::Error(ErrorCode::kInvalidArgument, "cache path is empty");
    const Status valid = ValidateFeatureSet(features);
    if (!valid.ok())
        return valid;
    cv::Mat descriptors = features.descriptors;
    if (!descriptors.empty() && !descriptors.isContinuous())
        descriptors = descriptors.clone();

    std::vector<std::uint8_t> payload;
    payload.reserve(features.features.size() * (32U + 128U * sizeof(float)));
    for (const Feature& feature : features.features)
    {
        Append(payload, feature.pixel.x());
        Append(payload, feature.pixel.y());
        Append(payload, feature.scale);
        Append(payload, feature.angle_degrees);
        Append(payload, feature.response);
        Append(payload, feature.octave);
    }
    if (!descriptors.empty())
    {
        const auto* begin = descriptors.ptr<std::uint8_t>();
        payload.insert(payload.end(), begin, begin + descriptors.total() * descriptors.elemSize());
    }
    if (payload.size() > kMaximumPayloadBytes)
    {
        return Status::Error(ErrorCode::kInvalidArgument, "feature cache payload is too large");
    }

    std::vector<std::uint8_t> file_bytes(kMagic.begin(), kMagic.end());
    const std::uint64_t count = features.features.size();
    const std::uint64_t core_count = features.core_feature_count;
    const std::uint32_t descriptor_columns =
        descriptors.empty() ? 128U : static_cast<std::uint32_t>(descriptors.cols);
    const std::uint64_t payload_size = payload.size();
    Append(file_bytes, kVersion);
    Append(file_bytes, kEndianMarker);
    Append(file_bytes, fingerprint.source);
    Append(file_bytes, fingerprint.configuration);
    Append(file_bytes, features.camera_id.value());
    Append(file_bytes, count);
    Append(file_bytes, core_count);
    Append(file_bytes, descriptor_columns);
    Append(file_bytes, payload_size);
    Append(file_bytes, Checksum(payload));
    file_bytes.insert(file_bytes.end(), payload.begin(), payload.end());

    std::error_code error;
    if (!cache_file.parent_path().empty())
        std::filesystem::create_directories(cache_file.parent_path(), error);
    if (error)
        return Status::Error(ErrorCode::kIoError,
                             "cannot create cache directory: " + error.message());
    const auto suffix = std::chrono::steady_clock::now().time_since_epoch().count();
    const std::filesystem::path temporary = cache_file.string() + ".tmp." + std::to_string(suffix);
    {
        std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
        if (!output)
            return Status::Error(ErrorCode::kIoError,
                                 "cannot open temporary cache: " + temporary.string());
        output.write(reinterpret_cast<const char*>(file_bytes.data()),
                     static_cast<std::streamsize>(file_bytes.size()));
        if (!output)
            return Status::Error(ErrorCode::kIoError,
                                 "cannot write temporary cache: " + temporary.string());
    }
    std::filesystem::rename(temporary, cache_file, error);
    if (error)
    {
        std::filesystem::remove(temporary);
        return Status::Error(ErrorCode::kIoError,
                             "cannot replace feature cache: " + error.message());
    }
    return Status::Ok();
}

Result<FeatureSet> FeatureProcessor::LoadFeatureCache(const std::filesystem::path& cache_file,
                                                      const FeatureCacheFingerprint& expected)
{
    std::ifstream input(cache_file, std::ios::binary | std::ios::ate);
    if (!input)
        return Result<FeatureSet>::Failure(Status::Error(
            ErrorCode::kNotFound, "cannot open feature cache: " + cache_file.string()));
    const std::streamsize size = input.tellg();
    if (size < 0 || static_cast<std::uint64_t>(size) > kMaximumPayloadBytes + 256U)
        return Result<FeatureSet>::Failure(
            Status::Error(ErrorCode::kDataLoss, "invalid feature cache size"));
    input.seekg(0);
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(size));
    input.read(reinterpret_cast<char*>(bytes.data()), size);
    if (!input)
        return Result<FeatureSet>::Failure(
            Status::Error(ErrorCode::kIoError, "failed to read feature cache"));
    std::size_t offset = 0;
    if (bytes.size() < kMagic.size() || !std::equal(kMagic.begin(), kMagic.end(), bytes.begin()))
        return Result<FeatureSet>::Failure(
            Status::Error(ErrorCode::kDataLoss, "feature cache magic mismatch"));
    offset += kMagic.size();
    std::uint32_t version = 0, endian = 0, columns = 0;
    std::uint64_t source = 0, config = 0, count = 0, core = 0, payload_size = 0, checksum = 0;
    std::int32_t camera = -1;
    if (!Read(bytes, offset, version) || !Read(bytes, offset, endian) ||
        !Read(bytes, offset, source) || !Read(bytes, offset, config) ||
        !Read(bytes, offset, camera) || !Read(bytes, offset, count) || !Read(bytes, offset, core) ||
        !Read(bytes, offset, columns) || !Read(bytes, offset, payload_size) ||
        !Read(bytes, offset, checksum))
        return Result<FeatureSet>::Failure(
            Status::Error(ErrorCode::kDataLoss, "truncated feature cache header"));
    if (version != kVersion || endian != kEndianMarker)
        return Result<FeatureSet>::Failure(Status::Error(
            ErrorCode::kFailedPrecondition, "unsupported feature cache schema or endianness"));
    if (source != expected.source || config != expected.configuration)
        return Result<FeatureSet>::Failure(
            Status::Error(ErrorCode::kFailedPrecondition, "feature cache fingerprint mismatch"));
    if (camera < 0 || count > kMaximumFeatures || core > count || columns != 128U ||
        payload_size != bytes.size() - offset)
        return Result<FeatureSet>::Failure(
            Status::Error(ErrorCode::kDataLoss, "invalid feature cache metadata"));
    std::vector<std::uint8_t> payload(bytes.begin() + static_cast<std::ptrdiff_t>(offset),
                                      bytes.end());
    if (Checksum(payload) != checksum)
        return Result<FeatureSet>::Failure(
            Status::Error(ErrorCode::kDataLoss, "feature cache checksum mismatch"));
    const std::uint64_t expected_size = count * (32U + columns * sizeof(float));
    if (payload_size != expected_size)
        return Result<FeatureSet>::Failure(
            Status::Error(ErrorCode::kDataLoss, "feature cache payload length mismatch"));

    FeatureSet result;
    result.camera_id = CameraId(camera);
    result.core_feature_count = static_cast<std::size_t>(core);
    result.features.reserve(static_cast<std::size_t>(count));
    std::size_t payload_offset = 0;
    for (std::uint64_t index = 0; index < count; ++index)
    {
        Feature feature;
        feature.id = FeatureId(static_cast<FeatureId::ValueType>(index));
        if (!Read(payload, payload_offset, feature.pixel.x()) ||
            !Read(payload, payload_offset, feature.pixel.y()) ||
            !Read(payload, payload_offset, feature.scale) ||
            !Read(payload, payload_offset, feature.angle_degrees) ||
            !Read(payload, payload_offset, feature.response) ||
            !Read(payload, payload_offset, feature.octave))
            return Result<FeatureSet>::Failure(
                Status::Error(ErrorCode::kDataLoss, "truncated feature records"));
        result.features.push_back(feature);
    }
    if (count > 0)
    {
        result.descriptors.create(static_cast<int>(count), static_cast<int>(columns), CV_32F);
        std::memcpy(result.descriptors.data, payload.data() + payload_offset,
                    static_cast<std::size_t>(count * columns * sizeof(float)));
    }
    const Status valid = ValidateFeatureSet(result);
    if (!valid.ok())
        return Result<FeatureSet>::Failure(valid.WithContext("invalid cached features"));
    return Result<FeatureSet>::Success(std::move(result));
}

} // namespace lio_visual_ba

// ---- Feature Matcher ----

namespace lio_visual_ba
{
namespace
{
std::mutex& OpenCvRngMutex()
{
    static std::mutex mutex;
    return mutex;
}
Status Validate(const FeatureSet& set)
{
    if (!set.camera_id.valid() || set.descriptors.rows != static_cast<int>(set.features.size()) ||
        (!set.descriptors.empty() &&
         (set.descriptors.type() != CV_32F || set.descriptors.cols != 128)))
        return Status::Error(ErrorCode::kInvalidArgument, "invalid SIFT feature set");
    for (std::size_t i = 0; i < set.features.size(); ++i)
        if (set.features[i].id != FeatureId(static_cast<FeatureId::ValueType>(i)) ||
            !set.features[i].pixel.array().isFinite().all())
            return Status::Error(ErrorCode::kInvalidArgument,
                                 "unstable feature ID or non-finite pixel");
    return Status::Ok();
}
Result<std::map<int, cv::DMatch>> Ratio(const cv::Mat& query, const cv::Mat& train, double ratio)
{
    std::map<int, cv::DMatch> accepted;
    if (query.empty() || train.rows < 2)
        return Result<std::map<int, cv::DMatch>>::Success(accepted);
    cv::BFMatcher matcher(cv::NORM_L2);
    std::vector<std::vector<cv::DMatch>> neighbors;
    matcher.knnMatch(query, train, neighbors, 2);
    for (const auto& pair : neighbors)
        if (pair.size() == 2 && pair[0].distance < static_cast<float>(ratio) * pair[1].distance)
            accepted.emplace(pair[0].queryIdx, pair[0]);
    return Result<std::map<int, cv::DMatch>>::Success(std::move(accepted));
}
} // namespace
Result<std::vector<FeatureMatch>>
FeatureProcessor::MatchFeatureDescriptors(const FeatureSet& first, const FeatureSet& second,
                                          const FeatureMatchingOptions& options)
{
    const Status a = Validate(first), b = Validate(second);
    if (!a.ok())
        return Result<std::vector<FeatureMatch>>::Failure(a);
    if (!b.ok())
        return Result<std::vector<FeatureMatch>>::Failure(b);
    if (first.camera_id == second.camera_id || !std::isfinite(options.ratio_threshold) ||
        options.ratio_threshold <= 0.0 || options.ratio_threshold >= 1.0 ||
        !std::isfinite(options.maximum_sampson_error_pixels) ||
        options.maximum_sampson_error_pixels < 0.0)
        return Result<std::vector<FeatureMatch>>::Failure(
            Status::Error(ErrorCode::kInvalidArgument, "invalid matching inputs or thresholds"));
    auto forward = Ratio(first.descriptors, second.descriptors, options.ratio_threshold);
    auto reverse = Ratio(second.descriptors, first.descriptors, options.ratio_threshold);
    std::vector<FeatureMatch> result;
    for (const auto& entry : forward.value())
    {
        const cv::DMatch& m = entry.second;
        if (options.require_mutual_match)
        {
            const auto it = reverse.value().find(m.trainIdx);
            if (it == reverse.value().end() || it->second.trainIdx != m.queryIdx)
                continue;
        }
        FeatureMatch out;
        out.first_camera_id = first.camera_id;
        out.first_feature_id = FeatureId(m.queryIdx);
        out.second_camera_id = second.camera_id;
        out.second_feature_id = FeatureId(m.trainIdx);
        out.descriptor_distance = m.distance;
        result.push_back(out);
    }
    return Result<std::vector<FeatureMatch>>::Success(std::move(result));
}
Result<std::vector<FeatureMatch>>
FeatureProcessor::MatchFeaturesWithPosePrior(const FeatureSet& first, const Pose3d& first_pose,
                                             const FeatureSet& second, const Pose3d& second_pose,
                                             const CameraIntrinsics& intrinsics,
                                             const FeatureMatchingOptions& options)
{
    auto matches = FeatureProcessor::MatchFeatureDescriptors(first, second, options);
    if (!matches.ok())
        return matches;
    auto fundamental = geometry::ComputeFundamentalMatrix(intrinsics, first_pose, second_pose);
    if (!fundamental.ok())
        return Result<std::vector<FeatureMatch>>::Failure(fundamental.status());
    std::vector<FeatureMatch> verified;
    for (FeatureMatch match : matches.value())
    {
        const auto i = static_cast<std::size_t>(match.first_feature_id.value());
        const auto j = static_cast<std::size_t>(match.second_feature_id.value());
        auto error = geometry::SampsonErrorPixels(first.features[i].pixel, second.features[j].pixel,
                                                  fundamental.value());
        if (!error.ok())
            return Result<std::vector<FeatureMatch>>::Failure(error.status());
        if (error.value() <= options.maximum_sampson_error_pixels)
        {
            match.geometric_error_pixels = error.value();
            verified.push_back(match);
        }
    }
    return Result<std::vector<FeatureMatch>>::Success(std::move(verified));
}

Result<VisualGeometryResult>
FeatureProcessor::VerifyMatchesWithVisualGeometry(const FeatureSet& first, const FeatureSet& second,
                                                  const std::vector<FeatureMatch>& candidates,
                                                  const VisualGeometryOptions& options)
{
    const Status a = Validate(first), b = Validate(second);
    if (!a.ok())
        return Result<VisualGeometryResult>::Failure(a);
    if (!b.ok())
        return Result<VisualGeometryResult>::Failure(b);
    if (!std::isfinite(options.ransac_threshold_pixels) || options.ransac_threshold_pixels <= 0 ||
        !std::isfinite(options.confidence) || options.confidence <= 0 || options.confidence >= 1 ||
        options.maximum_iterations <= 0 || options.minimum_inliers < 8 ||
        !std::isfinite(options.minimum_inlier_ratio) || options.minimum_inlier_ratio <= 0 ||
        options.minimum_inlier_ratio > 1 || options.coverage_columns <= 0 ||
        options.coverage_rows <= 0 || options.minimum_occupied_cells <= 0)
        return Result<VisualGeometryResult>::Failure(
            Status::Error(ErrorCode::kInvalidArgument, "invalid visual geometry options"));
    if (candidates.size() < static_cast<std::size_t>(options.minimum_inliers))
        return Result<VisualGeometryResult>::Failure(Status::Error(
            ErrorCode::kFailedPrecondition, "insufficient matches for visual geometry"));
    std::vector<cv::Point2f> p1, p2;
    p1.reserve(candidates.size());
    p2.reserve(candidates.size());
    for (const FeatureMatch& match : candidates)
    {
        if (match.first_camera_id != first.camera_id ||
            match.second_camera_id != second.camera_id || !match.first_feature_id.valid() ||
            !match.second_feature_id.valid() ||
            static_cast<std::size_t>(match.first_feature_id.value()) >= first.features.size() ||
            static_cast<std::size_t>(match.second_feature_id.value()) >= second.features.size())
            return Result<VisualGeometryResult>::Failure(Status::Error(
                ErrorCode::kInvalidArgument, "candidate match references an invalid feature"));
        const auto& x =
            first.features[static_cast<std::size_t>(match.first_feature_id.value())].pixel;
        const auto& y =
            second.features[static_cast<std::size_t>(match.second_feature_id.value())].pixel;
        p1.emplace_back(static_cast<float>(x.x()), static_cast<float>(x.y()));
        p2.emplace_back(static_cast<float>(y.x()), static_cast<float>(y.y()));
    }
    cv::Mat mask, f;
    {
        std::lock_guard<std::mutex> lock(OpenCvRngMutex());
        const std::uint64_t old = cv::theRNG().state;
        cv::theRNG().state = options.random_seed;
        f = cv::findFundamentalMat(p1, p2, cv::USAC_MAGSAC, options.ransac_threshold_pixels,
                                   options.confidence, options.maximum_iterations, mask);
        cv::theRNG().state = old;
    }
    if (f.empty() || f.rows != 3 || f.cols != 3 || mask.empty())
        return Result<VisualGeometryResult>::Failure(Status::Error(
            ErrorCode::kFailedPrecondition, "visual fundamental matrix estimation failed"));
    f.convertTo(f, CV_64F);
    Eigen::Matrix3d fundamental;
    for (int row = 0; row < 3; ++row)
        for (int column = 0; column < 3; ++column)
            fundamental(row, column) = f.at<double>(row, column);
    VisualGeometryResult result;
    result.fundamental_matrix = fundamental;
    std::vector<Eigen::Vector2d> inlier1, inlier2;
    for (std::size_t i = 0; i < candidates.size(); ++i)
        if (mask.at<std::uint8_t>(static_cast<int>(i)) != 0)
        {
            FeatureMatch match = candidates[i];
            auto error = geometry::SampsonErrorPixels(
                first.features[static_cast<std::size_t>(match.first_feature_id.value())].pixel,
                second.features[static_cast<std::size_t>(match.second_feature_id.value())].pixel,
                fundamental);
            if (!error.ok())
                return Result<VisualGeometryResult>::Failure(error.status());
            match.geometric_error_pixels = error.value();
            result.inlier_matches.push_back(match);
            inlier1.push_back(
                first.features[static_cast<std::size_t>(match.first_feature_id.value())].pixel);
            inlier2.push_back(
                second.features[static_cast<std::size_t>(match.second_feature_id.value())].pixel);
        }
    if (result.inlier_matches.size() < static_cast<std::size_t>(options.minimum_inliers) ||
        static_cast<double>(result.inlier_matches.size()) / static_cast<double>(candidates.size()) <
            options.minimum_inlier_ratio)
        return Result<VisualGeometryResult>::Failure(Status::Error(
            ErrorCode::kFailedPrecondition, "visual geometry has insufficient inlier support"));
    auto cells = [&](const std::vector<Eigen::Vector2d>& points)
    {
        Eigen::Vector2d low = points[0], high = points[0];
        for (const auto& p : points)
        {
            low = low.cwiseMin(p);
            high = high.cwiseMax(p);
        }
        Eigen::Vector2d span = (high - low).cwiseMax(Eigen::Vector2d::Ones());
        std::set<std::pair<int, int>> occupied;
        for (const auto& p : points)
        {
            int x =
                std::min(options.coverage_columns - 1,
                         static_cast<int>((p.x() - low.x()) / span.x() * options.coverage_columns));
            int y =
                std::min(options.coverage_rows - 1,
                         static_cast<int>((p.y() - low.y()) / span.y() * options.coverage_rows));
            occupied.emplace(x, y);
        }
        return static_cast<std::int32_t>(occupied.size());
    };
    result.first_occupied_cells = cells(inlier1);
    result.second_occupied_cells = cells(inlier2);
    if (result.first_occupied_cells < options.minimum_occupied_cells ||
        result.second_occupied_cells < options.minimum_occupied_cells)
        return Result<VisualGeometryResult>::Failure(Status::Error(
            ErrorCode::kFailedPrecondition, "visual inliers have insufficient spatial coverage"));
    return Result<VisualGeometryResult>::Success(std::move(result));
}
} // namespace lio_visual_ba

// ---- Pair Cache ----

namespace lio_visual_ba
{
namespace
{
constexpr std::array<std::uint8_t, 8> PairCachekMagic{{'L', 'V', 'B', 'A', 'P', 'A', 'R', '1'}};
constexpr std::uint32_t PairCachekVersion = 1, kEndian = 0x01020304U;
constexpr std::uint64_t kMaximumMatches = 100000000ULL;
template <class T> void PairCacheAppend(std::vector<std::uint8_t>& out, const T& value)
{
    static_assert(std::is_trivially_copyable<T>::value, "binary type");
    const auto* p = reinterpret_cast<const std::uint8_t*>(&value);
    out.insert(out.end(), p, p + sizeof(T));
}
template <class T>
bool PairCacheRead(const std::vector<std::uint8_t>& in, std::size_t& offset, T& value)
{
    if (offset > in.size() || sizeof(T) > in.size() - offset)
        return false;
    std::memcpy(&value, in.data() + offset, sizeof(T));
    offset += sizeof(T);
    return true;
}
std::uint64_t Hash(const std::vector<std::uint8_t>& bytes)
{
    std::uint64_t h = 14695981039346656037ULL;
    for (auto b : bytes)
    {
        h ^= b;
        h *= 1099511628211ULL;
    }
    return h;
}
Status ValidateKey(const PairCacheKey& key)
{
    if (!key.first_camera_id.valid() || !key.second_camera_id.valid() ||
        key.first_camera_id == key.second_camera_id ||
        key.first_feature_count >
            static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max()) ||
        key.second_feature_count >
            static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max()))
        return Status::Error(ErrorCode::kInvalidArgument, "invalid pair cache key");
    return Status::Ok();
}
Status ValidateMatches(const PairCacheKey& key, const std::vector<FeatureMatch>& matches)
{
    if (matches.size() > kMaximumMatches)
        return Status::Error(ErrorCode::kInvalidArgument, "too many pair matches");
    FeatureId previous = FeatureId::Invalid();
    for (const auto& m : matches)
    {
        if (m.first_camera_id != key.first_camera_id ||
            m.second_camera_id != key.second_camera_id || !m.first_feature_id.valid() ||
            !m.second_feature_id.valid() ||
            static_cast<std::uint64_t>(m.first_feature_id.value()) >= key.first_feature_count ||
            static_cast<std::uint64_t>(m.second_feature_id.value()) >= key.second_feature_count ||
            !std::isfinite(m.descriptor_distance) || m.descriptor_distance < 0 ||
            !std::isfinite(m.geometric_error_pixels) || m.geometric_error_pixels < 0)
            return Status::Error(ErrorCode::kInvalidArgument,
                                 "pair match has invalid references or errors");
        if (previous.valid() && m.first_feature_id < previous)
            return Status::Error(ErrorCode::kInvalidArgument,
                                 "pair matches are not deterministically ordered");
        previous = m.first_feature_id;
    }
    return Status::Ok();
}
} // namespace
Status FeatureProcessor::SavePairCache(const std::filesystem::path& path, const PairCacheKey& key,
                                       const std::vector<FeatureMatch>& matches)
{
    if (path.empty())
        return Status::Error(ErrorCode::kInvalidArgument, "pair cache path is empty");
    auto status = ValidateKey(key);
    if (!status.ok())
        return status;
    status = ValidateMatches(key, matches);
    if (!status.ok())
        return status;
    std::vector<std::uint8_t> payload;
    payload.reserve(matches.size() * 20U);
    for (const auto& m : matches)
    {
        PairCacheAppend(payload, m.first_feature_id.value());
        PairCacheAppend(payload, m.second_feature_id.value());
        PairCacheAppend(payload, m.descriptor_distance);
        PairCacheAppend(payload, m.geometric_error_pixels);
    }
    std::vector<std::uint8_t> bytes(PairCachekMagic.begin(), PairCachekMagic.end());
    PairCacheAppend(bytes, PairCachekVersion);
    PairCacheAppend(bytes, kEndian);
    PairCacheAppend(bytes, key.first_camera_id.value());
    PairCacheAppend(bytes, key.second_camera_id.value());
    PairCacheAppend(bytes, key.first_feature_fingerprint);
    PairCacheAppend(bytes, key.second_feature_fingerprint);
    PairCacheAppend(bytes, key.matching_configuration_fingerprint);
    PairCacheAppend(bytes, key.first_feature_count);
    PairCacheAppend(bytes, key.second_feature_count);
    const std::uint64_t count = matches.size(), size = payload.size();
    PairCacheAppend(bytes, count);
    PairCacheAppend(bytes, size);
    PairCacheAppend(bytes, Hash(payload));
    bytes.insert(bytes.end(), payload.begin(), payload.end());
    std::error_code error;
    if (!path.parent_path().empty())
        std::filesystem::create_directories(path.parent_path(), error);
    if (error)
        return Status::Error(ErrorCode::kIoError,
                             "cannot create pair cache directory: " + error.message());
    const auto temporary = std::filesystem::path(
        path.string() + ".tmp." +
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    {
        std::ofstream out(temporary, std::ios::binary | std::ios::trunc);
        if (!out)
            return Status::Error(ErrorCode::kIoError, "cannot open temporary pair cache");
        out.write(reinterpret_cast<const char*>(bytes.data()),
                  static_cast<std::streamsize>(bytes.size()));
        if (!out)
            return Status::Error(ErrorCode::kIoError, "cannot write temporary pair cache");
    }
    std::filesystem::rename(temporary, path, error);
    if (error)
    {
        std::filesystem::remove(temporary);
        return Status::Error(ErrorCode::kIoError, "cannot replace pair cache: " + error.message());
    }
    return Status::Ok();
}
Result<std::vector<FeatureMatch>> FeatureProcessor::LoadPairCache(const std::filesystem::path& path,
                                                                  const PairCacheKey& expected)
{
    auto key_status = ValidateKey(expected);
    if (!key_status.ok())
        return Result<std::vector<FeatureMatch>>::Failure(key_status);
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in)
        return Result<std::vector<FeatureMatch>>::Failure(
            Status::Error(ErrorCode::kNotFound, "cannot open pair cache: " + path.string()));
    const auto length = in.tellg();
    if (length < 0 || length > static_cast<std::streamsize>(kMaximumMatches * 20U + 256U))
        return Result<std::vector<FeatureMatch>>::Failure(
            Status::Error(ErrorCode::kDataLoss, "invalid pair cache size"));
    in.seekg(0);
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(length));
    in.read(reinterpret_cast<char*>(bytes.data()), length);
    if (!in)
        return Result<std::vector<FeatureMatch>>::Failure(
            Status::Error(ErrorCode::kIoError, "cannot read pair cache"));
    if (bytes.size() < PairCachekMagic.size() ||
        !std::equal(PairCachekMagic.begin(), PairCachekMagic.end(), bytes.begin()))
        return Result<std::vector<FeatureMatch>>::Failure(
            Status::Error(ErrorCode::kDataLoss, "pair cache magic mismatch"));
    std::size_t offset = PairCachekMagic.size();
    std::uint32_t version = 0, endian = 0;
    std::int32_t first = -1, second = -1;
    std::uint64_t ff = 0, sf = 0, cf = 0, fc = 0, sc = 0, count = 0, size = 0, checksum = 0;
    if (!PairCacheRead(bytes, offset, version) || !PairCacheRead(bytes, offset, endian) ||
        !PairCacheRead(bytes, offset, first) || !PairCacheRead(bytes, offset, second) ||
        !PairCacheRead(bytes, offset, ff) || !PairCacheRead(bytes, offset, sf) ||
        !PairCacheRead(bytes, offset, cf) || !PairCacheRead(bytes, offset, fc) ||
        !PairCacheRead(bytes, offset, sc) || !PairCacheRead(bytes, offset, count) ||
        !PairCacheRead(bytes, offset, size) || !PairCacheRead(bytes, offset, checksum))
        return Result<std::vector<FeatureMatch>>::Failure(
            Status::Error(ErrorCode::kDataLoss, "truncated pair cache header"));
    if (version != PairCachekVersion || endian != kEndian)
        return Result<std::vector<FeatureMatch>>::Failure(Status::Error(
            ErrorCode::kFailedPrecondition, "unsupported pair cache schema or endianness"));
    if (first != expected.first_camera_id.value() || second != expected.second_camera_id.value() ||
        ff != expected.first_feature_fingerprint || sf != expected.second_feature_fingerprint ||
        cf != expected.matching_configuration_fingerprint || fc != expected.first_feature_count ||
        sc != expected.second_feature_count)
        return Result<std::vector<FeatureMatch>>::Failure(
            Status::Error(ErrorCode::kFailedPrecondition, "pair cache key mismatch"));
    if (count > kMaximumMatches || size != count * 20U || size != bytes.size() - offset)
        return Result<std::vector<FeatureMatch>>::Failure(
            Status::Error(ErrorCode::kDataLoss, "invalid pair cache metadata"));
    std::vector<std::uint8_t> payload(bytes.begin() + static_cast<std::ptrdiff_t>(offset),
                                      bytes.end());
    if (Hash(payload) != checksum)
        return Result<std::vector<FeatureMatch>>::Failure(
            Status::Error(ErrorCode::kDataLoss, "pair cache checksum mismatch"));
    std::vector<FeatureMatch> matches;
    matches.reserve(static_cast<std::size_t>(count));
    std::size_t p = 0;
    for (std::uint64_t i = 0; i < count; ++i)
    {
        std::int32_t a = 0, b = 0;
        FeatureMatch m;
        m.first_camera_id = expected.first_camera_id;
        m.second_camera_id = expected.second_camera_id;
        if (!PairCacheRead(payload, p, a) || !PairCacheRead(payload, p, b) ||
            !PairCacheRead(payload, p, m.descriptor_distance) ||
            !PairCacheRead(payload, p, m.geometric_error_pixels))
            return Result<std::vector<FeatureMatch>>::Failure(
                Status::Error(ErrorCode::kDataLoss, "truncated pair match payload"));
        m.first_feature_id = FeatureId(a);
        m.second_feature_id = FeatureId(b);
        matches.push_back(m);
    }
    auto status = ValidateMatches(expected, matches);
    if (!status.ok())
        return Result<std::vector<FeatureMatch>>::Failure(
            Status::Error(ErrorCode::kDataLoss, status.message()));
    return Result<std::vector<FeatureMatch>>::Success(std::move(matches));
}
} // namespace lio_visual_ba

// ---- Track Builder ----

namespace lio_visual_ba
{
namespace
{
struct Node
{
    CameraId camera;
    FeatureId feature;
    Eigen::Vector2d pixel;
    bool supplemental = false;
};
class DisjointSet
{
  public:
    explicit DisjointSet(const std::vector<Node>& nodes)
        : parent_(nodes.size()), size_(nodes.size(), 1), cameras_(nodes.size())
    {
        std::iota(parent_.begin(), parent_.end(), 0U);
        for (std::size_t i = 0; i < nodes.size(); ++i)
            cameras_[i].insert(nodes[i].camera.value());
    }
    std::size_t Find(std::size_t x)
    {
        while (parent_[x] != x)
        {
            parent_[x] = parent_[parent_[x]];
            x = parent_[x];
        }
        return x;
    }
    enum class Merge
    {
        kMerged,
        kRedundant,
        kConflict
    };
    Merge Union(std::size_t a, std::size_t b)
    {
        a = Find(a);
        b = Find(b);
        if (a == b)
            return Merge::kRedundant;
        const auto& small = cameras_[a].size() < cameras_[b].size() ? cameras_[a] : cameras_[b];
        const auto& large = cameras_[a].size() < cameras_[b].size() ? cameras_[b] : cameras_[a];
        for (int camera : small)
            if (large.count(camera) > 0)
                return Merge::kConflict;
        if (size_[a] < size_[b])
            std::swap(a, b);
        parent_[b] = a;
        size_[a] += size_[b];
        cameras_[a].insert(cameras_[b].begin(), cameras_[b].end());
        cameras_[b].clear();
        return Merge::kMerged;
    }

  private:
    std::vector<std::size_t> parent_, size_;
    std::vector<std::set<int>> cameras_;
};
} // namespace
Result<TrackBuildResult>
FeatureProcessor::BuildFeatureTracks(const std::vector<FeatureSet>& sets,
                                     const std::vector<FeatureMatch>& input,
                                     const TrackBuilderOptions& options)
{
    if (options.minimum_track_length < 2)
        return Result<TrackBuildResult>::Failure(Status::Error(
            ErrorCode::kInvalidArgument, "minimum track length must be at least two"));
    std::map<CameraId, std::size_t> set_index;
    std::vector<Node> nodes;
    std::map<std::pair<int, int>, std::size_t> node_index;
    for (std::size_t si = 0; si < sets.size(); ++si)
    {
        const auto& set = sets[si];
        if (!set.camera_id.valid() || !set_index.emplace(set.camera_id, si).second ||
            set.core_feature_count > set.features.size())
            return Result<TrackBuildResult>::Failure(Status::Error(
                ErrorCode::kInvalidArgument, "invalid or duplicate feature-set camera ID"));
        for (std::size_t fi = 0; fi < set.features.size(); ++fi)
        {
            const auto& feature = set.features[fi];
            if (feature.id != FeatureId(static_cast<int>(fi)) ||
                !feature.pixel.array().isFinite().all())
                return Result<TrackBuildResult>::Failure(
                    Status::Error(ErrorCode::kInvalidArgument, "invalid feature ID or pixel"));
            node_index[{set.camera_id.value(), feature.id.value()}] = nodes.size();
            nodes.push_back(
                {set.camera_id, feature.id, feature.pixel, fi >= set.core_feature_count});
        }
    }
    DisjointSet dsu(nodes);
    TrackBuildResult result;
    // Preserve pair/matcher order. Conflict resolution is order-dependent and
    // the Python reference deliberately gives earlier temporal matches
    // priority over later loop/recovery matches.
    for (const auto& match : input)
    {
        if (match.first_camera_id == match.second_camera_id)
            return Result<TrackBuildResult>::Failure(
                Status::Error(ErrorCode::kInvalidArgument, "match connects a camera to itself"));
        auto a = node_index.find({match.first_camera_id.value(), match.first_feature_id.value()}),
             b = node_index.find({match.second_camera_id.value(), match.second_feature_id.value()});
        if (a == node_index.end() || b == node_index.end())
            return Result<TrackBuildResult>::Failure(
                Status::Error(ErrorCode::kInvalidArgument, "match references an unknown feature"));
        switch (dsu.Union(a->second, b->second))
        {
        case DisjointSet::Merge::kMerged:
            ++result.accepted_matches;
            break;
        case DisjointSet::Merge::kRedundant:
            ++result.redundant_matches;
            break;
        case DisjointSet::Merge::kConflict:
            ++result.conflict_rejected_matches;
            break;
        }
    }
    std::map<std::size_t, std::vector<std::size_t>> components;
    for (std::size_t i = 0; i < nodes.size(); ++i)
        components[dsu.Find(i)].push_back(i);
    std::vector<std::vector<std::size_t>> retained;
    for (auto& entry : components)
        if (entry.second.size() >= options.minimum_track_length)
        {
            std::sort(entry.second.begin(), entry.second.end(),
                      [&](std::size_t a, std::size_t b)
                      {
                          return std::tie(nodes[a].camera, nodes[a].feature) <
                                 std::tie(nodes[b].camera, nodes[b].feature);
                      });
            retained.push_back(std::move(entry.second));
        }
    std::sort(retained.begin(), retained.end(),
              [&](const auto& a, const auto& b)
              {
                  return std::tie(nodes[a[0]].camera, nodes[a[0]].feature) <
                         std::tie(nodes[b[0]].camera, nodes[b[0]].feature);
              });
    for (std::size_t ti = 0; ti < retained.size(); ++ti)
    {
        FeatureTrack track;
        track.id = TrackId(static_cast<int>(ti));
        for (std::size_t index : retained[ti])
        {
            const Node& node = nodes[index];
            track.observations.push_back({node.camera, node.feature, node.pixel});
            track.is_supplemental = track.is_supplemental || node.supplemental;
        }
        result.tracks.push_back(std::move(track));
    }
    return Result<TrackBuildResult>::Success(std::move(result));
}
} // namespace lio_visual_ba
