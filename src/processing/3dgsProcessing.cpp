#include "3dgsProcessing.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <utility>

namespace
{
enum class PlyFormat
{
    Ascii,
    BinaryLittleEndian
};

enum class ScalarType
{
    Int8,
    UInt8,
    Int16,
    UInt16,
    Int32,
    UInt32,
    Float32,
    Float64,
    Invalid
};

struct VertexProperty
{
    ScalarType type = ScalarType::Invalid;
    std::string name;
};

ScalarType parseScalarType(const std::string& name)
{
    if (name == "char" || name == "int8") return ScalarType::Int8;
    if (name == "uchar" || name == "uint8") return ScalarType::UInt8;
    if (name == "short" || name == "int16") return ScalarType::Int16;
    if (name == "ushort" || name == "uint16") return ScalarType::UInt16;
    if (name == "int" || name == "int32") return ScalarType::Int32;
    if (name == "uint" || name == "uint32") return ScalarType::UInt32;
    if (name == "float" || name == "float32") return ScalarType::Float32;
    if (name == "double" || name == "float64") return ScalarType::Float64;
    return ScalarType::Invalid;
}

template <typename T>
bool readBinaryValue(std::istream& input, double& value)
{
    T raw{};
    input.read(reinterpret_cast<char*>(&raw), sizeof(T));
    if (!input) return false;
    value = static_cast<double>(raw);
    return true;
}

bool readBinaryScalar(std::istream& input, ScalarType type, double& value)
{
    switch (type) {
    case ScalarType::Int8: return readBinaryValue<std::int8_t>(input, value);
    case ScalarType::UInt8: return readBinaryValue<std::uint8_t>(input, value);
    case ScalarType::Int16: return readBinaryValue<std::int16_t>(input, value);
    case ScalarType::UInt16: return readBinaryValue<std::uint16_t>(input, value);
    case ScalarType::Int32: return readBinaryValue<std::int32_t>(input, value);
    case ScalarType::UInt32: return readBinaryValue<std::uint32_t>(input, value);
    case ScalarType::Float32: return readBinaryValue<float>(input, value);
    case ScalarType::Float64: return readBinaryValue<double>(input, value);
    case ScalarType::Invalid: return false;
    }
    return false;
}

int propertyIndex(
    const std::vector<VertexProperty>& properties,
    const std::string& name)
{
    for (std::size_t index = 0; index < properties.size(); ++index) {
        if (properties[index].name == name) return static_cast<int>(index);
    }
    return -1;
}

std::uint8_t byteColor(double value)
{
    return static_cast<std::uint8_t>(
        std::lround(std::clamp(value, 0.0, 255.0)));
}

std::uint8_t sphericalHarmonicColor(double dc_value)
{
    // Degree-zero real spherical-harmonic coefficient used by 3DGS.
    constexpr double kShC0 = 0.28209479177387814;
    const double linear_color = std::clamp(0.5 + kShC0 * dc_value, 0.0, 1.0);
    return static_cast<std::uint8_t>(std::lround(linear_color * 255.0));
}

template <std::size_t Size>
bool hasCompleteGroup(const std::array<int, Size>& indices)
{
    return std::all_of(
        indices.begin(), indices.end(), [](int index) { return index >= 0; });
}

template <std::size_t Size>
bool hasAnyGroup(const std::array<int, Size>& indices)
{
    return std::any_of(
        indices.begin(), indices.end(), [](int index) { return index >= 0; });
}

int sphericalHarmonicDegree(std::size_t rest_coefficient_count)
{
    if (rest_coefficient_count == 0) return 0;
    if (rest_coefficient_count == 9) return 1;
    if (rest_coefficient_count == 24) return 2;
    if (rest_coefficient_count == 45) return 3;
    return -1;
}
} // namespace

bool GaussianSplatProcessing::loadPly(const std::string& path)
{
    clear();
    if (path.empty()) {
        last_error_ = "PLY path is empty.";
        return false;
    }

    std::ifstream input(path, std::ios::binary);
    if (!input) {
        last_error_ = "Could not open PLY file: " + path;
        return false;
    }

    std::string line;
    if (!std::getline(input, line) || (line != "ply" && line != "ply\r")) {
        last_error_ = "File does not begin with a valid PLY signature.";
        return false;
    }

    PlyFormat format = PlyFormat::Ascii;
    bool format_found = false;
    bool header_finished = false;
    std::string current_element;
    std::size_t vertex_count = 0;
    std::vector<VertexProperty> properties;

    while (std::getline(input, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line == "end_header") {
            header_finished = true;
            break;
        }

        std::istringstream tokens(line);
        std::string keyword;
        tokens >> keyword;
        if (keyword.empty() || keyword == "comment" || keyword == "obj_info") {
            continue;
        }
        if (keyword == "format") {
            std::string format_name;
            std::string version;
            tokens >> format_name >> version;
            if (format_name == "ascii") {
                format = PlyFormat::Ascii;
            } else if (format_name == "binary_little_endian") {
                format = PlyFormat::BinaryLittleEndian;
            } else {
                last_error_ = "Unsupported PLY format: " + format_name;
                return false;
            }
            format_found = true;
        } else if (keyword == "element") {
            std::size_t count = 0;
            tokens >> current_element >> count;
            if (!tokens) {
                last_error_ = "Malformed PLY element declaration.";
                return false;
            }
            if (current_element == "vertex") vertex_count = count;
        } else if (keyword == "property" && current_element == "vertex") {
            std::string type_name;
            std::string property_name;
            tokens >> type_name;
            if (type_name == "list") {
                last_error_ = "List properties are not supported in the vertex element.";
                return false;
            }
            tokens >> property_name;
            const ScalarType type = parseScalarType(type_name);
            if (!tokens || type == ScalarType::Invalid) {
                last_error_ = "Unsupported vertex property declaration: " + line;
                return false;
            }
            properties.push_back({type, property_name});
        }
    }

    if (!header_finished || !format_found) {
        last_error_ = "PLY header is incomplete.";
        return false;
    }
    if (vertex_count == 0) {
        last_error_ = "PLY file contains no vertices.";
        return false;
    }

    const int x_index = propertyIndex(properties, "x");
    const int y_index = propertyIndex(properties, "y");
    const int z_index = propertyIndex(properties, "z");
    if (x_index < 0 || y_index < 0 || z_index < 0) {
        last_error_ = "PLY vertex element must contain x, y, and z properties.";
        return false;
    }

    const std::array<int, 3> rgb_indices{
        propertyIndex(properties, "red"),
        propertyIndex(properties, "green"),
        propertyIndex(properties, "blue")};
    const std::array<int, 3> dc_indices{
        propertyIndex(properties, "f_dc_0"),
        propertyIndex(properties, "f_dc_1"),
        propertyIndex(properties, "f_dc_2")};
    const std::array<int, 3> scale_indices{
        propertyIndex(properties, "scale_0"),
        propertyIndex(properties, "scale_1"),
        propertyIndex(properties, "scale_2")};
    const std::array<int, 4> rotation_indices{
        propertyIndex(properties, "rot_0"),
        propertyIndex(properties, "rot_1"),
        propertyIndex(properties, "rot_2"),
        propertyIndex(properties, "rot_3")};
    const int opacity_index = propertyIndex(properties, "opacity");

    const bool has_rgb = hasCompleteGroup(rgb_indices);
    const bool has_dc = hasCompleteGroup(dc_indices);
    const bool has_scale = hasCompleteGroup(scale_indices);
    const bool has_rotation = hasCompleteGroup(rotation_indices);

    if (hasAnyGroup(rgb_indices) && !has_rgb) {
        last_error_ = "PLY contains an incomplete red/green/blue property group.";
        return false;
    }
    if (hasAnyGroup(dc_indices) && !has_dc) {
        last_error_ = "PLY contains an incomplete f_dc_0/f_dc_1/f_dc_2 property group.";
        return false;
    }
    if (hasAnyGroup(scale_indices) && !has_scale) {
        last_error_ = "PLY contains an incomplete scale_0/scale_1/scale_2 property group.";
        return false;
    }
    if (hasAnyGroup(rotation_indices) && !has_rotation) {
        last_error_ = "PLY contains an incomplete rot_0/rot_1/rot_2/rot_3 property group.";
        return false;
    }

    std::array<int, 45> rest_indices{};
    rest_indices.fill(-1);
    std::size_t rest_count = 0;
    bool rest_gap_found = false;
    for (std::size_t coefficient = 0;
         coefficient <= rest_indices.size(); ++coefficient) {
        const int index = propertyIndex(
            properties, "f_rest_" + std::to_string(coefficient));
        if (index < 0) {
            rest_gap_found = true;
            continue;
        }
        if (coefficient == rest_indices.size()) {
            last_error_ = "PLY spherical-harmonic degree above 3 is not supported.";
            return false;
        }
        if (rest_gap_found) {
            last_error_ = "PLY spherical-harmonic properties must be contiguous from f_rest_0.";
            return false;
        }
        rest_indices[coefficient] = index;
        ++rest_count;
    }
    const int sh_degree = sphericalHarmonicDegree(rest_count);
    if (sh_degree < 0) {
        last_error_ = "PLY has an unsupported spherical-harmonic coefficient count: " +
            std::to_string(rest_count) + ".";
        return false;
    }
    if (rest_count > 0 && !has_dc) {
        last_error_ = "PLY contains f_rest properties without the f_dc color group.";
        return false;
    }

    metadata_.has_scale = has_scale;
    metadata_.has_rotation = has_rotation;
    metadata_.has_opacity = opacity_index >= 0;
    metadata_.has_sh_dc = has_dc;
    metadata_.sh_degree = sh_degree;

    std::vector<GaussianPoint> loaded_points;
    try {
        loaded_points.reserve(vertex_count);
    } catch (const std::bad_alloc&) {
        last_error_ = "Not enough memory for the declared PLY vertex count.";
        return false;
    }

    std::vector<double> values(properties.size(), 0.0);
    for (std::size_t vertex = 0; vertex < vertex_count; ++vertex) {
        for (std::size_t property = 0; property < properties.size(); ++property) {
            const bool read_ok = format == PlyFormat::Ascii
                ? static_cast<bool>(input >> values[property])
                : readBinaryScalar(input, properties[property].type, values[property]);
            if (!read_ok) {
                last_error_ = "PLY vertex data ended at vertex " +
                    std::to_string(vertex) + " of " + std::to_string(vertex_count) + ".";
                return false;
            }
        }

        if (!std::all_of(values.begin(), values.end(), [](double value) {
                return std::isfinite(value);
            })) {
            last_error_ = "PLY contains a non-finite value at vertex " +
                std::to_string(vertex) + ".";
            return false;
        }

        const double x = values[static_cast<std::size_t>(x_index)];
        const double y = values[static_cast<std::size_t>(y_index)];
        const double z = values[static_cast<std::size_t>(z_index)];
        GaussianPoint point;
        point.x = static_cast<float>(x);
        point.y = static_cast<float>(y);
        point.z = static_cast<float>(z);
        if (has_rgb) {
            point.red = byteColor(values[static_cast<std::size_t>(rgb_indices[0])]);
            point.green = byteColor(values[static_cast<std::size_t>(rgb_indices[1])]);
            point.blue = byteColor(values[static_cast<std::size_t>(rgb_indices[2])]);
        } else if (has_dc) {
            point.red = sphericalHarmonicColor(
                values[static_cast<std::size_t>(dc_indices[0])]);
            point.green = sphericalHarmonicColor(
                values[static_cast<std::size_t>(dc_indices[1])]);
            point.blue = sphericalHarmonicColor(
                values[static_cast<std::size_t>(dc_indices[2])]);
        }

        if (has_scale) {
            for (std::size_t component = 0; component < point.scale.size(); ++component) {
                point.scale[component] = static_cast<float>(
                    values[static_cast<std::size_t>(scale_indices[component])]);
            }
        }
        if (has_rotation) {
            for (std::size_t component = 0; component < point.rotation.size(); ++component) {
                point.rotation[component] = static_cast<float>(
                    values[static_cast<std::size_t>(rotation_indices[component])]);
            }
        }
        if (opacity_index >= 0) {
            point.opacity = static_cast<float>(
                values[static_cast<std::size_t>(opacity_index)]);
        }
        if (has_dc) {
            for (std::size_t component = 0; component < point.sh_dc.size(); ++component) {
                point.sh_dc[component] = static_cast<float>(
                    values[static_cast<std::size_t>(dc_indices[component])]);
            }
        }
        for (std::size_t coefficient = 0; coefficient < rest_count; ++coefficient) {
            point.sh_rest[coefficient] = static_cast<float>(
                values[static_cast<std::size_t>(rest_indices[coefficient])]);
        }
        loaded_points.push_back(point);
    }

    points_ = std::move(loaded_points);
    return true;
}

void GaussianSplatProcessing::clear()
{
    std::vector<GaussianPoint>().swap(points_);
    metadata_ = {};
    last_error_.clear();
}

std::size_t GaussianSplatProcessing::splatCount() const
{
    return points_.size();
}

const std::vector<GaussianPoint>& GaussianSplatProcessing::points() const
{
    return points_;
}

const GaussianPlyMetadata& GaussianSplatProcessing::metadata() const
{
    return metadata_;
}

const std::string& GaussianSplatProcessing::lastError() const
{
    return last_error_;
}
