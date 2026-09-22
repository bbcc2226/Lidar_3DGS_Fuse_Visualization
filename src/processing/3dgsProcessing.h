#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

struct GaussianPoint
{
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;

    std::uint8_t red = 255;
    std::uint8_t green = 255;
    std::uint8_t blue = 255;

    // Raw parameters stored by the reference 3DGS PLY format. Activation
    // (exp for scale, sigmoid for opacity, normalization for rotation) belongs
    // to the renderer and is deliberately not applied by the file loader.
    std::array<float, 3> scale{};
    std::array<float, 4> rotation{1.0f, 0.0f, 0.0f, 0.0f}; // w, x, y, z
    float opacity = 0.0f;
    std::array<float, 3> sh_dc{};
    std::array<float, 45> sh_rest{};
};

struct GaussianPlyMetadata
{
    bool has_scale = false;
    bool has_rotation = false;
    bool has_opacity = false;
    bool has_sh_dc = false;
    int sh_degree = 0;

    bool isComplete3DGS() const
    {
        return has_scale && has_rotation && has_opacity && has_sh_dc;
    }
};

class GaussianSplatProcessing
{
public:
    // Supports ASCII and binary little-endian PLY vertex data. Direct RGB is
    // used when present; 3DGS f_dc coefficients are converted to display RGB.
    bool loadPly(const std::string& path,
                 const std::function<void(int)>& progress = {});

    void clear();

    std::size_t splatCount() const;
    const std::vector<GaussianPoint>& points() const;
    const GaussianPlyMetadata& metadata() const;
    const std::string& lastError() const;

private:
    std::vector<GaussianPoint> points_;
    GaussianPlyMetadata metadata_;
    std::string last_error_;
};
