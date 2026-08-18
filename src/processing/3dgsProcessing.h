#pragma once

#include <cstddef>
#include <cstdint>
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
};

class GaussianSplatProcessing
{
public:
    // Supports ASCII and binary little-endian PLY vertex data. Direct RGB is
    // used when present; 3DGS f_dc coefficients are converted to display RGB.
    bool loadPly(const std::string& path);

    void clear();

    std::size_t splatCount() const;
    const std::vector<GaussianPoint>& points() const;
    const std::string& lastError() const;

private:
    std::vector<GaussianPoint> points_;
    std::string last_error_;
};
