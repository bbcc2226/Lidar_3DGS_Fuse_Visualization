#include "3dgsProcessing.h"

bool GaussianSplatProcessing::loadPly(const std::string& path)
{
    clear();

    if (path.empty()) {
        last_error_ = "PLY path is empty.";
        return false;
    }

    last_error_ = "PLY loading is not implemented yet.";
    return false;
}

void GaussianSplatProcessing::clear()
{
    points_.clear();
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

const std::string& GaussianSplatProcessing::lastError() const
{
    return last_error_;
}
