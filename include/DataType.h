#pragma once

// Common Eigen and Sophus data types used throughout the project.

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <sophus/se3.hpp>
#include <sophus/so3.hpp>

// -----------------------------------------------------------------------------
// Eigen vectors
// -----------------------------------------------------------------------------

template <typename T>
using Vec2 = Eigen::Matrix<T, 2, 1>;

template <typename T>
using Vec3 = Eigen::Matrix<T, 3, 1>;

template <typename T>
using Vec6 = Eigen::Matrix<T, 6, 1>;

template <typename T>
using Vec18 = Eigen::Matrix<T, 18, 1>;

// -----------------------------------------------------------------------------
// Eigen matrices
// -----------------------------------------------------------------------------

template <typename T>
using Mat2 = Eigen::Matrix<T, 2, 2>;

template <typename T>
using Mat3 = Eigen::Matrix<T, 3, 3>;

template <typename T>
using Mat4 = Eigen::Matrix<T, 4, 4>;

template <typename T>
using Mat6 = Eigen::Matrix<T, 6, 6>;

template <typename T>
using Mat18 = Eigen::Matrix<T, 18, 18>;

// -----------------------------------------------------------------------------
// Eigen mapped vectors
// Useful for mapping Ceres raw parameter arrays without copying.
// -----------------------------------------------------------------------------

template <typename T>
using ConstVec3Map = Eigen::Map<const Vec3<T>>;

template <typename T>
using ConstVec6Map = Eigen::Map<const Vec6<T>>;

template <typename T>
using Vec6Map = Eigen::Map<Vec6<T>>;

// -----------------------------------------------------------------------------
// Common concrete Eigen types
// -----------------------------------------------------------------------------

using Vec2d = Vec2<double>;

using Vec3f = Vec3<float>;
using Vec3d = Vec3<double>;
using Vec3i = Vec3<int>;

using Vec6f = Vec6<float>;
using Vec6d = Vec6<double>;

using Vec18d = Vec18<double>;

using Mat2d = Mat2<double>;

using Mat3f = Mat3<float>;
using Mat3d = Mat3<double>;

using Mat4d = Mat4<double>;

using Mat6f = Mat6<float>;
using Mat6d = Mat6<double>;

using Mat18d = Mat18<double>;

using KeyType = Vec3<int>;
using MotionNoiseD = Mat18<double>;
using LidarNoiseD = Mat6<double>;

// -----------------------------------------------------------------------------
// Sophus Lie-group types
// -----------------------------------------------------------------------------

template <typename T>
using SE3 = Sophus::SE3<T>;

template <typename T>
using SO3 = Sophus::SO3<T>;

using SE3f = SE3<float>;
using SE3d = SE3<double>;

using SO3f = SO3<float>;
using SO3d = SO3<double>;