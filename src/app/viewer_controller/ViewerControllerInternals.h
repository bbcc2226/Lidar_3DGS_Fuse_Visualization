#pragma once

namespace viewer_controller_detail
{
constexpr float kRotationDegreesPerPixel = 0.5f;
constexpr float kConstrainedRotationDegreesPerPixel = 0.15f;
constexpr float kSemanticRotationDegreesPerPixel = 0.08f;
constexpr float kInitialCameraDistance = 3.0f;
constexpr float kNavigateInitialCameraDistance = 0.65f;
constexpr float kMinimumCameraDistance = 0.02f;
constexpr float kMaximumCameraDistance = 100.0f;
constexpr float kDollyFactorPerWheelStep = 0.90f;
constexpr float kMaximumWheelStepsPerEvent = 2.0f;
constexpr float kPanSensitivity = 1.75f;
constexpr float kHalfVerticalFieldOfViewRadians = 22.5f * 3.14159265f / 180.0f;
constexpr int kConstrainedDragThresholdPixels = 6;
constexpr float kVerticalMoveFraction = 0.05f;
}
