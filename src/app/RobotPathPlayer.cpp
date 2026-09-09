#include "RobotPathPlayer.h"

#include <QFile>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QtMath>

#include <algorithm>
#include <cmath>
#include <limits>

namespace
{
constexpr float kLookAheadMeters = 0.4f;
constexpr float kYawTimeConstantSeconds = 1.2f;
constexpr float kMaximumYawRateDegreesPerSecond = 20.0f;
constexpr float kMaximumPitchRateDegreesPerSecond = 12.0f;

float shortestAngleDifference(float target, float current)
{
    float difference = std::fmod(target - current + 180.0f, 360.0f);
    if (difference < 0.0f) difference += 360.0f;
    return difference - 180.0f;
}
} // namespace

RobotPathPlayer::RobotPathPlayer(QObject* parent)
    : QObject(parent)
{
    world_to_aligned_.setToIdentity();
    timer_.setInterval(16);
    timer_.setTimerType(Qt::PreciseTimer);
    connect(&timer_, &QTimer::timeout, this, &RobotPathPlayer::updateFrame);
}

bool RobotPathPlayer::loadPath(const QString& path)
{
    stop();
    world_points_.clear();
    cumulative_distance_.clear();
    last_error_.clear();

    QFile file(path);
    if (!file.open(QIODevice::ReadOnly)) {
        last_error_ = "Cannot open robot path: " + file.errorString();
        return false;
    }
    QJsonParseError parse_error;
    const QJsonDocument document =
        QJsonDocument::fromJson(file.readAll(), &parse_error);
    if (parse_error.error != QJsonParseError::NoError || !document.isObject()) {
        last_error_ = "Invalid robot path JSON: " + parse_error.errorString();
        return false;
    }
    const QJsonObject root = document.object();
    if (root.value("coordinate_frame").toString() != "3dgs_world") {
        last_error_ = "Robot path must use the 3dgs_world coordinate frame";
        return false;
    }
    const QJsonArray points = root.value("points").toArray();
    for (const QJsonValue& value : points) {
        const QJsonObject point = value.toObject();
        const double x = point.value("x").toDouble(
            std::numeric_limits<double>::quiet_NaN());
        const double y = point.value("y").toDouble(
            std::numeric_limits<double>::quiet_NaN());
        const double z = point.value("z").toDouble(
            std::numeric_limits<double>::quiet_NaN());
        if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) {
            last_error_ = "Robot path contains a non-finite point";
            world_points_.clear();
            return false;
        }
        const QVector3D candidate(
            static_cast<float>(x), static_cast<float>(y), static_cast<float>(z));
        if (world_points_.empty() ||
            (candidate - world_points_.back()).length() > 1.0e-5f)
            world_points_.push_back(candidate);
    }
    if (world_points_.size() < 2) {
        last_error_ = "Robot path needs at least two distinct points";
        world_points_.clear();
        return false;
    }
    has_final_look_target_ = false;
    final_turn_active_ = false;
    rebuildDistances();
    emit playbackStateChanged(
        QString("Path ready: %1 m, %2 points")
            .arg(total_distance_, 0, 'f', 2)
            .arg(static_cast<qulonglong>(world_points_.size())));
    return true;
}

void RobotPathPlayer::rebuildDistances()
{
    cumulative_distance_.clear();
    cumulative_distance_.reserve(world_points_.size());
    cumulative_distance_.push_back(0.0);
    for (std::size_t index = 1; index < world_points_.size(); ++index) {
        cumulative_distance_.push_back(
            cumulative_distance_.back() +
            static_cast<double>((world_points_[index] -
                                 world_points_[index - 1]).length()));
    }
    total_distance_ = cumulative_distance_.back();
    distance_ = 0.0;
    pitch_degrees_ = 0.0f;
    emit progressChanged(0.0f);
}

bool RobotPathPlayer::setPlannedPath(
    const std::vector<QVector3D>& world_points,
    const QVector3D& final_look_target_world)
{
    stop();
    world_points_.clear();
    for (const QVector3D& point : world_points) {
        if (world_points_.empty() ||
            (point - world_points_.back()).length() > 1.0e-5f)
            world_points_.push_back(point);
    }
    if (world_points_.empty()) {
        last_error_ = "Planned route contains no points";
        return false;
    }
    final_look_target_world_ = final_look_target_world;
    has_final_look_target_ = true;
    final_turn_active_ = false;
    last_error_.clear();
    rebuildDistances();
    emit playbackStateChanged(
        QString("Navigation ready: %1 m").arg(total_distance_, 0, 'f', 2));
    return true;
}

void RobotPathPlayer::setWorldToAlignedTransform(const QMatrix4x4& transform)
{
    world_to_aligned_ = transform;
}

void RobotPathPlayer::setSpeedMetersPerSecond(float speed)
{
    speed_ = std::clamp(speed, 0.05f, 2.0f);
}

QVector3D RobotPathPlayer::sampleWorldPosition(double distance) const
{
    distance = std::clamp(distance, 0.0, total_distance_);
    const auto upper = std::upper_bound(
        cumulative_distance_.begin(), cumulative_distance_.end(), distance);
    if (upper == cumulative_distance_.begin()) return world_points_.front();
    if (upper == cumulative_distance_.end()) return world_points_.back();
    const std::size_t next = static_cast<std::size_t>(
        upper - cumulative_distance_.begin());
    const std::size_t previous = next - 1;
    const double length = cumulative_distance_[next] -
        cumulative_distance_[previous];
    const float amount = length > 1.0e-9
        ? static_cast<float>((distance - cumulative_distance_[previous]) / length)
        : 0.0f;
    return world_points_[previous] * (1.0f - amount) +
        world_points_[next] * amount;
}

QVector3D RobotPathPlayer::currentWorldPosition() const
{
    return world_points_.empty() ? QVector3D() : sampleWorldPosition(distance_);
}

float RobotPathPlayer::desiredYaw(double distance) const
{
    if (world_points_.size() == 1) return yaw_degrees_;
    const QVector3D position =
        world_to_aligned_.map(sampleWorldPosition(distance));
    const QVector3D look_at = world_to_aligned_.map(sampleWorldPosition(
        std::min(distance + kLookAheadMeters, total_distance_)));
    QVector3D direction = look_at - position;
    direction.setZ(0.0f);
    if (direction.lengthSquared() < 1.0e-10f && distance > 0.0) {
        direction = position - world_to_aligned_.map(sampleWorldPosition(
            std::max(0.0, distance - kLookAheadMeters)));
        direction.setZ(0.0f);
    }
    return qRadiansToDegrees(std::atan2(direction.x(), direction.y()));
}

void RobotPathPlayer::emitPose(bool initialize_yaw)
{
    if (!hasPath()) return;
    if (initialize_yaw) yaw_degrees_ = desiredYaw(distance_);
    emit poseChanged(
        world_to_aligned_.map(sampleWorldPosition(distance_)),
        yaw_degrees_, pitch_degrees_);
}

void RobotPathPlayer::togglePlayPause()
{
    if (!hasPath()) {
        emit playbackStateChanged("Load a saved path before playback");
        return;
    }
    if (timer_.isActive()) {
        timer_.stop();
        emit playbackStateChanged("Playback paused");
        return;
    }
    if (total_distance_ <= 1.0e-9) {
        emitPose(true);
        if (has_final_look_target_) {
            final_turn_active_ = true;
            elapsed_.restart();
            timer_.start();
            emit playbackStateChanged(
                "Start is already at the destination; turning toward object");
        } else {
            emit playbackStateChanged("Start is already at the destination");
            emit playbackFinished();
        }
        return;
    }
    if (distance_ >= total_distance_) distance_ = 0.0;
    emitPose(distance_ == 0.0);
    elapsed_.restart();
    timer_.start();
    emit playbackStateChanged(
        QString("Playing at %1 m/s").arg(speed_, 0, 'f', 2));
}

void RobotPathPlayer::stop()
{
    timer_.stop();
    distance_ = 0.0;
    if (hasPath()) {
        emitPose(true);
        emit progressChanged(0.0f);
    }
    emit playbackStateChanged("Playback stopped");
}

void RobotPathPlayer::clearPath()
{
    timer_.stop();
    world_points_.clear();
    cumulative_distance_.clear();
    distance_ = 0.0;
    total_distance_ = 0.0;
    yaw_degrees_ = 0.0f;
    pitch_degrees_ = 0.0f;
    has_final_look_target_ = false;
    final_turn_active_ = false;
    last_error_.clear();
    emit progressChanged(0.0f);
    emit playbackStateChanged("No robot path loaded");
}

void RobotPathPlayer::updateFrame()
{
    const float dt = std::min(elapsed_.restart() / 1000.0f, 0.1f);
    if (final_turn_active_) {
        const QVector3D eye = world_to_aligned_.map(world_points_.back());
        const QVector3D target = world_to_aligned_.map(final_look_target_world_);
        const QVector3D direction = (target - eye).normalized();
        const float target_yaw = qRadiansToDegrees(
            std::atan2(direction.x(), direction.y()));
        // Route points live on the floor plane, while the renderer raises the
        // camera to eye height. Pitching from this floor-level position makes
        // the view sweep toward the ceiling. The controller performs the
        // eye-height pitch adjustment after the horizontal turn completes.
        const float target_pitch = 0.0f;
        float yaw_error = shortestAngleDifference(target_yaw, yaw_degrees_);
        float pitch_error = target_pitch - pitch_degrees_;
        yaw_degrees_ += std::clamp(yaw_error,
            -kMaximumYawRateDegreesPerSecond * dt,
             kMaximumYawRateDegreesPerSecond * dt);
        pitch_degrees_ += std::clamp(pitch_error,
            -kMaximumPitchRateDegreesPerSecond * dt,
             kMaximumPitchRateDegreesPerSecond * dt);
        emitPose(false);
        if (std::abs(yaw_error) < 0.5f && std::abs(pitch_error) < 0.5f) {
            final_turn_active_ = false;
            timer_.stop();
            emit playbackStateChanged("Navigation finished; object centered");
            emit playbackFinished();
        }
        return;
    }
    distance_ = std::min(total_distance_, distance_ + speed_ * dt);
    const float target_yaw = desiredYaw(distance_);
    const float smoothing = 1.0f - std::exp(-dt / kYawTimeConstantSeconds);
    float yaw_step = smoothing *
        shortestAngleDifference(target_yaw, yaw_degrees_);
    const float maximum_step = kMaximumYawRateDegreesPerSecond * dt;
    yaw_step = std::clamp(yaw_step, -maximum_step, maximum_step);
    yaw_degrees_ += yaw_step;
    emitPose(false);
    emit progressChanged(total_distance_ > 1.0e-9
        ? static_cast<float>(distance_ / total_distance_) : 1.0f);
    if (distance_ >= total_distance_) {
        if (has_final_look_target_) {
            final_turn_active_ = true;
            emit playbackStateChanged("At destination; turning toward object");
        } else {
            timer_.stop();
            emit playbackStateChanged("Playback finished at final point");
            emit playbackFinished();
        }
    }
}
