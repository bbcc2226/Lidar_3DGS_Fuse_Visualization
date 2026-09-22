#include "ViewerController.h"

#include "OpenGLWidget.h"
#include "OrientationGizmo.h"
#include "RobotPathPlayer.h"
#include "SemanticQuery.h"
#include "viewer_controller/ViewerControllerInternals.h"

#include <QApplication>
#include <QDir>
#include <QFileDialog>
#include <QKeyEvent>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QMouseEvent>
#include <QSaveFile>
#include <QFile>
#include <QtMath>
#include <QWheelEvent>
#include <QWidget>
#include <QVariantAnimation>
#include <QEasingCurve>
#include <QDateTime>
#include <QFileInfo>
#include <QPainterPath>
#include <QPolygonF>
#include <QVector2D>
#include <QtConcurrent/QtConcurrent>

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <queue>

// Core controller setup, camera state, and arrival-view transitions.
using namespace viewer_controller_detail;

ViewerController::ViewerController(OpenGLWidget* viewer, QObject* parent)
    : QObject(parent), viewer_(viewer)
{
    robot_path_player_ = new RobotPathPlayer(this);
    arrival_view_animation_ = new QVariantAnimation(this);
    lidar_watcher_ = new QFutureWatcher<std::shared_ptr<std::vector<GaussianPoint>>>(this);
    connect(lidar_watcher_, &QFutureWatcherBase::finished, this, [this]() {
        lidar_load_in_progress_ = false;
        const std::shared_ptr<std::vector<GaussianPoint>> result = lidar_watcher_->result();
        if (!result || result->empty()) {
            emit loadStatusChanged("LiDAR preparation failed", lidar_loading_path_);
            return;
        }
        lidar_points_ = *result;
        lidar_loaded_path_ = lidar_loading_path_;
        emit loadStatusChanged(QString("LiDAR prepared: %1 points — %2")
            .arg(static_cast<qulonglong>(lidar_points_.size()))
            .arg(QFileInfo(lidar_loaded_path_).fileName()), lidar_loaded_path_);
        if (lidar_view_enabled_) setLidarViewEnabled(true);
    });
    // Arrival is deliberately slower than path steering: a person naturally
    // settles their gaze after stopping instead of snapping toward the target.
    arrival_view_animation_->setDuration(1600);
    arrival_view_animation_->setStartValue(0.0);
    arrival_view_animation_->setEndValue(1.0);
    arrival_view_animation_->setEasingCurve(QEasingCurve::InOutCubic);
    connect(arrival_view_animation_, &QVariantAnimation::valueChanged, this,
            [this](const QVariant& value) {
                if (!has_navigation_target_) return;
                const float amount = value.toFloat();
                // Raise/lower first, then turn. Keeping these phases separate
                // avoids the unnatural ceiling sweep caused by simultaneous
                // height and pitch changes.
                const float height_amount = std::min(amount / 0.42f, 1.0f);
                const float look_amount = std::clamp(
                    (amount - 0.42f) / 0.58f, 0.0f, 1.0f);
                z_up_camera_position_ =
                    arrival_view_start_position_ * (1.0f - height_amount) +
                    arrival_view_end_position_ * height_amount;
                yaw_degrees_ = arrival_view_start_yaw_ * (1.0f - look_amount) +
                    arrival_view_end_yaw_ * look_amount;
                pitch_degrees_ = arrival_view_start_pitch_ * (1.0f - look_amount) +
                    arrival_view_end_pitch_ * look_amount;
                // Widen the arrival framing as the target is centered.
                viewer_->setVerticalFieldOfView(45.0f + 10.0f * look_amount);
                applyTransform();
            });
    connect(arrival_view_animation_, &QVariantAnimation::finished, this,
            [this]() { viewer_->setNavigationTargetTagVisible(true); });
    connect(robot_path_player_, &RobotPathPlayer::poseChanged, this,
            [this](const QVector3D& position, float yaw, float pitch) {
                robot_pose_initialized_ = true;
                constrained_z_up_navigation_ = true;
                path_editing_enabled_ = false;
                z_up_camera_position_ = position;
                z_up_camera_position_.setZ(
                    playbackCameraHeight(position.z()));
                yaw_degrees_ = yaw;
                pitch_degrees_ = pitch;
                viewer_->setPathEditMode(false);
                viewer_->setZUpGizmo(true);
                applyTransform();
            });
    connect(robot_path_player_, &RobotPathPlayer::playbackStateChanged, this,
            [this](const QString& text) {
                emit robotPlaybackStatusChanged(text, QString());
            });
    connect(robot_path_player_, &RobotPathPlayer::playbackFinished, this,
            [this]() {
                beginArrivalView();
            });
    viewer_->installEventFilter(this);
    applyTransform();
}

void ViewerController::setUseOriginalTrajectoryHeight(bool enabled)
{
    use_original_trajectory_height_ = enabled;
    arrival_view_animation_->stop();
    if (constrained_z_up_navigation_ && robot_path_player_->hasPath()) {
        const float height = viewer_->sceneWorldToAlignedTransform().map(
            robot_path_player_->currentWorldPosition()).z();
        z_up_camera_position_.setZ(playbackCameraHeight(height));
        applyTransform();
    }
    emit originalTrajectoryHeightChanged(enabled);
}

float ViewerController::playbackCameraHeight(float recorded_height) const
{
    if (use_original_trajectory_height_) {
        const float scale = viewer_->sceneWorldToAlignedTransform().mapVector(
            QVector3D(1.0f, 0.0f, 0.0f)).length();
        return recorded_height + recorded_height_offset_meters_ * scale;
    }
    return viewer_->robotEyeHeightAligned(navigation_eye_height_meters_, recorded_height);
}

void ViewerController::restoreNavigationView()
{
    arrival_view_animation_->stop();
    viewer_->setVerticalFieldOfView(45.0f);
    pitch_degrees_ = 0.0f;
    if (constrained_z_up_navigation_) {
        const float recorded_height = use_original_trajectory_height_
            ? viewer_->sceneWorldToAlignedTransform().map(
                robot_path_player_->currentWorldPosition()).z()
            : z_up_camera_position_.z();
        z_up_camera_position_.setZ(playbackCameraHeight(recorded_height));
    }
    applyTransform();
}

void ViewerController::beginArrivalView()
{
    if (!has_navigation_target_ || path_editing_enabled_ ||
        free_zone_editing_enabled_) return;
    arrival_view_start_position_ = z_up_camera_position_;
    arrival_view_start_yaw_ = yaw_degrees_;
    arrival_view_start_pitch_ = pitch_degrees_;
    const float units_per_meter = std::max(
        viewer_->sceneWorldToAlignedTransform().mapVector(
            QVector3D(1.0f, 0.0f, 0.0f)).length(), 1.0e-5f);
    const float floor = viewer_->robotEyeHeightAligned(
        0.0f, z_up_camera_position_.z() -
                  navigation_eye_height_meters_ * units_per_meter);
    const float target_height =
        navigation_target_center_aligned_.z() - floor;
    const float arrival_height = std::clamp(
        target_height, 0.85f * units_per_meter,
        1.35f * units_per_meter);
    arrival_view_end_position_ = z_up_camera_position_;
    if (!use_original_trajectory_height_)
        arrival_view_end_position_.setZ(floor + arrival_height);
    const QVector3D direction =
        (navigation_target_center_aligned_ - arrival_view_end_position_).normalized();
    const float target_yaw =
        qRadiansToDegrees(std::atan2(direction.x(), direction.y()));
    arrival_view_end_pitch_ = -qRadiansToDegrees(std::asin(
        std::clamp(direction.z(), -1.0f, 1.0f)));
    // Interpolate the shortest turn across the -180/180 degree boundary.
    float yaw_delta = std::fmod(target_yaw - arrival_view_start_yaw_ + 540.0f,
                                360.0f) - 180.0f;
    arrival_view_end_yaw_ = arrival_view_start_yaw_ + yaw_delta;
    arrival_view_animation_->start();
}
