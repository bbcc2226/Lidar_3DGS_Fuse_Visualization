#include "ViewerController.h"

#include "OpenGLWidget.h"
#include "OrientationGizmo.h"

#include <QApplication>
#include <QDir>
#include <QFileDialog>
#include <QKeyEvent>
#include <QMouseEvent>
#include <QtMath>
#include <QWheelEvent>
#include <QWidget>

#include <algorithm>
#include <cmath>

namespace
{
constexpr float kRotationDegreesPerPixel = 0.5f;
constexpr float kConstrainedRotationDegreesPerPixel = 0.15f;
constexpr float kInitialCameraDistance = 3.0f;
constexpr float kNavigateInitialCameraDistance = 0.65f;
constexpr float kMinimumCameraDistance = 0.02f;
constexpr float kMaximumCameraDistance = 100.0f;
constexpr float kDollyFactorPerWheelStep = 0.72f;
constexpr float kPanSensitivity = 1.75f;
constexpr float kHalfVerticalFieldOfViewRadians = 22.5f * 3.14159265f / 180.0f;
constexpr int kConstrainedDragThresholdPixels = 6;
constexpr float kVerticalMoveFraction = 0.05f;
}

ViewerController::ViewerController(OpenGLWidget* viewer, QObject* parent)
    : QObject(parent), viewer_(viewer)
{
    viewer_->installEventFilter(this);
    applyTransform();
}

void ViewerController::openPlyFile(QWidget* dialog_parent)
{
    QFileDialog dialog(
        dialog_parent, "Open Gaussian or point-cloud PLY", QDir::homePath());
    dialog.setAcceptMode(QFileDialog::AcceptOpen);
    dialog.setFileMode(QFileDialog::ExistingFile);
    dialog.setNameFilters({"PLY files (*.ply)", "All files (*)"});
    dialog.setOption(QFileDialog::DontUseNativeDialog, true);

    if (dialog.exec() != QDialog::Accepted) return;

    const QStringList selected_files = dialog.selectedFiles();
    if (selected_files.isEmpty()) return;
    const QString path = selected_files.constFirst();

    emit loadStatusChanged("Loading " + path + "...", path);
    QApplication::setOverrideCursor(Qt::WaitCursor);
    QApplication::processEvents();

    // Release the previous renderer scene before the loader allocates a new
    // large PLY representation, avoiding both scenes overlapping in memory.
    viewer_->setGaussianPoints({});
    const bool loaded = point_processing_.loadPly(path.toStdString());
    QApplication::restoreOverrideCursor();

    if (!loaded) {
        viewer_->setGaussianPoints({});
        emit loadStatusChanged(
            "Load failed: " + QString::fromStdString(point_processing_.lastError()),
            path);
        return;
    }

    const GaussianPlyMetadata metadata = point_processing_.metadata();
    const std::size_t splat_count = point_processing_.splatCount();
    viewer_->setGaussianPoints(
        point_processing_.points(), metadata.has_scale, metadata.has_sh_dc,
        metadata.sh_degree);
    if (constrained_z_up_navigation_) {
        // A newly loaded scene needs its own alignment and provisional start.
        setConstrainedZUpNavigation(true);
    }
    point_processing_.clear();
    const QString data_description = metadata.isComplete3DGS()
        ? QString("3DGS | SH degree %1").arg(metadata.sh_degree)
        : QString("point cloud");
    emit loadStatusChanged(
        QString("Loaded %1 points | uploaded %2 | %3")
            .arg(static_cast<qulonglong>(splat_count))
            .arg(static_cast<qulonglong>(viewer_->uploadedSplatCount()))
            .arg(data_description),
        path);
}

void ViewerController::resetView()
{
    yaw_degrees_ = 0.0f;
    pitch_degrees_ = 0.0f;
    translation_ = {};
    camera_distance_ = constrained_z_up_navigation_
        ? kNavigateInitialCameraDistance : kInitialCameraDistance;
    z_up_camera_position_ = QVector3D(0.0f, -camera_distance_, 0.0f);
    applyTransform();
}

void ViewerController::setConstrainedZUpNavigation(bool enabled)
{
    if (enabled) {
        if (viewer_->autoAlignSceneUp()) {
            emit floorAlignmentStatusChanged(
                "Automatically aligned the scene's shortest axis to +Z.");
        } else {
            emit floorAlignmentStatusChanged(
                "Automatic Z-up alignment failed; use three floor points.");
        }
    }
    constrained_z_up_navigation_ = enabled;
    yaw_degrees_ = 0.0f;
    pitch_degrees_ = 0.0f;
    if (enabled) {
        // Enter Navigate mode near the center of the normalized scene. This is
        // a provisional indoor pose until a trajectory start pose is supplied.
        translation_ = {};
        camera_distance_ = kNavigateInitialCameraDistance;
        z_up_camera_position_ = QVector3D(
            0.0f, -kNavigateInitialCameraDistance, 0.0f);
    } else {
        // Explore mode returns to a complete-scene orbit view.
        translation_ = {};
        camera_distance_ = kInitialCameraDistance;
    }
    drag_mode_ = DragMode::None;
    viewer_->setZUpGizmo(enabled);
    applyTransform();
}

void ViewerController::applyTransform()
{
    QMatrix4x4 transform;
    if (constrained_z_up_navigation_) {
        const float yaw_radians =
            qDegreesToRadians(yaw_degrees_);
        const float pitch_radians =
            qDegreesToRadians(pitch_degrees_);
        const float cos_pitch = std::cos(pitch_radians);
        const QVector3D forward(
            std::sin(yaw_radians) * cos_pitch,
            std::cos(yaw_radians) * cos_pitch,
            -std::sin(pitch_radians));
        const QVector3D world_up(0.0f, 0.0f, 1.0f);
        const QVector3D right =
            QVector3D::crossProduct(forward, world_up).normalized();
        const QVector3D camera_up =
            QVector3D::crossProduct(right, forward).normalized();

        QMatrix4x4 desired_view;
        desired_view.lookAt(
            z_up_camera_position_, z_up_camera_position_ + forward, camera_up);

        // OpenGLWidget owns a fixed view at (0,0,3). Pre-cancel it so the
        // resulting model-view matrix is the first-person Z-up camera above.
        QMatrix4x4 inverse_fixed_view;
        inverse_fixed_view.translate(0.0f, 0.0f, kInitialCameraDistance);
        transform = inverse_fixed_view * desired_view;
        viewer_->setInteractionTransform(
            transform, yaw_degrees_, pitch_degrees_);
        viewer_->setNavigationPose(z_up_camera_position_, yaw_degrees_);
        emit orientationChanged(yaw_degrees_, pitch_degrees_);
        return;
    }

    QVector3D model_offset = translation_;
    // The render camera stays at Z=3. Moving the normalized cloud toward it
    // is equivalent to dollying the camera toward the cloud.
    model_offset.setZ(kInitialCameraDistance - camera_distance_);
    transform.translate(model_offset);
    transform.rotate(pitch_degrees_, 1.0f, 0.0f, 0.0f);
    transform.rotate(yaw_degrees_, 0.0f, 1.0f, 0.0f);
    viewer_->setInteractionTransform(transform, yaw_degrees_, pitch_degrees_);
    emit orientationChanged(yaw_degrees_, pitch_degrees_);
}

void ViewerController::snapToAxis(int axis)
{
    switch (static_cast<OrientationGizmo::Axis>(axis)) {
    case OrientationGizmo::Axis::X:
        yaw_degrees_ = 90.0f;
        pitch_degrees_ = 0.0f;
        break;
    case OrientationGizmo::Axis::Y:
        yaw_degrees_ = 0.0f;
        pitch_degrees_ = -90.0f;
        break;
    case OrientationGizmo::Axis::Z:
        yaw_degrees_ = 0.0f;
        pitch_degrees_ = 0.0f;
        break;
    case OrientationGizmo::Axis::None:
        return;
    }
    applyTransform();
}

bool ViewerController::eventFilter(QObject* watched, QEvent* event)
{
    if (watched != viewer_) return QObject::eventFilter(watched, event);

    switch (event->type()) {
    case QEvent::KeyPress: {
        auto* key_event = static_cast<QKeyEvent*>(event);
        if (constrained_z_up_navigation_ &&
            (key_event->key() == Qt::Key_Up ||
             key_event->key() == Qt::Key_Down)) {
            const float vertical_step = std::max(
                camera_distance_, kMinimumCameraDistance) *
                kVerticalMoveFraction;
            z_up_camera_position_.setZ(
                z_up_camera_position_.z() +
                (key_event->key() == Qt::Key_Up
                    ? vertical_step
                    : -vertical_step));
            applyTransform();
            key_event->accept();
            return true;
        }
        break;
    }
    case QEvent::MouseButtonPress: {
        auto* mouse_event = static_cast<QMouseEvent*>(event);
        if (mouse_event->button() == Qt::LeftButton) {
            const auto axis = OrientationGizmo::hitTest(
                mouse_event->pos(), viewer_->size(), yaw_degrees_, pitch_degrees_,
                constrained_z_up_navigation_);
            if (axis != OrientationGizmo::Axis::None) {
                snapToAxis(static_cast<int>(axis));
                mouse_event->accept();
                return true;
            }
            last_mouse_position_ = mouse_event->pos();
            drag_start_position_ = mouse_event->pos();
            if (mouse_event->modifiers() & Qt::ShiftModifier) {
                drag_mode_ = DragMode::RotatePitchOnly;
            } else {
                drag_mode_ = constrained_z_up_navigation_
                    ? DragMode::RotatePending
                    : DragMode::RotateFree;
            }
            mouse_event->accept();
            return true;
        }
        if (mouse_event->button() == Qt::RightButton) {
            last_mouse_position_ = mouse_event->pos();
            drag_mode_ = DragMode::Pan;
            mouse_event->accept();
            return true;
        }
        break;
    }
    case QEvent::MouseMove: {
        auto* mouse_event = static_cast<QMouseEvent*>(event);
        if ((drag_mode_ == DragMode::RotateFree ||
             drag_mode_ == DragMode::RotatePending ||
             drag_mode_ == DragMode::RotateYawOnly ||
             drag_mode_ == DragMode::RotatePitchOnly) &&
            (mouse_event->buttons() & Qt::LeftButton)) {
            QPoint delta = mouse_event->pos() - last_mouse_position_;
            if (drag_mode_ == DragMode::RotatePending) {
                const QPoint total_delta =
                    mouse_event->pos() - drag_start_position_;
                const float horizontal =
                    static_cast<float>(std::abs(total_delta.x()));
                const float vertical =
                    static_cast<float>(std::abs(total_delta.y()));
                if (std::max(horizontal, vertical) <
                    kConstrainedDragThresholdPixels) {
                    mouse_event->accept();
                    return true;
                }
                // Select exactly one axis for the lifetime of this drag.
                // Later diagonal motion cannot introduce roll or alter the
                // other angle.
                drag_mode_ = horizontal >= vertical
                    ? DragMode::RotateYawOnly
                    : DragMode::RotatePitchOnly;
                delta = total_delta;
            }
            last_mouse_position_ = mouse_event->pos();
            if (drag_mode_ == DragMode::RotateFree ||
                drag_mode_ == DragMode::RotateYawOnly) {
                const float sensitivity = constrained_z_up_navigation_
                    ? kConstrainedRotationDegreesPerPixel
                    : kRotationDegreesPerPixel;
                yaw_degrees_ = std::fmod(
                    yaw_degrees_ + delta.x() * sensitivity,
                    360.0f);
            }
            if (drag_mode_ == DragMode::RotateFree ||
                drag_mode_ == DragMode::RotatePitchOnly) {
                const float sensitivity = constrained_z_up_navigation_
                    ? kConstrainedRotationDegreesPerPixel
                    : kRotationDegreesPerPixel;
                pitch_degrees_ = std::clamp(
                    pitch_degrees_ + delta.y() * sensitivity,
                    -89.0f, 89.0f);
            }
            applyTransform();
            mouse_event->accept();
            return true;
        }
        if (drag_mode_ == DragMode::Pan &&
            (mouse_event->buttons() & Qt::RightButton)) {
            const QPoint delta = mouse_event->pos() - last_mouse_position_;
            last_mouse_position_ = mouse_event->pos();
            const float visible_world_height =
                2.0f * std::tan(kHalfVerticalFieldOfViewRadians) *
                std::max(camera_distance_, kMinimumCameraDistance);
            const float world_units_per_pixel = kPanSensitivity *
                visible_world_height / std::max(1, viewer_->height());
            if (constrained_z_up_navigation_) {
                const float yaw_radians = qDegreesToRadians(yaw_degrees_);
                const QVector3D right(
                    std::cos(yaw_radians), -std::sin(yaw_radians), 0.0f);
                const QVector3D ground_forward(
                    std::sin(yaw_radians), std::cos(yaw_radians), 0.0f);
                z_up_camera_position_ +=
                    -right * (delta.x() * world_units_per_pixel) +
                    ground_forward * (delta.y() * world_units_per_pixel);
            } else {
                translation_ += QVector3D(
                    delta.x() * world_units_per_pixel,
                    -delta.y() * world_units_per_pixel,
                    0.0f);
            }
            applyTransform();
            mouse_event->accept();
            return true;
        }
        break;
    }
    case QEvent::MouseButtonRelease: {
        auto* mouse_event = static_cast<QMouseEvent*>(event);
        if ((mouse_event->button() == Qt::LeftButton &&
             (drag_mode_ == DragMode::RotateFree ||
              drag_mode_ == DragMode::RotatePending ||
              drag_mode_ == DragMode::RotateYawOnly ||
              drag_mode_ == DragMode::RotatePitchOnly)) ||
            (mouse_event->button() == Qt::RightButton &&
             drag_mode_ == DragMode::Pan)) {
            const bool finished_rotation = drag_mode_ != DragMode::Pan;
            drag_mode_ = DragMode::None;
            if (finished_rotation) viewer_->finalizeInteractionSort();
            mouse_event->accept();
            return true;
        }
        break;
    }
    case QEvent::Wheel: {
        auto* wheel_event = static_cast<QWheelEvent*>(event);
        const int angle_delta = wheel_event->angleDelta().y();
        const int pixel_delta = wheel_event->pixelDelta().y();
        const float wheel_steps = angle_delta != 0
            ? static_cast<float>(angle_delta) / 120.0f
            : static_cast<float>(pixel_delta) / 60.0f;
        if (std::abs(wheel_steps) < 1.0e-4f) break;
        const float previous_distance = camera_distance_;
        camera_distance_ = std::clamp(
            camera_distance_ * std::pow(
                kDollyFactorPerWheelStep, wheel_steps),
            kMinimumCameraDistance, kMaximumCameraDistance);
        if (constrained_z_up_navigation_) {
            const float yaw_radians = qDegreesToRadians(yaw_degrees_);
            const QVector3D ground_forward(
                std::sin(yaw_radians), std::cos(yaw_radians), 0.0f);
            z_up_camera_position_ +=
                ground_forward * (previous_distance - camera_distance_);
        }
        applyTransform();
        wheel_event->accept();
        return true;
    }
    default:
        break;
    }

    return QObject::eventFilter(watched, event);
}
