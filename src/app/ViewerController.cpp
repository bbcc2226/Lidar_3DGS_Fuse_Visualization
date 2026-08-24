#include "ViewerController.h"

#include "OpenGLWidget.h"
#include "OrientationGizmo.h"

#include <QApplication>
#include <QDir>
#include <QFileDialog>
#include <QMouseEvent>
#include <QWheelEvent>
#include <QWidget>

#include <algorithm>
#include <cmath>

namespace
{
constexpr float kRotationDegreesPerPixel = 0.5f;
constexpr float kInitialCameraDistance = 3.0f;
constexpr float kMinimumCameraDistance = 0.02f;
constexpr float kMaximumCameraDistance = 100.0f;
constexpr float kDollyFactor = 0.85f;
constexpr float kHalfVerticalFieldOfViewRadians = 22.5f * 3.14159265f / 180.0f;
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

    const bool loaded = point_processing_.loadPly(path.toStdString());
    QApplication::restoreOverrideCursor();

    if (!loaded) {
        viewer_->setGaussianPoints({});
        emit loadStatusChanged(
            "Load failed: " + QString::fromStdString(point_processing_.lastError()),
            path);
        return;
    }

    const GaussianPlyMetadata& metadata = point_processing_.metadata();
    viewer_->setGaussianPoints(
        point_processing_.points(), metadata.has_scale);
    const QString data_description = metadata.isComplete3DGS()
        ? QString("3DGS | SH degree %1").arg(metadata.sh_degree)
        : QString("point cloud");
    emit loadStatusChanged(
        QString("Loaded %1 points | uploaded %2 | %3")
            .arg(static_cast<qulonglong>(point_processing_.splatCount()))
            .arg(static_cast<qulonglong>(viewer_->uploadedSplatCount()))
            .arg(data_description),
        path);
}

void ViewerController::resetView()
{
    yaw_degrees_ = 0.0f;
    pitch_degrees_ = 0.0f;
    translation_ = {};
    camera_distance_ = kInitialCameraDistance;
    applyTransform();
}

void ViewerController::applyTransform()
{
    QMatrix4x4 transform;
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
    case QEvent::MouseButtonPress: {
        auto* mouse_event = static_cast<QMouseEvent*>(event);
        if (mouse_event->button() == Qt::LeftButton) {
            const auto axis = OrientationGizmo::hitTest(
                mouse_event->pos(), viewer_->size(), yaw_degrees_, pitch_degrees_);
            if (axis != OrientationGizmo::Axis::None) {
                snapToAxis(static_cast<int>(axis));
                mouse_event->accept();
                return true;
            }
            last_mouse_position_ = mouse_event->pos();
            drag_mode_ = DragMode::Rotate;
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
        if (drag_mode_ == DragMode::Rotate &&
            (mouse_event->buttons() & Qt::LeftButton)) {
            const QPoint delta = mouse_event->pos() - last_mouse_position_;
            last_mouse_position_ = mouse_event->pos();
            yaw_degrees_ = std::fmod(
                yaw_degrees_ + delta.x() * kRotationDegreesPerPixel, 360.0f);
            pitch_degrees_ = std::clamp(
                pitch_degrees_ + delta.y() * kRotationDegreesPerPixel,
                -89.0f, 89.0f);
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
            const float world_units_per_pixel = visible_world_height /
                std::max(1, viewer_->height());
            translation_ += QVector3D(
                delta.x() * world_units_per_pixel,
                -delta.y() * world_units_per_pixel,
                0.0f);
            applyTransform();
            mouse_event->accept();
            return true;
        }
        break;
    }
    case QEvent::MouseButtonRelease: {
        auto* mouse_event = static_cast<QMouseEvent*>(event);
        if ((mouse_event->button() == Qt::LeftButton &&
             drag_mode_ == DragMode::Rotate) ||
            (mouse_event->button() == Qt::RightButton &&
             drag_mode_ == DragMode::Pan)) {
            drag_mode_ = DragMode::None;
            mouse_event->accept();
            return true;
        }
        break;
    }
    case QEvent::Wheel: {
        auto* wheel_event = static_cast<QWheelEvent*>(event);
        camera_distance_ = std::clamp(
            camera_distance_ *
                (wheel_event->angleDelta().y() > 0
                    ? kDollyFactor
                    : 1.0f / kDollyFactor),
            kMinimumCameraDistance, kMaximumCameraDistance);
        applyTransform();
        wheel_event->accept();
        return true;
    }
    default:
        break;
    }

    return QObject::eventFilter(watched, event);
}
