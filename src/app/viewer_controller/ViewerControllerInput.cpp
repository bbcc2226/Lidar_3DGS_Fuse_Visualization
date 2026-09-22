#include "ViewerController.h"
#include "OpenGLWidget.h"
#include "OrientationGizmo.h"
#include "RobotPathPlayer.h"
#include "SemanticQuery.h"
#include "ViewerControllerInternals.h"

#include <QApplication>
#include <QDateTime>
#include <QEasingCurve>
#include <QFile>
#include <QFileDialog>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QKeyEvent>
#include <QMouseEvent>
#include <QPainterPath>
#include <QPolygonF>
#include <QSaveFile>
#include <QVariantAnimation>
#include <QVector2D>
#include <QWheelEvent>
#include <QWidget>
#include <QtConcurrent/QtConcurrent>
#include <QtMath>

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <queue>

using namespace viewer_controller_detail;

// Camera transforms, orientation controls, and viewport event handling.
void ViewerController::refreshLoadedRobotPathDisplay()
{
    if (!robot_path_player_->hasPath()) return;
    manual_path_.clear();
    manual_path_.reserve(robot_path_player_->worldPoints().size());
    const QMatrix4x4 transform = viewer_->sceneWorldToAlignedTransform();
    for (const QVector3D& world : robot_path_player_->worldPoints())
        manual_path_.push_back(transform.map(world));
    selected_manual_path_point_ = -1;
    viewer_->setManualPath(manual_path_, selected_manual_path_point_);
}

void ViewerController::applyTransform()
{
    //very important rendering function
    QMatrix4x4 transform;
    if (constrained_z_up_navigation_) {
        /* constrained_z_up_navigation_ == true
        │     → First-person / robot navigation camera
        │     → Camera actually has a world position
        │     → Z is always world-up*/

        /*
        z_up_camera_position_   // where am I?
        yaw_degrees_            // which horizontal direction am I facing?
        pitch_degrees_          // am I looking up/down?
        The goal of the following code is to turn those three things into a view matrix.
       */

        //You're converting spherical-angle-like yaw/pitch into a unit direction vector:
        const float yaw_radians =
            qDegreesToRadians(yaw_degrees_);
        const float pitch_radians =
            qDegreesToRadians(pitch_degrees_);
        const float cos_pitch = std::cos(pitch_radians);
        const QVector3D forward(
            std::sin(yaw_radians) * cos_pitch,
            std::cos(yaw_radians) * cos_pitch,
            -std::sin(pitch_radians));

        //The world is "z" up
        /*+Z
         ↑
         |
         |
         O────→ +X
        /
       /
     +Y*/
     /*
        That's appropriate for an indoor robot-navigation simulation: the robot can turn left/right and look up/down,
      but the world shouldn't suddenly tilt sideways.*/

        const QVector3D world_up(0.0f, 0.0f, 1.0f);


       /*These are the camera's coordinate axes:

       forward = where I'm looking
       right   = right side of my screen
       up      = top of my screen
        cross product gives you a vector perpendicular to both input vectors
      */
        const QVector3D right =
            QVector3D::crossProduct(forward, world_up).normalized();
        const QVector3D camera_up =
            QVector3D::crossProduct(right, forward).normalized();



        /*eye    = z_up_camera_position_
         center = z_up_camera_position_ + forward
         up     = camera_up
        */
        QMatrix4x4 desired_view;
        desired_view.lookAt(
            z_up_camera_position_, z_up_camera_position_ + forward, camera_up);

        // OpenGLWidget owns a fixed view at (0,0,3). Pre-cancel it so the
        // resulting model-view matrix is the first-person Z-up camera above.
        QMatrix4x4 inverse_fixed_view;
        inverse_fixed_view.translate(0.0f, 0.0f, kInitialCameraDistance);
        transform = inverse_fixed_view * desired_view;
        // “Render the 3D scene from this camera view.”
        viewer_->setInteractionTransform(
            transform, yaw_degrees_, pitch_degrees_);

        //The robot/navigation agent is logically here, facing this direction.
        viewer_->setNavigationPose(z_up_camera_position_, yaw_degrees_);
        emit orientationChanged(yaw_degrees_, pitch_degrees_);
        return;
    }
    /*constrained_z_up_navigation_ == false
      → Original viewer/orbit-style interaction
      → OpenGL camera remains fixed at (0,0,3)
      → Move/rotate the scene instead*/
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
 /*

 left mouse drag
     ↓
 eventFilter()
     ↓
 drag_mode_ = RotateFree or RotatePending
     ↓
 yaw_degrees_ / pitch_degrees_ change
     ↓
 applyTransform()
     ↓
 setInteractionTransform()
     ↓
 paintGL()

 For zoom:

 mouse wheel
     ↓
 eventFilter()
     ↓
 camera_distance_ changes
     ↓
 applyTransform()
     ↓
 setInteractionTransform()
     ↓
 paintGL()

 For Navigate mode:

 left drag / wheel / Up-Down key
     ↓
 eventFilter()
     ↓
 z_up_camera_position_, yaw, or pitch changes
     ↓
 applyTransform()
     ↓
 lookAt() creates first-person view
     ↓
 setInteractionTransform()
     ↓
 setNavigationPose()
     ↓
 paintGL()*/

    if (watched != viewer_) return QObject::eventFilter(watched, event);

    // During active playback the generated robot pose owns the camera. Pause
    // restores the normal Navigate controls; resuming emits the stored path
    // pose again and discards temporary inspection movement.
    if (robot_path_player_->isPlaying()) {
        switch (event->type()) {
        case QEvent::MouseButtonPress: {
            auto* mouse = static_cast<QMouseEvent*>(event);
            pressed_semantic_object_id_ = -1;
            if (mouse->button() == Qt::LeftButton) {
                drag_start_position_ = mouse->pos();
                pressed_semantic_object_id_ = viewer_->hitTestSemanticObject(mouse->pos());
            }
            event->accept();
            return true;
        }
        case QEvent::MouseMove: {
            auto* mouse = static_cast<QMouseEvent*>(event);
            if ((mouse->pos() - drag_start_position_).manhattanLength() >=
                QApplication::startDragDistance()) pressed_semantic_object_id_ = -1;
            event->accept();
            return true;
        }
        case QEvent::MouseButtonRelease: {
            auto* mouse = static_cast<QMouseEvent*>(event);
            if (mouse->button() == Qt::LeftButton && pressed_semantic_object_id_ >= 0 &&
                (mouse->pos() - drag_start_position_).manhattanLength() <
                    QApplication::startDragDistance()) {
                // Pick the box at press time: playback may move it before release.
                selectSemanticObject(pressed_semantic_object_id_, false);
            }
            pressed_semantic_object_id_ = -1;
            event->accept();
            return true;
        }
        case QEvent::KeyPress:
        case QEvent::Wheel:
            event->accept();
            return true;
        default:
            break;
        }
    }

    switch (event->type()) {
    case QEvent::KeyPress: {
        auto* key_event = static_cast<QKeyEvent*>(event);
        if (path_editing_enabled_ &&
            (key_event->key() == Qt::Key_Delete ||
             key_event->key() == Qt::Key_Backspace) &&
            selected_manual_path_point_ >= 0 &&
            selected_manual_path_point_ <
                static_cast<int>(manual_path_.size())) {
            manual_path_.erase(
                manual_path_.begin() + selected_manual_path_point_);
            selected_manual_path_point_ = -1;
            updateManualPathDisplay();
            key_event->accept();
            return true;
        }
        if (free_zone_editing_enabled_ &&
            (key_event->key() == Qt::Key_Delete ||
             key_event->key() == Qt::Key_Backspace) &&
            selected_free_zone_vertex_ >= 0 &&
            selected_free_zone_vertex_ < static_cast<int>(free_zone_vertices_.size())) {
            if (static_cast<std::size_t>(selected_free_zone_vertex_) <
                free_zone_seed_vertex_count_) {
                emit freeZoneStatusChanged(
                    "Shared-edge vertices are locked while drawing the joined polygon.",
                    QString());
                key_event->accept();
                return true;
            }
            free_zone_vertices_.erase(
                free_zone_vertices_.begin() + selected_free_zone_vertex_);
            selected_free_zone_vertex_ = -1;
            if (free_zone_vertices_.size() < 3) free_zone_closed_ = false;
            updateFreeZoneDisplay();
            key_event->accept();
            return true;
        }
        if (constrained_z_up_navigation_ &&
            (key_event->key() == Qt::Key_Up ||
             key_event->key() == Qt::Key_Down)) {
            const float vertical_step = std::max(
                camera_distance_, kMinimumCameraDistance) *
                kVerticalMoveFraction;
            const float height_delta = key_event->key() == Qt::Key_Up
                ? vertical_step : -vertical_step;
            const float units_per_meter = std::max(
                viewer_->sceneWorldToAlignedTransform().mapVector(
                    QVector3D(1.0f, 0.0f, 0.0f)).length(), 1.0e-5f);
            // Store the eye height in meters so playback poses and resume
            // preserve this adjustment, even when the scene is rescaled.
            if (use_original_trajectory_height_)
                recorded_height_offset_meters_ += height_delta / units_per_meter;
            else
                navigation_eye_height_meters_ += height_delta / units_per_meter;
            z_up_camera_position_.setZ(z_up_camera_position_.z() + height_delta);
            applyTransform();
            key_event->accept();
            return true;
        }
        break;
    }
    case QEvent::MouseButtonPress: {
        auto* mouse_event = static_cast<QMouseEvent*>(event);
        if (walkable_cell_editing_enabled_ &&
            (mouse_event->button() == Qt::LeftButton ||
             mouse_event->button() == Qt::RightButton)) {
            const auto point = viewer_->screenToPathPlane(
                mouse_event->pos(), walkable_floor_z_);
            if (point) {
                has_last_painted_walkable_cell_ = false;
                erasing_walkable_cells_ = mouse_event->button() == Qt::RightButton;
                extendWalkableCellStroke(walkableCellAt(*point), erasing_walkable_cells_);
            }
            painting_walkable_cells_ = true;
            mouse_event->accept(); return true;
        }
        if (path_editing_enabled_) {
            const int hit = viewer_->hitTestManualPathPoint(mouse_event->pos());
            if (mouse_event->button() == Qt::RightButton) {
                if (hit >= 0) {
                    manual_path_.erase(manual_path_.begin() + hit);
                    selected_manual_path_point_ = -1;
                    updateManualPathDisplay();
                }
                mouse_event->accept();
                return true;
            }
            if (mouse_event->button() == Qt::LeftButton) {
                if (hit >= 0) {
                    selected_manual_path_point_ = hit;
                    dragging_manual_path_point_ = true;
                } else {
                    const auto point = viewer_->screenToPathPlane(
                        mouse_event->pos(), manual_path_height_);
                    if (point) {
                        manual_path_.push_back(*point);
                        selected_manual_path_point_ =
                            static_cast<int>(manual_path_.size()) - 1;
                    }
                }
                updateManualPathDisplay();
                mouse_event->accept();
                return true;
            }
        }
        if (free_zone_editing_enabled_) {
            const int hit = viewer_->hitTestFreeZoneVertex(mouse_event->pos());
            if (mouse_event->button() == Qt::RightButton) {
                if (hit >= 0) {
                    if (static_cast<std::size_t>(hit) < free_zone_seed_vertex_count_) {
                        emit freeZoneStatusChanged(
                            "Shared-edge vertices are locked while drawing the joined polygon.",
                            QString());
                        mouse_event->accept();
                        return true;
                    }
                    free_zone_vertices_.erase(free_zone_vertices_.begin() + hit);
                    selected_free_zone_vertex_ = -1;
                    if (free_zone_vertices_.size() < 3) free_zone_closed_ = false;
                    updateFreeZoneDisplay();
                }
                mouse_event->accept();
                return true;
            }
            if (mouse_event->button() == Qt::LeftButton) {
                if (hit >= 0) {
                    selected_free_zone_vertex_ = hit;
                    dragging_free_zone_vertex_ =
                        static_cast<std::size_t>(hit) >= free_zone_seed_vertex_count_;
                } else if (!free_zone_closed_) {
                    const auto point = viewer_->screenToPathPlane(
                        mouse_event->pos(), free_zone_height_);
                    if (point) {
                        free_zone_vertices_.push_back(*point);
                        selected_free_zone_vertex_ =
                            static_cast<int>(free_zone_vertices_.size()) - 1;
                    }
                }
                updateFreeZoneDisplay();
                mouse_event->accept();
                return true;
            }
        }
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
            pressed_semantic_object_id_ = viewer_->hitTestSemanticObject(mouse_event->pos());
            if (semantic_verification_mode_) {
                // Semantic inspection is a camera free-look, never a model
                // orbit.  The slower response makes precise box inspection
                // practical even when the object is close to the camera.
                constrained_z_up_navigation_ = true;
                drag_mode_ = DragMode::SemanticLook;
            } else if (mouse_event->modifiers() & Qt::ShiftModifier) {
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
        if (walkable_cell_editing_enabled_ && painting_walkable_cells_ &&
            (mouse_event->buttons() & (Qt::LeftButton | Qt::RightButton))) {
            const auto point = viewer_->screenToPathPlane(
                mouse_event->pos(), walkable_floor_z_);
            if (point) {
                extendWalkableCellStroke(
                    walkableCellAt(*point), erasing_walkable_cells_);
            }
            mouse_event->accept(); return true;
        }
        if (path_editing_enabled_ && dragging_manual_path_point_ &&
            (mouse_event->buttons() & Qt::LeftButton)) {
            const auto point = viewer_->screenToPathPlane(
                mouse_event->pos(), manual_path_height_);
            if (point && selected_manual_path_point_ >= 0 &&
                selected_manual_path_point_ <
                    static_cast<int>(manual_path_.size())) {
                manual_path_[selected_manual_path_point_] = *point;
                updateManualPathDisplay();
            }
            mouse_event->accept();
            return true;
        }
        if (free_zone_editing_enabled_ && dragging_free_zone_vertex_ &&
            (mouse_event->buttons() & Qt::LeftButton)) {
            const auto point = viewer_->screenToPathPlane(
                mouse_event->pos(), free_zone_height_);
            if (point && selected_free_zone_vertex_ >= 0 &&
                selected_free_zone_vertex_ < static_cast<int>(free_zone_vertices_.size())) {
                free_zone_vertices_[selected_free_zone_vertex_] = *point;
                updateFreeZoneDisplay();
            }
            mouse_event->accept();
            return true;
        }
        if ((drag_mode_ == DragMode::RotateFree ||
             drag_mode_ == DragMode::RotatePending ||
             drag_mode_ == DragMode::RotateYawOnly ||
             drag_mode_ == DragMode::RotatePitchOnly ||
             drag_mode_ == DragMode::SemanticLook) &&
            (mouse_event->buttons() & Qt::LeftButton)) {
            if (pressed_semantic_object_id_ >= 0) {
                if ((mouse_event->pos() - drag_start_position_).manhattanLength() <
                    QApplication::startDragDistance()) {
                    mouse_event->accept();
                    return true;
                }
                pressed_semantic_object_id_ = -1;
            }
            QPoint delta = mouse_event->pos() - last_mouse_position_;
            if (viewer_->isOverviewMode() && !delta.isNull()) viewer_->setOverviewMode(false);
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
            if (drag_mode_ == DragMode::SemanticLook) {
                yaw_degrees_ = std::fmod(
                    yaw_degrees_ + delta.x() * kSemanticRotationDegreesPerPixel,
                    360.0f);
                pitch_degrees_ = std::clamp(
                    pitch_degrees_ + delta.y() * kSemanticRotationDegreesPerPixel,
                    -85.0f, 85.0f);
            } else if (drag_mode_ == DragMode::RotateFree ||
                drag_mode_ == DragMode::RotateYawOnly) {
                const float sensitivity = constrained_z_up_navigation_
                    ? kConstrainedRotationDegreesPerPixel
                    : kRotationDegreesPerPixel;
                yaw_degrees_ = std::fmod(
                    yaw_degrees_ + delta.x() * sensitivity,
                    360.0f);
            }
            if (drag_mode_ != DragMode::SemanticLook &&
                (drag_mode_ == DragMode::RotateFree ||
                 drag_mode_ == DragMode::RotatePitchOnly)) {
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
        if (walkable_cell_editing_enabled_) {
            painting_walkable_cells_ = false;
            has_last_painted_walkable_cell_ = false;
            mouse_event->accept(); return true;
        }
        if (path_editing_enabled_ && mouse_event->button() == Qt::LeftButton) {
            dragging_manual_path_point_ = false;
            mouse_event->accept();
            return true;
        }
        if (free_zone_editing_enabled_ && mouse_event->button() == Qt::LeftButton) {
            dragging_free_zone_vertex_ = false;
            mouse_event->accept();
            return true;
        }
        if ((mouse_event->button() == Qt::LeftButton &&
             (drag_mode_ == DragMode::RotateFree ||
              drag_mode_ == DragMode::RotatePending ||
              drag_mode_ == DragMode::RotateYawOnly ||
              drag_mode_ == DragMode::RotatePitchOnly ||
              drag_mode_ == DragMode::SemanticLook)) ||
            (mouse_event->button() == Qt::RightButton &&
             drag_mode_ == DragMode::Pan)) {
            if (mouse_event->button() == Qt::LeftButton &&
                pressed_semantic_object_id_ >= 0 &&
                (mouse_event->pos() - drag_start_position_).manhattanLength() <
                    QApplication::startDragDistance() &&
                viewer_->hitTestSemanticObject(mouse_event->pos()) == pressed_semantic_object_id_) {
                selectSemanticObject(pressed_semantic_object_id_, false);
            }
            pressed_semantic_object_id_ = -1;
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
        const float raw_wheel_steps = angle_delta != 0
            ? static_cast<float>(angle_delta) / 120.0f
            : static_cast<float>(pixel_delta) / 60.0f;
        const float wheel_steps = std::clamp(
            raw_wheel_steps, -kMaximumWheelStepsPerEvent,
            kMaximumWheelStepsPerEvent);
        if (std::abs(wheel_steps) < 1.0e-4f) break;
        if (path_editing_enabled_ || free_zone_editing_enabled_ ||
            walkable_cell_editing_enabled_ || viewer_->isOverviewMode()) {
            viewer_->zoomPathEditView(wheel_steps);
            wheel_event->accept();
            return true;
        }
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
