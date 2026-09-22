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

// Orthographic path, free-zone, and walkable-cell editing.
void ViewerController::resetView()
{
    yaw_degrees_ = 0.0f;
    pitch_degrees_ = 0.0f;
    translation_ = {};
    camera_distance_ = constrained_z_up_navigation_
        ? kNavigateInitialCameraDistance : kInitialCameraDistance;
    if (constrained_z_up_navigation_ &&
        (!navigation_path_world_.empty() || has_free_zone_default_start_)) {
        z_up_camera_position_ = viewer_->sceneWorldToAlignedTransform().map(
            !navigation_path_world_.empty() ? navigation_path_world_.front()
                                            : free_zone_default_start_world_);
        z_up_camera_position_.setZ(playbackCameraHeight(z_up_camera_position_.z()));
    } else {
        z_up_camera_position_ = QVector3D(0.0f, -camera_distance_, 0.0f);
    }
    applyTransform();
}

void ViewerController::setConstrainedZUpNavigation(bool enabled)
{
    viewer_->setOverviewMode(false);
    if (enabled) {
        ensureSceneAlignment();
    }
    constrained_z_up_navigation_ = enabled;
    yaw_degrees_ = 0.0f;
    pitch_degrees_ = 0.0f;
    if (enabled) {
        translation_ = {};
        camera_distance_ = kNavigateInitialCameraDistance;
        const QMatrix4x4 world_to_aligned =
            viewer_->sceneWorldToAlignedTransform();
        if (robot_pose_initialized_ && robot_path_player_->hasPath()) {
            z_up_camera_position_ = world_to_aligned.map(
                robot_path_player_->currentWorldPosition());
        } else if (!navigation_path_world_.empty()) {
            z_up_camera_position_ = world_to_aligned.map(
                navigation_path_world_.front());
        } else if (has_free_zone_default_start_) {
            z_up_camera_position_ = world_to_aligned.map(
                free_zone_default_start_world_);
        } else {
            z_up_camera_position_ = QVector3D(
                0.0f, -kNavigateInitialCameraDistance, 0.0f);
        }
        z_up_camera_position_.setZ(playbackCameraHeight(z_up_camera_position_.z()));
    } else {
        // Explore mode returns to a complete-scene orbit view.
        translation_ = {};
        camera_distance_ = kInitialCameraDistance;
    }
    drag_mode_ = DragMode::None;
    viewer_->setZUpGizmo(enabled);
    updateTrajectoryForScene();
    robot_path_player_->setWorldToAlignedTransform(
        viewer_->sceneWorldToAlignedTransform());
    refreshLoadedRobotPathDisplay();
    applyTransform();
}

void ViewerController::setPathEditingEnabled(bool enabled)
{
    if (enabled) {
        if (robot_path_player_->isPlaying()) robot_path_player_->togglePlayPause();
        if (ensureSceneAlignment()) {
            emit floorAlignmentStatusChanged(
                "Z-up aligned for orthographic path editing.");
        }
        constrained_z_up_navigation_ = false;
        path_editing_enabled_ = true;
        drag_mode_ = DragMode::None;
        selected_manual_path_point_ = -1;
        if (!trajectory_processing_.smoothedTrajectory().empty())
            manual_path_height_ = static_cast<float>(
                trajectory_processing_.trajectoryHeight());
        viewer_->setZUpGizmo(true);
        viewer_->setPathEditMode(true);
        updateTrajectoryForScene();
        robot_path_player_->setWorldToAlignedTransform(
            viewer_->sceneWorldToAlignedTransform());
        refreshLoadedRobotPathDisplay();
    } else {
        path_editing_enabled_ = false;
        dragging_manual_path_point_ = false;
        selected_manual_path_point_ = -1;
        viewer_->setPathEditMode(free_zone_editing_enabled_ || walkable_cell_editing_enabled_);
    }
    updateManualPathDisplay();
}

void ViewerController::setFreeZoneEditingEnabled(bool enabled)
{
    free_zone_editing_enabled_ = enabled;
    dragging_free_zone_vertex_ = false;
    selected_free_zone_vertex_ = -1;
    if (enabled) {
        if (robot_path_player_->isPlaying()) robot_path_player_->togglePlayPause();
        if (ensureSceneAlignment())
            emit floorAlignmentStatusChanged("Z-up aligned for free-zone editing.");
        constrained_z_up_navigation_ = false;
        path_editing_enabled_ = false;
        drag_mode_ = DragMode::None;
        free_zone_height_ = viewer_->robotEyeHeightAligned(
            0.02f, manual_path_height_);
        // A free zone represents floor area. Keep every component on one
        // detected floor plane; inherited path/camera heights must not leak
        // into polygon visualization.
        for (QVector3D& point : free_zone_vertices_)
            point.setZ(free_zone_height_);
        for (auto& polygon : completed_free_zone_polygons_)
            for (QVector3D& point : polygon) point.setZ(free_zone_height_);
        viewer_->setZUpGizmo(true);
        viewer_->setPathEditMode(true);
    } else {
        viewer_->setPathEditMode(path_editing_enabled_ || walkable_cell_editing_enabled_);
    }
    updateFreeZoneDisplay();
}

void ViewerController::updateFreeZoneDisplay()
{
    viewer_->setFreeZone(
        completed_free_zone_polygons_, free_zone_vertices_, free_zone_closed_,
        selected_free_zone_vertex_);
    const std::size_t polygon_count = completed_free_zone_polygons_.size() +
        (free_zone_closed_ ? 1U : 0U);
    emit freeZoneStatusChanged(
        QString("%1 merged polygon(s) | %2 active vertices | %3")
            .arg(polygon_count)
            .arg(free_zone_vertices_.size())
            .arg(free_zone_closed_ ? "closed walkable zone" : "open boundary"),
        QString());
}

void ViewerController::closeFreeZone()
{
    if (free_zone_vertices_.size() < 3) {
        emit freeZoneStatusChanged("A free zone needs at least three vertices.", QString());
        return;
    }
    if (free_zone_seed_vertex_count_ > 0 &&
        free_zone_vertices_.size() <= free_zone_seed_vertex_count_) {
        emit freeZoneStatusChanged(
            "Add at least one new outer-boundary vertex before closing.", QString());
        return;
    }
    bool overlaps_existing = completed_free_zone_polygons_.empty();
    if (!completed_free_zone_polygons_.empty()) {
        QPolygonF active_polygon;
        for (const QVector3D& point : free_zone_vertices_)
            active_polygon << QPointF(point.x(), point.y());
        QPainterPath active_path;
        active_path.addPolygon(active_polygon);
        active_path.closeSubpath();
        for (const auto& vertices : completed_free_zone_polygons_) {
            QPolygonF completed_polygon;
            for (const QVector3D& point : vertices)
                completed_polygon << QPointF(point.x(), point.y());
            QPainterPath completed_path;
            completed_path.addPolygon(completed_polygon);
            completed_path.closeSubpath();
            if (!active_path.intersected(completed_path).isEmpty()) {
                overlaps_existing = true;
                break;
            }
        }
    }
    free_zone_closed_ = true;
    free_zone_seed_vertex_count_ = 0;
    selected_free_zone_vertex_ = -1;
    updateFreeZoneDisplay();
    if (!overlaps_existing)
        emit freeZoneStatusChanged(
            "Polygon saved as another component, but it does not overlap the existing free zone; navigation may remain disconnected.",
            QString());
}

void ViewerController::reopenFreeZone()
{
    free_zone_closed_ = false;
    updateFreeZoneDisplay();
}

void ViewerController::startJoinedFreeZone(
    int shared_edge_count, bool reverse_direction)
{
    if (!free_zone_closed_ || free_zone_vertices_.size() < 3) {
        emit freeZoneStatusChanged(
            "Close the current polygon before starting a joined polygon.", QString());
        return;
    }
    if (selected_free_zone_vertex_ < 0 ||
        selected_free_zone_vertex_ >= static_cast<int>(free_zone_vertices_.size())) {
        emit freeZoneStatusChanged(
            "Select a vertex on the closed polygon first.", QString());
        return;
    }
    const int vertex_count = static_cast<int>(free_zone_vertices_.size());
    shared_edge_count = std::clamp(shared_edge_count, 1, vertex_count - 1);
    std::vector<QVector3D> shared_chain;
    shared_chain.reserve(static_cast<std::size_t>(shared_edge_count + 1));
    int index = selected_free_zone_vertex_;
    for (int edge = 0; edge <= shared_edge_count; ++edge) {
        shared_chain.push_back(free_zone_vertices_[static_cast<std::size_t>(index)]);
        index = reverse_direction
            ? (index - 1 + vertex_count) % vertex_count
            : (index + 1) % vertex_count;
    }
    completed_free_zone_polygons_.push_back(free_zone_vertices_);
    free_zone_vertices_ = std::move(shared_chain);
    free_zone_closed_ = false;
    free_zone_seed_vertex_count_ = free_zone_vertices_.size();
    selected_free_zone_vertex_ = static_cast<int>(free_zone_vertices_.size()) - 1;
    updateFreeZoneDisplay();
}

void ViewerController::startAdditiveFreeZone()
{
    if (!free_zone_closed_ || free_zone_vertices_.size() < 3) {
        emit freeZoneStatusChanged(
            "Close the current polygon before adding another region.", QString());
        return;
    }
    completed_free_zone_polygons_.push_back(free_zone_vertices_);
    free_zone_vertices_.clear();
    free_zone_closed_ = false;
    free_zone_seed_vertex_count_ = 0;
    selected_free_zone_vertex_ = -1;
    updateFreeZoneDisplay();
    emit freeZoneStatusChanged(
        "Draw the next polygon anywhere; overlap the existing green area for a safe union.",
        QString());
}

void ViewerController::clearFreeZone()
{
    free_zone_vertices_.clear();
    completed_free_zone_polygons_.clear();
    free_zone_closed_ = false;
    free_zone_seed_vertex_count_ = 0;
    has_free_zone_default_start_ = false;
    selected_free_zone_vertex_ = -1;
    updateFreeZoneDisplay();
}

void ViewerController::saveFreeZone(QWidget* dialog_parent)
{
    if (!free_zone_closed_ || free_zone_vertices_.size() < 3) {
        emit freeZoneStatusChanged("Close a polygon before saving it.", QString());
        return;
    }
    if (!ensureSceneAlignment()) {
        emit freeZoneStatusChanged("Cannot save without a stable scene alignment.", QString());
        return;
    }
    QFileDialog dialog(dialog_parent, "Save walkable free zone",
        QStringLiteral(PROJECT_ROOT_DIR) + "/data/free_zone.json");
    dialog.setAcceptMode(QFileDialog::AcceptSave);
    dialog.setFileMode(QFileDialog::AnyFile);
    dialog.setNameFilter("JSON free zones (*.json)");
    dialog.setDefaultSuffix("json");
    dialog.setOption(QFileDialog::DontUseNativeDialog, true);
    if (dialog.exec() != QDialog::Accepted || dialog.selectedFiles().isEmpty()) return;
    const QString path = dialog.selectedFiles().constFirst();
    bool invertible = false;
    const QMatrix4x4 aligned_to_world =
        viewer_->sceneWorldToAlignedTransform().inverted(&invertible);
    if (!invertible) {
        emit freeZoneStatusChanged("Cannot save: scene transform is not invertible.", path);
        return;
    }
    QJsonArray polygons;
    std::vector<std::vector<QVector3D>> all_polygons = completed_free_zone_polygons_;
    all_polygons.push_back(free_zone_vertices_);
    for (std::size_t polygon_index = 0; polygon_index < all_polygons.size(); ++polygon_index) {
        QJsonArray vertices;
        for (const QVector3D& aligned : all_polygons[polygon_index]) {
            const QVector3D world = aligned_to_world.map(aligned);
            QJsonObject vertex;
            vertex["x"] = world.x(); vertex["y"] = world.y(); vertex["z"] = world.z();
            vertices.append(vertex);
        }
        QJsonObject polygon;
        polygon["id"] = static_cast<int>(polygon_index);
        polygon["vertices"] = vertices;
        polygons.append(polygon);
    }
    QJsonObject root;
    root["version"] = 1;
    root["coordinate_frame"] = "3dgs_world";
    root["type"] = "walkable_polygons";
    root["merge_operation"] = "union";
    root["robot_radius_m"] = 0.30;
    QVector3D default_start_world;
    if (!navigation_path_world_.empty()) {
        default_start_world = navigation_path_world_.front();
    } else if (has_free_zone_default_start_) {
        default_start_world = free_zone_default_start_world_;
    } else {
        QVector3D centroid_aligned;
        for (const QVector3D& point : free_zone_vertices_) centroid_aligned += point;
        centroid_aligned /= static_cast<float>(free_zone_vertices_.size());
        default_start_world = aligned_to_world.map(centroid_aligned);
    }
    QJsonArray default_start;
    default_start.append(default_start_world.x());
    default_start.append(default_start_world.y());
    default_start.append(default_start_world.z());
    root["default_start_world_m"] = default_start;
    root["polygons"] = polygons;
    free_zone_default_start_world_ = default_start_world;
    has_free_zone_default_start_ = true;
    QSaveFile output(path);
    if (!output.open(QIODevice::WriteOnly)) {
        emit freeZoneStatusChanged("Cannot save free zone: " + output.errorString(), path);
        return;
    }
    const QByteArray data = QJsonDocument(root).toJson(QJsonDocument::Indented);
    if (output.write(data) != data.size() || !output.commit()) {
        emit freeZoneStatusChanged("Failed to save free zone: " + output.errorString(), path);
        return;
    }
    emit freeZoneStatusChanged(
        QString("Saved union of %1 walkable polygons").arg(all_polygons.size()),
        path);
}

void ViewerController::loadFreeZone(QWidget* dialog_parent)
{
    const QString path = QFileDialog::getOpenFileName(dialog_parent,
        "Load walkable free zone", QStringLiteral(PROJECT_ROOT_DIR) + "/data",
        "JSON free zones (*.json);;All files (*)");
    if (path.isEmpty()) return;
    if (!ensureSceneAlignment()) {
        emit freeZoneStatusChanged("Cannot load without a stable scene alignment.", path);
        return;
    }
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly)) {
        emit freeZoneStatusChanged("Cannot open free zone: " + file.errorString(), path);
        return;
    }
    QJsonParseError error;
    const QJsonDocument document = QJsonDocument::fromJson(file.readAll(), &error);
    const QJsonObject root = document.object();
    const QJsonArray polygons = root.value("polygons").toArray();
    if (error.error != QJsonParseError::NoError || !document.isObject() ||
        root.value("coordinate_frame").toString() != "3dgs_world" || polygons.isEmpty()) {
        emit freeZoneStatusChanged("Invalid 3dgs_world free-zone JSON.", path);
        return;
    }
    std::vector<std::vector<QVector3D>> loaded_polygons;
    for (const QJsonValue& polygon_value : polygons) {
        std::vector<QVector3D> loaded;
        for (const QJsonValue& value :
             polygon_value.toObject().value("vertices").toArray()) {
            const QJsonObject vertex = value.toObject();
            loaded.push_back(viewer_->sceneWorldToAlignedTransform().map(QVector3D(
                vertex.value("x").toDouble(), vertex.value("y").toDouble(),
                vertex.value("z").toDouble())));
        }
        if (loaded.size() >= 3) loaded_polygons.push_back(std::move(loaded));
    }
    if (loaded_polygons.empty()) {
        emit freeZoneStatusChanged("No valid free-zone polygons were found.", path);
        return;
    }
    free_zone_vertices_ = std::move(loaded_polygons.back());
    loaded_polygons.pop_back();
    completed_free_zone_polygons_ = std::move(loaded_polygons);
    const float loaded_fallback_height = free_zone_vertices_.front().z();
    free_zone_height_ = viewer_->robotEyeHeightAligned(
        0.02f, loaded_fallback_height);
    for (QVector3D& point : free_zone_vertices_) point.setZ(free_zone_height_);
    for (auto& polygon : completed_free_zone_polygons_)
        for (QVector3D& point : polygon) point.setZ(free_zone_height_);
    free_zone_closed_ = true;
    free_zone_seed_vertex_count_ = 0;
    has_free_zone_default_start_ = false;
    const QJsonArray saved_start = root.value("default_start_world_m").toArray();
    if (saved_start.size() == 3) {
        free_zone_default_start_world_ = QVector3D(
            saved_start[0].toDouble(), saved_start[1].toDouble(),
            saved_start[2].toDouble());
        has_free_zone_default_start_ = true;
    } else {
        // Migrate an older free-zone file from the conventional saved path
        // beside it, without requiring the user to load that path separately.
        QFile legacy_path(QFileInfo(path).absolutePath() + "/manual_path.json");
        if (legacy_path.open(QIODevice::ReadOnly)) {
            const QJsonArray points = QJsonDocument::fromJson(
                legacy_path.readAll()).object().value("points").toArray();
            if (!points.isEmpty()) {
                const QJsonObject point = points.first().toObject();
                free_zone_default_start_world_ = QVector3D(
                    point.value("x").toDouble(), point.value("y").toDouble(),
                    point.value("z").toDouble());
                has_free_zone_default_start_ = true;
            }
        }
        if (!has_free_zone_default_start_) {
            QVector3D centroid_aligned;
            for (const QVector3D& point : free_zone_vertices_) centroid_aligned += point;
            centroid_aligned /= static_cast<float>(free_zone_vertices_.size());
            bool transform_ok = false;
            const QMatrix4x4 aligned_to_world =
                viewer_->sceneWorldToAlignedTransform().inverted(&transform_ok);
            if (transform_ok) {
                free_zone_default_start_world_ = aligned_to_world.map(centroid_aligned);
                has_free_zone_default_start_ = true;
            }
        }
    }
    selected_free_zone_vertex_ = -1;
    viewer_->setFreeZone(
        completed_free_zone_polygons_, free_zone_vertices_, true, -1);
    emit freeZoneStatusChanged(
        QString("Loaded union of %1 free-zone polygons")
            .arg(completed_free_zone_polygons_.size() + 1), path);
}

void ViewerController::updateManualPathDisplay()
{
    viewer_->setManualPath(manual_path_, selected_manual_path_point_);
    emit manualPathStatusChanged(
        QString("%1 manual path points (unsaved)")
            .arg(static_cast<qulonglong>(manual_path_.size())),
        QString());
}

void ViewerController::updateWalkableCellDisplay()
{
    viewer_->setWalkableCells(
        walkable_cells_, walkable_cell_size_, walkable_floor_z_,
        walkable_grid_angle_radians_);
    const float units_per_meter = std::max(
        viewer_->sceneWorldToAlignedTransform().mapVector(
            QVector3D(1, 0, 0)).length(), 1.0e-5f);
    emit freeZoneStatusChanged(
        QString("%1 walkable cells; cell size %2 m")
            .arg(walkable_cells_.size())
            .arg(walkable_cell_size_ / units_per_meter, 0, 'f', 2), QString());
}

std::pair<int, int> ViewerController::walkableCellAt(
    const QVector3D& point) const
{
    const float cs=std::cos(walkable_grid_angle_radians_);
    const float sn=std::sin(walkable_grid_angle_radians_);
    return {static_cast<int>(std::floor((cs*point.x()+sn*point.y()) /
                                        walkable_cell_size_)),
            static_cast<int>(std::floor((-sn*point.x()+cs*point.y()) /
                                        walkable_cell_size_))};
}

void ViewerController::extendWalkableCellStroke(
    const std::pair<int, int>& destination, bool erase)
{
    const auto start = has_last_painted_walkable_cell_
        ? last_painted_walkable_cell_ : destination;
    const int dx=destination.first-start.first;
    const int dy=destination.second-start.second;
    const int steps=std::max(std::abs(dx),std::abs(dy));
    bool changed=false;
    for(int step=0;step<=steps;++step){
        const float t=steps ? static_cast<float>(step)/steps : 0.0f;
        const std::pair<int,int> cell{
            static_cast<int>(std::lround(start.first+t*dx)),
            static_cast<int>(std::lround(start.second+t*dy))};
        if(erase){changed=walkable_cells_.erase(cell)>0||changed;continue;}
        bool connected=walkable_cells_.empty();
        for(int x=-1;x<=1&&!connected;++x)
            for(int y=-1;y<=1&&!connected;++y)
                if((x||y)&&walkable_cells_.count({cell.first+x,cell.second+y}))
                    connected=true;
        if(connected) changed=walkable_cells_.insert(cell).second||changed;
    }
    last_painted_walkable_cell_=destination;
    has_last_painted_walkable_cell_=true;
    if(changed) updateWalkableCellDisplay();
}

void ViewerController::initializeWalkableCellsFromPath()
{
    if (navigation_path_world_.size() < 2) {
        emit freeZoneStatusChanged("Load a robot path before creating cells.", QString());
        return;
    }
    const QMatrix4x4 transform = viewer_->sceneWorldToAlignedTransform();
    const float units_per_meter = std::max(
        transform.mapVector(QVector3D(1, 0, 0)).length(), 1.0e-5f);
    walkable_cell_size_ = 0.12f * units_per_meter;
    const QVector3D first = transform.map(navigation_path_world_.front());
    const QVector3D second = transform.map(navigation_path_world_[1]);
    walkable_grid_angle_radians_ = std::atan2(
        second.y() - first.y(), second.x() - first.x());
    // The source path contains camera/robot poses above the floor. Using its Z
    // for the cell plane is invisible in the orthographic editor, but produces
    // a perspective parallax offset during navigation. Keep cells just above
    // the detected floor instead.
    walkable_floor_z_ = viewer_->robotEyeHeightAligned(0.02f, first.z());
    walkable_cells_.clear();
    for (std::size_t i = 0; i + 1 < navigation_path_world_.size(); ++i) {
        const QVector3D a = transform.map(navigation_path_world_[i]);
        const QVector3D b = transform.map(navigation_path_world_[i + 1]);
        const float length = QVector2D(b.x() - a.x(), b.y() - a.y()).length();
        const int samples = std::max(1, static_cast<int>(std::ceil(
            length / (walkable_cell_size_ * 0.35f))));
        for (int s = 0; s <= samples; ++s) {
            const float t = static_cast<float>(s) / samples;
            const float x = a.x() * (1-t) + b.x() * t;
            const float y = a.y() * (1-t) + b.y() * t;
            const float cs=std::cos(walkable_grid_angle_radians_);
            const float sn=std::sin(walkable_grid_angle_radians_);
            const int gx = static_cast<int>(std::floor((cs*x+sn*y) / walkable_cell_size_));
            const int gy = static_cast<int>(std::floor((-sn*x+cs*y) / walkable_cell_size_));
            // A four-cell-wide connected backbone.
            for (int dx = -1; dx <= 2; ++dx)
                for (int dy = -1; dy <= 2; ++dy)
                    walkable_cells_.insert({gx + dx, gy + dy});
        }
    }
    updateWalkableCellDisplay();
}

void ViewerController::setWalkableCellEditingEnabled(bool enabled)
{
    walkable_cell_editing_enabled_ = enabled;
    painting_walkable_cells_ = false;
    if (enabled) {
        if (robot_path_player_->isPlaying()) robot_path_player_->togglePlayPause();
        setPathEditingEnabled(false);
        setFreeZoneEditingEnabled(false);
        constrained_z_up_navigation_ = false;
        viewer_->setZUpGizmo(true);
        viewer_->setPathEditMode(true);
        if (walkable_cells_.empty()) initializeWalkableCellsFromPath();
    } else {
        viewer_->setPathEditMode(path_editing_enabled_ || free_zone_editing_enabled_);
    }
}

void ViewerController::clearWalkableCells()
{
    walkable_cells_.clear();
    updateWalkableCellDisplay();
}

void ViewerController::setWalkableCellsVisible(bool visible)
{
    viewer_->setWalkableCellsVisible(visible);
}

void ViewerController::saveWalkableCells(QWidget* parent)
{
    if (walkable_cells_.empty()) return;
    if (!ensureSceneAlignment()) {
        emit freeZoneStatusChanged("Cannot save cells without scene alignment.", QString());
        return;
    }
    bool invertible=false;
    const QMatrix4x4 world_to_aligned=viewer_->sceneWorldToAlignedTransform();
    const QMatrix4x4 aligned_to_world=world_to_aligned.inverted(&invertible);
    if(!invertible) return;
    QString path = QFileDialog::getSaveFileName(parent, "Save walkable cells",
        QStringLiteral(PROJECT_ROOT_DIR) + "/data/walkable_cells.json", "JSON (*.json)");
    if (path.isEmpty()) return;
    QJsonObject root; root["version"] = 2;
    root["coordinate_frame"] = "scene_z_up_aligned";
    root["cell_size_aligned"] = walkable_cell_size_;
    root["cell_size_m"] = 0.12;
    root["floor_z_aligned"] = walkable_floor_z_;
    root["grid_angle_radians"] = walkable_grid_angle_radians_;
    QJsonArray cells;
    QJsonArray world_centers;
    const float cs=std::cos(walkable_grid_angle_radians_);
    const float sn=std::sin(walkable_grid_angle_radians_);
    for (const auto& cell : walkable_cells_) {
        QJsonArray value; value.append(cell.first); value.append(cell.second);
        cells.append(value);
        const float x=(cell.first+0.5f)*walkable_cell_size_;
        const float y=(cell.second+0.5f)*walkable_cell_size_;
        const QVector3D world=aligned_to_world.map(
            QVector3D(cs*x-sn*y,sn*x+cs*y,walkable_floor_z_));
        QJsonArray center; center.append(world.x()); center.append(world.y());
        center.append(world.z()); world_centers.append(center);
    }
    root["walkable_cells"] = cells;
    root["cell_centers_world_m"] = world_centers;
    QVector3D world_axis=aligned_to_world.mapVector(QVector3D(cs,sn,0.0f));
    world_axis.normalize();
    QJsonArray axis; axis.append(world_axis.x()); axis.append(world_axis.y());
    axis.append(world_axis.z()); root["grid_x_axis_world"] = axis;
    QSaveFile file(path);
    if (!file.open(QIODevice::WriteOnly) ||
        file.write(QJsonDocument(root).toJson(QJsonDocument::Indented)) < 0 ||
        !file.commit()) {
        emit freeZoneStatusChanged("Could not save walkable cells.", QString()); return;
    }
    emit freeZoneStatusChanged(QString("Saved %1 walkable cells").arg(cells.size()), path);
}

void ViewerController::loadWalkableCells(QWidget* parent)
{
    const QString path = QFileDialog::getOpenFileName(parent, "Load walkable cells",
        QStringLiteral(PROJECT_ROOT_DIR) + "/data", "JSON (*.json)");
    if (path.isEmpty()) return;
    loadWalkableCellsFile(path);
}

bool ViewerController::loadWalkableCellsFile(const QString& path)
{
    if (!ensureSceneAlignment()) {
        emit freeZoneStatusChanged("Cannot load cells without scene alignment.", path);
        return false;
    }
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly)) {
        emit freeZoneStatusChanged(
            "Cannot open walkable cells: " + file.errorString(), path);
        return false;
    }
    const QJsonDocument doc = QJsonDocument::fromJson(file.readAll());
    const QJsonObject root = doc.object();
    if (root.isEmpty()) {
        emit freeZoneStatusChanged("Invalid walkable-cell JSON.", path);
        return false;
    }
    if (root.value("coordinate_frame").toString() != "scene_z_up_aligned") {
        emit freeZoneStatusChanged("Walkable-cell file is not Z-up aligned.", path);
        return false;
    }
    std::set<std::pair<int,int>> loaded;
    const QJsonArray world_centers=root.value("cell_centers_world_m").toArray();
    const QJsonArray world_axis=root.value("grid_x_axis_world").toArray();
    const bool has_world_geometry=!world_centers.isEmpty() && world_axis.size()==3;
    if(has_world_geometry){
        const QMatrix4x4 transform=viewer_->sceneWorldToAlignedTransform();
        const float units_per_meter=std::max(
            transform.mapVector(QVector3D(1,0,0)).length(),1.0e-5f);
        walkable_cell_size_=root.value("cell_size_m").toDouble(0.12)*units_per_meter;
        QVector3D axis=transform.mapVector(QVector3D(
            world_axis[0].toDouble(),world_axis[1].toDouble(),world_axis[2].toDouble()));
        axis.setZ(0.0f);
        walkable_grid_angle_radians_=std::atan2(axis.y(),axis.x());
        walkable_floor_z_=viewer_->robotEyeHeightAligned(0.02f,0.0f);
        for(const QJsonValue& value:world_centers){
            const QJsonArray c=value.toArray(); if(c.size()!=3) continue;
            const QVector3D aligned=transform.map(QVector3D(
                c[0].toDouble(),c[1].toDouble(),c[2].toDouble()));
            loaded.insert(walkableCellAt(aligned));
        }
    } else {
    for (const QJsonValue& value : root.value("walkable_cells").toArray()) {
        const QJsonArray c = value.toArray();
        if (c.size() == 2) loaded.insert({c[0].toInt(), c[1].toInt()});
    }
    }
    if (loaded.empty()) {
        emit freeZoneStatusChanged("Walkable-cell file contains no cells.", path);
        return false;
    }
    walkable_cells_ = std::move(loaded);
    if(!has_world_geometry) {
    walkable_cell_size_ = root.value("cell_size_aligned").toDouble(0.12);
    const float saved_floor_z =
        root.value("floor_z_aligned").toDouble();
    // Version-1 files may contain the old camera-height plane. Re-anchor them
    // to the current scene floor when floor estimation is available.
    walkable_floor_z_ =
        viewer_->robotEyeHeightAligned(0.02f, saved_floor_z);
    walkable_grid_angle_radians_ = root.value("grid_angle_radians").toDouble();
    }
    updateWalkableCellDisplay();
    emit freeZoneStatusChanged(
        QString("Loaded %1 walkable cells").arg(walkable_cells_.size()), path);
    return true;
}
