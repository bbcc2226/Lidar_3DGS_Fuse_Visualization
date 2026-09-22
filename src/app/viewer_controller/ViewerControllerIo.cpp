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

// Point-cloud, trajectory, robot-path, and LiDAR file operations.
void ViewerController::loadRobotPath(QWidget* dialog_parent)
{
    QFileDialog dialog(
        dialog_parent, "Load saved robot path",
        QStringLiteral(PROJECT_ROOT_DIR) + "/data");
    dialog.setAcceptMode(QFileDialog::AcceptOpen);
    dialog.setFileMode(QFileDialog::ExistingFile);
    dialog.setNameFilters({"JSON path files (*.json)", "All files (*)"});
    dialog.setOption(QFileDialog::DontUseNativeDialog, true);
    if (dialog.exec() != QDialog::Accepted) return;
    const QStringList files = dialog.selectedFiles();
    if (files.isEmpty()) return;
    loadRobotPathFile(files.constFirst());
}

bool ViewerController::loadRobotPathFile(const QString& path)
{
    viewer_->clearNavigationPlan();
    has_navigation_target_ = false;
    arrival_view_animation_->stop();
    viewer_->setVerticalFieldOfView(45.0f);
    navigation_path_world_.clear();
    if (!robot_path_player_->loadPath(path)) {
        emit robotPlaybackStatusChanged(
            robot_path_player_->lastError(), path);
        return false;
    }
    recorded_height_offset_meters_ = 0.0f;
    setUseOriginalTrajectoryHeight(robot_path_player_->usesRecordedHeight());
    navigation_path_world_ = robot_path_player_->worldPoints();
    navigation_has_started_ = false;
    if (!navigation_path_world_.empty()) {
        free_zone_default_start_world_ = navigation_path_world_.front();
        has_free_zone_default_start_ = true;
    }
    robot_pose_initialized_ = false;
    robot_path_player_->setWorldToAlignedTransform(
        viewer_->sceneWorldToAlignedTransform());
    refreshLoadedRobotPathDisplay();
    emit robotPlaybackStatusChanged("Robot path loaded", path);
    return true;
}

void ViewerController::toggleRobotPlayback()
{
    if (robot_path_player_->hasPath()) {
        if (!robot_path_player_->isPlaying()) restoreNavigationView();
        setPathEditingEnabled(false);
        constrained_z_up_navigation_ = true;
        viewer_->setZUpGizmo(true);
        robot_path_player_->setWorldToAlignedTransform(
            viewer_->sceneWorldToAlignedTransform());
    }
    robot_path_player_->togglePlayPause();
}

void ViewerController::stopRobotPlayback()
{
    robot_path_player_->stop();
}

void ViewerController::clearLoadedPath()
{
    viewer_->clearNavigationPlan();
    has_navigation_target_ = false;
    arrival_view_animation_->stop();
    viewer_->setVerticalFieldOfView(45.0f);
    robot_path_player_->clearPath();
    robot_pose_initialized_ = false;
    navigation_path_world_.clear();
    navigation_has_started_ = false;
    manual_path_.clear();
    selected_manual_path_point_ = -1;
    dragging_manual_path_point_ = false;
    viewer_->setManualPath({}, -1);
    emit manualPathStatusChanged("No manual path loaded", QString());
}

void ViewerController::setRobotPlaybackSpeed(float meters_per_second)
{
    robot_path_player_->setSpeedMetersPerSecond(meters_per_second);
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
    loadPlyFile(selected_files.constFirst());
}

bool ViewerController::loadPlyFile(const QString& path)
{
    emit loadStatusChanged("Loading " + path + "...", path);
    QApplication::setOverrideCursor(Qt::WaitCursor);
    QApplication::processEvents();

    // Release the previous renderer scene before the loader allocates a new
    // large PLY representation, avoiding both scenes overlapping in memory.
    viewer_->setGaussianPoints({});
    scene_alignment_ready_ = false;
    const bool loaded = point_processing_.loadPly(path.toStdString(), [this](int percent) {
        emit loadProgressChanged(percent * 90 / 100, "Reading 3DGS scene...");
    });
    QApplication::restoreOverrideCursor();

    if (!loaded) {
        viewer_->setGaussianPoints({});
        emit loadStatusChanged(
            "Load failed: " + QString::fromStdString(point_processing_.lastError()),
            path);
        return false;
    }

    emit loadProgressChanged(90, "Preparing 3DGS rendering...");
    const GaussianPlyMetadata metadata = point_processing_.metadata();
    const std::size_t splat_count = point_processing_.splatCount();
    gaussian_points_ = point_processing_.points();
    gaussian_has_scale_ = metadata.has_scale;
    gaussian_has_sh_dc_ = metadata.has_sh_dc;
    gaussian_sh_degree_ = metadata.sh_degree;
    lidar_view_enabled_ = false;
    viewer_->setGaussianPoints(
        point_processing_.points(), metadata.has_scale, metadata.has_sh_dc,
        metadata.sh_degree);
    scene_alignment_ready_ = false;
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
    updateTrajectoryForScene();
    if (!constrained_z_up_navigation_ && !path_editing_enabled_ &&
        !free_zone_editing_enabled_ && !walkable_cell_editing_enabled_) showPathOverview();
    emit loadProgressChanged(100, "3DGS scene loaded");
    return true;
}

bool ViewerController::loadLidarPointCloud(const QString& path)
{
    const QString absolute_path = QFileInfo(path).absoluteFilePath();
    if (!lidar_points_.empty() && lidar_loaded_path_ == absolute_path) {
        setLidarViewEnabled(true);
        return true;
    }
    if (lidar_load_in_progress_) return true;
    lidar_loading_path_ = absolute_path;
    lidar_view_enabled_ = true;
    lidar_load_in_progress_ = true;
    emit loadStatusChanged("Loading merged LiDAR " + path + "...", path);
    const QString requested_path = path;
    lidar_watcher_->setFuture(QtConcurrent::run([requested_path]() {
        GaussianSplatProcessing loader;
        if (!loader.loadPly(requested_path.toStdString()))
            return std::shared_ptr<std::vector<GaussianPoint>>();
        return std::make_shared<std::vector<GaussianPoint>>(loader.points());
    }));
    return true;
}

void ViewerController::setLidarViewEnabled(bool enabled)
{
    if (enabled) {
        if (lidar_points_.empty()) return;
        lidar_view_enabled_ = true;
        viewer_->setGaussianPoints(lidar_points_, false, false, 0, false);
    } else {
        if (gaussian_points_.empty()) return;
        lidar_view_enabled_ = false;
        viewer_->setGaussianPoints(gaussian_points_, gaussian_has_scale_,
                                   gaussian_has_sh_dc_, gaussian_sh_degree_, false);
    }
    scene_alignment_ready_ = false;
    updateTrajectoryForScene();
    viewer_->update();
}

bool ViewerController::ensureSceneAlignment()
{
    if (scene_alignment_ready_) return true;
    scene_alignment_ready_ = viewer_->autoAlignSceneUp();
    emit floorAlignmentStatusChanged(scene_alignment_ready_
        ? "Scene alignment fixed to +Z for this PLY."
        : "Automatic Z-up alignment failed; free-zone coordinates may be invalid.");
    return scene_alignment_ready_;
}

void ViewerController::openTrajectoryFile(QWidget* dialog_parent)
{
    QFileDialog dialog(
        dialog_parent, "Open optimized camera poses",
        QStringLiteral(PROJECT_ROOT_DIR) + "/data");
    dialog.setAcceptMode(QFileDialog::AcceptOpen);
    dialog.setFileMode(QFileDialog::ExistingFile);
    dialog.setNameFilters({"Camera pose files (*.txt)", "All files (*)"});
    dialog.setOption(QFileDialog::DontUseNativeDialog, true);
    if (dialog.exec() != QDialog::Accepted) return;

    const QStringList selected_files = dialog.selectedFiles();
    if (selected_files.isEmpty()) return;
    const QString path = selected_files.constFirst();
    if (!trajectory_processing_.loadOptimizedCameraPoses(path.toStdString())) {
        viewer_->setMiniMapTrajectory({}, {});
        emit trajectoryStatusChanged(
            "Trajectory load failed: " +
                QString::fromStdString(trajectory_processing_.lastError()),
            path);
        return;
    }
    updateTrajectoryForScene();
    emit trajectoryStatusChanged(
        QString("Loaded and smoothed %1 camera poses")
            .arg(static_cast<qulonglong>(trajectory_processing_.poses().size())),
        path);
}

void ViewerController::saveManualPath(QWidget* dialog_parent)
{
    if (manual_path_.empty()) {
        emit manualPathStatusChanged(
            "No manual path points to save", QString());
        return;
    }

    QFileDialog dialog(
        dialog_parent, "Save manual robot path",
        QStringLiteral(PROJECT_ROOT_DIR) + "/data/manual_path.json");
    dialog.setAcceptMode(QFileDialog::AcceptSave);
    dialog.setFileMode(QFileDialog::AnyFile);
    dialog.setNameFilters({"JSON path files (*.json)", "All files (*)"});
    dialog.setDefaultSuffix("json");
    dialog.setOption(QFileDialog::DontUseNativeDialog, true);
    if (dialog.exec() != QDialog::Accepted) return;
    const QStringList selected_files = dialog.selectedFiles();
    if (selected_files.isEmpty()) return;
    const QString path = selected_files.constFirst();

    bool invertible = false;
    const QMatrix4x4 world_from_aligned =
        viewer_->sceneWorldToAlignedTransform().inverted(&invertible);
    if (!invertible) {
        emit manualPathStatusChanged(
            "Cannot save path: scene transform is not invertible", path);
        return;
    }

    QJsonArray points;
    for (std::size_t index = 0; index < manual_path_.size(); ++index) {
        const QVector3D world = world_from_aligned.map(manual_path_[index]);
        QJsonObject point;
        point["index"] = static_cast<int>(index);
        point["x"] = static_cast<double>(world.x());
        point["y"] = static_cast<double>(world.y());
        point["z"] = static_cast<double>(world.z());
        points.append(point);
    }

    QJsonObject root;
    root["version"] = 1;
    root["coordinate_frame"] = "3dgs_world";
    root["path_type"] = "ordered_polyline";
    root["closed"] = false;
    root["suggested_interpolation"] = "linear_arc_length";
    root["point_count"] = static_cast<int>(manual_path_.size());
    root["points"] = points;

    QSaveFile output(path);
    if (!output.open(QIODevice::WriteOnly)) {
        emit manualPathStatusChanged(
            "Cannot open path file for writing: " + output.errorString(), path);
        return;
    }
    const QByteArray json = QJsonDocument(root).toJson(QJsonDocument::Indented);
    if (output.write(json) != json.size() || !output.commit()) {
        emit manualPathStatusChanged(
            "Failed to save path: " + output.errorString(), path);
        return;
    }
    emit manualPathStatusChanged(
        QString("Saved %1 ordered path points")
            .arg(static_cast<qulonglong>(manual_path_.size())),
        path);
}

void ViewerController::updateTrajectoryForScene()
{
    if (trajectory_processing_.poses().empty()) return;

    const QMatrix4x4 qt_transform = viewer_->sceneWorldToAlignedTransform();
    Mat4d world_to_aligned;
    for (int row = 0; row < 4; ++row)
        for (int column = 0; column < 4; ++column)
            world_to_aligned(row, column) = qt_transform(row, column);

    TrajectorySmoothingOptions options;
    options.preserve_height = true;
    if (!trajectory_processing_.smoothTrajectory(world_to_aligned, options)) {
        viewer_->setMiniMapTrajectory({}, {});
        emit trajectoryStatusChanged(
            "Trajectory smoothing failed: " +
                QString::fromStdString(trajectory_processing_.lastError()), {});
        return;
    }

    std::vector<QVector3D> raw_positions;
    raw_positions.reserve(trajectory_processing_.poses().size());
    for (const OptimizedCameraPose& pose : trajectory_processing_.poses()) {
        raw_positions.push_back(qt_transform.map(QVector3D(
            static_cast<float>(pose.position_world.x()),
            static_cast<float>(pose.position_world.y()),
            static_cast<float>(pose.position_world.z()))));
    }
    std::vector<QVector3D> smooth_positions;
    smooth_positions.reserve(trajectory_processing_.smoothedTrajectory().size());
    for (const TrajectoryPoint& point :
         trajectory_processing_.smoothedTrajectory()) {
        smooth_positions.emplace_back(
            static_cast<float>(point.position.x()),
            static_cast<float>(point.position.y()),
            static_cast<float>(point.position.z()));
    }
    // Recorded camera motion is a measured path, not an obstacle-grid route.
    viewer_->setMiniMapTrajectory(raw_positions, smooth_positions, false);
}

void ViewerController::showPathOverview()
{
    if (robot_path_player_->isPlaying()) robot_path_player_->togglePlayPause();
    setPathEditingEnabled(false);
    setFreeZoneEditingEnabled(false);
    setWalkableCellEditingEnabled(false);
    ensureSceneAlignment();
    setConstrainedZUpNavigation(false);
    if (robot_pose_initialized_ && robot_path_player_->hasPath()) {
        viewer_->setNavigationPose(
            viewer_->sceneWorldToAlignedTransform().map(
                robot_path_player_->currentWorldPosition()), yaw_degrees_);
    } else if (!navigation_path_world_.empty()) {
        viewer_->setNavigationPose(
            viewer_->sceneWorldToAlignedTransform().map(
                navigation_path_world_.front()), yaw_degrees_);
    }
    viewer_->setOverviewMode(true);
}
