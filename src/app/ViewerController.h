#pragma once

#include <QObject>
#include <QPoint>
#include <QVector3D>

#include "3dgsProcessing.h"
#include "TrajectoryProcessing.h"
#include "SemanticObject.h"
#include "NavigationIntentParser.h"
#include <QHash>
#include <QFutureWatcher>
#include <memory>
#include <future>
#include <set>

class OpenGLWidget;
class RobotPathPlayer;
class QWidget;
class QVariantAnimation;

class ViewerController final : public QObject
{
    Q_OBJECT

public:
    explicit ViewerController(OpenGLWidget* viewer, QObject* parent = nullptr);
    void setUseOriginalTrajectoryHeight(bool enabled);

    void openPlyFile(QWidget* dialog_parent);
    bool loadPlyFile(const QString& path);
    bool loadLidarPointCloud(const QString& path);
    void setLidarViewEnabled(bool enabled);
    void openTrajectoryFile(QWidget* dialog_parent);
    void saveManualPath(QWidget* dialog_parent);
    void loadRobotPath(QWidget* dialog_parent);
    bool loadRobotPathFile(const QString& path);
    void toggleRobotPlayback();
    void stopRobotPlayback();
    void clearLoadedPath();
    void setRobotPlaybackSpeed(float meters_per_second);
    void resetView();
    void showPathOverview();
    void setConstrainedZUpNavigation(bool enabled);
    void setPathEditingEnabled(bool enabled);
    void setFreeZoneEditingEnabled(bool enabled);
    void closeFreeZone();
    void reopenFreeZone();
    void startJoinedFreeZone(int shared_edge_count, bool reverse_direction);
    void startAdditiveFreeZone();
    void clearFreeZone();
    void saveFreeZone(QWidget* dialog_parent);
    void loadFreeZone(QWidget* dialog_parent);
    void setWalkableCellEditingEnabled(bool enabled);
    void initializeWalkableCellsFromPath();
    void clearWalkableCells();
    void saveWalkableCells(QWidget* dialog_parent);
    void loadWalkableCells(QWidget* dialog_parent);
    bool loadWalkableCellsFile(const QString& path);
    void setWalkableCellsVisible(bool visible);
    void loadSemanticDatabase(QWidget* dialog_parent);
    void loadDefaultSemanticDatabase();
    void setSemanticObjectsVisible(bool visible);
    void setSemanticClassFilter(const QString& filter);
    void selectSemanticObject(int object_id, bool focus_view = true);
    void setSemanticVerificationMode(bool enabled);
    void rotateSemanticView(float delta_degrees);
    void planNavigationRequest(const QString& request);
    void reviewSemanticObject(int object_id, SemanticReviewStatus status);
    void saveSemanticReviews(QWidget* dialog_parent, bool save_as = false);
    void saveSemanticObjectFile(QWidget* dialog_parent);
    const std::vector<SemanticObject>& semanticObjects() const { return semantic_objects_; }
    int selectedSemanticObjectId() const { return selected_semantic_object_id_; }

signals:
    void originalTrajectoryHeightChanged(bool enabled);
    void orientationChanged(float yaw_degrees, float pitch_degrees);
    void loadProgressChanged(int percent, const QString& stage);
    void loadStatusChanged(const QString& text, const QString& file_path);
    void floorAlignmentStatusChanged(const QString& text);
    void trajectoryStatusChanged(const QString& text, const QString& file_path);
    void manualPathStatusChanged(const QString& text, const QString& file_path);
    void robotPlaybackStatusChanged(const QString& text, const QString& file_path);
    void semanticDatabaseChanged(const QString& text, const QString& file_path);
    void semanticSelectionChanged(int object_id);
    void navigationPlanStatusChanged(const QString& text);
    void freeZoneStatusChanged(const QString& text, const QString& file_path);

protected:
    bool eventFilter(QObject* watched, QEvent* event) override;

private:
    struct SpatialRelation { int subject_id = -1; int reference_id = -1;
        QString type; float score = 0.0f; };
    enum class DragMode {
        None,
        RotateFree,
        RotatePending,
        RotateYawOnly,
        RotatePitchOnly,
        SemanticLook,
        Pan
    };

    void applyTransform();
    void snapToAxis(int axis);
    void updateTrajectoryForScene();
    void updateManualPathDisplay();
    void updateFreeZoneDisplay();
    void refreshLoadedRobotPathDisplay();
    void beginArrivalView();
    void restoreNavigationView();
    float playbackCameraHeight(float recorded_height) const;
    bool ensureSceneAlignment();
    bool loadSemanticDatabaseFile(const QString& path);
    void loadSemanticReviewsFile(const QString& path);
    bool planThroughFreeZone(const QString& canonical,
                             const QString& requested_destination);
    bool planThroughWalkableCells(const QString& canonical,
                                  const QString& requested_destination);
    void rebuildSpatialRelations();
    void updateWalkableCellDisplay();
    std::pair<int, int> walkableCellAt(const QVector3D& aligned_point) const;
    void extendWalkableCellStroke(const std::pair<int, int>& cell, bool erase);

    OpenGLWidget* viewer_ = nullptr;
    GaussianSplatProcessing point_processing_;
    std::vector<GaussianPoint> gaussian_points_;
    std::vector<GaussianPoint> lidar_points_;
    bool lidar_view_enabled_ = false;
    QFutureWatcher<std::shared_ptr<std::vector<GaussianPoint>>>* lidar_watcher_ = nullptr;
    bool lidar_load_in_progress_ = false;
    QString lidar_loaded_path_;
    QString lidar_loading_path_;
    bool gaussian_has_scale_ = false;
    bool gaussian_has_sh_dc_ = false;
    int gaussian_sh_degree_ = 0;
    TrajectoryProcessing trajectory_processing_;
    RobotPathPlayer* robot_path_player_ = nullptr;
    QPoint last_mouse_position_;
    QPoint drag_start_position_;
    int pressed_semantic_object_id_ = -1;
    QVector3D translation_;
    QVector3D z_up_camera_position_{0.0f, -3.0f, 0.0f};
    std::vector<QVector3D> manual_path_;
    std::vector<QVector3D> navigation_path_world_;
    std::vector<QVector3D> free_zone_vertices_;
    std::vector<std::vector<QVector3D>> completed_free_zone_polygons_;
    int selected_free_zone_vertex_ = -1;
    float free_zone_height_ = 0.0f;
    bool free_zone_closed_ = false;
    bool free_zone_editing_enabled_ = false;
    std::set<std::pair<int, int>> walkable_cells_;
    float walkable_cell_size_ = 0.12f;
    float walkable_floor_z_ = 0.0f;
    float walkable_grid_angle_radians_ = 0.0f;
    bool walkable_cell_editing_enabled_ = false;
    bool painting_walkable_cells_ = false;
    std::pair<int, int> last_painted_walkable_cell_{};
    bool has_last_painted_walkable_cell_ = false;
    bool erasing_walkable_cells_ = false;
    bool navigation_has_started_ = false;
    bool dragging_free_zone_vertex_ = false;
    std::size_t free_zone_seed_vertex_count_ = 0;
    QVector3D free_zone_default_start_world_;
    bool has_free_zone_default_start_ = false;
    std::vector<SemanticObject> semantic_objects_;
    std::vector<SpatialRelation> spatial_relations_;
    std::set<int> relation_target_ids_;
    int preferred_navigation_object_id_ = -1;
    int selected_semantic_object_id_ = -1;
    QString semantic_database_path_;
    QString semantic_review_path_;
    bool semantic_reviews_dirty_ = false;
    bool semantic_verification_mode_ = false;
    bool robot_pose_initialized_ = false;
    bool scene_alignment_ready_ = false;
    QVariantAnimation* arrival_view_animation_ = nullptr;
    QVector3D navigation_target_center_aligned_;
    QVector3D arrival_view_start_position_;
    QVector3D arrival_view_end_position_;
    float arrival_view_start_yaw_ = 0.0f;
    float arrival_view_end_yaw_ = 0.0f;
    float arrival_view_start_pitch_ = 0.0f;
    float arrival_view_end_pitch_ = 0.0f;
    bool has_navigation_target_ = false;
    QHash<QString, QString> semantic_aliases_;
    NavigationIntentParser navigation_intent_parser_;
    int selected_manual_path_point_ = -1;
    float manual_path_height_ = 0.0f;
    float yaw_degrees_ = 0.0f;
    float pitch_degrees_ = 0.0f;
    float camera_distance_ = 3.0f;
    float navigation_eye_height_meters_ = 1.10f;
    float recorded_height_offset_meters_ = 0.0f;
    bool use_original_trajectory_height_ = false;
    bool constrained_z_up_navigation_ = false;
    bool path_editing_enabled_ = false;
    bool dragging_manual_path_point_ = false;
    DragMode drag_mode_ = DragMode::None;
};
