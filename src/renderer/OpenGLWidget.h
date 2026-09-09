#pragma once

#include <QColor>
#include <QMatrix4x4>
#include <QOpenGLBuffer>
#include <QOpenGLExtraFunctions>
#include <QOpenGLVertexArrayObject>
#include <QOpenGLWidget>
#include <QPoint>
#include <QVector3D>

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>
#include <set>

#include "3dgsProcessing.h"
#include "SemanticObject.h"

class QOpenGLShaderProgram;
class QPainter;

class OpenGLWidget final : public QOpenGLWidget, protected QOpenGLExtraFunctions
{
public:
    explicit OpenGLWidget(QWidget* parent = nullptr);
    ~OpenGLWidget() override;

    void setBackgroundColor(const QColor& color);
    void setExposure(float exposure_ev);
    void setLowPassVariance(float variance);
    void setMaximumSplatHalfSize(float pixels);
    void setMinimumSplatOpacity(float opacity);
    void setVerticalFieldOfView(float degrees);
    void setLargeSplatHighlightEnabled(bool enabled);
    void setOversizedSplatSuppressionEnabled(bool enabled);
    void setViewportOverlayEnabled(bool enabled);
    void setGaussianPoints(const std::vector<GaussianPoint>& points,
                           bool has_trained_scale = false,
                           bool has_sh_dc = false,
                           int sh_degree = 0);
    void setInteractionTransform(const QMatrix4x4& transform,
                                 float yaw_degrees, float pitch_degrees);
    void setZUpGizmo(bool enabled);
    void setNavigationPose(const QVector3D& position, float yaw_degrees);
    void setMiniMapTrajectory(const std::vector<QVector3D>& raw_positions,
                              const std::vector<QVector3D>& smooth_positions);
    QMatrix4x4 sceneWorldToAlignedTransform() const;
    float robotEyeHeightAligned(float height_meters,
                                float fallback_height) const;
    void setPathEditMode(bool enabled);
    void zoomPathEditView(float wheel_steps);
    std::optional<QVector3D> screenToPathPlane(
        const QPoint& screen_position, float aligned_height) const;
    int hitTestManualPathPoint(const QPoint& screen_position,
                               float radius_pixels = 11.0f) const;
    void setManualPath(const std::vector<QVector3D>& points,
                       int selected_index = -1);
    void setFreeZone(
        const std::vector<std::vector<QVector3D>>& completed_polygons,
        const std::vector<QVector3D>& active_vertices, bool active_closed,
        int selected_index = -1);
    void setWalkableCells(const std::set<std::pair<int, int>>& cells,
                          float cell_size, float floor_z, float angle_radians);
    void setWalkableCellsVisible(bool visible);
    int hitTestFreeZoneVertex(const QPoint& screen_position,
                              float radius_pixels = 11.0f) const;
    void setSemanticObjects(const std::vector<SemanticObject>& objects);
    void setSemanticObjectsVisible(bool visible);
    void setSemanticClassFilter(const QString& filter);
    void setSelectedSemanticObject(int object_id);
    void setSemanticSelectedOnly(bool selected_only);
    void setNavigationPlan(const std::vector<QVector3D>& aligned_route,
                           const QVector3D& aligned_target,
                           const QString& destination_name);
    void setNavigationTargetTagVisible(bool visible);
    void clearNavigationPlan();
    int hitTestSemanticObject(const QPoint& screen_position) const;
    void finalizeInteractionSort();
    bool autoAlignSceneUp();

    std::size_t uploadedSplatCount() const
    {
        return source_gpu_splat_data_.size();
    }
    bool isSplatShaderReady() const;

protected:
    void initializeGL() override;
    void resizeGL(int width, int height) override;
    void paintGL() override;

private:
    // Compact upload format owned by the renderer. The loader's GaussianPoint
    // can evolve without changing the OpenGL vertex layout.
    struct GpuSplatData
    {
        float x = 0.0f;
        float y = 0.0f;
        float z = 0.0f;
        std::uint8_t red = 255;
        std::uint8_t green = 255;
        std::uint8_t blue = 255;
        std::uint8_t color_padding = 255;
        float opacity = 0.0f;
        float scale_x = 0.0f;
        float scale_y = 0.0f;
        float scale_z = 0.0f;
        float rotation_w = 1.0f;
        float rotation_x = 0.0f;
        float rotation_y = 0.0f;
        float rotation_z = 0.0f;
        std::uint32_t original_index = 0;
    };

    bool createSplatBuffers();
    bool createSplatShaderProgram();
    bool createGpuReorderProgram();
    bool createTrajectoryShaderProgram();
    bool uploadSplatBuffer();
    bool uploadShTextureBuffer();
    bool sortAndUploadSplats(const QMatrix4x4& model_view);
    void destroySplatResources();
    void resetCameraMatrices();
    void fitPointCloudToView();
    void renderGaussianSplats();
    void renderTrajectory();
    void paintDemoScene(QPainter& painter);
    void rebuildMiniMapLandscape();
    void rebuildWalkability();
    void rebuildTrajectoryVertexData();
    void paintMiniMap(QPainter& painter);
    void paintManualPath(QPainter& painter);
    void paintSemanticObjects(QPainter& painter);
    void paintNavigationPlan(QPainter& painter);
    void paintFreeZone(QPainter& painter);
    void paintWalkableCells(QPainter& painter);
    QMatrix4x4 activeAlignedViewProjection() const;

    static constexpr int kMiniMapGridSize = 72;

    QOpenGLVertexArrayObject splat_vao_;
    QOpenGLVertexArrayObject trajectory_vao_;
    QOpenGLBuffer quad_vbo_{QOpenGLBuffer::VertexBuffer};
    QOpenGLBuffer splat_instance_vbo_{QOpenGLBuffer::VertexBuffer};
    QOpenGLBuffer trajectory_vbo_{QOpenGLBuffer::VertexBuffer};
    GLuint sh_buffer_ = 0;
    GLuint sh_texture_ = 0;
    GLuint source_splat_buffer_ = 0;
    GLuint sorted_index_buffer_ = 0;
    GLuint alternate_sorted_index_buffer_ = 0;
    GLuint splat_data_buffer_ = 0;
    GLuint splat_data_texture_ = 0;
    GLuint sorted_index_texture_ = 0;
    GLuint alternate_sorted_index_texture_ = 0;
    std::unique_ptr<QOpenGLShaderProgram> splat_shader_program_;
    std::unique_ptr<QOpenGLShaderProgram> gpu_reorder_program_;
    std::unique_ptr<QOpenGLShaderProgram> trajectory_shader_program_;
    std::vector<GpuSplatData> source_gpu_splat_data_;
    std::vector<GpuSplatData> gpu_splat_data_;
    std::vector<std::uint32_t> splat_sort_indices_;
    std::vector<std::uint32_t> splat_sort_scratch_;
    std::vector<float> splat_sort_depths_;
    std::vector<float> sh_coefficient_data_;
    std::vector<float> static_splat_texture_data_;
    QMatrix4x4 model_matrix_;
    QMatrix4x4 scene_alignment_matrix_;
    QMatrix4x4 view_matrix_;
    QMatrix4x4 projection_matrix_;
    QMatrix4x4 interaction_matrix_;
    QVector3D camera_position_{0.0f, 0.0f, 3.0f};
    QVector3D camera_target_{0.0f, 0.0f, 0.0f};
    QVector3D camera_up_{0.0f, 1.0f, 0.0f};
    QVector3D navigation_position_{0.0f, -0.65f, 0.0f};
    std::array<std::uint16_t,
               kMiniMapGridSize * kMiniMapGridSize> minimap_occupancy_{};
    // 0 unknown, 1 obstacle, 2 inflated/insufficient clearance, 3 candidate
    // walkable, 4 trajectory/obstacle conflict.
    std::array<std::uint8_t,
               kMiniMapGridSize * kMiniMapGridSize> walkability_{};
    float minimap_min_x_ = -1.0f;
    float minimap_max_x_ = 1.0f;
    float minimap_min_y_ = -1.0f;
    float minimap_max_y_ = 1.0f;
    std::uint16_t minimap_peak_occupancy_ = 0;
    std::vector<QVector3D> raw_trajectory_;
    std::vector<QVector3D> proposed_trajectory_;
    std::vector<QVector3D> smooth_trajectory_;
    std::vector<float> trajectory_vertex_data_;
    std::vector<QVector3D> manual_path_;
    std::vector<SemanticObject> semantic_objects_;
    QString semantic_class_filter_;
    int selected_semantic_object_id_ = -1;
    bool semantic_objects_visible_ = false;
    bool semantic_selected_only_ = true;
    std::vector<QVector3D> navigation_plan_;
    QVector3D navigation_target_;
    bool has_navigation_target_ = false;
    bool navigation_target_tag_visible_ = false;
    QString navigation_destination_name_;
    int selected_manual_path_point_ = -1;
    std::vector<QVector3D> free_zone_vertices_;
    std::vector<std::vector<QVector3D>> completed_free_zone_polygons_;
    int selected_free_zone_vertex_ = -1;
    bool free_zone_closed_ = false;
    std::set<std::pair<int, int>> walkable_cells_;
    float walkable_cell_size_ = 0.12f;
    float walkable_floor_z_ = 0.0f;
    float walkable_grid_angle_radians_ = 0.0f;
    bool walkable_cells_visible_ = true;
    bool trajectory_buffer_dirty_ = true;
    bool path_edit_mode_ = false;
    float path_edit_ortho_half_height_ = 1.0f;
    float path_edit_ceiling_cutoff_ = 0.5f;
    float estimated_floor_z_ = 0.0f;
    bool has_estimated_floor_ = false;
    QColor background_color_{25, 30, 42};
    float yaw_degrees_ = 0.0f;
    float pitch_degrees_ = 0.0f;
    float point_cloud_scale_ = 1.0f;
    float exposure_ev_ = 0.0f;
    float low_pass_variance_ = 0.3f;
    float maximum_splat_half_size_ = 256.0f;
    float minimum_splat_opacity_ = 0.0f;
    float vertical_field_of_view_degrees_ = 45.0f;
    bool highlight_large_splats_ = false;
    bool suppress_oversized_splats_ = false;
    bool viewport_overlay_enabled_ = false;
    bool has_trained_scale_ = false;
    bool has_sh_dc_ = false;
    int sh_degree_ = 0;
    bool z_up_gizmo_ = false;
    bool navigation_pose_visible_ = false;
    bool sh_buffer_dirty_ = false;
    bool source_splat_buffer_dirty_ = false;
    bool static_splat_texture_dirty_ = false;
    bool gpu_reorder_available_ = false;
    bool splat_sort_dirty_ = true;
    bool alternate_sorted_index_active_ = false;
    QVector3D last_sort_depth_direction_;
    std::size_t allocated_splat_bytes_ = 0;
    std::size_t uploaded_splat_count_ = 0;
};
