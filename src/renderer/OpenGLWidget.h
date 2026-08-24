#pragma once

#include <QColor>
#include <QMatrix4x4>
#include <QOpenGLBuffer>
#include <QOpenGLExtraFunctions>
#include <QOpenGLVertexArrayObject>
#include <QOpenGLWidget>
#include <QVector3D>

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "3dgsProcessing.h"

class QOpenGLShaderProgram;
class QPainter;

class OpenGLWidget final : public QOpenGLWidget, protected QOpenGLExtraFunctions
{
public:
    explicit OpenGLWidget(QWidget* parent = nullptr);
    ~OpenGLWidget() override;

    void setBackgroundColor(const QColor& color);
    void setGaussianPoints(const std::vector<GaussianPoint>& points,
                           bool has_trained_scale = false,
                           bool has_sh_dc = false,
                           int sh_degree = 0);
    void setInteractionTransform(const QMatrix4x4& transform,
                                 float yaw_degrees, float pitch_degrees);
    void setZUpGizmo(bool enabled);
    void finalizeInteractionSort();
    std::optional<QVector3D> pickGaussianAt(const QPoint& position) const;
    bool alignFloorFromPoints(const std::vector<QVector3D>& points);
    void clearFloorAlignment();
    void setFloorSelectionPoints(const std::vector<QVector3D>& points);

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
        float sh_dc_r = 0.0f;
        float sh_dc_g = 0.0f;
        float sh_dc_b = 0.0f;
        float sh_l1_0_r = 0.0f;
        float sh_l1_0_g = 0.0f;
        float sh_l1_0_b = 0.0f;
        float sh_l1_1_r = 0.0f;
        float sh_l1_1_g = 0.0f;
        float sh_l1_1_b = 0.0f;
        float sh_l1_2_r = 0.0f;
        float sh_l1_2_g = 0.0f;
        float sh_l1_2_b = 0.0f;
    };

    bool createSplatBuffers();
    bool createSplatShaderProgram();
    bool uploadSplatBuffer();
    bool sortAndUploadSplats(const QMatrix4x4& model_view);
    void destroySplatResources();
    void resetCameraMatrices();
    void fitPointCloudToView();
    void renderGaussianSplats();
    void paintDemoScene(QPainter& painter);

    QOpenGLVertexArrayObject splat_vao_;
    QOpenGLBuffer quad_vbo_{QOpenGLBuffer::VertexBuffer};
    QOpenGLBuffer splat_instance_vbo_{QOpenGLBuffer::VertexBuffer};
    std::unique_ptr<QOpenGLShaderProgram> splat_shader_program_;
    std::vector<GpuSplatData> source_gpu_splat_data_;
    std::vector<GpuSplatData> gpu_splat_data_;
    std::vector<std::uint32_t> splat_sort_indices_;
    std::vector<std::uint32_t> splat_sort_scratch_;
    std::vector<float> splat_sort_depths_;
    QMatrix4x4 model_matrix_;
    QMatrix4x4 scene_alignment_matrix_;
    QMatrix4x4 view_matrix_;
    QMatrix4x4 projection_matrix_;
    QMatrix4x4 interaction_matrix_;
    QVector3D camera_position_{0.0f, 0.0f, 3.0f};
    QVector3D camera_target_{0.0f, 0.0f, 0.0f};
    QVector3D camera_up_{0.0f, 1.0f, 0.0f};
    QColor background_color_{25, 30, 42};
    float yaw_degrees_ = 0.0f;
    float pitch_degrees_ = 0.0f;
    float point_cloud_scale_ = 1.0f;
    bool has_trained_scale_ = false;
    bool has_sh_dc_ = false;
    int sh_degree_ = 0;
    bool z_up_gizmo_ = false;
    bool splat_sort_dirty_ = true;
    QVector3D last_sort_depth_direction_;
    std::vector<QVector3D> floor_selection_points_;
    std::size_t allocated_splat_bytes_ = 0;
    std::size_t uploaded_splat_count_ = 0;
};
