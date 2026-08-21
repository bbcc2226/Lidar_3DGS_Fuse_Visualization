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
    void setGaussianPoints(const std::vector<GaussianPoint>& points);
    void setInteractionTransform(const QMatrix4x4& transform,
                                 float yaw_degrees, float pitch_degrees);

    std::size_t uploadedPointCount() const { return uploaded_point_count_; }
    bool isPointShaderReady() const;

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
    };

    bool createPointBuffers();
    bool createPointShaderProgram();
    bool uploadPointBuffer();
    void destroyPointResources();
    void resetCameraMatrices();
    void fitPointCloudToView();
    void renderGaussianPoints();
    void paintDemoScene(QPainter& painter);

    QOpenGLVertexArrayObject point_vao_;
    QOpenGLBuffer quad_vbo_{QOpenGLBuffer::VertexBuffer};
    QOpenGLBuffer point_vbo_{QOpenGLBuffer::VertexBuffer};
    std::unique_ptr<QOpenGLShaderProgram> point_shader_program_;
    std::vector<GpuSplatData> gpu_splat_data_;
    QMatrix4x4 model_matrix_;
    QMatrix4x4 view_matrix_;
    QMatrix4x4 projection_matrix_;
    QMatrix4x4 interaction_matrix_;
    QVector3D camera_position_{0.0f, 0.0f, 3.0f};
    QVector3D camera_target_{0.0f, 0.0f, 0.0f};
    QVector3D camera_up_{0.0f, 1.0f, 0.0f};
    QColor background_color_{25, 30, 42};
    float yaw_degrees_ = 0.0f;
    float pitch_degrees_ = 0.0f;
    std::size_t uploaded_point_count_ = 0;
};
