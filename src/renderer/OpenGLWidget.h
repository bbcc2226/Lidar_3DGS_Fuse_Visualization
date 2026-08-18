#pragma once

#include <QColor>
#include <QMatrix4x4>
#include <QOpenGLBuffer>
#include <QOpenGLFunctions>
#include <QOpenGLVertexArrayObject>
#include <QOpenGLWidget>
#include <QPoint>
#include <QTimer>
#include <QVector3D>

#include <memory>
#include <vector>

#include "3dgsProcessing.h"

class QMouseEvent;
class QOpenGLShaderProgram;
class QPainter;
class QWheelEvent;

class OpenGLWidget final : public QOpenGLWidget, protected QOpenGLFunctions
{
public:
    explicit OpenGLWidget(QWidget* parent = nullptr);
    ~OpenGLWidget() override;

    double angle() const { return angle_degrees_; }
    bool isAnimating() const { return animation_timer_.isActive(); }

    void setAnimating(bool enabled);
    void setAnimationSpeed(int degrees_per_second);
    void setBackgroundColor(const QColor& color);
    void setGaussianPoints(const std::vector<GaussianPoint>& points);
    void resetView();

    std::size_t uploadedPointCount() const { return uploaded_point_count_; }
    bool isPointShaderReady() const;

protected:
    void initializeGL() override;
    void resizeGL(int width, int height) override;
    void paintGL() override;
    void mousePressEvent(QMouseEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;
    void wheelEvent(QWheelEvent* event) override;

private:
    bool createPointBuffers();
    bool createPointShaderProgram();
    bool uploadPointBuffer();
    void destroyPointResources();
    void resetCameraMatrices();
    void renderGaussianPoints();
    void paintDemoScene(QPainter& painter);

    QTimer animation_timer_;
    QOpenGLVertexArrayObject point_vao_;
    QOpenGLBuffer point_vbo_{QOpenGLBuffer::VertexBuffer};
    std::unique_ptr<QOpenGLShaderProgram> point_shader_program_;
    std::vector<GaussianPoint> point_data_;
    QMatrix4x4 model_matrix_;
    QMatrix4x4 view_matrix_;
    QMatrix4x4 projection_matrix_;
    QVector3D camera_position_{0.0f, 0.0f, 3.0f};
    QVector3D camera_target_{0.0f, 0.0f, 0.0f};
    QVector3D camera_up_{0.0f, 1.0f, 0.0f};
    QColor background_color_{25, 30, 42};
    QPoint last_mouse_position_;
    double angle_degrees_ = 0.0;
    double zoom_ = 1.0;
    int animation_speed_ = 45;
    std::size_t uploaded_point_count_ = 0;
};
