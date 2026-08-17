#pragma once

#include <QColor>
#include <QOpenGLBuffer>
#include <QOpenGLFunctions>
#include <QOpenGLVertexArrayObject>
#include <QOpenGLWidget>
#include <QPoint>
#include <QTimer>

#include <vector>

#include "3dgsProcessing.h"

class QMouseEvent;
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

protected:
    void initializeGL() override;
    void resizeGL(int width, int height) override;
    void paintGL() override;
    void mousePressEvent(QMouseEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;
    void wheelEvent(QWheelEvent* event) override;

private:
    void createPointBuffers();
    void uploadPointBuffer();

    QTimer animation_timer_;
    QOpenGLVertexArrayObject point_vao_;
    QOpenGLBuffer point_vbo_{QOpenGLBuffer::VertexBuffer};
    std::vector<GaussianPoint> pending_points_;
    QColor background_color_{25, 30, 42};
    QPoint last_mouse_position_;
    double angle_degrees_ = 0.0;
    double zoom_ = 1.0;
    int animation_speed_ = 45;
    std::size_t uploaded_point_count_ = 0;
};
