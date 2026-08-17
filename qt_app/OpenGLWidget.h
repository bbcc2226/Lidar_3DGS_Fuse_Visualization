#pragma once

#include <QColor>
#include <QOpenGLFunctions>
#include <QOpenGLWidget>
#include <QPoint>
#include <QTimer>

class QMouseEvent;
class QWheelEvent;

class OpenGLWidget final : public QOpenGLWidget, protected QOpenGLFunctions
{
public:
    explicit OpenGLWidget(QWidget* parent = nullptr);

    double angle() const { return angle_degrees_; }
    bool isAnimating() const { return animation_timer_.isActive(); }

    void setAnimating(bool enabled);
    void setAnimationSpeed(int degrees_per_second);
    void setBackgroundColor(const QColor& color);
    void resetView();

protected:
    void initializeGL() override;
    void resizeGL(int width, int height) override;
    void paintGL() override;
    void mousePressEvent(QMouseEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;
    void wheelEvent(QWheelEvent* event) override;

private:
    QTimer animation_timer_;
    QColor background_color_{25, 30, 42};
    QPoint last_mouse_position_;
    double angle_degrees_ = 0.0;
    double zoom_ = 1.0;
    int animation_speed_ = 45;
};
