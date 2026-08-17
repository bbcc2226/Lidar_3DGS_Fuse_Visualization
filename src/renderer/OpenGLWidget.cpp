#include "OpenGLWidget.h"

#include <QLinearGradient>
#include <QMouseEvent>
#include <QPainter>
#include <QPolygonF>
#include <QWheelEvent>

#include <algorithm>

OpenGLWidget::OpenGLWidget(QWidget* parent)
    : QOpenGLWidget(parent)
{
    setMinimumSize(640, 480);
    setFocusPolicy(Qt::StrongFocus);
    animation_timer_.setInterval(16);
    connect(&animation_timer_, &QTimer::timeout, this, [this]() {
        angle_degrees_ += animation_speed_ * animation_timer_.interval() / 1000.0;
        if (angle_degrees_ >= 360.0) angle_degrees_ -= 360.0;
        update();
    });
    animation_timer_.start();
}

void OpenGLWidget::setAnimating(bool enabled)
{
    enabled ? animation_timer_.start() : animation_timer_.stop();
}

void OpenGLWidget::setAnimationSpeed(int degrees_per_second)
{
    animation_speed_ = degrees_per_second;
}

void OpenGLWidget::setBackgroundColor(const QColor& color)
{
    if (color.isValid()) {
        background_color_ = color;
        update();
    }
}

void OpenGLWidget::resetView()
{
    angle_degrees_ = 0.0;
    zoom_ = 1.0;
    update();
}

void OpenGLWidget::initializeGL()
{
    initializeOpenGLFunctions();
    glDisable(GL_DEPTH_TEST);
}

void OpenGLWidget::resizeGL(int width, int height)
{
    glViewport(0, 0, width, height);
}

void OpenGLWidget::paintGL()
{
    glClearColor(background_color_.redF(), background_color_.greenF(),
                 background_color_.blueF(), 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing);

    painter.setPen(QPen(QColor(255, 255, 255, 22), 1.0));
    constexpr int grid_spacing = 40;
    for (int x = width() / 2 % grid_spacing; x < width(); x += grid_spacing)
        painter.drawLine(x, 0, x, height());
    for (int y = height() / 2 % grid_spacing; y < height(); y += grid_spacing)
        painter.drawLine(0, y, width(), y);

    painter.translate(width() * 0.5, height() * 0.5);
    painter.scale(zoom_, zoom_);
    painter.rotate(angle_degrees_);

    const double radius = std::min(width(), height()) * 0.22;
    QPolygonF triangle;
    triangle << QPointF(0.0, -radius)
             << QPointF(radius * 0.866, radius * 0.5)
             << QPointF(-radius * 0.866, radius * 0.5);

    QLinearGradient fill(-radius, -radius, radius, radius);
    fill.setColorAt(0.0, QColor(52, 211, 153));
    fill.setColorAt(0.5, QColor(59, 130, 246));
    fill.setColorAt(1.0, QColor(168, 85, 247));
    painter.setBrush(fill);
    painter.setPen(QPen(QColor(235, 245, 255), 3.0));
    painter.drawPolygon(triangle);

    painter.setBrush(QColor(255, 255, 255));
    painter.setPen(Qt::NoPen);
    painter.drawEllipse(QPointF(0.0, 0.0), 6.0, 6.0);

    painter.resetTransform();
    painter.setPen(QColor(215, 225, 240));
    painter.drawText(18, 28, "Qt OpenGL viewport  |  drag to rotate  |  wheel to zoom");
}

void OpenGLWidget::mousePressEvent(QMouseEvent* event)
{
    last_mouse_position_ = event->pos();
}

void OpenGLWidget::mouseMoveEvent(QMouseEvent* event)
{
    const QPoint delta = event->pos() - last_mouse_position_;
    angle_degrees_ += delta.x() * 0.5;
    last_mouse_position_ = event->pos();
    update();
}

void OpenGLWidget::wheelEvent(QWheelEvent* event)
{
    zoom_ = std::max(0.25, std::min(3.0,
        zoom_ * (event->angleDelta().y() > 0 ? 1.1 : 0.9)));
    update();
    event->accept();
}
