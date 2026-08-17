#include "OpenGLWidget.h"

#include <QLinearGradient>
#include <QMouseEvent>
#include <QPainter>
#include <QPolygonF>
#include <QWheelEvent>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <type_traits>

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

OpenGLWidget::~OpenGLWidget()
{
    // OpenGL resources must be destroyed while this widget's context is
    // current. Qt may otherwise defer deletion until after the context dies.
    if (context()) {
        makeCurrent();
        point_vbo_.destroy();
        point_vao_.destroy();
        doneCurrent();
    }
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

void OpenGLWidget::setGaussianPoints(const std::vector<GaussianPoint>& points)
{
    pending_points_ = points;

    // Calls made before initializeGL() are retained and uploaded when the
    // context becomes available.
    if (isValid()) {
        makeCurrent();
        uploadPointBuffer();
        doneCurrent();
    }
    update();
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
    createPointBuffers();
    uploadPointBuffer();
}

void OpenGLWidget::createPointBuffers()
{
    static_assert(std::is_standard_layout<GaussianPoint>::value,
                  "GaussianPoint must remain usable as an interleaved vertex type.");

    if (!point_vao_.isCreated() && !point_vao_.create()) {
        qWarning("Failed to create Gaussian point VAO.");
        return;
    }
    if (!point_vbo_.isCreated() && !point_vbo_.create()) {
        qWarning("Failed to create Gaussian point VBO.");
        return;
    }

    QOpenGLVertexArrayObject::Binder vao_binder(&point_vao_);
    if (!point_vbo_.bind()) {
        qWarning("Failed to bind Gaussian point VBO.");
        return;
    }
    point_vbo_.setUsagePattern(QOpenGLBuffer::StaticDraw);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(
        0, 3, GL_FLOAT, GL_FALSE, sizeof(GaussianPoint),
        reinterpret_cast<const void*>(offsetof(GaussianPoint, x)));

    glEnableVertexAttribArray(1);
    glVertexAttribPointer(
        1, 3, GL_UNSIGNED_BYTE, GL_TRUE, sizeof(GaussianPoint),
        reinterpret_cast<const void*>(offsetof(GaussianPoint, red)));

    point_vbo_.release();
}

void OpenGLWidget::uploadPointBuffer()
{
    if (!point_vbo_.isCreated()) return;

    const std::size_t byte_count = pending_points_.size() * sizeof(GaussianPoint);
    if (byte_count > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        qWarning("Gaussian point buffer exceeds QOpenGLBuffer's allocation limit.");
        uploaded_point_count_ = 0;
        return;
    }

    if (!point_vbo_.bind()) {
        qWarning("Failed to bind Gaussian point VBO for upload.");
        uploaded_point_count_ = 0;
        return;
    }
    point_vbo_.allocate(
        pending_points_.empty() ? nullptr : pending_points_.data(),
        static_cast<int>(byte_count));
    point_vbo_.release();
    uploaded_point_count_ = pending_points_.size();
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
