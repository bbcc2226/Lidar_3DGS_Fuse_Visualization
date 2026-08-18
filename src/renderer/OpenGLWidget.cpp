#include "OpenGLWidget.h"

#include <QLinearGradient>
#include <QMouseEvent>
#include <QOpenGLContext>
#include <QOpenGLShader>
#include <QOpenGLShaderProgram>
#include <QPainter>
#include <QPolygonF>
#include <QWheelEvent>

#include <algorithm>
#include <cstddef>
#include <cmath>
#include <limits>
#include <type_traits>

namespace
{
constexpr int kAnimationIntervalMs = 16;
constexpr int kPositionAttribute = 0;
constexpr int kColorAttribute = 1;

constexpr char kPointVertexShader[] = R"GLSL(
#version 330 core

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_color;

out vec3 vertex_color;

void main()
{
    gl_Position = vec4(in_position, 1.0);
    gl_PointSize = 3.0;
    vertex_color = in_color;
}
)GLSL";

constexpr char kPointFragmentShader[] = R"GLSL(
#version 330 core

in vec3 vertex_color;
out vec4 fragment_color;

void main()
{
    fragment_color = vec4(vertex_color, 1.0);
}
)GLSL";
} // namespace

OpenGLWidget::OpenGLWidget(QWidget* parent)
    : QOpenGLWidget(parent)
{
    setMinimumSize(640, 480);
    setFocusPolicy(Qt::StrongFocus);
    animation_timer_.setInterval(kAnimationIntervalMs);
    connect(&animation_timer_, &QTimer::timeout, this, [this]() {
        angle_degrees_ += animation_speed_ * animation_timer_.interval() / 1000.0;
        angle_degrees_ = std::fmod(angle_degrees_, 360.0);
        update();
    });
    animation_timer_.start();
}

OpenGLWidget::~OpenGLWidget()
{
    if (context() && context()->isValid()) {
        makeCurrent();
        destroyPointResources();
        doneCurrent();
    }
}

bool OpenGLWidget::isPointShaderReady() const
{
    return point_shader_program_ && point_shader_program_->isLinked();
}

void OpenGLWidget::setAnimating(bool enabled)
{
    if (enabled) {
        animation_timer_.start();
    } else {
        animation_timer_.stop();
    }
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
    point_data_ = points;

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
    glEnable(GL_PROGRAM_POINT_SIZE);

    if (createPointBuffers()) {
        uploadPointBuffer();
    }
    createPointShaderProgram();
}

bool OpenGLWidget::createPointShaderProgram()
{
    point_shader_program_ = std::make_unique<QOpenGLShaderProgram>();
    if (!point_shader_program_->addShaderFromSourceCode(
            QOpenGLShader::Vertex, kPointVertexShader)) {
        qWarning("Gaussian point vertex shader compilation failed: %s",
                 qPrintable(point_shader_program_->log()));
        point_shader_program_.reset();
        return false;
    }

    if (!point_shader_program_->addShaderFromSourceCode(
            QOpenGLShader::Fragment, kPointFragmentShader)) {
        qWarning("Gaussian point fragment shader compilation failed: %s",
                 qPrintable(point_shader_program_->log()));
        point_shader_program_.reset();
        return false;
    }

    if (!point_shader_program_->link()) {
        qWarning("Gaussian point shader link failed: %s",
                 qPrintable(point_shader_program_->log()));
        point_shader_program_.reset();
        return false;
    }

    return true;
}

bool OpenGLWidget::createPointBuffers()
{
    static_assert(std::is_standard_layout<GaussianPoint>::value,
                  "GaussianPoint must remain usable as an interleaved vertex type.");

    if (!point_vao_.isCreated() && !point_vao_.create()) {
        qWarning("Failed to create Gaussian point VAO.");
        return false;
    }
    if (!point_vbo_.isCreated() && !point_vbo_.create()) {
        qWarning("Failed to create Gaussian point VBO.");
        return false;
    }

    QOpenGLVertexArrayObject::Binder vao_binder(&point_vao_);
    if (!point_vbo_.bind()) {
        qWarning("Failed to bind Gaussian point VBO.");
        return false;
    }
    point_vbo_.setUsagePattern(QOpenGLBuffer::StaticDraw);

    glEnableVertexAttribArray(kPositionAttribute);
    glVertexAttribPointer(
        kPositionAttribute, 3, GL_FLOAT, GL_FALSE, sizeof(GaussianPoint),
        reinterpret_cast<const void*>(offsetof(GaussianPoint, x)));

    glEnableVertexAttribArray(kColorAttribute);
    glVertexAttribPointer(
        kColorAttribute, 3, GL_UNSIGNED_BYTE, GL_TRUE, sizeof(GaussianPoint),
        reinterpret_cast<const void*>(offsetof(GaussianPoint, red)));

    point_vbo_.release();
    return true;
}

bool OpenGLWidget::uploadPointBuffer()
{
    uploaded_point_count_ = 0;
    if (!point_vbo_.isCreated()) return false;

    const std::size_t byte_count = point_data_.size() * sizeof(GaussianPoint);
    if (byte_count > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        qWarning("Gaussian point buffer exceeds QOpenGLBuffer's allocation limit.");
        return false;
    }

    if (!point_vbo_.bind()) {
        qWarning("Failed to bind Gaussian point VBO for upload.");
        return false;
    }
    point_vbo_.allocate(
        point_data_.empty() ? nullptr : point_data_.data(),
        static_cast<int>(byte_count));
    point_vbo_.release();
    uploaded_point_count_ = point_data_.size();
    return true;
}

void OpenGLWidget::destroyPointResources()
{
    uploaded_point_count_ = 0;
    point_shader_program_.reset();
    point_vbo_.destroy();
    point_vao_.destroy();
}

void OpenGLWidget::renderGaussianPoints()
{
    if (uploaded_point_count_ == 0 || !isPointShaderReady() ||
        !point_vao_.isCreated()) {
        return;
    }

    if (!point_shader_program_->bind()) {
        qWarning("Failed to bind Gaussian point shader for drawing.");
        return;
    }

    {
        QOpenGLVertexArrayObject::Binder vao_binder(&point_vao_);
        glDrawArrays(
            GL_POINTS, 0, static_cast<GLsizei>(uploaded_point_count_));
    }

    point_shader_program_->release();
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

    renderGaussianPoints();

    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing);

    paintDemoScene(painter);
}

void OpenGLWidget::paintDemoScene(QPainter& painter)
{

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
    zoom_ = std::clamp(
        zoom_ * (event->angleDelta().y() > 0 ? 1.1 : 0.9), 0.25, 3.0);
    update();
    event->accept();
}
