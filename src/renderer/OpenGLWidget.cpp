#include "OpenGLWidget.h"

#include "OrientationGizmo.h"

#include <QOpenGLContext>
#include <QOpenGLShader>
#include <QOpenGLShaderProgram>
#include <QPainter>
#include <QVector2D>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <type_traits>

namespace
{
constexpr int kCornerAttribute = 0;
constexpr int kPositionAttribute = 1;
constexpr int kColorAttribute = 2;
constexpr float kSquareHalfSizePixels = 3.0f;

constexpr float kQuadCorners[] = {
    -1.0f, -1.0f,
     1.0f, -1.0f,
    -1.0f,  1.0f,
     1.0f,  1.0f,
};

constexpr char kPointVertexShader[] = R"GLSL(
#version 330 core

layout(location = 0) in vec2 in_corner;
layout(location = 1) in vec3 in_position;
layout(location = 2) in vec3 in_color;

uniform mat4 u_mvp;
uniform vec2 u_viewport_size;
uniform float u_square_half_size_pixels;

out vec3 vertex_color;

void main()
{
    vec4 center_clip = u_mvp * vec4(in_position, 1.0);
    vec2 offset_ndc = in_corner * u_square_half_size_pixels * 2.0
        / u_viewport_size;
    gl_Position = center_clip;
    gl_Position.xy += offset_ndc * center_clip.w;
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

void OpenGLWidget::setBackgroundColor(const QColor& color)
{
    if (color.isValid()) {
        background_color_ = color;
        update();
    }
}

void OpenGLWidget::setGaussianPoints(const std::vector<GaussianPoint>& points)
{
    gpu_splat_data_.clear();
    gpu_splat_data_.reserve(points.size());
    for (const GaussianPoint& point : points) {
        GpuSplatData gpu_splat;
        gpu_splat.x = point.x;
        gpu_splat.y = point.y;
        gpu_splat.z = point.z;
        gpu_splat.red = point.red;
        gpu_splat.green = point.green;
        gpu_splat.blue = point.blue;
        gpu_splat.opacity = point.opacity;
        gpu_splat_data_.push_back(gpu_splat);
    }
    fitPointCloudToView();

    // Calls made before initializeGL() are retained and uploaded when the
    // context becomes available.
    if (isValid()) {
        makeCurrent();
        uploadPointBuffer();
        doneCurrent();
    }
    update();
}

void OpenGLWidget::setInteractionTransform(
    const QMatrix4x4& transform, float yaw_degrees, float pitch_degrees)
{
    interaction_matrix_ = transform;
    yaw_degrees_ = yaw_degrees;
    pitch_degrees_ = pitch_degrees;
    update();
}

void OpenGLWidget::initializeGL()
{
    initializeOpenGLFunctions();
    glEnable(GL_DEPTH_TEST);
    resetCameraMatrices();

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
    static_assert(std::is_standard_layout<GpuSplatData>::value,
                  "GpuSplatData must be an interleaved vertex type.");

    if (!point_vao_.isCreated() && !point_vao_.create()) {
        qWarning("Failed to create Gaussian point VAO.");
        return false;
    }
    if (!quad_vbo_.isCreated() && !quad_vbo_.create()) {
        qWarning("Failed to create shared quad VBO.");
        return false;
    }
    if (!point_vbo_.isCreated() && !point_vbo_.create()) {
        qWarning("Failed to create Gaussian point VBO.");
        return false;
    }

    QOpenGLVertexArrayObject::Binder vao_binder(&point_vao_);
    if (!quad_vbo_.bind()) {
        qWarning("Failed to bind shared quad VBO.");
        return false;
    }
    quad_vbo_.setUsagePattern(QOpenGLBuffer::StaticDraw);
    quad_vbo_.allocate(kQuadCorners, static_cast<int>(sizeof(kQuadCorners)));

    glEnableVertexAttribArray(kCornerAttribute);
    glVertexAttribPointer(
        kCornerAttribute, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), nullptr);
    glVertexAttribDivisor(kCornerAttribute, 0);
    quad_vbo_.release();

    if (!point_vbo_.bind()) {
        qWarning("Failed to bind Gaussian point VBO.");
        return false;
    }
    point_vbo_.setUsagePattern(QOpenGLBuffer::StaticDraw);

    glEnableVertexAttribArray(kPositionAttribute);
    glVertexAttribPointer(
        kPositionAttribute, 3, GL_FLOAT, GL_FALSE, sizeof(GpuSplatData),
        reinterpret_cast<const void*>(offsetof(GpuSplatData, x)));
    glVertexAttribDivisor(kPositionAttribute, 1);

    glEnableVertexAttribArray(kColorAttribute);
    glVertexAttribPointer(
        kColorAttribute, 3, GL_UNSIGNED_BYTE, GL_TRUE, sizeof(GpuSplatData),
        reinterpret_cast<const void*>(offsetof(GpuSplatData, red)));
    glVertexAttribDivisor(kColorAttribute, 1);

    point_vbo_.release();
    return true;
}

bool OpenGLWidget::uploadPointBuffer()
{
    uploaded_point_count_ = 0;
    if (!point_vbo_.isCreated()) return false;

    const std::size_t byte_count =
        gpu_splat_data_.size() * sizeof(GpuSplatData);
    if (byte_count > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        qWarning("Gaussian point buffer exceeds QOpenGLBuffer's allocation limit.");
        return false;
    }

    if (!point_vbo_.bind()) {
        qWarning("Failed to bind Gaussian point VBO for upload.");
        return false;
    }
    point_vbo_.allocate(
        gpu_splat_data_.empty() ? nullptr : gpu_splat_data_.data(),
        static_cast<int>(byte_count));
    point_vbo_.release();
    uploaded_point_count_ = gpu_splat_data_.size();
    return true;
}

void OpenGLWidget::destroyPointResources()
{
    uploaded_point_count_ = 0;
    point_shader_program_.reset();
    point_vbo_.destroy();
    quad_vbo_.destroy();
    point_vao_.destroy();
}

void OpenGLWidget::resetCameraMatrices()
{
    model_matrix_.setToIdentity();

    view_matrix_.setToIdentity();
    view_matrix_.lookAt(camera_position_, camera_target_, camera_up_);
}

void OpenGLWidget::fitPointCloudToView()
{
    model_matrix_.setToIdentity();
    if (gpu_splat_data_.empty()) return;

    QVector3D minimum(
        gpu_splat_data_.front().x,
        gpu_splat_data_.front().y,
        gpu_splat_data_.front().z);
    QVector3D maximum = minimum;

    for (const GpuSplatData& point : gpu_splat_data_) {
        minimum.setX(std::min(minimum.x(), point.x));
        minimum.setY(std::min(minimum.y(), point.y));
        minimum.setZ(std::min(minimum.z(), point.z));
        maximum.setX(std::max(maximum.x(), point.x));
        maximum.setY(std::max(maximum.y(), point.y));
        maximum.setZ(std::max(maximum.z(), point.z));
    }

    const QVector3D center = (minimum + maximum) * 0.5f;
    const QVector3D extent = maximum - minimum;
    const float largest_extent = std::max({extent.x(), extent.y(), extent.z()});

    // Map the largest dimension to 1.6 world units. The remaining 20% margin
    // keeps points away from the viewport edges with the default camera/FOV.
    constexpr float kTargetExtent = 1.6f;
    constexpr float kMinimumExtent = 1.0e-6f;
    const float scale = largest_extent > kMinimumExtent
        ? kTargetExtent / largest_extent
        : 1.0f;

    // QMatrix4x4 post-multiplies these operations, producing Scale *
    // Translation. A point is therefore centered first and scaled second.
    model_matrix_.scale(scale);
    model_matrix_.translate(-center);
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

    const QMatrix4x4 model_view_projection =
        projection_matrix_ * view_matrix_ * interaction_matrix_ * model_matrix_;
    point_shader_program_->setUniformValue("u_mvp", model_view_projection);
    point_shader_program_->setUniformValue(
        "u_viewport_size", QVector2D(width(), height()));
    point_shader_program_->setUniformValue(
        "u_square_half_size_pixels", kSquareHalfSizePixels);

    {
        QOpenGLVertexArrayObject::Binder vao_binder(&point_vao_);
        glDrawArraysInstanced(
            GL_TRIANGLE_STRIP, 0, 4,
            static_cast<GLsizei>(uploaded_point_count_));
    }

    point_shader_program_->release();
}

void OpenGLWidget::resizeGL(int width, int height)
{
    glViewport(0, 0, width, height);

    projection_matrix_.setToIdentity();
    projection_matrix_.perspective(
        45.0f,
        static_cast<float>(width) / static_cast<float>(std::max(1, height)),
        0.01f,
        1000.0f);
}

void OpenGLWidget::paintGL()
{
    glClearColor(background_color_.redF(), background_color_.greenF(),
                 background_color_.blueF(), 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glEnable(GL_DEPTH_TEST);
    renderGaussianPoints();
    glDisable(GL_DEPTH_TEST);

    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing);

    paintDemoScene(painter);
    OrientationGizmo::paint(painter, size(), yaw_degrees_, pitch_degrees_);
}

void OpenGLWidget::paintDemoScene(QPainter& painter)
{

    painter.setPen(QPen(QColor(255, 255, 255, 22), 1.0));
    constexpr int grid_spacing = 40;
    for (int x = width() / 2 % grid_spacing; x < width(); x += grid_spacing)
        painter.drawLine(x, 0, x, height());
    for (int y = height() / 2 % grid_spacing; y < height(); y += grid_spacing)
        painter.drawLine(0, y, width(), y);

    painter.setPen(QColor(215, 225, 240));
    painter.drawText(18, 28, "Qt OpenGL point-cloud viewport");
}
