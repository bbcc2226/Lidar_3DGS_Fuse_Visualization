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
constexpr int kOpacityAttribute = 3;
constexpr int kScaleAttribute = 4;
constexpr int kRotationAttribute = 5;
constexpr float kFixedSplatHalfSizePixels = 3.0f;

constexpr float kQuadCorners[] = {
    -1.0f, -1.0f,
     1.0f, -1.0f,
    -1.0f,  1.0f,
     1.0f,  1.0f,
};

constexpr char kSplatVertexShader[] = R"GLSL(
#version 330 core

layout(location = 0) in vec2 in_corner;
layout(location = 1) in vec3 in_position;
layout(location = 2) in vec3 in_color;
layout(location = 3) in float in_opacity;
layout(location = 4) in vec3 in_scale;
layout(location = 5) in vec4 in_rotation;

uniform mat4 u_model_view;
uniform mat4 u_projection;
uniform vec2 u_viewport_size;
uniform float u_fixed_half_size_pixels;
uniform float u_point_cloud_scale;
uniform vec2 u_focal_pixels;
uniform bool u_use_trained_scale;

out vec3 vertex_color;
out vec2 splat_coordinate;
flat out float splat_opacity;

vec4 normalizedQuaternion(vec4 raw_quaternion)
{
    float norm_squared = dot(raw_quaternion, raw_quaternion);
    return norm_squared > 1.0e-12
        ? raw_quaternion * inversesqrt(norm_squared)
        : vec4(1.0, 0.0, 0.0, 0.0);
}

mat3 rotationMatrixFromQuaternion(vec4 quaternion_wxyz)
{
    float w = quaternion_wxyz.x;
    float x = quaternion_wxyz.y;
    float y = quaternion_wxyz.z;
    float z = quaternion_wxyz.w;

    float r00 = 1.0 - 2.0 * (y * y + z * z);
    float r01 = 2.0 * (x * y - w * z);
    float r02 = 2.0 * (x * z + w * y);
    float r10 = 2.0 * (x * y + w * z);
    float r11 = 1.0 - 2.0 * (x * x + z * z);
    float r12 = 2.0 * (y * z - w * x);
    float r20 = 2.0 * (x * z - w * y);
    float r21 = 2.0 * (y * z + w * x);
    float r22 = 1.0 - 2.0 * (x * x + y * y);

    // GLSL matrix constructors take columns, not rows.
    return mat3(
        vec3(r00, r10, r20),
        vec3(r01, r11, r21),
        vec3(r02, r12, r22));
}

void main()
{
    vec4 quaternion_wxyz = normalizedQuaternion(in_rotation);
    vec4 center_view = u_model_view * vec4(in_position, 1.0);
    vec4 center_clip = u_projection * center_view;

    vec2 half_size_pixels = vec2(u_fixed_half_size_pixels);
    if (u_use_trained_scale) {
        vec3 activated_scale_3d = exp(in_scale);
        vec3 variance = activated_scale_3d * activated_scale_3d;
        mat3 scale_covariance = mat3(
            variance.x, 0.0, 0.0,
            0.0, variance.y, 0.0,
            0.0, 0.0, variance.z);
        mat3 rotation = rotationMatrixFromQuaternion(quaternion_wxyz);
        mat3 covariance_3d =
            rotation * scale_covariance * transpose(rotation);
        mat3 model_view_linear = mat3(u_model_view);
        mat3 covariance_view = model_view_linear * covariance_3d
            * transpose(model_view_linear);

        vec2 activated_scale = activated_scale_3d.xy;
        float camera_distance = max(-center_view.z, 0.01);
        mat3x2 projection_jacobian = mat3x2(
            vec2(u_focal_pixels.x / camera_distance, 0.0),
            vec2(0.0, u_focal_pixels.y / camera_distance),
            vec2(
                u_focal_pixels.x * center_view.x /
                    (camera_distance * camera_distance),
                u_focal_pixels.y * center_view.y /
                    (camera_distance * camera_distance)));
        mat2 covariance_screen = projection_jacobian * covariance_view
            * transpose(projection_jacobian);

        vec2 sigma_pixels = u_focal_pixels.y * activated_scale
            * u_point_cloud_scale / camera_distance;
        half_size_pixels = clamp(
            3.0 * sigma_pixels, vec2(1.0), vec2(256.0));
    }

    vec2 offset_ndc = in_corner * half_size_pixels * 2.0
        / u_viewport_size;
    gl_Position = center_clip;
    gl_Position.xy += offset_ndc * center_clip.w;
    vertex_color = in_color;
    splat_coordinate = in_corner;
    splat_opacity = in_opacity;
}
)GLSL";

constexpr char kSplatFragmentShader[] = R"GLSL(
#version 330 core

in vec3 vertex_color;
in vec2 splat_coordinate;
flat in float splat_opacity;
out vec4 fragment_color;

void main()
{
    vec2 gaussian_position = splat_coordinate * 3.0;
    float radius_squared = dot(gaussian_position, gaussian_position);
    if (radius_squared > 9.0)
        discard;

    float opacity = 1.0 / (1.0 + exp(-splat_opacity));
    float gaussian_weight = exp(-0.5 * radius_squared);
    float alpha = opacity * gaussian_weight;
    fragment_color = vec4(vertex_color * alpha, alpha);
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
        destroySplatResources();
        doneCurrent();
    }
}

bool OpenGLWidget::isSplatShaderReady() const
{
    return splat_shader_program_ && splat_shader_program_->isLinked();
}

void OpenGLWidget::setBackgroundColor(const QColor& color)
{
    if (color.isValid()) {
        background_color_ = color;
        update();
    }
}

void OpenGLWidget::setGaussianPoints(
    const std::vector<GaussianPoint>& points, bool has_trained_scale)
{
    has_trained_scale_ = has_trained_scale;
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
        gpu_splat.scale_x = point.scale[0];
        gpu_splat.scale_y = point.scale[1];
        gpu_splat.scale_z = point.scale[2];
        gpu_splat.rotation_w = point.rotation[0];
        gpu_splat.rotation_x = point.rotation[1];
        gpu_splat.rotation_y = point.rotation[2];
        gpu_splat.rotation_z = point.rotation[3];
        gpu_splat_data_.push_back(gpu_splat);
    }
    fitPointCloudToView();

    // Calls made before initializeGL() are retained and uploaded when the
    // context becomes available.
    if (isValid()) {
        makeCurrent();
        uploadSplatBuffer();
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

    if (createSplatBuffers()) {
        uploadSplatBuffer();
    }
    createSplatShaderProgram();
}

bool OpenGLWidget::createSplatShaderProgram()
{
    splat_shader_program_ = std::make_unique<QOpenGLShaderProgram>();
    if (!splat_shader_program_->addShaderFromSourceCode(
            QOpenGLShader::Vertex, kSplatVertexShader)) {
        qWarning("Gaussian splat vertex shader compilation failed: %s",
                 qPrintable(splat_shader_program_->log()));
        splat_shader_program_.reset();
        return false;
    }

    if (!splat_shader_program_->addShaderFromSourceCode(
            QOpenGLShader::Fragment, kSplatFragmentShader)) {
        qWarning("Gaussian splat fragment shader compilation failed: %s",
                 qPrintable(splat_shader_program_->log()));
        splat_shader_program_.reset();
        return false;
    }

    if (!splat_shader_program_->link()) {
        qWarning("Gaussian splat shader link failed: %s",
                 qPrintable(splat_shader_program_->log()));
        splat_shader_program_.reset();
        return false;
    }

    return true;
}

bool OpenGLWidget::createSplatBuffers()
{
    static_assert(std::is_standard_layout<GpuSplatData>::value,
                  "GpuSplatData must be an interleaved vertex type.");
    static_assert(sizeof(GpuSplatData) == 48,
                  "GpuSplatData layout must match the configured attributes.");
    static_assert(offsetof(GpuSplatData, scale_x) == 20,
                  "GpuSplatData scale offset must remain stable.");
    static_assert(offsetof(GpuSplatData, rotation_w) == 32,
                  "GpuSplatData rotation offset must remain stable.");

    if (!splat_vao_.isCreated() && !splat_vao_.create()) {
        qWarning("Failed to create Gaussian splat VAO.");
        return false;
    }
    if (!quad_vbo_.isCreated() && !quad_vbo_.create()) {
        qWarning("Failed to create shared quad VBO.");
        return false;
    }
    if (!splat_instance_vbo_.isCreated() && !splat_instance_vbo_.create()) {
        qWarning("Failed to create Gaussian splat instance VBO.");
        return false;
    }

    QOpenGLVertexArrayObject::Binder vao_binder(&splat_vao_);
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

    if (!splat_instance_vbo_.bind()) {
        qWarning("Failed to bind Gaussian splat instance VBO.");
        return false;
    }
    splat_instance_vbo_.setUsagePattern(QOpenGLBuffer::StaticDraw);

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

    glEnableVertexAttribArray(kOpacityAttribute);
    glVertexAttribPointer(
        kOpacityAttribute, 1, GL_FLOAT, GL_FALSE, sizeof(GpuSplatData),
        reinterpret_cast<const void*>(offsetof(GpuSplatData, opacity)));
    glVertexAttribDivisor(kOpacityAttribute, 1);

    glEnableVertexAttribArray(kScaleAttribute);
    glVertexAttribPointer(
        kScaleAttribute, 3, GL_FLOAT, GL_FALSE, sizeof(GpuSplatData),
        reinterpret_cast<const void*>(offsetof(GpuSplatData, scale_x)));
    glVertexAttribDivisor(kScaleAttribute, 1);

    glEnableVertexAttribArray(kRotationAttribute);
    glVertexAttribPointer(
        kRotationAttribute, 4, GL_FLOAT, GL_FALSE, sizeof(GpuSplatData),
        reinterpret_cast<const void*>(offsetof(GpuSplatData, rotation_w)));
    glVertexAttribDivisor(kRotationAttribute, 1);

    splat_instance_vbo_.release();
    return true;
}

bool OpenGLWidget::uploadSplatBuffer()
{
    uploaded_splat_count_ = 0;
    if (!splat_instance_vbo_.isCreated()) return false;

    const std::size_t byte_count =
        gpu_splat_data_.size() * sizeof(GpuSplatData);
    if (byte_count > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        qWarning("Gaussian splat buffer exceeds QOpenGLBuffer's allocation limit.");
        return false;
    }

    if (!splat_instance_vbo_.bind()) {
        qWarning("Failed to bind Gaussian splat VBO for upload.");
        return false;
    }
    splat_instance_vbo_.allocate(
        gpu_splat_data_.empty() ? nullptr : gpu_splat_data_.data(),
        static_cast<int>(byte_count));
    splat_instance_vbo_.release();
    uploaded_splat_count_ = gpu_splat_data_.size();
    return true;
}

void OpenGLWidget::destroySplatResources()
{
    uploaded_splat_count_ = 0;
    splat_shader_program_.reset();
    splat_instance_vbo_.destroy();
    quad_vbo_.destroy();
    splat_vao_.destroy();
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
    point_cloud_scale_ = 1.0f;
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
    point_cloud_scale_ = largest_extent > kMinimumExtent
        ? kTargetExtent / largest_extent
        : 1.0f;

    // QMatrix4x4 post-multiplies these operations, producing Scale *
    // Translation. A point is therefore centered first and scaled second.
    model_matrix_.scale(point_cloud_scale_);
    model_matrix_.translate(-center);
}

void OpenGLWidget::renderGaussianSplats()
{
    if (uploaded_splat_count_ == 0 || !isSplatShaderReady() ||
        !splat_vao_.isCreated()) {
        return;
    }

    if (!splat_shader_program_->bind()) {
        qWarning("Failed to bind Gaussian splat shader for drawing.");
        return;
    }

    const QMatrix4x4 model_view =
        view_matrix_ * interaction_matrix_ * model_matrix_;
    splat_shader_program_->setUniformValue("u_model_view", model_view);
    splat_shader_program_->setUniformValue("u_projection", projection_matrix_);
    splat_shader_program_->setUniformValue(
        "u_viewport_size", QVector2D(width(), height()));
    splat_shader_program_->setUniformValue(
        "u_fixed_half_size_pixels", kFixedSplatHalfSizePixels);
    splat_shader_program_->setUniformValue(
        "u_point_cloud_scale", point_cloud_scale_);
    splat_shader_program_->setUniformValue(
        "u_focal_pixels",
        QVector2D(
            projection_matrix_(0, 0) * static_cast<float>(width()) * 0.5f,
            projection_matrix_(1, 1) * static_cast<float>(height()) * 0.5f));
    splat_shader_program_->setUniformValue(
        "u_use_trained_scale", has_trained_scale_);

    {
        QOpenGLVertexArrayObject::Binder vao_binder(&splat_vao_);
        glDrawArraysInstanced(
            GL_TRIANGLE_STRIP, 0, 4,
            static_cast<GLsizei>(uploaded_splat_count_));
    }

    splat_shader_program_->release();
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
    glDepthMask(GL_FALSE);
    glEnable(GL_BLEND);
    glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
    renderGaussianSplats();
    glDisable(GL_BLEND);
    glDepthMask(GL_TRUE);
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
