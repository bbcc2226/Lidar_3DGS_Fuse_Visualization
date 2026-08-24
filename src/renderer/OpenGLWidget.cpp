#include "OpenGLWidget.h"

#include "OrientationGizmo.h"

#include <QOpenGLContext>
#include <QOpenGLShader>
#include <QOpenGLShaderProgram>
#include <QPainter>
#include <QQuaternion>
#include <QVector2D>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>
#include <numeric>
#include <type_traits>

namespace
{
constexpr int kCornerAttribute = 0;
constexpr int kPositionAttribute = 1;
constexpr int kColorAttribute = 2;
constexpr int kOpacityAttribute = 3;
constexpr int kScaleAttribute = 4;
constexpr int kRotationAttribute = 5;
constexpr int kShDcAttribute = 6;
constexpr int kShL1FirstAttribute = 7;
constexpr float kFixedSplatHalfSizePixels = 3.0f;
constexpr float kInteractiveSortAngleDegrees = 0.75f;

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
layout(location = 6) in vec3 in_sh_dc;
layout(location = 7) in vec3 in_sh_l1_0;
layout(location = 8) in vec3 in_sh_l1_1;
layout(location = 9) in vec3 in_sh_l1_2;

uniform mat4 u_model_view;
uniform mat4 u_projection;
uniform vec2 u_viewport_size;
uniform float u_fixed_half_size_pixels;
uniform vec2 u_focal_pixels;
uniform bool u_use_trained_scale;
uniform bool u_use_sh_dc;
uniform int u_sh_degree;
uniform vec3 u_camera_position_model;

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

    // The camera looks down view-space -Z. Reject centers on or behind the
    // camera instead of clamping their projection depth to a tiny value,
    // which would otherwise create very large, misplaced splats.
    if (center_view.z >= -0.01) {
        gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
        vertex_color = in_color;
        splat_coordinate = in_corner;
        splat_opacity = 0.0;
        return;
    }

    vec4 center_clip = u_projection * center_view;

    vec2 half_size_pixels = vec2(u_fixed_half_size_pixels);
    vec2 major_axis = vec2(1.0, 0.0);
    vec2 minor_axis = vec2(0.0, 1.0);
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
        covariance_screen[0][0] += 0.3;
        covariance_screen[1][1] += 0.3;

        float covariance_xx = covariance_screen[0][0];
        float covariance_xy = 0.5 *
            (covariance_screen[0][1] + covariance_screen[1][0]);
        float covariance_yy = covariance_screen[1][1];
        float trace = covariance_xx + covariance_yy;
        float difference = covariance_xx - covariance_yy;
        float discriminant = sqrt(max(
            difference * difference +
                4.0 * covariance_xy * covariance_xy,
            0.0));
        vec2 eigenvalues = max(
            vec2(
                0.5 * (trace + discriminant),
                0.5 * (trace - discriminant)),
            vec2(1.0e-6));
        vec2 sigma_axes_pixels = sqrt(eigenvalues);

        vec2 major_candidate_a = vec2(
            covariance_xy, eigenvalues.x - covariance_xx);
        vec2 major_candidate_b = vec2(
            eigenvalues.x - covariance_yy, covariance_xy);
        major_axis = dot(major_candidate_a, major_candidate_a) >
                dot(major_candidate_b, major_candidate_b)
            ? major_candidate_a
            : major_candidate_b;
        float major_axis_length_squared = dot(major_axis, major_axis);
        major_axis = major_axis_length_squared > 1.0e-12
            ? major_axis * inversesqrt(major_axis_length_squared)
            : vec2(1.0, 0.0);
        minor_axis = vec2(-major_axis.y, major_axis.x);

        half_size_pixels = clamp(
            3.0 * sigma_axes_pixels, vec2(1.0), vec2(256.0));
    }

    vec2 center_ndc = center_clip.xy / center_clip.w;
    // Convert the rotated quad's half-extents to an axis-aligned screen-space
    // bound. Cull only when the complete bound is outside the viewport; an
    // off-screen center can still own an ellipse that overlaps visible pixels.
    vec2 extent_pixels =
        abs(major_axis) * half_size_pixels.x +
        abs(minor_axis) * half_size_pixels.y;
    vec2 extent_ndc = extent_pixels * 2.0 / u_viewport_size;
    if (center_ndc.x + extent_ndc.x < -1.0 ||
        center_ndc.x - extent_ndc.x > 1.0 ||
        center_ndc.y + extent_ndc.y < -1.0 ||
        center_ndc.y - extent_ndc.y > 1.0) {
        gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
        vertex_color = in_color;
        splat_coordinate = in_corner;
        splat_opacity = 0.0;
        return;
    }

    vec2 offset_pixels =
        major_axis * in_corner.x * half_size_pixels.x +
        minor_axis * in_corner.y * half_size_pixels.y;
    vec2 offset_ndc = offset_pixels * 2.0 / u_viewport_size;
    gl_Position = center_clip;
    gl_Position.xy += offset_ndc * center_clip.w;
    const float sh_c0 = 0.2820947918;
    if (u_use_sh_dc) {
        vec3 evaluated_color = sh_c0 * in_sh_dc;
        if (u_sh_degree >= 1) {
            const float sh_c1 = 0.4886025119;
            vec3 direction = normalize(in_position - u_camera_position_model);
            evaluated_color +=
                (-sh_c1 * direction.y) * in_sh_l1_0 +
                ( sh_c1 * direction.z) * in_sh_l1_1 +
                (-sh_c1 * direction.x) * in_sh_l1_2;
        }
        vertex_color = max(vec3(0.5) + evaluated_color, vec3(0.0));
    } else {
        vertex_color = in_color;
    }
    splat_coordinate = in_corner;
    splat_opacity = in_opacity;
}
)GLSL";

constexpr char kSplatFragmentShader[] = R"GLSL(
#version 330 core

in vec3 vertex_color;
in vec2 splat_coordinate;
flat in float splat_opacity;
uniform float u_alpha_density_scale;
out vec4 fragment_color;

void main()
{
    vec2 gaussian_position = splat_coordinate * 3.0;
    float radius_squared = dot(gaussian_position, gaussian_position);
    if (radius_squared > 9.0)
        discard;

    float opacity = 1.0 / (1.0 + exp(-splat_opacity));
    float gaussian_weight = exp(-0.5 * radius_squared);
    float base_alpha = clamp(opacity * gaussian_weight, 0.0, 0.999);
    // A low-detail interaction frame keeps one of every four splats. Treat
    // each retained alpha sample as representative of the omitted density so
    // the preview does not become four times darker.
    float alpha = 1.0 - pow(1.0 - base_alpha, u_alpha_density_scale);
    fragment_color = vec4(vertex_color * alpha, alpha);
}
)GLSL";
} // namespace

OpenGLWidget::OpenGLWidget(QWidget* parent)
    : QOpenGLWidget(parent)
{
    scene_alignment_matrix_.setToIdentity();
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
    const std::vector<GaussianPoint>& points, bool has_trained_scale,
    bool has_sh_dc, int sh_degree)
{
    has_trained_scale_ = has_trained_scale;
    has_sh_dc_ = has_sh_dc;
    sh_degree_ = std::clamp(sh_degree, 0, 3);
    std::vector<GpuSplatData>().swap(source_gpu_splat_data_);
    source_gpu_splat_data_.reserve(points.size());
    std::vector<GpuSplatData>().swap(gpu_splat_data_);
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
        gpu_splat.sh_dc_r = point.sh_dc[0];
        gpu_splat.sh_dc_g = point.sh_dc[1];
        gpu_splat.sh_dc_b = point.sh_dc[2];
        if (sh_degree_ >= 1) {
            const std::size_t coefficients_per_channel =
                static_cast<std::size_t>((sh_degree_ + 1) * (sh_degree_ + 1) - 1);
            const auto coefficient = [&point, coefficients_per_channel](
                                         std::size_t channel,
                                         std::size_t basis) {
                return point.sh_rest[channel * coefficients_per_channel + basis];
            };
            gpu_splat.sh_l1_0_r = coefficient(0, 0);
            gpu_splat.sh_l1_0_g = coefficient(1, 0);
            gpu_splat.sh_l1_0_b = coefficient(2, 0);
            gpu_splat.sh_l1_1_r = coefficient(0, 1);
            gpu_splat.sh_l1_1_g = coefficient(1, 1);
            gpu_splat.sh_l1_1_b = coefficient(2, 1);
            gpu_splat.sh_l1_2_r = coefficient(0, 2);
            gpu_splat.sh_l1_2_g = coefficient(1, 2);
            gpu_splat.sh_l1_2_b = coefficient(2, 2);
        }
        source_gpu_splat_data_.push_back(gpu_splat);
    }
    splat_sort_indices_.resize(points.size());
    splat_sort_scratch_.resize(points.size());
    std::iota(splat_sort_indices_.begin(), splat_sort_indices_.end(), 0U);
    splat_sort_depths_.resize(points.size());
    splat_sort_dirty_ = true;
    fitPointCloudToView();

    // Calls made before initializeGL() are retained and uploaded when the
    // context becomes available.
    uploaded_splat_count_ = 0;
    if (points.empty() && isValid()) {
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

void OpenGLWidget::setZUpGizmo(bool enabled)
{
    z_up_gizmo_ = enabled;
    update();
}

void OpenGLWidget::finalizeInteractionSort()
{
    splat_sort_dirty_ = true;
    update();
}

std::optional<QVector3D> OpenGLWidget::pickGaussianAt(
    const QPoint& position) const
{
    if (source_gpu_splat_data_.empty() || width() <= 0 || height() <= 0) {
        return std::nullopt;
    }

    const QMatrix4x4 model_view =
        view_matrix_ * interaction_matrix_ * scene_alignment_matrix_ *
        model_matrix_;
    constexpr float kPickRadiusPixels = 12.0f;
    float best_distance_squared = kPickRadiusPixels * kPickRadiusPixels;
    std::optional<QVector3D> best_point;
    for (const GpuSplatData& splat : source_gpu_splat_data_) {
        const QVector3D raw_position(splat.x, splat.y, splat.z);
        const QVector4D view_position =
            model_view * QVector4D(raw_position, 1.0f);
        if (view_position.z() >= -0.01f) continue;
        const QVector4D clip_position = projection_matrix_ * view_position;
        if (clip_position.w() <= 0.0f) continue;
        const float inverse_w = 1.0f / clip_position.w();
        const QPointF screen_position(
            (clip_position.x() * inverse_w * 0.5f + 0.5f) * width(),
            (0.5f - clip_position.y() * inverse_w * 0.5f) * height());
        const float dx = static_cast<float>(screen_position.x() - position.x());
        const float dy = static_cast<float>(screen_position.y() - position.y());
        const float distance_squared = dx * dx + dy * dy;
        if (distance_squared < best_distance_squared) {
            best_distance_squared = distance_squared;
            best_point = model_matrix_.map(raw_position);
        }
    }
    return best_point;
}

bool OpenGLWidget::alignFloorFromPoints(
    const std::vector<QVector3D>& points)
{
    if (points.size() != 3) return false;
    QVector3D normal = QVector3D::crossProduct(
        points[1] - points[0], points[2] - points[0]);
    if (normal.lengthSquared() < 1.0e-10f) return false;
    normal.normalize();
    if (normal.z() < 0.0f) normal = -normal;

    scene_alignment_matrix_.setToIdentity();
    scene_alignment_matrix_.rotate(QQuaternion::rotationTo(
        normal, QVector3D(0.0f, 0.0f, 1.0f)));
    floor_selection_points_ = points;
    splat_sort_dirty_ = true;
    update();
    return true;
}

void OpenGLWidget::clearFloorAlignment()
{
    scene_alignment_matrix_.setToIdentity();
    floor_selection_points_.clear();
    splat_sort_dirty_ = true;
    update();
}

void OpenGLWidget::setFloorSelectionPoints(
    const std::vector<QVector3D>& points)
{
    floor_selection_points_ = points;
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
    static_assert(sizeof(GpuSplatData) == 96,
                  "GpuSplatData layout must match the configured attributes.");
    static_assert(offsetof(GpuSplatData, scale_x) == 20,
                  "GpuSplatData scale offset must remain stable.");
    static_assert(offsetof(GpuSplatData, rotation_w) == 32,
                  "GpuSplatData rotation offset must remain stable.");
    static_assert(offsetof(GpuSplatData, sh_dc_r) == 48,
                  "GpuSplatData SH DC offset must remain stable.");
    static_assert(offsetof(GpuSplatData, sh_l1_0_r) == 60,
                  "GpuSplatData degree-one SH offset must remain stable.");

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
    // The instance order changes with the camera for back-to-front alpha
    // compositing, so this buffer is rewritten while interacting with a scene.
    splat_instance_vbo_.setUsagePattern(QOpenGLBuffer::DynamicDraw);

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

    glEnableVertexAttribArray(kShDcAttribute);
    glVertexAttribPointer(
        kShDcAttribute, 3, GL_FLOAT, GL_FALSE, sizeof(GpuSplatData),
        reinterpret_cast<const void*>(offsetof(GpuSplatData, sh_dc_r)));
    glVertexAttribDivisor(kShDcAttribute, 1);

    for (int basis = 0; basis < 3; ++basis) {
        const int attribute = kShL1FirstAttribute + basis;
        glEnableVertexAttribArray(attribute);
        glVertexAttribPointer(
            attribute, 3, GL_FLOAT, GL_FALSE, sizeof(GpuSplatData),
            reinterpret_cast<const void*>(
                offsetof(GpuSplatData, sh_l1_0_r) +
                static_cast<std::size_t>(basis) * 3 * sizeof(float)));
        glVertexAttribDivisor(attribute, 1);
    }

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
    if (byte_count == allocated_splat_bytes_ && byte_count > 0) {
        // Preserve the existing GPU allocation while only its sorted contents
        // change. Reallocating a large buffer on every mouse event can cause
        // driver stalls and transient GPU-memory exhaustion.
        splat_instance_vbo_.write(
            0, gpu_splat_data_.data(), static_cast<int>(byte_count));
    } else {
        splat_instance_vbo_.allocate(
            gpu_splat_data_.empty() ? nullptr : gpu_splat_data_.data(),
            static_cast<int>(byte_count));
        allocated_splat_bytes_ = byte_count;
    }
    splat_instance_vbo_.release();
    uploaded_splat_count_ = gpu_splat_data_.size();
    return true;
}

bool OpenGLWidget::sortAndUploadSplats(const QMatrix4x4& model_view)
{
    const QVector3D depth_direction = QVector3D(
        model_view(2, 0), model_view(2, 1), model_view(2, 2)).normalized();
    // Translation adds the same value to every depth and cannot change their
    // order. Avoid sorting and uploading while the user only pans or dollies.
    const float sort_angle_radians =
        kInteractiveSortAngleDegrees * 3.14159265f / 180.0f;
    const float direction_threshold_squared =
        2.0f - 2.0f * std::cos(sort_angle_radians);
    if (!splat_sort_dirty_ &&
        (depth_direction - last_sort_depth_direction_).lengthSquared() <
            direction_threshold_squared) {
        return true;
    }

    // OpenGL view space looks down -Z. More-negative Z values are farther
    // from the camera and must be blended first. Compute each transformed
    // depth once, then sort 32-bit indices instead of moving 96-byte records.
    const float depth_x = model_view(2, 0);
    const float depth_y = model_view(2, 1);
    const float depth_z = model_view(2, 2);
    const float depth_offset = model_view(2, 3);
    const std::size_t sorted_count = source_gpu_splat_data_.size();
    splat_sort_indices_.resize(sorted_count);
    gpu_splat_data_.resize(sorted_count);
    for (std::size_t sorted_index = 0;
         sorted_index < sorted_count; ++sorted_index) {
        const std::size_t index = sorted_index;
        splat_sort_indices_[sorted_index] = static_cast<std::uint32_t>(index);
        const GpuSplatData& splat = source_gpu_splat_data_[index];
        splat_sort_depths_[index] =
            depth_x * splat.x + depth_y * splat.y +
            depth_z * splat.z + depth_offset;
    }

    const auto depth_key = [this](std::uint32_t index) {
        std::uint32_t bits = 0;
        static_assert(sizeof(bits) == sizeof(float));
        std::memcpy(&bits, &splat_sort_depths_[index], sizeof(bits));
        const std::uint32_t sign_mask = static_cast<std::uint32_t>(
            -static_cast<std::int32_t>(bits >> 31));
        return bits ^ (sign_mask | 0x80000000U);
    };
    splat_sort_scratch_.resize(sorted_count);
    for (unsigned pass = 0; pass < 4; ++pass) {
        std::array<std::size_t, 256> offsets{};
        const unsigned shift = pass * 8;
        for (std::uint32_t index : splat_sort_indices_) {
            ++offsets[(depth_key(index) >> shift) & 0xffU];
        }
        std::size_t running_offset = 0;
        for (std::size_t& offset : offsets) {
            const std::size_t count = offset;
            offset = running_offset;
            running_offset += count;
        }
        for (std::uint32_t index : splat_sort_indices_) {
            const std::uint32_t byte = (depth_key(index) >> shift) & 0xffU;
            splat_sort_scratch_[offsets[byte]++] = index;
        }
        splat_sort_indices_.swap(splat_sort_scratch_);
    }
    for (std::size_t destination = 0;
         destination < splat_sort_indices_.size(); ++destination) {
        gpu_splat_data_[destination] =
            source_gpu_splat_data_[splat_sort_indices_[destination]];
    }

    if (!uploadSplatBuffer()) return false;
    last_sort_depth_direction_ = depth_direction;
    splat_sort_dirty_ = false;
    return true;
}

void OpenGLWidget::destroySplatResources()
{
    uploaded_splat_count_ = 0;
    allocated_splat_bytes_ = 0;
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
    if (source_gpu_splat_data_.empty()) return;

    QVector3D minimum(
        source_gpu_splat_data_.front().x,
        source_gpu_splat_data_.front().y,
        source_gpu_splat_data_.front().z);
    QVector3D maximum = minimum;

    for (const GpuSplatData& point : source_gpu_splat_data_) {
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
    if (source_gpu_splat_data_.empty() || !isSplatShaderReady() ||
        !splat_vao_.isCreated()) {
        return;
    }

    const QMatrix4x4 model_view =
        view_matrix_ * interaction_matrix_ * scene_alignment_matrix_ *
        model_matrix_;
    if (!sortAndUploadSplats(model_view)) {
        qWarning("Failed to upload depth-sorted Gaussian splats.");
        return;
    }

    if (!splat_shader_program_->bind()) {
        qWarning("Failed to bind Gaussian splat shader for drawing.");
        return;
    }
    splat_shader_program_->setUniformValue("u_model_view", model_view);
    splat_shader_program_->setUniformValue("u_projection", projection_matrix_);
    splat_shader_program_->setUniformValue(
        "u_viewport_size", QVector2D(width(), height()));
    splat_shader_program_->setUniformValue(
        "u_fixed_half_size_pixels", kFixedSplatHalfSizePixels);
    splat_shader_program_->setUniformValue(
        "u_focal_pixels",
        QVector2D(
            projection_matrix_(0, 0) * static_cast<float>(width()) * 0.5f,
            projection_matrix_(1, 1) * static_cast<float>(height()) * 0.5f));
    splat_shader_program_->setUniformValue(
        "u_use_trained_scale", has_trained_scale_);
    splat_shader_program_->setUniformValue("u_use_sh_dc", has_sh_dc_);
    splat_shader_program_->setUniformValue("u_sh_degree", sh_degree_);
    splat_shader_program_->setUniformValue("u_alpha_density_scale", 1.0f);
    bool inverse_ok = false;
    const QMatrix4x4 inverse_model_view = model_view.inverted(&inverse_ok);
    const QVector3D camera_position_model = inverse_ok
        ? inverse_model_view.map(QVector3D(0.0f, 0.0f, 0.0f))
        : QVector3D(0.0f, 0.0f, 0.0f);
    splat_shader_program_->setUniformValue(
        "u_camera_position_model", camera_position_model);

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
    const QMatrix4x4 marker_mvp =
        projection_matrix_ * view_matrix_ * interaction_matrix_ *
        scene_alignment_matrix_;
    painter.setPen(QPen(QColor(255, 255, 255), 2.0f));
    painter.setBrush(QColor(250, 204, 21, 220));
    for (std::size_t index = 0; index < floor_selection_points_.size(); ++index) {
        const QVector4D clip = marker_mvp *
            QVector4D(floor_selection_points_[index], 1.0f);
        if (clip.w() <= 0.0f) continue;
        const QPointF screen(
            (clip.x() / clip.w() * 0.5f + 0.5f) * width(),
            (0.5f - clip.y() / clip.w() * 0.5f) * height());
        painter.drawEllipse(screen, 7.0, 7.0);
        painter.drawText(screen + QPointF(10.0, -8.0),
                         QString::number(index + 1));
    }
    OrientationGizmo::paint(
        painter, size(), yaw_degrees_, pitch_degrees_, z_up_gizmo_);
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
