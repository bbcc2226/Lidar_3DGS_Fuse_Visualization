#include "OpenGLWidget.h"

#include "OrientationGizmo.h"

#include <QOpenGLContext>
#include <QOpenGLShader>
#include <QOpenGLShaderProgram>
#include <QPainter>
#include <QPainterPath>
#include <QQuaternion>
#include <QSurfaceFormat>
#include <QtMath>
#include <QtGlobal>
#include <QVector2D>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>
#include <numeric>
#include <thread>
#include <type_traits>

namespace
{
constexpr int kCornerAttribute = 0;
constexpr int kPositionAttribute = 1;
constexpr int kColorAttribute = 2;
constexpr int kOpacityAttribute = 3;
constexpr int kScaleAttribute = 4;
constexpr int kRotationAttribute = 5;
constexpr int kGaussianIndexAttribute = 6;
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

uniform mat4 u_model_view;
uniform mat4 u_projection;
uniform vec2 u_viewport_size;
uniform float u_fixed_half_size_pixels;
uniform vec2 u_focal_pixels;
uniform bool u_use_trained_scale;
uniform bool u_use_sh_dc;
uniform int u_sh_degree;
uniform vec3 u_camera_position_model;
uniform samplerBuffer u_sh_coefficients;
uniform samplerBuffer u_splat_data;
uniform usamplerBuffer u_sorted_indices;
uniform float u_exposure_ev;
uniform float u_low_pass_variance;
uniform float u_maximum_splat_half_size;
uniform bool u_highlight_large_splats;
uniform bool u_suppress_oversized_splats;

out vec3 vertex_color;
flat out float splat_opacity;
flat out float splat_opacity_scale;
flat out vec2 splat_center_pixels;
flat out vec3 splat_conic;

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
    uint gaussian_index = texelFetch(
        u_sorted_indices, gl_InstanceID).r;
    int splat_offset = int(gaussian_index) * 4;
    vec4 position_opacity = texelFetch(u_splat_data, splat_offset);
    vec4 color_scale_x = texelFetch(u_splat_data, splat_offset + 1);
    vec4 scale_yz_rotation_wx = texelFetch(u_splat_data, splat_offset + 2);
    vec4 rotation_yz = texelFetch(u_splat_data, splat_offset + 3);
    vec3 splat_position = position_opacity.xyz;
    float raw_opacity = position_opacity.w;
    vec3 splat_color = color_scale_x.rgb;
    vec3 splat_scale = vec3(
        color_scale_x.w, scale_yz_rotation_wx.xy);
    vec4 splat_rotation = vec4(
        scale_yz_rotation_wx.zw, rotation_yz.xy);

    vec4 quaternion_wxyz = normalizedQuaternion(splat_rotation);
    vec4 center_view = u_model_view * vec4(splat_position, 1.0);

    // The camera looks down view-space -Z. Reject centers on or behind the
    // camera instead of clamping their projection depth to a tiny value,
    // which would otherwise create very large, misplaced splats.
    if (center_view.z >= -0.01) {
        gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
        vertex_color = splat_color;
        splat_opacity = 0.0;
        splat_opacity_scale = 0.0;
        splat_center_pixels = vec2(0.0);
        splat_conic = vec3(1.0, 0.0, 1.0);
        return;
    }

    vec4 center_clip = u_projection * center_view;

    vec2 half_size_pixels = vec2(u_fixed_half_size_pixels);
    vec2 major_axis = vec2(1.0, 0.0);
    vec2 minor_axis = vec2(0.0, 1.0);
    float largest_half_size = u_fixed_half_size_pixels;
    splat_conic = vec3(1.0, 0.0, 1.0);
    splat_opacity_scale = 1.0;
    if (u_use_trained_scale) {
        vec3 activated_scale_3d = exp(splat_scale);
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
        vec2 tan_half_fov = u_viewport_size / (2.0 * u_focal_pixels);
        vec2 projected_ratio = clamp(
            center_view.xy / camera_distance,
            -1.3 * tan_half_fov,
             1.3 * tan_half_fov);
        vec2 jacobian_position = projected_ratio * camera_distance;
        mat3x2 projection_jacobian = mat3x2(
            vec2(u_focal_pixels.x / camera_distance, 0.0),
            vec2(0.0, u_focal_pixels.y / camera_distance),
            vec2(
                u_focal_pixels.x * jacobian_position.x /
                    (camera_distance * camera_distance),
                u_focal_pixels.y * jacobian_position.y /
                    (camera_distance * camera_distance)));
        mat2 covariance_screen = projection_jacobian * covariance_view
            * transpose(projection_jacobian);
        covariance_screen[0][0] += u_low_pass_variance;
        covariance_screen[1][1] += u_low_pass_variance;

        float covariance_xx = covariance_screen[0][0];
        float covariance_xy = 0.5 *
            (covariance_screen[0][1] + covariance_screen[1][0]);
        float covariance_yy = covariance_screen[1][1];
        float covariance_determinant = max(
            covariance_xx * covariance_yy - covariance_xy * covariance_xy,
            1.0e-12);
        splat_conic = vec3(
            covariance_yy / covariance_determinant,
            -covariance_xy / covariance_determinant,
            covariance_xx / covariance_determinant);
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

        // SuperSplat's compact normalized kernel ends at exp(-4), which maps
        // to radius^2 == 8 for exp(-0.5 * radius^2).
        const float compact_radius = 2.8284271247;
        vec2 raw_half_size_pixels = max(
            compact_radius * sigma_axes_pixels, vec2(1.0));
        largest_half_size = max(
            raw_half_size_pixels.x, raw_half_size_pixels.y);
        half_size_pixels = min(
            raw_half_size_pixels, vec2(u_maximum_splat_half_size));
        // If either projected axis is capped, the quad and conic must describe
        // the same ellipse. Keeping the original conic would leave substantial
        // alpha at the smaller quad boundary, exposing its straight triangle
        // and rectangle edges. Rebuild the inverse covariance from the axes
        // actually drawn on screen.
        vec2 displayed_sigma = half_size_pixels / compact_radius;
        vec2 inverse_variance = 1.0 /
            max(displayed_sigma * displayed_sigma, vec2(1.0e-12));
        splat_conic = vec3(
            major_axis.x * major_axis.x * inverse_variance.x +
                minor_axis.x * minor_axis.x * inverse_variance.y,
            major_axis.x * major_axis.y * inverse_variance.x +
                minor_axis.x * minor_axis.y * inverse_variance.y,
            major_axis.y * major_axis.y * inverse_variance.x +
                minor_axis.y * minor_axis.y * inverse_variance.y);
        if (u_suppress_oversized_splats) {
            // Optional diagnostic for extreme projection outliers. Keep this
            // disabled for the reference rendering path: screen size alone is
            // not a reliable indication that a trained Gaussian is invalid.
            float oversized_ratio =
                largest_half_size / u_maximum_splat_half_size;
            splat_opacity_scale *= 1.0 - smoothstep(
                1.0, 2.0, oversized_ratio);
        }
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
        vertex_color = splat_color;
        splat_opacity = 0.0;
        splat_opacity_scale = 0.0;
        splat_center_pixels = vec2(0.0);
        splat_conic = vec3(1.0, 0.0, 1.0);
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
        int sh_offset = int(gaussian_index) * 16;
        vec3 sh_dc = texelFetch(u_sh_coefficients, sh_offset).rgb;
        vec3 evaluated_color = sh_c0 * sh_dc;
        if (u_sh_degree >= 1) {
            const float sh_c1 = 0.4886025119;
            vec3 direction = normalize(splat_position - u_camera_position_model);
            vec3 sh_l1_0 = texelFetch(
                u_sh_coefficients, sh_offset + 1).rgb;
            vec3 sh_l1_1 = texelFetch(
                u_sh_coefficients, sh_offset + 2).rgb;
            vec3 sh_l1_2 = texelFetch(
                u_sh_coefficients, sh_offset + 3).rgb;
            evaluated_color +=
                (-sh_c1 * direction.y) * sh_l1_0 +
                ( sh_c1 * direction.z) * sh_l1_1 +
                (-sh_c1 * direction.x) * sh_l1_2;
            float x = direction.x;
            float y = direction.y;
            float z = direction.z;
            float xx = x * x;
            float yy = y * y;
            float zz = z * z;

            if (u_sh_degree >= 2) {
                evaluated_color +=
                    1.0925484306 * x * y * texelFetch(
                        u_sh_coefficients, sh_offset + 4).rgb +
                   -1.0925484306 * y * z * texelFetch(
                        u_sh_coefficients, sh_offset + 5).rgb +
                    0.3153915653 * (2.0 * zz - xx - yy) * texelFetch(
                        u_sh_coefficients, sh_offset + 6).rgb +
                   -1.0925484306 * x * z * texelFetch(
                        u_sh_coefficients, sh_offset + 7).rgb +
                    0.5462742153 * (xx - yy) * texelFetch(
                        u_sh_coefficients, sh_offset + 8).rgb;
            }

            if (u_sh_degree >= 3) {
                evaluated_color +=
                   -0.5900435899 * y * (3.0 * xx - yy) * texelFetch(
                        u_sh_coefficients, sh_offset + 9).rgb +
                    2.8906114426 * x * y * z * texelFetch(
                        u_sh_coefficients, sh_offset + 10).rgb +
                   -0.4570457995 * y * (4.0 * zz - xx - yy) * texelFetch(
                        u_sh_coefficients, sh_offset + 11).rgb +
                    0.3731763326 * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) *
                        texelFetch(u_sh_coefficients, sh_offset + 12).rgb +
                   -0.4570457995 * x * (4.0 * zz - xx - yy) * texelFetch(
                        u_sh_coefficients, sh_offset + 13).rgb +
                    1.4453057213 * z * (xx - yy) * texelFetch(
                        u_sh_coefficients, sh_offset + 14).rgb +
                   -0.5900435899 * x * (xx - 3.0 * yy) * texelFetch(
                        u_sh_coefficients, sh_offset + 15).rgb;
            }
        }
        vertex_color = max(vec3(0.5) + evaluated_color, vec3(0.0));
    } else {
        vertex_color = splat_color;
    }
    vertex_color *= exp2(u_exposure_ev);
    if (u_highlight_large_splats) {
        float large_splat_weight = smoothstep(
            0.5, 1.0, largest_half_size / u_maximum_splat_half_size);
        vertex_color = mix(
            vertex_color, vec3(1.0, 0.05, 0.0), large_splat_weight);
    }
    splat_center_pixels =
        (center_ndc * 0.5 + 0.5) * u_viewport_size;
    splat_opacity = raw_opacity;
}
)GLSL";

constexpr char kSplatFragmentShader[] = R"GLSL(
#version 330 core

in vec3 vertex_color;
flat in float splat_opacity;
flat in float splat_opacity_scale;
flat in vec2 splat_center_pixels;
flat in vec3 splat_conic;
uniform float u_alpha_density_scale;
uniform float u_minimum_splat_opacity;
out vec4 fragment_color;

void main()
{
    vec2 delta = gl_FragCoord.xy - splat_center_pixels;
    float radius_squared =
        splat_conic.x * delta.x * delta.x +
        2.0 * splat_conic.y * delta.x * delta.y +
        splat_conic.z * delta.y * delta.y;
    if (radius_squared > 8.0)
        discard;

    float opacity =
        (1.0 / (1.0 + exp(-splat_opacity))) * splat_opacity_scale;
    if (opacity < u_minimum_splat_opacity)
        discard;
    // Normalize the compact Gaussian so it is exactly one at the center and
    // exactly zero at the ellipse boundary. This prevents millions of weak
    // tails from accumulating into a low-frequency haze.
    const float boundary_weight = 0.0183156389; // exp(-4)
    float gaussian_weight =
        (exp(-0.5 * radius_squared) - boundary_weight) /
        (1.0 - boundary_weight);
    float base_alpha = min(0.99, opacity * gaussian_weight);
    // A low-detail interaction frame keeps one of every four splats. Treat
    // each retained alpha sample as representative of the omitted density so
    // the preview does not become four times darker.
    float alpha = 1.0 - pow(1.0 - base_alpha, u_alpha_density_scale);
    if (alpha < 1.0 / 255.0)
        discard;
    fragment_color = vec4(vertex_color * alpha, alpha);
}
)GLSL";

constexpr char kGpuReorderShader[] = R"GLSL(
#version 430 core
layout(local_size_x = 256) in;
layout(std430, binding = 0) readonly buffer SourceSplats {
    uint source_words[];
};
layout(std430, binding = 1) readonly buffer SortedIndices {
    uint sorted_indices[];
};
layout(std430, binding = 2) writeonly buffer SortedSplats {
    uint sorted_words[];
};
uniform uint u_splat_count;

void main()
{
    uint destination = gl_GlobalInvocationID.x;
    if (destination >= u_splat_count) return;
    const uint words_per_splat = 13u;
    uint source = sorted_indices[destination];
    for (uint word = 0u; word < words_per_splat; ++word) {
        sorted_words[destination * words_per_splat + word] =
            source_words[source * words_per_splat + word];
    }
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

void OpenGLWidget::setExposure(float exposure_ev)
{
    exposure_ev_ = std::clamp(exposure_ev, -4.0f, 4.0f);
    update();
}

void OpenGLWidget::setLowPassVariance(float variance)
{
    low_pass_variance_ = std::clamp(variance, 0.0f, 1.0f);
    update();
}

void OpenGLWidget::setMaximumSplatHalfSize(float pixels)
{
    maximum_splat_half_size_ = std::clamp(pixels, 16.0f, 256.0f);
    update();
}

void OpenGLWidget::setMinimumSplatOpacity(float opacity)
{
    minimum_splat_opacity_ = std::clamp(opacity, 0.0f, 0.2f);
    update();
}

void OpenGLWidget::setLargeSplatHighlightEnabled(bool enabled)
{
    highlight_large_splats_ = enabled;
    update();
}

void OpenGLWidget::setOversizedSplatSuppressionEnabled(bool enabled)
{
    suppress_oversized_splats_ = enabled;
    update();
}

void OpenGLWidget::setViewportOverlayEnabled(bool enabled)
{
    viewport_overlay_enabled_ = enabled;
    update();
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
    std::vector<float>().swap(sh_coefficient_data_);
    std::vector<float>().swap(static_splat_texture_data_);
    constexpr std::size_t kShRgbCoefficientCount = 16;
    sh_coefficient_data_.reserve(
        points.size() * kShRgbCoefficientCount * 3);
    static_splat_texture_data_.reserve(points.size() * 16);
    for (std::size_t point_index = 0; point_index < points.size(); ++point_index) {
        const GaussianPoint& point = points[point_index];
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
        gpu_splat.original_index = static_cast<std::uint32_t>(point_index);
        static_splat_texture_data_.insert(
            static_splat_texture_data_.end(),
            {point.x, point.y, point.z, point.opacity,
             point.red / 255.0f, point.green / 255.0f,
             point.blue / 255.0f, point.scale[0],
             point.scale[1], point.scale[2], point.rotation[0],
             point.rotation[1], point.rotation[2], point.rotation[3],
             0.0f, 0.0f});
        sh_coefficient_data_.insert(
            sh_coefficient_data_.end(), point.sh_dc.begin(), point.sh_dc.end());
        const std::size_t coefficients_per_channel = sh_degree_ > 0
            ? static_cast<std::size_t>(
                (sh_degree_ + 1) * (sh_degree_ + 1) - 1)
            : 0;
        for (std::size_t basis = 0;
             basis < kShRgbCoefficientCount - 1; ++basis) {
            for (std::size_t channel = 0; channel < 3; ++channel) {
                sh_coefficient_data_.push_back(
                    basis < coefficients_per_channel
                    ? point.sh_rest[channel * coefficients_per_channel + basis]
                    : 0.0f);
            }
        }
        source_gpu_splat_data_.push_back(gpu_splat);
    }
    splat_sort_indices_.resize(points.size());
    splat_sort_scratch_.resize(points.size());
    std::iota(splat_sort_indices_.begin(), splat_sort_indices_.end(), 0U);
    splat_sort_depths_.resize(points.size());
    splat_sort_dirty_ = true;
    sh_buffer_dirty_ = true;
    source_splat_buffer_dirty_ = true;
    static_splat_texture_dirty_ = true;
    fitPointCloudToView();
    rebuildMiniMapLandscape();

    // Calls made before initializeGL() are retained and uploaded when the
    // context becomes available.
    uploaded_splat_count_ = 0;
    if (points.empty() && isValid()) {
        makeCurrent();
        uploadSplatBuffer();
        uploadShTextureBuffer();
        if (source_splat_buffer_ != 0) {
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, source_splat_buffer_);
            glBufferData(GL_SHADER_STORAGE_BUFFER, 0, nullptr, GL_STATIC_DRAW);
        }
        if (sorted_index_buffer_ != 0) {
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, sorted_index_buffer_);
            glBufferData(GL_SHADER_STORAGE_BUFFER, 0, nullptr, GL_DYNAMIC_DRAW);
        }
        if (splat_data_buffer_ != 0) {
            glBindBuffer(GL_TEXTURE_BUFFER, splat_data_buffer_);
            glBufferData(GL_TEXTURE_BUFFER, 0, nullptr, GL_STATIC_DRAW);
        }
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
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
    navigation_pose_visible_ = enabled;
    update();
}

void OpenGLWidget::setNavigationPose(
    const QVector3D& position, float yaw_degrees)
{
    navigation_position_ = position;
    yaw_degrees_ = yaw_degrees;
    update();
}

void OpenGLWidget::setMiniMapTrajectory(
    const std::vector<QVector3D>& raw_positions,
    const std::vector<QVector3D>& smooth_positions)
{
    raw_trajectory_ = raw_positions;
    smooth_trajectory_ = smooth_positions;
    update();
}

QMatrix4x4 OpenGLWidget::sceneWorldToAlignedTransform() const
{
    return scene_alignment_matrix_ * model_matrix_;
}

void OpenGLWidget::finalizeInteractionSort()
{
    splat_sort_dirty_ = true;
    update();
}

bool OpenGLWidget::autoAlignSceneUp()
{
    if (source_gpu_splat_data_.size() < 3) return false;

    QVector3D mean;
    for (const GpuSplatData& splat : source_gpu_splat_data_) {
        mean += model_matrix_.map(QVector3D(splat.x, splat.y, splat.z));
    }
    mean /= static_cast<float>(source_gpu_splat_data_.size());

    double covariance[3][3]{};
    for (const GpuSplatData& splat : source_gpu_splat_data_) {
        const QVector3D delta =
            model_matrix_.map(QVector3D(splat.x, splat.y, splat.z)) - mean;
        const double value[3]{delta.x(), delta.y(), delta.z()};
        for (int row = 0; row < 3; ++row) {
            for (int column = row; column < 3; ++column) {
                covariance[row][column] += value[row] * value[column];
            }
        }
    }
    for (int row = 0; row < 3; ++row) {
        for (int column = row; column < 3; ++column) {
            covariance[row][column] /= source_gpu_splat_data_.size();
            covariance[column][row] = covariance[row][column];
        }
    }

    // Jacobi diagonalization of the symmetric 3x3 covariance matrix. The
    // eigenvector with least variance is the likely vertical axis indoors.
    double eigenvectors[3][3]{{1.0, 0.0, 0.0},
                              {0.0, 1.0, 0.0},
                              {0.0, 0.0, 1.0}};
    for (int iteration = 0; iteration < 12; ++iteration) {
        int p = 0;
        int q = 1;
        if (std::abs(covariance[0][2]) > std::abs(covariance[p][q])) {
            p = 0; q = 2;
        }
        if (std::abs(covariance[1][2]) > std::abs(covariance[p][q])) {
            p = 1; q = 2;
        }
        if (std::abs(covariance[p][q]) < 1.0e-12) break;

        const double angle = 0.5 * std::atan2(
            2.0 * covariance[p][q], covariance[q][q] - covariance[p][p]);
        const double cosine = std::cos(angle);
        const double sine = std::sin(angle);
        for (int index = 0; index < 3; ++index) {
            if (index == p || index == q) continue;
            const double aip = covariance[index][p];
            const double aiq = covariance[index][q];
            covariance[index][p] = covariance[p][index] =
                cosine * aip - sine * aiq;
            covariance[index][q] = covariance[q][index] =
                sine * aip + cosine * aiq;
        }
        const double app = covariance[p][p];
        const double aqq = covariance[q][q];
        const double apq = covariance[p][q];
        covariance[p][p] = cosine * cosine * app -
            2.0 * sine * cosine * apq + sine * sine * aqq;
        covariance[q][q] = sine * sine * app +
            2.0 * sine * cosine * apq + cosine * cosine * aqq;
        covariance[p][q] = covariance[q][p] = 0.0;
        for (int row = 0; row < 3; ++row) {
            const double vip = eigenvectors[row][p];
            const double viq = eigenvectors[row][q];
            eigenvectors[row][p] = cosine * vip - sine * viq;
            eigenvectors[row][q] = sine * vip + cosine * viq;
        }
    }

    int smallest = 0;
    if (covariance[1][1] < covariance[smallest][smallest]) smallest = 1;
    if (covariance[2][2] < covariance[smallest][smallest]) smallest = 2;
    QVector3D up(
        eigenvectors[0][smallest],
        eigenvectors[1][smallest],
        eigenvectors[2][smallest]);
    if (up.lengthSquared() < 1.0e-10f) return false;
    up.normalize();
    if (up.z() < 0.0f) up = -up;

    scene_alignment_matrix_.setToIdentity();
    scene_alignment_matrix_.rotate(QQuaternion::rotationTo(
        up, QVector3D(0.0f, 0.0f, 1.0f)));
    rebuildMiniMapLandscape();
    splat_sort_dirty_ = true;
    update();
    return true;
}

void OpenGLWidget::initializeGL()
{
    initializeOpenGLFunctions();
    glEnable(GL_DEPTH_TEST);
    resetCameraMatrices();

    if (createSplatBuffers()) {
        uploadSplatBuffer();
        uploadShTextureBuffer();
    }
    createSplatShaderProgram();
    createGpuReorderProgram();
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

bool OpenGLWidget::createGpuReorderProgram()
{
    if (qEnvironmentVariableIntValue("LIDAR_3DGS_GPU_REORDER") != 1) {
        gpu_reorder_available_ = false;
        qInfo("GPU-assisted splat reordering disabled; using CPU fallback.");
        return false;
    }
    const QSurfaceFormat context_format = context()->format();
    if (context_format.majorVersion() < 4 ||
        (context_format.majorVersion() == 4 && context_format.minorVersion() < 3)) {
        gpu_reorder_available_ = false;
        return false;
    }
    gpu_reorder_program_ = std::make_unique<QOpenGLShaderProgram>();
    if (!gpu_reorder_program_->addShaderFromSourceCode(
            QOpenGLShader::Compute, kGpuReorderShader) ||
        !gpu_reorder_program_->link()) {
        qWarning("GPU splat reorder shader failed: %s",
                 qPrintable(gpu_reorder_program_->log()));
        gpu_reorder_program_.reset();
        gpu_reorder_available_ = false;
        return false;
    }
    gpu_reorder_available_ = true;
    qInfo("GPU-assisted splat reordering enabled (OpenGL %d.%d).",
          context_format.majorVersion(), context_format.minorVersion());
    return true;
}

bool OpenGLWidget::createSplatBuffers()
{
    static_assert(std::is_standard_layout<GpuSplatData>::value,
                  "GpuSplatData must be an interleaved vertex type.");
    static_assert(sizeof(GpuSplatData) == 52,
                  "GpuSplatData layout must match the configured attributes.");
    static_assert(offsetof(GpuSplatData, scale_x) == 20,
                  "GpuSplatData scale offset must remain stable.");
    static_assert(offsetof(GpuSplatData, rotation_w) == 32,
                  "GpuSplatData rotation offset must remain stable.");
    static_assert(offsetof(GpuSplatData, original_index) == 48,
                  "GpuSplatData index offset must remain stable.");

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
    if (sh_buffer_ == 0) glGenBuffers(1, &sh_buffer_);
    if (sh_texture_ == 0) glGenTextures(1, &sh_texture_);
    if (source_splat_buffer_ == 0) glGenBuffers(1, &source_splat_buffer_);
    if (sorted_index_buffer_ == 0) glGenBuffers(1, &sorted_index_buffer_);
    if (alternate_sorted_index_buffer_ == 0)
        glGenBuffers(1, &alternate_sorted_index_buffer_);
    if (splat_data_buffer_ == 0) glGenBuffers(1, &splat_data_buffer_);
    if (splat_data_texture_ == 0) glGenTextures(1, &splat_data_texture_);
    if (sorted_index_texture_ == 0) glGenTextures(1, &sorted_index_texture_);
    if (alternate_sorted_index_texture_ == 0)
        glGenTextures(1, &alternate_sorted_index_texture_);
    if (sh_buffer_ == 0 || sh_texture_ == 0 ||
        source_splat_buffer_ == 0 || sorted_index_buffer_ == 0 ||
        alternate_sorted_index_buffer_ == 0 ||
        splat_data_buffer_ == 0 || splat_data_texture_ == 0 ||
        sorted_index_texture_ == 0 || alternate_sorted_index_texture_ == 0) {
        qWarning("Failed to create spherical-harmonic texture buffer.");
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

    glEnableVertexAttribArray(kGaussianIndexAttribute);
    glVertexAttribIPointer(
        kGaussianIndexAttribute, 1, GL_UNSIGNED_INT, sizeof(GpuSplatData),
        reinterpret_cast<const void*>(offsetof(GpuSplatData, original_index)));
    glVertexAttribDivisor(kGaussianIndexAttribute, 1);

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

bool OpenGLWidget::uploadShTextureBuffer()
{
    if (!sh_buffer_dirty_) return true;
    if (sh_buffer_ == 0 || sh_texture_ == 0) return false;

    glBindBuffer(GL_TEXTURE_BUFFER, sh_buffer_);
    glBufferData(
        GL_TEXTURE_BUFFER,
        static_cast<GLsizeiptr>(sh_coefficient_data_.size() * sizeof(float)),
        sh_coefficient_data_.empty() ? nullptr : sh_coefficient_data_.data(),
        GL_STATIC_DRAW);
    glBindTexture(GL_TEXTURE_BUFFER, sh_texture_);
    glTexBuffer(GL_TEXTURE_BUFFER, GL_RGB32F, sh_buffer_);
    glBindTexture(GL_TEXTURE_BUFFER, 0);
    glBindBuffer(GL_TEXTURE_BUFFER, 0);
    sh_buffer_dirty_ = false;
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
    // depth once, then sort 32-bit indices instead of moving 52-byte records.
    const float depth_x = model_view(2, 0);
    const float depth_y = model_view(2, 1);
    const float depth_z = model_view(2, 2);
    const float depth_offset = model_view(2, 3);
    const std::size_t sorted_count = source_gpu_splat_data_.size();
    splat_sort_indices_.resize(sorted_count);
    const std::size_t hardware_threads = std::max(
        1U, std::thread::hardware_concurrency());
    const std::size_t worker_count = std::min(
        hardware_threads, std::max<std::size_t>(1, sorted_count / 65536));
    const auto parallel_chunks = [worker_count, sorted_count](auto&& operation) {
        std::vector<std::thread> workers;
        workers.reserve(worker_count > 0 ? worker_count - 1 : 0);
        for (std::size_t worker = 1; worker < worker_count; ++worker) {
            workers.emplace_back([&, worker]() {
                const std::size_t begin = sorted_count * worker / worker_count;
                const std::size_t end =
                    sorted_count * (worker + 1) / worker_count;
                operation(worker, begin, end);
            });
        }
        operation(0, 0, sorted_count / worker_count);
        for (std::thread& worker : workers) worker.join();
    };
    parallel_chunks([&](std::size_t, std::size_t begin, std::size_t end) {
        for (std::size_t index = begin; index < end; ++index) {
            splat_sort_indices_[index] = static_cast<std::uint32_t>(index);
            const GpuSplatData& splat = source_gpu_splat_data_[index];
            splat_sort_depths_[index] =
                depth_x * splat.x + depth_y * splat.y +
                depth_z * splat.z + depth_offset;
        }
    });

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
        std::vector<std::array<std::size_t, 256>>
            worker_offsets(worker_count);
        const unsigned shift = pass * 8;
        parallel_chunks([&](std::size_t worker, std::size_t begin,
                            std::size_t end) {
            auto& counts = worker_offsets[worker];
            for (std::size_t position = begin; position < end; ++position) {
                const std::uint32_t index = splat_sort_indices_[position];
                ++counts[(depth_key(index) >> shift) & 0xffU];
            }
        });
        std::size_t running_offset = 0;
        for (std::size_t byte = 0; byte < 256; ++byte) {
            for (std::size_t worker = 0; worker < worker_count; ++worker) {
                const std::size_t count = worker_offsets[worker][byte];
                worker_offsets[worker][byte] = running_offset;
                running_offset += count;
            }
        }
        parallel_chunks([&](std::size_t worker, std::size_t begin,
                            std::size_t end) {
            auto offsets = worker_offsets[worker];
            for (std::size_t position = begin; position < end; ++position) {
                const std::uint32_t index = splat_sort_indices_[position];
                const std::uint32_t byte =
                    (depth_key(index) >> shift) & 0xffU;
                splat_sort_scratch_[offsets[byte]++] = index;
            }
        });
        splat_sort_indices_.swap(splat_sort_scratch_);
    }
    if (static_splat_texture_dirty_) {
        glBindBuffer(GL_TEXTURE_BUFFER, splat_data_buffer_);
        glBufferData(
            GL_TEXTURE_BUFFER,
            static_cast<GLsizeiptr>(
                static_splat_texture_data_.size() * sizeof(float)),
            static_splat_texture_data_.data(), GL_STATIC_DRAW);
        glBindTexture(GL_TEXTURE_BUFFER, splat_data_texture_);
        glTexBuffer(GL_TEXTURE_BUFFER, GL_RGBA32F, splat_data_buffer_);
        static_splat_texture_dirty_ = false;
    }
    alternate_sorted_index_active_ = !alternate_sorted_index_active_;
    const GLuint upload_buffer = alternate_sorted_index_active_
        ? alternate_sorted_index_buffer_ : sorted_index_buffer_;
    const GLuint upload_texture = alternate_sorted_index_active_
        ? alternate_sorted_index_texture_ : sorted_index_texture_;
    glBindBuffer(GL_TEXTURE_BUFFER, upload_buffer);
    glBufferData(
        GL_TEXTURE_BUFFER,
        static_cast<GLsizeiptr>(sorted_count * sizeof(std::uint32_t)),
        splat_sort_indices_.data(), GL_DYNAMIC_DRAW);
    glBindTexture(GL_TEXTURE_BUFFER, upload_texture);
    glTexBuffer(GL_TEXTURE_BUFFER, GL_R32UI, upload_buffer);
    glBindTexture(GL_TEXTURE_BUFFER, 0);
    glBindBuffer(GL_TEXTURE_BUFFER, 0);
    uploaded_splat_count_ = sorted_count;
    last_sort_depth_direction_ = depth_direction;
    splat_sort_dirty_ = false;
    return true;
}

void OpenGLWidget::destroySplatResources()
{
    uploaded_splat_count_ = 0;
    allocated_splat_bytes_ = 0;
    splat_shader_program_.reset();
    gpu_reorder_program_.reset();
    if (sh_texture_ != 0) glDeleteTextures(1, &sh_texture_);
    if (sh_buffer_ != 0) glDeleteBuffers(1, &sh_buffer_);
    if (source_splat_buffer_ != 0) glDeleteBuffers(1, &source_splat_buffer_);
    if (sorted_index_buffer_ != 0) glDeleteBuffers(1, &sorted_index_buffer_);
    if (alternate_sorted_index_buffer_ != 0)
        glDeleteBuffers(1, &alternate_sorted_index_buffer_);
    if (splat_data_buffer_ != 0) glDeleteBuffers(1, &splat_data_buffer_);
    if (splat_data_texture_ != 0) glDeleteTextures(1, &splat_data_texture_);
    if (sorted_index_texture_ != 0) glDeleteTextures(1, &sorted_index_texture_);
    if (alternate_sorted_index_texture_ != 0)
        glDeleteTextures(1, &alternate_sorted_index_texture_);
    sh_texture_ = 0;
    sh_buffer_ = 0;
    source_splat_buffer_ = 0;
    sorted_index_buffer_ = 0;
    alternate_sorted_index_buffer_ = 0;
    splat_data_buffer_ = 0;
    splat_data_texture_ = 0;
    sorted_index_texture_ = 0;
    alternate_sorted_index_texture_ = 0;
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

void OpenGLWidget::rebuildMiniMapLandscape()
{
    minimap_occupancy_.fill(0);
    minimap_peak_occupancy_ = 0;
    if (source_gpu_splat_data_.empty()) return;

    QVector3D first = scene_alignment_matrix_.map(
        model_matrix_.map(QVector3D(
            source_gpu_splat_data_.front().x,
            source_gpu_splat_data_.front().y,
            source_gpu_splat_data_.front().z)));
    minimap_min_x_ = minimap_max_x_ = first.x();
    minimap_min_y_ = minimap_max_y_ = first.y();
    for (const GpuSplatData& splat : source_gpu_splat_data_) {
        const QVector3D point = scene_alignment_matrix_.map(
            model_matrix_.map(QVector3D(splat.x, splat.y, splat.z)));
        minimap_min_x_ = std::min(minimap_min_x_, point.x());
        minimap_max_x_ = std::max(minimap_max_x_, point.x());
        minimap_min_y_ = std::min(minimap_min_y_, point.y());
        minimap_max_y_ = std::max(minimap_max_y_, point.y());
    }

    constexpr float kMinimumSpan = 1.0e-5f;
    const float span_x = std::max(minimap_max_x_ - minimap_min_x_, kMinimumSpan);
    const float span_y = std::max(minimap_max_y_ - minimap_min_y_, kMinimumSpan);
    for (const GpuSplatData& splat : source_gpu_splat_data_) {
        const QVector3D point = scene_alignment_matrix_.map(
            model_matrix_.map(QVector3D(splat.x, splat.y, splat.z)));
        const int x = std::clamp(static_cast<int>(
            (point.x() - minimap_min_x_) / span_x * kMiniMapGridSize),
            0, kMiniMapGridSize - 1);
        const int y = std::clamp(static_cast<int>(
            (point.y() - minimap_min_y_) / span_y * kMiniMapGridSize),
            0, kMiniMapGridSize - 1);
        std::uint16_t& occupancy =
            minimap_occupancy_[y * kMiniMapGridSize + x];
        if (occupancy < std::numeric_limits<std::uint16_t>::max()) ++occupancy;
        minimap_peak_occupancy_ = std::max(minimap_peak_occupancy_, occupancy);
    }
}

void OpenGLWidget::renderGaussianSplats()
{
    if (source_gpu_splat_data_.empty() || !isSplatShaderReady() ||
        !splat_vao_.isCreated()) {
        return;
    }

    if (!uploadShTextureBuffer()) {
        qWarning("Failed to upload spherical-harmonic texture buffer.");
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
    splat_shader_program_->setUniformValue("u_sh_coefficients", 0);
    splat_shader_program_->setUniformValue("u_splat_data", 1);
    splat_shader_program_->setUniformValue("u_sorted_indices", 2);
    splat_shader_program_->setUniformValue("u_exposure_ev", exposure_ev_);
    splat_shader_program_->setUniformValue(
        "u_low_pass_variance", low_pass_variance_);
    splat_shader_program_->setUniformValue(
        "u_maximum_splat_half_size", maximum_splat_half_size_);
    splat_shader_program_->setUniformValue(
        "u_minimum_splat_opacity", minimum_splat_opacity_);
    splat_shader_program_->setUniformValue(
        "u_highlight_large_splats", highlight_large_splats_);
    splat_shader_program_->setUniformValue(
        "u_suppress_oversized_splats", suppress_oversized_splats_);
    splat_shader_program_->setUniformValue("u_alpha_density_scale", 1.0f);
    bool inverse_ok = false;
    const QMatrix4x4 inverse_model_view = model_view.inverted(&inverse_ok);
    const QVector3D camera_position_model = inverse_ok
        ? inverse_model_view.map(QVector3D(0.0f, 0.0f, 0.0f))
        : QVector3D(0.0f, 0.0f, 0.0f);
    splat_shader_program_->setUniformValue(
        "u_camera_position_model", camera_position_model);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_BUFFER, sh_texture_);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_BUFFER, splat_data_texture_);
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_BUFFER, alternate_sorted_index_active_
        ? alternate_sorted_index_texture_ : sorted_index_texture_);
    {
        QOpenGLVertexArrayObject::Binder vao_binder(&splat_vao_);
        glDrawArraysInstanced(
            GL_TRIANGLE_STRIP, 0, 4,
            static_cast<GLsizei>(uploaded_splat_count_));
    }
    glBindTexture(GL_TEXTURE_BUFFER, 0);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_BUFFER, 0);
    glActiveTexture(GL_TEXTURE0);

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

    if (viewport_overlay_enabled_) paintDemoScene(painter);
    paintMiniMap(painter);
    OrientationGizmo::paint(
        painter, size(), yaw_degrees_, pitch_degrees_, z_up_gizmo_);
}

void OpenGLWidget::paintMiniMap(QPainter& painter)
{
    if (minimap_peak_occupancy_ == 0) return;

    constexpr int kMapSize = 190;
    constexpr int kMargin = 16;
    constexpr int kPadding = 12;
    const QRectF frame(
        width() - kMapSize - kMargin, height() - kMapSize - kMargin,
        kMapSize, kMapSize);
    const QRectF map = frame.adjusted(kPadding, kPadding, -kPadding, -kPadding);

    painter.save();
    painter.setPen(QPen(QColor(255, 255, 255, 75), 1.0));
    painter.setBrush(QColor(12, 17, 25, 210));
    painter.drawRoundedRect(frame, 8.0, 8.0);

    const float cell_width = map.width() / kMiniMapGridSize;
    const float cell_height = map.height() / kMiniMapGridSize;
    painter.setPen(Qt::NoPen);
    for (int y = 0; y < kMiniMapGridSize; ++y) {
        for (int x = 0; x < kMiniMapGridSize; ++x) {
            const std::uint16_t occupancy =
                minimap_occupancy_[y * kMiniMapGridSize + x];
            if (occupancy == 0) continue;
            const float density = std::log1p(static_cast<float>(occupancy)) /
                std::log1p(static_cast<float>(minimap_peak_occupancy_));
            painter.setBrush(QColor(105, 177, 166,
                static_cast<int>(55.0f + 165.0f * density)));
            painter.drawRect(QRectF(
                map.left() + x * cell_width,
                map.bottom() - (y + 1) * cell_height,
                cell_width + 0.5, cell_height + 0.5));
        }
    }

    const float span_x = std::max(minimap_max_x_ - minimap_min_x_, 1.0e-5f);
    const float span_y = std::max(minimap_max_y_ - minimap_min_y_, 1.0e-5f);
    const auto to_map = [&](const QVector3D& position) {
        return QPointF(
            map.left() + (position.x() - minimap_min_x_) / span_x * map.width(),
            map.bottom() - (position.y() - minimap_min_y_) / span_y * map.height());
    };
    const auto draw_path = [&](const std::vector<QVector3D>& positions,
                               const QPen& pen) {
        if (positions.size() < 2) return;
        QPainterPath path(to_map(positions.front()));
        for (std::size_t index = 1; index < positions.size(); ++index)
            path.lineTo(to_map(positions[index]));
        painter.setPen(pen);
        painter.setBrush(Qt::NoBrush);
        painter.drawPath(path);
    };
    painter.save();
    painter.setClipRect(map);
    draw_path(raw_trajectory_, QPen(QColor(220, 225, 235, 105), 1.0));
    draw_path(smooth_trajectory_, QPen(QColor(58, 218, 238), 2.5));
    if (!smooth_trajectory_.empty()) {
        painter.setPen(QPen(QColor(255, 255, 255, 190), 1.0));
        painter.setBrush(QColor(71, 215, 120));
        painter.drawEllipse(to_map(smooth_trajectory_.front()), 4.0, 4.0);
        painter.setBrush(QColor(245, 86, 86));
        painter.drawEllipse(to_map(smooth_trajectory_.back()), 4.0, 4.0);
    }
    painter.restore();

    if (navigation_pose_visible_) {
        const QPointF robot(
            std::clamp(
                map.left() + (navigation_position_.x() - minimap_min_x_) /
                    span_x * map.width(),
                map.left() + 8.0, map.right() - 8.0),
            std::clamp(
                map.bottom() - (navigation_position_.y() - minimap_min_y_) /
                    span_y * map.height(),
                map.top() + 8.0, map.bottom() - 8.0));
        const float yaw = qDegreesToRadians(yaw_degrees_);
        const QPointF forward(std::sin(yaw), -std::cos(yaw));
        const QPointF right(-forward.y(), forward.x());
        QPainterPath marker;
        marker.moveTo(robot + forward * 9.0);
        marker.lineTo(robot - forward * 6.0 + right * 5.0);
        marker.lineTo(robot - forward * 6.0 - right * 5.0);
        marker.closeSubpath();
        painter.setPen(QPen(QColor(255, 255, 255), 1.5));
        painter.setBrush(QColor(255, 153, 48));
        painter.drawPath(marker);
    }

    painter.setPen(QColor(225, 234, 240));
    painter.drawText(frame.adjusted(8, 5, -8, -5),
                     Qt::AlignLeft | Qt::AlignTop, "Top view");
    painter.restore();
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
