#include "OrientationGizmo.h"

#include <QColor>
#include <QFont>
#include <QMatrix4x4>
#include <QPainter>
#include <QPointF>
#include <QVector3D>

#include <algorithm>
#include <array>
#include <cmath>

namespace
{
constexpr qreal kRadius = 42.0;
constexpr qreal kHitRadius = 13.0;
constexpr int kMargin = 24;

struct ProjectedAxis
{
    OrientationGizmo::Axis axis;
    QPointF end;
    float depth;
    QColor color;
    const char* label;
};

QPointF gizmoCenter(const QSize& viewport)
{
    return {kMargin + kRadius, viewport.height() - kMargin - kRadius};
}

std::array<ProjectedAxis, 3> projectedAxes(
    const QSize& viewport, float yaw_degrees, float pitch_degrees, bool z_up)
{
    QMatrix4x4 rotation;
    rotation.rotate(pitch_degrees, 1.0f, 0.0f, 0.0f);
    if (z_up) {
        rotation.rotate(-90.0f, 1.0f, 0.0f, 0.0f);
        rotation.rotate(yaw_degrees, 0.0f, 0.0f, 1.0f);
    } else {
        rotation.rotate(yaw_degrees, 0.0f, 1.0f, 0.0f);
    }

    const QPointF center = gizmoCenter(viewport);
    const auto project = [center](const QVector3D& direction) {
        // Use a true orthographic screen projection. Adding depth to the
        // screen coordinates made the world-up axis appear to lean sideways
        // during pitch, which incorrectly suggested camera roll.
        return center + QPointF(
            direction.x() * kRadius,
            -direction.y() * kRadius);
    };

    const QVector3D x = rotation.mapVector({1.0f, 0.0f, 0.0f});
    const QVector3D y = rotation.mapVector({0.0f, 1.0f, 0.0f});
    const QVector3D z = rotation.mapVector({0.0f, 0.0f, 1.0f});
    return {{
        {OrientationGizmo::Axis::X, project(x), x.z(), QColor(239, 68, 68), "X"},
        {OrientationGizmo::Axis::Y, project(y), y.z(), QColor(34, 197, 94), "Y"},
        {OrientationGizmo::Axis::Z, project(z), z.z(), QColor(59, 130, 246), "Z"},
    }};
}
} // namespace

void OrientationGizmo::paint(QPainter& painter, const QSize& viewport,
                             float yaw_degrees, float pitch_degrees, bool z_up)
{
    const QPointF center = gizmoCenter(viewport);
    auto axes = projectedAxes(viewport, yaw_degrees, pitch_degrees, z_up);
    std::sort(axes.begin(), axes.end(), [](const auto& left, const auto& right) {
        return left.depth < right.depth;
    });

    painter.save();
    painter.setBrush(QColor(10, 14, 24, 150));
    painter.setPen(QPen(QColor(255, 255, 255, 35), 1.0));
    painter.drawEllipse(center, kRadius + 12.0, kRadius + 12.0);

    QFont label_font = painter.font();
    label_font.setBold(true);
    painter.setFont(label_font);
    for (const ProjectedAxis& axis : axes) {
        painter.setPen(QPen(axis.color, 3.0, Qt::SolidLine, Qt::RoundCap));
        painter.drawLine(center, axis.end);
        painter.setBrush(axis.color);
        painter.setPen(Qt::NoPen);
        painter.drawEllipse(axis.end, 7.0, 7.0);
        painter.setPen(Qt::white);
        painter.drawText(axis.end + QPointF(8.0, -7.0), axis.label);
    }

    painter.setBrush(Qt::white);
    painter.setPen(Qt::NoPen);
    painter.drawEllipse(center, 3.5, 3.5);
    painter.restore();
}

OrientationGizmo::Axis OrientationGizmo::hitTest(
    const QPoint& position, const QSize& viewport,
    float yaw_degrees, float pitch_degrees, bool z_up)
{
    const auto axes = projectedAxes(
        viewport, yaw_degrees, pitch_degrees, z_up);
    Axis closest_axis = Axis::None;
    qreal closest_distance = kHitRadius;
    for (const ProjectedAxis& axis : axes) {
        const qreal dx = position.x() - axis.end.x();
        const qreal dy = position.y() - axis.end.y();
        const qreal distance = std::hypot(dx, dy);
        if (distance <= closest_distance) {
            closest_distance = distance;
            closest_axis = axis.axis;
        }
    }
    return closest_axis;
}
