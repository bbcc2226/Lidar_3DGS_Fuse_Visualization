#pragma once

#include <QPoint>
#include <QSize>

class QPainter;

class OrientationGizmo
{
public:
    enum class Axis { None, X, Y, Z };

    static void paint(QPainter& painter, const QSize& viewport,
                      float yaw_degrees, float pitch_degrees);
    static Axis hitTest(const QPoint& position, const QSize& viewport,
                        float yaw_degrees, float pitch_degrees);
};
