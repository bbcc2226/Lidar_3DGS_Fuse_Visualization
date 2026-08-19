#pragma once

#include <QObject>
#include <QPoint>
#include <QVector3D>

#include "3dgsProcessing.h"

class OpenGLWidget;
class QWidget;

class ViewerController final : public QObject
{
    Q_OBJECT

public:
    explicit ViewerController(OpenGLWidget* viewer, QObject* parent = nullptr);

    void openPlyFile(QWidget* dialog_parent);
    void resetView();

signals:
    void orientationChanged(float yaw_degrees, float pitch_degrees);
    void loadStatusChanged(const QString& text, const QString& file_path);

protected:
    bool eventFilter(QObject* watched, QEvent* event) override;

private:
    enum class DragMode { None, Rotate, Pan };

    void applyTransform();
    void snapToAxis(int axis);

    OpenGLWidget* viewer_ = nullptr;
    GaussianSplatProcessing point_processing_;
    QPoint last_mouse_position_;
    QVector3D translation_;
    float yaw_degrees_ = 0.0f;
    float pitch_degrees_ = 0.0f;
    float camera_distance_ = 3.0f;
    DragMode drag_mode_ = DragMode::None;
};
