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
    void setConstrainedZUpNavigation(bool enabled);
    void beginFloorAlignment();

signals:
    void orientationChanged(float yaw_degrees, float pitch_degrees);
    void loadStatusChanged(const QString& text, const QString& file_path);
    void floorAlignmentStatusChanged(const QString& text);

protected:
    bool eventFilter(QObject* watched, QEvent* event) override;

private:
    enum class DragMode {
        None,
        RotateFree,
        RotatePending,
        RotateYawOnly,
        RotatePitchOnly,
        Pan
    };

    void applyTransform();
    void snapToAxis(int axis);

    OpenGLWidget* viewer_ = nullptr;
    GaussianSplatProcessing point_processing_;
    QPoint last_mouse_position_;
    QPoint drag_start_position_;
    QVector3D translation_;
    QVector3D z_up_camera_position_{0.0f, -3.0f, 0.0f};
    float yaw_degrees_ = 0.0f;
    float pitch_degrees_ = 0.0f;
    float camera_distance_ = 3.0f;
    bool constrained_z_up_navigation_ = false;
    bool selecting_floor_points_ = false;
    std::vector<QVector3D> floor_points_;
    DragMode drag_mode_ = DragMode::None;
};
