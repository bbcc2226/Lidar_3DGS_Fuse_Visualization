#pragma once

#include <QMainWindow>

class QLabel;
class OpenGLWidget;
class ViewerController;

class MainWindow final : public QMainWindow
{
public:
    explicit MainWindow(QWidget* parent = nullptr);

private:
    OpenGLWidget* gl_widget_ = nullptr;
    ViewerController* viewer_controller_ = nullptr;
    QLabel* orientation_label_ = nullptr;
    QLabel* load_status_label_ = nullptr;
    QLabel* trajectory_status_label_ = nullptr;
    QLabel* manual_path_status_label_ = nullptr;
    QLabel* robot_playback_status_label_ = nullptr;
    QLabel* floor_alignment_label_ = nullptr;
    QLabel* free_zone_status_label_ = nullptr;
};
