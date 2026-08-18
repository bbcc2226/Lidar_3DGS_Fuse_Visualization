#pragma once

#include <QMainWindow>

#include "3dgsProcessing.h"

class QLabel;
class QPushButton;
class OpenGLWidget;

class MainWindow final : public QMainWindow
{
public:
    explicit MainWindow(QWidget* parent = nullptr);

private:
    void openPlyFile();

    GaussianSplatProcessing point_processing_;
    OpenGLWidget* gl_widget_ = nullptr;
    QPushButton* animation_button_ = nullptr;
    QLabel* angle_label_ = nullptr;
    QLabel* load_status_label_ = nullptr;
};
