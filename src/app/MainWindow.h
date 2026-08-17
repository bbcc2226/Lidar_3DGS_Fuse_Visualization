#pragma once

#include <QMainWindow>

class QLabel;
class QPushButton;
class OpenGLWidget;

class MainWindow final : public QMainWindow
{
public:
    explicit MainWindow(QWidget* parent = nullptr);

private:
    OpenGLWidget* gl_widget_ = nullptr;
    QPushButton* animation_button_ = nullptr;
    QLabel* angle_label_ = nullptr;
};
