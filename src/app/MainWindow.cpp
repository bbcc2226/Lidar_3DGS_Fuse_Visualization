#include "MainWindow.h"

#include "OpenGLWidget.h"

#include <QApplication>
#include <QColorDialog>
#include <QFileDialog>
#include <QFrame>
#include <QHBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QSlider>
#include <QVBoxLayout>
#include <QWidget>

MainWindow::MainWindow(QWidget* parent)
    : QMainWindow(parent)
{
    setWindowTitle("3DGS Qt OpenGL Viewer");
    resize(1100, 700);

    auto* central_widget = new QWidget(this);
    auto* root_layout = new QHBoxLayout(central_widget);
    root_layout->setContentsMargins(8, 8, 8, 8);
    root_layout->setSpacing(8);

    gl_widget_ = new OpenGLWidget(central_widget);
    root_layout->addWidget(gl_widget_, 1);

    auto* panel = new QFrame(central_widget);
    panel->setFrameShape(QFrame::StyledPanel);
    panel->setMinimumWidth(250);
    panel->setMaximumWidth(320);
    auto* panel_layout = new QVBoxLayout(panel);

    auto* title = new QLabel("Viewer Controls", panel);
    QFont title_font = title->font();
    title_font.setPointSize(title_font.pointSize() + 3);
    title_font.setBold(true);
    title->setFont(title_font);
    panel_layout->addWidget(title);

    auto* open_ply_button = new QPushButton("Open PLY...", panel);
    load_status_label_ = new QLabel("No point cloud loaded", panel);
    load_status_label_->setWordWrap(true);
    panel_layout->addWidget(open_ply_button);
    panel_layout->addWidget(load_status_label_);
    panel_layout->addSpacing(12);

    panel_layout->addWidget(new QLabel("Animation speed", panel));
    auto* speed_slider = new QSlider(Qt::Horizontal, panel);
    speed_slider->setRange(0, 180);
    speed_slider->setValue(45);
    speed_slider->setTickInterval(30);
    speed_slider->setTickPosition(QSlider::TicksBelow);
    panel_layout->addWidget(speed_slider);

    animation_button_ = new QPushButton("Pause animation", panel);
    auto* reset_button = new QPushButton("Reset view", panel);
    auto* background_button = new QPushButton("Background color...", panel);
    angle_label_ = new QLabel("Angle: 0.0 degrees", panel);
    panel_layout->addWidget(animation_button_);
    panel_layout->addWidget(reset_button);
    panel_layout->addWidget(background_button);
    panel_layout->addSpacing(12);
    panel_layout->addWidget(angle_label_);
    panel_layout->addStretch(1);

    auto* hint = new QLabel(
        "This panel is ready for dataset loading, rendering modes, and BA controls.", panel);
    hint->setWordWrap(true);
    hint->setStyleSheet("color: palette(mid);");
    panel_layout->addWidget(hint);
    root_layout->addWidget(panel);
    setCentralWidget(central_widget);

    connect(speed_slider, &QSlider::valueChanged,
            gl_widget_, &OpenGLWidget::setAnimationSpeed);
    connect(open_ply_button, &QPushButton::clicked,
            this, &MainWindow::openPlyFile);
    connect(animation_button_, &QPushButton::clicked, this, [this]() {
        gl_widget_->setAnimating(!gl_widget_->isAnimating());
        animation_button_->setText(
            gl_widget_->isAnimating() ? "Pause animation" : "Resume animation");
    });
    connect(reset_button, &QPushButton::clicked,
            gl_widget_, &OpenGLWidget::resetView);
    connect(background_button, &QPushButton::clicked, this, [this]() {
        const QColor color = QColorDialog::getColor(QColor(25, 30, 42), this,
                                                     "Choose viewport background");
        gl_widget_->setBackgroundColor(color);
    });
    connect(gl_widget_, &QOpenGLWidget::frameSwapped, this, [this]() {
        angle_label_->setText(
            QString("Angle: %1 degrees").arg(gl_widget_->angle(), 0, 'f', 1));
    });
}

void MainWindow::openPlyFile()
{
    const QString path = QFileDialog::getOpenFileName(
        this,
        "Open Gaussian or point-cloud PLY",
        QString(),
        "PLY files (*.ply);;All files (*)");
    if (path.isEmpty()) return;

    load_status_label_->setText("Loading " + path + "...");
    load_status_label_->setToolTip(path);
    QApplication::setOverrideCursor(Qt::WaitCursor);
    QApplication::processEvents();

    const bool loaded = point_processing_.loadPly(path.toStdString());
    QApplication::restoreOverrideCursor();

    if (!loaded) {
        gl_widget_->setGaussianPoints({});
        load_status_label_->setText(
            "Load failed: " +
            QString::fromStdString(point_processing_.lastError()));
        return;
    }

    gl_widget_->setGaussianPoints(point_processing_.points());
    load_status_label_->setText(
        QString("Loaded %1 points | uploaded %2")
            .arg(static_cast<qulonglong>(point_processing_.splatCount()))
            .arg(static_cast<qulonglong>(gl_widget_->uploadedPointCount())));
}
