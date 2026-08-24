#include "MainWindow.h"

#include "OpenGLWidget.h"
#include "ViewerController.h"

#include <QColorDialog>
#include <QCheckBox>
#include <QFrame>
#include <QHBoxLayout>
#include <QLabel>
#include <QPushButton>
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
    viewer_controller_ = new ViewerController(gl_widget_, this);
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

    auto* reset_button = new QPushButton("Reset view", panel);
    auto* align_floor_button = new QPushButton("Align floor: 3 points", panel);
    floor_alignment_label_ = new QLabel("Floor alignment not set", panel);
    floor_alignment_label_->setWordWrap(true);
    auto* constrained_rotation = new QCheckBox(
        "Z-up constrained rotation", panel);
    constrained_rotation->setToolTip(
        "Reset to a level Z-up view; clear horizontal drags turn around and "
        "clear vertical drags look up or down. Up/Down arrow keys change "
        "camera height.");
    auto* background_button = new QPushButton("Background color...", panel);
    orientation_label_ = new QLabel("Yaw: 0.0 | Pitch: 0.0 degrees", panel);
    panel_layout->addWidget(reset_button);
    panel_layout->addWidget(align_floor_button);
    panel_layout->addWidget(floor_alignment_label_);
    panel_layout->addWidget(constrained_rotation);
    panel_layout->addWidget(background_button);
    panel_layout->addSpacing(12);
    panel_layout->addWidget(orientation_label_);
    panel_layout->addStretch(1);

    auto* hint = new QLabel(
        "This panel is ready for dataset loading, rendering modes, and BA controls.", panel);
    hint->setWordWrap(true);
    hint->setStyleSheet("color: palette(mid);");
    panel_layout->addWidget(hint);
    root_layout->addWidget(panel);
    setCentralWidget(central_widget);

    connect(open_ply_button, &QPushButton::clicked,
            this, [this]() { viewer_controller_->openPlyFile(this); });
    connect(reset_button, &QPushButton::clicked,
            viewer_controller_, &ViewerController::resetView);
    connect(align_floor_button, &QPushButton::clicked,
            viewer_controller_, &ViewerController::beginFloorAlignment);
    connect(constrained_rotation, &QCheckBox::toggled,
            viewer_controller_, &ViewerController::setConstrainedZUpNavigation);
    connect(background_button, &QPushButton::clicked, this, [this]() {
        const QColor color = QColorDialog::getColor(QColor(25, 30, 42), this,
                                                     "Choose viewport background");
        gl_widget_->setBackgroundColor(color);
    });
    connect(viewer_controller_, &ViewerController::orientationChanged, this,
            [this](float yaw, float pitch) {
        orientation_label_->setText(
            QString("Yaw: %1 | Pitch: %2 degrees")
                .arg(yaw, 0, 'f', 1)
                .arg(pitch, 0, 'f', 1));
    });
    connect(viewer_controller_, &ViewerController::loadStatusChanged, this,
            [this](const QString& text, const QString& path) {
        load_status_label_->setText(text);
        load_status_label_->setToolTip(path);
    });
    connect(viewer_controller_, &ViewerController::floorAlignmentStatusChanged,
            floor_alignment_label_, &QLabel::setText);
}
