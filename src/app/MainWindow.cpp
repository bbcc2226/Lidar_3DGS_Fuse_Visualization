#include "MainWindow.h"

#include "OpenGLWidget.h"
#include "ViewerController.h"

#include <QColorDialog>
#include <QCheckBox>
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QFrame>
#include <QHBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QSpinBox>
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
    auto* open_trajectory_button = new QPushButton("Open trajectory...", panel);
    trajectory_status_label_ = new QLabel("No trajectory loaded", panel);
    trajectory_status_label_->setWordWrap(true);
    panel_layout->addWidget(open_trajectory_button);
    panel_layout->addWidget(trajectory_status_label_);
    panel_layout->addSpacing(12);

    auto* reset_button = new QPushButton("Reset view", panel);
    auto* working_mode_label = new QLabel("Working mode", panel);
    auto* working_mode = new QComboBox(panel);
    working_mode->addItems({"Explore", "Navigate"});
    working_mode->setToolTip(
        "Explore uses free orbit controls. Navigate automatically aligns +Z "
        "and starts from a provisional indoor first-person pose.");
    floor_alignment_label_ = new QLabel("Automatic Z-up alignment not run", panel);
    floor_alignment_label_->setWordWrap(true);
    auto* background_button = new QPushButton("Background color...", panel);
    auto* exposure_label = new QLabel("Exposure (EV)", panel);
    auto* exposure_control = new QDoubleSpinBox(panel);
    exposure_control->setRange(-4.0, 4.0);
    exposure_control->setSingleStep(0.1);
    exposure_control->setDecimals(1);
    exposure_control->setValue(0.0);
    auto* low_pass_label = new QLabel("Low-pass variance", panel);
    auto* low_pass_control = new QDoubleSpinBox(panel);
    low_pass_control->setRange(0.0, 1.0);
    low_pass_control->setSingleStep(0.05);
    low_pass_control->setValue(0.3);
    low_pass_control->setToolTip(
        "Lower values look sharper but may alias or flicker.");
    auto* maximum_size_label = new QLabel("Maximum splat half-size (px)", panel);
    auto* maximum_size_control = new QSpinBox(panel);
    maximum_size_control->setRange(16, 256);
    maximum_size_control->setSingleStep(16);
    maximum_size_control->setValue(256);
    maximum_size_control->setToolTip(
        "Lower values suppress very large projected floaters.");
    auto* minimum_opacity_label = new QLabel("Minimum splat opacity", panel);
    auto* minimum_opacity_control = new QDoubleSpinBox(panel);
    minimum_opacity_control->setRange(0.0, 0.2);
    minimum_opacity_control->setSingleStep(0.01);
    minimum_opacity_control->setDecimals(2);
    minimum_opacity_control->setValue(0.0);
    minimum_opacity_control->setToolTip(
        "Raise slightly to hide weak floaters; high values create holes.");
    auto* highlight_large_splats = new QCheckBox(
        "Highlight large splats", panel);
    highlight_large_splats->setToolTip(
        "Tint splats red when their projected radius approaches the maximum "
        "splat-size limit. This is a diagnostic and does not change geometry.");
    auto* suppress_oversized_splats = new QCheckBox(
        "Suppress oversized splats", panel);
    suppress_oversized_splats->setToolTip(
        "Diagnostic: fade splats beyond the selected maximum size. This can "
        "remove valid scene detail, so normal rendering leaves it disabled.");
    auto* show_viewport_grid = new QCheckBox("Show viewport grid", panel);
    show_viewport_grid->setToolTip(
        "Show the decorative screen-space grid and viewport title.");
    orientation_label_ = new QLabel("Yaw: 0.0 | Pitch: 0.0 degrees", panel);
    panel_layout->addWidget(reset_button);
    panel_layout->addWidget(working_mode_label);
    panel_layout->addWidget(working_mode);
    panel_layout->addWidget(floor_alignment_label_);
    panel_layout->addWidget(background_button);
    panel_layout->addWidget(exposure_label);
    panel_layout->addWidget(exposure_control);
    panel_layout->addWidget(low_pass_label);
    panel_layout->addWidget(low_pass_control);
    panel_layout->addWidget(maximum_size_label);
    panel_layout->addWidget(maximum_size_control);
    panel_layout->addWidget(minimum_opacity_label);
    panel_layout->addWidget(minimum_opacity_control);
    panel_layout->addWidget(highlight_large_splats);
    panel_layout->addWidget(suppress_oversized_splats);
    panel_layout->addWidget(show_viewport_grid);
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
    connect(open_trajectory_button, &QPushButton::clicked,
            this, [this]() { viewer_controller_->openTrajectoryFile(this); });
    connect(reset_button, &QPushButton::clicked,
            viewer_controller_, &ViewerController::resetView);
    connect(working_mode,
            QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, [this](int index) {
                viewer_controller_->setConstrainedZUpNavigation(index == 1);
            });
    connect(background_button, &QPushButton::clicked, this, [this]() {
        const QColor color = QColorDialog::getColor(QColor(25, 30, 42), this,
                                                     "Choose viewport background");
        gl_widget_->setBackgroundColor(color);
    });
    connect(exposure_control,
            QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            this, [this](double value) {
                gl_widget_->setExposure(static_cast<float>(value));
            });
    connect(low_pass_control,
            QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            this, [this](double value) {
                gl_widget_->setLowPassVariance(static_cast<float>(value));
            });
    connect(maximum_size_control,
            QOverload<int>::of(&QSpinBox::valueChanged),
            this, [this](int value) {
                gl_widget_->setMaximumSplatHalfSize(static_cast<float>(value));
            });
    connect(minimum_opacity_control,
            QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            this, [this](double value) {
                gl_widget_->setMinimumSplatOpacity(static_cast<float>(value));
            });
    connect(highlight_large_splats, &QCheckBox::toggled,
            gl_widget_, &OpenGLWidget::setLargeSplatHighlightEnabled);
    connect(suppress_oversized_splats, &QCheckBox::toggled,
            gl_widget_, &OpenGLWidget::setOversizedSplatSuppressionEnabled);
    connect(show_viewport_grid, &QCheckBox::toggled,
            gl_widget_, &OpenGLWidget::setViewportOverlayEnabled);
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
    connect(viewer_controller_, &ViewerController::trajectoryStatusChanged, this,
            [this](const QString& text, const QString& path) {
                trajectory_status_label_->setText(text);
                trajectory_status_label_->setToolTip(path);
            });
}
