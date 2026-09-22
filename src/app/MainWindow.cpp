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
#include <QLineEdit>
#include <QListWidget>
#include <QPlainTextEdit>
#include <QPushButton>
#include <QProgressDialog>
#include <QApplication>
#include <QMessageBox>
#include <QSpinBox>
#include <QSlider>
#include <QShortcut>
#include <QScrollArea>
#include <QSignalBlocker>
#include <QTabWidget>
#include <QTimer>
#include <QVBoxLayout>
#include <QWidget>

#include <algorithm>

MainWindow::MainWindow(bool demo_mode, QWidget* parent, const QString& lidar_path)
    : QMainWindow(parent), demo_mode_(demo_mode)
{
    setWindowTitle(demo_mode_ ? "chabot-gpt — Navigation Demo"
                              : "chabot-gpt");
    setWindowFlags(windowFlags() | Qt::WindowMinMaxButtonsHint |
                   Qt::WindowMaximizeButtonHint);
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

    auto* tabs = new QTabWidget(panel);
    auto* explore_tab = new QWidget(tabs);
    auto* navigate_tab = new QScrollArea(tabs);
    auto* navigate_content = new QWidget(navigate_tab);
    navigate_tab->setWidget(navigate_content);
    navigate_tab->setWidgetResizable(true);
    navigate_tab->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    navigate_tab->setFrameShape(QFrame::NoFrame);
    auto* semantic_tab = new QWidget(tabs);
    auto* explore_layout = new QVBoxLayout(explore_tab);
    auto* navigate_layout = new QVBoxLayout(navigate_content);
    auto* semantic_layout = new QVBoxLayout(semantic_tab);
    explore_layout->setContentsMargins(8, 10, 8, 8);
    navigate_layout->setContentsMargins(8, 10, 8, 8);
    semantic_layout->setContentsMargins(8, 10, 8, 8);
    tabs->addTab(explore_tab, "Explore & Display");
    tabs->addTab(navigate_tab, "Navigate & Path");
    tabs->addTab(semantic_tab, "Semantic");
    panel_layout->addWidget(tabs, 1);

    auto* open_ply_button = new QPushButton("Open PLY...", panel);
    load_status_label_ = new QLabel("No point cloud loaded", panel);
    load_status_label_->setWordWrap(true);
    explore_layout->addWidget(open_ply_button);
    explore_layout->addWidget(load_status_label_);
    auto* source_switch = new QSlider(Qt::Horizontal, panel);
    source_switch->setObjectName("sourceViewSwitch");
    source_switch->setRange(0, 1);
    source_switch->setValue(0);
    source_switch->setFixedWidth(62);
    source_switch->setStyleSheet(
        "QSlider::groove:horizontal { height: 18px; border-radius: 9px; "
        "background: #58616b; border: 1px solid #7c8792; }"
        "QSlider::handle:horizontal { width: 26px; margin: -5px 0; "
        "border-radius: 13px; background: #e8f4f5; border: 2px solid #369f99; }");
    source_switch->setToolTip(
        "Switch between the reconstructed 3DGS scene and the merged colored LiDAR scene.");
    auto* source_3dgs_label = new QLabel("3DGS", panel);
    auto* source_lidar_label = new QLabel("Colored LiDAR", panel);
    auto* lidar_spinner = new QLabel("◐", panel);
    lidar_spinner->setObjectName("lidarLoadingSpinner");
    lidar_spinner->setStyleSheet("font-size: 18px; color: #65d5cb;");
    lidar_spinner->setToolTip("Preparing LiDAR points...");
    lidar_spinner->hide();
    auto* lidar_spinner_timer = new QTimer(panel);
    lidar_spinner_timer->setInterval(120);
    const QStringList spinner_frames = {"◐", "◓", "◑", "◒"};
    auto* spinner_frame = new int(0);
    connect(lidar_spinner_timer, &QTimer::timeout, this,
            [lidar_spinner, spinner_frames, spinner_frame]() {
                *spinner_frame = (*spinner_frame + 1) % spinner_frames.size();
                lidar_spinner->setText(spinner_frames.at(*spinner_frame));
            });
    auto* source_row = new QHBoxLayout;
    source_row->addWidget(source_3dgs_label);
    source_row->addWidget(source_switch);
    source_row->addWidget(source_lidar_label);
    source_row->addWidget(lidar_spinner);
    source_row->addStretch(1);
    explore_layout->addLayout(source_row);
    auto* open_trajectory_button = new QPushButton("Open trajectory...", panel);
    trajectory_status_label_ = new QLabel("No trajectory loaded", panel);
    trajectory_status_label_->setWordWrap(true);
    navigate_layout->addWidget(open_trajectory_button);
    navigate_layout->addWidget(trajectory_status_label_);
    auto* save_path_button = new QPushButton("Save path...", panel);
    manual_path_status_label_ = new QLabel("Manual path not saved", panel);
    manual_path_status_label_->setWordWrap(true);
    navigate_layout->addWidget(save_path_button);
    navigate_layout->addWidget(manual_path_status_label_);
    auto* close_free_zone_button = new QPushButton("Close free-zone polygon", panel);
    auto* reopen_free_zone_button = new QPushButton("Reopen polygon", panel);
    auto* join_free_zone_button = new QPushButton("Start joined polygon", panel);
    auto* add_free_zone_button = new QPushButton("Add overlapping polygon", panel);
    auto* shared_edge_count = new QSpinBox(panel);
    shared_edge_count->setRange(1, 32);
    shared_edge_count->setValue(1);
    shared_edge_count->setPrefix("Shared edges: ");
    auto* shared_edge_direction = new QComboBox(panel);
    shared_edge_direction->addItems({"Boundary direction: forward",
                                     "Boundary direction: reverse"});
    auto* clear_free_zone_button = new QPushButton("Clear free zone", panel);
    auto* save_free_zone_button = new QPushButton("Save free zone...", panel);
    auto* load_free_zone_button = new QPushButton("Load free zone...", panel);
    free_zone_status_label_ = new QLabel("No free zone created", panel);
    free_zone_status_label_->setWordWrap(true);
    navigate_layout->addWidget(close_free_zone_button);
    navigate_layout->addWidget(reopen_free_zone_button);
    navigate_layout->addWidget(shared_edge_count);
    navigate_layout->addWidget(shared_edge_direction);
    navigate_layout->addWidget(join_free_zone_button);
    navigate_layout->addWidget(add_free_zone_button);
    navigate_layout->addWidget(clear_free_zone_button);
    navigate_layout->addWidget(save_free_zone_button);
    navigate_layout->addWidget(load_free_zone_button);
    navigate_layout->addWidget(free_zone_status_label_);
    auto* initialize_cells_button = new QPushButton("Create cells from path", panel);
    auto* clear_cells_button = new QPushButton("Clear walkable cells", panel);
    auto* save_cells_button = new QPushButton("Save walkable cells...", panel);
    auto* load_cells_button = new QPushButton("Load walkable cells...", panel);
    auto* show_cells_checkbox = new QCheckBox("Show walkable cells", panel);
    show_cells_checkbox->setChecked(true);
    navigate_layout->addWidget(initialize_cells_button);
    navigate_layout->addWidget(clear_cells_button);
    navigate_layout->addWidget(save_cells_button);
    navigate_layout->addWidget(load_cells_button);
    navigate_layout->addWidget(show_cells_checkbox);
    auto* load_robot_path_button = new QPushButton("Load robot path...", panel);
    auto* clear_robot_path_button = new QPushButton("Clear loaded path", panel);
    auto* play_pause_button = new QPushButton("Play / Pause", panel);
    auto* stop_playback_button = new QPushButton("Stop", panel);
    auto* playback_speed = new QDoubleSpinBox(panel);
    playback_speed->setRange(0.05, 2.0);
    playback_speed->setSingleStep(0.05);
    playback_speed->setDecimals(2);
    playback_speed->setSuffix(" m/s");
    playback_speed->setValue(1.0 / 6.0);
    auto* navigation_speed = new QDoubleSpinBox(panel);
    navigation_speed->setRange(0.05, 2.0);
    navigation_speed->setSingleStep(0.05);
    navigation_speed->setDecimals(2);
    navigation_speed->setSuffix(" m/s");
    navigation_speed->setValue(playback_speed->value());
    navigation_speed->hide();
    robot_playback_status_label_ = new QLabel("No robot path loaded", panel);
    robot_playback_status_label_->setWordWrap(true);
    navigate_layout->addWidget(load_robot_path_button);
    navigate_layout->addWidget(clear_robot_path_button);
    navigate_layout->addWidget(playback_speed);
    auto* height_mode = new QComboBox(panel);
    height_mode->setObjectName("trajectoryHeightMode");
    height_mode->addItems({"Constant height", "Original trajectory height"});
    height_mode->setToolTip(
        "Original follows the height of each recorded pose. Constant uses a fixed "
        "height above the estimated floor. Pause and use Up/Down to adjust either mode.");
    navigate_layout->addWidget(new QLabel("Playback camera height", panel));
    navigate_layout->addWidget(height_mode);
    connect(height_mode, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, [this](int index) {
                viewer_controller_->setUseOriginalTrajectoryHeight(index == 1);
            });
    connect(viewer_controller_, &ViewerController::originalTrajectoryHeightChanged,
            height_mode, [height_mode](bool original) {
                const QSignalBlocker blocker(height_mode);
                height_mode->setCurrentIndex(original ? 1 : 0);
            });
    navigate_layout->addWidget(play_pause_button);
    navigate_layout->addWidget(stop_playback_button);
    navigate_layout->addWidget(robot_playback_status_label_);
    navigate_layout->addSpacing(12);

    auto* overview_button = new QPushButton("Path overview", panel);
    explore_layout->addWidget(overview_button);
    auto* reset_button = new QPushButton("Reset view", panel);
    auto* working_mode_label = new QLabel("Working mode", panel);
    auto* working_mode = new QComboBox(panel);
    working_mode->addItems({"Explore", "Navigate", "Edit Path", "Edit Free Zone",
                            "Edit Walkable Cells"});
    working_mode->setToolTip(
        "Explore uses free orbit controls. Navigate uses a Z-up first-person "
        "view. Edit Path and Edit Free Zone use a Z-up orthographic view.");
    floor_alignment_label_ = new QLabel("Automatic Z-up alignment not run", panel);
    floor_alignment_label_->setWordWrap(true);
    auto* destination_input = new QPlainTextEdit(panel);
    destination_input->setObjectName("navigationPrompt");
    destination_input->setPlaceholderText(
        "Where would you like the robot to go?");
    destination_input->setFixedHeight(92);
    destination_input->setTabChangesFocus(true);
    auto* navigate_button = new QPushButton("Navigate", panel);
    navigate_button->setObjectName("navigateAction");
    navigate_button->setToolTip(
        "Preview a route to a confirmed semantic object.");
    auto* navigation_plan_status = new QLabel(
        "Enter a request to preview a route.", panel);
    navigation_plan_status->setWordWrap(true);
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
    explore_layout->addWidget(reset_button);
    explore_layout->addWidget(background_button);
    explore_layout->addWidget(exposure_label);
    explore_layout->addWidget(exposure_control);
    explore_layout->addWidget(low_pass_label);
    explore_layout->addWidget(low_pass_control);
    explore_layout->addWidget(maximum_size_label);
    explore_layout->addWidget(maximum_size_control);
    explore_layout->addWidget(minimum_opacity_label);
    explore_layout->addWidget(minimum_opacity_control);
    explore_layout->addWidget(highlight_large_splats);
    explore_layout->addWidget(suppress_oversized_splats);
    explore_layout->addWidget(show_viewport_grid);
    explore_layout->addStretch(1);

    navigate_layout->insertWidget(0, working_mode_label);
    navigate_layout->insertWidget(1, working_mode);
    navigate_layout->insertWidget(2, floor_alignment_label_);
    navigate_layout->addSpacing(8);
    navigate_layout->addWidget(orientation_label_);
    navigate_layout->addStretch(1);

    auto* hint = new QLabel(
        "Edit Path: left-click to append, drag a point to move it, and "
        "right-click or Delete to remove it. Wheel zooms the top view.", panel);
    hint->setWordWrap(true);
    hint->setStyleSheet("color: palette(mid);");
    navigate_layout->addWidget(hint);
    navigate_layout->addSpacing(6);
    navigate_layout->addWidget(destination_input);
    navigate_layout->addWidget(navigate_button);
    navigate_layout->addWidget(navigation_plan_status);

    auto* open_semantic_button = new QPushButton("Open semantic database...", panel);
    auto* update_semantic_file = new QPushButton("Update current object file", panel);
    update_semantic_file->setToolTip("Save verification to the loaded object database and its review file.");
    auto* save_semantic_button = new QPushButton("Save reviews", panel);
    auto* save_semantic_as_button = new QPushButton("Save reviews as...", panel);
    auto* show_semantic_boxes = new QCheckBox("Show selected object box", panel);
    show_semantic_boxes->setChecked(true);
    auto* show_all_semantic_labels = new QCheckBox("Show all objects in view", panel);
    show_all_semantic_labels->setToolTip(
        "Show labels for all objects in the current view for screenshots, "
        "ignoring the name filter and keeping the camera in place. "
        "Enable 2D boxes to add approximate boundaries.");
    auto* verified_semantic_only = new QCheckBox("Only verified objects", panel);
    verified_semantic_only->setToolTip("Only display objects marked Confirmed in the view.");
    auto* semantic_2d_boxes = new QCheckBox("Use approximate 2D boxes", panel);
    semantic_2d_boxes->setToolTip(
        "Draw screen-aligned rectangles around projected object bounds. "
        "Also adds boxes when showing all objects.");
    auto* semantic_filter = new QLineEdit(panel);
    semantic_filter->setPlaceholderText("Filter by object name...");
    auto* semantic_list = new QListWidget(panel);
    semantic_list->setAlternatingRowColors(true);
    auto* semantic_details = new QLabel("Select an object to inspect it.", panel);
    semantic_details->setWordWrap(true);
    semantic_details->setTextInteractionFlags(Qt::TextSelectableByMouse);
    auto* review_row = new QHBoxLayout;
    auto* confirm_button = new QPushButton("Confirm", panel);
    auto* uncertain_button = new QPushButton("Unsure", panel);
    auto* incorrect_button = new QPushButton("Incorrect", panel);
    review_row->addWidget(confirm_button);
    review_row->addWidget(uncertain_button);
    review_row->addWidget(incorrect_button);
    auto* semantic_status = new QLabel("Loading default semantic database...", panel);
    semantic_status->setWordWrap(true);
    semantic_layout->addWidget(open_semantic_button);
    auto* semantic_save_row = new QHBoxLayout;
    semantic_save_row->addWidget(save_semantic_button);
    semantic_save_row->addWidget(save_semantic_as_button);
    semantic_layout->addLayout(semantic_save_row);
    semantic_layout->addWidget(update_semantic_file);
    semantic_layout->addWidget(show_semantic_boxes);
    semantic_layout->addWidget(show_all_semantic_labels);
    semantic_layout->addWidget(verified_semantic_only);
    semantic_layout->addWidget(semantic_2d_boxes);
    semantic_layout->addWidget(semantic_filter);
    semantic_layout->addWidget(semantic_list, 1);
    semantic_layout->addWidget(semantic_details);
    semantic_layout->addLayout(review_row);
    semantic_layout->addWidget(semantic_status);

    root_layout->addWidget(panel);
    setCentralWidget(central_widget);

    connect(open_ply_button, &QPushButton::clicked,
            this, [this]() { viewer_controller_->openPlyFile(this); });
    connect(source_switch, &QSlider::valueChanged, this,
            [this, source_3dgs_label, source_lidar_label, lidar_path,
             lidar_spinner, lidar_spinner_timer](int value) {
                source_3dgs_label->setStyleSheet(value == 0
                    ? "font-weight: bold; color: #65d5cb;" : "color: palette(text);");
                source_lidar_label->setStyleSheet(value == 1
                    ? "font-weight: bold; color: #65d5cb;" : "color: palette(text);");
                lidar_spinner->setVisible(value == 1);
                if (value == 1) lidar_spinner_timer->start();
                else lidar_spinner_timer->stop();
                const QString selected_lidar_path = lidar_path.isEmpty()
                    ? QStringLiteral(PROJECT_ROOT_DIR) +
                        "/data/lidar_related/dense_lidar_rgb_latest_50MiB.ply"
                    : lidar_path;
                if (value == 1) viewer_controller_->loadLidarPointCloud(selected_lidar_path);
                else viewer_controller_->setLidarViewEnabled(false);
            });
    connect(viewer_controller_, &ViewerController::loadStatusChanged, this,
            [lidar_spinner, lidar_spinner_timer](const QString& text, const QString&) {
                if (!text.contains("LiDAR", Qt::CaseInsensitive)) return;
                const bool loading = text.contains("Loading", Qt::CaseInsensitive);
                lidar_spinner->setVisible(loading);
                if (!loading) lidar_spinner_timer->stop();
            });
    source_3dgs_label->setStyleSheet("font-weight: bold; color: #65d5cb;");
    if (!lidar_path.isEmpty())
        QTimer::singleShot(0, source_switch, [source_switch]() {
            source_switch->setValue(1);
        });
    connect(open_trajectory_button, &QPushButton::clicked,
            this, [this]() { viewer_controller_->openTrajectoryFile(this); });
    connect(save_path_button, &QPushButton::clicked,
            this, [this]() { viewer_controller_->saveManualPath(this); });
    connect(close_free_zone_button, &QPushButton::clicked,
            viewer_controller_, &ViewerController::closeFreeZone);
    connect(reopen_free_zone_button, &QPushButton::clicked,
            viewer_controller_, &ViewerController::reopenFreeZone);
    connect(join_free_zone_button, &QPushButton::clicked, this,
            [this, shared_edge_count, shared_edge_direction]() {
                viewer_controller_->startJoinedFreeZone(
                    shared_edge_count->value(),
                    shared_edge_direction->currentIndex() == 1);
            });
    connect(add_free_zone_button, &QPushButton::clicked,
            viewer_controller_, &ViewerController::startAdditiveFreeZone);
    connect(clear_free_zone_button, &QPushButton::clicked,
            viewer_controller_, &ViewerController::clearFreeZone);
    connect(save_free_zone_button, &QPushButton::clicked, this,
            [this]() { viewer_controller_->saveFreeZone(this); });
    connect(load_free_zone_button, &QPushButton::clicked, this,
            [this]() { viewer_controller_->loadFreeZone(this); });
    connect(initialize_cells_button, &QPushButton::clicked,
            viewer_controller_, &ViewerController::initializeWalkableCellsFromPath);
    connect(clear_cells_button, &QPushButton::clicked,
            viewer_controller_, &ViewerController::clearWalkableCells);
    connect(save_cells_button, &QPushButton::clicked, this,
            [this]() { viewer_controller_->saveWalkableCells(this); });
    connect(load_cells_button, &QPushButton::clicked, this,
            [this]() { viewer_controller_->loadWalkableCells(this); });
    connect(show_cells_checkbox, &QCheckBox::toggled,
            viewer_controller_, &ViewerController::setWalkableCellsVisible);
    connect(load_robot_path_button, &QPushButton::clicked,
            this, [this]() { viewer_controller_->loadRobotPath(this); });
    connect(clear_robot_path_button, &QPushButton::clicked,
            viewer_controller_, &ViewerController::clearLoadedPath);
    connect(play_pause_button, &QPushButton::clicked,
            this, [this, working_mode]() {
                if (working_mode->currentIndex() != 1)
                    working_mode->setCurrentIndex(1);
                viewer_controller_->toggleRobotPlayback();
            });
    connect(stop_playback_button, &QPushButton::clicked,
            viewer_controller_, &ViewerController::stopRobotPlayback);
    connect(navigate_button, &QPushButton::clicked, this,
            [this, destination_input, working_mode]() {
                if (working_mode->currentIndex() != 1)
                    working_mode->setCurrentIndex(1);
                viewer_controller_->planNavigationRequest(
                    destination_input->toPlainText());
            });
    connect(playback_speed,
            QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            this, [this, navigation_speed](double speed) {
                navigation_speed->setValue(speed);
                viewer_controller_->setRobotPlaybackSpeed(
                    static_cast<float>(speed));
            });
    connect(navigation_speed,
            QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            this, [this, playback_speed](double speed) {
                playback_speed->setValue(speed);
                viewer_controller_->setRobotPlaybackSpeed(
                    static_cast<float>(speed));
            });
    connect(overview_button, &QPushButton::clicked, this, [this, working_mode]() {
        working_mode->setCurrentIndex(0);
        viewer_controller_->showPathOverview();
    });
    connect(reset_button, &QPushButton::clicked,
            viewer_controller_, &ViewerController::resetView);
    connect(working_mode,
            QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, [this](int index) {
                viewer_controller_->setPathEditingEnabled(index == 2);
                viewer_controller_->setFreeZoneEditingEnabled(index == 3);
                viewer_controller_->setWalkableCellEditingEnabled(index == 4);
                if (index == 0) viewer_controller_->showPathOverview();
                else if (index == 1) viewer_controller_->setConstrainedZUpNavigation(true);
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
    connect(open_semantic_button, &QPushButton::clicked, this,
            [this]() { viewer_controller_->loadSemanticDatabase(this); });
    connect(update_semantic_file, &QPushButton::clicked, this,
            [this]() { viewer_controller_->saveSemanticObjectFile(this); });
    connect(save_semantic_button, &QPushButton::clicked, this,
            [this]() { viewer_controller_->saveSemanticReviews(this); });
    connect(save_semantic_as_button, &QPushButton::clicked, this,
            [this]() { viewer_controller_->saveSemanticReviews(this, true); });
    const auto update_semantic_display =
        [this, tabs, semantic_tab, show_semantic_boxes, show_all_semantic_labels]() {
            const bool semantic_mode = tabs->currentWidget() == semantic_tab;
            const bool labels_only = show_all_semantic_labels->isChecked();
            show_semantic_boxes->setEnabled(!labels_only);
            gl_widget_->setSemanticLabelsOnly(labels_only);
            viewer_controller_->setSemanticVerificationMode(semantic_mode && !labels_only);
            viewer_controller_->setSemanticObjectsVisible(
                semantic_mode && (labels_only || show_semantic_boxes->isChecked()));
        };
    connect(show_semantic_boxes, &QCheckBox::toggled, this, update_semantic_display);
    connect(show_all_semantic_labels, &QCheckBox::toggled, this, update_semantic_display);
    connect(verified_semantic_only, &QCheckBox::toggled,
            gl_widget_, &OpenGLWidget::setSemanticVerifiedOnly);
    connect(semantic_2d_boxes, &QCheckBox::toggled,
            gl_widget_, &OpenGLWidget::setSemantic2DBoxes);
    auto* rotate_left = new QShortcut(QKeySequence(Qt::Key_Left), this);
    auto* rotate_right = new QShortcut(QKeySequence(Qt::Key_Right), this);
    rotate_left->setContext(Qt::WindowShortcut);
    rotate_right->setContext(Qt::WindowShortcut);
    rotate_left->setEnabled(false);
    rotate_right->setEnabled(false);
    connect(rotate_left, &QShortcut::activated, this,
            [this]() { viewer_controller_->rotateSemanticView(-5.0f); });
    connect(rotate_right, &QShortcut::activated, this,
            [this]() { viewer_controller_->rotateSemanticView(5.0f); });
    connect(tabs, &QTabWidget::currentChanged, this,
            [this, tabs, semantic_tab, show_all_semantic_labels, semantic_list,
             rotate_left, rotate_right, update_semantic_display](int index) {
                const bool semantic_mode = tabs->widget(index) == semantic_tab;
                rotate_left->setEnabled(semantic_mode);
                rotate_right->setEnabled(semantic_mode);
                update_semantic_display();
                if (semantic_mode && !show_all_semantic_labels->isChecked() &&
                    semantic_list->currentItem())
                    viewer_controller_->selectSemanticObject(
                        semantic_list->currentItem()->data(Qt::UserRole).toInt());
            });
    connect(semantic_filter, &QLineEdit::textChanged, this,
            [viewer_controller = viewer_controller_, semantic_list](const QString& text) {
                viewer_controller->setSemanticClassFilter(text);
                for (int row = 0; row < semantic_list->count(); ++row)
                    semantic_list->item(row)->setHidden(
                        !semantic_list->item(row)->text().contains(text, Qt::CaseInsensitive));
            });
    connect(semantic_list, &QListWidget::currentItemChanged, this,
            [this](QListWidgetItem* current) {
                if (current) viewer_controller_->selectSemanticObject(
                    current->data(Qt::UserRole).toInt());
            });
    const auto review_selected = [this, semantic_list](SemanticReviewStatus status) {
        if (auto* item = semantic_list->currentItem())
            viewer_controller_->reviewSemanticObject(
                item->data(Qt::UserRole).toInt(), status);
    };
    connect(confirm_button, &QPushButton::clicked, this,
            [review_selected]() { review_selected(SemanticReviewStatus::Confirmed); });
    connect(uncertain_button, &QPushButton::clicked, this,
            [review_selected]() { review_selected(SemanticReviewStatus::Uncertain); });
    connect(incorrect_button, &QPushButton::clicked, this,
            [review_selected]() { review_selected(SemanticReviewStatus::Incorrect); });
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
    connect(viewer_controller_, &ViewerController::manualPathStatusChanged, this,
            [this](const QString& text, const QString& path) {
                manual_path_status_label_->setText(text);
                manual_path_status_label_->setToolTip(path);
            });
    connect(viewer_controller_, &ViewerController::freeZoneStatusChanged, this,
            [this](const QString& text, const QString& path) {
                free_zone_status_label_->setText(text);
                if (!path.isEmpty()) free_zone_status_label_->setToolTip(path);
            });
    connect(viewer_controller_, &ViewerController::robotPlaybackStatusChanged,
            this, [this](const QString& text, const QString& path) {
                robot_playback_status_label_->setText(text);
                if (!path.isEmpty()) robot_playback_status_label_->setToolTip(path);
            });
    connect(viewer_controller_, &ViewerController::semanticDatabaseChanged,
            this, [this, semantic_list, semantic_status](const QString& text,
                                                         const QString& path) {
                semantic_status->setText(text);
                if (!path.isEmpty()) semantic_status->setToolTip(path);
                if (path.isEmpty()) return;
                semantic_list->clear();
                for (const SemanticObject& object : viewer_controller_->semanticObjects()) {
                    auto* item = new QListWidgetItem(
                        QString("%1  [#%2]").arg(object.name).arg(object.id), semantic_list);
                    item->setData(Qt::UserRole, object.id);
                }
            });
    connect(viewer_controller_, &ViewerController::semanticSelectionChanged,
            this, [this, semantic_list, semantic_details, semantic_filter](int object_id) {
                const SemanticObject* selected = nullptr;
                for (const SemanticObject& object : viewer_controller_->semanticObjects())
                    if (object.id == object_id) { selected = &object; break; }
                if (!selected) { semantic_details->setText("Select an object to inspect it."); return; }
                for (int row = 0; row < semantic_list->count(); ++row) {
                    auto* item = semantic_list->item(row);
                    if (item->data(Qt::UserRole).toInt() == object_id) {
                        if (item->isHidden()) semantic_filter->clear();
                        const QSignalBlocker blocker(semantic_list);
                        semantic_list->setCurrentItem(item);
                        semantic_list->scrollToItem(item, QAbstractItemView::EnsureVisible);
                        QColor color(190, 194, 200);
                        if (selected->review == SemanticReviewStatus::Confirmed) color = QColor(55, 220, 105);
                        else if (selected->review == SemanticReviewStatus::Uncertain) color = QColor(255, 195, 55);
                        else if (selected->review == SemanticReviewStatus::Incorrect) color = QColor(245, 75, 75);
                        item->setForeground(color);
                        break;
                    }
                }
                const QVector3D size = selected->bounds_max_world - selected->bounds_min_world;
                semantic_details->setText(QString(
                    "<b>%1 [#%2]</b><br>%3<br>Confidence: %4 (%5)<br>"
                    "Size: %6 × %7 × %8 m<br>Observations: %9<br>Images: %10")
                    .arg(selected->name).arg(selected->id).arg(selected->description)
                    .arg(selected->confidence).arg(selected->confidence_score, 0, 'f', 2)
                    .arg(size.x(), 0, 'f', 2).arg(size.y(), 0, 'f', 2)
                    .arg(size.z(), 0, 'f', 2).arg(selected->observation_count)
                    .arg(selected->supporting_images.mid(0, 4).join(", ")));
            });
    connect(viewer_controller_, &ViewerController::navigationPlanStatusChanged,
            navigation_plan_status, &QLabel::setText);
    viewer_controller_->setSemanticObjectsVisible(false);
    viewer_controller_->loadDefaultSemanticDatabase();

    if (demo_mode_) {
        // Keep the full editor untouched for the normal launch. Demo mode
        // exposes only room exploration/playback and destination navigation.
        tabs->setTabText(tabs->indexOf(explore_tab), "Explore");
        tabs->setTabText(tabs->indexOf(navigate_tab), "Navigate");
        tabs->removeTab(tabs->indexOf(semantic_tab));

        open_ply_button->hide();
        load_status_label_->hide();
        open_trajectory_button->hide();
        trajectory_status_label_->hide();
        save_path_button->hide();
        manual_path_status_label_->hide();
        close_free_zone_button->hide();
        reopen_free_zone_button->hide();
        join_free_zone_button->hide();
        add_free_zone_button->hide();
        shared_edge_count->hide();
        shared_edge_direction->hide();
        clear_free_zone_button->hide();
        save_free_zone_button->hide();
        load_free_zone_button->hide();
        free_zone_status_label_->hide();
        initialize_cells_button->hide();
        clear_cells_button->hide();
        save_cells_button->hide();
        load_cells_button->hide();
        show_cells_checkbox->hide();
        load_robot_path_button->hide();
        clear_robot_path_button->hide();
        hint->hide();
        working_mode_label->hide();
        working_mode->hide();

        background_button->hide();
        exposure_label->hide();
        exposure_control->hide();
        low_pass_label->hide();
        low_pass_control->hide();
        maximum_size_label->hide();
        maximum_size_control->hide();
        minimum_opacity_label->hide();
        minimum_opacity_control->hide();
        highlight_large_splats->hide();
        suppress_oversized_splats->hide();
        show_viewport_grid->hide();

        // Playback follows the preloaded hand-authored exploration route.
        // Moving these existing widgets preserves their original connections,
        // speed behavior, and play/pause/resume semantics.
        const int before_stretch = std::max(0, explore_layout->count() - 1);
        explore_layout->insertWidget(before_stretch, playback_speed);
        explore_layout->insertWidget(before_stretch + 1, play_pause_button);
        explore_layout->insertWidget(before_stretch + 2, stop_playback_button);
        explore_layout->insertWidget(before_stretch + 3,
                                     robot_playback_status_label_);
        auto* show_demo_semantics = new QCheckBox("Show semantic objects (2D boxes)", panel);
        show_demo_semantics->setObjectName("demoSemanticOverlay");
        show_demo_semantics->setToolTip(
            "Show verified (Confirmed), high-confidence objects while exploring. "
            "Click a box to highlight it and view its information.");
        auto* demo_object_details = new QLabel("Click an object box to view its description.", panel);
        demo_object_details->setObjectName("demoSemanticDetails");
        demo_object_details->setWordWrap(true);
        demo_object_details->setTextFormat(Qt::PlainText);
        demo_object_details->setTextInteractionFlags(Qt::TextSelectableByMouse);
        demo_object_details->setAlignment(Qt::AlignTop | Qt::AlignLeft);
        auto* demo_details_scroll = new QScrollArea(panel);
        demo_details_scroll->setWidgetResizable(true);
        demo_details_scroll->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
        demo_details_scroll->setWidget(demo_object_details);
        demo_details_scroll->setMinimumHeight(120);
        demo_details_scroll->setMaximumHeight(260);
        demo_details_scroll->hide();
        explore_layout->insertWidget(before_stretch + 4, show_demo_semantics);
        explore_layout->insertWidget(before_stretch + 5, demo_details_scroll);
        const auto update_demo_semantics =
            [this, tabs, explore_tab, show_demo_semantics, demo_details_scroll]() {
                const bool visible = tabs->currentWidget() == explore_tab &&
                                     show_demo_semantics->isChecked();
                viewer_controller_->setSemanticVerificationMode(false);
                gl_widget_->setSemanticLabelsOnly(true);
                gl_widget_->setSemantic2DBoxes(true);
                gl_widget_->setSemanticVerifiedOnly(true);
                gl_widget_->setSemanticHighConfidenceOnly(true);
                viewer_controller_->setSemanticObjectsVisible(visible);
                demo_details_scroll->setVisible(visible);
            };
        connect(show_demo_semantics, &QCheckBox::toggled, this, update_demo_semantics);
        connect(tabs, &QTabWidget::currentChanged, this, update_demo_semantics);
        connect(viewer_controller_, &ViewerController::semanticSelectionChanged, this,
                [this, demo_object_details](int object_id) {
                    for (const SemanticObject& object : viewer_controller_->semanticObjects()) {
                        if (object.id != object_id) continue;
                        demo_object_details->setText(object.description.isEmpty()
                            ? "No description available." : object.description);
                        return;
                    }
                    demo_object_details->setText("Click an object box to view its description.");
                });
        update_demo_semantics();

        auto* navigation_speed_label = new QLabel("Navigation speed", panel);
        const int prompt_index = navigate_layout->indexOf(destination_input);
        navigate_layout->insertWidget(prompt_index, navigation_speed_label);
        navigate_layout->insertWidget(prompt_index + 1, navigation_speed);
        navigation_speed->show();

        connect(tabs, &QTabWidget::currentChanged, this,
                [this, tabs, explore_tab](int) {
                    gl_widget_->setManualPathVisible(
                        tabs->currentWidget() == explore_tab);
                });
        gl_widget_->setManualPathVisible(true);
        gl_widget_->setWalkableCellsVisible(false);
    }
}

void MainWindow::loadDemoAssets()
{
    if (!demo_mode_) return;
    QProgressDialog progress("Reading 3DGS scene...", QString(), 0, 100, this);
    progress.setWindowTitle("Loading demo");
    progress.setWindowModality(Qt::ApplicationModal);
    progress.setCancelButton(nullptr);
    progress.setMinimumDuration(0);
    progress.setAutoClose(false);
    progress.setAutoReset(false);
    progress.setValue(0);
    progress.show();
    connect(viewer_controller_, &ViewerController::loadProgressChanged, &progress,
            [&progress](int percent, const QString& stage) {
                progress.setLabelText(stage);
                progress.setValue(percent);
                QApplication::processEvents(QEventLoop::ExcludeUserInputEvents);
            });
    QApplication::processEvents(QEventLoop::ExcludeUserInputEvents);
    const QString data = QStringLiteral(PROJECT_ROOT_DIR) + "/data/";
    if (!viewer_controller_->loadPlyFile(data + "point_cloud_curr_best.ply")) {
        progress.close();
        QMessageBox::warning(this, "Demo loading failed", load_status_label_->text());
        return;
    }
    progress.setLabelText("Preparing exploration path...");
    QApplication::processEvents(QEventLoop::ExcludeUserInputEvents);
    viewer_controller_->setConstrainedZUpNavigation(true);
    viewer_controller_->setConstrainedZUpNavigation(false);
    if (!viewer_controller_->loadRobotPathFile(data + "9_12_manual_path.json"))
        return;
    viewer_controller_->loadWalkableCellsFile(
        data + "9_10walkable_cells.json");
    viewer_controller_->showPathOverview();
}
