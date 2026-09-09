#include "MainWindow.h"

#include <QApplication>
#include <QFont>
#include <QPalette>
#include <QStyleFactory>
#include <QSurfaceFormat>
#include <QtGlobal>

int main(int argc, char* argv[])
{
    QSurfaceFormat format;
    format.setRenderableType(QSurfaceFormat::OpenGL);
    const bool enable_experimental_gpu_reorder =
        qEnvironmentVariableIntValue("LIDAR_3DGS_GPU_REORDER") == 1;
    format.setVersion(enable_experimental_gpu_reorder ? 4 : 3,
                      enable_experimental_gpu_reorder ? 3 : 3);
    format.setProfile(QSurfaceFormat::CoreProfile);
    format.setDepthBufferSize(24);
    // Gaussian footprints are analytically antialiased in the fragment
    // shader. Multisampling this pass adds cost and softens the resolved image.
    format.setSamples(0);
    QSurfaceFormat::setDefaultFormat(format);

    QApplication application(argc, argv);
    application.setApplicationName("3DGS Qt OpenGL Viewer");
    application.setOrganizationName("Lidar3DGS");
    application.setStyle(QStyleFactory::create("Fusion"));
    application.setFont(QFont("Noto Sans", 10));

    QPalette palette;
    palette.setColor(QPalette::Window, QColor(34, 36, 40));
    palette.setColor(QPalette::WindowText, QColor(232, 234, 237));
    palette.setColor(QPalette::Base, QColor(24, 26, 29));
    palette.setColor(QPalette::AlternateBase, QColor(42, 45, 50));
    palette.setColor(QPalette::ToolTipBase, QColor(46, 49, 54));
    palette.setColor(QPalette::ToolTipText, QColor(245, 246, 247));
    palette.setColor(QPalette::Text, QColor(226, 229, 232));
    palette.setColor(QPalette::Button, QColor(52, 55, 61));
    palette.setColor(QPalette::ButtonText, QColor(238, 240, 242));
    palette.setColor(QPalette::BrightText, QColor(255, 110, 110));
    palette.setColor(QPalette::Highlight, QColor(54, 160, 154));
    palette.setColor(QPalette::HighlightedText, QColor(255, 255, 255));
    palette.setColor(QPalette::Link, QColor(91, 205, 198));
    palette.setColor(QPalette::Disabled, QPalette::Text, QColor(116, 120, 126));
    palette.setColor(
        QPalette::Disabled, QPalette::ButtonText, QColor(116, 120, 126));
    application.setPalette(palette);

    application.setStyleSheet(R"QSS(
        QWidget {
            color: #e8eaed;
            font-family: "Noto Sans", "DejaVu Sans", sans-serif;
            font-size: 10pt;
        }
        QMainWindow, QDialog { background: #222428; }
        QFrame[frameShape="6"] {
            background: #2a2d32;
            border: 1px solid #444850;
            border-radius: 8px;
        }
        QLabel { background: transparent; }
        QPushButton, QLineEdit, QPlainTextEdit,
        QComboBox, QSpinBox, QDoubleSpinBox {
            min-height: 25px;
            padding: 3px 8px;
            color: #eef0f2;
            background: #383c42;
            border: 1px solid #555a63;
            border-radius: 5px;
        }
        QPushButton:hover, QLineEdit:hover, QPlainTextEdit:hover, QComboBox:hover,
        QSpinBox:hover, QDoubleSpinBox:hover {
            background: #444950;
            border-color: #6d747e;
        }
        QPushButton:pressed {
            background: #2d817c;
            border-color: #48aaa4;
        }
        QPlainTextEdit#navigationPrompt {
            padding: 10px 12px;
            color: #f1f3f4;
            background: #303338;
            border: 1px solid #565b63;
            border-radius: 12px;
            selection-background-color: #369f99;
        }
        QPlainTextEdit#navigationPrompt:focus {
            background: #34383d;
            border: 1px solid #5fc0ba;
        }
        QPushButton#navigateAction {
            min-height: 30px;
            font-weight: 600;
            background: #368f89;
            border-color: #55b5ae;
        }
        QPushButton#navigateAction:hover { background: #40a39c; }
        QPushButton:disabled {
            color: #74787e;
            background: #303237;
            border-color: #3d4046;
        }
        QComboBox::drop-down {
            width: 24px;
            border: 0;
        }
        QComboBox QAbstractItemView {
            color: #eef0f2;
            background: #303339;
            border: 1px solid #555a63;
            selection-background-color: #369f99;
        }
        QTabWidget::pane {
            background: #2a2d32;
            border: 1px solid #484c54;
            border-radius: 5px;
            top: -1px;
        }
        QTabBar::tab {
            padding: 7px 9px;
            color: #aeb3ba;
            background: #292c31;
            border: 1px solid #42464d;
            border-bottom: 0;
            border-top-left-radius: 5px;
            border-top-right-radius: 5px;
        }
        QTabBar::tab:selected {
            color: #ffffff;
            background: #383c42;
            border-top: 2px solid #48aaa4;
        }
        QTabBar::tab:hover:!selected { background: #31343a; }
        QCheckBox { spacing: 7px; }
        QCheckBox::indicator {
            width: 15px;
            height: 15px;
            border: 1px solid #686e77;
            border-radius: 3px;
            background: #25272b;
        }
        QCheckBox::indicator:checked {
            background: #369f99;
            border-color: #5fc0ba;
        }
        QToolTip {
            color: #f5f6f7;
            background: #36393f;
            border: 1px solid #60656e;
            padding: 4px;
        }
    )QSS");

    MainWindow window;
    window.show();
    return application.exec();
}
