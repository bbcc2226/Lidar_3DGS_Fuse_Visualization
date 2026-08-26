#include "MainWindow.h"

#include <QApplication>
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

    MainWindow window;
    window.show();
    return application.exec();
}
