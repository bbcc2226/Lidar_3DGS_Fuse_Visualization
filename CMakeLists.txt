cmake_minimum_required(VERSION 3.16)

project(
    lio_visual_ba
    VERSION 0.1.0
    DESCRIPTION "C++ feature matching and bundle adjustment pipeline"
    LANGUAGES CXX
)

include(CTest)

option(LIO_VISUAL_BA_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)
option(LIO_VISUAL_BA_ENABLE_ASAN "Enable AddressSanitizer in supported configurations" OFF)
option(LIO_VISUAL_BA_ENABLE_UBSAN "Enable UndefinedBehaviorSanitizer in supported configurations" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(OpenCV REQUIRED COMPONENTS core imgcodecs imgproc features2d calib3d video)
find_package(Ceres REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(Threads REQUIRED)
find_package(Python3 COMPONENTS Interpreter)

option(LIO_VISUAL_BA_ENABLE_QT_VIEWER "Build the optional Qt/OpenGL viewer" OFF)
if(LIO_VISUAL_BA_ENABLE_QT_VIEWER)
    find_package(TBB REQUIRED)
    find_package(Qt5 5.12 REQUIRED COMPONENTS Core Gui Widgets OpenGL Concurrent)
endif()

# Every production target links this interface target to receive one consistent
# language and platform configuration.
add_library(lio_visual_ba_project_options INTERFACE)
add_library(lio_visual_ba::project_options ALIAS lio_visual_ba_project_options)
target_compile_features(lio_visual_ba_project_options INTERFACE cxx_std_17)

# Warnings are kept separate from project options so third-party code never
# inherits them accidentally.
add_library(lio_visual_ba_project_warnings INTERFACE)
add_library(lio_visual_ba::project_warnings ALIAS lio_visual_ba_project_warnings)

if(MSVC)
    target_compile_options(lio_visual_ba_project_warnings INTERFACE /W4 /permissive-)
    if(LIO_VISUAL_BA_WARNINGS_AS_ERRORS)
        target_compile_options(lio_visual_ba_project_warnings INTERFACE /WX)
    endif()
else()
    target_compile_options(
        lio_visual_ba_project_warnings
        INTERFACE
            -Wall
            -Wextra
            -Wpedantic
            -Wconversion
            -Wsign-conversion
            -Wshadow
            -Wnon-virtual-dtor
    )
    if(LIO_VISUAL_BA_WARNINGS_AS_ERRORS)
        target_compile_options(lio_visual_ba_project_warnings INTERFACE -Werror)
    endif()
endif()

if(LIO_VISUAL_BA_ENABLE_ASAN OR LIO_VISUAL_BA_ENABLE_UBSAN)
    if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
        message(FATAL_ERROR "Sanitizers are supported only with Clang or GCC")
    endif()

    set(lio_visual_ba_sanitizers "")
    if(LIO_VISUAL_BA_ENABLE_ASAN)
        list(APPEND lio_visual_ba_sanitizers address)
    endif()
    if(LIO_VISUAL_BA_ENABLE_UBSAN)
        list(APPEND lio_visual_ba_sanitizers undefined)
    endif()
    list(JOIN lio_visual_ba_sanitizers "," lio_visual_ba_sanitizer_list)

    target_compile_options(
        lio_visual_ba_project_options
        INTERFACE
            -fno-omit-frame-pointer
            "-fsanitize=${lio_visual_ba_sanitizer_list}"
    )
    target_link_options(
        lio_visual_ba_project_options
        INTERFACE
            -fno-omit-frame-pointer
            "-fsanitize=${lio_visual_ba_sanitizer_list}"
    )
endif()

# Centralize external dependencies. Future library and application targets
# link lio_visual_ba::dependencies instead of repeating package-specific target names.
add_library(lio_visual_ba_dependencies INTERFACE)
add_library(lio_visual_ba::dependencies ALIAS lio_visual_ba_dependencies)
target_include_directories(
    lio_visual_ba_dependencies
    SYSTEM
    INTERFACE
        ${OpenCV_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}/3rd_party/sophus
)
target_link_libraries(
    lio_visual_ba_dependencies
    INTERFACE
        Eigen3::Eigen
        ${OpenCV_LIBS}
        Ceres::ceres
        nlohmann_json::nlohmann_json
        yaml-cpp
        Threads::Threads
)

if(BUILD_TESTING)
    # A Conda GTest package can inject Conda's libstdc++ into binaries built by
    # the system compiler. Prefer the distribution package when it exists so
    # the compiler and test runtime use the same C++ standard library.
    set(lio_visual_ba_system_gtest_dir "/usr/lib/${CMAKE_LIBRARY_ARCHITECTURE}/cmake/GTest")
    if(UNIX AND EXISTS "${lio_visual_ba_system_gtest_dir}/GTestConfig.cmake")
        find_package(
            GTest REQUIRED CONFIG
            PATHS "${lio_visual_ba_system_gtest_dir}"
            NO_DEFAULT_PATH
        )
    else()
        find_package(GTest REQUIRED)
    endif()
    include(GoogleTest)
endif()

add_library(
    lio_visual_ba_core
    STATIC
        src/ReconstructionExporter.cpp
        src/FeatureProcessor.cpp
        src/Geometry.cpp
        src/DatasetIo.cpp
        src/Mapper.cpp
        src/Optimizer.cpp
        src/Pipeline.cpp
        src/PipelineBackend.cpp
        src/PipelineFrontend.cpp
        src/PipelineRematcher.cpp
)
add_library(lio_visual_ba::core ALIAS lio_visual_ba_core)
target_include_directories(
    lio_visual_ba_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(
    lio_visual_ba_core
    PUBLIC
        lio_visual_ba::dependencies
        lio_visual_ba::project_options
    PRIVATE
        lio_visual_ba::project_warnings
)

add_executable(lio_visual_ba_pipeline src/Main.cpp)
target_link_libraries(
    lio_visual_ba_pipeline
    PRIVATE
        lio_visual_ba::core
        lio_visual_ba::project_warnings
)

if(LIO_VISUAL_BA_ENABLE_QT_VIEWER)
    set(CMAKE_AUTOMOC ON)
    set(CMAKE_AUTOUIC ON)
    set(CMAKE_AUTORCC ON)

    add_executable(
        3dgs_qt_viewer
        src/app/Main.cpp
        src/app/MainWindow.cpp
        src/app/MainWindow.h
        src/app/ViewerController.cpp
        src/app/ViewerController.h
        src/app/viewer_controller/ViewerControllerSemantic.cpp
        src/app/viewer_controller/ViewerControllerNavigation.cpp
        src/app/viewer_controller/ViewerControllerIo.cpp
        src/app/viewer_controller/ViewerControllerPathEditing.cpp
        src/app/viewer_controller/ViewerControllerInput.cpp
        src/app/viewer_controller/ViewerControllerInternals.h
        src/app/RobotPathPlayer.cpp
        src/app/RobotPathPlayer.h
        src/app/SemanticObject.h
        src/renderer/OpenGLWidget.cpp
        src/renderer/OpenGLWidget.h
        src/renderer/OrientationGizmo.cpp
        src/renderer/OrientationGizmo.h
        src/processing/3dgsProcessing.cpp
        src/processing/3dgsProcessing.h
        src/processing/TrajectoryProcessing.cpp
        include/TrajectoryProcessing.h
        src/processing/NavigationIntentParser.cpp
        include/NavigationIntentParser.h
    )
    target_include_directories(
        3dgs_qt_viewer
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src/app
            ${CMAKE_CURRENT_SOURCE_DIR}/src/renderer
            ${CMAKE_CURRENT_SOURCE_DIR}/src/processing
            ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    target_link_libraries(
        3dgs_qt_viewer
        PRIVATE
            lio_visual_ba::core
            lio_visual_ba::project_warnings
            Threads::Threads
            TBB::tbb
            Qt5::Core
            Qt5::Gui
            Qt5::Widgets
            Qt5::OpenGL
            Qt5::Concurrent
    )
    target_compile_definitions(
        3dgs_qt_viewer
        PRIVATE PROJECT_ROOT_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
    )
endif()

if(BUILD_TESTING)
    add_executable(
        lio_visual_ba_unit_tests
        tests/PoseTest.cpp
        tests/CameraModelTest.cpp
        tests/ColmapTextWriterTest.cpp
        tests/ReconstructionValidatorTest.cpp
        tests/EpipolarTest.cpp
        tests/TriangulationTest.cpp
        tests/FeatureExtractorTest.cpp
        tests/FeatureCacheTest.cpp
        tests/FeatureMatcherTest.cpp
        tests/PairCacheTest.cpp
        tests/TrackBuilderTest.cpp
        tests/CalibrationIoTest.cpp
        tests/ImageListIoTest.cpp
        tests/ReconstructionIoTest.cpp
        tests/TrajectoryIoTest.cpp
        tests/PoseInitializerTest.cpp
        tests/BundleAdjusterTest.cpp
        tests/IncrementalOptimizerTest.cpp
        tests/local_BundleAdjusterTest.cpp
        tests/LocalWindowSelectorTest.cpp
        tests/PipelineConfigTest.cpp
        tests/PipelineBackendTest.cpp
        tests/PipelineFrontendTest.cpp
        tests/PipelineTest.cpp
        tests/LandmarkBuilderTest.cpp
        tests/PairSelectorTest.cpp
    )
    target_link_libraries(
        lio_visual_ba_unit_tests
        PRIVATE
            lio_visual_ba::core
            lio_visual_ba::project_warnings
            GTest::gtest_main
    )
    gtest_discover_tests(lio_visual_ba_unit_tests)
    if(Python3_Interpreter_FOUND)
        add_test(
            NAME compare_sparse_outputs_test
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tools/test_compare_sparse_outputs.py
        )
        add_test(
            NAME prepare_kitti_pipeline_test
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_prepare_kitti.py
        )
        set_tests_properties(
            compare_sparse_outputs_test
            PROPERTIES ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/tools"
        )
    endif()
endif()

# Future source files remain explicit in the target lists above. This makes
# every addition visible during review and avoids implicit file(GLOB) behavior.
