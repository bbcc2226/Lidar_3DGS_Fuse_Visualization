cmake_minimum_required(VERSION 3.16)

project(3dgs_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Let CMake run Qt's code generators automatically for QObject subclasses,
# Designer forms, and resource collections added to any target below.
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)

find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(GTest REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(TBB REQUIRED)
find_package(Qt5 5.12 REQUIRED COMPONENTS Core Gui Widgets OpenGL)

file(GLOB_RECURSE PROJECT_SOURCES CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
)

list(REMOVE_ITEM PROJECT_SOURCES
        "${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp"
)

# The desktop application and renderer have their own executable target and
# must not be folded into the core library by the recursive source glob.
list(FILTER PROJECT_SOURCES EXCLUDE REGEX
        "${CMAKE_CURRENT_SOURCE_DIR}/src/(app|renderer|processing)/.*\\.cpp$"
)

add_library(${PROJECT_NAME}_lib
        ${PROJECT_SOURCES}
)

target_include_directories(${PROJECT_NAME}_lib
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/3rd_party/sophus
        ${OpenCV_INCLUDE_DIRS}
        ${CERES_INCLUDE_DIRS}
)

target_link_libraries(${PROJECT_NAME}_lib
        PUBLIC
        ${OpenCV_LIBS}
        ${CERES_LIBRARIES}
        Eigen3::Eigen
        yaml-cpp
        TBB::tbb
        Qt5::Core
        Qt5::Gui
        Qt5::Widgets
        Qt5::OpenGL
)

target_compile_definitions(${PROJECT_NAME}_lib
        PUBLIC
        PROJECT_ROOT_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
)

add_executable(${PROJECT_NAME}
        src/main.cpp
)

target_link_libraries(${PROJECT_NAME}
        PRIVATE
        ${PROJECT_NAME}_lib
        nlohmann_json::nlohmann_json
)

# Standalone Qt/OpenGL desktop front end. Application layout code lives in
# src/app, while OpenGL-specific implementation lives in src/renderer.
add_executable(${PROJECT_NAME}_qt_viewer
        src/app/main.cpp
        src/app/MainWindow.cpp
        src/app/MainWindow.h
        src/app/ViewerController.cpp
        src/app/ViewerController.h
        src/renderer/OpenGLWidget.cpp
        src/renderer/OpenGLWidget.h
        src/renderer/OrientationGizmo.cpp
        src/renderer/OrientationGizmo.h
        src/processing/3dgsProcessing.cpp
        src/processing/3dgsProcessing.h
)

target_include_directories(${PROJECT_NAME}_qt_viewer
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src/app
        ${CMAKE_CURRENT_SOURCE_DIR}/src/renderer
        ${CMAKE_CURRENT_SOURCE_DIR}/src/processing
)

target_link_libraries(${PROJECT_NAME}_qt_viewer
        PRIVATE
        Qt5::Core
        Qt5::Gui
        Qt5::Widgets
        Qt5::OpenGL
)

enable_testing()

file(GLOB_RECURSE TEST_SOURCES CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cpp"
)

if(TEST_SOURCES)
    add_executable(${PROJECT_NAME}_tests
            ${TEST_SOURCES}
    )

    target_link_libraries(${PROJECT_NAME}_tests
            PRIVATE
            ${PROJECT_NAME}_lib
            GTest::gtest
            GTest::gtest_main
    )

    include(GoogleTest)
    gtest_discover_tests(${PROJECT_NAME}_tests)
endif()
