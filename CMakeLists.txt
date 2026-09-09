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
target_include_directories(lio_visual_ba_dependencies SYSTEM INTERFACE ${OpenCV_INCLUDE_DIRS})
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
        src/reconstruction_exporter.cpp
        src/feature_processor.cpp
        src/geometry.cpp
        src/dataset_io.cpp
        src/mapper.cpp
        src/optimizer.cpp
        src/pipeline.cpp
        src/pipeline_backend.cpp
        src/pipeline_frontend.cpp
        src/pipeline_rematcher.cpp
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

add_executable(lio_visual_ba_pipeline src/main.cpp)
target_link_libraries(
    lio_visual_ba_pipeline
    PRIVATE
        lio_visual_ba::core
        lio_visual_ba::project_warnings
)

if(BUILD_TESTING)
    add_executable(
        lio_visual_ba_unit_tests
        tests/pose_test.cpp
        tests/camera_model_test.cpp
        tests/colmap_text_writer_test.cpp
        tests/reconstruction_validator_test.cpp
        tests/epipolar_test.cpp
        tests/triangulation_test.cpp
        tests/feature_extractor_test.cpp
        tests/feature_cache_test.cpp
        tests/feature_matcher_test.cpp
        tests/pair_cache_test.cpp
        tests/track_builder_test.cpp
        tests/calibration_io_test.cpp
        tests/image_list_io_test.cpp
        tests/reconstruction_io_test.cpp
        tests/trajectory_io_test.cpp
        tests/pose_initializer_test.cpp
        tests/bundle_adjuster_test.cpp
        tests/incremental_optimizer_test.cpp
        tests/local_bundle_adjuster_test.cpp
        tests/local_window_selector_test.cpp
        tests/pipeline_config_test.cpp
        tests/pipeline_backend_test.cpp
        tests/pipeline_frontend_test.cpp
        tests/pipeline_test.cpp
        tests/landmark_builder_test.cpp
        tests/pair_selector_test.cpp
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
        set_tests_properties(
            compare_sparse_outputs_test
            PROPERTIES ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/tools"
        )
    endif()
endif()

# Future source files remain explicit in the target lists above. This makes
# every addition visible during review and avoids implicit file(GLOB) behavior.
