cmake_minimum_required(VERSION 3.16)

project(3dgs_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(GTest REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(TBB REQUIRED)

file(GLOB_RECURSE PROJECT_SOURCES CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
)

list(REMOVE_ITEM PROJECT_SOURCES
        "${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp"
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

enable_testing()

# Temporarily build only this test file; restore the glob below to
# re-enable all tests under tests/.
# file(GLOB_RECURSE TEST_SOURCES CONFIGURE_DEPENDS
#         "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cpp"
# )
set(TEST_SOURCES
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/EpipolarPriorPoseTest.cpp"
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