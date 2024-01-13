cmake_minimum_required(VERSION 3.6)

project(KinectFusion)

set(CMAKE_CXX_STANDARD 14)

include(${PROJECT_SOURCE_DIR}/config.cmake)

find_package(OpenCV 4.0 REQUIRED)
if (OpenCV_INCLUDE_DIRS)
    include_directories("${OpenCV_INCLUDE_DIRS}")
    link_directories(${OpenCV_DIR}/lib)
endif (OpenCV_INCLUDE_DIRS)

find_package(Eigen3 REQUIRED)
if (EIGEN3_INCLUDE_DIR)
    include_directories("${EIGEN3_INCLUDE_DIR}")
endif (EIGEN3_INCLUDE_DIR)

include_directories(${PROJECT_SOURCE_DIR}/include)

file(GLOB KINECTFUSION_HEADER_FILES "${PROJECT_SOURCE_DIR}/include/*.hpp")
file(GLOB KINECTFUSION_SOURCE_FILES "${PROJECT_SOURCE_DIR}/src/*.cpp")

add_executable(KinectFusion ${KINECTFUSION_HEADER_FILES} ${KINECTFUSION_SOURCE_FILES})

target_link_libraries(KinectFusion ${OpenCV_LIBS})