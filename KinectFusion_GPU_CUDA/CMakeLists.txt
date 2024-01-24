cmake_minimum_required(VERSION 3.6)

project(KinectFusion_CUDA)

set(CMAKE_CXX_STANDARD 14)

include(${PROJECT_SOURCE_DIR}/config.cmake)

find_package(CUDA 11.0 REQUIRED)
if (CUDA_FOUND)
    include_directories("${CUDA_INCLUDE_DIRS}")
    # message(STATUS "==> KinectFusion_CUDA Found CUDA " ${CUDA_INCLUDE_DIRS})
endif (CUDA_FOUND)

find_package(OpenCV 4.0 REQUIRED)
if (OpenCV_INCLUDE_DIRS)
    include_directories("${OpenCV_INCLUDE_DIRS}")
    link_directories(${OpenCV_DIR}/lib)
endif (OpenCV_INCLUDE_DIRS)

find_package(Eigen3 REQUIRED)
if (EIGEN3_INCLUDE_DIR)
    include_directories("${EIGEN3_INCLUDE_DIR}")
    # message(STATUS "==> KinectFusion_CUDA Found EIGEN3 " ${EIGEN3_INCLUDE_DIR})
endif (EIGEN3_INCLUDE_DIR)

SET(CUDA_PROPAGATE_HOST_FLAGS OFF)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Weffc++")

include_directories(${PROJECT_SOURCE_DIR}/include)

file(GLOB KINECTFUSION_HEADER_FILES "${PROJECT_SOURCE_DIR}/include/*.hpp")
file(GLOB KINECTFUSION_SOURCE_FILES "${PROJECT_SOURCE_DIR}/src/*.cpp")
file(GLOB KINECTFUSION_CUDA_FILES "${PROJECT_SOURCE_DIR}/cuda/*.cu")

CUDA_ADD_EXECUTABLE(KinectFusion_CUDA ${KINECTFUSION_HEADER_FILES} ${KINECTFUSION_SOURCE_FILES} ${KINECTFUSION_CUDA_FILES})

target_link_libraries(KinectFusion_CUDA ${OpenCV_LIBS} ${CUDA_LIBRARIES} ${CUDA_CUFFT_LIBRARIES})