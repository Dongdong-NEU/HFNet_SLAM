cmake_minimum_required(VERSION 3.10)
project(test_match_global_feats_standalone)

# 设置C++标准
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置编译类型
set(DEFAULT_BUILD_TYPE "Debug")
if (NOT CMAKE_BUILD_TYPE)
    message(STATUS "Setting build type to '${DEFAULT_BUILD_TYPE}' as none was specified.")
    set(CMAKE_BUILD_TYPE "${DEFAULT_BUILD_TYPE}" CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif ()

# 编译标志
set(CMAKE_CXX_FLAGS "-std=c++14 -Wall -fopenmp -march=native -pthread")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -g")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -O3 -DNDEBUG")

# 查找依赖包
find_package(OpenCV REQUIRED)
find_package(Eigen3 3.1.0 REQUIRED)
find_package(Pangolin REQUIRED)

# TensorRT支持（如不需要可注释掉）
set(USE_TENSORRT ON)
if(${USE_TENSORRT})
    add_definitions(-DUSE_TENSORRT)
    find_package(CUDA REQUIRED)
    set(TENSORRT_INCLUDE_DIR "/home/xihuidong/usr_lib/TensorRT-8.5.3.1/include")
    include_directories(${TENSORRT_INCLUDE_DIR} ${CUDA_INCLUDE_DIRS})
endif()

# 包含目录
include_directories(
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/include/Extractors
    ${EIGEN3_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
    ${Pangolin_INCLUDE_DIRS}
)

# 自动收集src目录下所有源文件
file(GLOB_RECURSE LIBRARY_SOURCES
    "${PROJECT_SOURCE_DIR}/src/*.cpp"
    "${PROJECT_SOURCE_DIR}/src/*.cc"
)

# app目录下的主程序文件
set(APP_SOURCES
    ${PROJECT_SOURCE_DIR}/app/test_match_global_feats.cc
)

# 创建可执行文件
add_executable(test_match_global_feats ${APP_SOURCES} ${LIBRARY_SOURCES})

# 链接库
target_link_libraries(test_match_global_feats
    ${OpenCV_LIBS}
    ${EIGEN3_LIBS}
    ${Pangolin_LIBRARIES}
)

# TensorRT链接（如需要）
if(${USE_TENSORRT})
    target_link_libraries(test_match_global_feats
        /home/xihuidong/usr_lib/TensorRT-8.5.3.1/lib/libnvinfer.so.8
        /home/xihuidong/usr_lib/TensorRT-8.5.3.1/lib/libnvonnxparser.so
        ${CUDA_LIBRARIES}
    )
endif()

# 安装规则
install(TARGETS test_match_global_feats DESTINATION bin)

# 打印配置信息
message(STATUS "===============================================")
message(STATUS "Project: ${PROJECT_NAME}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "TensorRT: ${USE_TENSORRT}")
message(STATUS "===============================================")