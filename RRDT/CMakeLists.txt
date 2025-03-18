
cmake_minimum_required(VERSION 3.1...3.20)
project(MyProject)

# 设置C++标准
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


find_package(Threads REQUIRED)

# 查找Assimp库
find_package(assimp REQUIRED)
include_directories(${ASSIMP_INCLUDE_DIRS})
# 查找并包含Pangolin库
find_package(Pangolin REQUIRED NO_MODULE)
include_directories(${Pangolin_INCLUDE_DIRS})

find_package(Eigen3 3.1.91 QUIET) #(requires 3.1.91 or greater)

# 查找 Boost 库
find_package(Boost REQUIRED COMPONENTS system)

# 如果 Boost 未找到，报错
if(NOT Boost_FOUND)
    message(FATAL_ERROR "Boost library not found!")
endif()

find_package(OpenCV 4.0 REQUIRED)

include_directories(${Boost_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
message(STATUS "OpenCV is set to ${OpenCV_INCLUDE_DIRS}")

include_directories("${CMAKE_SOURCE_DIR}/include")
include_directories(
${PROJECT_SOURCE_DIR}
${PROJECT_SOURCE_DIR}/lib
${PROJECT_SOURCE_DIR}/lib/rrf
${EIGEN3_INCLUDE_DIR}
)
# 用于测试的！！！！添加可执行文件
# 用于测试的！！！！添加可执行文件
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples)


add_executable(main2D
test/main2D.cpp
lib/rrf/rrf_common.cpp
)

target_link_libraries(main2D PUBLIC 
${OpenCV_LIBS}
Threads::Threads
)

add_executable(main3D
test/main3D.cpp
lib/rrf/rrf_common.cpp
lib/rrf/debug3D.cpp
)

target_link_libraries(main3D PUBLIC 
Boost::system
${OpenCV_LIBS}
Threads::Threads
${Pangolin_LIBRARIES}
${ASSIMP_LIBRARIES}
)

add_executable(mainSE2
test/mainSE2.cpp
lib/rrf/rrf_common.cpp
)

target_link_libraries(mainSE2 PUBLIC 
${OpenCV_LIBS}
Threads::Threads
)

