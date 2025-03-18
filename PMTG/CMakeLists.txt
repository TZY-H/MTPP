
cmake_minimum_required(VERSION 3.1...3.20)
project(MyProject)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 找到CGAL
# 添加局部CGAL库的路径
set(CGAL_DIR "${CMAKE_SOURCE_DIR}/lib/CGAL/lib/cmake/CGAL")
find_package(CGAL REQUIRED )
message(STATUS "CGAL_UDIR is set to ${CGAL_DIR}")
message(STATUS "CGAL is set to ${CGAL_INCLUDE_DIRS}")

find_package(Threads REQUIRED)

find_package(TBB QUIET)
include(CGAL_TBB_support)

find_package(OpenCV 4.0 REQUIRED)
message(STATUS "OpenCV is set to ${OpenCV_INCLUDE_DIRS}")

include_directories("${CMAKE_SOURCE_DIR}/include")
include_directories(
${PROJECT_SOURCE_DIR}
${CGAL_DIR}/lib/CGAL/include/CGAL
${PROJECT_SOURCE_DIR}/lib
${PROJECT_SOURCE_DIR}/lib/PMTG
${PROJECT_SOURCE_DIR}/lib/Gen3
${EIGEN3_INCLUDE_DIR}
)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples)

# 查找Assimp库
find_package(assimp REQUIRED)
include_directories(${ASSIMP_INCLUDE_DIRS})
# 查找并包含Pangolin库
find_package(Pangolin REQUIRED NO_MODULE)
include_directories(${Pangolin_INCLUDE_DIRS})
# 查找Eigen库
find_package(Eigen3 REQUIRED NO_MODULE)
include_directories(${EIGEN3_INCLUDE_DIR})

 
add_executable(ShowArmPath
test/ShowArmPath.cpp
lib/Gen3/uGen3Ctrl.cpp
lib/PMTG/debug3D.cpp
)

target_link_libraries(ShowArmPath PUBLIC 
CGAL::CGAL 
CGAL::TBB_support
${OpenCV_LIBS}
pthread
${Pangolin_LIBRARIES}
${ASSIMP_LIBRARIES}
)


add_executable(main
test/main.cpp
lib/Gen3/uGen3Ctrl.cpp
lib/PMTG/debug3D.cpp
)

target_link_libraries(main PUBLIC 
CGAL::CGAL 
CGAL::TBB_support
${OpenCV_LIBS}
pthread
${Pangolin_LIBRARIES}
${ASSIMP_LIBRARIES}
)
