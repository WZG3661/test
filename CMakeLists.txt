cmake_minimum_required (VERSION 2.8)
set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -Wall -Wfatal-errors -ggdb")
set(CMAKE_EXPORT_COMPILE_COMMANDS on)

project (test)

#=========================外部库
find_package(Eigen3 3 REQUIRED )
find_package(OpenCV 4 REQUIRED )
find_package(Ceres REQUIRED)

#=========================库
include_directories(lib_src)
aux_source_directory (lib_src/ LIB_SRC_LIST)
add_library (test_shared SHARED ${LIB_SRC_LIST})# 动态库
add_library (test_static STATIC ${LIB_SRC_LIST})# 静态库
set (LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)# 输出路径

#========================项目
include_directories(include)
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CERES_INCLUDE_DIRS})
aux_source_directory (src/ SRC_LIST)

#========================可执行程序
add_executable(main main.cpp ${SRC_LIST})
target_link_libraries(main test_static)
target_link_libraries(main ${EIGEN3_LIBS} ${OpenCV_LIBS} ${CERES_LIBRARIES})