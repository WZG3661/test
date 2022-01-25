cmake_minimum_required( VERSION 2.8 )
project ( my_proj CXX C)

set( CMAKE_CXX_COMPILER "g++" )
set( CMAKE_BUILD_TYPE "Debug" )
set( CMAKE_CXX_FLAGS "-std=c++11 -O0 -Wall -Wfatal-errors -ggdb" )
set(CMAKE_EXPORT_COMPILE_COMMANDS on)

# 设置可执行程序输出位置
set( EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin )

####################添加包
# Eigen
find_package(Eigen3 3 REQUIRED )

# OpenCV
find_package( OpenCV 4 REQUIRED )

####################添加头文件
include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${EIGEN3_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

####################添加库文件
# 生成内部库
add_library( my_proj SHARED func.cpp)

# 内部库链接上外部库
target_link_libraries( my_proj
    ${EIGEN3_LIBS}
    ${OpenCV_LIBS}
)

#####################添加可执行程序
add_executable( run_my_proj test/main.cpp )
#可执行程序链接上内部库(已包含外部库)
target_link_libraries( run_my_proj  my_proj )