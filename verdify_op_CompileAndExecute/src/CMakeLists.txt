# Copyright (c) Deepglint Technologies Co., Ltd. 2019. All rights reserved.

# CMake lowest version requirement
cmake_minimum_required(VERSION 3.5.1)

# project information
project(verdify_facealign_op)

# Compile options
add_compile_options(-std=c++11)

# Specify target generation path
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY  "../../../out")
set(CMAKE_CXX_FLAGS_DEBUG "-fPIC -O0 -g -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "-fPIC -O2 -Wall")

find_package(OpenCV REQUIRED)

# Header path
set(INC_PATH $ENV{DDK_PATH})
if (NOT DEFINED ENV{DDK_PATH})
    set(INC_PATH "/usr/local/Ascend")
    message(STATUS "set default INC_PATH: ${INC_PATH}")
else ()
    message(STATUS "env INC_PATH: ${INC_PATH}")
endif ()

include_directories(
    ${INC_PATH}/ascend-toolkit/latest/acllib/include/
    ${INC_PATH}/ascend-toolkit/latest/include/
    ${OpenCV_INCLUDE_DIRS}
)

# add host lib path
set(LIB_PATH $ENV{NPU_HOST_LIB})
# Dynamic libraries in the stub directory can only be used for compilation
if (NOT DEFINED ENV{NPU_HOST_LIB})
    set(LIB_PATH "${INC_PATH}/ascend-toolkit/latest/acllib/lib64/stub/")
    message(STATUS "set default LIB_PATH: ${LIB_PATH}")
else ()
    message(STATUS "env LIB_PATH: ${LIB_PATH}")
endif ()

link_directories(${LIB_PATH})
link_directories(${INC_PATH}/ascend-toolkit/latest/arm64-linux/compiler/lib64/stub/)
add_executable(main main.cpp AclProcess.cpp ModelProcess.cpp)

if (${target} MATCHES "Simulator_Function")
        target_link_libraries(main
            funcsim
            ${OpenCV_LIBS})
else ()
        target_link_libraries(main
            ascendcl acl_op_compiler stdc++
            ${OpenCV_LIBS})
endif ()

install(TARGETS main DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
