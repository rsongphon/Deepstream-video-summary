cmake_minimum_required(VERSION 3.19)
project(deepstream-video-summary VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA version from environment
if(NOT DEFINED CUDA_VER)
    set(CUDA_VER "12.6")
endif()

# Find required packages
find_package(PkgConfig REQUIRED)

# Find OpenCV using pkg-config (optional for now)
pkg_check_modules(OpenCV opencv4)
if(NOT OpenCV_FOUND)
    pkg_check_modules(OpenCV opencv)
endif()
if(NOT OpenCV_FOUND)
    message(STATUS "OpenCV not found, some features will be disabled")
endif()

# Find yaml-cpp using pkg-config
pkg_check_modules(YAML_CPP yaml-cpp)
if(NOT YAML_CPP_FOUND)
    find_package(yaml-cpp QUIET)
    if(NOT yaml-cpp_FOUND)
        message(STATUS "yaml-cpp not found, will try to continue without it")
    endif()
endif()

# Find CUDA
find_package(CUDA REQUIRED)
enable_language(CUDA)

# Find spdlog for logging
find_package(spdlog QUIET)
if(NOT spdlog_FOUND)
    message(STATUS "spdlog not found via find_package, trying pkg-config")
    pkg_check_modules(SPDLOG spdlog)
    if(NOT SPDLOG_FOUND)
        message(STATUS "spdlog not found, will use system headers if available")
    endif()
endif()

# Find GStreamer
pkg_check_modules(GSTREAMER REQUIRED gstreamer-1.0)
pkg_check_modules(GSTREAMER_APP REQUIRED gstreamer-app-1.0)
pkg_check_modules(GSTREAMER_VIDEO REQUIRED gstreamer-video-1.0)

# Find DeepStream
set(DEEPSTREAM_DIR "/opt/nvidia/deepstream/deepstream-7.1")
set(DEEPSTREAM_LIB_DIR "${DEEPSTREAM_DIR}/lib")
set(DEEPSTREAM_INCLUDE_DIR "${DEEPSTREAM_DIR}/sources/includes")

# Check if DeepStream is installed
if(NOT EXISTS ${DEEPSTREAM_DIR})
    message(FATAL_ERROR "DeepStream SDK not found at ${DEEPSTREAM_DIR}")
endif()

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${DEEPSTREAM_INCLUDE_DIR}
    ${GSTREAMER_INCLUDE_DIRS}
)

# Add OpenCV includes if found
if(OpenCV_FOUND)
    include_directories(${OpenCV_INCLUDE_DIRS})
endif()

# Link directories
link_directories(
    ${DEEPSTREAM_LIB_DIR}
    ${GSTREAMER_LIBRARY_DIRS}
    ${CUDA_TOOLKIT_ROOT_DIR}/targets/x86_64-linux/lib
)

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Werror")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -DDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# Add definitions
add_definitions(${GSTREAMER_CFLAGS})

# Phase 1 source files - Foundation components
set(PHASE1_SOURCES
    src/utils/Logger.cpp
    src/utils/ConfigManager.cpp
    src/pipeline/DeepStreamPipeline.cpp
)

# Create Phase 1 library for foundation components
add_library(deepstream_video_summary_foundation ${PHASE1_SOURCES})

# Include Phase 1 headers
target_include_directories(deepstream_video_summary_foundation
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src
    PUBLIC ${DEEPSTREAM_INCLUDE_DIR}
    PUBLIC ${GSTREAMER_INCLUDE_DIRS}
)

# Add OpenCV includes if found
if(OpenCV_FOUND)
    target_include_directories(deepstream_video_summary_foundation PUBLIC ${OpenCV_INCLUDE_DIRS})
endif()

# Link Phase 1 libraries
target_link_libraries(deepstream_video_summary_foundation
    ${GSTREAMER_LIBRARIES}
    pthread
)

# Link yaml-cpp if found
if(YAML_CPP_FOUND)
    target_link_libraries(deepstream_video_summary_foundation ${YAML_CPP_LIBRARIES})
    target_include_directories(deepstream_video_summary_foundation PUBLIC ${YAML_CPP_INCLUDE_DIRS})
elseif(yaml-cpp_FOUND)
    target_link_libraries(deepstream_video_summary_foundation yaml-cpp)
endif()

# Link spdlog if found
if(spdlog_FOUND)
    target_link_libraries(deepstream_video_summary_foundation spdlog::spdlog)
elseif(SPDLOG_FOUND)
    target_link_libraries(deepstream_video_summary_foundation ${SPDLOG_LIBRARIES})
    target_include_directories(deepstream_video_summary_foundation PUBLIC ${SPDLOG_INCLUDE_DIRS})
endif()

# Libraries to link
set(DEEPSTREAM_LIBS
    nvdsgst_meta
    nvds_meta
    nvdsgst_helper
    nvds_infer
    nvds_infer_server
    nvds_msgbroker
    nvds_msgconv
    nvds_nvmultiobjecttracker
    nvbufsurface
    nvbufsurftransform
)

# Link libraries (will be uncommented when executable is created)
# target_link_libraries(${PROJECT_NAME}
#     ${DEEPSTREAM_LIBS}
#     ${GSTREAMER_LIBRARIES}
#     ${OpenCV_LIBS}
#     yaml-cpp
#     pthread
# )

# Add Triton client libraries when available
# find_library(TRITON_CLIENT_LIB tritonclient)
# if(TRITON_CLIENT_LIB)
#     target_link_libraries(${PROJECT_NAME} ${TRITON_CLIENT_LIB})
# endif()

# Install targets
# install(TARGETS ${PROJECT_NAME} DESTINATION bin)
# install(DIRECTORY configs/ DESTINATION share/${PROJECT_NAME}/configs)
# install(DIRECTORY models/ DESTINATION share/${PROJECT_NAME}/models)

# Testing (Google Test)
option(BUILD_TESTS "Build tests" ON)
if(BUILD_TESTS)
    find_package(GTest)
    if(GTest_FOUND)
        enable_testing()
        
        # Environment validation test
        add_executable(test_environment tests/test_environment.cpp)
        target_include_directories(test_environment PRIVATE ${CUDA_INCLUDE_DIRS})
        target_link_libraries(test_environment
            deepstream_video_summary_foundation
            ${DEEPSTREAM_LIBS}
            ${GSTREAMER_LIBRARIES}
            ${CUDA_LIBRARIES}
            cudart
            dl
            GTest::gtest_main
        )
        
        # Add OpenCV libs if found
        if(OpenCV_FOUND)
            target_link_libraries(test_environment ${OpenCV_LIBS})
        endif()
        add_test(NAME EnvironmentTest COMMAND test_environment)
        
        # Basic pipeline test
        add_executable(test_pipeline_basic tests/test_pipeline_basic.cpp)
        target_include_directories(test_pipeline_basic PRIVATE ${CUDA_INCLUDE_DIRS})
        target_link_libraries(test_pipeline_basic
            deepstream_video_summary_foundation
            ${DEEPSTREAM_LIBS}
            ${GSTREAMER_LIBRARIES}
            ${CUDA_LIBRARIES}
            cudart
            dl
            GTest::gtest_main
        )
        add_test(NAME BasicPipelineTest COMMAND test_pipeline_basic)
        
    else()
        message(STATUS "Google Test not found, tests will not be built")
    endif()
endif()

# Print configuration summary
message(STATUS "")
message(STATUS "Configuration Summary:")
message(STATUS "  Project:           ${PROJECT_NAME}")
message(STATUS "  Version:           ${PROJECT_VERSION}")
message(STATUS "  Build Type:        ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ Standard:      ${CMAKE_CXX_STANDARD}")
message(STATUS "  CUDA Version:      ${CUDA_VER}")
message(STATUS "  DeepStream Dir:    ${DEEPSTREAM_DIR}")
message(STATUS "  OpenCV Version:    ${OpenCV_VERSION}")
message(STATUS "  Build Tests:       ${BUILD_TESTS}")
message(STATUS "")
message(STATUS "Note: This is a template CMakeLists.txt")
message(STATUS "      Source files will be added according to PLAN.md")
message(STATUS "")