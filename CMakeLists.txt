cmake_minimum_required(VERSION 3.18)
project(DeepStreamMultiSourceInference VERSION 1.0.0 LANGUAGES CXX C)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set default build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags for optimization and performance
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -mtune=native -ffast-math -funroll-loops")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall -Wextra -fsanitize=address")

# Enable OpenMP for parallel processing
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(target OpenMP::OpenMP_CXX)
endif()

# CUDA version check and setup
if(NOT DEFINED CUDA_VER)
    if(DEFINED ENV{CUDA_VER})
        set(CUDA_VER $ENV{CUDA_VER})
    else()
        set(CUDA_VER "12.6")
        message(WARNING "CUDA_VER not set, defaulting to ${CUDA_VER}")
    endif()
endif()

message(STATUS "Using CUDA version: ${CUDA_VER}")

# NVIDIA DeepStream paths
set(DEEPSTREAM_ROOT "/opt/nvidia/deepstream/deepstream-7.1")
set(DEEPSTREAM_SOURCES "${DEEPSTREAM_ROOT}/sources")
set(DEEPSTREAM_INCLUDES "${DEEPSTREAM_SOURCES}/includes")
set(DEEPSTREAM_LIBS "${DEEPSTREAM_ROOT}/lib")

# CUDA paths
set(CUDA_ROOT "/usr/local/cuda-${CUDA_VER}")
set(CUDA_INCLUDE_DIRS "${CUDA_ROOT}/include")
set(CUDA_LIBRARY_DIRS "${CUDA_ROOT}/lib64")

# Find required packages
find_package(PkgConfig REQUIRED)

# GStreamer packages
pkg_check_modules(GSTREAMER REQUIRED gstreamer-1.0)
pkg_check_modules(GSTREAMER_BASE REQUIRED gstreamer-base-1.0)
pkg_check_modules(GSTREAMER_VIDEO REQUIRED gstreamer-video-1.0)
pkg_check_modules(GLIB REQUIRED glib-2.0)
pkg_check_modules(GOBJECT REQUIRED gobject-2.0)

# Check for yaml-cpp
find_path(YAML_CPP_INCLUDE_DIR yaml-cpp/yaml.h)
find_library(YAML_CPP_LIBRARY yaml-cpp)

if(YAML_CPP_INCLUDE_DIR AND YAML_CPP_LIBRARY)
    message(STATUS "Found yaml-cpp: ${YAML_CPP_LIBRARY}")
else()
    message(FATAL_ERROR "yaml-cpp not found. Install with: sudo apt-get install libyaml-cpp-dev")
endif()

# Check for OpenCV (optional, for enhanced processing)
find_package(OpenCV QUIET)
if(OpenCV_FOUND)
    message(STATUS "Found OpenCV: ${OpenCV_VERSION}")
    add_definitions(-DHAVE_OPENCV)
else()
    message(STATUS "OpenCV not found - some features will be disabled")
endif()

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cpp
    ${DEEPSTREAM_INCLUDES}
    ${CUDA_INCLUDE_DIRS}
    ${GSTREAMER_INCLUDE_DIRS}
    ${GLIB_INCLUDE_DIRS}
    ${YAML_CPP_INCLUDE_DIR}
)

if(OpenCV_FOUND)
    include_directories(${OpenCV_INCLUDE_DIRS})
endif()

# Library directories
link_directories(
    ${DEEPSTREAM_LIBS}
    ${CUDA_LIBRARY_DIRS}
    ${GSTREAMER_LIBRARY_DIRS}
)

# Source files for the flexible C++ application
set(CPP_SOURCES
    src/cpp/main.cpp
    src/cpp/pipeline_builder.cpp
    src/cpp/tensor_processor.cpp
)

set(CPP_HEADERS
    src/cpp/pipeline_builder.h
    src/cpp/tensor_processor.h
)

# Create the C++ executable
add_executable(deepstream-multi-source-cpp ${CPP_SOURCES})

# Set target properties
set_target_properties(deepstream-multi-source-cpp PROPERTIES
    OUTPUT_NAME "deepstream-multi-source-cpp"
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

# Link libraries for C++ application
target_link_libraries(deepstream-multi-source-cpp
    # DeepStream libraries
    nvdsgst_meta
    nvds_meta
    nvdsgst_helper
    nvdsgst_smartrecord
    nvds_utils
    nvds_msgbroker
    
    # CUDA libraries  
    cuda
    cudart
    
    # GStreamer libraries
    ${GSTREAMER_LIBRARIES}
    ${GLIB_LIBRARIES}
    ${GOBJECT_LIBRARIES}
    
    # System libraries
    pthread
    dl
    m
    
    # yaml-cpp
    ${YAML_CPP_LIBRARY}
)

# Link OpenCV if available
if(OpenCV_FOUND)
    target_link_libraries(deepstream-multi-source-cpp ${OpenCV_LIBS})
endif()

# Link OpenMP if available
if(OpenMP_CXX_FOUND)
    target_link_libraries(deepstream-multi-source-cpp OpenMP::OpenMP_CXX)
endif()

# Compiler definitions
target_compile_definitions(deepstream-multi-source-cpp PRIVATE
    PLATFORM_TEGRA=0
    NVDS_VERSION_MAJOR=7
    NVDS_VERSION_MINOR=1
)

# Also build the original C application for comparison/fallback
add_executable(deepstream-multi-inference-c deepstream_multi_inference_app.c)

set_target_properties(deepstream-multi-inference-c PROPERTIES
    OUTPUT_NAME "deepstream-multi-inference-c"
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

target_link_libraries(deepstream-multi-inference-c
    nvdsgst_meta
    nvds_meta  
    nvdsgst_helper
    nvds_utils
    cuda
    cudart
    ${GSTREAMER_LIBRARIES}
    ${GLIB_LIBRARIES}
    ${GOBJECT_LIBRARIES}
    pthread
    dl
    m
)

# Install targets
install(TARGETS deepstream-multi-source-cpp deepstream-multi-inference-c
    RUNTIME DESTINATION bin
)

# Install configuration files
install(DIRECTORY config/
    DESTINATION share/deepstream-multi-source/config
    FILES_MATCHING PATTERN "*.txt" PATTERN "*.yaml" PATTERN "*.yml"
)

# Install scripts
install(FILES test_app.sh
    DESTINATION share/deepstream-multi-source/scripts
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ GROUP_EXECUTE GROUP_READ
)

# Custom targets for development

# Check dependencies
add_custom_target(check-deps
    COMMAND ${CMAKE_COMMAND} -E echo "Checking DeepStream installation..."
    COMMAND test -d ${DEEPSTREAM_ROOT} || (echo "ERROR: DeepStream not found at ${DEEPSTREAM_ROOT}" && exit 1)
    COMMAND test -d ${CUDA_ROOT} || (echo "ERROR: CUDA not found at ${CUDA_ROOT}" && exit 1)
    COMMAND pkg-config --exists gstreamer-1.0 || (echo "ERROR: GStreamer development packages not found" && exit 1)
    COMMAND ${CMAKE_COMMAND} -E echo "All dependencies found!"
)

# Clean output files  
add_custom_target(clean-output
    COMMAND rm -rf output/*.csv output/*.json output/*.bin
    COMMAND ${CMAKE_COMMAND} -E echo "Cleaned output files"
)

# Performance benchmark
add_custom_target(benchmark
    COMMAND echo "Running performance benchmark..."
    COMMAND ./deepstream-multi-source-cpp --perf --max-tensor-values 10 ${BENCHMARK_SOURCES}
    DEPENDS deepstream-multi-source-cpp
)

# Memory check with valgrind
add_custom_target(memcheck
    COMMAND valgrind --tool=memcheck --leak-check=full --track-origins=yes 
            ./deepstream-multi-source-cpp --max-tensor-values 10 ${TEST_SOURCES}
    DEPENDS deepstream-multi-source-cpp
)

# Static analysis with cppcheck
add_custom_target(analyze
    COMMAND cppcheck --enable=all --inconclusive --std=c++17 
            --suppress=missingIncludeSystem --suppress=unmatchedSuppression
            -I ${CMAKE_CURRENT_SOURCE_DIR}/src/cpp
            -I ${DEEPSTREAM_INCLUDES}
            ${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/
)

# Generate documentation with doxygen
find_package(Doxygen QUIET)
if(DOXYGEN_FOUND)
    add_custom_target(docs
        COMMAND doxygen
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Generating documentation with Doxygen"
    )
endif()

# Print build information
message(STATUS "=====================================")
message(STATUS "Build Configuration:")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  CUDA Version: ${CUDA_VER}")
message(STATUS "  DeepStream Root: ${DEEPSTREAM_ROOT}")
message(STATUS "  OpenCV Found: ${OpenCV_FOUND}")
message(STATUS "  OpenMP Found: ${OpenMP_CXX_FOUND}")
message(STATUS "=====================================")

# Build configuration summary
add_custom_target(build-info
    COMMAND ${CMAKE_COMMAND} -E echo ""
    COMMAND ${CMAKE_COMMAND} -E echo "=== Build Information ==="
    COMMAND ${CMAKE_COMMAND} -E echo "C++ Application: deepstream-multi-source-cpp"
    COMMAND ${CMAKE_COMMAND} -E echo "C Application: deepstream-multi-inference-c"
    COMMAND ${CMAKE_COMMAND} -E echo "Build Type: ${CMAKE_BUILD_TYPE}"
    COMMAND ${CMAKE_COMMAND} -E echo "Install Prefix: ${CMAKE_INSTALL_PREFIX}"
    COMMAND ${CMAKE_COMMAND} -E echo "========================="
    COMMAND ${CMAKE_COMMAND} -E echo ""
)