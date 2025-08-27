################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
################################################################################

# DeepStream Multi-Source Batched Inference Application Makefile
# Optimized build configuration for maximum performance

# CUDA Version - Required for DeepStream compilation
CUDA_VER = 12.6
ifeq ($(CUDA_VER),)
  $(error "CUDA_VER is not set")
endif

# Application Configuration
APP_C := deepstream-multi-inference-app
APP_CPP := deepstream-multi-source-cpp
TARGET_DEVICE = $(shell gcc -dumpmachine | cut -f1 -d -)

# DeepStream SDK Version
NVDS_VERSION := 7.1

# Installation Directories
LIB_INSTALL_DIR ?= /opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib/
APP_INSTALL_DIR ?= /opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/bin/

# Source and Object Files
# C Application (original)
SRCS_C := $(wildcard *.c)
INCS_C := $(wildcard *.h)
OBJS_C := $(SRCS_C:.c=.o)

# C++ Application (flexible multi-source)
SRCS_CPP := $(wildcard src/cpp/*.cpp)
INCS_CPP := $(wildcard src/cpp/*.h)
OBJS_CPP := $(SRCS_CPP:.cpp=.o)

# Package Configuration
PKGS := gstreamer-1.0

# Compiler Flags - C Application
CFLAGS += -O3 -DNDEBUG                          # Optimization flags for performance
CFLAGS += -Wall -Wextra -Wno-unused-parameter   # Warning flags
CFLAGS += -std=c99                              # C99 standard
CFLAGS += -I../../includes                      # DeepStream includes
CFLAGS += -I/usr/local/cuda-$(CUDA_VER)/include # CUDA includes
CFLAGS += $(shell pkg-config --cflags $(PKGS))  # GStreamer flags

# Compiler Flags - C++ Application
CXXFLAGS += -O3 -DNDEBUG                        # Optimization flags for performance
CXXFLAGS += -Wall -Wextra -Wno-unused-parameter # Warning flags
CXXFLAGS += -std=c++17                          # C++17 standard
CXXFLAGS += -I../../includes                    # DeepStream includes
CXXFLAGS += -Isrc/cpp                           # Local C++ headers
CXXFLAGS += -I/usr/local/cuda-$(CUDA_VER)/include # CUDA includes
CXXFLAGS += $(shell pkg-config --cflags $(PKGS))  # GStreamer flags

# Performance and Optimization Flags (both C and C++)
PERF_FLAGS = -march=native -mtune=native -ffast-math -funroll-loops -fomit-frame-pointer
CFLAGS += $(PERF_FLAGS)
CXXFLAGS += $(PERF_FLAGS)

# Multi-threading Support (both C and C++)
THREAD_FLAGS = -pthread -fopenmp
CFLAGS += $(THREAD_FLAGS)
CXXFLAGS += $(THREAD_FLAGS)

# DeepStream Specific Flags
DEEPSTREAM_FLAGS = -DWITH_OPENCV -DGST_DISABLE_DEPRECATED
CFLAGS += $(DEEPSTREAM_FLAGS)
CXXFLAGS += $(DEEPSTREAM_FLAGS)

# C++ Specific Libraries
YAML_CPP_FLAGS = $(shell pkg-config --cflags yaml-cpp 2>/dev/null || echo "-I/usr/include/yaml-cpp")
CXXFLAGS += $(YAML_CPP_FLAGS)

# Linker Flags and Libraries - Common
LIBS_COMMON := $(shell pkg-config --libs $(PKGS))  # GStreamer libraries

# CUDA Libraries
LIBS_COMMON += -L/usr/local/cuda-$(CUDA_VER)/lib64/ # CUDA library path
LIBS_COMMON += -lcudart                             # CUDA runtime
LIBS_COMMON += -lcuda                               # CUDA driver

# DeepStream Libraries
LIBS_COMMON += -L$(LIB_INSTALL_DIR)                 # DeepStream library path
LIBS_COMMON += -lnvdsgst_helper                     # DeepStream GStreamer helper
LIBS_COMMON += -lnvdsgst_meta                       # DeepStream metadata
LIBS_COMMON += -lnvds_meta                          # DeepStream core metadata
LIBS_COMMON += -lnvds_yml_parser                    # YAML configuration parser
LIBS_COMMON += -lnvds_infer                         # Inference library  
LIBS_COMMON += -lnvds_inferutils                    # Inference utilities

# System Libraries
LIBS_COMMON += -lm                                  # Math library
LIBS_COMMON += -ldl                                 # Dynamic loading
LIBS_COMMON += -lpthread                            # POSIX threads
LIBS_COMMON += -lgomp                               # OpenMP runtime

# Runtime Path
LIBS_COMMON += -Wl,-rpath,$(LIB_INSTALL_DIR)        # Runtime library path

# C Application Libraries
LIBS_C = $(LIBS_COMMON)

# C++ Application Libraries (includes yaml-cpp)
YAML_CPP_LIBS = $(shell pkg-config --libs yaml-cpp 2>/dev/null || echo "-lyaml-cpp")
LIBS_CPP = $(LIBS_COMMON) $(YAML_CPP_LIBS)

# Build Rules
.PHONY: all install clean debug release help cpp c app-c app-cpp

# Default target - build both applications
all: release

# Release build (optimized) - both applications
release: CFLAGS += -O3 -DNDEBUG
release: CXXFLAGS += -O3 -DNDEBUG
release: app-c app-cpp

# Debug build - both applications
debug: CFLAGS += -g -O0 -DDEBUG -fsanitize=address
debug: CXXFLAGS += -g -O0 -DDEBUG -fsanitize=address
debug: LIBS_C += -fsanitize=address
debug: LIBS_CPP += -fsanitize=address
debug: app-c app-cpp

# Build only C application
app-c: $(APP_C)
c: app-c

# Build only C++ application  
app-cpp: $(APP_CPP)
cpp: app-cpp

# C Object file compilation
%.o: %.c $(INCS_C) Makefile
	@echo "Compiling C file $<..."
	$(CC) -c -o $@ $(CFLAGS) $<

# C++ Object file compilation
src/cpp/%.o: src/cpp/%.cpp $(INCS_CPP) Makefile
	@echo "Compiling C++ file $<..."
	$(CXX) -c -o $@ $(CXXFLAGS) $<

# C Application linking
$(APP_C): $(OBJS_C) Makefile
	@echo "Linking C application $(APP_C)..."
	$(CC) -o $(APP_C) $(OBJS_C) $(LIBS_C)
	@echo "Build complete: $(APP_C)"

# C++ Application linking
$(APP_CPP): $(OBJS_CPP) Makefile
	@echo "Linking C++ application $(APP_CPP)..."
	$(CXX) -o $(APP_CPP) $(OBJS_CPP) $(LIBS_CPP)
	@echo "Build complete: $(APP_CPP)"

# Installation
install: $(APP_C) $(APP_CPP)
	@echo "Installing applications to $(APP_INSTALL_DIR)..."
	@mkdir -p $(APP_INSTALL_DIR)
	cp -v $(APP_C) $(APP_CPP) $(APP_INSTALL_DIR)
	@echo "Installation complete."

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	rm -rf $(OBJS_C) $(OBJS_CPP) $(APP_C) $(APP_CPP) *.log tensor_output*.csv output/*.csv
	@echo "Clean complete."

# Performance build with profiling
profile: CFLAGS += -O3 -pg -DNDEBUG
profile: CXXFLAGS += -O3 -pg -DNDEBUG
profile: LIBS_C += -pg
profile: LIBS_CPP += -pg
profile: app-c app-cpp

# Static analysis
analyze:
	@echo "Running static analysis..."
	@if command -v cppcheck >/dev/null 2>&1; then \
		echo "Analyzing C files..."; \
		cppcheck --enable=all --std=c99 --platform=unix64 $(SRCS_C); \
		echo "Analyzing C++ files..."; \
		cppcheck --enable=all --std=c++17 --platform=unix64 $(SRCS_CPP); \
	else \
		echo "cppcheck not found, skipping static analysis"; \
	fi

# Memory check (requires valgrind)
memcheck: debug
	@echo "Running memory check..."
	@if command -v valgrind >/dev/null 2>&1; then \
		echo "Testing C application..."; \
		valgrind --tool=memcheck --leak-check=full --show-leak-kinds=all ./$(APP_C) --help; \
		echo "Testing C++ application..."; \
		valgrind --tool=memcheck --leak-check=full --show-leak-kinds=all ./$(APP_CPP) --help; \
	else \
		echo "valgrind not found, skipping memory check"; \
	fi

# Performance benchmark
benchmark: release
	@echo "Running performance benchmark..."
	@echo "Testing both C and C++ applications..."
	@if [ -n "$(SOURCES)" ]; then \
		echo "C Application Benchmark:"; \
		./$(APP_C) --perf $(SOURCES); \
		echo "C++ Application Benchmark:"; \
		./$(APP_CPP) --perf $(SOURCES); \
	else \
		echo "No sources provided. Usage examples:"; \
		echo "  make benchmark SOURCES='vid1.mp4 vid2.mp4'"; \
		echo ""; \
		echo "C Application help:"; \
		./$(APP_C) --help; \
		echo ""; \
		echo "C++ Application help:"; \
		./$(APP_CPP) --help; \
	fi

# Check dependencies
check-deps:
	@echo "Checking dependencies..."
	@echo "CUDA Version: $(CUDA_VER)"
	@echo "DeepStream Version: $(NVDS_VERSION)"
	@echo "Target Device: $(TARGET_DEVICE)"
	@pkg-config --exists $(PKGS) && echo "GStreamer: OK" || echo "GStreamer: MISSING"
	@test -d /usr/local/cuda-$(CUDA_VER) && echo "CUDA: OK" || echo "CUDA: MISSING"
	@test -d $(LIB_INSTALL_DIR) && echo "DeepStream libs: OK" || echo "DeepStream libs: MISSING"
	@pkg-config --exists yaml-cpp && echo "yaml-cpp: OK" || echo "yaml-cpp: MISSING (install libyaml-cpp-dev)"
	@echo "C Compiler: $(CC) $(shell $(CC) --version 2>/dev/null | head -1 || echo 'NOT FOUND')"
	@echo "C++ Compiler: $(CXX) $(shell $(CXX) --version 2>/dev/null | head -1 || echo 'NOT FOUND')"
	@echo ""
	@echo "Build Summary:"
	@echo "  C Application: $(APP_C)"
	@echo "  C++ Application: $(APP_CPP)"

# Help target
help:
	@echo "DeepStream Multi-Source Flexible Inference Application Build System"
	@echo ""
	@echo "Available targets:"
	@echo "  all         - Build both applications (same as release)"
	@echo "  release     - Build optimized release versions (default)"
	@echo "  debug       - Build debug versions with sanitizers"
	@echo "  app-c (c)   - Build only C application (hardcoded 4 sources)"
	@echo "  app-cpp (cpp) - Build only C++ application (flexible sources)"
	@echo "  profile     - Build with profiling enabled"
	@echo "  install     - Install both applications to system directory"
	@echo "  clean       - Remove build artifacts"
	@echo "  analyze     - Run static analysis (requires cppcheck)"
	@echo "  memcheck    - Run memory leak detection (requires valgrind)"
	@echo "  benchmark   - Run performance benchmark on both apps"
	@echo "  check-deps  - Check build dependencies"
	@echo "  help        - Show this help message"
	@echo ""
	@echo "Build Configuration:"
	@echo "  CUDA_VER    = $(CUDA_VER)"
	@echo "  NVDS_VERSION= $(NVDS_VERSION)"
	@echo "  CC          = $(CC)"
	@echo ""
	@echo "Usage Examples:"
	@echo "  make                    # Build both applications (release)"
	@echo "  make cpp                # Build only C++ flexible application"
	@echo "  make c                  # Build only C hardcoded application"
	@echo "  make debug              # Build debug versions"
	@echo "  make install            # Build and install both"
	@echo "  make clean              # Clean build artifacts"
	@echo "  make check-deps         # Check dependencies"
	@echo ""
	@echo "Performance Optimization:"
	@echo "  - Release build uses -O3 optimization"
	@echo "  - Native CPU optimizations enabled"
	@echo "  - OpenMP multi-threading support"
	@echo "  - Fast math optimizations"
	@echo ""
	@echo "After building, run the applications with:"
	@echo ""
	@echo "C++ Application (flexible sources):"
	@echo "  ./$(APP_CPP) video1.mp4 video2.mp4            # 2 sources"
	@echo "  ./$(APP_CPP) -d vid1.mp4 vid2.mp4 vid3.mp4    # 3 sources with display"
	@echo "  ./$(APP_CPP) -p rtsp://cam1 rtsp://cam2       # Live streams with perf"
	@echo "  ./$(APP_CPP) --help                           # Full help"
	@echo ""
	@echo "C Application (exactly 4 sources):"
	@echo "  ./$(APP_C) video1.mp4 video2.mp4 video3.mp4 video4.mp4"
	@echo "  ./$(APP_C) --enable-display video1.mp4 video2.mp4 video3.mp4 video4.mp4"
	@echo "  ./$(APP_C) --help"

################################################################################
# Build Optimization Notes:
#
# 1. Compiler Optimizations:
#    - -O3: Aggressive optimization for maximum performance
#    - -march=native: Optimize for current CPU architecture
#    - -ffast-math: Enable fast floating-point math
#    - -funroll-loops: Unroll loops for better performance
#
# 2. Memory Optimizations:
#    - Unified memory support for GPU-CPU data sharing
#    - Efficient buffer management
#    - Zero-copy operations where possible
#
# 3. Multi-threading:
#    - OpenMP support for parallel processing
#    - POSIX threads for GStreamer pipeline management
#
# 4. Debug Support:
#    - Address sanitizer for memory error detection
#    - Debug symbols and verbose error reporting
#    - Static analysis integration
#
# 5. Profiling:
#    - gprof support for performance profiling
#    - Valgrind integration for memory analysis
#    - Built-in performance measurement
#
# Environment Variables:
#   CUDA_VER        - CUDA version (required)
#   LIB_INSTALL_DIR - DeepStream library directory
#   APP_INSTALL_DIR - Application installation directory
#   CC              - C compiler (default: gcc)
#
# Prerequisites:
#   - NVIDIA DeepStream SDK 7.1
#   - CUDA Toolkit 12.6
#   - GStreamer 1.0 development packages
#   - Standard build tools (gcc, make)
################################################################################