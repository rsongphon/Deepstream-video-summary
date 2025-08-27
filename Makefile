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
APP := deepstream-multi-inference-app
TARGET_DEVICE = $(shell gcc -dumpmachine | cut -f1 -d -)

# DeepStream SDK Version
NVDS_VERSION := 7.1

# Installation Directories
LIB_INSTALL_DIR ?= /opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib/
APP_INSTALL_DIR ?= /opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/bin/

# Source and Object Files
SRCS := $(wildcard *.c)
INCS := $(wildcard *.h)
OBJS := $(SRCS:.c=.o)

# Package Configuration
PKGS := gstreamer-1.0

# Compiler Flags
CFLAGS += -O3 -DNDEBUG                          # Optimization flags for performance
CFLAGS += -Wall -Wextra -Wno-unused-parameter   # Warning flags
CFLAGS += -std=c99                              # C99 standard
CFLAGS += -I../../includes                      # DeepStream includes
CFLAGS += -I/usr/local/cuda-$(CUDA_VER)/include # CUDA includes
CFLAGS += $(shell pkg-config --cflags $(PKGS))  # GStreamer flags

# Performance and Optimization Flags
CFLAGS += -march=native                         # Optimize for current CPU
CFLAGS += -mtune=native                         # Tune for current CPU
CFLAGS += -ffast-math                           # Fast math optimizations
CFLAGS += -funroll-loops                        # Loop optimizations
CFLAGS += -fomit-frame-pointer                  # Frame pointer optimizations

# Multi-threading Support
CFLAGS += -pthread                              # POSIX threads
CFLAGS += -fopenmp                              # OpenMP support

# DeepStream Specific Flags
CFLAGS += -DWITH_OPENCV                         # Enable OpenCV support
CFLAGS += -DGST_DISABLE_DEPRECATED              # Disable deprecated GStreamer APIs

# Linker Flags and Libraries
LIBS := $(shell pkg-config --libs $(PKGS))     # GStreamer libraries

# CUDA Libraries
LIBS += -L/usr/local/cuda-$(CUDA_VER)/lib64/   # CUDA library path
LIBS += -lcudart                               # CUDA runtime
LIBS += -lcuda                                 # CUDA driver

# DeepStream Libraries
LIBS += -L$(LIB_INSTALL_DIR)                   # DeepStream library path
LIBS += -lnvdsgst_helper                       # DeepStream GStreamer helper
LIBS += -lnvdsgst_meta                         # DeepStream metadata
LIBS += -lnvds_meta                            # DeepStream core metadata
LIBS += -lnvds_yml_parser                      # YAML configuration parser
LIBS += -lnvds_infer                           # Inference library  
LIBS += -lnvds_inferutils                      # Inference utilities

# System Libraries
LIBS += -lm                                    # Math library
LIBS += -ldl                                   # Dynamic loading
LIBS += -lpthread                              # POSIX threads
LIBS += -lgomp                                 # OpenMP runtime

# Runtime Path
LIBS += -Wl,-rpath,$(LIB_INSTALL_DIR)          # Runtime library path

# Build Rules
.PHONY: all install clean debug release help

# Default target
all: release

# Release build (optimized)
release: CFLAGS += -O3 -DNDEBUG
release: $(APP)

# Debug build
debug: CFLAGS += -g -O0 -DDEBUG -fsanitize=address
debug: LIBS += -fsanitize=address
debug: $(APP)

# Object file compilation
%.o: %.c $(INCS) Makefile
	@echo "Compiling $<..."
	$(CC) -c -o $@ $(CFLAGS) $<

# Application linking
$(APP): $(OBJS) Makefile
	@echo "Linking $(APP)..."
	$(CC) -o $(APP) $(OBJS) $(LIBS)
	@echo "Build complete: $(APP)"

# Installation
install: $(APP)
	@echo "Installing $(APP) to $(APP_INSTALL_DIR)..."
	@mkdir -p $(APP_INSTALL_DIR)
	cp -v $(APP) $(APP_INSTALL_DIR)
	@echo "Installation complete."

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	rm -rf $(OBJS) $(APP) *.log tensor_output.csv
	@echo "Clean complete."

# Performance build with profiling
profile: CFLAGS += -O3 -pg -DNDEBUG
profile: LIBS += -pg
profile: $(APP)

# Static analysis
analyze:
	@echo "Running static analysis..."
	@if command -v cppcheck >/dev/null 2>&1; then \
		cppcheck --enable=all --std=c99 --platform=unix64 $(SRCS); \
	else \
		echo "cppcheck not found, skipping static analysis"; \
	fi

# Memory check (requires valgrind)
memcheck: debug
	@echo "Running memory check..."
	@if command -v valgrind >/dev/null 2>&1; then \
		valgrind --tool=memcheck --leak-check=full --show-leak-kinds=all ./$(APP) --help; \
	else \
		echo "valgrind not found, skipping memory check"; \
	fi

# Performance benchmark
benchmark: release
	@echo "Running performance benchmark..."
	@echo "Note: Provide 4 video sources for actual benchmarking"
	@echo "Example: make benchmark SOURCES='vid1.mp4 vid2.mp4 vid3.mp4 vid4.mp4'"
	@if [ -n "$(SOURCES)" ]; then \
		./$(APP) --perf $(SOURCES); \
	else \
		./$(APP) --help; \
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
	@echo "Compiler: $(CC) $(shell $(CC) --version | head -1)"

# Help target
help:
	@echo "DeepStream Multi-Source Batched Inference Application Build System"
	@echo ""
	@echo "Available targets:"
	@echo "  all         - Build application (same as release)"
	@echo "  release     - Build optimized release version (default)"
	@echo "  debug       - Build debug version with sanitizers"
	@echo "  profile     - Build with profiling enabled"
	@echo "  install     - Install application to system directory"
	@echo "  clean       - Remove build artifacts"
	@echo "  analyze     - Run static analysis (requires cppcheck)"
	@echo "  memcheck    - Run memory leak detection (requires valgrind)"
	@echo "  benchmark   - Run performance benchmark"
	@echo "  check-deps  - Check build dependencies"
	@echo "  help        - Show this help message"
	@echo ""
	@echo "Build Configuration:"
	@echo "  CUDA_VER    = $(CUDA_VER)"
	@echo "  NVDS_VERSION= $(NVDS_VERSION)"
	@echo "  CC          = $(CC)"
	@echo ""
	@echo "Usage Examples:"
	@echo "  make                    # Build release version"
	@echo "  make debug              # Build debug version"
	@echo "  make install            # Build and install"
	@echo "  make clean              # Clean build"
	@echo "  make check-deps         # Check dependencies"
	@echo ""
	@echo "Performance Optimization:"
	@echo "  - Release build uses -O3 optimization"
	@echo "  - Native CPU optimizations enabled"
	@echo "  - OpenMP multi-threading support"
	@echo "  - Fast math optimizations"
	@echo ""
	@echo "After building, run the application with:"
	@echo "  ./$(APP) video1.mp4 video2.mp4 video3.mp4 video4.mp4"
	@echo "  ./$(APP) --enable-display video1.mp4 video2.mp4 video3.mp4 video4.mp4"
	@echo "  ./$(APP) --help"

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