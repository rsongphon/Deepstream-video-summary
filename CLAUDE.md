# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a specialized DeepStream Multi-Source Batched Inference Application that processes exactly 4 video sources simultaneously with hardware-accelerated inference and tensor extraction. The application is built on NVIDIA DeepStream SDK 7.1 and optimized for maximum performance with batched processing.

## Prerequisites and System Requirements

- Ubuntu 22.04 LTS
- NVIDIA GPU with compute capability 6.0+
- NVIDIA Driver 535+
- CUDA 12.6 (required - must export CUDA_VER=12.6)
- NVIDIA DeepStream SDK 7.1
- TensorRT 10.3.0.26
- GStreamer 1.20.3 development packages

## Build System and Common Commands

### Environment Setup (Critical)

```bash
# Always required before building
export CUDA_VER=12.6
```

### Building the Application

```bash
# Check dependencies first
make check-deps

# Build optimized release version (default)
make

# Build debug version with address sanitizer
make debug

# Build with profiling support
make profile

# Clean build artifacts
make clean
```

### Testing and Running

```bash
# Run test script
./test_app.sh

# Basic usage (exactly 4 sources required)
./deepstream-multi-inference-app video1.mp4 video2.mp4 video3.mp4 video4.mp4

# With display output
./deepstream-multi-inference-app --enable-display video1.mp4 video2.mp4 video3.mp4 video4.mp4

# With performance monitoring
./deepstream-multi-inference-app --perf video1.mp4 video2.mp4 video3.mp4 video4.mp4

# Show help
./deepstream-multi-inference-app --help
```

### Development Tools

```bash
# Static analysis
make analyze

# Memory leak checking
make memcheck

# Performance benchmark
make benchmark SOURCES='vid1.mp4 vid2.mp4 vid3.mp4 vid4.mp4'
```

## Application Architecture

### Core Design Principles

1. **Fixed 4-Source Processing**: Application is specifically designed for exactly 4 video sources
2. **Batched Inference**: Uses batch-size=4 for optimal GPU utilization
3. **Hardware Acceleration**: NVDEC decoding + TensorRT inference
4. **Unified Memory**: Zero-copy operations between CPU and GPU
5. **Dual Output**: Tensor extraction + optional display visualization

### Pipeline Architecture

```
[Source 0] ──┐
[Source 1] ──┼─→ [nvstreammux] ─→ [nvinfer] ─→ [tee] ─┬─→ [tensor_probe] → CSV Output
[Source 2] ──┤    (batch=4)        (TensorRT)          │
[Source 3] ──┘                                        └─→ [tiler] → [osd] → [display]
                                                            (optional branch)
```

### Key Components

- **Source Bins**: Hardware-accelerated video decoding for each input
- **nvstreammux**: Batches 4 sources into single inference batch (1920x1080, 40ms timeout)
- **nvinfer**: TensorRT inference engine with INT8 precision
- **Tensor Probe**: Extracts inference metadata and writes to CSV
- **Display Branch**: Optional 2x2 tiled visualization

### File Structure

```
deepstream_multi_inference_app.c    # Main application (1000+ lines C code)
├── main()                          # Entry point and CLI parsing
├── setup_pipeline()               # GStreamer pipeline construction
├── create_source_bin()            # Individual source bin creation
├── tensor_extract_probe()         # Tensor extraction callback
└── bus_call()                     # Pipeline message handling

configs/
├── multi_inference_pgie_config.txt    # TensorRT inference configuration
├── multi_inference_config.yml         # Pipeline configuration
└── labels.txt                         # Classification labels

Makefile                            # Advanced build system with optimization
test_app.sh                         # Test and demonstration script
```

## Configuration System

### Model Configuration (multi_inference_pgie_config.txt)

```ini
# Key settings that must be maintained
batch-size=4                    # Fixed for 4-source processing
network-mode=1                  # INT8 precision
output-tensor-meta=1            # Enable tensor extraction
nvbuf-memory-type=2            # Unified memory
```

### Pipeline Configuration (multi_inference_config.yml)

```yaml
# Critical batch settings
streammux:
  batch-size: 4                 # Must match model batch-size
  batched-push-timeout: 40000   # 40ms batch formation timeout
  nvbuf-memory-type: 2         # Unified memory

primary-gie:
  batch-size: 4                 # Must match streammux
  interval: 0                   # Process every frame
```

## Performance Optimization

### Compiler Optimizations (Makefile)

- `-O3`: Aggressive optimization
- `-march=native -mtune=native`: CPU-specific optimizations  
- `-ffast-math`: Fast floating-point math
- `-funroll-loops`: Loop optimizations
- `-fopenmp`: OpenMP multi-threading

### Runtime Optimizations

1. **Memory Management**: Unified memory (type 2) for zero-copy GPU-CPU transfers
2. **Hardware Acceleration**: NVDEC for decoding, TensorRT for inference
3. **Batching Strategy**: Fixed batch-size=4 for optimal GPU utilization
4. **Asynchronous Processing**: Non-blocking pipeline with queue-based buffering

### Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| Throughput | 30+ FPS per source | For 1080p input |
| Latency | <100ms end-to-end | Including tensor extraction |
| GPU Utilization | >80% | Optimal with batching |
| Memory Usage | <4GB per stream | Unified memory efficiency |

## Output and Results

### Tensor Output (tensor_output.csv)

The application generates CSV output with extracted tensor data:
```csv
Source,Batch,Frame,Layer,LayerName,NumDims,Dimensions
Source_0,1,1,0,output_coverage/Sigmoid,3,1 1 16800
Source_1,1,1,0,output_coverage/Sigmoid,3,1 1 16800
```

### Console Performance Metrics

When using `--perf` flag, displays real-time performance statistics:
```
=== Performance Metrics ===
Total Batches: 150
Average FPS per source: 30.2
Total throughput: 120.8 FPS
```

## Development Guidelines

### Critical Requirements

1. **Exactly 4 Sources**: The application is hardcoded for 4 video sources
2. **Batch Size Consistency**: All batch-size settings must be 4 throughout pipeline
3. **CUDA_VER Environment**: Must export CUDA_VER=12.6 before building
4. **Memory Type**: Use nvbuf-memory-type=2 (unified memory) for performance

### Code Structure

- **C99 Standard**: Application written in C99 with GStreamer
- **Error Handling**: Comprehensive error checking and recovery
- **Performance Focus**: Optimized for maximum throughput and minimal latency
- **Hardware Utilization**: Full GPU acceleration throughout pipeline

### Extension Points

1. **Custom Models**: Modify `configs/multi_inference_pgie_config.txt`
2. **Tensor Processing**: Extend `tensor_extract_probe()` function
3. **Output Formats**: Add custom output handlers
4. **Display Options**: Modify optional display branch

### Common Issues

1. **Model Loading**: Verify model paths in configuration files
2. **Memory Issues**: Check GPU memory with `nvidia-smi`
3. **Pipeline Deadlock**: Test sources individually with `gst-launch-1.0`
4. **Hardware Decoder**: Ensure NVDEC support for input formats

## Important Notes

- The application requires exactly 4 video sources - no more, no less
- All batch-size configurations must be consistent throughout the pipeline
- This is a production-optimized application focused on throughput and efficiency
- The tensor extraction runs in real-time and outputs to CSV for analysis
- Optional display mode provides 2x2 tiled visualization for monitoring
- Performance monitoring is built-in and can be enabled with `--perf` flag