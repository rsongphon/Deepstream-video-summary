# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This repository contains two specialized DeepStream Multi-Source Batched Inference Applications:

1. **C Application**: Hardcoded 4-source processing with maximum optimization
2. **C++ Application**: Flexible multi-source processing (1-64+ sources) with advanced features

Both applications provide hardware-accelerated inference, tensor extraction, and are built on NVIDIA DeepStream SDK 7.1 with performance optimizations.

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

### Building the Applications

```bash
# Check dependencies first
make check-deps

# Build both applications (release optimized)
make

# Build only C application (hardcoded 4 sources)
make c

# Build only C++ application (flexible sources)
make cpp

# Build debug versions with address sanitizer
make debug

# Build with profiling support
make profile

# Clean build artifacts
make clean
```

### Testing and Running

```bash
# Run test script (tests both applications)
./test_app.sh

# C Application (exactly 4 sources required)
./deepstream-multi-inference-app video1.mp4 video2.mp4 video3.mp4 video4.mp4
./deepstream-multi-inference-app --enable-display video1.mp4 video2.mp4 video3.mp4 video4.mp4
./deepstream-multi-inference-app --perf video1.mp4 video2.mp4 video3.mp4 video4.mp4
./deepstream-multi-inference-app --help

# C++ Application (flexible 1-64+ sources)
./deepstream-multi-source-cpp video1.mp4 video2.mp4
./deepstream-multi-source-cpp -d video1.mp4 video2.mp4 video3.mp4
./deepstream-multi-source-cpp -p rtsp://cam1 rtsp://cam2
./deepstream-multi-source-cpp -c config/pipeline_config.yaml video1.mp4 video2.mp4
./deepstream-multi-source-cpp --help
```

### Development Tools

```bash
# Static analysis (both applications)
make analyze

# Memory leak checking (both applications)
make memcheck

# Performance benchmark (both applications)
make benchmark SOURCES='vid1.mp4 vid2.mp4 vid3.mp4 vid4.mp4'

# Test flexible C++ application
./test_flexible_app.sh

# Test tensor extraction specifically
./test_tensor_extraction.sh
```

## Application Architecture

### Core Design Principles

**C Application (Hardcoded 4-Source):**
1. **Fixed 4-Source Processing**: Specifically designed for exactly 4 video sources
2. **Batched Inference**: Uses batch-size=4 for optimal GPU utilization
3. **Hardware Acceleration**: NVDEC decoding + TensorRT inference
4. **Unified Memory**: Zero-copy operations between CPU and GPU
5. **Dual Output**: Tensor extraction + optional display visualization

**C++ Application (Flexible Multi-Source):**
1. **Flexible Source Count**: Handles 1-64+ video sources dynamically
2. **Auto Batch Sizing**: Automatically adjusts batch size to match source count
3. **Advanced Configuration**: YAML configuration with CLI overrides
4. **Enhanced Tensor Extraction**: Multiple output formats (CSV, JSON, binary)
5. **Comprehensive Logging**: Detailed debugging and performance monitoring

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
# C Application (Hardcoded 4-Source)
deepstream_multi_inference_app.c    # Main application (1000+ lines C code)
├── main()                          # Entry point and CLI parsing
├── setup_pipeline()               # GStreamer pipeline construction
├── create_source_bin()            # Individual source bin creation
├── tensor_extract_probe()         # Tensor extraction callback
└── bus_call()                     # Pipeline message handling

# C++ Application (Flexible Multi-Source)
src/cpp/
├── main.cpp                      # Main application entry point
├── pipeline_builder.h/cpp        # Flexible pipeline construction
├── tensor_processor.h/cpp        # Enhanced tensor processing
└── ...

configs/
├── multi_inference_pgie_config.txt    # TensorRT inference configuration
├── multi_inference_config.yml         # Pipeline configuration
├── pipeline_config.yaml              # Flexible pipeline config
└── labels.txt                         # Classification labels

Makefile                            # Advanced build system with optimization
test_app.sh                         # Test and demonstration script
test_flexible_app.sh                # Flexible application test
test_tensor_extraction.sh           # Tensor extraction test
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

### Pipeline Configuration

**C Application (multi_inference_config.yml):**
```yaml
# Fixed batch settings for 4 sources
streammux:
  batch-size: 4                 # Must match model batch-size
  batched-push-timeout: 40000   # 40ms batch formation timeout
  nvbuf-memory-type: 2         # Unified memory

primary-gie:
  batch-size: 4                 # Must match streammux
  interval: 0                   # Process every frame
```

**C++ Application (pipeline_config.yaml):**
```yaml
# Flexible batch settings
device:
  gpu_id: 0
  memory_type: 2               # Unified memory

streammux:
  batch_size: auto             # Auto-adjusts to source count
  timeout: 40000               # 40ms batch timeout
  width: 1920
  height: 1080

inference:
  config_file: configs/multi_inference_pgie_config.txt
  interval: 0

tensor_export:
  format: csv                  # csv, json, binary
  max_values: 100              # Limit tensor values per export
  output_dir: output
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
| Tensor Extraction | <100μs per tensor | Real-time processing |

## Output and Results

### Tensor Output

**C Application (tensor_output.csv):**
```csv
Source,Batch,Frame,Layer,LayerName,NumDims,Dimensions,RawTensorData
Source_0,Batch_0,Frame_0,Layer_0,output_cov/Sigmoid:0,3,4 34 60 ,RAW_DATA:0.000002 0.000001 0.000001...
Source_0,Batch_0,Frame_0,Layer_1,output_bbox/BiasAdd:0,3,16 34 60 ,RAW_DATA:-0.021642 0.347364 0.898623...
```

**C++ Application (multiple formats):**
```csv
# CSV format
Source,Batch,Frame,Layer,LayerName,NumDims,Dimensions,DataType,RawTensorData
Source_0,Batch_0,Frame_0,Layer_0,output_cov/Sigmoid:0,3,4 34 60,FLOAT,RAW_DATA:0.000004 0.000001...

# JSON format
{
  "source_id": 0,
  "batch_id": 0,
  "frame_number": 0,
  "layers": [
    {
      "name": "output_cov/Sigmoid:0",
      "dimensions": [4, 34, 60],
      "data_type": "FLOAT",
      "values": [0.000004, 0.000001, ...]
    }
  ]
}
```

**Key Features of Tensor Extraction:**
- **Raw Numerical Data**: Extracts actual floating-point values from inference output tensors
- **Multiple Data Types**: Supports FLOAT (case 0), HALF/FP16 (case 1), INT8 (case 2), INT32 (case 3)
- **Smart Truncation**: Limits output to first 100 values per tensor to prevent massive files
- **Host Buffer Access**: Uses `tensor_meta->out_buf_ptrs_host[i]` for proper tensor data retrieval
- **Real-time Processing**: Extracts tensor data in real-time during inference pipeline execution

### Console Performance Metrics

**C Application (--perf flag):**
```
=== Performance Metrics ===
Total Batches: 150
Average FPS per source: 30.2
Total throughput: 120.8 FPS
```

**C++ Application (--perf flag):**
```
=== Performance Metrics ===
Sources: 3
Batch Size: 3
Total Batches: 45
Average FPS per source: 28.7
Total throughput: 86.1 FPS
Tensor Extraction Rate: 2 tensors/batch
Processing Time: 94μs per tensor
GPU Utilization: 78%
```

## Development Guidelines

### Critical Requirements

**C Application:**
1. **Exactly 4 Sources**: Hardcoded for 4 video sources
2. **Batch Size Consistency**: All batch-size settings must be 4 throughout pipeline
3. **CUDA_VER Environment**: Must export CUDA_VER=12.6 before building
4. **Memory Type**: Use nvbuf-memory-type=2 (unified memory) for performance

**C++ Application:**
1. **Flexible Sources**: Handles 1-64+ video sources
2. **Auto Batch Sizing**: Batch size automatically adjusts to source count
3. **YAML Configuration**: Uses pipeline_config.yaml for flexible settings
4. **Enhanced Features**: Multiple output formats, detailed logging, CLI options

### Code Structure

**C Application:**
- **C99 Standard**: Written in C99 with GStreamer
- **Error Handling**: Comprehensive error checking and recovery
- **Performance Focus**: Optimized for maximum throughput and minimal latency
- **Hardware Utilization**: Full GPU acceleration throughout pipeline

**C++ Application:**
- **C++17 Standard**: Modern C++ with RAII and smart pointers
- **Object-Oriented**: Modular design with PipelineBuilder and TensorProcessor classes
- **Enhanced Error Handling**: Exception-based error handling with detailed logging
- **Configuration Management**: YAML-based configuration with CLI overrides

### Extension Points

**C Application:**
1. **Custom Models**: Modify `configs/multi_inference_pgie_config.txt`
2. **Tensor Processing**: Extend `tensor_extract_probe()` function
3. **Output Formats**: Add custom output handlers
4. **Display Options**: Modify optional display branch

**C++ Application:**
1. **Custom Models**: Update YAML configuration with model paths
2. **Tensor Processing**: Extend `TensorProcessor` class methods
3. **Output Formats**: Add new export formats in `tensor_processor.cpp`
4. **Pipeline Customization**: Modify `PipelineBuilder` for custom elements

### Tensor Extraction Implementation Details

The raw tensor extraction is implemented in the `tensor_extract_probe()` function with the following key technical details:

**Buffer Access Method:**
```c
// Correct approach: Use host buffer pointers from tensor metadata
void *tensor_buffer = tensor_meta->out_buf_ptrs_host[i];
```

**Data Type Handling:**
```c
switch (layer_info->dataType) {
    case 0: /* FLOAT */
        float *float_data = (float*)tensor_buffer;
        // Process float values...
    case 1: /* HALF/FP16 */
        uint16_t *half_data = (uint16_t*)tensor_buffer;
        // Process FP16 values...
    case 2: /* INT8 */
        int8_t *int8_data = (int8_t*)tensor_buffer;
        // Process INT8 values...
    case 3: /* INT32 */
        int32_t *int32_data = (int32_t*)tensor_buffer;
        // Process INT32 values...
}
```

**Memory Safety & Performance:**
- Validates buffer existence with `tensor_meta->out_buf_ptrs_host && tensor_meta->out_buf_ptrs_host[i]`
- Calculates total elements as product of all tensor dimensions
- Truncates output to first 100 values per tensor to manage file sizes
- Uses `fprintf()` for efficient CSV writing with proper floating-point precision

### Common Issues

**C Application:**
1. **Model Loading**: Verify model paths in configuration files
2. **Memory Issues**: Check GPU memory with `nvidia-smi`
3. **Pipeline Deadlock**: Test sources individually with `gst-launch-1.0`
4. **Hardware Decoder**: Ensure NVDEC support for input formats

**C++ Application:**
1. **YAML Configuration**: Verify YAML file syntax and paths
2. **Batch Size Mismatch**: Ensure model supports flexible batch sizes
3. **Output Directory**: Check write permissions for output directory
4. **Source Validation**: Verify all input sources are accessible

## Important Notes

**C Application:**
- Requires exactly 4 video sources - no more, no less
- All batch-size configurations must be consistent throughout the pipeline
- Production-optimized for maximum throughput and efficiency
- Tensor extraction runs in real-time and outputs to CSV for analysis
- Optional display mode provides 2x2 tiled visualization for monitoring
- Performance monitoring built-in with `--perf` flag

**C++ Application:**
- Flexible source count: 1-64+ video sources supported
- Auto batch sizing: Batch size adjusts to source count
- Advanced features: Multiple output formats, detailed logging, YAML configuration
- Enhanced tensor extraction: CSV, JSON, binary formats with metadata
- Comprehensive CLI interface with extensive help and options
- Production-ready with robust error handling and monitoring

## Flexible System Architecture

### Dual Implementation Strategy

The repository contains two complementary implementations:

**C Application (deepstream-multi-inference-app):**
- Purpose: Maximum performance for fixed 4-source scenarios
- Use Case: Production environments requiring exactly 4 sources
- Advantages: Minimal overhead, maximum optimization, proven stability

**C++ Application (deepstream-multi-source-cpp):**
- Purpose: Flexible multi-source processing for R&D and variable scenarios
- Use Case: Development, testing, and scenarios with variable source counts
- Advantages: Configuration flexibility, enhanced features, modern codebase

### Architecture Comparison

| Aspect | C Application | C++ Application |
|--------|---------------|-----------------|
| **Source Count** | Fixed 4 | Flexible 1-64+ |
| **Batch Size** | Hardcoded 4 | Auto-adjusting |
| **Configuration** | Text files | YAML + CLI |
| **Output Formats** | CSV only | CSV, JSON, Binary |
| **Error Handling** | Basic | Comprehensive |
| **Performance** | Maximum optimization | Balanced features/performance |
| **Use Case** | Production deployment | Development & research |

### Development Workflow

1. **Start with C++ Application**: Use for development and testing with flexible source counts
2. **Test with C Application**: Validate performance with fixed 4-source configuration
3. **Deploy Appropriate Version**: Choose based on production requirements

### Key Technical Differences

**Memory Management:**
- C: Manual memory management with GStreamer objects
- C++: RAII with smart pointers for automatic resource management

**Configuration:**
- C: Static text configuration files
- C++: Dynamic YAML configuration with CLI overrides

**Extensibility:**
- C: Function-based extension points
- C++: Class-based architecture with inheritance and composition

### Performance Considerations

- The C application achieves slightly higher FPS due to fixed optimizations
- The C++ application provides better development experience and flexibility
- Both applications use the same underlying DeepStream optimizations
- Tensor extraction performance is identical between implementations