# DeepStream Multi-Source Batched Inference - Technical Architecture

## System Architecture Overview

This document provides a detailed technical architecture overview of the DeepStream Multi-Source Batched Inference application, focusing on the internal design, data flow, and optimization strategies.

## Table of Contents

1. [High-Level Architecture](#high-level-architecture)
2. [Pipeline Architecture](#pipeline-architecture)
3. [Memory Management](#memory-management)
4. [Batching Strategy](#batching-strategy)
5. [Tensor Processing](#tensor-processing)
6. [Performance Optimizations](#performance-optimizations)
7. [Threading Model](#threading-model)
8. [Error Handling](#error-handling)

## High-Level Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────────┐
│                    APPLICATION LAYER                             │
├─────────────────────────────────────────────────────────────────┤
│  Command Line Interface  │  Configuration Manager  │  Logger    │
└─────────────────────────────────────────────────────────────────┘
                                    │
┌─────────────────────────────────────────────────────────────────┐
│                    PIPELINE LAYER                                │
├─────────────────────────────────────────────────────────────────┤
│  Source Managers   │  Stream Multiplexer  │  Inference Engine  │
│  (4x Video Inputs) │  (Batching Logic)    │  (TensorRT)        │
└─────────────────────────────────────────────────────────────────┘
                                    │
┌─────────────────────────────────────────────────────────────────┐
│                    PROCESSING LAYER                              │
├─────────────────────────────────────────────────────────────────┤
│  Tensor Extractor  │  Metadata Processor  │  Display Renderer  │
│  (Probe Callbacks) │  (Per-Source Data)   │  (Optional)        │
└─────────────────────────────────────────────────────────────────┘
                                    │
┌─────────────────────────────────────────────────────────────────┐
│                    HARDWARE LAYER                                │
├─────────────────────────────────────────────────────────────────┤
│  NVDEC Decoders    │  GPU Memory          │  TensorRT Engine   │
│  (Hardware Decode) │  (Unified Memory)    │  (INT8 Inference)  │
└─────────────────────────────────────────────────────────────────┘
```

### Key Architectural Principles

1. **Fixed Batching**: Designed specifically for 4-source processing
2. **Hardware Acceleration**: Maximum use of GPU acceleration
3. **Zero-Copy Operations**: Unified memory for efficient data transfer
4. **Asynchronous Processing**: Non-blocking pipeline operations
5. **Modular Design**: Pluggable components for extensibility

## Pipeline Architecture

### GStreamer Pipeline Graph

```
                    ┌─ Source Bin 0 ─┐
                    │  uridecodebin  │
                    │  nvvidconv     │──┐
                    └────────────────┘  │
                                        │
                    ┌─ Source Bin 1 ─┐  │
                    │  uridecodebin  │  │
                    │  nvvidconv     │──┤
                    └────────────────┘  │    ┌─ nvstreammux ─┐
                                        ├─→  │   batch=4     │
                    ┌─ Source Bin 2 ─┐  │    │   1920x1080   │
                    │  uridecodebin  │  │    └───────────────┘
                    │  nvvidconv     │──┤              │
                    └────────────────┘  │              ▼
                                        │    ┌─────── nvinfer ────────┐
                    ┌─ Source Bin 3 ─┐  │    │     TensorRT Engine    │
                    │  uridecodebin  │  │    │     batch-size=4       │
                    │  nvvidconv     │──┘    │     INT8 Precision     │
                    └────────────────┘       └────────────────────────┘
                                                        │
                                               ┌────── tee ──────┐
                                               │                 │
                                               ▼                 ▼
                                    ┌─ Tensor Branch ─┐   ┌─ Display Branch ─┐
                                    │   queue         │   │   nvmultistreamtiler
                                    │   fakesink      │   │   nvvideoconvert   │
                                    │   [probe]       │   │   nvdsosd          │
                                    └─────────────────┘   │   sink             │
                                                         └────────────────────┘
```

### Pipeline Element Details

#### Source Bins (x4)

Each source bin handles one video input:

- **Element**: `nvurisrcbin` (preferred) or `uridecodebin`
- **Function**: Hardware-accelerated video decoding
- **Output**: NVMM memory surfaces
- **Optimization**: Hardware NVDEC decoder selection

```c
// Source bin configuration
g_object_set(G_OBJECT(uri_decode_bin),
    "file-loop", TRUE,           // Loop playback for files
    "cudadec-memtype", 0,        // Device memory
    "drop-on-latency", TRUE,     // Drop frames on latency
    NULL);
```

#### Stream Multiplexer (nvstreammux)

Central batching component:

- **Batch Formation**: Combines 4 sources into single batch
- **Resolution**: Standardizes all inputs to 1920x1080
- **Memory Type**: Unified memory for zero-copy
- **Timeout**: 40ms batch formation timeout

```c
// Streammux configuration
g_object_set(G_OBJECT(streammux),
    "batch-size", 4,                    // Fixed batch size
    "width", 1920,
    "height", 1080,
    "batched-push-timeout", 40000,      // 40ms timeout
    "nvbuf-memory-type", 2,             // Unified memory
    NULL);
```

#### Primary Inference Engine (nvinfer)

TensorRT-accelerated inference:

- **Engine**: Pre-built TensorRT INT8 engine
- **Batch Processing**: Simultaneous inference on 4 frames
- **Memory**: GPU memory with unified access
- **Optimization**: INT8 precision for speed

```c
// Inference engine configuration
g_object_set(G_OBJECT(pgie),
    "config-file-path", "configs/multi_inference_pgie_config.txt",
    "batch-size", 4,                    // Match streammux
    "interval", 0,                      // Process every frame
    NULL);
```

#### Tee Element

Pipeline branching for dual output:

- **Tensor Branch**: Extracts inference results
- **Display Branch**: Optional visualization
- **Memory Efficiency**: Shared buffers between branches

## Memory Management

### Memory Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                        CPU Memory                               │
├────────────────────────────────────────────────────────────────┤
│  Application Data  │  GStreamer Objects  │  Configuration     │
└────────────────────────────────────────────────────────────────┘
                                   │
                              Unified Memory
                                   │
┌────────────────────────────────────────────────────────────────┐
│                       GPU Memory                                │
├────────────────────────────────────────────────────────────────┤
│  Video Buffers     │  Inference Tensors  │  TensorRT Engine   │
│  (NVMM Surfaces)   │  (Input/Output)     │  (Model Weights)   │
└────────────────────────────────────────────────────────────────┘
```

### Memory Optimization Strategies

1. **Unified Memory (Type 2)**
   - Zero-copy operations between CPU and GPU
   - Automatic memory migration
   - Reduced latency and bandwidth usage

2. **Buffer Pooling**
   - Pre-allocated buffer pools
   - Efficient buffer reuse
   - Minimized allocation overhead

3. **NVMM Surfaces**
   - Hardware-optimized memory format
   - Direct GPU access
   - Minimal format conversion

### Memory Configuration

```c
// Throughout pipeline
nvbuf-memory-type = 2    // Unified memory

// For optimal performance
enable-memory-pool = 1   // Enable buffer pooling
pool-size = 8           // Buffer pool size
```

## Batching Strategy

### Batch Formation Logic

```
Time →
     ┌─ Source 0 ─┬─ Source 0 ─┬─ Source 0 ─┬─ Source 0 ─┐
     │  Frame 1   │  Frame 2   │  Frame 3   │  Frame 4   │
     └────────────┴────────────┴────────────┴────────────┘
     ┌─ Source 1 ─┬─ Source 1 ─┬─ Source 1 ─┬─ Source 1 ─┐
     │  Frame 1   │  Frame 2   │  Frame 3   │  Frame 4   │
     └────────────┴────────────┴────────────┴────────────┘
     ┌─ Source 2 ─┬─ Source 2 ─┬─ Source 2 ─┬─ Source 2 ─┐
     │  Frame 1   │  Frame 2   │  Frame 3   │  Frame 4   │
     └────────────┴────────────┴────────────┴────────────┘
     ┌─ Source 3 ─┬─ Source 3 ─┬─ Source 3 ─┬─ Source 3 ─┐
     │  Frame 1   │  Frame 2   │  Frame 3   │  Frame 4   │
     └────────────┴────────────┴────────────┴────────────┘
            │           │           │           │
            ▼           ▼           ▼           ▼
     ┌────────────┬────────────┬────────────┬────────────┐
     │  Batch 1   │  Batch 2   │  Batch 3   │  Batch 4   │
     │ [0,1,2,3]  │ [0,1,2,3]  │ [0,1,2,3]  │ [0,1,2,3]  │
     │  Frame 1   │  Frame 2   │  Frame 3   │  Frame 4   │
     └────────────┴────────────┴────────────┴────────────┘
```

### Synchronization Strategy

1. **Frame Alignment**: Ensures frames from all sources are batched together
2. **Timeout Handling**: 40ms maximum wait for batch completion
3. **Dropped Frame Handling**: Graceful handling of source delays
4. **EOS Management**: Proper end-of-stream handling for all sources

### Batch Processing Benefits

- **GPU Utilization**: Maximum parallelism with 4 concurrent inferences
- **Memory Bandwidth**: Efficient use of GPU memory bandwidth
- **Latency Optimization**: Reduced per-frame processing overhead
- **Throughput Maximization**: Optimal inference engine utilization

## Tensor Processing

### Tensor Extraction Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Inference Output                              │
├─────────────────────────────────────────────────────────────────┤
│  Batch Tensor [4, C, H, W]  →  Split by Source ID  →  Process  │
└─────────────────────────────────────────────────────────────────┘
                                          │
                        ┌─────────────────┼─────────────────┐
                        │                 │                 │
                        ▼                 ▼                 ▼
              ┌─ Source 0 Tensor ──┬─ Source 1 Tensor ──┬─ Source N... ─┐
              │  Layer Info        │  Layer Info        │               │
              │  Dimensions        │  Dimensions        │               │
              │  Data Pointer      │  Data Pointer      │               │
              └────────────────────┴────────────────────┴───────────────┘
                        │                 │                 │
                        ▼                 ▼                 ▼
              ┌─────────────────────────────────────────────────────────┐
              │               Tensor Output Processing                   │
              │  - CSV Export                                           │
              │  - Real-time Analysis                                   │
              │  - Performance Metrics                                  │
              │  - Custom Post-processing                               │
              └─────────────────────────────────────────────────────────┘
```

### Probe Callback Implementation

```c
static GstPadProbeReturn
tensor_extract_probe(GstPad *pad, GstPadProbeInfo *info, gpointer user_data)
{
    GstBuffer *buf = (GstBuffer *)info->data;
    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buf);
    
    // Process each frame in the batch
    for (NvDsMetaList *l_frame = batch_meta->frame_meta_list; 
         l_frame != NULL; l_frame = l_frame->next) {
        
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)(l_frame->data);
        guint source_id = frame_meta->source_id;
        
        // Extract tensor metadata for this source
        process_tensor_metadata(frame_meta, source_id);
    }
    
    return GST_PAD_PROBE_OK;
}
```

### Tensor Data Structure

```c
typedef struct {
    guint source_id;                    // Source identifier (0-3)
    guint batch_id;                     // Batch number
    guint frame_number;                 // Frame sequence number
    guint num_layers;                   // Number of output layers
    NvDsInferLayerInfo *layers;         // Layer information array
    gdouble processing_time_ms;         // Processing time
    GstClockTime timestamp;             // Buffer timestamp
} TensorData;
```

## Performance Optimizations

### Compiler Optimizations

```makefile
# Performance flags
CFLAGS += -O3 -DNDEBUG              # Maximum optimization
CFLAGS += -march=native -mtune=native  # CPU-specific optimizations
CFLAGS += -ffast-math               # Fast floating-point math
CFLAGS += -funroll-loops            # Loop unrolling
CFLAGS += -fomit-frame-pointer      # Frame pointer optimization
```

### Runtime Optimizations

1. **TensorRT Engine Pre-compilation**
   - INT8 precision for maximum speed
   - Optimized for specific batch size (4)
   - Platform-specific optimizations

2. **Memory Access Patterns**
   - Sequential memory access
   - Cache-friendly data structures
   - Minimized memory copies

3. **Pipeline Optimization**
   - Asynchronous processing
   - Queue-based buffering
   - Optimal element ordering

### GPU Optimizations

```
┌─────────────────────────────────────────────────────────────────┐
│                        GPU Utilization                          │
├─────────────────────────────────────────────────────────────────┤
│  NVDEC (Hardware Decoding)  │  CUDA Cores (Inference)          │
│  - 4x Parallel Decode       │  - Batched Inference             │
│  - Hardware Acceleration    │  - INT8 Precision                │
│  - Minimal CPU Load         │  - Optimized Kernels             │
└─────────────────────────────────────────────────────────────────┘
```

## Threading Model

### Thread Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Main Thread                                   │
├─────────────────────────────────────────────────────────────────┤
│  - Application Control                                          │
│  - Message Bus Handling                                         │
│  - Performance Monitoring                                       │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                 GStreamer Threads                                │
├─────────────────────────────────────────────────────────────────┤
│  Source Thread 0  │  Source Thread 1  │  Source Thread 2       │
│  Source Thread 3  │  Inference Thread │  Display Thread        │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    CUDA Threads                                 │
├─────────────────────────────────────────────────────────────────┤
│  - Hardware Decode Threads                                      │
│  - Inference Engine Threads                                     │
│  - Memory Copy Threads                                          │
└─────────────────────────────────────────────────────────────────┘
```

### Thread Synchronization

1. **GStreamer Pipeline**: Automatic thread management
2. **Probe Callbacks**: Executed in pipeline threads
3. **CUDA Streams**: Asynchronous GPU operations
4. **Message Bus**: Thread-safe communication

### Concurrency Considerations

- **Lock-free Design**: Minimized mutex usage
- **Atomic Operations**: Thread-safe counters
- **Pipeline Buffers**: Natural synchronization points
- **GPU Streams**: Concurrent kernel execution

## Error Handling

### Error Categories

1. **Configuration Errors**
   - Invalid model paths
   - Incorrect batch size settings
   - Missing dependencies

2. **Runtime Errors**
   - Source connection failures
   - GPU memory exhaustion
   - Pipeline state changes

3. **Performance Errors**
   - Frame drops
   - Timeout conditions
   - Resource contention

### Error Recovery Strategies

```c
// Example error handling
static void
handle_pipeline_error(GstMessage *msg, AppContext *ctx)
{
    GError *error = NULL;
    gchar *debug_info = NULL;
    
    gst_message_parse_error(msg, &error, &debug_info);
    
    // Log detailed error information
    g_printerr("Pipeline Error: %s\n", error->message);
    g_printerr("Debug Info: %s\n", debug_info);
    
    // Attempt recovery based on error type
    if (is_recoverable_error(error)) {
        attempt_pipeline_recovery(ctx);
    } else {
        initiate_graceful_shutdown(ctx);
    }
    
    g_error_free(error);
    g_free(debug_info);
}
```

### Monitoring and Diagnostics

1. **Health Checks**: Regular pipeline health monitoring
2. **Performance Metrics**: Real-time performance tracking
3. **Resource Monitoring**: GPU and memory usage tracking
4. **Logging System**: Comprehensive error and debug logging

## Extensibility

### Plugin Architecture

The application is designed for easy extension:

1. **Custom Models**: Easy model swapping via configuration
2. **Post-processing**: Pluggable tensor processing callbacks
3. **Output Formats**: Multiple output format support
4. **Display Modes**: Customizable visualization options

### Integration Points

1. **Probe Callbacks**: Custom processing injection points
2. **Configuration System**: Flexible parameter management
3. **Output Interfaces**: Multiple output format support
4. **Message System**: Event-driven architecture

This architecture provides a solid foundation for high-performance, multi-source video processing with batched inference while maintaining flexibility for customization and extension.