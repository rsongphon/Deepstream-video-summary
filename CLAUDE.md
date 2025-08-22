# CLAUDE.md - DeepStream Video Summary System

This file provides guidance for Claude Code when working on the DeepStream Video Summary System project.

## Project Overview

This is a video summarization system built on NVIDIA DeepStream SDK 7.1 that:
- Accepts multiple video sources (files, RTSP, USB/CSI cameras, HTTP streams)
- Comprehends video content using AI models via Triton Inference Server
- Generates coherent text summaries using multimodal analysis
- Uses C++17 for core implementation with GStreamer and DeepStream plugins

## Build System and Commands

### Prerequisites
```bash
# Ensure CUDA_VER is set (required for all DeepStream builds)
export CUDA_VER=12.6

# Verify DeepStream installation
ls /opt/nvidia/deepstream/deepstream-7.1/lib/

# Check Triton client libraries
python3 -c "import tritonclient.grpc"
```

### Building the Project
```bash
# Navigate to project directory
cd /opt/nvidia/deepstream/deepstream-7.1/sources/apps/deepstream-video-summary

# Create build directory
mkdir -p build && cd build

# Configure with CMake
cmake .. -DCMAKE_BUILD_TYPE=Debug

# Build the project
make -j$(nproc)

# Install (optional)
make install
```

### Running the Application
```bash
# Basic usage with video file
./video-summary-app -c configs/basic_config.txt -i input_video.mp4

# With custom Triton server
./video-summary-app -c configs/triton_config.txt -t localhost:8001

# Debug mode with verbose logging
GST_DEBUG=3 ./video-summary-app -c configs/debug_config.txt -v
```

## Architecture Components

### Core Classes and Responsibilities

1. **DeepStreamPipeline** (`src/pipeline/`)
   - Manages GStreamer pipeline lifecycle
   - Handles element creation, linking, and state changes
   - Integrates nvinferserver for Triton communication

2. **TritonClient** (`src/inference/`)
   - Manages gRPC connections to Triton Inference Server
   - Handles model loading, batching, and inference requests
   - Provides async inference capabilities

3. **FeatureExtractor** (`src/processing/`)
   - Extracts visual features from video frames
   - Performs keyframe selection and scene detection
   - Handles temporal analysis and feature aggregation

4. **SummaryGenerator** (`src/summary/`)
   - Coordinates multimodal feature fusion
   - Interfaces with language models for text generation
   - Formats and outputs final summaries

5. **Utilities** (`src/utils/`)
   - ConfigManager: YAML/INI configuration parsing
   - Logger: Structured logging with performance metrics
   - MetadataParser: DeepStream metadata extraction

### Data Flow Architecture
```
Video Input → DeepStream Pipeline → Feature Extraction → Triton Inference → Summary Generation → Text Output
     ↓              ↓                    ↓                 ↓                    ↓
GStreamer      Metadata Probes     Frame Processing    AI Models         Text Formatting
Elements       NvDsBatchMeta       Visual Features     Transformers      JSON/Plain Text
```

## Development Patterns

### GStreamer Element Creation Pattern
```cpp
// Standard element creation with error checking
GstElement* createElement(const std::string& factory, const std::string& name) {
    GstElement* element = gst_element_factory_make(factory.c_str(), name.c_str());
    if (!element) {
        throw std::runtime_error("Failed to create element: " + factory);
    }
    return element;
}

// Pipeline linking with capability negotiation
bool linkElements(GstElement* src, GstElement* dest) {
    if (!gst_element_link(src, dest)) {
        GST_ERROR("Failed to link %s to %s", GST_ELEMENT_NAME(src), GST_ELEMENT_NAME(dest));
        return false;
    }
    return true;
}
```

### Metadata Probe Pattern
```cpp
static GstPadProbeReturn metadataProbe(GstPad* pad, GstPadProbeInfo* info, gpointer user_data) {
    GstBuffer* buffer = GST_PAD_PROBE_INFO_BUFFER(info);
    NvDsBatchMeta* batch_meta = gst_buffer_get_nvds_batch_meta(buffer);
    
    if (!batch_meta) {
        return GST_PAD_PROBE_OK;
    }
    
    // Process frame metadata
    for (NvDsMetaList* l_frame = batch_meta->frame_meta_list; l_frame; l_frame = l_frame->next) {
        NvDsFrameMeta* frame_meta = (NvDsFrameMeta*)l_frame->data;
        // Extract features, store frame data, etc.
    }
    
    return GST_PAD_PROBE_OK;
}
```

### Triton Inference Pattern
```cpp
// Async inference with callback
class TritonInferenceCallback {
public:
    void onComplete(const InferenceResponse& response) {
        // Process inference results
        processInferenceResults(response);
    }
    
    void onError(const std::string& error) {
        LOG_ERROR("Inference failed: {}", error);
    }
};

// Batch inference setup
auto callback = std::make_shared<TritonInferenceCallback>();
tritonClient->inferAsync("video_summarizer", input_tensors, callback);
```

### Configuration Management Pattern
```cpp
// YAML configuration loading
class VideoSummaryConfig {
private:
    YAML::Node config_;
    
public:
    bool load(const std::string& config_path) {
        try {
            config_ = YAML::LoadFile(config_path);
            return validateConfig();
        } catch (const YAML::Exception& e) {
            LOG_ERROR("Config error: {}", e.what());
            return false;
        }
    }
    
    template<typename T>
    T get(const std::string& key, const T& default_value = T{}) const {
        return config_[key].as<T>(default_value);
    }
};
```

## Debugging Guidelines

### Common Issues and Solutions

#### 1. Pipeline State Changes
**Problem**: Pipeline fails to change state to PLAYING
**Debug Steps**:
```bash
# Enable GStreamer debug
export GST_DEBUG=3
export GST_DEBUG_FILE=pipeline_debug.log

# Check element states
gst-inspect-1.0 nvstreammux
gst-inspect-1.0 nvinferserver
```

#### 2. Triton Connection Issues
**Problem**: Cannot connect to Triton server
**Debug Steps**:
```bash
# Test Triton server connectivity
curl -v localhost:8000/v2/health/ready

# Check model repository
curl localhost:8000/v2/models

# Validate model configs
tritonserver --model-repository=./models/triton_repo --log-verbose=1
```

#### 3. Memory Issues
**Problem**: GPU memory exhaustion or leaks
**Debug Steps**:
```bash
# Monitor GPU memory
nvidia-smi -l 1

# Check for memory leaks with Valgrind
valgrind --tool=memcheck --leak-check=full ./video-summary-app

# Profile GPU usage
nsys profile --trace=cuda,nvtx ./video-summary-app
```

#### 4. Performance Issues
**Problem**: Low FPS or high latency
**Debug Areas**:
- Batch size configuration
- GPU utilization
- Memory bandwidth
- Pipeline buffer pools
- Inference model optimization

### Debugging Tools and Techniques

#### GStreamer Pipeline Debugging
```bash
# Visualize pipeline graph
export GST_DEBUG_DUMP_DOT_DIR=./debug
# Pipeline will generate .dot files that can be converted to images
dot -Tpng pipeline.dot -o pipeline.png
```

#### DeepStream Metadata Debugging
```cpp
// Add debug prints in metadata probes
void debugFrameMetadata(NvDsFrameMeta* frame_meta) {
    g_print("Frame %d: %dx%d, pts=%lu\n", 
            frame_meta->frame_num,
            frame_meta->source_frame_width,
            frame_meta->source_frame_height,
            frame_meta->buf_pts);
}
```

#### Performance Profiling
```cpp
// Add timing measurements
class Timer {
private:
    std::chrono::high_resolution_clock::time_point start_;
    
public:
    Timer() : start_(std::chrono::high_resolution_clock::now()) {}
    
    double elapsed() const {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start_).count();
    }
};

// Usage
Timer timer;
performInference();
LOG_INFO("Inference took {:.2f}ms", timer.elapsed());
```

## Configuration Files

### DeepStream Configuration Structure
```ini
[application]
enable-perf-measurement=1
perf-measurement-interval-sec=5

[source0]
enable=1
type=3  # File source
uri=file:///path/to/video.mp4
gpu-id=0

[streammux]
gpu-id=0
batch-size=4
width=1920
height=1080
batched-push-timeout=40000

[primary-gie]
enable=1
gpu-id=0
gie-unique-id=1
config-file=config_infer_primary.txt

[tracker]
enable=1
tracker-width=960
tracker-height=544
ll-lib-file=/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so
```

### Triton Model Configuration
```protobuf
name: "video_summarizer"
backend: "ensemble"
max_batch_size: 8

input [
  {
    name: "input_frames"
    data_type: TYPE_FP32
    dims: [ 3, 224, 224 ]
  }
]

output [
  {
    name: "summary_text"
    data_type: TYPE_STRING
    dims: [ 1 ]
  }
]
```

## Testing Strategy

### Unit Testing
```cpp
// Use Google Test framework
TEST(DeepStreamPipelineTest, InitializationTest) {
    PipelineConfig config;
    config.batch_size = 4;
    config.input_width = 1920;
    config.input_height = 1080;
    
    DeepStreamPipeline pipeline;
    ASSERT_TRUE(pipeline.initialize(config));
    ASSERT_EQ(pipeline.getState(), GST_STATE_NULL);
}

TEST(TritonClientTest, ConnectionTest) {
    TritonClient client("localhost:8001");
    ASSERT_TRUE(client.connect());
    ASSERT_TRUE(client.isModelReady("video_summarizer"));
}
```

### Integration Testing
```bash
# Test with sample video
./scripts/test_integration.sh sample_videos/test.mp4

# Performance benchmarking
./scripts/benchmark.sh --input-dir test_videos/ --output-dir results/
```

## Important Notes

### DeepStream-Specific Considerations
- Always check element capabilities before linking
- Use nvbuf-memory-type=0 for unified memory
- Monitor pipeline bus for error messages
- Properly handle EOS (End of Stream) events

### Triton Integration Best Practices
- Use model versioning for rollback capability
- Implement proper request batching for efficiency
- Monitor model performance metrics
- Handle network failures gracefully

### Performance Optimization
- Enable FP16 inference when possible
- Use optimal batch sizes (typically 4-8)
- Minimize CPU-GPU memory transfers
- Profile bottlenecks regularly

### Error Handling Patterns
- Always check return values from GStreamer functions
- Implement proper cleanup in destructors
- Use RAII for resource management
- Log errors with sufficient context for debugging

## File Naming Conventions

- Headers: PascalCase.h (e.g., `DeepStreamPipeline.h`)
- Sources: PascalCase.cpp (e.g., `DeepStreamPipeline.cpp`)
- Configs: snake_case.txt/yaml (e.g., `video_summary_config.yaml`)
- Tests: test_snake_case.cpp (e.g., `test_pipeline.cpp`)

## Dependencies and Libraries

### Required Libraries
- DeepStream SDK 7.1
- GStreamer 1.20.3+
- OpenCV 4.5+
- Triton Client Libraries
- YAML-cpp
- spdlog (for logging)
- Google Test (for testing)

### Optional Libraries
- Prometheus C++ client (for metrics)
- gRPC (for remote management)
- OpenTelemetry (for tracing)

Remember to follow the step-by-step plan in PLAN.md and implement features incrementally with thorough testing at each phase.