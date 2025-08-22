# DeepStream Video Summary System

A high-performance video summarization system built on NVIDIA DeepStream SDK 7.1 that accepts multiple video sources, comprehends content using state-of-the-art AI models, and generates text summaries using Triton Inference Server.

## Features

- **Multi-Source Input**: Supports files, RTSP streams, USB/CSI cameras, HTTP streams
- **AI-Powered Analysis**: Uses advanced computer vision and NLP models
- **High Performance**: Leverages GPU acceleration and optimized inference
- **Scalable Architecture**: Modular design with Triton Inference Server
- **Real-time Processing**: Optimized for low-latency video analysis

## Quick Start

⚠️ **Note**: This project is currently in the planning phase. Implementation will follow the step-by-step plan outlined in [PLAN.md](PLAN.md).

### Prerequisites

- NVIDIA DeepStream SDK 7.1
- CUDA 12.6+
- TensorRT 8.6+
- GStreamer 1.20.3+
- Triton Inference Server 24.08+
- OpenCV 4.5+
- CMake 3.22+

### Installation (Future)

```bash
# Set required environment variable
export CUDA_VER=12.6

# Navigate to project directory
cd /opt/nvidia/deepstream/deepstream-7.1/sources/apps/deepstream-video-summary

# Build the project
mkdir build && cd build
cmake ..
make -j$(nproc)

# Run with sample video
./deepstream-video-summary -c ../configs/sample_config.txt -i sample_video.mp4
```

## Project Structure

```
deepstream-video-summary/
├── src/                   # Source code
│   ├── pipeline/          # DeepStream pipeline management
│   ├── inference/         # Triton client and model management
│   ├── processing/        # Video processing and feature extraction
│   ├── summary/           # Summary generation logic
│   └── utils/             # Utilities and helpers
├── models/                # AI models and configurations
│   ├── triton_repo/       # Triton model repository
│   └── configs/           # Model configuration files
├── configs/               # Application configuration files
├── tests/                 # Unit and integration tests
├── docker/                # Containerization files
├── PLAN.md               # Step-by-step implementation plan
├── CLAUDE.md             # Development guidance for Claude Code
├── README.md             # This file
└── CMakeLists.txt        # Build configuration
```

## Implementation Phases

This project follows a structured development approach outlined in [PLAN.md](PLAN.md):

1. **Phase 1**: Foundation and Basic Pipeline (Week 1-2)
2. **Phase 2**: Video Processing and Feature Extraction (Week 3-4)
3. **Phase 3**: Triton Integration (Week 5-6)
4. **Phase 4**: Summary Generation (Week 7-8)
5. **Phase 5**: Advanced Features and Optimization (Week 9-10)
6. **Phase 6**: Testing and Integration (Week 11-12)

## Development Guidelines

### Getting Started with Development

1. **Read the Plan**: Start with [PLAN.md](PLAN.md) for the complete implementation roadmap
2. **Check Development Guide**: Refer to [CLAUDE.md](CLAUDE.md) for coding patterns and debugging tips
3. **Follow Phases**: Implement features according to the phase-by-phase plan
4. **Test Incrementally**: Test each component thoroughly before moving to the next phase

### Build Instructions

```bash
# Ensure environment is set up
export CUDA_VER=12.6

# Create build directory
mkdir -p build && cd build

# Configure build
cmake .. -DCMAKE_BUILD_TYPE=Debug

# Build (when source files are added)
make -j$(nproc)
```

### Testing

```bash
# Run unit tests (when implemented)
cd build && ctest

# Run with debug logging
GST_DEBUG=3 ./deepstream-video-summary --verbose

# Performance profiling
nsys profile ./deepstream-video-summary -c config.txt
```

## Architecture Overview

### High-Level Data Flow

```
Video Input → DeepStream → Feature → Triton → Summary → Text
Sources       Pipeline     Extraction  Inference  Generation  Output
```

### Core Components

- **DeepStreamPipeline**: Manages GStreamer pipeline and metadata flow
- **TritonClient**: Handles AI model inference via Triton server
- **FeatureExtractor**: Processes video frames and extracts features
- **SummaryGenerator**: Coordinates multimodal analysis and text generation
- **ConfigManager**: Manages application and model configurations

## Configuration

### DeepStream Configuration Example

```ini
[application]
enable-perf-measurement=1

[source0]
enable=1
type=3
uri=file:///path/to/video.mp4

[streammux]
batch-size=4
width=1920
height=1080

[primary-gie]
enable=1
config-file=config_infer_primary.txt
```

### Triton Model Configuration Example

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
```

## Performance Targets

- **Throughput**: ≥30 FPS for 1080p video
- **Latency**: <100ms end-to-end processing
- **Memory Usage**: <4GB per video stream
- **Accuracy**: >85% summary quality score

## Current Status

🚧 **Development Status**: Planning Phase

- [x] Project structure created
- [x] Implementation plan developed
- [x] Development guidelines established
- [ ] Phase 1: Foundation (In Progress)
- [ ] Phase 2: Video Processing
- [ ] Phase 3: Triton Integration
- [ ] Phase 4: Summary Generation
- [ ] Phase 5: Advanced Features
- [ ] Phase 6: Testing and Deployment

## Contributing

This project follows a structured development approach. Before contributing:

1. Review [PLAN.md](PLAN.md) for current phase requirements
2. Check [CLAUDE.md](CLAUDE.md) for coding standards and patterns
3. Ensure all tests pass before submitting changes
4. Follow the incremental development approach

## Debugging

Common debugging commands:

```bash
# GStreamer pipeline debugging
export GST_DEBUG=3
export GST_DEBUG_DUMP_DOT_DIR=./debug

# Monitor GPU memory
nvidia-smi -l 1

# Check Triton server status
curl localhost:8000/v2/health/ready
```

For detailed debugging guidance, see [CLAUDE.md](CLAUDE.md).

## License

This project is part of the NVIDIA DeepStream SDK ecosystem. Please refer to NVIDIA's licensing terms for usage guidelines.

## Support

- **Documentation**: See [PLAN.md](PLAN.md) and [CLAUDE.md](CLAUDE.md)
- **NVIDIA DeepStream**: [Official Documentation](https://docs.nvidia.com/metropolis/deepstream/dev-guide/index.html)
- **Triton Inference Server**: [Official Documentation](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/index.html)

---

**Note**: This README will be updated as development progresses through each phase.