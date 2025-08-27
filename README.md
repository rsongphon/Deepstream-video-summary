# DeepStream Multi-Source Batched Inference Application

A high-performance NVIDIA DeepStream application designed for simultaneous processing of exactly 4 video sources with batched inference and tensor extraction capabilities.

## Overview

This application demonstrates optimized multi-source video processing using DeepStream SDK 7.1. It processes 4 video sources simultaneously through a batched inference pipeline, extracting tensor data for each source while providing optional visual output for monitoring and debugging.

### Key Features

- **Fixed 4-Source Processing**: Designed specifically for processing exactly 4 video sources simultaneously
- **Batched Inference**: Optimal batch processing with batch-size=4 for maximum GPU utilization
- **Tensor Extraction**: Real-time extraction and output of inference tensor data for each source
- **Hardware Acceleration**: Full hardware-accelerated video decoding (NVDEC) and inference (TensorRT)
- **Optional Display**: 2x2 tiled display output for visualization and monitoring
- **Performance Optimized**: Maximum optimization for throughput and low latency
- **Flexible Input**: Supports various video formats, RTSP streams, and camera inputs

## Architecture

```
[Source 1] ──┐
[Source 2] ──┼─→ [nvstreammux] ─→ [nvinfer] ─→ [tee] ─┬─→ [tensor_probe] → TENSOR OUTPUT
[Source 3] ──┤    (batch=4)        (TRT)              │
[Source 4] ──┘                                        │
                                                       └─→ [tiler] → [osd] → [display]
                                                            (optional branch)
```

### Pipeline Components

1. **Source Bins**: Hardware-accelerated video decoding for each input
2. **nvstreammux**: Batches 4 sources into single inference batch
3. **nvinfer**: TensorRT-optimized inference engine
4. **Tensor Probe**: Extracts and processes inference tensor metadata
5. **Display Branch**: Optional 2x2 tiled visualization (when enabled)

## Prerequisites

### System Requirements

- **Operating System**: Ubuntu 22.04 LTS
- **GPU**: NVIDIA GPU with compute capability 6.0+
- **Driver**: NVIDIA Driver 535+
- **CUDA**: CUDA Toolkit 12.6
- **DeepStream**: NVIDIA DeepStream SDK 7.1
- **TensorRT**: TensorRT 10.3.0.26
- **GStreamer**: GStreamer 1.20.3

### Software Dependencies

```bash
# Core dependencies
sudo apt-get update
sudo apt-get install build-essential cmake git
sudo apt-get install libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev
sudo apt-get install libgstreamer-plugins-bad1.0-dev libgstreamer-plugins-good1.0-dev

# Optional dependencies for analysis
sudo apt-get install cppcheck valgrind
```

## Installation

### 1. Clone/Navigate to Project

```bash
cd /opt/nvidia/deepstream/deepstream-7.1/sources/apps/deepstream-multi-inference
```

### 2. Check Dependencies

```bash
make check-deps
```

### 3. Build Application

```bash
# Build optimized release version
make

# Or build debug version
make debug

# Or build with profiling
make profile
```

### 4. Install (Optional)

```bash
sudo make install
```

## Usage

### Basic Usage

```bash
# Process 4 video files (headless mode)
./deepstream-multi-inference-app video1.mp4 video2.mp4 video3.mp4 video4.mp4

# Process with display output
./deepstream-multi-inference-app --enable-display video1.mp4 video2.mp4 video3.mp4 video4.mp4

# Process RTSP streams
./deepstream-multi-inference-app rtsp://cam1/stream rtsp://cam2/stream rtsp://cam3/stream rtsp://cam4/stream
```

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--enable-display` | Enable 2x2 tiled visual output | Disabled |
| `--no-display` | Disable display (headless mode) | Enabled |
| `--config FILE` | Use custom configuration file | Built-in config |
| `--model FILE` | Use custom model configuration | Built-in model config |
| `--gpu-id ID` | Specify GPU device ID | 0 |
| `--perf` | Enable performance measurement | Disabled |
| `--help`, `-h` | Show help message | - |

### Advanced Usage Examples

```bash
# Custom configuration with performance monitoring
./deepstream-multi-inference-app --config configs/multi_inference_config.yml \
  --model configs/custom_model.txt --perf \
  video1.mp4 video2.mp4 video3.mp4 video4.mp4

# Multi-GPU setup (specify GPU)
./deepstream-multi-inference-app --gpu-id 1 \
  rtsp://camera1 rtsp://camera2 rtsp://camera3 rtsp://camera4

# Display mode with custom model
./deepstream-multi-inference-app --enable-display \
  --model configs/yolo_config.txt \
  sample1.mp4 sample2.mp4 sample3.mp4 sample4.mp4
```

## Configuration

### Model Configuration

Edit `configs/multi_inference_pgie_config.txt` to customize the inference model:

```ini
# Model paths
onnx-file=path/to/your/model.onnx
model-engine-file=path/to/your/model_b4.engine
labelfile-path=path/to/labels.txt

# Batch configuration (must be 4)
batch-size=4

# Performance settings
network-mode=1              # INT8 mode
output-tensor-meta=1        # Enable tensor extraction
nvbuf-memory-type=2         # Unified memory
```

### Pipeline Configuration

Edit `configs/multi_inference_config.yml` for pipeline settings:

```yaml
streammux:
  batch-size: 4                    # Fixed for 4 sources
  batched-push-timeout: 40000      # 40ms batch timeout
  width: 1920                      # Input resolution
  height: 1080
  nvbuf-memory-type: 2            # Unified memory

primary-gie:
  config-file-path: configs/multi_inference_pgie_config.txt
  batch-size: 4                    # Must match streammux
  interval: 0                      # Process every frame
```

## Output

### Tensor Output

The application generates `tensor_output.csv` with extracted tensor data:

```csv
Source,Batch,Frame,Layer,LayerName,NumDims,Dimensions
Source_0,1,1,0,output_coverage/Sigmoid,3,1 1 16800
Source_1,1,1,0,output_coverage/Sigmoid,3,1 1 16800
Source_2,1,1,0,output_coverage/Sigmoid,3,1 1 16800
Source_3,1,1,0,output_coverage/Sigmoid,3,1 1 16800
```

### Console Output

Real-time processing information:

```
=== Batch #1 - Tensor Extraction ===
Source 0 - Frame 1:
  Tensor Output Layers: 2
    Layer 0: output_coverage/Sigmoid
      Data Type: 1
      Dimensions: 1 1 16800

=== Performance Metrics ===
Total Batches: 30
Average FPS per source: 28.5
Total throughput: 114.0 FPS
Source 0 frames: 30
Source 1 frames: 30
Source 2 frames: 30
Source 3 frames: 30
```

### Display Output (Optional)

When `--enable-display` is used:
- 2x2 tiled layout showing all 4 sources
- Bounding boxes and detection results
- Source ID and object count overlays
- Real-time performance metrics

## Performance Optimization

### Hardware Optimizations

1. **GPU Memory**: Uses unified memory for zero-copy operations
2. **Hardware Decoding**: NVDEC hardware video decoding
3. **TensorRT**: Optimized inference engines
4. **Batch Processing**: 4-source simultaneous processing

### Software Optimizations

1. **Compiler Flags**: `-O3`, `-march=native`, `-ffast-math`
2. **Memory Management**: Efficient buffer pooling and reuse
3. **Threading**: Multi-threaded processing with OpenMP
4. **Pipeline**: Asynchronous processing pipeline

### Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| Throughput | 30+ FPS per source | For 1080p input |
| Latency | <100ms end-to-end | Including tensor extraction |
| GPU Utilization | >80% | Optimal GPU usage |
| Memory Usage | <4GB per stream | Efficient memory management |

### Benchmarking

```bash
# Build performance-optimized version
make release

# Run benchmark
make benchmark SOURCES='vid1.mp4 vid2.mp4 vid3.mp4 vid4.mp4'

# Profile performance
make profile
./deepstream-multi-inference-app --perf video1.mp4 video2.mp4 video3.mp4 video4.mp4
gprof deepstream-multi-inference-app gmon.out > performance_profile.txt
```

## Troubleshooting

### Common Issues

#### 1. Model Loading Failures

```
Error: Failed to create PGIE
```

**Solution**: Check model paths in configuration files
```bash
# Verify model files exist
ls -la configs/
ls -la ../../../../samples/models/Primary_Detector/
```

#### 2. Hardware Decoder Issues

```
Error: Hardware decoder not selected
```

**Solution**: Ensure NVDEC is available and supported format is used
```bash
# Check GPU capabilities
nvidia-smi

# Use supported formats (H.264, H.265)
ffprobe your_video.mp4
```

#### 3. Memory Issues

```
Error: Failed to allocate memory
```

**Solution**: Check GPU memory and reduce batch size if needed
```bash
# Check GPU memory
nvidia-smi

# Monitor memory usage
watch -n 1 nvidia-smi
```

#### 4. Pipeline Deadlock

```
Pipeline stuck, no output
```

**Solution**: Check source connectivity and format compatibility
```bash
# Test sources individually
gst-launch-1.0 uridecodebin uri=file:///path/to/video.mp4 ! fakesink

# Check GStreamer debug output
GST_DEBUG=3 ./deepstream-multi-inference-app video1.mp4 video2.mp4 video3.mp4 video4.mp4
```

### Debug Mode

Build and run in debug mode for detailed information:

```bash
make debug
./deepstream-multi-inference-app --perf video1.mp4 video2.mp4 video3.mp4 video4.mp4
```

### Memory Analysis

```bash
make memcheck
# or manually:
valgrind --tool=memcheck --leak-check=full ./deepstream-multi-inference-app --help
```

## Development

### Build Targets

```bash
make help           # Show all available targets
make all            # Build release version (default)
make debug          # Build debug version
make profile        # Build with profiling
make clean          # Clean build artifacts
make install        # Install to system directory
make analyze        # Run static analysis
make memcheck       # Memory leak detection
make check-deps     # Check dependencies
```

### Code Structure

```
deepstream_multi_inference_app.c    # Main application source
├── main()                          # Entry point and argument parsing
├── setup_pipeline()               # Pipeline construction
├── create_source_bin()            # Source bin creation
├── tensor_extract_probe()         # Tensor extraction callback
├── osd_sink_pad_buffer_probe()    # Display overlay callback
└── bus_call()                     # Message handling
```

### Extending the Application

#### Adding Custom Models

1. Update model configuration in `configs/multi_inference_pgie_config.txt`
2. Ensure batch-size=4 for your model
3. Generate TensorRT engine with batch size 4:

```bash
/usr/src/tensorrt/bin/trtexec --onnx=your_model.onnx --batch=4 \
  --int8 --workspace=1024 --saveEngine=your_model_b4.engine
```

#### Adding Custom Post-Processing

Modify the `tensor_extract_probe()` function to add custom tensor processing:

```c
// Extract tensor data
if (user_meta->base_meta.meta_type == NVDSINFER_TENSOR_OUTPUT_META) {
    NvDsInferTensorMeta *tensor_meta = (NvDsInferTensorMeta *)user_meta->user_meta_data;
    
    // Add your custom processing here
    process_custom_tensor(tensor_meta, source_id);
}
```

## Performance Analysis

### Profiling Tools

1. **Built-in Performance Measurement**: Use `--perf` flag
2. **NVIDIA Nsight Systems**: For detailed GPU profiling
3. **gprof**: For CPU profiling (build with `make profile`)
4. **Valgrind**: For memory analysis

### Performance Metrics

The application provides detailed performance metrics:

- **Batch Processing Rate**: Batches processed per second
- **Per-Source FPS**: Frame rate for each individual source
- **Total Throughput**: Combined processing rate
- **GPU Utilization**: GPU compute and memory utilization
- **Tensor Extraction Time**: Time spent on tensor processing

## Support and Contributing

### Getting Help

1. Check troubleshooting section above
2. Review NVIDIA DeepStream documentation
3. Check application logs and error messages
4. Use debug mode for detailed information

### Filing Issues

When reporting issues, please include:

1. System specifications (GPU, driver versions)
2. Input video specifications
3. Configuration files used
4. Complete error messages
5. Steps to reproduce

### Contributing

1. Follow C99 coding standards
2. Add comprehensive comments for new features
3. Include performance impact analysis
4. Test with various input formats
5. Update documentation as needed

## License

This application is provided under the NVIDIA Proprietary License. See the license header in source files for details.

## Changelog

### Version 1.0.0
- Initial release with 4-source batched inference
- Hardware-accelerated processing pipeline
- Tensor extraction and CSV output
- Optional display mode with 2x2 tiling
- Comprehensive configuration system
- Performance optimization and monitoring
- Full documentation and examples