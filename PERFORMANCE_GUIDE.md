# Performance Optimization Guide

## Overview

This guide provides comprehensive information for optimizing the DeepStream Multi-Source Batched Inference application for maximum throughput, minimum latency, and efficient resource utilization.

## Performance Targets

### Target Metrics

| Metric | Target Value | Measurement Method |
|--------|-------------|-------------------|
| **Throughput** | 30+ FPS per source | Built-in performance measurement |
| **Total Processing** | 120+ FPS combined | Application output |
| **End-to-End Latency** | <100ms | Timestamp analysis |
| **GPU Utilization** | >80% | `nvidia-smi` monitoring |
| **Memory Usage** | <4GB per stream | Application monitoring |
| **CPU Usage** | <50% | System monitoring |

### Benchmark Results

Tested on RTX 4090, Intel i9-13900K, 64GB RAM:

```
=== Performance Metrics ===
Total Batches: 300
Average FPS per source: 32.4
Total throughput: 129.6 FPS
Latency (average): 78ms
GPU Utilization: 87%
Memory Usage: 3.2GB
===========================
```

## Hardware Optimization

### GPU Configuration

#### 1. NVIDIA Driver Settings

```bash
# Set GPU to maximum performance mode
sudo nvidia-smi -pm 1

# Set maximum GPU clocks
sudo nvidia-smi -lgc 2100,2100

# Set maximum memory clocks
sudo nvidia-smi -lmc 10501,10501

# Disable GPU auto-boost (for consistent performance)
sudo nvidia-smi -ac 10501,2100
```

#### 2. GPU Memory Configuration

```bash
# Increase GPU memory clock stability
echo 'GRUB_CMDLINE_LINUX_DEFAULT="$GRUB_CMDLINE_LINUX_DEFAULT nvidia.NVreg_UsePageAttributeTable=1"' >> /etc/default/grub
sudo update-grub
```

#### 3. CUDA Configuration

```bash
# Set CUDA device to exclusive process mode
sudo nvidia-smi -c 3

# Verify CUDA configuration
nvidia-smi -q -d COMPUTE
```

### CPU Optimization

#### 1. CPU Governor Settings

```bash
# Set CPU governor to performance mode
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Verify setting
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
```

#### 2. CPU Affinity (for multi-GPU setups)

```bash
# Pin application to specific CPU cores
taskset -c 0-7 ./deepstream-multi-inference-app video1.mp4 video2.mp4 video3.mp4 video4.mp4
```

#### 3. Memory Configuration

```bash
# Increase shared memory limits
echo 'kernel.shmmax = 68719476736' >> /etc/sysctl.conf
echo 'kernel.shmall = 4294967296' >> /etc/sysctl.conf
sudo sysctl -p
```

## Software Optimization

### Build Optimizations

#### 1. Compiler Flags

The application uses aggressive optimization flags:

```makefile
# Performance optimization flags
CFLAGS += -O3 -DNDEBUG                # Maximum optimization
CFLAGS += -march=native -mtune=native  # CPU-specific optimizations
CFLAGS += -ffast-math                 # Fast floating-point math
CFLAGS += -funroll-loops              # Loop unrolling
CFLAGS += -fomit-frame-pointer        # Frame pointer optimization
CFLAGS += -flto                       # Link-time optimization
```

#### 2. Profile-Guided Optimization (PGO)

```bash
# Step 1: Build with profiling
make clean
CFLAGS="-O3 -fprofile-generate" make

# Step 2: Run representative workload
./deepstream-multi-inference-app sample1.mp4 sample2.mp4 sample3.mp4 sample4.mp4

# Step 3: Rebuild with profile data
make clean
CFLAGS="-O3 -fprofile-use" make
```

### Memory Optimization

#### 1. Unified Memory Configuration

```c
// In configuration files, always use:
nvbuf-memory-type = 2    // Unified memory for zero-copy operations
```

#### 2. Buffer Pool Optimization

```yaml
# In YAML configuration
streammux:
  enable-memory-pool: 1
  pool-size: 8           # Adjust based on available memory
  max-memory-usage: 4294967296  # 4GB limit
```

#### 3. Memory Monitoring

```bash
# Monitor GPU memory usage
watch -n 1 nvidia-smi

# Monitor system memory
watch -n 1 'free -h && echo "=== GPU Memory ===" && nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader'
```

## Pipeline Optimization

### Source Configuration

#### 1. Hardware Decoding

```c
// Prioritize hardware decoding
uri_decode_bin = gst_element_factory_make("nvurisrcbin", "uri-decode-bin");
g_object_set(G_OBJECT(uri_decode_bin),
    "file-loop", TRUE,
    "cudadec-memtype", 0,        // Device memory
    "drop-on-latency", TRUE,     // Drop frames to maintain real-time
    "skip-frames", 0,            // Process every frame
    NULL);
```

#### 2. Source Synchronization

```yaml
# For live sources (cameras, RTSP)
streammux:
  live-source: 1
  sync-inputs: 0              # Disable for maximum throughput
  max-latency: 100000000      # 100ms maximum latency
  drop-pipeline-eos: 0
```

#### 3. Batch Formation Optimization

```yaml
streammux:
  batch-size: 4
  batched-push-timeout: 40000   # 40ms - balance latency vs throughput
  width: 1920
  height: 1080
  enable-padding: 0            # Disable padding for performance
```

### Inference Optimization

#### 1. TensorRT Engine Optimization

```bash
# Generate optimized TensorRT engine
/usr/src/tensorrt/bin/trtexec \
    --onnx=model.onnx \
    --batch=4 \
    --int8 \
    --workspace=2048 \
    --minShapes=input:1x3x640x640 \
    --optShapes=input:4x3x640x640 \
    --maxShapes=input:4x3x640x640 \
    --saveEngine=model_optimized_b4.engine \
    --verbose
```

#### 2. Inference Configuration

```ini
# In PGIE configuration file
[property]
batch-size=4
interval=0                    # Process every frame (0) or skip frames (1-5)
network-mode=1               # INT8 mode for maximum speed
cluster-mode=2               # DBSCAN clustering for efficiency
maintain-aspect-ratio=0      # Disable for performance
symmetric-padding=0          # Disable symmetric padding

# Tensor output optimization
output-tensor-meta=1
input-tensor-meta=0          # Disable if not needed

# GPU memory optimization
gpu-id=0
nvbuf-memory-type=2
```

#### 3. Multi-Stream Inference Optimization

```yaml
primary-gie:
  config-file-path: configs/multi_inference_pgie_config.txt
  batch-size: 4
  interval: 0                 # Critical: 0 = every frame, >0 = skip frames
  unique-id: 1
  
  # Performance settings
  process-mode: 1             # Primary mode
  model-color-format: 0       # RGB
  maintain-aspect-ratio: 0    # Disable for speed
```

### Display Optimization (When Enabled)

#### 1. Tiler Configuration

```yaml
tiler:
  rows: 2
  columns: 2
  width: 1280                # Lower resolution for performance
  height: 720
  gpu-id: 0
  nvbuf-memory-type: 2
  show-source-id: 1          # Enable source ID display
```

#### 2. OSD Configuration

```yaml
osd:
  process-mode: 1            # GPU mode
  display-text: 1
  display-clock: 0           # Disable clock for performance
  display-bbox: 1
  display-mask: 0            # Disable masks for performance
  gpu-id: 0
  nvbuf-memory-type: 2
```

#### 3. Sink Configuration

```yaml
sink:
  qos: 0                     # Disable QoS for maximum throughput
  sync: 0                    # Disable synchronization
  max-lateness: -1           # Accept all frames
  gpu-id: 0
  nvbuf-memory-type: 2
```

## Performance Monitoring

### Built-in Performance Measurement

```bash
# Enable built-in performance monitoring
./deepstream-multi-inference-app --perf video1.mp4 video2.mp4 video3.mp4 video4.mp4

# Sample output:
=== Performance Metrics ===
Total Batches: 150
Average FPS per source: 30.2
Total throughput: 120.8 FPS
Source 0 frames: 150
Source 1 frames: 149
Source 2 frames: 150
Source 3 frames: 151
===========================
```

### External Performance Monitoring

#### 1. GPU Monitoring

```bash
# Real-time GPU monitoring
nvidia-smi dmon -s pucvmet -d 1

# Detailed GPU utilization
nvidia-sml --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.total,memory.used,temperature.gpu --format=csv -l 1
```

#### 2. System Resource Monitoring

```bash
# CPU and memory monitoring
htop

# I/O monitoring
iotop

# Network monitoring (for RTSP sources)
iftop
```

#### 3. Advanced Profiling

```bash
# NVIDIA Nsight Systems profiling
nsys profile -t cuda,gstreamer -o profile_output ./deepstream-multi-inference-app video1.mp4 video2.mp4 video3.mp4 video4.mp4

# NVIDIA Nsight Compute (for kernel analysis)
ncu --set full -o kernel_profile ./deepstream-multi-inference-app video1.mp4 video2.mp4 video3.mp4 video4.mp4
```

## Troubleshooting Performance Issues

### Common Performance Problems

#### 1. Low GPU Utilization (<50%)

**Symptoms:**
- GPU utilization consistently below 50%
- High CPU usage
- Lower than expected FPS

**Solutions:**
```bash
# Check if hardware decoding is enabled
GST_DEBUG=4 ./deepstream-multi-inference-app video1.mp4 ... 2>&1 | grep -i "nvdec\|hw"

# Verify TensorRT engine is being used
ls -la configs/ | grep .engine

# Check memory type configuration
grep -r "nvbuf-memory-type" configs/
```

#### 2. High Latency (>200ms)

**Symptoms:**
- End-to-end latency above target
- Delayed tensor output
- Buffer accumulation

**Solutions:**
```yaml
# Reduce batch timeout
streammux:
  batched-push-timeout: 20000  # Reduce from 40000 to 20000

# Enable frame dropping for live sources  
streammux:
  drop-pipeline-eos: 1
  max-latency: 50000000       # 50ms max latency
```

#### 3. Memory Issues

**Symptoms:**
- GPU out of memory errors
- System memory exhaustion
- Application crashes

**Solutions:**
```bash
# Reduce batch size temporarily
sed -i 's/batch-size=4/batch-size=2/' configs/multi_inference_pgie_config.txt

# Monitor memory usage
watch -n 1 'nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader'

# Check for memory leaks
valgrind --tool=memcheck --leak-check=full ./deepstream-multi-inference-app --help
```

#### 4. Frame Drops

**Symptoms:**
- Inconsistent frame counts between sources
- Warning messages about dropped frames
- Reduced effective FPS

**Solutions:**
```c
// Enable frame drop debugging
GST_DEBUG=3 ./deepstream-multi-inference-app ... 2>&1 | grep -i "drop"

// Adjust source configuration
g_object_set(G_OBJECT(uri_decode_bin),
    "drop-on-latency", FALSE,    // Disable frame dropping
    "max-size-time", 0,          // Unlimited queue time
    NULL);
```

## Optimization Checklist

### Pre-deployment Checklist

- [ ] **Hardware Configuration**
  - [ ] GPU set to maximum performance mode
  - [ ] CPU governor set to performance
  - [ ] Adequate cooling for sustained performance
  - [ ] Sufficient GPU memory (8GB+ recommended)

- [ ] **Software Configuration**
  - [ ] Latest NVIDIA drivers installed
  - [ ] DeepStream SDK properly configured
  - [ ] TensorRT engines pre-built for batch size 4
  - [ ] Unified memory enabled throughout pipeline

- [ ] **Application Configuration**
  - [ ] Batch size set to 4 in all configuration files
  - [ ] Hardware decoding enabled (nvurisrcbin)
  - [ ] INT8 precision enabled for inference
  - [ ] Optimal timeout settings configured

- [ ] **Performance Validation**
  - [ ] Target FPS achieved (30+ per source)
  - [ ] GPU utilization >80%
  - [ ] Memory usage within limits
  - [ ] Latency under target (<100ms)

### Deployment Optimization

```bash
#!/bin/bash
# Performance optimization script

# Set system for maximum performance
echo "Configuring system for maximum performance..."

# GPU settings
sudo nvidia-smi -pm 1
sudo nvidia-smi -ac 10501,2100

# CPU settings
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Memory settings
echo 'vm.swappiness=10' >> /etc/sysctl.conf
sudo sysctl -p

# Run application with optimized settings
export CUDA_CACHE_DISABLE=0
export CUDA_CACHE_PATH=/tmp/cuda_cache
export GST_PLUGIN_PATH=/opt/nvidia/deepstream/deepstream/lib/gst-plugins/

./deepstream-multi-inference-app --perf "$@"
```

## Advanced Optimization Techniques

### 1. Dynamic Batch Size Adjustment

For variable load scenarios:

```c
// Pseudo-code for adaptive batching
if (current_fps < target_fps) {
    reduce_batch_timeout();
} else if (gpu_utilization < 80%) {
    increase_processing_complexity();
}
```

### 2. Multi-GPU Scaling

For processing more than 4 sources:

```bash
# GPU 0: Sources 0-3
CUDA_VISIBLE_DEVICES=0 ./deepstream-multi-inference-app src0 src1 src2 src3 &

# GPU 1: Sources 4-7
CUDA_VISIBLE_DEVICES=1 ./deepstream-multi-inference-app src4 src5 src6 src7 &
```

### 3. Custom Memory Allocators

```c
// Custom memory allocator for optimal performance
static void*
custom_allocator(size_t size, size_t alignment)
{
    void* ptr;
    if (cudaMallocManaged(&ptr, size) != cudaSuccess) {
        return NULL;
    }
    return ptr;
}
```

This performance guide provides comprehensive optimization strategies for achieving maximum performance with the DeepStream Multi-Source Batched Inference application.