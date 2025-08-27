# DeepStream Multi-Source Batched Inference Pipeline - Implementation Plan

## Project Overview
Create an optimized DeepStream pipeline that processes multiple video sources simultaneously with batched inference, outputting tensor data for each source.

## Architecture Design

```
┌─────────────────────────────────────────────────────────────────┐
│                     DeepStream Pipeline                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  [Source 1] ──┐                                                 │
│  [Source 2] ──┼──> [StreamMux] ──> [PGIE] ──> [Tensor Extract] │
│  [Source 3] ──┤    (Batch=(Multiple))     (Infer)      (X Outputs)     │
│  [Source X] ──┘                                                 │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Phase 1: Project Setup and Infrastructure

### 1.1 Environment Setup
```bash
# Required dependencies
- DeepStream SDK 6.3 or later
- CUDA 11.8+
- TensorRT 8.5+
- GStreamer 1.16+
- Python bindings (pyds) for Python implementation
```

### 1.2 Directory Structure
```
deepstream_multi_source/
├── src/
│   ├── cpp/
│   │   ├── main.cpp
│   │   ├── pipeline_builder.cpp
│   │   ├── pipeline_builder.h
│   │   ├── tensor_processor.cpp
│   │   └── tensor_processor.h
│   └── python/
│       ├── main.py
│       ├── pipeline_builder.py
│       └── tensor_processor.py
├── config/
│   ├── model_config.txt
│   └── pipeline_config.yaml
├── models/
│   ├── model.onnx
│   └── model.engine (generated)
├── test_data/
│   └── videos/
├── output/
├── CMakeLists.txt
├── Makefile
└── README.md
```

### 1.3 Configuration Files to Create
- **model_config.txt**: PGIE configuration for inference
- **pipeline_config.yaml**: Pipeline parameters (sources, batch size, dimensions)
- **optimization_config.yaml**: Performance tuning parameters

## Phase 2: Core Components Implementation

### 2.1 Pipeline Builder Component
**File: `pipeline_builder.py/cpp`**

Key responsibilities:
- Initialize GStreamer pipeline
- Create and configure source bins for multiple inputs.
- Set up StreamMux with optimal batching
- Configure PGIE for inference
- Implement probe functions for tensor extraction

**Critical Functions:**
```python
class PipelineBuilder:
    def __init__(self, config):
        # Initialize with configuration
        
    def create_source_bin(self, source_id: int, uri: str):
        # Create source bin with hardware decoding
        
    def setup_streammux(self, batch_size: int):
        # Configure batching parameters
        
    def setup_inference(self, model_config: str):
        # Configure PGIE with TensorRT
        
    def add_probe_for_tensor_extraction(self):
        # Attach probe to PGIE src pad
        
    def build(self):
        # Assemble complete pipeline
```

### 2.2 Tensor Processor Component
**File: `tensor_processor.py/cpp`**

Key responsibilities:
- Extract tensor metadata from GStreamer buffer
- Organize tensors by source ID
- Provide interface for tensor access
- Implement batch processing logic

**Critical Functions:**
```python
class TensorProcessor:
    def __init__(self):
        # Initialize tensor storage
        
    def extract_tensor_meta(self, buffer):
        # Extract NvDsInferTensorMeta from buffer
        
    def process_batch(self, batch_meta):
        # Process all multiple sources in batch
        
    def get_tensor_by_source(self, source_id: int):
        # Return tensor for specific source
        
    def export_tensors(self, format='numpy'):
        # Export tensors in desired format
```

### 2.3 Optimization Manager Component
**File: `optimization_manager.py/cpp`**

Key responsibilities:
- Configure GPU/CUDA settings
- Manage buffer pools
- Set up zero-copy operations
- Monitor performance metrics

**Critical Functions:**
```python
class OptimizationManager:
    def configure_gpu(self, gpu_id: int):
        # Set CUDA device and memory type
        
    def setup_unified_memory(self):
        # Configure unified memory for zero-copy
        
    def optimize_decoder(self, decoder_element):
        # Set hardware decoder parameters
        
    def monitor_performance(self):
        # Track FPS, latency, GPU usage
```

## Phase 3: Implementation Steps

### Step 1: Basic Pipeline Setup
```python
# Pseudocode structure
def create_basic_pipeline():
    # 1. Create pipeline element
    # 2. Create multiple source bins
    # 3. Create streammux
    # 4. Link sources to streammux
    # 5. Add fakesink for testing
    # 6. Verify pipeline runs
```

### Step 2: Add Inference Component
```python
def add_inference():
    # 1. Create PGIE element
    # 2. Configure with model config
    # 3. Set batch-size = depend on input sources
    # 4. Link to streammux
    # 5. Test inference execution
```

### Step 3: Implement Tensor Extraction
```python
def add_tensor_extraction():
    # 1. Add probe to PGIE src pad
    # 2. Extract NvDsInferTensorMeta
    # 3. Parse tensor dimensions
    # 4. Copy tensor data
    # 5. Organize by source_id
```

### Step 4: Optimize Performance
```python
def optimize_pipeline():
    # 1. Enable hardware decoding (NVDEC)
    # 2. Configure unified memory
    # 3. Set up buffer pools
    # 4. Enable async processing
    # 5. Tune batch timeout
```

## Phase 4: Critical Optimizations

### 4.1 Memory Optimizations
```yaml
memory_config:
  nvbuf_memory_type: NVBUF_MEM_CUDA_UNIFIED  # Type 3
  num_surfaces_per_frame: 1
  gpu_id: 0
  buffer_pool_size: depend on input sources
```

### 4.2 Decoder Optimizations
```yaml
decoder_config:
  enable_max_performance: true
  drop_frame_interval: 0
  num_extra_surfaces: 0
  cudadec_memtype: 2  # Unified memory
  skip_frames: 0
```

### 4.3 StreamMux Optimizations
```yaml
streammux_config:
  batch_size: depend on input sources
  batched_push_timeout: 40000  # microseconds
  width: 1920
  height: 1080
  enable_padding: 0
  live_source: 0
  attach_sys_ts: 1
```

### 4.4 Inference Optimizations
```yaml
inference_config:
  gpu_id: 0
  batch_size: depend on input sources
  model_engine_file: "model.engine"  # Pre-built TensorRT engine
  input_tensor_meta: true
  output_tensor_meta: true
  interval: 0  # Process every frame
  cluster_mode: 2  # DBSCAN
```

## Phase 5: Testing Strategy

### 5.1 Unit Tests
```python
# Test individual components
- test_source_creation()
- test_streammux_batching()
- test_tensor_extraction()
- test_memory_allocation()
```

### 5.2 Integration Tests
```python
# Test pipeline flow
- test_pipeline_construction()
- test_multiple_source_synchronization()
- test_batch_processing()
- test_tensor_output_validity()
```

### 5.3 Performance Tests
```python
# Measure optimization impact
- measure_fps_per_source()
- measure_end_to_end_latency()
- measure_gpu_utilization()
- measure_memory_usage()
```

## Phase 6: Advanced Features (Optional)

### 6.1 Dynamic Source Management
- Add/remove sources at runtime
- Handle source failures gracefully
- Implement source reconnection

### 6.2 Multi-Model Support
- Sequential inference pipeline
- Parallel model execution
- Model switching at runtime

### 6.3 Output Options
- Tensor serialization (protobuf, msgpack)
- Network streaming (RTSP, WebRTC)
- File storage with metadata
- Real-time visualization

## Implementation Checklist

### Core Requirements
- [ ] multiple video source inputs
- [ ] Batch size = multiple processing
- [ ] Model inference execution
- [ ] Multiple tensor outputs extraction
- [ ] Maximum optimization applied

### Optimizations
- [ ] Hardware video decoding (NVDEC)
- [ ] TensorRT inference engine
- [ ] Unified/Zero-copy memory
- [ ] Asynchronous processing
- [ ] Buffer pool management
- [ ] GPU Direct transfers

### Code Quality
- [ ] Error handling for all components
- [ ] Resource cleanup on shutdown
- [ ] Configuration validation
- [ ] Logging and debugging
- [ ] Performance monitoring

### Documentation
- [ ] API documentation
- [ ] Configuration guide
- [ ] Performance tuning guide
- [ ] Troubleshooting guide

## Key Performance Metrics to Track

```python
metrics = {
    "fps_per_source": [],      # Target: 30+ fps
    "batch_latency_ms": 0,     # Target: <50ms
    "gpu_utilization_%": 0,    # Target: 70-90%
    "memory_usage_mb": 0,       # Monitor for leaks
    "tensor_extraction_ms": 0, # Target: <5ms
    "dropped_frames": 0         # Target: 0
}
```

## Common Issues and Solutions

### Issue 1: Sources not synchronized
**Solution**: Adjust batched-push-timeout and ensure live-source=0

### Issue 2: High latency
**Solution**: Enable async processing, reduce interval, use pre-built TensorRT engine

### Issue 3: Memory leaks
**Solution**: Properly unref GStreamer objects, use buffer pools, cleanup tensor data

### Issue 4: Low FPS
**Solution**: Enable hardware decoding, use unified memory, optimize model

## Next Steps for Claude Code Implementation

1. **Start with Phase 1**: Set up project structure
2. **Implement Phase 2.1**: Create basic pipeline builder
3. **Test with Phase 3, Step 1**: Verify basic pipeline works
4. **Incrementally add**: Inference, tensor extraction, optimizations
5. **Measure and tune**: Use performance metrics to guide optimization

## Example Command to Run

```bash
# Python version
python3 src/python/main.py \
    --sources video1.mp4 video2.mp4 video3.mp4 video4.mp4 \
    --model models/model.onnx \
    --batch-size 4 \
    --output-tensors

# C++ version
./build/deepstream_multi_source \
    --config config/pipeline_config.yaml \
    --model-config config/model_config.txt \
    --enable-perf-measurement
```

## Resources and References

- DeepStream SDK Documentation
- NVIDIA TensorRT Optimization Guide
- GStreamer Pipeline Optimization
- CUDA Unified Memory Programming
- NvBufSurface API Reference
