#pragma once

#include <string>
#include <vector>
#include <memory>

namespace VideoSummary {

/**
 * @brief Pipeline-specific configuration structures
 * 
 * These structures define configuration parameters for the DeepStream pipeline
 * components, separate from the general ConfigManager structures for modularity.
 */

enum class SourceType {
    FILE_H264 = 1,      // H.264 elementary stream
    FILE_MP4 = 2,       // MP4 container
    URI_RTSP = 3,       // RTSP stream
    USB_CAMERA = 4,     // USB camera (V4L2)
    CSI_CAMERA = 5,     // CSI camera (Jetson)
    URI_HTTP = 6        // HTTP stream
};

enum class PipelineState {
    UNINITIALIZED,      // Pipeline not created
    NULL_STATE,         // GStreamer NULL state
    READY,              // GStreamer READY state  
    PAUSED,             // GStreamer PAUSED state
    PLAYING,            // GStreamer PLAYING state
    ERROR_STATE         // Error occurred
};

/**
 * @brief Source configuration for video inputs
 */
struct SourceConfiguration {
    SourceType type = SourceType::FILE_H264;
    std::string uri;                    // File path or stream URI
    int gpu_id = 0;                     // GPU device ID
    bool drop_frame_interval = false;   // Drop frames for performance
    int framerate_num = 30;             // Numerator for framerate
    int framerate_den = 1;              // Denominator for framerate
    
    // File-specific options
    bool loop = false;                  // Loop file playback
    
    // Stream-specific options
    int latency = 200;                  // Latency in ms for streams
    int timeout = 5000;                 // Connection timeout in ms
    
    bool isValid() const {
        return !uri.empty() && gpu_id >= 0;
    }
};

/**
 * @brief Stream multiplexer configuration
 */
struct StreamMuxConfig {
    int batch_size = 1;                 // Number of streams to batch
    int width = 1920;                   // Output frame width
    int height = 1080;                  // Output frame height
    int gpu_id = 0;                     // GPU device ID
    int buffer_pool_size = 4;           // Number of buffers in pool
    int batched_push_timeout = 40000;   // Timeout in microseconds
    bool enable_padding = false;        // Pad frames to maintain aspect ratio
    int nvbuf_memory_type = 0;          // NVMM memory type
    
    bool isValid() const {
        return batch_size > 0 && batch_size <= 32 && 
               width > 0 && height > 0 && gpu_id >= 0;
    }
};

/**
 * @brief Sink configuration for output
 */
struct SinkConfig {
    enum Type {
        FAKE_SINK,      // Discard output (for testing)
        FILE_SINK,      // Write to file
        DISPLAY_SINK,   // Display on screen
        RTMP_SINK       // Stream to RTMP
    };
    
    Type type = FAKE_SINK;
    std::string location;               // Output file path (if FILE_SINK)
    bool sync = false;                  // Synchronize with clock
    bool async = false;                 // Asynchronous operation
    int max_lateness = -1;              // Maximum lateness in ns (-1 = unlimited)
    
    bool isValid() const {
        if (type == FILE_SINK) {
            return !location.empty();
        }
        return true;
    }
};

/**
 * @brief Complete pipeline configuration
 */
struct PipelineConfiguration {
    std::string name = "video-summary-pipeline";
    
    // Component configurations
    SourceConfiguration source;
    StreamMuxConfig streammux;
    SinkConfig sink;
    
    // Performance settings
    bool enable_perf_measurement = true;
    int perf_measurement_interval_sec = 5;
    
    // Debug settings
    bool enable_debug = false;
    std::string debug_dump_dir = "./debug_dumps";
    
    // Timeout settings
    int pipeline_ready_timeout_sec = 10;
    int pipeline_eos_timeout_sec = 30;
    
    bool isValid() const {
        return source.isValid() && 
               streammux.isValid() && 
               sink.isValid() &&
               !name.empty();
    }
    
    void setDefaults() {
        if (name.empty()) {
            name = "video-summary-pipeline";
        }
        
        // Set reasonable defaults for source
        if (source.uri.empty()) {
            source.uri = "/opt/nvidia/deepstream/deepstream-7.1/samples/streams/sample_720p.h264";
            source.type = SourceType::FILE_H264;
        }
    }
};

/**
 * @brief Pipeline statistics and performance metrics
 */
struct PipelineStats {
    // Frame statistics
    uint64_t frames_processed = 0;
    uint64_t frames_dropped = 0;
    double current_fps = 0.0;
    double average_fps = 0.0;
    
    // Timing statistics  
    double pipeline_latency_ms = 0.0;
    double processing_time_ms = 0.0;
    
    // Memory statistics
    size_t memory_usage_mb = 0;
    size_t gpu_memory_usage_mb = 0;
    
    // Error statistics
    uint32_t error_count = 0;
    uint32_t warning_count = 0;
    
    void reset() {
        frames_processed = 0;
        frames_dropped = 0;
        current_fps = 0.0;
        average_fps = 0.0;
        pipeline_latency_ms = 0.0;
        processing_time_ms = 0.0;
        memory_usage_mb = 0;
        gpu_memory_usage_mb = 0;
        error_count = 0;
        warning_count = 0;
    }
};

} // namespace VideoSummary