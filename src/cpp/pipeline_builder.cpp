#include "pipeline_builder.h"
#include "async_processor.h"
#include <iostream>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <iomanip>

PipelineBuilder::PipelineBuilder() 
    : pipeline(nullptr), streammux(nullptr), pgie(nullptr), nvvidconv(nullptr),
      nvosd(nullptr), tiler(nullptr), sink(nullptr), tee(nullptr),
      queue1(nullptr), queue2(nullptr), bus(nullptr), bus_watch_id(0),
      tensor_callback(nullptr), callback_user_data(nullptr), async_processor(nullptr) {
    // Initialize GStreamer
    gst_init(nullptr, nullptr);
}

PipelineBuilder::~PipelineBuilder() {
    cleanup();
}

bool PipelineBuilder::initialize(const PipelineConfig& pipeline_config) {
    config = pipeline_config;
    
    // Validate configuration
    if (config.sources.empty()) {
        std::cerr << "Error: No sources configured!" << std::endl;
        return false;
    }
    
    if (config.batch_size <= 0 || config.batch_size > 128) {
        std::cerr << "Error: Invalid batch size: " << config.batch_size << std::endl;
        return false;
    }
    
    // Update batch_size to match number of sources if not explicitly set
    if (config.batch_size != static_cast<int>(config.sources.size())) {
        std::cout << "Adjusting batch_size from " << config.batch_size 
                  << " to " << config.sources.size() << " to match source count" << std::endl;
        config.batch_size = static_cast<int>(config.sources.size());
    }
    
    std::cout << "Initialized PipelineBuilder with " << config.sources.size() 
              << " sources, batch_size=" << config.batch_size << std::endl;
    
    return true;
}

bool PipelineBuilder::create_pipeline() {
    // Create pipeline
    pipeline = gst_pipeline_new("deepstream-multi-source-pipeline");
    if (!check_element_creation(pipeline, "pipeline")) {
        return false;
    }
    
    // Create all pipeline components
    if (!setup_streammux() ||
        !setup_inference() ||
        !setup_tensor_extraction()) {
        return false;
    }
    
    // Create source bins for all configured sources
    for (const auto& source : config.sources) {
        if (!create_source_bin(source)) {
            std::cerr << "Error: Failed to create source bin for: " << source.uri << std::endl;
            return false;
        }
    }
    
    // Setup display branch if enabled
    if (config.enable_display) {
        if (!setup_display_branch()) {
            std::cerr << "Warning: Failed to setup display branch" << std::endl;
        }
    }
    
    // Link all components
    if (!link_pipeline_components()) {
        return false;
    }
    
    // Setup bus message handling
    bus = gst_element_get_bus(pipeline);
    bus_watch_id = gst_bus_add_watch(bus, bus_call, this);
    gst_object_unref(bus);
    
    print_pipeline_info();
    return true;
}

bool PipelineBuilder::create_source_bin(const SourceConfig& source_config) {
    std::stringstream bin_name;
    bin_name << "source-bin-" << source_config.source_id;
    
    GstElement* source_bin = gst_bin_new(bin_name.str().c_str());
    if (!check_element_creation(source_bin, bin_name.str())) {
        return false;
    }
    
    // Create source elements
    GstElement* source = nullptr;
    GstElement* h264parser = nullptr;
    GstElement* decoder = nullptr;
    GstElement* nvvidconv_src = nullptr;
    GstElement* capsfilter = nullptr;
    
    // Determine source type and create appropriate element
    if (source_config.uri.find("rtsp://") == 0) {
        source = gst_element_factory_make("rtspsrc", nullptr);
        g_object_set(G_OBJECT(source), "location", source_config.uri.c_str(), 
                     "latency", 2000, nullptr);
    } else if (source_config.uri.find("/dev/video") == 0) {
        source = gst_element_factory_make("v4l2src", nullptr);
        g_object_set(G_OBJECT(source), "device", source_config.uri.c_str(), nullptr);
    } else {
        source = gst_element_factory_make("filesrc", nullptr);
        g_object_set(G_OBJECT(source), "location", source_config.uri.c_str(), nullptr);
    }
    
    if (!check_element_creation(source, "source")) {
        gst_object_unref(source_bin);
        return false;
    }
    
    // Create decoder chain
    GstElement* final_element = nullptr;
    if (source_config.uri.find("rtsp://") != 0 && source_config.uri.find("/dev/video") != 0) {
        GstElement* qtdemux = gst_element_factory_make("qtdemux", nullptr);
        h264parser = gst_element_factory_make("h264parse", nullptr);
        decoder = gst_element_factory_make("nvv4l2decoder", nullptr);
        GstElement* queue_after_decoder = gst_element_factory_make("queue", nullptr);
        
        if (!check_element_creation(qtdemux, "qtdemux") ||
            !check_element_creation(h264parser, "h264parser") ||
            !check_element_creation(decoder, "nvv4l2decoder") ||
            !check_element_creation(queue_after_decoder, "queue")) {
            gst_object_unref(source_bin);
            return false;
        }
        
        // Configure decoder for optimal performance
        // Note: Some properties may not be available on all decoder elements
        g_object_set(G_OBJECT(decoder), 
                     "drop-frame-interval", 0,
                     "num-extra-surfaces", 0,
                     nullptr);
                     
        // Configure queue to help with timing
        g_object_set(G_OBJECT(queue_after_decoder),
                     "max-size-buffers", 2,
                     "max-size-bytes", 0,
                     "max-size-time", 0,
                     nullptr);
        
        gst_bin_add_many(GST_BIN(source_bin), source, qtdemux, h264parser, decoder, queue_after_decoder, nullptr);
        
        // Link source elements
        if (!gst_element_link(source, qtdemux)) {
            std::cerr << "Error: Failed to link source to qtdemux" << std::endl;
            gst_object_unref(source_bin);
            return false;
        }
        
        // Connect qtdemux pad dynamically
        g_signal_connect(qtdemux, "pad-added", 
                        G_CALLBACK(qtdemux_pad_added_callback), h264parser);
        
        if (!gst_element_link_many(h264parser, decoder, queue_after_decoder, nullptr)) {
            std::cerr << "Error: Failed to link parser to decoder to queue" << std::endl;
            gst_object_unref(source_bin);
            return false;
        }
        
        final_element = queue_after_decoder;
    } else {
        gst_bin_add(GST_BIN(source_bin), source);
        final_element = source; // For RTSP and V4L2, source acts as decoder
    }
    
    // Create converter and capsfilter
    nvvidconv_src = gst_element_factory_make("nvvideoconvert", nullptr);
    capsfilter = gst_element_factory_make("capsfilter", nullptr);
    
    if (!check_element_creation(nvvidconv_src, "nvvideoconvert") ||
        !check_element_creation(capsfilter, "capsfilter")) {
        gst_object_unref(source_bin);
        return false;
    }
    
    // Configure caps for streammux compatibility
    std::stringstream caps_str;
    caps_str << "video/x-raw(memory:NVMM), format=NV12";
    GstCaps* caps = gst_caps_from_string(caps_str.str().c_str());
    g_object_set(G_OBJECT(capsfilter), "caps", caps, nullptr);
    gst_caps_unref(caps);
    
    // Configure nvvidconv for optimal performance  
    g_object_set(G_OBJECT(nvvidconv_src), 
                 "gpu-id", config.gpu_id,
                 nullptr);
    
    gst_bin_add_many(GST_BIN(source_bin), nvvidconv_src, capsfilter, nullptr);
    
    // Link converter chain
    if (!gst_element_link_many(final_element, nvvidconv_src, capsfilter, nullptr)) {
        std::cerr << "Error: Failed to link converter chain" << std::endl;
        gst_object_unref(source_bin);
        return false;
    }
    
    // Create ghost pad
    GstPad* src_pad = gst_element_get_static_pad(capsfilter, "src");
    GstPad* ghost_pad = gst_ghost_pad_new("src", src_pad);
    gst_element_add_pad(source_bin, ghost_pad);
    gst_object_unref(src_pad);
    
    // Add to pipeline and connect to streammux
    gst_bin_add(GST_BIN(pipeline), source_bin);
    source_bins.push_back(source_bin);
    
    // Get streammux sink pad and link
    std::stringstream pad_name;
    pad_name << "sink_" << source_config.source_id;
    GstPad* sinkpad = gst_element_request_pad_simple(streammux, pad_name.str().c_str());
    GstPad* srcpad = gst_element_get_static_pad(source_bin, "src");
    
    if (gst_pad_link(srcpad, sinkpad) != GST_PAD_LINK_OK) {
        std::cerr << "Error: Failed to link source bin to streammux" << std::endl;
        gst_object_unref(srcpad);
        gst_object_unref(sinkpad);
        return false;
    }
    
    gst_object_unref(srcpad);
    gst_object_unref(sinkpad);
    
    std::cout << "Created source bin " << source_config.source_id 
              << " for: " << source_config.uri << std::endl;
    
    return true;
}

bool PipelineBuilder::setup_streammux() {
    streammux = gst_element_factory_make("nvstreammux", "stream-muxer");
    if (!check_element_creation(streammux, "nvstreammux")) {
        return false;
    }
    
    // Configure streammux properties for optimal performance
    g_object_set(G_OBJECT(streammux),
                 "batch-size", config.batch_size,
                 "width", config.width,
                 "height", config.height,
                 "batched-push-timeout", config.batched_push_timeout,
                 "live-source", FALSE,
                 "enable-padding", FALSE,
                 "nvbuf-memory-type", config.nvbuf_memory_type,
                 "gpu-id", config.gpu_id,
                 "attach-sys-ts", FALSE,  // Don't attach system timestamps for files
                 "sync-inputs", TRUE,     // Synchronize inputs for proper playback
                 "max-latency", 40000000, // 40ms max latency in nanoseconds
                 nullptr);
    
    gst_bin_add(GST_BIN(pipeline), streammux);
    
    std::cout << "Setup nvstreammux: batch-size=" << config.batch_size
              << ", resolution=" << config.width << "x" << config.height << std::endl;
    
    return true;
}

bool PipelineBuilder::setup_inference() {
    pgie = create_nvinfer_element();
    if (!pgie) {
        return false;
    }
    
    gst_bin_add(GST_BIN(pipeline), pgie);
    return true;
}

GstElement* PipelineBuilder::create_nvinfer_element() {
    GstElement* nvinfer = gst_element_factory_make("nvinfer", "primary-infer");
    if (!check_element_creation(nvinfer, "nvinfer")) {
        return nullptr;
    }
    
    // Configure inference properties
    g_object_set(G_OBJECT(nvinfer),
                 "config-file-path", config.model_config_path.c_str(),
                 "batch-size", config.batch_size,
                 "unique-id", 1,
                 "gpu-id", config.gpu_id,
                 "output-tensor-meta", TRUE,
                 "interval", 0, // Process every frame
                 nullptr);
    
    if (!config.model_engine_path.empty()) {
        g_object_set(G_OBJECT(nvinfer),
                     "model-engine-file", config.model_engine_path.c_str(),
                     nullptr);
    }
    
    std::cout << "Setup nvinfer: config=" << config.model_config_path
              << ", batch-size=" << config.batch_size << std::endl;
    
    return nvinfer;
}

bool PipelineBuilder::setup_tensor_extraction() {
    // Create tee for branching after inference
    tee = gst_element_factory_make("tee", "inference-tee");
    queue1 = gst_element_factory_make("queue", "tensor-queue");
    
    // CRITICAL FIX: Add fakesink to complete tensor extraction branch
    GstElement* fakesink_tensor = gst_element_factory_make("fakesink", "fakesink-tensor");
    
    if (!check_element_creation(tee, "tee") ||
        !check_element_creation(queue1, "queue") ||
        !check_element_creation(fakesink_tensor, "fakesink-tensor")) {
        return false;
    }
    
    // Configure queue properties
    g_object_set(G_OBJECT(queue1),
                 "max-size-buffers", 1000,
                 "max-size-bytes", 0,
                 "max-size-time", 0,
                 nullptr);
                 
    // Configure fakesink properties
    g_object_set(G_OBJECT(fakesink_tensor),
                 "sync", FALSE,
                 "async", FALSE,
                 nullptr);
    
    gst_bin_add_many(GST_BIN(pipeline), tee, queue1, fakesink_tensor, nullptr);
    
    // Link tensor extraction branch: queue1 -> fakesink
    if (!gst_element_link(queue1, fakesink_tensor)) {
        std::cerr << "Error: Failed to link tensor queue to fakesink" << std::endl;
        return false;
    }
    
    // CRITICAL FIX: Add probe to PGIE src pad for tensor extraction (like C version)
    // This ensures we get raw tensor metadata directly from inference engine
    GstPad* tensor_probe_pad = gst_element_get_static_pad(pgie, "src");
    if (tensor_probe_pad) {
        gst_pad_add_probe(tensor_probe_pad, GST_PAD_PROBE_TYPE_BUFFER,
                         tensor_extract_probe, this, nullptr);
        gst_object_unref(tensor_probe_pad);
        std::cout << "Setup tensor extraction probe on PGIE src pad" << std::endl;
    } else {
        std::cerr << "Error: Failed to get PGIE src pad for tensor extraction" << std::endl;
        return false;
    }
    
    std::cout << "Setup tensor extraction infrastructure complete" << std::endl;
    return true;
}

bool PipelineBuilder::setup_display_branch() {
    if (!config.enable_display) {
        return true;
    }
    
    // Create display elements
    queue2 = gst_element_factory_make("queue", "display-queue");
    tiler = gst_element_factory_make("nvmultistreamtiler", "nvtiler");
    nvvidconv = gst_element_factory_make("nvvideoconvert", "nvvideo-converter");
    nvosd = gst_element_factory_make("nvdsosd", "nv-onscreendisplay");
    sink = gst_element_factory_make("nveglglessink", "nvvideo-renderer");
    
    if (!check_element_creation(queue2, "display-queue") ||
        !check_element_creation(tiler, "nvmultistreamtiler") ||
        !check_element_creation(nvvidconv, "nvvideoconvert") ||
        !check_element_creation(nvosd, "nvdsosd") ||
        !check_element_creation(sink, "nveglglessink")) {
        return false;
    }
    
    // Configure tiler for multi-source display
    int rows = static_cast<int>(std::ceil(std::sqrt(config.sources.size())));
    int cols = static_cast<int>(std::ceil(static_cast<double>(config.sources.size()) / rows));
    
    g_object_set(G_OBJECT(tiler),
                 "rows", rows,
                 "columns", cols,
                 "width", 1920,
                 "height", 1080,
                 "gpu-id", config.gpu_id,
                 "nvbuf-memory-type", config.nvbuf_memory_type,
                 nullptr);
    
    // Configure other display elements
    g_object_set(G_OBJECT(nvvidconv),
                 "gpu-id", config.gpu_id,
                 nullptr);
    
    g_object_set(G_OBJECT(nvosd),
                 "gpu-id", config.gpu_id,
                 nullptr);
    
    g_object_set(G_OBJECT(sink),
                 "sync", TRUE,   // Enable sync for proper timing
                 "async", FALSE, // Disable async for better sync
                 "qos", TRUE,    // Enable quality of service for frame dropping if needed
                 nullptr);
    
    gst_bin_add_many(GST_BIN(pipeline), queue2, tiler, nvvidconv, nvosd, sink, nullptr);
    
    std::cout << "Setup display branch: " << rows << "x" << cols 
              << " grid for " << config.sources.size() << " sources" << std::endl;
    
    return true;
}

bool PipelineBuilder::link_pipeline_components() {
    // Link basic pipeline: streammux -> pgie -> tee
    if (!gst_element_link_many(streammux, pgie, tee, nullptr)) {
        std::cerr << "Error: Failed to link basic pipeline components" << std::endl;
        return false;
    }
    
    // Link tensor extraction branch: tee -> queue1
    if (!gst_element_link(tee, queue1)) {
        std::cerr << "Error: Failed to link tensor extraction branch" << std::endl;
        return false;
    }
    
    // Link display branch if enabled
    if (config.enable_display && queue2 && tiler && nvvidconv && nvosd && sink) {
        if (!gst_element_link_many(tee, queue2, tiler, nvvidconv, nvosd, sink, nullptr)) {
            std::cerr << "Error: Failed to link display branch" << std::endl;
            return false;
        }
    }
    
    std::cout << "Successfully linked all pipeline components" << std::endl;
    return true;
}

bool PipelineBuilder::start_pipeline() {
    if (!pipeline) {
        std::cerr << "Error: Pipeline not created" << std::endl;
        return false;
    }
    
    std::cout << "Starting pipeline..." << std::endl;
    GstStateChangeReturn ret = gst_element_set_state(pipeline, GST_STATE_PLAYING);
    
    if (ret == GST_STATE_CHANGE_FAILURE) {
        std::cerr << "Error: Failed to start pipeline" << std::endl;
        return false;
    }
    
    std::cout << "Pipeline started successfully" << std::endl;
    return true;
}

bool PipelineBuilder::stop_pipeline() {
    if (!pipeline) {
        return true;
    }
    
    std::cout << "Stopping pipeline..." << std::endl;
    gst_element_set_state(pipeline, GST_STATE_NULL);
    std::cout << "Pipeline stopped" << std::endl;
    return true;
}

bool PipelineBuilder::pause_pipeline() {
    if (!pipeline) {
        return false;
    }
    
    GstStateChangeReturn ret = gst_element_set_state(pipeline, GST_STATE_PAUSED);
    return ret != GST_STATE_CHANGE_FAILURE;
}

GstStateChangeReturn PipelineBuilder::get_pipeline_state() {
    if (!pipeline) {
        return GST_STATE_CHANGE_FAILURE;
    }
    
    GstState state;
    return gst_element_get_state(pipeline, &state, nullptr, GST_CLOCK_TIME_NONE);
}

void PipelineBuilder::set_tensor_extraction_callback(TensorExtractCallback callback, gpointer user_data) {
    tensor_callback = callback;
    callback_user_data = user_data;
}

gboolean PipelineBuilder::bus_call(GstBus *bus, GstMessage *msg, gpointer data) {
    PipelineBuilder* builder = static_cast<PipelineBuilder*>(data);
    
    switch (GST_MESSAGE_TYPE(msg)) {
        case GST_MESSAGE_EOS:
            std::cout << "End of stream" << std::endl;
            g_main_loop_quit(g_main_loop_new(nullptr, FALSE));
            break;
            
        case GST_MESSAGE_ERROR: {
            gchar *debug;
            GError *error;
            gst_message_parse_error(msg, &error, &debug);
            std::cerr << "Pipeline Error: " << error->message << std::endl;
            if (debug) {
                std::cerr << "Debug: " << debug << std::endl;
                g_free(debug);
            }
            g_error_free(error);
            g_main_loop_quit(g_main_loop_new(nullptr, FALSE));
            break;
        }
        
        case GST_MESSAGE_WARNING: {
            gchar *debug;
            GError *error;
            gst_message_parse_warning(msg, &error, &debug);
            std::cout << "Pipeline Warning: " << error->message << std::endl;
            if (debug) {
                std::cout << "Debug: " << debug << std::endl;
                g_free(debug);
            }
            g_error_free(error);
            break;
        }
        
        case GST_MESSAGE_STATE_CHANGED: {
            if (GST_OBJECT(msg->src) == GST_OBJECT(builder->pipeline)) {
                GstState old_state, new_state, pending_state;
                gst_message_parse_state_changed(msg, &old_state, &new_state, &pending_state);
                std::cout << "Pipeline state changed from " 
                         << gst_element_state_get_name(old_state) << " to "
                         << gst_element_state_get_name(new_state) << std::endl;
            }
            break;
        }
        
        default:
            break;
    }
    
    return TRUE;
}

GstPadProbeReturn PipelineBuilder::tensor_extract_probe(GstPad *pad, GstPadProbeInfo *info, gpointer user_data) {
    PipelineBuilder* builder = static_cast<PipelineBuilder*>(user_data);
    
    // Priority 1: Try async processing if enabled
    if (builder->async_processor && builder->async_processor->is_running()) {
        GstBuffer *buffer = static_cast<GstBuffer*>(info->data);
        if (buffer) {
            // Process tensor metadata asynchronously
            NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buffer);
            if (batch_meta) {
                // Submit async tasks for each frame with tensor metadata
                NvDsMetaList* l_frame = batch_meta->frame_meta_list;
                while (l_frame != nullptr) {
                    NvDsFrameMeta* frame_meta = (NvDsFrameMeta*)(l_frame->data);
                    
                    // Look for tensor metadata in frame's user metadata list
                    NvDsMetaList* l_user_meta = frame_meta->frame_user_meta_list;
                    while (l_user_meta != nullptr) {
                        NvDsUserMeta* user_meta = (NvDsUserMeta*)(l_user_meta->data);
                        
                        if (user_meta && user_meta->base_meta.meta_type == NVDSINFER_TENSOR_OUTPUT_META) {
                            void* tensor_meta_ptr = user_meta->user_meta_data;
                            
                            // Task submitted to async processor
                            
                            // Submit async task (use void* to avoid header conflicts)
                            auto future = builder->async_processor->submit_tensor_task(
                                tensor_meta_ptr,
                                frame_meta->source_id,
                                0,  // Use 0 as default batch ID (or we could use frame_meta->batch_id if available)
                                frame_meta->frame_num,
                                frame_meta->ntp_timestamp
                            );
                            
                            // Note: We don't wait for the future here to maintain non-blocking behavior
                            // The async processor handles the actual processing in background threads
                        }
                        
                        l_user_meta = l_user_meta->next;
                    }
                    
                    l_frame = l_frame->next;
                }
            }
        }
    }
    // Priority 2: Fall back to legacy callback if async not available
    else if (builder->tensor_callback) {
        builder->tensor_callback(pad, info, builder->callback_user_data);
    }
    
    return GST_PAD_PROBE_OK;
}

void PipelineBuilder::qtdemux_pad_added_callback(GstElement* src, GstPad* new_pad, gpointer data) {
    GstElement* parser = static_cast<GstElement*>(data);
    GstPad* sink_pad = gst_element_get_static_pad(parser, "sink");
    
    if (gst_pad_link(new_pad, sink_pad) != GST_PAD_LINK_OK) {
        std::cerr << "Error: Failed to link qtdemux to parser" << std::endl;
    }
    
    gst_object_unref(sink_pad);
}

bool PipelineBuilder::check_element_creation(GstElement* element, const std::string& element_name) {
    if (!element) {
        std::cerr << "Error: Failed to create element: " << element_name << std::endl;
        return false;
    }
    return true;
}

void PipelineBuilder::print_pipeline_info() {
    std::cout << "\n=== Pipeline Information ===" << std::endl;
    std::cout << "Sources: " << config.sources.size() << std::endl;
    std::cout << "Batch size: " << config.batch_size << std::endl;
    std::cout << "Resolution: " << config.width << "x" << config.height << std::endl;
    std::cout << "GPU ID: " << config.gpu_id << std::endl;
    std::cout << "Display enabled: " << (config.enable_display ? "Yes" : "No") << std::endl;
    std::cout << "Model config: " << config.model_config_path << std::endl;
    std::cout << "=============================" << std::endl;
}

void PipelineBuilder::cleanup() {
    if (bus_watch_id > 0) {
        g_source_remove(bus_watch_id);
        bus_watch_id = 0;
    }
    
    if (pipeline) {
        gst_element_set_state(pipeline, GST_STATE_NULL);
        gst_object_unref(GST_OBJECT(pipeline));
        pipeline = nullptr;
    }
    
    source_bins.clear();
    
    // Reset pointers (they're owned by pipeline)
    streammux = nullptr;
    pgie = nullptr;
    nvvidconv = nullptr;
    nvosd = nullptr;
    tiler = nullptr;
    sink = nullptr;
    tee = nullptr;
    queue1 = nullptr;
    queue2 = nullptr;
    bus = nullptr;
}

void PipelineBuilder::enable_performance_monitoring(bool enable) {
    if (!pgie) return;
    
    g_object_set(G_OBJECT(pgie), "enable-perf-measurement", enable, nullptr);
}

void PipelineBuilder::print_performance_stats() {
    // Performance stats would be collected and displayed here
    std::cout << "=== Performance Statistics ===" << std::endl;
    std::cout << "Pipeline running with " << config.sources.size() << " sources" << std::endl;
    std::cout << "Batch processing: " << config.batch_size << " streams" << std::endl;
    
    if (async_processor && async_processor->is_running()) {
        auto stats = async_processor->get_stats();
        std::cout << "Async Processing Stats:" << std::endl;
        std::cout << "  Tasks Submitted: " << stats.tasks_submitted << std::endl;
        std::cout << "  Tasks Completed: " << stats.tasks_completed << std::endl;
        std::cout << "  Success Rate: " << std::fixed << std::setprecision(1) 
                  << stats.get_success_rate() << "%" << std::endl;
        std::cout << "  Avg Processing Time: " << std::fixed << std::setprecision(2) 
                  << stats.get_avg_processing_time_ms() << "ms" << std::endl;
    }
    
    std::cout << "===============================" << std::endl;
}

// ============================================================================
// Async Processing Integration
// ============================================================================

bool PipelineBuilder::enable_async_processing(std::shared_ptr<AsyncProcessor> processor) {
    if (!processor) {
        std::cerr << "[PipelineBuilder] ERROR: Null async processor provided" << std::endl;
        return false;
    }
    
    async_processor = processor;
    
    if (!async_processor->start()) {
        std::cerr << "[PipelineBuilder] ERROR: Failed to start async processor" << std::endl;
        async_processor.reset();
        return false;
    }
    
    std::cout << "[PipelineBuilder] Async processing enabled successfully" << std::endl;
    return true;
}

void PipelineBuilder::disable_async_processing() {
    if (async_processor) {
        async_processor->stop();
        async_processor.reset();
        std::cout << "[PipelineBuilder] Async processing disabled" << std::endl;
    }
}

bool PipelineBuilder::is_async_processing_enabled() const {
    return async_processor && async_processor->is_running();
}