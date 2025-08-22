#include "DeepStreamPipeline.h"
#include <sstream>
#include <algorithm>
#include <cstring>

namespace VideoSummary {

DeepStreamPipeline::DeepStreamPipeline() {
    // Initialize GStreamer if not already done
    if (!gst_is_initialized()) {
        gst_init(nullptr, nullptr);
    }
    
    LOG_DEBUG("DeepStreamPipeline constructed");
}

DeepStreamPipeline::~DeepStreamPipeline() {
    cleanup();
    LOG_DEBUG("DeepStreamPipeline destroyed");
}

bool DeepStreamPipeline::initialize(const PipelineConfiguration& config) {
    LOG_INFO_FMT("Initializing DeepStream pipeline: {}", config.name);
    
    // Validate configuration
    if (!validateConfiguration(config)) {
        last_error_ = "Invalid configuration provided";
        LOG_ERROR(last_error_);
        return false;
    }
    
    // Clean up any existing pipeline
    cleanup();
    
    // Store configuration
    config_ = config;
    config_.setDefaults();
    
    // Create pipeline elements
    if (!createElements()) {
        cleanup();
        return false;
    }
    
    // Configure elements
    if (!configureElements()) {
        cleanup();
        return false;
    }
    
    // Link elements
    if (!linkElements()) {
        cleanup();
        return false;
    }
    
    // Setup bus for message handling
    if (!setupBus()) {
        cleanup();
        return false;
    }
    
    // Initialize statistics
    stats_.reset();
    start_time_ = std::chrono::steady_clock::now();
    last_stats_time_ = start_time_;
    
    current_state_ = PipelineState::NULL_STATE;
    
    LOG_INFO("Pipeline initialized successfully");
    return true;
}

bool DeepStreamPipeline::initializeFromFile(const std::string& config_file) {
    LOG_INFO_FMT("Loading pipeline configuration from: {}", config_file);
    
    try {
        PipelineConfiguration config = loadConfigurationFromFile(config_file);
        return initialize(config);
    } catch (const std::exception& e) {
        last_error_ = "Failed to load configuration: " + std::string(e.what());
        LOG_ERROR(last_error_);
        return false;
    }
}

bool DeepStreamPipeline::start() {
    if (current_state_ == PipelineState::UNINITIALIZED) {
        last_error_ = "Pipeline not initialized";
        LOG_ERROR(last_error_);
        return false;
    }
    
    LOG_INFO("Starting pipeline playback");
    
    // Start main loop in separate thread if not running
    if (!main_loop_ || !g_main_loop_is_running(main_loop_)) {
        main_loop_ = g_main_loop_new(nullptr, FALSE);
        loop_thread_ = std::thread(&DeepStreamPipeline::runMainLoop, this);
    }
    
    // Set pipeline to PLAYING state
    if (!setState(GST_STATE_PLAYING)) {
        last_error_ = "Failed to set pipeline to PLAYING state";
        LOG_ERROR(last_error_);
        return false;
    }
    
    // Enable performance measurement if configured
    if (config_.enable_perf_measurement) {
        enablePerformanceMeasurement(true, config_.perf_measurement_interval_sec);
    }
    
    LOG_INFO("Pipeline started successfully");
    return true;
}

bool DeepStreamPipeline::pause() {
    if (current_state_ != PipelineState::PLAYING) {
        last_error_ = "Pipeline is not playing";
        LOG_WARN(last_error_);
        return false;
    }
    
    LOG_INFO("Pausing pipeline");
    
    if (!setState(GST_STATE_PAUSED)) {
        last_error_ = "Failed to pause pipeline";
        LOG_ERROR(last_error_);
        return false;
    }
    
    LOG_INFO("Pipeline paused successfully");
    return true;
}

bool DeepStreamPipeline::stop() {
    LOG_INFO("Stopping pipeline");
    
    // Stop performance measurement
    perf_measurement_enabled_ = false;
    
    // Set pipeline to NULL state
    if (pipeline_ && !setState(GST_STATE_NULL)) {
        LOG_WARN("Failed to set pipeline to NULL state cleanly");
    }
    
    // Stop main loop
    stopMainLoop();
    
    current_state_ = PipelineState::NULL_STATE;
    
    LOG_INFO("Pipeline stopped successfully");
    return true;
}

PipelineState DeepStreamPipeline::getState() const {
    return current_state_.load();
}

bool DeepStreamPipeline::waitForEOS(int timeout_sec) {
    if (!pipeline_) {
        return false;
    }
    
    LOG_INFO_FMT("Waiting for EOS (timeout: {}s)", timeout_sec);
    
    GstBus* bus = gst_element_get_bus(pipeline_);
    GstMessage* msg;
    
    GstClockTime timeout = (timeout_sec > 0) ? 
                          timeout_sec * GST_SECOND : GST_CLOCK_TIME_NONE;
    
    bool eos_received = false;
    
    while (!eos_received) {
        msg = gst_bus_timed_pop_filtered(bus, timeout,
                                        static_cast<GstMessageType>(
                                            GST_MESSAGE_ERROR | GST_MESSAGE_EOS));
        
        if (!msg) {
            // Timeout
            break;
        }
        
        switch (GST_MESSAGE_TYPE(msg)) {
            case GST_MESSAGE_EOS:
                LOG_INFO("Received EOS message");
                eos_received = true;
                break;
            case GST_MESSAGE_ERROR:
                handleErrorMessage(msg);
                break;
            default:
                break;
        }
        
        gst_message_unref(msg);
    }
    
    gst_object_unref(bus);
    return eos_received;
}

bool DeepStreamPipeline::seek(double position_sec) {
    if (!pipeline_) {
        last_error_ = "Pipeline not initialized";
        return false;
    }
    
    LOG_INFO_FMT("Seeking to position: {:.2f}s", position_sec);
    
    gint64 seek_pos = static_cast<gint64>(position_sec * GST_SECOND);
    
    if (!gst_element_seek_simple(pipeline_, GST_FORMAT_TIME,
                                GST_SEEK_FLAG_FLUSH, seek_pos)) {
        last_error_ = "Seek operation failed";
        LOG_ERROR(last_error_);
        return false;
    }
    
    return true;
}

double DeepStreamPipeline::getCurrentPosition() const {
    if (!pipeline_) {
        return -1.0;
    }
    
    gint64 position;
    if (!gst_element_query_position(pipeline_, GST_FORMAT_TIME, &position)) {
        return -1.0;
    }
    
    return static_cast<double>(position) / GST_SECOND;
}

double DeepStreamPipeline::getDuration() const {
    if (!pipeline_) {
        return -1.0;
    }
    
    gint64 duration;
    if (!gst_element_query_duration(pipeline_, GST_FORMAT_TIME, &duration)) {
        return -1.0;
    }
    
    return static_cast<double>(duration) / GST_SECOND;
}

void DeepStreamPipeline::setEventCallback(EventCallback callback, gpointer user_data) {
    event_callback_ = callback;
    callback_user_data_ = user_data;
}

PipelineStats DeepStreamPipeline::getStats() const {
    updateStats();
    return stats_;
}

void DeepStreamPipeline::enablePerformanceMeasurement(bool enable, int interval_sec) {
    perf_measurement_enabled_ = enable;
    
    if (enable) {
        LOG_INFO_FMT("Performance measurement enabled (interval: {}s)", interval_sec);
        
        // Start performance monitoring thread
        std::thread([this, interval_sec]() {
            while (perf_measurement_enabled_) {
                updateStats();
                
                auto stats = getStats();
                LOG_INFO_FMT("Performance: FPS={:.2f}, Latency={:.2f}ms, Memory={}MB",
                           stats.current_fps, stats.pipeline_latency_ms, stats.memory_usage_mb);
                
                std::this_thread::sleep_for(std::chrono::seconds(interval_sec));
            }
        }).detach();
    } else {
        LOG_INFO("Performance measurement disabled");
    }
}

bool DeepStreamPipeline::exportPipelineGraph(const std::string& filename) const {
    if (!pipeline_) {
        return false;
    }
    
    GST_DEBUG_BIN_TO_DOT_FILE(GST_BIN(pipeline_), GST_DEBUG_GRAPH_SHOW_ALL, filename.c_str());
    LOG_INFO_FMT("Pipeline graph exported to: {}.dot", filename);
    return true;
}

// Private implementation methods

bool DeepStreamPipeline::createElements() {
    LOG_DEBUG("Creating GStreamer elements");
    
    // Create pipeline
    pipeline_ = gst_pipeline_new(config_.name.c_str());
    if (!pipeline_) {
        last_error_ = "Failed to create pipeline";
        LOG_ERROR(last_error_);
        return false;
    }
    
    // Create source element
    std::string source_name = getSourceElementName();
    source_ = createElement(source_name, "source");
    if (!source_) return false;
    
    // Create parser if needed
    if (requiresParser()) {
        std::string parser_name = getParserElementName();
        parser_ = createElement(parser_name, "parser");
        if (!parser_) return false;
    }
    
    // Create decoder if needed
    if (requiresDecoder()) {
        std::string decoder_name = getDecoderElementName();
        decoder_ = createElement(decoder_name, "decoder");
        if (!decoder_) return false;
    }
    
    // Create stream multiplexer
    streammux_ = createElement("nvstreammux", "stream-muxer");
    if (!streammux_) return false;
    
    // Create sink
    std::string sink_name = getSinkElementName();
    sink_ = createElement(sink_name, "sink");
    if (!sink_) return false;
    
    LOG_DEBUG("All elements created successfully");
    return true;
}

bool DeepStreamPipeline::configureElements() {
    LOG_DEBUG("Configuring GStreamer elements");
    
    // Configure source
    if (!setElementProperties(source_, "source")) return false;
    
    // Configure parser
    if (parser_ && !setElementProperties(parser_, "parser")) return false;
    
    // Configure decoder  
    if (decoder_ && !setElementProperties(decoder_, "decoder")) return false;
    
    // Configure stream multiplexer
    if (!setElementProperties(streammux_, "streammux")) return false;
    
    // Configure sink
    if (!setElementProperties(sink_, "sink")) return false;
    
    LOG_DEBUG("All elements configured successfully");
    return true;
}

bool DeepStreamPipeline::linkElements() {
    LOG_DEBUG("Linking GStreamer elements");
    
    // Add all elements to pipeline
    std::vector<GstElement*> elements = {source_};
    if (parser_) elements.push_back(parser_);
    if (decoder_) elements.push_back(decoder_);
    elements.push_back(streammux_);
    elements.push_back(sink_);
    
    for (GstElement* element : elements) {
        gst_bin_add(GST_BIN(pipeline_), element);
    }
    
    // Link elements in sequence
    GstElement* current = source_;
    
    if (parser_) {
        if (!gst_element_link(current, parser_)) {
            last_error_ = "Failed to link source to parser";
            LOG_ERROR(last_error_);
            return false;
        }
        current = parser_;
    }
    
    if (decoder_) {
        if (!gst_element_link(current, decoder_)) {
            last_error_ = "Failed to link to decoder";
            LOG_ERROR(last_error_);
            return false;
        }
        current = decoder_;
    }
    
    // Link to stream mux (requires special handling for request pads)
    GstPad* srcpad = gst_element_get_static_pad(current, "src");
    GstPad* sinkpad = gst_element_request_pad_simple(streammux_, "sink_0");
    
    if (!srcpad || !sinkpad) {
        last_error_ = "Failed to get pads for stream mux linking";
        LOG_ERROR(last_error_);
        return false;
    }
    
    if (gst_pad_link(srcpad, sinkpad) != GST_PAD_LINK_OK) {
        last_error_ = "Failed to link to stream multiplexer";
        LOG_ERROR(last_error_);
        gst_object_unref(srcpad);
        gst_object_unref(sinkpad);
        return false;
    }
    
    gst_object_unref(srcpad);
    gst_object_unref(sinkpad);
    
    // Link stream mux to sink
    if (!gst_element_link(streammux_, sink_)) {
        last_error_ = "Failed to link stream multiplexer to sink";
        LOG_ERROR(last_error_);
        return false;
    }
    
    LOG_DEBUG("All elements linked successfully");
    return true;
}

bool DeepStreamPipeline::setupBus() {
    LOG_DEBUG("Setting up message bus");
    
    bus_ = gst_element_get_bus(pipeline_);
    if (!bus_) {
        last_error_ = "Failed to get pipeline bus";
        LOG_ERROR(last_error_);
        return false;
    }
    
    bus_watch_id_ = gst_bus_add_watch(bus_, busCallback, this);
    
    LOG_DEBUG("Message bus configured successfully");
    return true;
}

void DeepStreamPipeline::cleanup() {
    LOG_DEBUG("Cleaning up pipeline resources");
    
    // Stop main loop
    stopMainLoop();
    
    // Remove bus watch
    if (bus_watch_id_ > 0) {
        g_source_remove(bus_watch_id_);
        bus_watch_id_ = 0;
    }
    
    // Cleanup bus
    if (bus_) {
        gst_object_unref(bus_);
        bus_ = nullptr;
    }
    
    // Set pipeline to NULL and cleanup
    if (pipeline_) {
        gst_element_set_state(pipeline_, GST_STATE_NULL);
        gst_object_unref(pipeline_);
        pipeline_ = nullptr;
    }
    
    // Reset element pointers (they're owned by pipeline)
    source_ = nullptr;
    parser_ = nullptr;
    decoder_ = nullptr;
    streammux_ = nullptr;
    sink_ = nullptr;
    
    current_state_ = PipelineState::UNINITIALIZED;
    
    LOG_DEBUG("Cleanup completed");
}

GstElement* DeepStreamPipeline::createElement(const std::string& factory_name,
                                            const std::string& element_name) {
    GstElement* element = gst_element_factory_make(factory_name.c_str(), element_name.c_str());
    
    if (!element) {
        std::stringstream ss;
        ss << "Failed to create element: " << factory_name << " (" << element_name << ")";
        last_error_ = ss.str();
        LOG_ERROR(last_error_);
        return nullptr;
    }
    
    LOG_DEBUG_FMT("Created element: {} ({})", factory_name, element_name);
    return element;
}

bool DeepStreamPipeline::setElementProperties(GstElement* element, const std::string& element_name) {
    if (!element) return false;
    
    LOG_DEBUG_FMT("Configuring element: {}", element_name);
    
    if (element_name == "source") {
        g_object_set(G_OBJECT(element),
                    "location", config_.source.uri.c_str(),
                    NULL);
                    
        if (config_.source.loop && isFileSource()) {
            // Note: loop property is not available on filesrc
            // This would need to be handled at application level
        }
    }
    else if (element_name == "streammux") {
        g_object_set(G_OBJECT(element),
                    "batch-size", config_.streammux.batch_size,
                    "width", config_.streammux.width,
                    "height", config_.streammux.height,
                    "batched-push-timeout", config_.streammux.batched_push_timeout,
                    "enable-padding", config_.streammux.enable_padding,
                    "nvbuf-memory-type", config_.streammux.nvbuf_memory_type,
                    NULL);
    }
    else if (element_name == "sink") {
        g_object_set(G_OBJECT(element),
                    "sync", config_.sink.sync,
                    "async", config_.sink.async,
                    NULL);
                    
        if (config_.sink.type == SinkConfig::FILE_SINK && !config_.sink.location.empty()) {
            g_object_set(G_OBJECT(element),
                        "location", config_.sink.location.c_str(),
                        NULL);
        }
    }
    
    return true;
}

bool DeepStreamPipeline::setState(GstState state) {
    if (!pipeline_) {
        return false;
    }
    
    GstStateChangeReturn ret = gst_element_set_state(pipeline_, state);
    
    if (ret == GST_STATE_CHANGE_FAILURE) {
        last_error_ = "Failed to change pipeline state";
        current_state_ = PipelineState::ERROR_STATE;
        return false;
    }
    
    // Wait for state change to complete
    if (ret == GST_STATE_CHANGE_ASYNC) {
        ret = gst_element_get_state(pipeline_, nullptr, nullptr, GST_CLOCK_TIME_NONE);
        if (ret == GST_STATE_CHANGE_FAILURE) {
            last_error_ = "Pipeline state change failed (async)";
            current_state_ = PipelineState::ERROR_STATE;
            return false;
        }
    }
    
    current_state_ = gstStateToPipelineState(state);
    return true;
}

PipelineState DeepStreamPipeline::gstStateToPipelineState(GstState state) const {
    switch (state) {
        case GST_STATE_NULL:
            return PipelineState::NULL_STATE;
        case GST_STATE_READY:
            return PipelineState::READY;
        case GST_STATE_PAUSED:
            return PipelineState::PAUSED;
        case GST_STATE_PLAYING:
            return PipelineState::PLAYING;
        default:
            return PipelineState::ERROR_STATE;
    }
}

void DeepStreamPipeline::updateStats() const {
    auto now = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_stats_time_);
    
    if (duration.count() > 0) {
        // Update FPS calculation
        // This is a simplified calculation - in real implementation you'd track actual frames
        // double elapsed_sec = duration.count() / 1000.0; // Unused for now
        stats_.current_fps = 30.0; // Placeholder - would be calculated from actual frame count
        
        // Update average FPS
        auto total_duration = std::chrono::duration_cast<std::chrono::seconds>(now - start_time_);
        if (total_duration.count() > 0) {
            stats_.average_fps = stats_.frames_processed / total_duration.count();
        }
        
        // Update latency (placeholder)
        stats_.pipeline_latency_ms = 33.3; // Placeholder value
        
        // Cast away const for mutable member (stats update pattern)
        const_cast<DeepStreamPipeline*>(this)->last_stats_time_ = now;
    }
}

gboolean DeepStreamPipeline::busCallback(GstBus* bus, GstMessage* message, gpointer user_data) {
    // Suppress unused parameter warning
    (void)bus;
    
    DeepStreamPipeline* pipeline = static_cast<DeepStreamPipeline*>(user_data);
    pipeline->handleBusMessage(message);
    return TRUE;
}

void DeepStreamPipeline::handleBusMessage(GstMessage* message) {
    switch (GST_MESSAGE_TYPE(message)) {
        case GST_MESSAGE_ERROR:
            handleErrorMessage(message);
            break;
        case GST_MESSAGE_EOS:
            handleEOSMessage();
            break;
        case GST_MESSAGE_STATE_CHANGED:
            handleStateChangedMessage(message);
            break;
        case GST_MESSAGE_WARNING:
            handleWarningMessage(message);
            break;
        default:
            break;
    }
}

void DeepStreamPipeline::handleErrorMessage(GstMessage* message) {
    GError* error;
    gchar* debug;
    
    gst_message_parse_error(message, &error, &debug);
    
    std::stringstream ss;
    ss << "Pipeline error: " << error->message;
    if (debug) {
        ss << " (Debug: " << debug << ")";
    }
    
    last_error_ = ss.str();
    LOG_ERROR(last_error_);
    
    current_state_ = PipelineState::ERROR_STATE;
    stats_.error_count++;
    
    // Notify callback
    if (event_callback_) {
        event_callback_("error", error, callback_user_data_);
    }
    
    g_clear_error(&error);
    g_free(debug);
}

void DeepStreamPipeline::handleEOSMessage() {
    LOG_INFO("Received End-Of-Stream");
    
    // Notify callback
    if (event_callback_) {
        event_callback_("eos", nullptr, callback_user_data_);
    }
}

void DeepStreamPipeline::handleStateChangedMessage(GstMessage* message) {
    GstState old_state, new_state, pending_state;
    gst_message_parse_state_changed(message, &old_state, &new_state, &pending_state);
    
    // Only handle pipeline state changes
    if (GST_MESSAGE_SRC(message) == GST_OBJECT(pipeline_)) {
        current_state_ = gstStateToPipelineState(new_state);
        
        LOG_DEBUG_FMT("Pipeline state changed: {} -> {}", 
                     gst_element_state_get_name(old_state),
                     gst_element_state_get_name(new_state));
        
        // Notify callback
        if (event_callback_) {
            event_callback_("state-change", message, callback_user_data_);
        }
    }
}

void DeepStreamPipeline::handleWarningMessage(GstMessage* message) {
    GError* warning;
    gchar* debug;
    
    gst_message_parse_warning(message, &warning, &debug);
    
    LOG_WARN_FMT("Pipeline warning: {} (Debug: {})", 
                warning->message, debug ? debug : "none");
    
    stats_.warning_count++;
    
    g_clear_error(&warning);
    g_free(debug);
}

void DeepStreamPipeline::runMainLoop() {
    if (main_loop_) {
        LOG_DEBUG("Starting main loop");
        g_main_loop_run(main_loop_);
        LOG_DEBUG("Main loop stopped");
    }
}

void DeepStreamPipeline::stopMainLoop() {
    if (main_loop_ && g_main_loop_is_running(main_loop_)) {
        LOG_DEBUG("Stopping main loop");
        g_main_loop_quit(main_loop_);
    }
    
    if (loop_thread_.joinable()) {
        loop_thread_.join();
    }
    
    if (main_loop_) {
        g_main_loop_unref(main_loop_);
        main_loop_ = nullptr;
    }
}

// Helper methods for element names

std::string DeepStreamPipeline::getSourceElementName() const {
    return "filesrc";  // For now, only support file sources
}

std::string DeepStreamPipeline::getParserElementName() const {
    switch (config_.source.type) {
        case SourceType::FILE_H264:
            return "h264parse";
        case SourceType::FILE_MP4:
            return "qtdemux";
        default:
            return "h264parse";
    }
}

std::string DeepStreamPipeline::getDecoderElementName() const {
    return "nvv4l2decoder";  // NVIDIA hardware decoder
}

std::string DeepStreamPipeline::getSinkElementName() const {
    switch (config_.sink.type) {
        case SinkConfig::FILE_SINK:
            return "filesink";
        case SinkConfig::DISPLAY_SINK:
            return "nveglglessink";
        case SinkConfig::FAKE_SINK:
        default:
            return "fakesink";
    }
}

bool DeepStreamPipeline::isFileSource() const {
    return config_.source.type == SourceType::FILE_H264 || 
           config_.source.type == SourceType::FILE_MP4;
}

bool DeepStreamPipeline::requiresParser() const {
    return config_.source.type == SourceType::FILE_H264;
}

bool DeepStreamPipeline::requiresDecoder() const {
    return isFileSource();  // File sources typically need decoding
}

PipelineConfiguration DeepStreamPipeline::loadConfigurationFromFile(const std::string& config_file) {
    // Use existing ConfigManager to load configuration
    auto& config_manager = ConfigManager::getInstance();
    
    if (!config_manager.loadConfig(config_file)) {
        throw std::runtime_error("Failed to load configuration file: " + config_file);
    }
    
    PipelineConfiguration config;
    
    // Load pipeline settings
    config.name = config_manager.getString("pipeline.name", config.name);
    
    // Load source settings
    std::string source_uri = config_manager.getString("source.uri", "");
    if (!source_uri.empty()) {
        config.source.uri = source_uri;
        
        // Determine source type from URI
        if (source_uri.find(".h264") != std::string::npos) {
            config.source.type = SourceType::FILE_H264;
        } else if (source_uri.find(".mp4") != std::string::npos) {
            config.source.type = SourceType::FILE_MP4;
        }
    }
    
    config.source.gpu_id = config_manager.getInt("source.gpu_id", config.source.gpu_id);
    
    // Load streammux settings
    config.streammux.batch_size = config_manager.getInt("streammux.batch_size", config.streammux.batch_size);
    config.streammux.width = config_manager.getInt("streammux.width", config.streammux.width);
    config.streammux.height = config_manager.getInt("streammux.height", config.streammux.height);
    
    // Load sink settings
    std::string sink_type = config_manager.getString("sink.type", "fake");
    if (sink_type == "file") {
        config.sink.type = SinkConfig::FILE_SINK;
        config.sink.location = config_manager.getString("sink.location", "");
    } else if (sink_type == "display") {
        config.sink.type = SinkConfig::DISPLAY_SINK;
    } else {
        config.sink.type = SinkConfig::FAKE_SINK;
    }
    
    config.sink.sync = config_manager.getBool("sink.sync", config.sink.sync);
    
    return config;
}

bool DeepStreamPipeline::validateConfiguration(const PipelineConfiguration& config) const {
    if (!config.isValid()) {
        LOG_ERROR("Configuration validation failed");
        return false;
    }
    
    // Additional validation can be added here
    return true;
}

} // namespace VideoSummary