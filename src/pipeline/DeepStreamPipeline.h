#pragma once

#include <gst/gst.h>
#include <glib.h>
#include <memory>
#include <functional>
#include <atomic>
#include <chrono>
#include <thread>

#include "PipelineConfig.h"
#include "../utils/Logger.h"
#include "../utils/ConfigManager.h"

namespace VideoSummary {

/**
 * @brief Main DeepStream pipeline class for video processing
 * 
 * This class manages the complete GStreamer pipeline lifecycle, including:
 * - Element creation and configuration
 * - Pipeline state management
 * - Error handling and recovery  
 * - Performance monitoring
 * - Resource cleanup
 */
class DeepStreamPipeline {
public:
    /**
     * @brief Pipeline event callback function type
     * @param event_type Type of event (e.g., "eos", "error", "state-change")
     * @param data Event-specific data
     * @param user_data User-provided data pointer
     */
    using EventCallback = std::function<void(const std::string& event_type, 
                                           gpointer data, gpointer user_data)>;

    DeepStreamPipeline();
    ~DeepStreamPipeline();

    // Non-copyable but movable
    DeepStreamPipeline(const DeepStreamPipeline&) = delete;
    DeepStreamPipeline& operator=(const DeepStreamPipeline&) = delete;
    DeepStreamPipeline(DeepStreamPipeline&&) = default;
    DeepStreamPipeline& operator=(DeepStreamPipeline&&) = default;

    /**
     * @brief Initialize the pipeline with given configuration
     * @param config Pipeline configuration parameters
     * @return true if initialization successful, false otherwise
     */
    bool initialize(const PipelineConfiguration& config);

    /**
     * @brief Initialize pipeline from configuration file
     * @param config_file Path to configuration file (INI or YAML)
     * @return true if initialization successful, false otherwise
     */
    bool initializeFromFile(const std::string& config_file);

    /**
     * @brief Start pipeline playback
     * @return true if started successfully, false otherwise
     */
    bool start();

    /**
     * @brief Pause pipeline playback
     * @return true if paused successfully, false otherwise
     */
    bool pause();

    /**
     * @brief Stop pipeline playback
     * @return true if stopped successfully, false otherwise
     */
    bool stop();

    /**
     * @brief Get current pipeline state
     * @return Current PipelineState
     */
    PipelineState getState() const;

    /**
     * @brief Check if pipeline is playing
     * @return true if pipeline is in PLAYING state
     */
    bool isPlaying() const { return getState() == PipelineState::PLAYING; }

    /**
     * @brief Wait for pipeline to reach EOS (End of Stream)
     * @param timeout_sec Timeout in seconds (0 = wait indefinitely)
     * @return true if EOS reached, false if timeout or error
     */
    bool waitForEOS(int timeout_sec = 0);

    /**
     * @brief Seek to specific position in stream
     * @param position_sec Position in seconds
     * @return true if seek successful, false otherwise
     */
    bool seek(double position_sec);

    /**
     * @brief Get current playback position
     * @return Current position in seconds, -1.0 if unavailable
     */
    double getCurrentPosition() const;

    /**
     * @brief Get total stream duration
     * @return Duration in seconds, -1.0 if unavailable
     */
    double getDuration() const;

    /**
     * @brief Set event callback for pipeline events
     * @param callback Callback function to receive events
     * @param user_data User data to pass to callback
     */
    void setEventCallback(EventCallback callback, gpointer user_data = nullptr);

    /**
     * @brief Get current pipeline statistics
     * @return PipelineStats structure with current metrics
     */
    PipelineStats getStats() const;

    /**
     * @brief Enable/disable performance measurement
     * @param enable true to enable performance measurement
     * @param interval_sec Measurement interval in seconds
     */
    void enablePerformanceMeasurement(bool enable, int interval_sec = 5);

    /**
     * @brief Export pipeline graph to DOT file for debugging
     * @param filename Output filename (without .dot extension)
     * @return true if export successful, false otherwise
     */
    bool exportPipelineGraph(const std::string& filename) const;

    /**
     * @brief Get last error message
     * @return Error message string, empty if no error
     */
    std::string getLastError() const { return last_error_; }

    /**
     * @brief Get pipeline configuration
     * @return Current pipeline configuration
     */
    const PipelineConfiguration& getConfiguration() const { return config_; }

private:
    // GStreamer elements
    GstElement* pipeline_ = nullptr;
    GstElement* source_ = nullptr;
    GstElement* parser_ = nullptr;
    GstElement* decoder_ = nullptr;
    GstElement* streammux_ = nullptr;
    GstElement* sink_ = nullptr;

    // GStreamer infrastructure
    GstBus* bus_ = nullptr;
    guint bus_watch_id_ = 0;
    GMainLoop* main_loop_ = nullptr;
    std::thread loop_thread_;

    // Configuration and state
    PipelineConfiguration config_;
    std::atomic<PipelineState> current_state_{PipelineState::UNINITIALIZED};
    std::string last_error_;

    // Event handling
    EventCallback event_callback_;
    gpointer callback_user_data_ = nullptr;

    // Performance monitoring
    mutable PipelineStats stats_;
    std::atomic<bool> perf_measurement_enabled_{false};
    std::chrono::steady_clock::time_point start_time_;
    std::chrono::steady_clock::time_point last_stats_time_;

    // Internal methods
    bool createElements();
    bool configureElements();
    bool linkElements();
    bool setupBus();
    void cleanup();

    // Element creation helpers
    GstElement* createElement(const std::string& factory_name, 
                            const std::string& element_name);
    bool setElementProperties(GstElement* element, const std::string& element_name);

    // State management
    bool setState(GstState state);
    PipelineState gstStateToPipelineState(GstState state) const;
    void updateStats() const;

    // Bus message handlers
    static gboolean busCallback(GstBus* bus, GstMessage* message, gpointer user_data);
    void handleBusMessage(GstMessage* message);
    void handleErrorMessage(GstMessage* message);
    void handleEOSMessage();
    void handleStateChangedMessage(GstMessage* message);
    void handleWarningMessage(GstMessage* message);

    // Main loop management
    void runMainLoop();
    void stopMainLoop();

    // Utility methods
    std::string getSourceElementName() const;
    std::string getParserElementName() const;
    std::string getDecoderElementName() const;
    std::string getSinkElementName() const;
    bool isFileSource() const;
    bool requiresParser() const;
    bool requiresDecoder() const;

    // Configuration helpers
    PipelineConfiguration loadConfigurationFromFile(const std::string& config_file);
    bool validateConfiguration(const PipelineConfiguration& config) const;
};

} // namespace VideoSummary