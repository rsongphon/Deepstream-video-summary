#include <gtest/gtest.h>
#include <chrono>
#include <thread>

#include "../src/pipeline/DeepStreamPipeline.h"
#include "../src/utils/Logger.h"

class BasicPipelineTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize GStreamer
        gst_init(nullptr, nullptr);
        
        // Set logger to debug level for detailed output
        VideoSummary::Logger::getInstance().setLevel(VideoSummary::LogLevel::DEBUG);
        
        LOG_INFO("=== Starting Pipeline Test ===");
    }
    
    void TearDown() override {
        LOG_INFO("=== Pipeline Test Complete ===");
    }
};

TEST_F(BasicPipelineTest, PipelineCreationAndDestruction) {
    LOG_INFO("Testing pipeline creation and destruction");
    
    VideoSummary::DeepStreamPipeline pipeline;
    
    // Pipeline should start in uninitialized state
    ASSERT_EQ(pipeline.getState(), VideoSummary::PipelineState::UNINITIALIZED);
    
    LOG_INFO("Pipeline created successfully");
}

TEST_F(BasicPipelineTest, BasicConfiguration) {
    LOG_INFO("Testing basic pipeline configuration");
    
    VideoSummary::PipelineConfiguration config;
    config.name = "test-pipeline";
    
    // Configure source
    config.source.type = VideoSummary::SourceType::FILE_H264;
    config.source.uri = "/opt/nvidia/deepstream/deepstream-7.1/samples/streams/sample_720p.h264";
    config.source.gpu_id = 0;
    
    // Configure stream mux
    config.streammux.batch_size = 1;
    config.streammux.width = 1920;
    config.streammux.height = 1080;
    config.streammux.gpu_id = 0;
    
    // Configure sink (fake sink for testing)
    config.sink.type = VideoSummary::SinkConfig::FAKE_SINK;
    config.sink.sync = false;
    config.sink.async = true;
    
    // Validate configuration
    ASSERT_TRUE(config.isValid()) << "Configuration should be valid";
    
    LOG_INFO("Configuration validation passed");
}

TEST_F(BasicPipelineTest, PipelineInitialization) {
    LOG_INFO("Testing pipeline initialization");
    
    VideoSummary::DeepStreamPipeline pipeline;
    VideoSummary::PipelineConfiguration config;
    
    // Set up valid configuration
    config.name = "test-init-pipeline";
    config.source.type = VideoSummary::SourceType::FILE_H264;
    config.source.uri = "/opt/nvidia/deepstream/deepstream-7.1/samples/streams/sample_720p.h264";
    config.sink.type = VideoSummary::SinkConfig::FAKE_SINK;
    config.sink.sync = false;
    
    // Initialize pipeline
    bool init_result = pipeline.initialize(config);
    ASSERT_TRUE(init_result) << "Pipeline initialization should succeed. Error: " << pipeline.getLastError();
    
    // Check state after initialization
    ASSERT_EQ(pipeline.getState(), VideoSummary::PipelineState::NULL_STATE);
    
    LOG_INFO("Pipeline initialized successfully");
}

TEST_F(BasicPipelineTest, PipelineFromConfigFile) {
    LOG_INFO("Testing pipeline initialization from config file");
    
    VideoSummary::DeepStreamPipeline pipeline;
    std::string config_file = "/opt/nvidia/deepstream/deepstream-7.1/sources/apps/deepstream-video-summary/configs/basic_pipeline_config.txt";
    
    // Initialize from config file
    bool init_result = pipeline.initializeFromFile(config_file);
    ASSERT_TRUE(init_result) << "Pipeline initialization from config should succeed. Error: " << pipeline.getLastError();
    
    // Verify pipeline configuration
    const auto& config = pipeline.getConfiguration();
    ASSERT_FALSE(config.name.empty());
    ASSERT_FALSE(config.source.uri.empty());
    
    LOG_INFO_FMT("Pipeline loaded with name: {}", config.name);
    LOG_INFO_FMT("Source URI: {}", config.source.uri);
}

TEST_F(BasicPipelineTest, PipelineStateTransitions) {
    LOG_INFO("Testing pipeline state transitions");
    
    VideoSummary::DeepStreamPipeline pipeline;
    VideoSummary::PipelineConfiguration config;
    
    // Set up configuration
    config.name = "test-state-pipeline";
    config.source.type = VideoSummary::SourceType::FILE_H264;
    config.source.uri = "/opt/nvidia/deepstream/deepstream-7.1/samples/streams/sample_720p.h264";
    config.sink.type = VideoSummary::SinkConfig::FAKE_SINK;
    config.sink.sync = false;
    
    // Initialize pipeline
    ASSERT_TRUE(pipeline.initialize(config)) << "Pipeline initialization failed: " << pipeline.getLastError();
    ASSERT_EQ(pipeline.getState(), VideoSummary::PipelineState::NULL_STATE);
    
    // Start pipeline
    bool start_result = pipeline.start();
    ASSERT_TRUE(start_result) << "Pipeline start should succeed. Error: " << pipeline.getLastError();
    
    // Wait a moment for state transition
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    
    // Check if pipeline is playing
    ASSERT_TRUE(pipeline.isPlaying()) << "Pipeline should be in PLAYING state";
    
    // Pause pipeline
    bool pause_result = pipeline.pause();
    ASSERT_TRUE(pause_result) << "Pipeline pause should succeed. Error: " << pipeline.getLastError();
    
    // Wait for state change
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    ASSERT_EQ(pipeline.getState(), VideoSummary::PipelineState::PAUSED);
    
    // Stop pipeline
    bool stop_result = pipeline.stop();
    ASSERT_TRUE(stop_result) << "Pipeline stop should succeed";
    
    LOG_INFO("All state transitions completed successfully");
}

TEST_F(BasicPipelineTest, PipelinePerformanceStats) {
    LOG_INFO("Testing pipeline performance statistics");
    
    VideoSummary::DeepStreamPipeline pipeline;
    VideoSummary::PipelineConfiguration config;
    
    // Configure pipeline
    config.name = "test-perf-pipeline";
    config.source.type = VideoSummary::SourceType::FILE_H264;
    config.source.uri = "/opt/nvidia/deepstream/deepstream-7.1/samples/streams/sample_720p.h264";
    config.sink.type = VideoSummary::SinkConfig::FAKE_SINK;
    config.enable_perf_measurement = true;
    config.perf_measurement_interval_sec = 1;
    
    ASSERT_TRUE(pipeline.initialize(config));
    ASSERT_TRUE(pipeline.start());
    
    // Let pipeline run for a short time
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    
    // Get performance statistics
    auto stats = pipeline.getStats();
    
    // Check that stats structure is properly initialized
    ASSERT_GE(stats.current_fps, 0.0);
    ASSERT_GE(stats.average_fps, 0.0);
    ASSERT_GE(stats.pipeline_latency_ms, 0.0);
    
    LOG_INFO_FMT("Performance Stats - FPS: {:.2f}, Latency: {:.2f}ms", 
                stats.current_fps, stats.pipeline_latency_ms);
    
    // Stop pipeline
    ASSERT_TRUE(pipeline.stop());
    
    LOG_INFO("Performance statistics test completed");
}

TEST_F(BasicPipelineTest, PipelineErrorHandling) {
    LOG_INFO("Testing pipeline error handling");
    
    VideoSummary::DeepStreamPipeline pipeline;
    VideoSummary::PipelineConfiguration config;
    
    // Set up invalid configuration (non-existent file)
    config.name = "test-error-pipeline";
    config.source.type = VideoSummary::SourceType::FILE_H264;
    config.source.uri = "/non/existent/file.h264";
    config.sink.type = VideoSummary::SinkConfig::FAKE_SINK;
    
    // Initialize should succeed (elements are created but not validated yet)
    bool init_result = pipeline.initialize(config);
    ASSERT_TRUE(init_result) << "Pipeline initialization should succeed even with bad file";
    
    // Starting should detect the error
    bool start_result = pipeline.start();
    
    if (start_result) {
        // If start succeeded, wait a bit and check for error state
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        
        // Pipeline might enter error state due to missing file
        auto state = pipeline.getState();
        LOG_INFO_FMT("Pipeline state with missing file: {}", static_cast<int>(state));
        
        pipeline.stop();
    } else {
        // Expected case - start failed due to missing file
        std::string error = pipeline.getLastError();
        ASSERT_FALSE(error.empty()) << "Error message should be provided";
        LOG_INFO_FMT("Expected error caught: {}", error);
    }
    
    LOG_INFO("Error handling test completed");
}

TEST_F(BasicPipelineTest, PipelineEventCallback) {
    LOG_INFO("Testing pipeline event callbacks");
    
    VideoSummary::DeepStreamPipeline pipeline;
    VideoSummary::PipelineConfiguration config;
    
    // Set up valid configuration
    config.name = "test-event-pipeline";
    config.source.type = VideoSummary::SourceType::FILE_H264;
    config.source.uri = "/opt/nvidia/deepstream/deepstream-7.1/samples/streams/sample_720p.h264";
    config.sink.type = VideoSummary::SinkConfig::FAKE_SINK;
    
    // Set up event callback
    bool callback_called = false;
    std::string last_event;
    
    pipeline.setEventCallback([&](const std::string& event_type, gpointer data, gpointer user_data) {
        (void)data;      // Suppress unused parameter warning
        (void)user_data; // Suppress unused parameter warning
        
        callback_called = true;
        last_event = event_type;
        LOG_INFO_FMT("Event callback received: {}", event_type);
    });
    
    ASSERT_TRUE(pipeline.initialize(config));
    ASSERT_TRUE(pipeline.start());
    
    // Wait for potential events
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    
    // Stop pipeline
    ASSERT_TRUE(pipeline.stop());
    
    LOG_INFO_FMT("Event callback test completed. Callback called: {}, Last event: {}", 
                callback_called, last_event);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    
    std::cout << "=== DeepStream Basic Pipeline Tests ===" << std::endl;
    std::cout << "Testing basic pipeline functionality..." << std::endl << std::endl;
    
    int result = RUN_ALL_TESTS();
    
    if (result == 0) {
        std::cout << std::endl << "🎉 All basic pipeline tests passed!" << std::endl;
        std::cout << "✅ Basic DeepStream pipeline functionality verified" << std::endl;
    } else {
        std::cout << std::endl << "❌ Some basic pipeline tests failed!" << std::endl;
        std::cout << "Please check the test output above for details" << std::endl;
    }
    
    return result;
}