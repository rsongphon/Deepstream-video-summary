#include <gtest/gtest.h>
#include <gst/gst.h>
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <dlfcn.h>
#include <sys/stat.h>

// Include our project headers
#include "../src/utils/Logger.h"
#include "../src/utils/ConfigManager.h"

class EnvironmentTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize GStreamer
        gst_init(nullptr, nullptr);
        
        // Initialize logger
        VideoSummary::Logger::getInstance().setLevel(VideoSummary::LogLevel::DEBUG);
    }
    
    void TearDown() override {
        // Note: GStreamer cleanup handled automatically, 
        // calling gst_deinit() can cause issues in test environments
    }
    
    bool fileExists(const std::string& path) {
        struct stat buffer;
        return (stat(path.c_str(), &buffer) == 0);
    }
    
    bool libraryExists(const std::string& lib_path) {
        void* handle = dlopen(lib_path.c_str(), RTLD_LAZY);
        if (handle) {
            dlclose(handle);
            return true;
        }
        return false;
    }
};

// Test CUDA environment
TEST_F(EnvironmentTest, CUDAEnvironment) {
    int device_count = 0;
    cudaError_t error = cudaGetDeviceCount(&device_count);
    
    ASSERT_EQ(error, cudaSuccess) << "CUDA not properly initialized: " << cudaGetErrorString(error);
    ASSERT_GT(device_count, 0) << "No CUDA devices found";
    
    // Test device properties
    for (int i = 0; i < device_count; ++i) {
        cudaDeviceProp prop;
        error = cudaGetDeviceProperties(&prop, i);
        ASSERT_EQ(error, cudaSuccess) << "Failed to get device properties for device " << i;
        
        std::cout << "CUDA Device " << i << ": " << prop.name 
                  << " (Compute Capability: " << prop.major << "." << prop.minor << ")" << std::endl;
        
        // Ensure compute capability is sufficient for DeepStream
        ASSERT_GE(prop.major, 6) << "Device " << i << " has insufficient compute capability (need >= 6.0)";
    }
}

// Test GStreamer environment
TEST_F(EnvironmentTest, GStreamerEnvironment) {
    // Check GStreamer version
    guint major, minor, micro, nano;
    gst_version(&major, &minor, &micro, &nano);
    
    std::cout << "GStreamer version: " << major << "." << minor << "." << micro << std::endl;
    
    // Ensure minimum GStreamer version (1.20.3 required for DeepStream 7.1)
    ASSERT_GE(major, 1);
    if (major == 1) {
        ASSERT_GE(minor, 20);
        if (minor == 20) {
            ASSERT_GE(micro, 3);
        }
    }
}

// Test DeepStream plugin availability
TEST_F(EnvironmentTest, DeepStreamPlugins) {
    std::vector<std::string> required_plugins = {
        "nvstreammux",
        "nvdsosd",
        "nvinfer",
        "nvinferserver",
        "nvtracker",
        "nvdsanalytics",
        "nvdspreprocess",
        "nvdspostprocess",
        "nvmsgconv",
        "nvmsgbroker"
    };
    
    for (const auto& plugin_name : required_plugins) {
        GstElementFactory* factory = gst_element_factory_find(plugin_name.c_str());
        ASSERT_NE(factory, nullptr) << "DeepStream plugin not found: " << plugin_name;
        
        // Try to create the element
        GstElement* element = gst_element_factory_create(factory, nullptr);
        ASSERT_NE(element, nullptr) << "Failed to create element: " << plugin_name;
        
        gst_object_unref(element);
        gst_object_unref(factory);
        
        std::cout << "✓ DeepStream plugin available: " << plugin_name << std::endl;
    }
}

// Test standard GStreamer plugins
TEST_F(EnvironmentTest, StandardGStreamerPlugins) {
    std::vector<std::string> required_plugins = {
        "filesrc",
        "filesink",
        "uridecodebin",
        "videoconvert",
        "videoscale",
        "queue",
        "tee",
        "capsfilter",
        "fakesink"
    };
    
    for (const auto& plugin_name : required_plugins) {
        GstElementFactory* factory = gst_element_factory_find(plugin_name.c_str());
        ASSERT_NE(factory, nullptr) << "Standard GStreamer plugin not found: " << plugin_name;
        gst_object_unref(factory);
        
        std::cout << "✓ Standard plugin available: " << plugin_name << std::endl;
    }
}

// Test DeepStream library files
TEST_F(EnvironmentTest, DeepStreamLibraries) {
    std::string deepstream_lib_path = "/opt/nvidia/deepstream/deepstream-7.1/lib/";
    
    std::vector<std::string> required_libs = {
        "libnvdsgst_meta.so",
        "libnvds_meta.so",
        "libnvbufsurface.so",
        "libnvbufsurftransform.so",
        "libnvdsgst_helper.so",
        "libnvdsgst_smartrecord.so",
        "libnvds_msgbroker.so",
        "libnvds_logger.so"
    };
    
    for (const auto& lib : required_libs) {
        std::string full_path = deepstream_lib_path + lib;
        ASSERT_TRUE(fileExists(full_path)) << "DeepStream library not found: " << full_path;
        ASSERT_TRUE(libraryExists(full_path)) << "DeepStream library not loadable: " << full_path;
        
        std::cout << "✓ DeepStream library available: " << lib << std::endl;
    }
}

// Test DeepStream plugin libraries
TEST_F(EnvironmentTest, DeepStreamPluginLibraries) {
    std::string plugin_path = "/opt/nvidia/deepstream/deepstream-7.1/lib/gst-plugins/";
    
    std::vector<std::string> required_plugin_libs = {
        "libgstnvinfer.so",
        "libgstnvinferserver.so",
        "libgstnvstreammux.so",
        "libgstnvdsosd.so",
        "libgstnvtracker.so",
        "libgstnvdsanalytics.so",
        "libgstnvdspreprocess.so",
        "libgstnvdspostprocess.so"
    };
    
    for (const auto& lib : required_plugin_libs) {
        std::string full_path = plugin_path + lib;
        ASSERT_TRUE(fileExists(full_path)) << "DeepStream plugin library not found: " << full_path;
        
        std::cout << "✓ DeepStream plugin library available: " << lib << std::endl;
    }
}

// Test TensorRT availability
TEST_F(EnvironmentTest, TensorRTEnvironment) {
    // Try to load TensorRT library
    std::vector<std::string> tensorrt_libs = {
        "/usr/lib/x86_64-linux-gnu/libnvinfer.so",
        "/usr/lib/x86_64-linux-gnu/libnvinfer_plugin.so",
        "/usr/lib/x86_64-linux-gnu/libnvonnxparser.so"
    };
    
    bool tensorrt_found = false;
    for (const auto& lib : tensorrt_libs) {
        if (fileExists(lib)) {
            tensorrt_found = true;
            std::cout << "✓ TensorRT library found: " << lib << std::endl;
            break;
        }
    }
    
    // Also check in alternative locations
    if (!tensorrt_found) {
        std::vector<std::string> alt_locations = {
            "/usr/local/lib/libnvinfer.so",
            "/opt/tensorrt/lib/libnvinfer.so"
        };
        
        for (const auto& lib : alt_locations) {
            if (fileExists(lib)) {
                tensorrt_found = true;
                std::cout << "✓ TensorRT library found: " << lib << std::endl;
                break;
            }
        }
    }
    
    ASSERT_TRUE(tensorrt_found) << "TensorRT libraries not found. Please ensure TensorRT is installed.";
}

// Test environment variables
TEST_F(EnvironmentTest, EnvironmentVariables) {
    // Check CUDA_VER
    const char* cuda_ver = std::getenv("CUDA_VER");
    if (cuda_ver) {
        std::cout << "CUDA_VER: " << cuda_ver << std::endl;
        ASSERT_STREQ(cuda_ver, "12.6") << "CUDA_VER should be set to 12.6 for DeepStream 7.1";
    } else {
        std::cout << "Warning: CUDA_VER environment variable not set" << std::endl;
    }
    
    // Check LD_LIBRARY_PATH
    const char* ld_path = std::getenv("LD_LIBRARY_PATH");
    if (ld_path) {
        std::string ld_path_str(ld_path);
        ASSERT_NE(ld_path_str.find("/opt/nvidia/deepstream/deepstream-7.1/lib"), std::string::npos)
            << "LD_LIBRARY_PATH should include DeepStream lib directory";
        std::cout << "✓ LD_LIBRARY_PATH includes DeepStream libraries" << std::endl;
    }
    
    // Check GST_PLUGIN_PATH
    const char* gst_path = std::getenv("GST_PLUGIN_PATH");
    if (gst_path) {
        std::string gst_path_str(gst_path);
        ASSERT_NE(gst_path_str.find("/opt/nvidia/deepstream/deepstream-7.1/lib/gst-plugins"), std::string::npos)
            << "GST_PLUGIN_PATH should include DeepStream plugin directory";
        std::cout << "✓ GST_PLUGIN_PATH includes DeepStream plugins" << std::endl;
    }
}

// Test sample files availability
TEST_F(EnvironmentTest, SampleFiles) {
    std::string samples_path = "/opt/nvidia/deepstream/deepstream-7.1/samples/";
    
    ASSERT_TRUE(fileExists(samples_path + "streams/sample_720p.h264"))
        << "Sample video file not found";
    
    ASSERT_TRUE(fileExists(samples_path + "configs/deepstream-app/config_infer_primary.txt"))
        << "Sample config file not found";
    
    std::cout << "✓ Sample files available" << std::endl;
}

// Test our project utilities
TEST_F(EnvironmentTest, ProjectUtilities) {
    // Test Logger
    auto& logger = VideoSummary::Logger::getInstance();
    ASSERT_NO_THROW(logger.info("Test log message"));
    ASSERT_NO_THROW(logger.logPipelineMetrics(30.0, 33.3, 75.5));
    
    std::cout << "✓ Logger functionality working" << std::endl;
    
    // Test ConfigManager
    auto& config_manager = VideoSummary::ConfigManager::getInstance();
    
    // Test default values
    auto pipeline_config = config_manager.getPipelineConfig();
    ASSERT_GT(pipeline_config.batch_size, 0);
    ASSERT_GT(pipeline_config.width, 0);
    ASSERT_GT(pipeline_config.height, 0);
    
    std::cout << "✓ ConfigManager functionality working" << std::endl;
}

// Test build system requirements
TEST_F(EnvironmentTest, BuildRequirements) {
    // Check for required header files
    std::vector<std::string> required_headers = {
        "/opt/nvidia/deepstream/deepstream-7.1/sources/includes/nvdsmeta.h",
        "/opt/nvidia/deepstream/deepstream-7.1/sources/includes/gstnvdsmeta.h",
        "/opt/nvidia/deepstream/deepstream-7.1/sources/includes/nvds_tracker_meta.h"
    };
    
    for (const auto& header : required_headers) {
        ASSERT_TRUE(fileExists(header)) << "Required header file not found: " << header;
    }
    
    std::cout << "✓ Build requirements satisfied" << std::endl;
}

// Test GPU memory allocation
TEST_F(EnvironmentTest, GPUMemoryAllocation) {
    size_t free_mem, total_mem;
    cudaError_t error = cudaMemGetInfo(&free_mem, &total_mem);
    
    ASSERT_EQ(error, cudaSuccess) << "Failed to get GPU memory info";
    
    std::cout << "GPU Memory - Total: " << (total_mem / 1024 / 1024) << " MB, "
              << "Free: " << (free_mem / 1024 / 1024) << " MB" << std::endl;
    
    // Ensure we have at least 2GB of GPU memory available
    ASSERT_GE(total_mem, 2ULL * 1024 * 1024 * 1024) 
        << "Insufficient GPU memory (need at least 2GB)";
    
    // Test simple GPU memory allocation
    void* gpu_ptr = nullptr;
    error = cudaMalloc(&gpu_ptr, 1024 * 1024); // Allocate 1MB
    ASSERT_EQ(error, cudaSuccess) << "Failed to allocate GPU memory";
    
    if (gpu_ptr) {
        cudaFree(gpu_ptr);
    }
    
    std::cout << "✓ GPU memory allocation working" << std::endl;
}

// Performance baseline test
TEST_F(EnvironmentTest, PerformanceBaseline) {
    // Create a simple pipeline to test basic performance
    GstElement* pipeline = gst_pipeline_new("test-pipeline");
    GstElement* videotestsrc = gst_element_factory_make("videotestsrc", "source");
    GstElement* fakesink = gst_element_factory_make("fakesink", "sink");
    
    ASSERT_NE(pipeline, nullptr);
    ASSERT_NE(videotestsrc, nullptr);
    ASSERT_NE(fakesink, nullptr);
    
    // Configure test source for performance test
    g_object_set(G_OBJECT(videotestsrc),
                 "num-buffers", 300, // 10 seconds at 30fps
                 "pattern", 0, // SMPTE color bars
                 NULL);
    
    g_object_set(G_OBJECT(fakesink),
                 "sync", FALSE,
                 NULL);
    
    // Build pipeline
    gst_bin_add_many(GST_BIN(pipeline), videotestsrc, fakesink, NULL);
    ASSERT_TRUE(gst_element_link(videotestsrc, fakesink));
    
    // Run pipeline
    GstStateChangeReturn ret = gst_element_set_state(pipeline, GST_STATE_PLAYING);
    ASSERT_NE(ret, GST_STATE_CHANGE_FAILURE);
    
    // Wait for completion or timeout
    GstBus* bus = gst_element_get_bus(pipeline);
    GstMessage* msg = gst_bus_timed_pop_filtered(bus, 15 * GST_SECOND,
                                                (GstMessageType)(GST_MESSAGE_ERROR | GST_MESSAGE_EOS));
    
    if (msg) {
        if (GST_MESSAGE_TYPE(msg) == GST_MESSAGE_ERROR) {
            GError* error;
            gchar* debug;
            gst_message_parse_error(msg, &error, &debug);
            FAIL() << "Pipeline error: " << error->message;
        }
        gst_message_unref(msg);
    }
    
    // Cleanup
    gst_element_set_state(pipeline, GST_STATE_NULL);
    gst_object_unref(bus);
    gst_object_unref(pipeline);
    
    std::cout << "✓ Basic pipeline performance test passed" << std::endl;
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    
    std::cout << "=== DeepStream Video Summary Environment Validation ===" << std::endl;
    std::cout << "Running comprehensive environment tests..." << std::endl << std::endl;
    
    int result = RUN_ALL_TESTS();
    
    if (result == 0) {
        std::cout << std::endl << "🎉 All environment validation tests passed!" << std::endl;
        std::cout << "✅ Environment is ready for DeepStream Video Summary development" << std::endl;
    } else {
        std::cout << std::endl << "❌ Some environment validation tests failed!" << std::endl;
        std::cout << "Please fix the issues above before proceeding with development" << std::endl;
    }
    
    return result;
}