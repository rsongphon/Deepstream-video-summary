#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <signal.h>
#include <unistd.h>
#include <getopt.h>
#include <fstream>
#include <yaml-cpp/yaml.h>

#include "pipeline_builder.h"
#include "tensor_processor.h"

// Global variables for signal handling
std::unique_ptr<PipelineBuilder> g_pipeline;
std::unique_ptr<TensorProcessor> g_tensor_processor;
GMainLoop *g_main_loop = nullptr;
bool g_interrupted = false;

// Signal handler for graceful shutdown
void signal_handler(int signum) {
    std::cout << "\nReceived signal " << signum << ", shutting down gracefully..." << std::endl;
    g_interrupted = true;
    
    if (g_main_loop) {
        g_main_loop_quit(g_main_loop);
    }
}

// Print application usage
void print_usage(const char* program_name) {
    std::cout << "\nDeepStream Multi-Source Batched Inference Application\n" << std::endl;
    std::cout << "Usage: " << program_name << " [OPTIONS] SOURCE1 SOURCE2 ... SOURCEN\n" << std::endl;
    
    std::cout << "Options:" << std::endl;
    std::cout << "  -c, --config FILE          Configuration file (YAML format)" << std::endl;
    std::cout << "  -m, --model-config FILE    Model configuration file" << std::endl;
    std::cout << "  -e, --model-engine FILE    Pre-built TensorRT engine file" << std::endl;
    std::cout << "  -b, --batch-size N         Batch size (auto-detected from sources if not specified)" << std::endl;
    std::cout << "  -g, --gpu-id N             GPU ID to use (default: 0)" << std::endl;
    std::cout << "  -w, --width N              Input width (default: 1920)" << std::endl;
    std::cout << "  -h, --height N             Input height (default: 1080)" << std::endl;
    std::cout << "  -d, --enable-display       Enable display output" << std::endl;
    std::cout << "  -p, --perf                 Enable performance measurement" << std::endl;
    std::cout << "  -o, --output-dir DIR       Output directory for tensor data (default: output)" << std::endl;
    std::cout << "  -f, --output-format FORMAT Export format: csv, json, binary (default: csv)" << std::endl;
    std::cout << "  --max-tensor-values N      Maximum tensor values to log (default: 100)" << std::endl;
    std::cout << "  --detailed-logging         Enable detailed logging" << std::endl;
    std::cout << "  --timeout N                Batch formation timeout in microseconds (default: 40000)" << std::endl;
    std::cout << "  --help                     Show this help message" << std::endl;
    
    std::cout << "\nSources:" << std::endl;
    std::cout << "  File:     /path/to/video.mp4" << std::endl;
    std::cout << "  RTSP:     rtsp://192.168.1.100:554/stream" << std::endl;
    std::cout << "  USB Cam:  /dev/video0" << std::endl;
    std::cout << "  HTTP:     http://example.com/stream" << std::endl;
    
    std::cout << "\nExamples:" << std::endl;
    std::cout << "  # Process 4 video files with display" << std::endl;
    std::cout << "  " << program_name << " -d video1.mp4 video2.mp4 video3.mp4 video4.mp4" << std::endl;
    std::cout << "\n  # Process 2 RTSP streams with custom batch size" << std::endl;
    std::cout << "  " << program_name << " -b 2 rtsp://cam1/stream rtsp://cam2/stream" << std::endl;
    std::cout << "\n  # Process mixed sources with custom configuration" << std::endl;
    std::cout << "  " << program_name << " -c config.yaml -p video1.mp4 rtsp://cam/stream /dev/video0" << std::endl;
    
    std::cout << std::endl;
}

// Load configuration from YAML file
bool load_config_from_file(const std::string& config_file, PipelineConfig& config) {
    try {
        YAML::Node yaml_config = YAML::LoadFile(config_file);
        
        // System configuration
        if (yaml_config["system"]) {
            if (yaml_config["system"]["gpu_id"]) {
                config.gpu_id = yaml_config["system"]["gpu_id"].as<int>();
            }
            if (yaml_config["system"]["enable_perf_measurement"]) {
                config.enable_perf_measurement = yaml_config["system"]["enable_perf_measurement"].as<bool>();
            }
        }
        
        // Pipeline configuration
        if (yaml_config["pipeline"]) {
            if (yaml_config["pipeline"]["batch_size"]) {
                config.batch_size = yaml_config["pipeline"]["batch_size"].as<int>();
            }
            if (yaml_config["pipeline"]["width"]) {
                config.width = yaml_config["pipeline"]["width"].as<int>();
            }
            if (yaml_config["pipeline"]["height"]) {
                config.height = yaml_config["pipeline"]["height"].as<int>();
            }
            if (yaml_config["pipeline"]["enable_display"]) {
                config.enable_display = yaml_config["pipeline"]["enable_display"].as<bool>();
            }
        }
        
        // StreamMux configuration
        if (yaml_config["streammux"]) {
            if (yaml_config["streammux"]["batched_push_timeout"]) {
                config.batched_push_timeout = yaml_config["streammux"]["batched_push_timeout"].as<int>();
            }
            if (yaml_config["streammux"]["nvbuf_memory_type"]) {
                config.nvbuf_memory_type = yaml_config["streammux"]["nvbuf_memory_type"].as<int>();
            }
        }
        
        // Primary GIE configuration
        if (yaml_config["primary_gie"]) {
            if (yaml_config["primary_gie"]["config_file_path"]) {
                config.model_config_path = yaml_config["primary_gie"]["config_file_path"].as<std::string>();
            }
            if (yaml_config["primary_gie"]["model_engine_file"]) {
                config.model_engine_path = yaml_config["primary_gie"]["model_engine_file"].as<std::string>();
            }
        }
        
        std::cout << "Loaded configuration from: " << config_file << std::endl;
        return true;
        
    } catch (const YAML::Exception& e) {
        std::cerr << "Error loading configuration file: " << e.what() << std::endl;
        return false;
    }
}

// Create source configurations from command line arguments
std::vector<SourceConfig> create_source_configs(const std::vector<std::string>& source_uris) {
    std::vector<SourceConfig> sources;
    
    for (size_t i = 0; i < source_uris.size(); i++) {
        SourceConfig source;
        source.source_id = static_cast<int>(i);
        source.uri = source_uris[i];
        
        // Determine if source is live based on URI
        if (source.uri.find("rtsp://") == 0 || 
            source.uri.find("/dev/video") == 0 ||
            source.uri.find("http://") == 0) {
            source.is_live = true;
            source.framerate = 30; // Default for live sources
        } else {
            source.is_live = false;
            source.framerate = 30; // Will be auto-detected for files
        }
        
        sources.push_back(source);
    }
    
    return sources;
}

// Tensor extraction callback function
void tensor_extraction_callback(GstPad *pad, GstPadProbeInfo *info, gpointer user_data) {
    TensorProcessor* processor = static_cast<TensorProcessor*>(user_data);
    GstBuffer *buf = static_cast<GstBuffer*>(info->data);
    
    if (processor && buf) {
        std::vector<TensorBatchData> batch_data;
        if (processor->extract_tensor_meta(buf, batch_data)) {
            // Tensors are automatically exported by the processor
            // Additional processing could be done here
        }
    }
}

// Performance monitoring thread
void* performance_monitor_thread(void* arg) {
    PipelineBuilder* pipeline = static_cast<PipelineBuilder*>(arg);
    
    while (!g_interrupted) {
        sleep(30); // Report every 30 seconds
        
        if (!g_interrupted && pipeline) {
            pipeline->print_performance_stats();
            if (g_tensor_processor) {
                g_tensor_processor->print_statistics();
            }
        }
    }
    
    return nullptr;
}

int main(int argc, char *argv[]) {
    // Default configuration
    PipelineConfig config;
    config.batch_size = 0; // Will be auto-set based on source count
    config.width = 1920;
    config.height = 1080;
    config.gpu_id = 0;
    config.model_config_path = "config/model_config.txt";
    config.model_engine_path = "";
    config.enable_display = false;
    config.enable_perf_measurement = false;
    config.batched_push_timeout = 40000;
    config.nvbuf_memory_type = 2; // Unified memory
    
    // Tensor processor configuration
    std::string output_dir = "output";
    ExportFormat export_format = ExportFormat::CSV;
    int max_tensor_values = 100;
    bool detailed_logging = false;
    
    // Command line options
    std::string config_file = "";
    std::vector<std::string> source_uris;
    
    // Parse command line options
    static struct option long_options[] = {
        {"config", required_argument, 0, 'c'},
        {"model-config", required_argument, 0, 'm'},
        {"model-engine", required_argument, 0, 'e'},
        {"batch-size", required_argument, 0, 'b'},
        {"gpu-id", required_argument, 0, 'g'},
        {"width", required_argument, 0, 'w'},
        {"height", required_argument, 0, 'h'},
        {"enable-display", no_argument, 0, 'd'},
        {"perf", no_argument, 0, 'p'},
        {"output-dir", required_argument, 0, 'o'},
        {"output-format", required_argument, 0, 'f'},
        {"max-tensor-values", required_argument, 0, 1001},
        {"detailed-logging", no_argument, 0, 1002},
        {"timeout", required_argument, 0, 1003},
        {"help", no_argument, 0, 1004},
        {0, 0, 0, 0}
    };
    
    int opt;
    while ((opt = getopt_long(argc, argv, "c:m:e:b:g:w:h:dpo:f:", long_options, nullptr)) != -1) {
        switch (opt) {
            case 'c':
                config_file = optarg;
                break;
            case 'm':
                config.model_config_path = optarg;
                break;
            case 'e':
                config.model_engine_path = optarg;
                break;
            case 'b':
                config.batch_size = std::stoi(optarg);
                break;
            case 'g':
                config.gpu_id = std::stoi(optarg);
                break;
            case 'w':
                config.width = std::stoi(optarg);
                break;
            case 'h':
                config.height = std::stoi(optarg);
                break;
            case 'd':
                config.enable_display = true;
                break;
            case 'p':
                config.enable_perf_measurement = true;
                break;
            case 'o':
                output_dir = optarg;
                break;
            case 'f':
                if (std::string(optarg) == "json") {
                    export_format = ExportFormat::JSON;
                } else if (std::string(optarg) == "binary") {
                    export_format = ExportFormat::BINARY;
                } else {
                    export_format = ExportFormat::CSV;
                }
                break;
            case 1001:
                max_tensor_values = std::stoi(optarg);
                break;
            case 1002:
                detailed_logging = true;
                break;
            case 1003:
                config.batched_push_timeout = std::stoi(optarg);
                break;
            case 1004:
                print_usage(argv[0]);
                return 0;
            default:
                print_usage(argv[0]);
                return 1;
        }
    }
    
    // Get source URIs from remaining arguments
    for (int i = optind; i < argc; i++) {
        source_uris.push_back(argv[i]);
    }
    
    // Validate arguments
    if (source_uris.empty()) {
        std::cerr << "Error: No source URIs provided!" << std::endl;
        print_usage(argv[0]);
        return 1;
    }
    
    if (source_uris.size() > 64) {
        std::cerr << "Error: Too many sources (max 64 supported)" << std::endl;
        return 1;
    }
    
    // Load configuration from file if specified
    if (!config_file.empty()) {
        if (!load_config_from_file(config_file, config)) {
            std::cerr << "Failed to load configuration file, using defaults" << std::endl;
        }
    }
    
    // Create source configurations
    config.sources = create_source_configs(source_uris);
    
    // Auto-adjust batch size to match source count if not explicitly set
    if (config.batch_size == 0 || config.batch_size != static_cast<int>(config.sources.size())) {
        config.batch_size = static_cast<int>(config.sources.size());
        std::cout << "Auto-adjusted batch size to " << config.batch_size << " (matching source count)" << std::endl;
    }
    
    // Print configuration summary
    std::cout << "\n=== Configuration Summary ===" << std::endl;
    std::cout << "Sources: " << config.sources.size() << std::endl;
    std::cout << "Batch Size: " << config.batch_size << std::endl;
    std::cout << "Resolution: " << config.width << "x" << config.height << std::endl;
    std::cout << "GPU ID: " << config.gpu_id << std::endl;
    std::cout << "Display: " << (config.enable_display ? "Enabled" : "Disabled") << std::endl;
    std::cout << "Model Config: " << config.model_config_path << std::endl;
    std::cout << "Output Directory: " << output_dir << std::endl;
    std::cout << "=============================" << std::endl;
    
    // Setup signal handlers
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    try {
        // Initialize tensor processor
        g_tensor_processor = std::make_unique<TensorProcessor>();
        if (!g_tensor_processor->initialize(output_dir, export_format)) {
            std::cerr << "Failed to initialize tensor processor" << std::endl;
            return 1;
        }
        
        g_tensor_processor->set_max_tensor_values(max_tensor_values);
        g_tensor_processor->set_detailed_logging(detailed_logging);
        g_tensor_processor->enable_tensor_export(true);
        
        // Initialize pipeline
        g_pipeline = std::make_unique<PipelineBuilder>();
        if (!g_pipeline->initialize(config)) {
            std::cerr << "Failed to initialize pipeline" << std::endl;
            return 1;
        }
        
        // Set tensor extraction callback
        g_pipeline->set_tensor_extraction_callback(tensor_extraction_callback, g_tensor_processor.get());
        
        // Create and start pipeline
        if (!g_pipeline->create_pipeline()) {
            std::cerr << "Failed to create pipeline" << std::endl;
            return 1;
        }
        
        if (!g_pipeline->start_pipeline()) {
            std::cerr << "Failed to start pipeline" << std::endl;
            return 1;
        }
        
        // Enable performance monitoring
        if (config.enable_perf_measurement) {
            g_pipeline->enable_performance_monitoring(true);
        }
        
        std::cout << "\nPipeline started successfully!" << std::endl;
        std::cout << "Processing " << config.sources.size() << " sources..." << std::endl;
        std::cout << "Press Ctrl+C to stop." << std::endl;
        
        // Create and run main loop
        g_main_loop = g_main_loop_new(nullptr, FALSE);
        
        // Start performance monitoring thread if enabled
        pthread_t perf_thread;
        if (config.enable_perf_measurement) {
            pthread_create(&perf_thread, nullptr, performance_monitor_thread, g_pipeline.get());
        }
        
        // Run main loop
        g_main_loop_run(g_main_loop);
        
        // Cleanup
        std::cout << "\nShutting down..." << std::endl;
        
        if (config.enable_perf_measurement) {
            pthread_join(perf_thread, nullptr);
        }
        
        g_pipeline->stop_pipeline();
        
        // Print final statistics
        std::cout << "\n=== Final Statistics ===" << std::endl;
        g_tensor_processor->print_statistics();
        
        g_main_loop_unref(g_main_loop);
        g_tensor_processor.reset();
        g_pipeline.reset();
        
        std::cout << "Application finished successfully." << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}