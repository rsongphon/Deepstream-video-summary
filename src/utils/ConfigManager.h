#pragma once

#include <string>
#include <memory>
#include <mutex>
#include <map>
#include <vector>
#include <functional>

// Try to use yaml-cpp if available, otherwise fallback to simple INI parsing
#ifdef YAML_CPP_API
#include <yaml-cpp/yaml.h>
#define HAS_YAML_CPP
#else
#include <fstream>
#include <sstream>
#include <algorithm>
#endif

namespace VideoSummary {

class ConfigManager {
public:
    static ConfigManager& getInstance();
    
    // Configuration loading
    bool loadConfig(const std::string& config_path);
    bool loadYAMLConfig(const std::string& yaml_path);
    bool loadINIConfig(const std::string& ini_path);
    
    // Configuration access methods
    template<typename T>
    T get(const std::string& key) const;
    
    template<typename T>
    T get(const std::string& key, const T& default_value) const;
    
    // Specialized getters for common types
    std::string getString(const std::string& key, const std::string& default_value = "") const;
    int getInt(const std::string& key, int default_value = 0) const;
    double getDouble(const std::string& key, double default_value = 0.0) const;
    bool getBool(const std::string& key, bool default_value = false) const;
    std::vector<std::string> getStringArray(const std::string& key) const;
    
    // Configuration validation
    bool hasKey(const std::string& key) const;
    bool validateConfig() const;
    
    // Configuration watching (hot reload)
    void watchForChanges(const std::string& config_path);
    void setChangeCallback(std::function<void(const std::string&)> callback);
    
    // Configuration sections for DeepStream
    struct PipelineConfig {
        int batch_size = 1;
        int width = 1920;
        int height = 1080;
        int gpu_id = 0;
        int num_surfaces = 20;
        bool enable_perf_measurement = true;
        int perf_measurement_interval_sec = 5;
    };
    
    struct SourceConfig {
        bool enable = true;
        int type = 3; // File source by default
        std::string uri;
        int gpu_id = 0;
        int cudadec_memtype = 0;
        int num_sources = 1;
    };
    
    struct InferenceConfig {
        bool enable = true;
        int gpu_id = 0;
        int gie_unique_id = 1;
        std::string config_file;
        std::string model_engine_file;
        int batch_size = 1;
        int interval = 0;
    };
    
    struct TritonConfig {
        std::string server_url = "localhost:8001";
        std::string model_repository = "./models";
        int max_batch_size = 8;
        std::vector<int> preferred_batch_sizes = {4, 8};
        int client_timeout_ms = 5000;
        bool enable_cuda_memory_pool = true;
    };
    
    struct TrackerConfig {
        bool enable = true;
        int tracker_width = 960;
        int tracker_height = 544;
        std::string ll_lib_file = "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so";
        std::string ll_config_file = "config_tracker_NvDCF.yml";
        int gpu_id = 0;
        bool display_tracking_id = true;
    };
    
    // Structured configuration getters
    PipelineConfig getPipelineConfig() const;
    SourceConfig getSourceConfig(int source_id = 0) const;
    InferenceConfig getPrimaryInferenceConfig() const;
    InferenceConfig getSecondaryInferenceConfig(int sgie_id = 0) const;
    TritonConfig getTritonConfig() const;
    TrackerConfig getTrackerConfig() const;
    
    // Configuration validation rules
    bool validatePipelineConfig(const PipelineConfig& config) const;
    bool validateSourceConfig(const SourceConfig& config) const;
    bool validateInferenceConfig(const InferenceConfig& config) const;
    bool validateTritonConfig(const TritonConfig& config) const;

private:
    ConfigManager();
    ~ConfigManager() = default;
    ConfigManager(const ConfigManager&) = delete;
    ConfigManager& operator=(const ConfigManager&) = delete;
    
    // Internal parsing methods
    std::vector<std::string> splitKey(const std::string& key) const;
    bool fileExists(const std::string& path) const;
    std::string detectConfigFormat(const std::string& config_path) const;
    
#ifdef HAS_YAML_CPP
    YAML::Node config_;
    YAML::Node getNestedNode(const std::string& key) const;
#else
    // Fallback: simple key-value storage
    std::map<std::string, std::string> config_map_;
    bool parseINIFile(const std::string& ini_path);
    std::string getValue(const std::string& key) const;
#endif
    
    mutable std::mutex config_mutex_;
    std::string current_config_path_;
    std::function<void(const std::string&)> change_callback_;
    
    // File watching (simplified implementation)
    void startFileWatcher(const std::string& file_path);
    bool isFileModified(const std::string& file_path) const;
    std::time_t last_modified_time_;
    
    // Default configurations
    void setDefaultValues();
    void validateRequiredKeys() const;
};

// Template implementations
template<typename T>
T ConfigManager::get(const std::string& key) const {
    std::lock_guard<std::mutex> lock(config_mutex_);
    
#ifdef HAS_YAML_CPP
    try {
        auto node = getNestedNode(key);
        if (!node) {
            throw std::runtime_error("Configuration key not found: " + key);
        }
        return node.as<T>();
    } catch (const YAML::Exception& e) {
        throw std::runtime_error("YAML parsing error for key '" + key + "': " + e.what());
    }
#else
    std::string value = getValue(key);
    if (value.empty()) {
        throw std::runtime_error("Configuration key not found: " + key);
    }
    
    // Simple type conversion for fallback implementation
    if constexpr (std::is_same_v<T, std::string>) {
        return value;
    } else if constexpr (std::is_same_v<T, int>) {
        return std::stoi(value);
    } else if constexpr (std::is_same_v<T, double>) {
        return std::stod(value);
    } else if constexpr (std::is_same_v<T, bool>) {
        std::string lower_value = value;
        std::transform(lower_value.begin(), lower_value.end(), lower_value.begin(), ::tolower);
        return (lower_value == "true" || lower_value == "1" || lower_value == "yes");
    } else {
        throw std::runtime_error("Unsupported type for configuration key: " + key);
    }
#endif
}

template<typename T>
T ConfigManager::get(const std::string& key, const T& default_value) const {
    try {
        return get<T>(key);
    } catch (const std::exception&) {
        return default_value;
    }
}

} // namespace VideoSummary