#include "ConfigManager.h"
#include "Logger.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <stdexcept>
#include <sys/stat.h>

namespace VideoSummary {

ConfigManager& ConfigManager::getInstance() {
    static ConfigManager instance;
    return instance;
}

ConfigManager::ConfigManager() : last_modified_time_(0) {
    setDefaultValues();
}

bool ConfigManager::loadConfig(const std::string& config_path) {
    std::lock_guard<std::mutex> lock(config_mutex_);
    
    if (!fileExists(config_path)) {
        LOG_ERROR_FMT("Configuration file not found: {}", config_path);
        return false;
    }
    
    std::string format = detectConfigFormat(config_path);
    current_config_path_ = config_path;
    
    bool success = false;
    if (format == "yaml") {
        success = loadYAMLConfig(config_path);
    } else if (format == "ini") {
        success = loadINIConfig(config_path);
    } else {
        LOG_ERROR_FMT("Unsupported configuration format for file: {}", config_path);
        return false;
    }
    
    if (success) {
        LOG_INFO_FMT("Configuration loaded successfully from: {}", config_path);
        validateConfig();
    }
    
    return success;
}

bool ConfigManager::loadYAMLConfig(const std::string& yaml_path) {
#ifdef HAS_YAML_CPP
    try {
        config_ = YAML::LoadFile(yaml_path);
        return true;
    } catch (const YAML::Exception& e) {
        LOG_ERROR_FMT("Failed to load YAML config: {}", e.what());
        return false;
    }
#else
    LOG_WARN("YAML-cpp not available, falling back to INI parsing");
    return loadINIConfig(yaml_path);
#endif
}

bool ConfigManager::loadINIConfig(const std::string& ini_path) {
#ifdef HAS_YAML_CPP
    // If YAML is available, try to parse as simple key-value pairs
    try {
        std::ifstream file(ini_path);
        if (!file.is_open()) {
            return false;
        }
        
        YAML::Node node;
        std::string line;
        std::string current_section;
        
        while (std::getline(file, line)) {
            // Remove whitespace
            line.erase(0, line.find_first_not_of(" \t"));
            line.erase(line.find_last_not_of(" \t") + 1);
            
            // Skip empty lines and comments
            if (line.empty() || line[0] == '#' || line[0] == ';') {
                continue;
            }
            
            // Check for section header
            if (line[0] == '[' && line.back() == ']') {
                current_section = line.substr(1, line.length() - 2);
                continue;
            }
            
            // Parse key-value pair
            size_t eq_pos = line.find('=');
            if (eq_pos != std::string::npos) {
                std::string key = line.substr(0, eq_pos);
                std::string value = line.substr(eq_pos + 1);
                
                // Remove whitespace
                key.erase(0, key.find_first_not_of(" \t"));
                key.erase(key.find_last_not_of(" \t") + 1);
                value.erase(0, value.find_first_not_of(" \t"));
                value.erase(value.find_last_not_of(" \t") + 1);
                
                // Build full key with section
                std::string full_key = current_section.empty() ? key : current_section + "." + key;
                
                // Convert to YAML format
                auto keys = splitKey(full_key);
                YAML::Node current = config_;
                
                for (size_t i = 0; i < keys.size() - 1; ++i) {
                    current = current[keys[i]];
                }
                
                // Try to parse as different types
                if (value == "true" || value == "false") {
                    current[keys.back()] = (value == "true");
                } else {
                    try {
                        // Try integer
                        int int_val = std::stoi(value);
                        current[keys.back()] = int_val;
                    } catch (...) {
                        try {
                            // Try double
                            double double_val = std::stod(value);
                            current[keys.back()] = double_val;
                        } catch (...) {
                            // Default to string
                            current[keys.back()] = value;
                        }
                    }
                }
            }
        }
        
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR_FMT("Failed to parse INI config: {}", e.what());
        return false;
    }
#else
    return parseINIFile(ini_path);
#endif
}

std::string ConfigManager::getString(const std::string& key, const std::string& default_value) const {
    return get<std::string>(key, default_value);
}

int ConfigManager::getInt(const std::string& key, int default_value) const {
    return get<int>(key, default_value);
}

double ConfigManager::getDouble(const std::string& key, double default_value) const {
    return get<double>(key, default_value);
}

bool ConfigManager::getBool(const std::string& key, bool default_value) const {
    return get<bool>(key, default_value);
}

std::vector<std::string> ConfigManager::getStringArray(const std::string& key) const {
    std::lock_guard<std::mutex> lock(config_mutex_);
    
#ifdef HAS_YAML_CPP
    try {
        auto node = getNestedNode(key);
        if (!node || !node.IsSequence()) {
            return {};
        }
        
        std::vector<std::string> result;
        for (const auto& item : node) {
            result.push_back(item.as<std::string>());
        }
        return result;
    } catch (const YAML::Exception& e) {
        LOG_ERROR_FMT("Error reading array for key '{}': {}", key, e.what());
        return {};
    }
#else
    // Fallback: parse comma-separated values
    std::string value = getValue(key);
    if (value.empty()) {
        return {};
    }
    
    std::vector<std::string> result;
    std::stringstream ss(value);
    std::string item;
    
    while (std::getline(ss, item, ',')) {
        // Trim whitespace
        item.erase(0, item.find_first_not_of(" \t"));
        item.erase(item.find_last_not_of(" \t") + 1);
        if (!item.empty()) {
            result.push_back(item);
        }
    }
    
    return result;
#endif
}

bool ConfigManager::hasKey(const std::string& key) const {
    std::lock_guard<std::mutex> lock(config_mutex_);
    
#ifdef HAS_YAML_CPP
    try {
        auto node = getNestedNode(key);
        return static_cast<bool>(node);
    } catch (...) {
        return false;
    }
#else
    return config_map_.find(key) != config_map_.end();
#endif
}

bool ConfigManager::validateConfig() const {
    try {
        validateRequiredKeys();
        
        // Validate specific configurations
        auto pipeline_config = getPipelineConfig();
        if (!validatePipelineConfig(pipeline_config)) {
            LOG_ERROR("Pipeline configuration validation failed");
            return false;
        }
        
        auto triton_config = getTritonConfig();
        if (!validateTritonConfig(triton_config)) {
            LOG_ERROR("Triton configuration validation failed");
            return false;
        }
        
        LOG_INFO("Configuration validation passed");
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR_FMT("Configuration validation failed: {}", e.what());
        return false;
    }
}

void ConfigManager::watchForChanges(const std::string& config_path) {
    startFileWatcher(config_path);
}

void ConfigManager::setChangeCallback(std::function<void(const std::string&)> callback) {
    change_callback_ = callback;
}

ConfigManager::PipelineConfig ConfigManager::getPipelineConfig() const {
    PipelineConfig config;
    
    config.batch_size = getInt("pipeline.batch_size", config.batch_size);
    config.width = getInt("pipeline.width", config.width);
    config.height = getInt("pipeline.height", config.height);
    config.gpu_id = getInt("pipeline.gpu_id", config.gpu_id);
    config.num_surfaces = getInt("pipeline.num_surfaces", config.num_surfaces);
    config.enable_perf_measurement = getBool("pipeline.enable_perf_measurement", config.enable_perf_measurement);
    config.perf_measurement_interval_sec = getInt("pipeline.perf_measurement_interval_sec", config.perf_measurement_interval_sec);
    
    return config;
}

ConfigManager::SourceConfig ConfigManager::getSourceConfig(int source_id) const {
    SourceConfig config;
    std::string prefix = "source" + std::to_string(source_id);
    
    config.enable = getBool(prefix + ".enable", config.enable);
    config.type = getInt(prefix + ".type", config.type);
    config.uri = getString(prefix + ".uri", config.uri);
    config.gpu_id = getInt(prefix + ".gpu_id", config.gpu_id);
    config.cudadec_memtype = getInt(prefix + ".cudadec_memtype", config.cudadec_memtype);
    config.num_sources = getInt(prefix + ".num_sources", config.num_sources);
    
    return config;
}

ConfigManager::InferenceConfig ConfigManager::getPrimaryInferenceConfig() const {
    InferenceConfig config;
    std::string prefix = "primary_gie";
    
    config.enable = getBool(prefix + ".enable", config.enable);
    config.gpu_id = getInt(prefix + ".gpu_id", config.gpu_id);
    config.gie_unique_id = getInt(prefix + ".gie_unique_id", config.gie_unique_id);
    config.config_file = getString(prefix + ".config_file", config.config_file);
    config.model_engine_file = getString(prefix + ".model_engine_file", config.model_engine_file);
    config.batch_size = getInt(prefix + ".batch_size", config.batch_size);
    config.interval = getInt(prefix + ".interval", config.interval);
    
    return config;
}

ConfigManager::InferenceConfig ConfigManager::getSecondaryInferenceConfig(int sgie_id) const {
    InferenceConfig config;
    std::string prefix = "secondary_gie" + std::to_string(sgie_id);
    
    config.enable = getBool(prefix + ".enable", config.enable);
    config.gpu_id = getInt(prefix + ".gpu_id", config.gpu_id);
    config.gie_unique_id = getInt(prefix + ".gie_unique_id", config.gie_unique_id + sgie_id + 1);
    config.config_file = getString(prefix + ".config_file", config.config_file);
    config.model_engine_file = getString(prefix + ".model_engine_file", config.model_engine_file);
    config.batch_size = getInt(prefix + ".batch_size", config.batch_size);
    config.interval = getInt(prefix + ".interval", config.interval);
    
    return config;
}

ConfigManager::TritonConfig ConfigManager::getTritonConfig() const {
    TritonConfig config;
    std::string prefix = "triton";
    
    config.server_url = getString(prefix + ".server_url", config.server_url);
    config.model_repository = getString(prefix + ".model_repository", config.model_repository);
    config.max_batch_size = getInt(prefix + ".max_batch_size", config.max_batch_size);
    config.client_timeout_ms = getInt(prefix + ".client_timeout_ms", config.client_timeout_ms);
    config.enable_cuda_memory_pool = getBool(prefix + ".enable_cuda_memory_pool", config.enable_cuda_memory_pool);
    
    // Parse preferred batch sizes
    auto batch_sizes = getStringArray(prefix + ".preferred_batch_sizes");
    if (!batch_sizes.empty()) {
        config.preferred_batch_sizes.clear();
        for (const auto& size_str : batch_sizes) {
            try {
                config.preferred_batch_sizes.push_back(std::stoi(size_str));
            } catch (...) {
                LOG_WARN_FMT("Invalid batch size value: {}", size_str);
            }
        }
    }
    
    return config;
}

ConfigManager::TrackerConfig ConfigManager::getTrackerConfig() const {
    TrackerConfig config;
    std::string prefix = "tracker";
    
    config.enable = getBool(prefix + ".enable", config.enable);
    config.tracker_width = getInt(prefix + ".tracker_width", config.tracker_width);
    config.tracker_height = getInt(prefix + ".tracker_height", config.tracker_height);
    config.ll_lib_file = getString(prefix + ".ll_lib_file", config.ll_lib_file);
    config.ll_config_file = getString(prefix + ".ll_config_file", config.ll_config_file);
    config.gpu_id = getInt(prefix + ".gpu_id", config.gpu_id);
    config.display_tracking_id = getBool(prefix + ".display_tracking_id", config.display_tracking_id);
    
    return config;
}

bool ConfigManager::validatePipelineConfig(const PipelineConfig& config) const {
    if (config.batch_size <= 0 || config.batch_size > 32) {
        LOG_ERROR_FMT("Invalid batch_size: {} (must be 1-32)", config.batch_size);
        return false;
    }
    
    if (config.width <= 0 || config.height <= 0) {
        LOG_ERROR_FMT("Invalid dimensions: {}x{}", config.width, config.height);
        return false;
    }
    
    if (config.gpu_id < 0) {
        LOG_ERROR_FMT("Invalid gpu_id: {}", config.gpu_id);
        return false;
    }
    
    return true;
}

bool ConfigManager::validateSourceConfig(const SourceConfig& config) const {
    if (config.type < 1 || config.type > 6) {
        LOG_ERROR_FMT("Invalid source type: {} (must be 1-6)", config.type);
        return false;
    }
    
    if (config.uri.empty() && config.type != 1) { // USB camera doesn't need URI
        LOG_ERROR("Source URI is required for this source type");
        return false;
    }
    
    return true;
}

bool ConfigManager::validateInferenceConfig(const InferenceConfig& config) const {
    if (config.gie_unique_id <= 0) {
        LOG_ERROR_FMT("Invalid gie_unique_id: {}", config.gie_unique_id);
        return false;
    }
    
    if (config.batch_size <= 0) {
        LOG_ERROR_FMT("Invalid inference batch_size: {}", config.batch_size);
        return false;
    }
    
    return true;
}

bool ConfigManager::validateTritonConfig(const TritonConfig& config) const {
    if (config.server_url.empty()) {
        LOG_ERROR("Triton server URL is required");
        return false;
    }
    
    if (config.max_batch_size <= 0) {
        LOG_ERROR_FMT("Invalid max_batch_size: {}", config.max_batch_size);
        return false;
    }
    
    return true;
}

// Private methods
std::vector<std::string> ConfigManager::splitKey(const std::string& key) const {
    std::vector<std::string> keys;
    std::stringstream ss(key);
    std::string item;
    
    while (std::getline(ss, item, '.')) {
        if (!item.empty()) {
            keys.push_back(item);
        }
    }
    
    return keys;
}

bool ConfigManager::fileExists(const std::string& path) const {
    struct stat buffer;
    return (stat(path.c_str(), &buffer) == 0);
}

std::string ConfigManager::detectConfigFormat(const std::string& config_path) const {
    size_t dot_pos = config_path.find_last_of('.');
    if (dot_pos != std::string::npos) {
        std::string extension = config_path.substr(dot_pos + 1);
        std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
        
        if (extension == "yaml" || extension == "yml") {
            return "yaml";
        } else if (extension == "ini" || extension == "txt" || extension == "conf") {
            return "ini";
        }
    }
    
    // Default to INI format
    return "ini";
}

#ifdef HAS_YAML_CPP
YAML::Node ConfigManager::getNestedNode(const std::string& key) const {
    auto keys = splitKey(key);
    YAML::Node node = config_;
    
    for (const auto& k : keys) {
        if (!node[k]) {
            return YAML::Node();
        }
        node = node[k];
    }
    
    return node;
}
#else
bool ConfigManager::parseINIFile(const std::string& ini_path) {
    std::ifstream file(ini_path);
    if (!file.is_open()) {
        return false;
    }
    
    config_map_.clear();
    std::string line;
    std::string current_section;
    
    while (std::getline(file, line)) {
        // Remove whitespace
        line.erase(0, line.find_first_not_of(" \t"));
        line.erase(line.find_last_not_of(" \t") + 1);
        
        // Skip empty lines and comments
        if (line.empty() || line[0] == '#' || line[0] == ';') {
            continue;
        }
        
        // Check for section header
        if (line[0] == '[' && line.back() == ']') {
            current_section = line.substr(1, line.length() - 2);
            continue;
        }
        
        // Parse key-value pair
        size_t eq_pos = line.find('=');
        if (eq_pos != std::string::npos) {
            std::string key = line.substr(0, eq_pos);
            std::string value = line.substr(eq_pos + 1);
            
            // Remove whitespace
            key.erase(0, key.find_first_not_of(" \t"));
            key.erase(key.find_last_not_of(" \t") + 1);
            value.erase(0, value.find_first_not_of(" \t"));
            value.erase(value.find_last_not_of(" \t") + 1);
            
            // Build full key with section
            std::string full_key = current_section.empty() ? key : current_section + "." + key;
            config_map_[full_key] = value;
        }
    }
    
    return true;
}

std::string ConfigManager::getValue(const std::string& key) const {
    auto it = config_map_.find(key);
    return (it != config_map_.end()) ? it->second : "";
}
#endif

void ConfigManager::startFileWatcher(const std::string& file_path) {
    // Simple file modification time checking
    // In a production system, you might want to use inotify or similar
    struct stat file_stat;
    if (stat(file_path.c_str(), &file_stat) == 0) {
        last_modified_time_ = file_stat.st_mtime;
    }
}

bool ConfigManager::isFileModified(const std::string& file_path) const {
    struct stat file_stat;
    if (stat(file_path.c_str(), &file_stat) == 0) {
        return file_stat.st_mtime > last_modified_time_;
    }
    return false;
}

void ConfigManager::setDefaultValues() {
#ifdef HAS_YAML_CPP
    // Set some basic default values
    config_["pipeline"]["batch_size"] = 1;
    config_["pipeline"]["width"] = 1920;
    config_["pipeline"]["height"] = 1080;
    config_["pipeline"]["gpu_id"] = 0;
    config_["triton"]["server_url"] = "localhost:8001";
    config_["triton"]["max_batch_size"] = 8;
#else
    config_map_["pipeline.batch_size"] = "1";
    config_map_["pipeline.width"] = "1920";
    config_map_["pipeline.height"] = "1080";
    config_map_["pipeline.gpu_id"] = "0";
    config_map_["triton.server_url"] = "localhost:8001";
    config_map_["triton.max_batch_size"] = "8";
#endif
}

void ConfigManager::validateRequiredKeys() const {
    // Define required configuration keys
    std::vector<std::string> required_keys = {
        "pipeline.batch_size",
        "pipeline.width",
        "pipeline.height"
    };
    
    for (const auto& key : required_keys) {
        if (!hasKey(key)) {
            throw std::runtime_error("Required configuration key missing: " + key);
        }
    }
}

} // namespace VideoSummary