#include "Logger.h"
#include <stdexcept>

namespace VideoSummary {

Logger& Logger::getInstance() {
    static Logger instance;
    return instance;
}

Logger::Logger() {
    initializeLogger();
}

void Logger::initializeLogger() {
#ifdef HAS_SPDLOG
    try {
        // Create console and file sinks
        auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
        auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>("deepstream_video_summary.log", true);
        
        // Set patterns
        console_sink->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%l] [%n] %v");
        file_sink->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%l] [%n] %v");
        
        // Create logger with multiple sinks
        std::vector<spdlog::sink_ptr> sinks{console_sink, file_sink};
        spdlog_logger_ = std::make_shared<spdlog::logger>("video_summary", sinks.begin(), sinks.end());
        
        // Set default level to info
        spdlog_logger_->set_level(spdlog::level::info);
        spdlog_logger_->flush_on(spdlog::level::warn);
        
        // Register the logger
        spdlog::register_logger(spdlog_logger_);
        
    } catch (const spdlog::spdlog_ex& ex) {
        std::cerr << "Logger initialization failed: " << ex.what() << std::endl;
        // Fallback will be used
    }
#else
    // Initialize fallback logging
    current_level_ = LogLevel::INFO;
    console_logging_ = true;
    file_logging_ = true;
    log_filename_ = "deepstream_video_summary.log";
    
    if (file_logging_) {
        log_file_.open(log_filename_, std::ios::app);
        if (!log_file_.is_open()) {
            std::cerr << "Warning: Could not open log file: " << log_filename_ << std::endl;
            file_logging_ = false;
        }
    }
#endif
}

void Logger::setLevel(LogLevel level) {
#ifdef HAS_SPDLOG
    if (spdlog_logger_) {
        switch (level) {
            case LogLevel::TRACE:
                spdlog_logger_->set_level(spdlog::level::trace);
                break;
            case LogLevel::DEBUG:
                spdlog_logger_->set_level(spdlog::level::debug);
                break;
            case LogLevel::INFO:
                spdlog_logger_->set_level(spdlog::level::info);
                break;
            case LogLevel::WARN:
                spdlog_logger_->set_level(spdlog::level::warn);
                break;
            case LogLevel::ERROR:
                spdlog_logger_->set_level(spdlog::level::err);
                break;
            case LogLevel::CRITICAL:
                spdlog_logger_->set_level(spdlog::level::critical);
                break;
        }
    }
#else
    current_level_ = level;
#endif
}

void Logger::setConsoleLogging(bool enable) {
#ifdef HAS_SPDLOG
    // For spdlog, we'd need to recreate the logger with different sinks
    // This is a simplified implementation
    if (spdlog_logger_) {
        if (enable) {
            spdlog_logger_->info("Console logging enabled");
        } else {
            spdlog_logger_->info("Console logging disabled");
        }
    }
#else
    console_logging_ = enable;
#endif
}

void Logger::setFileLogging(bool enable, const std::string& filename) {
#ifdef HAS_SPDLOG
    // For spdlog, we'd need to recreate the logger with different sinks
    // This is a simplified implementation
    if (spdlog_logger_) {
        spdlog_logger_->info("File logging {} for file: {}", 
                           enable ? "enabled" : "disabled", filename);
    }
#else
    std::lock_guard<std::mutex> lock(log_mutex_);
    
    if (file_logging_ && log_file_.is_open()) {
        log_file_.close();
    }
    
    file_logging_ = enable;
    log_filename_ = filename;
    
    if (file_logging_) {
        log_file_.open(log_filename_, std::ios::app);
        if (!log_file_.is_open()) {
            std::cerr << "Warning: Could not open log file: " << log_filename_ << std::endl;
            file_logging_ = false;
        }
    }
#endif
}

void Logger::setRotatingFileLogging(bool enable, const std::string& filename, 
                                   size_t max_size, size_t max_files) {
#ifdef HAS_SPDLOG
    if (enable) {
        try {
            auto rotating_sink = std::make_shared<spdlog::sinks::rotating_file_sink_mt>(
                filename, max_size, max_files);
            rotating_sink->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%l] [%n] %v");
            
            // Add to existing sinks or replace
            if (spdlog_logger_) {
                spdlog_logger_->info("Rotating file logging enabled: {} (max_size: {}, max_files: {})", 
                                   filename, max_size, max_files);
            }
        } catch (const spdlog::spdlog_ex& ex) {
            if (spdlog_logger_) {
                spdlog_logger_->error("Failed to enable rotating file logging: {}", ex.what());
            }
        }
    }
#else
    // Fallback doesn't support rotating files, just use regular file logging
    // Suppress unused parameter warnings
    (void)max_size;
    (void)max_files;
    setFileLogging(enable, filename);
#endif
}

void Logger::trace(const std::string& message) {
#ifdef HAS_SPDLOG
    if (spdlog_logger_) {
        spdlog_logger_->trace(message);
    }
#else
    log(LogLevel::TRACE, message);
#endif
}

void Logger::debug(const std::string& message) {
#ifdef HAS_SPDLOG
    if (spdlog_logger_) {
        spdlog_logger_->debug(message);
    }
#else
    log(LogLevel::DEBUG, message);
#endif
}

void Logger::info(const std::string& message) {
#ifdef HAS_SPDLOG
    if (spdlog_logger_) {
        spdlog_logger_->info(message);
    }
#else
    log(LogLevel::INFO, message);
#endif
}

void Logger::warn(const std::string& message) {
#ifdef HAS_SPDLOG
    if (spdlog_logger_) {
        spdlog_logger_->warn(message);
    }
#else
    log(LogLevel::WARN, message);
#endif
}

void Logger::error(const std::string& message) {
#ifdef HAS_SPDLOG
    if (spdlog_logger_) {
        spdlog_logger_->error(message);
    }
#else
    log(LogLevel::ERROR, message);
#endif
}

void Logger::critical(const std::string& message) {
#ifdef HAS_SPDLOG
    if (spdlog_logger_) {
        spdlog_logger_->critical(message);
    }
#else
    log(LogLevel::CRITICAL, message);
#endif
}

void Logger::logPipelineMetrics(double fps, double latency_ms, double gpu_utilization) {
    std::ostringstream oss;
    oss << "Pipeline Metrics - FPS: " << std::fixed << std::setprecision(2) << fps
        << ", Latency: " << latency_ms << "ms";
    
    if (gpu_utilization >= 0.0) {
        oss << ", GPU Utilization: " << gpu_utilization << "%";
    }
    
    info(oss.str());
}

void Logger::logMemoryUsage(size_t gpu_memory_mb, size_t cpu_memory_mb) {
    std::ostringstream oss;
    oss << "Memory Usage - GPU: " << gpu_memory_mb << "MB, CPU: " << cpu_memory_mb << "MB";
    info(oss.str());
}

void Logger::logInferenceMetrics(const std::string& model_name, double inference_time_ms, size_t batch_size) {
    std::ostringstream oss;
    oss << "Inference Metrics - Model: " << model_name
        << ", Time: " << std::fixed << std::setprecision(2) << inference_time_ms << "ms"
        << ", Batch Size: " << batch_size;
    info(oss.str());
}

// Private methods
void Logger::log(LogLevel level, const std::string& message) {
#ifndef HAS_SPDLOG
    // Fallback logging implementation
    if (level < current_level_) {
        return;
    }
    
    std::string formatted_message = formatMessage(level, message);
    
    std::lock_guard<std::mutex> lock(log_mutex_);
    
    if (console_logging_) {
        std::cout << formatted_message << std::endl;
    }
    
    if (file_logging_ && log_file_.is_open()) {
        log_file_ << formatted_message << std::endl;
        log_file_.flush();
    }
#endif
}

std::string Logger::formatMessage(LogLevel level, const std::string& message) {
    std::ostringstream oss;
    oss << "[" << getCurrentTimestamp() << "] "
        << "[" << levelToString(level) << "] "
        << "[video_summary] " << message;
    return oss.str();
}

std::string Logger::getCurrentTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;
    
    std::ostringstream oss;
    oss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
    oss << "." << std::setfill('0') << std::setw(3) << ms.count();
    return oss.str();
}

std::string Logger::levelToString(LogLevel level) {
    switch (level) {
        case LogLevel::TRACE:    return "TRACE";
        case LogLevel::DEBUG:    return "DEBUG";
        case LogLevel::INFO:     return "INFO";
        case LogLevel::WARN:     return "WARN";
        case LogLevel::ERROR:    return "ERROR";
        case LogLevel::CRITICAL: return "CRITICAL";
        default:                 return "UNKNOWN";
    }
}

} // namespace VideoSummary