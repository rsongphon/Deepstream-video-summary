#pragma once

#include <string>
#include <memory>
#include <sstream>

// Try to use spdlog if available, otherwise fallback to simple logging
#ifdef SPDLOG_VERSION
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/rotating_file_sink.h>
#define HAS_SPDLOG
#else
#include <iostream>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <mutex>
#endif

namespace VideoSummary {

enum class LogLevel {
    TRACE = 0,
    DEBUG = 1,
    INFO = 2,
    WARN = 3,
    ERROR = 4,
    CRITICAL = 5
};

class Logger {
public:
    static Logger& getInstance();
    
    // Configuration methods
    void setLevel(LogLevel level);
    void setConsoleLogging(bool enable);
    void setFileLogging(bool enable, const std::string& filename = "deepstream_video_summary.log");
    void setRotatingFileLogging(bool enable, const std::string& filename = "deepstream_video_summary.log", 
                               size_t max_size = 1048576 * 5, size_t max_files = 3);
    
    // Logging methods
    void trace(const std::string& message);
    void debug(const std::string& message);
    void info(const std::string& message);
    void warn(const std::string& message);
    void error(const std::string& message);
    void critical(const std::string& message);
    
    // Template methods for formatted logging
    template<typename... Args>
    void trace(const std::string& format, Args&&... args);
    
    template<typename... Args>
    void debug(const std::string& format, Args&&... args);
    
    template<typename... Args>
    void info(const std::string& format, Args&&... args);
    
    template<typename... Args>
    void warn(const std::string& format, Args&&... args);
    
    template<typename... Args>
    void error(const std::string& format, Args&&... args);
    
    template<typename... Args>
    void critical(const std::string& format, Args&&... args);
    
    // Performance metrics logging
    void logPipelineMetrics(double fps, double latency_ms, double gpu_utilization = -1.0);
    void logMemoryUsage(size_t gpu_memory_mb, size_t cpu_memory_mb);
    void logInferenceMetrics(const std::string& model_name, double inference_time_ms, size_t batch_size);

private:
    Logger();
    ~Logger() = default;
    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;
    
    void initializeLogger();
    void log(LogLevel level, const std::string& message);
    std::string formatMessage(LogLevel level, const std::string& message);
    std::string getCurrentTimestamp();
    std::string levelToString(LogLevel level);
    
#ifdef HAS_SPDLOG
    std::shared_ptr<spdlog::logger> spdlog_logger_;
#else
    // Fallback logging implementation
    LogLevel current_level_;
    bool console_logging_;
    bool file_logging_;
    std::string log_filename_;
    std::mutex log_mutex_;
    std::ofstream log_file_;
#endif
};

// Template implementations
template<typename... Args>
void Logger::trace(const std::string& format, Args&&... args) {
#ifdef HAS_SPDLOG
    spdlog_logger_->trace(format, std::forward<Args>(args)...);
#else
    std::ostringstream oss;
    ((oss << args << " "), ...);
    trace(format + " " + oss.str());
#endif
}

template<typename... Args>
void Logger::debug(const std::string& format, Args&&... args) {
#ifdef HAS_SPDLOG
    spdlog_logger_->debug(format, std::forward<Args>(args)...);
#else
    std::ostringstream oss;
    ((oss << args << " "), ...);
    debug(format + " " + oss.str());
#endif
}

template<typename... Args>
void Logger::info(const std::string& format, Args&&... args) {
#ifdef HAS_SPDLOG
    spdlog_logger_->info(format, std::forward<Args>(args)...);
#else
    std::ostringstream oss;
    ((oss << args << " "), ...);
    info(format + " " + oss.str());
#endif
}

template<typename... Args>
void Logger::warn(const std::string& format, Args&&... args) {
#ifdef HAS_SPDLOG
    spdlog_logger_->warn(format, std::forward<Args>(args)...);
#else
    std::ostringstream oss;
    ((oss << args << " "), ...);
    warn(format + " " + oss.str());
#endif
}

template<typename... Args>
void Logger::error(const std::string& format, Args&&... args) {
#ifdef HAS_SPDLOG
    spdlog_logger_->error(format, std::forward<Args>(args)...);
#else
    std::ostringstream oss;
    ((oss << args << " "), ...);
    error(format + " " + oss.str());
#endif
}

template<typename... Args>
void Logger::critical(const std::string& format, Args&&... args) {
#ifdef HAS_SPDLOG
    spdlog_logger_->critical(format, std::forward<Args>(args)...);
#else
    std::ostringstream oss;
    ((oss << args << " "), ...);
    critical(format + " " + oss.str());
#endif
}

} // namespace VideoSummary

// Convenience macros for logging
#define LOG_TRACE(msg) VideoSummary::Logger::getInstance().trace(msg)
#define LOG_DEBUG(msg) VideoSummary::Logger::getInstance().debug(msg)
#define LOG_INFO(msg) VideoSummary::Logger::getInstance().info(msg)
#define LOG_WARN(msg) VideoSummary::Logger::getInstance().warn(msg)
#define LOG_ERROR(msg) VideoSummary::Logger::getInstance().error(msg)
#define LOG_CRITICAL(msg) VideoSummary::Logger::getInstance().critical(msg)

// Formatted logging macros
#define LOG_TRACE_FMT(fmt, ...) VideoSummary::Logger::getInstance().trace(fmt, __VA_ARGS__)
#define LOG_DEBUG_FMT(fmt, ...) VideoSummary::Logger::getInstance().debug(fmt, __VA_ARGS__)
#define LOG_INFO_FMT(fmt, ...) VideoSummary::Logger::getInstance().info(fmt, __VA_ARGS__)
#define LOG_WARN_FMT(fmt, ...) VideoSummary::Logger::getInstance().warn(fmt, __VA_ARGS__)
#define LOG_ERROR_FMT(fmt, ...) VideoSummary::Logger::getInstance().error(fmt, __VA_ARGS__)
#define LOG_CRITICAL_FMT(fmt, ...) VideoSummary::Logger::getInstance().critical(fmt, __VA_ARGS__)