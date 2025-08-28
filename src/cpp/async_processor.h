#ifndef ASYNC_PROCESSOR_H
#define ASYNC_PROCESSOR_H

#include <functional>
#include <future>
#include <queue>
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <memory>
#include <chrono>

// Forward declarations
struct TensorData;
class TensorProcessor;

/**
 * @brief Task structure for async tensor processing
 */
struct AsyncTensorTask {
    void* tensor_meta;  // NvDsInferTensorMeta* cast as void* to avoid header conflicts
    int source_id;
    int batch_id;
    int frame_number;
    uint64_t timestamp;
    std::promise<bool> completion_promise;
    
    AsyncTensorTask(void* meta, int src_id, int b_id, int f_num, uint64_t ts)
        : tensor_meta(meta), source_id(src_id), batch_id(b_id), frame_number(f_num), timestamp(ts) {}
};

/**
 * @brief Performance statistics for async processing (copyable version)
 */
struct AsyncProcessingStats {
    uint64_t tasks_submitted{0};
    uint64_t tasks_completed{0};
    uint64_t tasks_failed{0};
    uint64_t total_processing_time_us{0};
    uint32_t current_queue_size{0};
    uint32_t max_queue_size{0};
    
    void reset() {
        tasks_submitted = 0;
        tasks_completed = 0;
        tasks_failed = 0;
        total_processing_time_us = 0;
        current_queue_size = 0;
        max_queue_size = 0;
    }
    
    double get_success_rate() const {
        return tasks_submitted > 0 ? (double)tasks_completed / tasks_submitted * 100.0 : 0.0;
    }
    
    double get_avg_processing_time_ms() const {
        return tasks_completed > 0 ? (double)total_processing_time_us / tasks_completed / 1000.0 : 0.0;
    }
};

/**
 * @brief Thread-safe statistics tracking (internal use)
 */
struct AsyncProcessingStatsAtomic {
    std::atomic<uint64_t> tasks_submitted{0};
    std::atomic<uint64_t> tasks_completed{0};
    std::atomic<uint64_t> tasks_failed{0};
    std::atomic<uint64_t> total_processing_time_us{0};
    std::atomic<uint32_t> current_queue_size{0};
    std::atomic<uint32_t> max_queue_size{0};
    
    void reset() {
        tasks_submitted = 0;
        tasks_completed = 0;
        tasks_failed = 0;
        total_processing_time_us = 0;
        current_queue_size = 0;
        max_queue_size = 0;
    }
    
    AsyncProcessingStats to_copyable() const {
        AsyncProcessingStats copy;
        copy.tasks_submitted = tasks_submitted.load();
        copy.tasks_completed = tasks_completed.load();
        copy.tasks_failed = tasks_failed.load();
        copy.total_processing_time_us = total_processing_time_us.load();
        copy.current_queue_size = current_queue_size.load();
        copy.max_queue_size = max_queue_size.load();
        return copy;
    }
};

/**
 * @brief Asynchronous tensor processing framework
 * 
 * Inspired by Python script's async processing approach, this class provides
 * a thread pool for non-blocking tensor processing to avoid pipeline stalls.
 */
class AsyncProcessor {
public:
    /**
     * @brief Constructor
     * @param num_threads Number of worker threads (default: auto-detect)
     * @param max_queue_size Maximum number of queued tasks (default: 1000)
     */
    explicit AsyncProcessor(size_t num_threads = 0, size_t max_queue_size = 1000);
    
    /**
     * @brief Destructor - ensures clean shutdown
     */
    ~AsyncProcessor();
    
    /**
     * @brief Initialize the async processor
     * @param tensor_processor Pointer to tensor processor instance
     * @return true if initialization successful
     */
    bool initialize(std::shared_ptr<TensorProcessor> tensor_processor);
    
    /**
     * @brief Submit tensor processing task asynchronously
     * @param tensor_meta Tensor metadata to process
     * @param source_id Source ID
     * @param batch_id Batch ID
     * @param frame_number Frame number
     * @param timestamp Frame timestamp
     * @return Future for task completion (true if successful)
     */
    std::future<bool> submit_tensor_task(void* tensor_meta,  // NvDsInferTensorMeta*
                                        int source_id,
                                        int batch_id,
                                        int frame_number,
                                        uint64_t timestamp = 0);
    
    /**
     * @brief Start the async processing threads
     * @return true if started successfully
     */
    bool start();
    
    /**
     * @brief Stop the async processing threads
     * @param timeout_ms Timeout for graceful shutdown (default: 5000ms)
     * @return true if stopped gracefully
     */
    bool stop(uint32_t timeout_ms = 5000);
    
    /**
     * @brief Check if processor is running
     * @return true if running
     */
    bool is_running() const { return running_.load(); }
    
    /**
     * @brief Get current processing statistics
     * @return Copy of current statistics
     */
    AsyncProcessingStats get_stats() const { return stats_.to_copyable(); }
    
    /**
     * @brief Reset processing statistics
     */
    void reset_stats() { stats_.reset(); }
    
    /**
     * @brief Configure processing options
     * @param enable_detailed_logging Enable detailed logging
     * @param enable_performance_tracking Enable performance tracking
     */
    void configure(bool enable_detailed_logging = false, 
                  bool enable_performance_tracking = true);
    
    /**
     * @brief Wait for all pending tasks to complete
     * @param timeout_ms Maximum time to wait in milliseconds
     * @return true if all tasks completed within timeout
     */
    bool wait_for_completion(uint32_t timeout_ms = 10000);

private:
    // Configuration
    size_t num_threads_;
    size_t max_queue_size_;
    bool enable_detailed_logging_;
    bool enable_performance_tracking_;
    
    // Thread management
    std::vector<std::thread> worker_threads_;
    std::atomic<bool> running_{false};
    std::atomic<bool> shutdown_requested_{false};
    
    // Task queue
    std::queue<std::unique_ptr<AsyncTensorTask>> task_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_condition_;
    std::condition_variable completion_condition_;
    
    // Processing components
    std::shared_ptr<TensorProcessor> tensor_processor_;
    
    // Statistics
    mutable AsyncProcessingStatsAtomic stats_;
    
    // Worker thread function
    void worker_thread_function();
    
    // Task processing
    bool process_tensor_task(AsyncTensorTask* task);
    
    // Utility functions
    uint64_t get_current_time_us();
    void log_task_info(const AsyncTensorTask* task, const std::string& message);
    void update_queue_stats();
};

#endif // ASYNC_PROCESSOR_H