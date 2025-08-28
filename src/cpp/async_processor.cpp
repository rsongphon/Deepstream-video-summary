#include "tensor_processor.h"
#include "async_processor.h"
#include <iostream>
#include <algorithm>
#include <cassert>

AsyncProcessor::AsyncProcessor(size_t num_threads, size_t max_queue_size)
    : num_threads_(num_threads > 0 ? num_threads : std::thread::hardware_concurrency())
    , max_queue_size_(max_queue_size)
    , enable_detailed_logging_(false)
    , enable_performance_tracking_(true) {
    
    // Ensure reasonable thread count
    if (num_threads_ == 0 || num_threads_ > 32) {
        num_threads_ = std::min(8u, std::thread::hardware_concurrency());
    }
    
    if (enable_detailed_logging_) {
        std::cout << "[AsyncProcessor] Initialized with " << num_threads_ 
                  << " threads, max queue size: " << max_queue_size_ << std::endl;
    }
}

AsyncProcessor::~AsyncProcessor() {
    if (is_running()) {
        stop(2000);  // 2 second timeout for destructor
    }
}

bool AsyncProcessor::initialize(std::shared_ptr<TensorProcessor> tensor_processor) {
    if (!tensor_processor) {
        std::cerr << "[AsyncProcessor] ERROR: Null tensor processor provided" << std::endl;
        return false;
    }
    
    tensor_processor_ = tensor_processor;
    
    if (enable_detailed_logging_) {
        std::cout << "[AsyncProcessor] Successfully initialized with tensor processor" << std::endl;
    }
    
    return true;
}

bool AsyncProcessor::start() {
    if (running_.load()) {
        if (enable_detailed_logging_) {
            std::cout << "[AsyncProcessor] Already running" << std::endl;
        }
        return true;
    }
    
    if (!tensor_processor_) {
        std::cerr << "[AsyncProcessor] ERROR: Cannot start without tensor processor" << std::endl;
        return false;
    }
    
    // Reset state
    shutdown_requested_ = false;
    running_ = true;
    stats_.reset();
    
    // Start worker threads
    worker_threads_.clear();
    worker_threads_.reserve(num_threads_);
    
    try {
        for (size_t i = 0; i < num_threads_; ++i) {
            worker_threads_.emplace_back(&AsyncProcessor::worker_thread_function, this);
        }
        
        if (enable_detailed_logging_) {
            std::cout << "[AsyncProcessor] Started " << num_threads_ << " worker threads" << std::endl;
        }
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "[AsyncProcessor] ERROR: Failed to start threads: " << e.what() << std::endl;
        running_ = false;
        return false;
    }
}

bool AsyncProcessor::stop(uint32_t timeout_ms) {
    if (!running_.load()) {
        return true;
    }
    
    if (enable_detailed_logging_) {
        std::cout << "[AsyncProcessor] Stopping with " << timeout_ms << "ms timeout..." << std::endl;
    }
    
    // Signal shutdown
    shutdown_requested_ = true;
    running_ = false;
    
    // Wake up all worker threads
    queue_condition_.notify_all();
    
    // Wait for threads to finish with timeout
    auto start_time = std::chrono::steady_clock::now();
    bool all_joined = true;
    
    for (auto& thread : worker_threads_) {
        if (thread.joinable()) {
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - start_time).count();
            
            if (elapsed >= timeout_ms) {
                all_joined = false;
                break;
            }
            
            thread.join();
        }
    }
    
    worker_threads_.clear();
    
    // Clear remaining tasks
    std::lock_guard<std::mutex> lock(queue_mutex_);
    while (!task_queue_.empty()) {
        auto task = std::move(task_queue_.front());
        task_queue_.pop();
        // Set promise to indicate failure due to shutdown
        task->completion_promise.set_value(false);
        stats_.tasks_failed++;
    }
    
    if (enable_detailed_logging_) {
        std::cout << "[AsyncProcessor] Stopped. Final stats - Submitted: " << stats_.tasks_submitted
                  << ", Completed: " << stats_.tasks_completed 
                  << ", Failed: " << stats_.tasks_failed << std::endl;
    }
    
    return all_joined;
}

std::future<bool> AsyncProcessor::submit_tensor_task(void* tensor_meta,
                                                     int source_id,
                                                     int batch_id,
                                                     int frame_number,
                                                     uint64_t timestamp) {
    if (!running_.load()) {
        // Return failed future
        std::promise<bool> promise;
        promise.set_value(false);
        return promise.get_future();
    }
    
    if (!tensor_meta) {
        std::promise<bool> promise;
        promise.set_value(false);
        return promise.get_future();
    }
    
    // Create task
    auto task = std::make_unique<AsyncTensorTask>(
        tensor_meta, source_id, batch_id, frame_number, 
        timestamp == 0 ? get_current_time_us() : timestamp);
    
    auto future = task->completion_promise.get_future();
    
    // Try to enqueue task
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        
        if (task_queue_.size() >= max_queue_size_) {
            if (enable_detailed_logging_) {
                std::cout << "[AsyncProcessor] Queue full, dropping task for source " 
                          << source_id << std::endl;
            }
            task->completion_promise.set_value(false);
            stats_.tasks_failed++;
            return future;
        }
        
        task_queue_.push(std::move(task));
        update_queue_stats();
    }
    
    // Notify worker threads
    queue_condition_.notify_one();
    stats_.tasks_submitted++;
    
    if (enable_detailed_logging_) {
        log_task_info(task_queue_.back().get(), "Task submitted");
    }
    
    return future;
}

bool AsyncProcessor::wait_for_completion(uint32_t timeout_ms) {
    auto start_time = std::chrono::steady_clock::now();
    
    std::unique_lock<std::mutex> lock(queue_mutex_);
    return completion_condition_.wait_for(lock, std::chrono::milliseconds(timeout_ms),
        [this] { return task_queue_.empty() || !running_.load(); });
}

void AsyncProcessor::configure(bool enable_detailed_logging, bool enable_performance_tracking) {
    enable_detailed_logging_ = enable_detailed_logging;
    enable_performance_tracking_ = enable_performance_tracking;
}

// Private methods implementation

void AsyncProcessor::worker_thread_function() {
    if (enable_detailed_logging_) {
        std::cout << "[AsyncProcessor] Worker thread " << std::this_thread::get_id() 
                  << " started" << std::endl;
    }
    
    while (running_.load() || !shutdown_requested_.load()) {
        std::unique_ptr<AsyncTensorTask> task;
        
        // Get task from queue
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            
            queue_condition_.wait(lock, [this] {
                return !task_queue_.empty() || shutdown_requested_.load();
            });
            
            if (shutdown_requested_.load() && task_queue_.empty()) {
                break;
            }
            
            if (!task_queue_.empty()) {
                task = std::move(task_queue_.front());
                task_queue_.pop();
                update_queue_stats();
            }
        }
        
        // Process task if we got one
        if (task) {
            bool success = process_tensor_task(task.get());
            task->completion_promise.set_value(success);
            
            if (success) {
                stats_.tasks_completed++;
            } else {
                stats_.tasks_failed++;
            }
            
            // Notify completion waiters
            completion_condition_.notify_all();
        }
    }
    
    if (enable_detailed_logging_) {
        std::cout << "[AsyncProcessor] Worker thread " << std::this_thread::get_id() 
                  << " finished" << std::endl;
    }
}

bool AsyncProcessor::process_tensor_task(AsyncTensorTask* task) {
    if (!task || !task->tensor_meta || !tensor_processor_) {
        return false;
    }
    
    auto start_time = enable_performance_tracking_ ? get_current_time_us() : 0;
    
    try {
        // Cast back to proper type for processing
        NvDsInferTensorMeta* tensor_meta = static_cast<NvDsInferTensorMeta*>(task->tensor_meta);
        
        // Extract tensors using the tensor processor
        std::vector<TensorData> frame_tensors;
        
        if (enable_detailed_logging_) {
            std::cout << "[AsyncProcessor] Processing tensor task - Source: " << task->source_id 
                      << ", Batch: " << task->batch_id << ", Frame: " << task->frame_number << std::endl;
        }
        
        bool success = tensor_processor_->extract_tensor_from_meta(
            tensor_meta,
            task->source_id,
            task->batch_id,
            task->frame_number,
            frame_tensors
        );
        
        if (enable_detailed_logging_) {
            std::cout << "[AsyncProcessor] Tensor extraction " << (success ? "succeeded" : "failed") 
                      << ", extracted " << frame_tensors.size() << " tensors" << std::endl;
        }
        
        if (enable_performance_tracking_) {
            auto processing_time = get_current_time_us() - start_time;
            stats_.total_processing_time_us += processing_time;
        }
        
        if (enable_detailed_logging_) {
            log_task_info(task, success ? "Task completed successfully" : "Task failed");
        }
        
        return success;
        
    } catch (const std::exception& e) {
        if (enable_detailed_logging_) {
            std::cout << "[AsyncProcessor] Exception processing task: " << e.what() << std::endl;
        }
        return false;
    }
}

uint64_t AsyncProcessor::get_current_time_us() {
    return std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

void AsyncProcessor::log_task_info(const AsyncTensorTask* task, const std::string& message) {
    if (task && enable_detailed_logging_) {
        std::cout << "[AsyncProcessor] " << message 
                  << " - Source: " << task->source_id
                  << ", Batch: " << task->batch_id
                  << ", Frame: " << task->frame_number << std::endl;
    }
}

void AsyncProcessor::update_queue_stats() {
    uint32_t current_size = static_cast<uint32_t>(task_queue_.size());
    stats_.current_queue_size = current_size;
    
    uint32_t max_size = stats_.max_queue_size.load();
    while (current_size > max_size && 
           !stats_.max_queue_size.compare_exchange_weak(max_size, current_size)) {
        // Keep trying until we successfully update or someone else updated with a higher value
    }
}