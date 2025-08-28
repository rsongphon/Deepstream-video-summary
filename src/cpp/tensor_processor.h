#ifndef TENSOR_PROCESSOR_H
#define TENSOR_PROCESSOR_H

#include <gst/gst.h>
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <fstream>

#include "gstnvdsmeta.h"
#include "gstnvdsinfer.h"
#include "nvdsmeta.h"
#include "nvbufsurface.h"
#include "nvdsinfer_context.h"

// Tensor data structure
struct TensorData {
    int source_id;
    int batch_id;
    int frame_number;
    int layer_index;
    std::string layer_name;
    int num_dims;
    std::vector<int> dimensions;
    NvDsInferDataType data_type;
    std::vector<float> raw_data;
    size_t total_elements;
};

struct TensorBatchData {
    int batch_id;
    std::vector<TensorData> tensors;
    guint64 timestamp;
};

// Export formats
enum class ExportFormat {
    CSV,
    JSON,
    BINARY,
    NUMPY
};

class TensorProcessor {
public:
    // Statistics tracking structure
    struct ProcessingStats {
        int total_batches_processed;
        int total_tensors_extracted;
        int total_frames_processed;
        double avg_processing_time_ms;
        guint64 start_timestamp;
    };

    TensorProcessor();
    ~TensorProcessor();

private:
    std::string output_directory;
    ExportFormat export_format;
    bool enable_export;
    
    // Output file streams
    std::unique_ptr<std::ofstream> csv_file;
    
    ProcessingStats stats;
    
    // Configuration
    int max_tensor_values_to_log;
    bool enable_detailed_logging;
    std::vector<std::string> layer_name_filter;
    
    // Batch tracking (like C version batch_num)
    guint global_batch_num;

public:
    
    // Initialization and configuration
    bool initialize(const std::string& output_dir = "output", 
                   ExportFormat format = ExportFormat::CSV);
    void cleanup();
    
    // Core processing functions
    bool extract_tensor_meta(GstBuffer* buffer, std::vector<TensorBatchData>& batch_data);
    bool process_batch(NvDsBatchMeta* batch_meta, std::vector<TensorBatchData>& processed_data);
    
    // Tensor data extraction
    bool extract_tensor_from_meta(NvDsInferTensorMeta* tensor_meta, 
                                  int source_id, 
                                  int batch_id, 
                                  int frame_number,
                                  std::vector<TensorData>& tensor_data);
    
    // Export functions
    bool export_tensors(const std::vector<TensorBatchData>& batch_data);
    bool export_to_csv(const std::vector<TensorBatchData>& batch_data);
    bool export_to_json(const std::vector<TensorBatchData>& batch_data);
    bool export_to_binary(const std::vector<TensorBatchData>& batch_data);
    
    // Tensor access by source
    std::vector<TensorData> get_tensor_by_source(const std::vector<TensorBatchData>& batch_data, 
                                                  int source_id);
    
    // Configuration setters
    void set_export_format(ExportFormat format) { export_format = format; }
    void set_max_tensor_values(int max_values) { max_tensor_values_to_log = max_values; }
    void set_detailed_logging(bool enable) { enable_detailed_logging = enable; }
    void set_layer_filter(const std::vector<std::string>& filters) { layer_name_filter = filters; }
    void enable_tensor_export(bool enable) { enable_export = enable; }
    
    // Statistics and monitoring
    const ProcessingStats& get_statistics() const;
    void print_statistics();
    void reset_statistics();
    
    // Utility functions
    static std::string data_type_to_string(NvDsInferDataType data_type);
    static size_t get_data_type_size(NvDsInferDataType data_type);
    static std::string tensor_to_string(const TensorData& tensor, int max_values = 10);
    
    // Data validation
    bool validate_tensor_meta(NvDsInferTensorMeta* tensor_meta);
    bool validate_layer_info(NvDsInferLayerInfo* layer_info, int layer_index);

private:
    // Internal processing helpers
    bool extract_raw_tensor_data(void* tensor_buffer, 
                                 NvDsInferDataType data_type, 
                                 size_t total_elements,
                                 std::vector<float>& output_data);
    
    bool copy_tensor_data_by_type(void* buffer, 
                                  NvDsInferDataType data_type, 
                                  size_t num_elements, 
                                  std::vector<float>& output);
    
    // File I/O helpers
    bool initialize_csv_output();
    bool write_csv_header();
    bool write_tensor_to_csv(const TensorData& tensor);
    
    std::string generate_output_filename(const std::string& base_name, 
                                         const std::string& extension);
    
    // Statistics helpers
    void update_processing_stats(int batch_count, int tensor_count, int frame_count);
    double calculate_processing_time();
    
    // Memory management
    void cleanup_output_streams();
    
    // Validation helpers
    bool is_layer_filtered(const std::string& layer_name);
    bool validate_tensor_dimensions(const std::vector<int>& dimensions);
    
    // Debug and logging helpers
    void log_tensor_info(const TensorData& tensor);
    void log_processing_error(const std::string& error_msg);
};

#endif // TENSOR_PROCESSOR_H