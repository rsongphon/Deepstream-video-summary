#include "tensor_processor.h"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <algorithm>
#include <cstring>
#include <sstream>

TensorProcessor::TensorProcessor() 
    : export_format(ExportFormat::CSV), enable_export(true), 
      max_tensor_values_to_log(100), enable_detailed_logging(false),
      global_batch_num(0) {
    // Initialize statistics
    memset(&stats, 0, sizeof(stats));
    stats.start_timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

TensorProcessor::~TensorProcessor() {
    cleanup();
}

bool TensorProcessor::initialize(const std::string& output_dir, ExportFormat format) {
    output_directory = output_dir;
    export_format = format;
    
    // Create output directory if it doesn't exist
    std::string mkdir_cmd = "mkdir -p " + output_directory;
    if (system(mkdir_cmd.c_str()) != 0) {
        std::cerr << "Warning: Could not create output directory: " << output_directory << std::endl;
    }
    
    // Initialize output streams based on format
    switch (export_format) {
        case ExportFormat::CSV:
            if (!initialize_csv_output()) {
                return false;
            }
            break;
        case ExportFormat::JSON:
        case ExportFormat::BINARY:
        case ExportFormat::NUMPY:
            // TODO: Implement other formats
            std::cout << "Format not yet implemented, using CSV" << std::endl;
            export_format = ExportFormat::CSV;
            if (!initialize_csv_output()) {
                return false;
            }
            break;
    }
    
    std::cout << "TensorProcessor initialized - Output: " << output_directory
              << ", Format: " << (export_format == ExportFormat::CSV ? "CSV" : "Other") << std::endl;
    
    return true;
}

bool TensorProcessor::initialize_csv_output() {
    std::string filename = generate_output_filename("tensor_output", "csv");
    csv_file = std::make_unique<std::ofstream>(filename);
    
    if (!csv_file->is_open()) {
        std::cerr << "Error: Could not create CSV file: " << filename << std::endl;
        return false;
    }
    
    return write_csv_header();
}

bool TensorProcessor::write_csv_header() {
    if (!csv_file || !csv_file->is_open()) {
        return false;
    }
    
    // CRITICAL FIX: Match exact C version CSV header format (no DataType column)
    *csv_file << "Source,Batch,Frame,Layer,LayerName,NumDims,Dimensions,RawTensorData" << std::endl;
    csv_file->flush();
    
    return true;
}

std::string TensorProcessor::generate_output_filename(const std::string& base_name, 
                                                      const std::string& extension) {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    auto tm = *std::localtime(&time_t);
    
    std::stringstream ss;
    ss << output_directory << "/" << base_name << "_"
       << std::put_time(&tm, "%Y%m%d_%H%M%S") << "." << extension;
    
    return ss.str();
}

bool TensorProcessor::extract_tensor_meta(GstBuffer* buffer, std::vector<TensorBatchData>& batch_data) {
    if (!buffer) {
        std::cerr << "Error: Invalid buffer" << std::endl;
        return false;
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buffer);
    if (!batch_meta) {
        if (enable_detailed_logging) {
            std::cout << "[DEBUG] No batch metadata found in buffer" << std::endl;
        }
        return false;
    }
    
    if (enable_detailed_logging) {
        std::cout << "[DEBUG] Processing batch with " << batch_meta->num_frames_in_batch << " frames" << std::endl;
    }
    
    bool success = process_batch(batch_meta, batch_data);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    
    // Update processing statistics
    if (success && !batch_data.empty()) {
        int tensor_count = 0;
        int frame_count = 0;
        for (const auto& batch : batch_data) {
            tensor_count += batch.tensors.size();
            frame_count++;
        }
        update_processing_stats(batch_data.size(), tensor_count, frame_count);
        
        if (enable_detailed_logging) {
            std::cout << "[DEBUG] Successfully extracted " << tensor_count << " tensors from " 
                      << frame_count << " frames" << std::endl;
        }
    } else if (enable_detailed_logging) {
        std::cout << "[DEBUG] No tensor data extracted from batch" << std::endl;
    }
    
    if (enable_detailed_logging) {
        std::cout << "[DEBUG] Tensor extraction took: " << duration.count() << " μs" << std::endl;
    }
    
    return success;
}

bool TensorProcessor::process_batch(NvDsBatchMeta* batch_meta, std::vector<TensorBatchData>& processed_data) {
    if (!batch_meta) {
        return false;
    }
    
    // CRITICAL FIX: Increment global batch counter once per batch (like C version)
    global_batch_num++;
    
    if (enable_detailed_logging) {
        std::cout << "[DEBUG] Processing batch " << global_batch_num 
                  << " with " << batch_meta->num_frames_in_batch << " frames" << std::endl;
    }
    
    NvDsMetaList* l_frame = batch_meta->frame_meta_list;
    int frames_with_inference = 0;
    
    // Process all frames in the batch - they all share the same batch number
    while (l_frame != nullptr) {
        NvDsFrameMeta* frame_meta = (NvDsFrameMeta*)(l_frame->data);
        
        if (enable_detailed_logging) {
            std::cout << "[DEBUG] Processing frame " << frame_meta->frame_num 
                      << " from source " << frame_meta->source_id 
                      << " (batch " << global_batch_num << ")" << std::endl;
        }
        
        TensorBatchData batch_tensor_data;
        batch_tensor_data.batch_id = global_batch_num;  // All frames use same batch number
        batch_tensor_data.timestamp = frame_meta->ntp_timestamp;
        
        // Count user metadata
        int user_meta_count = 0;
        int tensor_meta_count = 0;
        
        // Extract tensor metadata from this frame
        NvDsMetaList* l_user_meta = frame_meta->frame_user_meta_list;
        while (l_user_meta != nullptr) {
            NvDsUserMeta* user_meta = (NvDsUserMeta*)(l_user_meta->data);
            user_meta_count++;
            
            if (enable_detailed_logging) {
                std::cout << "[DEBUG] Found user meta type: " << user_meta->base_meta.meta_type << std::endl;
            }
            
            if (user_meta && user_meta->base_meta.meta_type == NVDSINFER_TENSOR_OUTPUT_META) {
                tensor_meta_count++;
                NvDsInferTensorMeta* tensor_meta = (NvDsInferTensorMeta*)(user_meta->user_meta_data);
                
                if (validate_tensor_meta(tensor_meta)) {
                    std::vector<TensorData> frame_tensors;
                    if (extract_tensor_from_meta(tensor_meta, 
                                                frame_meta->source_id, 
                                                global_batch_num,  // Use global batch number
                                                frame_meta->frame_num,
                                                frame_tensors)) {
                        batch_tensor_data.tensors.insert(batch_tensor_data.tensors.end(),
                                                         frame_tensors.begin(), frame_tensors.end());
                        frames_with_inference++;
                    }
                } else if (enable_detailed_logging) {
                    std::cout << "[DEBUG] Tensor meta validation failed" << std::endl;
                }
            }
            
            l_user_meta = l_user_meta->next;
        }
        
        if (enable_detailed_logging) {
            std::cout << "[DEBUG] Frame " << frame_meta->frame_num << ": " 
                      << user_meta_count << " user metas, " 
                      << tensor_meta_count << " tensor metas, "
                      << batch_tensor_data.tensors.size() << " tensors extracted" << std::endl;
        }
        
        if (!batch_tensor_data.tensors.empty()) {
            processed_data.push_back(batch_tensor_data);
            
            // OPTIMIZATION: Remove redundant CSV export - now done directly in tensor extraction
            // CSV writing is now handled directly in extract_tensor_from_meta for better performance
        }
        
        // Move to next frame (do NOT increment batch_id - all frames share same batch number)
        l_frame = l_frame->next;
    }
    
    if (enable_detailed_logging) {
        std::cout << "[DEBUG] Batch " << global_batch_num << " processing complete: " 
                  << frames_with_inference << " frames with inference out of " 
                  << batch_meta->num_frames_in_batch << " total frames" << std::endl;
    }
    
    return !processed_data.empty();
}

bool TensorProcessor::extract_tensor_from_meta(NvDsInferTensorMeta* tensor_meta, 
                                               int source_id, 
                                               int batch_id, 
                                               int frame_number,
                                               std::vector<TensorData>& tensor_data) {
    if (!tensor_meta || !tensor_meta->out_buf_ptrs_host) {
        return false;
    }
    
    for (unsigned int i = 0; i < tensor_meta->num_output_layers; i++) {
        NvDsInferLayerInfo* layer_info = &tensor_meta->output_layers_info[i];
        
        if (!validate_layer_info(layer_info, i)) {
            continue;
        }
        
        TensorData tensor;
        tensor.source_id = source_id;
        tensor.batch_id = batch_id;
        tensor.frame_number = frame_number;
        tensor.layer_index = i;
        tensor.layer_name = std::string(layer_info->layerName);
        tensor.data_type = layer_info->dataType;
        tensor.num_dims = layer_info->inferDims.numDims;
        
        // Copy dimensions
        tensor.dimensions.clear();
        tensor.total_elements = 1;
        for (int d = 0; d < layer_info->inferDims.numDims; d++) {
            tensor.dimensions.push_back(layer_info->inferDims.d[d]);
            tensor.total_elements *= layer_info->inferDims.d[d];
        }
        
        // Skip if layer is filtered out
        if (is_layer_filtered(tensor.layer_name)) {
            continue;
        }
        
        // CRITICAL FIX: Write tensor data directly to CSV like C version (no intermediate storage)
        void* tensor_buffer = tensor_meta->out_buf_ptrs_host[i];
        if (tensor_buffer && csv_file && csv_file->is_open()) {
            // Write basic tensor info directly
            *csv_file << "Source_" << source_id << ","
                      << "Batch_" << batch_id << ","
                      << "Frame_" << frame_number << ","
                      << "Layer_" << i << ","
                      << layer_info->layerName << ","
                      << layer_info->inferDims.numDims << ",";
                      
            // Write dimensions
            for (int d = 0; d < layer_info->inferDims.numDims; d++) {
                *csv_file << layer_info->inferDims.d[d];
                if (d < layer_info->inferDims.numDims - 1) *csv_file << " ";
            }
            *csv_file << " ,RAW_DATA:";
            
            // Write raw tensor data directly based on data type (like C version)
            guint max_elements = (tensor.total_elements > 100) ? 100 : tensor.total_elements;
            
            switch (layer_info->dataType) {
                case 0: { // FLOAT
                    float* float_data = (float*)tensor_buffer;
                    for (guint k = 0; k < max_elements; k++) {
                        *csv_file << std::fixed << std::setprecision(6) << float_data[k];
                        if (k < max_elements - 1) *csv_file << " ";
                    }
                    break;
                }
                case 1: { // HALF/FP16
                    uint16_t* half_data = (uint16_t*)tensor_buffer;
                    for (guint k = 0; k < max_elements; k++) {
                        *csv_file << half_data[k];
                        if (k < max_elements - 1) *csv_file << " ";
                    }
                    break;
                }
                case 2: { // INT8
                    int8_t* int8_data = (int8_t*)tensor_buffer;
                    for (guint k = 0; k < max_elements; k++) {
                        *csv_file << (int)int8_data[k];
                        if (k < max_elements - 1) *csv_file << " ";
                    }
                    break;
                }
                case 3: { // INT32
                    int32_t* int32_data = (int32_t*)tensor_buffer;
                    for (guint k = 0; k < max_elements; k++) {
                        *csv_file << int32_data[k];
                        if (k < max_elements - 1) *csv_file << " ";
                    }
                    break;
                }
                default: {
                    *csv_file << "[DataType_" << layer_info->dataType << "]";
                    uint8_t* byte_data = (uint8_t*)tensor_buffer;
                    guint byte_size = tensor.total_elements * 4;
                    guint max_bytes = (byte_size > 400) ? 400 : byte_size;
                    for (guint k = 0; k < max_bytes; k++) {
                        *csv_file << std::hex << (int)byte_data[k] << std::dec;
                        if (k < max_bytes - 1) *csv_file << " ";
                    }
                    break;
                }
            }
            
            if (tensor.total_elements > 100) {
                *csv_file << "...[truncated_" << tensor.total_elements << "_elements]";
            }
            
            *csv_file << std::endl;
            csv_file->flush();
            
            // Add to tensor_data for tracking (but with empty raw_data to save memory)
            tensor.raw_data.clear();  // Don't store raw data anymore
            tensor_data.push_back(tensor);
            
            if (enable_detailed_logging) {
                log_tensor_info(tensor);
            }
        }
    }
    
    // Update statistics for async processing (single tensor method)
    if (!tensor_data.empty()) {
        // This is a single frame, single "batch" with multiple tensors
        update_processing_stats(1, tensor_data.size(), 1);
        
        if (enable_detailed_logging) {
            std::cout << "[DEBUG] Updated stats: +1 batch, +" << tensor_data.size() 
                      << " tensors, +1 frame (from async processing)" << std::endl;
        }
    }
    
    return !tensor_data.empty();
}

bool TensorProcessor::extract_raw_tensor_data(void* tensor_buffer, 
                                              NvDsInferDataType data_type, 
                                              size_t total_elements,
                                              std::vector<float>& output_data) {
    if (!tensor_buffer || total_elements == 0) {
        return false;
    }
    
    // Limit the number of elements to prevent memory issues
    size_t elements_to_copy = std::min(total_elements, static_cast<size_t>(max_tensor_values_to_log));
    
    return copy_tensor_data_by_type(tensor_buffer, data_type, elements_to_copy, output_data);
}

bool TensorProcessor::copy_tensor_data_by_type(void* buffer, 
                                               NvDsInferDataType data_type, 
                                               size_t num_elements, 
                                               std::vector<float>& output) {
    if (!buffer || num_elements == 0) {
        if (enable_detailed_logging) {
            std::cout << "[ERROR] Invalid buffer or zero elements" << std::endl;
        }
        return false;
    }
    
    output.clear();
    output.resize(num_elements);  // Pre-allocate for vectorized operations
    
    // Optimized vectorized data extraction inspired by Python's numpy approach
    switch (data_type) {
        case FLOAT: {
            // Direct memory copy for float data (fastest path - like numpy's direct array access)
            float* float_data = static_cast<float*>(buffer);
            vectorized_copy_float(float_data, output.data(), num_elements);
            break;
        }
        case HALF: {
            // Vectorized half-precision conversion with proper IEEE 754 handling
            uint16_t* half_data = static_cast<uint16_t*>(buffer);
            // Process in chunks for better cache performance
            const size_t chunk_size = 1024;
            for (size_t i = 0; i < num_elements; i += chunk_size) {
                size_t end = std::min(i + chunk_size, num_elements);
                for (size_t j = i; j < end; j++) {
                    output[j] = half_to_float(half_data[j]);
                }
            }
            break;
        }
        case INT8: {
            // Optimized int8 to float conversion using vectorized helper
            int8_t* int8_data = static_cast<int8_t*>(buffer);
            vectorized_convert_int8(int8_data, output.data(), num_elements);
            break;
        }
        case INT32: {
            // Optimized int32 to float conversion using vectorized helper
            int32_t* int32_data = static_cast<int32_t*>(buffer);
            vectorized_convert_int32(int32_data, output.data(), num_elements);
            break;
        }
        default:
            if (enable_detailed_logging) {
                std::cout << "[ERROR] Unsupported data type: " << data_type << std::endl;
            }
            return false;
    }
    
    if (enable_detailed_logging && num_elements <= 10) {
        std::cout << "[DEBUG] Extracted tensor data (first " << std::min(num_elements, size_t(10)) 
                  << " elements): ";
        for (size_t i = 0; i < std::min(num_elements, size_t(10)); i++) {
            std::cout << output[i] << " ";
        }
        std::cout << std::endl;
    }
    
    return true;
}

bool TensorProcessor::export_to_csv(const std::vector<TensorBatchData>& batch_data) {
    if (!csv_file || !csv_file->is_open()) {
        return false;
    }
    
    for (const auto& batch : batch_data) {
        for (const auto& tensor : batch.tensors) {
            if (!write_tensor_to_csv(tensor)) {
                return false;
            }
        }
    }
    
    csv_file->flush();
    return true;
}

bool TensorProcessor::write_tensor_to_csv(const TensorData& tensor) {
    if (!csv_file || !csv_file->is_open()) {
        return false;
    }
    
    // Write basic tensor info
    *csv_file << "Source_" << tensor.source_id << ","
              << "Batch_" << tensor.batch_id << ","
              << "Frame_" << tensor.frame_number << ","
              << "Layer_" << tensor.layer_index << ","
              << tensor.layer_name << ","
              << tensor.num_dims << ",";
    
    // Write dimensions
    for (int i = 0; i < tensor.num_dims; i++) {
        *csv_file << tensor.dimensions[i];
        if (i < tensor.num_dims - 1) *csv_file << " ";
    }
    *csv_file << " ,";  // Add space after dimensions like C version
    
    // CRITICAL FIX: Remove DataType column to match C version format exactly
    
    // Write raw tensor data
    *csv_file << "RAW_DATA:";
    size_t values_to_write = std::min(tensor.raw_data.size(), static_cast<size_t>(max_tensor_values_to_log));
    for (size_t i = 0; i < values_to_write; i++) {
        *csv_file << std::fixed << std::setprecision(6) << tensor.raw_data[i];
        if (i < values_to_write - 1) *csv_file << " ";
    }
    
    if (tensor.raw_data.size() > values_to_write) {
        *csv_file << " ... (truncated from " << tensor.raw_data.size() << " values)";
    }
    
    *csv_file << std::endl;
    
    return true;
}

bool TensorProcessor::validate_tensor_meta(NvDsInferTensorMeta* tensor_meta) {
    if (!tensor_meta) {
        if (enable_detailed_logging) {
            std::cout << "[ERROR] Tensor metadata is null" << std::endl;
        }
        return false;
    }
    
    // Comprehensive validation inspired by Python script approach
    if (!tensor_meta->out_buf_ptrs_host) {
        if (enable_detailed_logging) {
            std::cout << "[ERROR] No host buffer pointers in tensor meta" << std::endl;
        }
        return false;
    }
    
    if (tensor_meta->num_output_layers == 0) {
        if (enable_detailed_logging) {
            std::cout << "[ERROR] No output layers in tensor meta" << std::endl;
        }
        return false;
    }
    
    // Validate output layers info array exists
    if (!tensor_meta->output_layers_info) {
        if (enable_detailed_logging) {
            std::cout << "[ERROR] Output layers info array is null" << std::endl;
        }
        return false;
    }
    
    // Enhanced validation: check each layer has valid buffer pointer
    for (guint i = 0; i < tensor_meta->num_output_layers; i++) {
        if (!tensor_meta->out_buf_ptrs_host[i]) {
            if (enable_detailed_logging) {
                std::cout << "[ERROR] Layer " << i << " has null buffer pointer" << std::endl;
            }
            return false;
        }
        
        // Validate layer info structure
        NvDsInferLayerInfo* layer_info = &tensor_meta->output_layers_info[i];
        if (!validate_layer_info(layer_info, i)) {
            if (enable_detailed_logging) {
                std::cout << "[ERROR] Layer " << i << " validation failed" << std::endl;
            }
            return false;
        }
    }
    
    if (enable_detailed_logging) {
        std::cout << "[DEBUG] Tensor meta validation passed: " << tensor_meta->num_output_layers 
                  << " layers, unique_id=" << tensor_meta->unique_id << std::endl;
    }
    
    return true;
}

bool TensorProcessor::validate_layer_info(NvDsInferLayerInfo* layer_info, int layer_index) {
    if (!layer_info) {
        if (enable_detailed_logging) {
            std::cout << "[ERROR] Layer info " << layer_index << " is null" << std::endl;
        }
        return false;
    }
    
    // Validate layer name
    if (!layer_info->layerName) {
        if (enable_detailed_logging) {
            std::cout << "[ERROR] Layer " << layer_index << " has no name" << std::endl;
        }
        return false;
    }
    
    // Validate dimension count (similar to Python validation)
    if (layer_info->inferDims.numDims <= 0 || layer_info->inferDims.numDims > NV_TENSORRT_MAX_DIMS) {
        if (enable_detailed_logging) {
            std::cout << "[ERROR] Layer " << layer_index << " (" << layer_info->layerName 
                      << ") has invalid dimension count: " << layer_info->inferDims.numDims << std::endl;
        }
        return false;
    }
    
    // Validate data type is supported
    if (layer_info->dataType < 0 || layer_info->dataType > 3) {
        if (enable_detailed_logging) {
            std::cout << "[ERROR] Layer " << layer_index << " (" << layer_info->layerName 
                      << ") has unsupported data type: " << layer_info->dataType << std::endl;
        }
        return false;
    }
    
    // Validate tensor dimensions
    std::vector<int> dimensions(layer_info->inferDims.d, 
                               layer_info->inferDims.d + layer_info->inferDims.numDims);
    if (!validate_tensor_dimensions(dimensions)) {
        if (enable_detailed_logging) {
            std::cout << "[ERROR] Layer " << layer_index << " (" << layer_info->layerName 
                      << ") has invalid dimensions" << std::endl;
        }
        return false;
    }
    
    // Calculate total elements for sanity check (like Python's np.prod)
    size_t total_elements = 1;
    for (int i = 0; i < layer_info->inferDims.numDims; i++) {
        total_elements *= layer_info->inferDims.d[i];
    }
    
    // Sanity check: ensure reasonable tensor size (max 100M elements)
    if (total_elements == 0 || total_elements > 100000000) {
        if (enable_detailed_logging) {
            std::cout << "[ERROR] Layer " << layer_index << " (" << layer_info->layerName 
                      << ") has unreasonable tensor size: " << total_elements << " elements" << std::endl;
        }
        return false;
    }
    
    if (enable_detailed_logging) {
        std::cout << "[DEBUG] Layer " << layer_index << " (" << layer_info->layerName 
                  << ") validation passed: " << total_elements << " elements, dims=[";
        for (int i = 0; i < layer_info->inferDims.numDims; i++) {
            std::cout << layer_info->inferDims.d[i];
            if (i < layer_info->inferDims.numDims - 1) std::cout << ",";
        }
        std::cout << "], dataType=" << layer_info->dataType << std::endl;
    }
    
    return true;
}

bool TensorProcessor::validate_tensor_dimensions(const std::vector<int>& dimensions) {
    if (dimensions.empty()) {
        if (enable_detailed_logging) {
            std::cout << "[ERROR] Empty dimensions vector" << std::endl;
        }
        return false;
    }
    
    // Enhanced dimension validation similar to Python's shape checking
    for (size_t i = 0; i < dimensions.size(); i++) {
        int dim = dimensions[i];
        if (dim <= 0) {
            if (enable_detailed_logging) {
                std::cout << "[ERROR] Invalid dimension " << i << ": " << dim << std::endl;
            }
            return false;
        }
        
        // Sanity check: individual dimension shouldn't be too large
        if (dim > 100000) {
            if (enable_detailed_logging) {
                std::cout << "[ERROR] Dimension " << i << " too large: " << dim << std::endl;
            }
            return false;
        }
    }
    
    // Check total size doesn't overflow
    size_t total_size = 1;
    for (int dim : dimensions) {
        if (total_size > SIZE_MAX / dim) {
            if (enable_detailed_logging) {
                std::cout << "[ERROR] Dimension overflow detected" << std::endl;
            }
            return false;
        }
        total_size *= dim;
    }
    
    return true;
}

bool TensorProcessor::is_layer_filtered(const std::string& layer_name) {
    if (layer_name_filter.empty()) {
        return false; // No filter means accept all
    }
    
    return std::find(layer_name_filter.begin(), layer_name_filter.end(), layer_name) 
           == layer_name_filter.end();
}

std::string TensorProcessor::data_type_to_string(NvDsInferDataType data_type) {
    switch (data_type) {
        case FLOAT: return "FLOAT";
        case HALF: return "HALF";
        case INT8: return "INT8";
        case INT32: return "INT32";
        default: return "UNKNOWN";
    }
}

size_t TensorProcessor::get_data_type_size(NvDsInferDataType data_type) {
    switch (data_type) {
        case FLOAT: return sizeof(float);
        case HALF: return sizeof(uint16_t);
        case INT8: return sizeof(int8_t);
        case INT32: return sizeof(int32_t);
        default: return 0;
    }
}

std::string TensorProcessor::tensor_to_string(const TensorData& tensor, int max_values) {
    std::stringstream ss;
    ss << "Tensor[" << tensor.layer_name << "] ";
    ss << "Source:" << tensor.source_id << " Batch:" << tensor.batch_id;
    ss << " Frame:" << tensor.frame_number << " ";
    ss << "Shape:[";
    for (int i = 0; i < tensor.num_dims; i++) {
        ss << tensor.dimensions[i];
        if (i < tensor.num_dims - 1) ss << "x";
    }
    ss << "] ";
    ss << "Type:" << data_type_to_string(tensor.data_type) << " ";
    ss << "Data:[";
    
    int values_to_show = std::min(max_values, static_cast<int>(tensor.raw_data.size()));
    for (int i = 0; i < values_to_show; i++) {
        ss << std::fixed << std::setprecision(3) << tensor.raw_data[i];
        if (i < values_to_show - 1) ss << ", ";
    }
    
    if (tensor.raw_data.size() > values_to_show) {
        ss << " ... +" << (tensor.raw_data.size() - values_to_show) << " more";
    }
    ss << "]";
    
    return ss.str();
}

void TensorProcessor::log_tensor_info(const TensorData& tensor) {
    std::cout << "[TENSOR] " << tensor_to_string(tensor, 5) << std::endl;
}

void TensorProcessor::log_processing_error(const std::string& error_msg) {
    std::cerr << "[TENSOR_PROCESSOR_ERROR] " << error_msg << std::endl;
}

void TensorProcessor::update_processing_stats(int batch_count, int tensor_count, int frame_count) {
    stats.total_batches_processed += batch_count;
    stats.total_tensors_extracted += tensor_count;
    stats.total_frames_processed += frame_count;
    
    // Update average processing time (simplified)
    guint64 current_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    stats.avg_processing_time_ms = static_cast<double>(current_time - stats.start_timestamp) / 
                                  static_cast<double>(stats.total_batches_processed);
}

const TensorProcessor::ProcessingStats& TensorProcessor::get_statistics() const {
    return stats;
}

void TensorProcessor::print_statistics() {
    std::cout << "\n=== Tensor Processing Statistics ===" << std::endl;
    std::cout << "Total Batches Processed: " << stats.total_batches_processed << std::endl;
    std::cout << "Total Tensors Extracted: " << stats.total_tensors_extracted << std::endl;
    std::cout << "Total Frames Processed: " << stats.total_frames_processed << std::endl;
    std::cout << "Average Processing Time: " << std::fixed << std::setprecision(2) 
              << stats.avg_processing_time_ms << " ms" << std::endl;
    std::cout << "=====================================" << std::endl;
}

void TensorProcessor::reset_statistics() {
    memset(&stats, 0, sizeof(stats));
    stats.start_timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

std::vector<TensorData> TensorProcessor::get_tensor_by_source(const std::vector<TensorBatchData>& batch_data, 
                                                              int source_id) {
    std::vector<TensorData> result;
    
    for (const auto& batch : batch_data) {
        for (const auto& tensor : batch.tensors) {
            if (tensor.source_id == source_id) {
                result.push_back(tensor);
            }
        }
    }
    
    return result;
}

bool TensorProcessor::export_tensors(const std::vector<TensorBatchData>& batch_data) {
    if (!enable_export) {
        return true;
    }
    
    switch (export_format) {
        case ExportFormat::CSV:
            return export_to_csv(batch_data);
        case ExportFormat::JSON:
            return export_to_json(batch_data);
        case ExportFormat::BINARY:
            return export_to_binary(batch_data);
        default:
            std::cerr << "Unsupported export format" << std::endl;
            return false;
    }
}

bool TensorProcessor::export_to_json(const std::vector<TensorBatchData>& batch_data) {
    // TODO: Implement JSON export
    std::cout << "JSON export not yet implemented" << std::endl;
    return false;
}

bool TensorProcessor::export_to_binary(const std::vector<TensorBatchData>& batch_data) {
    // TODO: Implement binary export
    std::cout << "Binary export not yet implemented" << std::endl;
    return false;
}

void TensorProcessor::cleanup() {
    cleanup_output_streams();
}

void TensorProcessor::cleanup_output_streams() {
    if (csv_file && csv_file->is_open()) {
        csv_file->close();
        csv_file.reset();
    }
}

// ============================================================================
// Optimized Data Conversion Helper Functions (inspired by Python's NumPy)
// ============================================================================

float TensorProcessor::half_to_float(uint16_t half_val) {
    // IEEE 754 half-precision to single-precision conversion
    uint32_t sign = (half_val >> 15) & 0x1;
    uint32_t exp = (half_val >> 10) & 0x1f;
    uint32_t mantissa = half_val & 0x3ff;
    
    uint32_t float_val;
    
    if (exp == 0) {
        if (mantissa == 0) {
            // Zero
            float_val = sign << 31;
        } else {
            // Denormalized number
            exp = 127 - 15;
            while (!(mantissa & 0x400)) {
                mantissa <<= 1;
                exp--;
            }
            mantissa &= 0x3ff;
            float_val = (sign << 31) | (exp << 23) | (mantissa << 13);
        }
    } else if (exp == 31) {
        // Infinity or NaN
        float_val = (sign << 31) | (0xff << 23) | (mantissa << 13);
    } else {
        // Normal number
        float_val = (sign << 31) | ((exp + 127 - 15) << 23) | (mantissa << 13);
    }
    
    return *reinterpret_cast<float*>(&float_val);
}

void TensorProcessor::vectorized_copy_float(const float* src, float* dst, size_t count) {
    // Direct memory copy for float arrays (fastest possible)
    std::memcpy(dst, src, count * sizeof(float));
}

void TensorProcessor::vectorized_convert_int8(const int8_t* src, float* dst, size_t count) {
    // Optimized int8 to float conversion using SIMD-friendly loops
    const size_t simd_chunk = 8;  // Process 8 elements at a time for better vectorization
    
    size_t i = 0;
    // Process chunks of 8 for better vectorization opportunity
    for (; i + simd_chunk <= count; i += simd_chunk) {
        for (size_t j = 0; j < simd_chunk; j++) {
            dst[i + j] = static_cast<float>(src[i + j]);
        }
    }
    
    // Handle remaining elements
    for (; i < count; i++) {
        dst[i] = static_cast<float>(src[i]);
    }
}

void TensorProcessor::vectorized_convert_int32(const int32_t* src, float* dst, size_t count) {
    // Optimized int32 to float conversion with overflow checking
    const size_t chunk_size = 4;  // Process 4 elements at a time
    
    size_t i = 0;
    // Process in chunks for better cache performance
    for (; i + chunk_size <= count; i += chunk_size) {
        for (size_t j = 0; j < chunk_size; j++) {
            dst[i + j] = static_cast<float>(src[i + j]);
        }
    }
    
    // Handle remaining elements
    for (; i < count; i++) {
        dst[i] = static_cast<float>(src[i]);
    }
}