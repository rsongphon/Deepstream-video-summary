#ifndef PIPELINE_BUILDER_H
#define PIPELINE_BUILDER_H

#include <gst/gst.h>
#include <glib.h>
#include <string>
#include <vector>
#include <memory>

#include "gstnvdsmeta.h"
#include "nvdsmeta.h"
#include "nvbufsurface.h"

struct SourceConfig {
    int source_id;
    std::string uri;
    bool is_live;
    int framerate;
};

struct PipelineConfig {
    int batch_size;
    int width;
    int height;
    int gpu_id;
    std::string model_config_path;
    std::string model_engine_path;
    bool enable_display;
    bool enable_perf_measurement;
    int batched_push_timeout;
    int nvbuf_memory_type;
    std::vector<SourceConfig> sources;
};

class PipelineBuilder {
private:
    GstElement *pipeline;
    GstElement *streammux;
    GstElement *pgie;
    GstElement *nvvidconv;
    GstElement *nvosd;
    GstElement *tiler;
    GstElement *sink;
    GstElement *tee;
    GstElement *queue1;
    GstElement *queue2;
    
    GstBus *bus;
    guint bus_watch_id;
    
    PipelineConfig config;
    std::vector<GstElement*> source_bins;
    
    // Callback function type for tensor extraction
    typedef void (*TensorExtractCallback)(GstPad *pad, GstPadProbeInfo *info, gpointer user_data);
    TensorExtractCallback tensor_callback;
    gpointer callback_user_data;

public:
    PipelineBuilder();
    ~PipelineBuilder();
    
    // Configuration
    bool initialize(const PipelineConfig& config);
    void cleanup();
    
    // Pipeline building
    bool create_pipeline();
    bool create_source_bin(const SourceConfig& source_config);
    bool setup_streammux();
    bool setup_inference();
    bool setup_display_branch();
    bool setup_tensor_extraction();
    bool link_pipeline_components();
    
    // Pipeline control
    bool start_pipeline();
    bool stop_pipeline();
    bool pause_pipeline();
    GstStateChangeReturn get_pipeline_state();
    
    // Tensor extraction
    void set_tensor_extraction_callback(TensorExtractCallback callback, gpointer user_data);
    
    // Utility functions
    static gboolean bus_call(GstBus *bus, GstMessage *msg, gpointer data);
    static GstPadProbeReturn tensor_extract_probe(GstPad *pad, GstPadProbeInfo *info, gpointer user_data);
    static void qtdemux_pad_added_callback(GstElement* src, GstPad* new_pad, gpointer data);
    
    // Configuration getters
    const PipelineConfig& get_config() const { return config; }
    int get_source_count() const { return source_bins.size(); }
    
    // Performance monitoring
    void enable_performance_monitoring(bool enable);
    void print_performance_stats();

private:
    // Internal helper functions
    GstElement* create_nvinfer_element();
    GstElement* create_display_elements();
    bool configure_streammux_properties();
    bool add_probe_to_element(GstElement* element, const std::string& pad_name);
    void print_pipeline_info();
    
    // Memory management helpers
    void unref_elements();
    void cleanup_source_bins();
    
    // Error handling
    void handle_pipeline_error(const std::string& error_msg);
    bool check_element_creation(GstElement* element, const std::string& element_name);
};

#endif // PIPELINE_BUILDER_H