/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

/**
 * @file deepstream_multi_inference_app.c
 * @brief DeepStream Multi-Source Batched Inference Application
 *
 * This application demonstrates optimized multi-source processing with batched inference.
 * It accepts exactly 4 video sources, processes them simultaneously through a batched
 * inference pipeline, and outputs tensor data for each source with optional display.
 *
 * Key Features:
 * - Fixed 4-source input with automatic batching
 * - Hardware-accelerated video decoding
 * - TensorRT optimized inference
 * - Tensor extraction and output for each source
 * - Optional display output for visualization
 * - Maximum performance optimizations
 */

#include <gst/gst.h>
#include <glib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>
#include <cuda_runtime_api.h>
#include <stdlib.h>
#include <unistd.h>
#include <signal.h>

#include "gstnvdsmeta.h"
#include "nvds_yml_parser.h"
#include "gst-nvmessage.h"
#include "nvdsinfer.h"
#include "gstnvdsinfer.h"

/* Application constants */
#define MAX_SOURCES 4
#define MAX_DISPLAY_LEN 64
#define MUXER_OUTPUT_WIDTH 1920
#define MUXER_OUTPUT_HEIGHT 1080
#define MUXER_BATCH_TIMEOUT_USEC 40000
#define TILED_OUTPUT_WIDTH 1280
#define TILED_OUTPUT_HEIGHT 720

/* NVIDIA Decoder source pad memory feature */
#define GST_CAPS_FEATURES_NVMM "memory:NVMM"

/* Application configuration */
typedef struct {
    gboolean enable_display;
    gboolean enable_perf_measurement;
    gchar *config_file;
    gchar *model_config_file;
    gchar *source_uris[MAX_SOURCES];
    guint num_sources;
    guint gpu_id;
} AppConfig;

/* Application context */
typedef struct {
    GMainLoop *loop;
    GstElement *pipeline;
    GstElement *streammux;
    GstElement *pgie;
    GstElement *tiler;
    GstElement *nvvidconv;
    GstElement *nvosd;
    GstElement *sink;
    GstElement *tee;
    GstElement *queue_tensor;
    GstElement *fakesink_tensor;
    GstBus *bus;
    guint bus_watch_id;
    AppConfig config;
    
    /* Performance metrics */
    guint frame_count[MAX_SOURCES];
    gdouble total_fps;
    GTimer *timer;
    
    /* Tensor output data */
    guint batch_num;
    FILE *tensor_output_file;
} AppContext;

/* Global application context */
static AppContext *g_app_ctx = NULL;

/* Forward declarations */
static gboolean bus_call(GstBus *bus, GstMessage *msg, gpointer data);
static GstElement *create_source_bin(guint index, const gchar *uri);
static void cb_newpad(GstElement *decodebin, GstPad *decoder_src_pad, gpointer data);
static void decodebin_child_added(GstChildProxy *child_proxy, GObject *object, gchar *name, gpointer user_data);
static GstPadProbeReturn tensor_extract_probe(GstPad *pad, GstPadProbeInfo *info, gpointer user_data);
static GstPadProbeReturn osd_sink_pad_buffer_probe(GstPad *pad, GstPadProbeInfo *info, gpointer user_data);
static void print_usage(const char *prog_name);
static gboolean parse_args(int argc, char *argv[], AppConfig *config);
static gboolean setup_pipeline(AppContext *ctx);
static void cleanup_and_exit(AppContext *ctx);

/**
 * @brief Tensor extraction probe callback
 *
 * This function extracts tensor metadata from the inference output and
 * saves it for each of the 4 sources in the batch.
 */
static GstPadProbeReturn
tensor_extract_probe(GstPad *pad, GstPadProbeInfo *info, gpointer user_data)
{
    AppContext *ctx = (AppContext *)user_data;
    GstBuffer *buf = (GstBuffer *)info->data;
    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buf);
    NvDsMetaList *l_frame = NULL;
    NvDsMetaList *l_user = NULL;
    
    if (!batch_meta) {
        return GST_PAD_PROBE_OK;
    }
    
    g_print("=== Batch #%d - Tensor Extraction ===\n", ctx->batch_num++);
    
    /* Iterate through each frame in the batch */
    for (l_frame = batch_meta->frame_meta_list; l_frame != NULL; l_frame = l_frame->next) {
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)(l_frame->data);
        guint source_id = frame_meta->source_id;
        
        g_print("Source %d - Frame %d:\n", source_id, frame_meta->frame_num);
        
        /* Extract inference tensor metadata */
        for (l_user = frame_meta->frame_user_meta_list; l_user != NULL; l_user = l_user->next) {
            NvDsUserMeta *user_meta = (NvDsUserMeta *)l_user->data;
            
            if (user_meta->base_meta.meta_type == NVDSINFER_TENSOR_OUTPUT_META) {
                NvDsInferTensorMeta *tensor_meta = (NvDsInferTensorMeta *)user_meta->user_meta_data;
                
                g_print("  Tensor Output Layers: %d\n", tensor_meta->num_output_layers);
                
                /* Process each output layer */
                for (guint i = 0; i < tensor_meta->num_output_layers; i++) {
                    NvDsInferLayerInfo *layer_info = &tensor_meta->output_layers_info[i];
                    
                    g_print("    Layer %d: %s\n", i, layer_info->layerName);
                    g_print("      Data Type: %d\n", layer_info->inferDims.d[0]);
                    g_print("      Dimensions: ");
                    
                    for (guint j = 0; j < layer_info->inferDims.numDims; j++) {
                        g_print("%d ", layer_info->inferDims.d[j]);
                    }
                    g_print("\n");
                    
                    /* Extract tensor data for this source */
                    if (ctx->tensor_output_file) {
                        fprintf(ctx->tensor_output_file, 
                               "Source_%d,Batch_%d,Frame_%d,Layer_%d,%s,",
                               source_id, ctx->batch_num - 1, frame_meta->frame_num, i, 
                               layer_info->layerName);
                        
                        /* Write dimension info */
                        fprintf(ctx->tensor_output_file, "%d,", layer_info->inferDims.numDims);
                        for (guint j = 0; j < layer_info->inferDims.numDims; j++) {
                            fprintf(ctx->tensor_output_file, "%d ", layer_info->inferDims.d[j]);
                        }
                        fprintf(ctx->tensor_output_file, "\n");
                    }
                }
            }
        }
        
        /* Update frame count for performance tracking */
        if (source_id < MAX_SOURCES) {
            ctx->frame_count[source_id]++;
        }
    }
    
    /* Calculate and display performance metrics */
    if (ctx->batch_num % 30 == 0) { // Every 30 batches
        gdouble elapsed_time = g_timer_elapsed(ctx->timer, NULL);
        if (elapsed_time > 0) {
            ctx->total_fps = (ctx->batch_num * MAX_SOURCES) / elapsed_time;
            g_print("\n=== Performance Metrics ===\n");
            g_print("Total Batches: %d\n", ctx->batch_num);
            g_print("Average FPS per source: %.2f\n", ctx->total_fps / MAX_SOURCES);
            g_print("Total throughput: %.2f FPS\n", ctx->total_fps);
            
            for (guint i = 0; i < MAX_SOURCES; i++) {
                g_print("Source %d frames: %d\n", i, ctx->frame_count[i]);
            }
            g_print("===========================\n\n");
        }
    }
    
    return GST_PAD_PROBE_OK;
}

/**
 * @brief OSD sink pad buffer probe (for display mode)
 *
 * This probe is used when display is enabled to show inference results.
 */
static GstPadProbeReturn
osd_sink_pad_buffer_probe(GstPad *pad, GstPadProbeInfo *info, gpointer user_data)
{
    GstBuffer *buf = (GstBuffer *)info->data;
    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buf);
    NvDsMetaList *l_frame = NULL;
    NvDsMetaList *l_obj = NULL;
    
    if (!batch_meta) {
        return GST_PAD_PROBE_OK;
    }
    
    /* Process each frame for display overlay */
    for (l_frame = batch_meta->frame_meta_list; l_frame != NULL; l_frame = l_frame->next) {
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)(l_frame->data);
        guint obj_count = 0;
        
        /* Count objects for display */
        for (l_obj = frame_meta->obj_meta_list; l_obj != NULL; l_obj = l_obj->next) {
            obj_count++;
        }
        
        /* Add display metadata showing object count */
        NvDsDisplayMeta *display_meta = nvds_acquire_display_meta_from_pool(batch_meta);
        NvOSD_TextParams *txt_params = &display_meta->text_params[0];
        txt_params->display_text = g_malloc0(MAX_DISPLAY_LEN);
        
        g_snprintf(txt_params->display_text, MAX_DISPLAY_LEN,
                  "Source %d: Objects=%d", frame_meta->source_id, obj_count);
        
        /* Position text based on source ID */
        txt_params->x_offset = (frame_meta->source_id % 2) * (TILED_OUTPUT_WIDTH / 2) + 10;
        txt_params->y_offset = (frame_meta->source_id / 2) * (TILED_OUTPUT_HEIGHT / 2) + 30;
        
        /* Text styling */
        txt_params->font_params.font_name = "Serif";
        txt_params->font_params.font_size = 14;
        txt_params->font_params.font_color.red = 1.0;
        txt_params->font_params.font_color.green = 1.0;
        txt_params->font_params.font_color.blue = 0.0;
        txt_params->font_params.font_color.alpha = 1.0;
        
        /* Background */
        txt_params->set_bg_clr = 1;
        txt_params->text_bg_clr.red = 0.0;
        txt_params->text_bg_clr.green = 0.0;
        txt_params->text_bg_clr.blue = 0.0;
        txt_params->text_bg_clr.alpha = 0.7;
        
        display_meta->num_labels = 1;
        nvds_add_display_meta_to_frame(frame_meta, display_meta);
    }
    
    return GST_PAD_PROBE_OK;
}

/**
 * @brief Bus callback for handling pipeline messages
 */
static gboolean
bus_call(GstBus *bus, GstMessage *msg, gpointer data)
{
    GMainLoop *loop = (GMainLoop *)data;
    
    switch (GST_MESSAGE_TYPE(msg)) {
        case GST_MESSAGE_EOS:
            g_print("End of stream\n");
            g_main_loop_quit(loop);
            break;
            
        case GST_MESSAGE_WARNING: {
            gchar *debug = NULL;
            GError *error = NULL;
            gst_message_parse_warning(msg, &error, &debug);
            g_printerr("WARNING from element %s: %s\n",
                      GST_OBJECT_NAME(msg->src), error->message);
            g_free(debug);
            g_printerr("Warning: %s\n", error->message);
            g_error_free(error);
            break;
        }
        
        case GST_MESSAGE_ERROR: {
            gchar *debug = NULL;
            GError *error = NULL;
            gst_message_parse_error(msg, &error, &debug);
            g_printerr("ERROR from element %s: %s\n",
                      GST_OBJECT_NAME(msg->src), error->message);
            if (debug) {
                g_printerr("Error details: %s\n", debug);
            }
            g_free(debug);
            g_error_free(error);
            g_main_loop_quit(loop);
            break;
        }
        
        case GST_MESSAGE_ELEMENT: {
            if (gst_nvmessage_is_stream_eos(msg)) {
                guint stream_id = 0;
                if (gst_nvmessage_parse_stream_eos(msg, &stream_id)) {
                    g_print("Got EOS from stream %d\n", stream_id);
                }
            }
            break;
        }
        
        default:
            break;
    }
    
    return TRUE;
}

/**
 * @brief Callback for new pad creation in decodebin
 */
static void
cb_newpad(GstElement *decodebin, GstPad *decoder_src_pad, gpointer data)
{
    GstCaps *caps = gst_pad_get_current_caps(decoder_src_pad);
    if (!caps) {
        caps = gst_pad_query_caps(decoder_src_pad, NULL);
    }
    
    const GstStructure *str = gst_caps_get_structure(caps, 0);
    const gchar *name = gst_structure_get_name(str);
    GstElement *source_bin = (GstElement *)data;
    GstCapsFeatures *features = gst_caps_get_features(caps, 0);
    
    /* Check for video stream */
    if (!strncmp(name, "video", 5)) {
        /* Ensure hardware decoder is used */
        if (gst_caps_features_contains(features, GST_CAPS_FEATURES_NVMM)) {
            GstPad *bin_ghost_pad = gst_element_get_static_pad(source_bin, "src");
            if (!gst_ghost_pad_set_target(GST_GHOST_PAD(bin_ghost_pad), decoder_src_pad)) {
                g_printerr("Failed to link decoder src pad to source bin ghost pad\n");
            }
            gst_object_unref(bin_ghost_pad);
        } else {
            g_printerr("Error: Hardware decoder not selected. Please check your input format.\n");
        }
    }
    
    gst_caps_unref(caps);
}

/**
 * @brief Callback for decodebin child addition
 */
static void
decodebin_child_added(GstChildProxy *child_proxy, GObject *object, gchar *name, gpointer user_data)
{
    if (g_strrstr(name, "decodebin") == name) {
        g_signal_connect(G_OBJECT(object), "child-added",
                        G_CALLBACK(decodebin_child_added), user_data);
    }
    if (g_strrstr(name, "source") == name) {
        g_object_set(G_OBJECT(object), "drop-on-latency", TRUE, NULL);
    }
}

/**
 * @brief Create source bin for video input
 */
static GstElement *
create_source_bin(guint index, const gchar *uri)
{
    GstElement *bin = NULL, *uri_decode_bin = NULL;
    gchar bin_name[32] = {};
    
    g_snprintf(bin_name, 31, "source-bin-%02d", index);
    bin = gst_bin_new(bin_name);
    
    /* Use nvurisrcbin for optimal performance */
    uri_decode_bin = gst_element_factory_make("nvurisrcbin", "uri-decode-bin");
    if (!uri_decode_bin) {
        g_printerr("Failed to create nvurisrcbin, falling back to uridecodebin\n");
        uri_decode_bin = gst_element_factory_make("uridecodebin", "uri-decode-bin");
    } else {
        /* Configure nvurisrcbin for optimal performance */
        g_object_set(G_OBJECT(uri_decode_bin), "file-loop", TRUE, NULL);
        g_object_set(G_OBJECT(uri_decode_bin), "cudadec-memtype", 0, NULL);
    }
    
    if (!bin || !uri_decode_bin) {
        g_printerr("Failed to create source bin elements\n");
        return NULL;
    }
    
    g_object_set(G_OBJECT(uri_decode_bin), "uri", uri, NULL);
    
    /* Connect callbacks */
    g_signal_connect(G_OBJECT(uri_decode_bin), "pad-added",
                    G_CALLBACK(cb_newpad), bin);
    g_signal_connect(G_OBJECT(uri_decode_bin), "child-added",
                    G_CALLBACK(decodebin_child_added), bin);
    
    gst_bin_add(GST_BIN(bin), uri_decode_bin);
    
    /* Create ghost pad */
    if (!gst_element_add_pad(bin, gst_ghost_pad_new_no_target("src", GST_PAD_SRC))) {
        g_printerr("Failed to add ghost pad in source bin\n");
        return NULL;
    }
    
    return bin;
}

/**
 * @brief Setup the complete pipeline
 */
static gboolean
setup_pipeline(AppContext *ctx)
{
    GstElement *queue1, *queue2, *queue3, *queue4, *queue5;
    GstPad *tensor_probe_pad = NULL;
    GstPad *osd_sink_pad = NULL;
    
    /* Create pipeline */
    ctx->pipeline = gst_pipeline_new("multi-inference-pipeline");
    
    /* Create nvstreammux for batching */
    ctx->streammux = gst_element_factory_make("nvstreammux", "stream-muxer");
    
    /* Create primary inference engine */
    ctx->pgie = gst_element_factory_make("nvinfer", "primary-nvinference-engine");
    
    /* Create queues */
    queue1 = gst_element_factory_make("queue", "queue1");
    queue2 = gst_element_factory_make("queue", "queue2");
    queue3 = gst_element_factory_make("queue", "queue3");
    queue4 = gst_element_factory_make("queue", "queue4");
    queue5 = gst_element_factory_make("queue", "queue5");
    
    if (!ctx->pipeline || !ctx->streammux || !ctx->pgie || !queue1 || !queue2 || !queue3) {
        g_printerr("Failed to create basic pipeline elements\n");
        return FALSE;
    }
    
    /* Configure streammux for 4-source batching */
    g_object_set(G_OBJECT(ctx->streammux),
                "batch-size", MAX_SOURCES,
                "width", MUXER_OUTPUT_WIDTH,
                "height", MUXER_OUTPUT_HEIGHT,
                "batched-push-timeout", MUXER_BATCH_TIMEOUT_USEC,
                "nvbuf-memory-type", 2, /* Unified memory for optimal performance */
                NULL);
    
    /* Configure inference engine */
    g_object_set(G_OBJECT(ctx->pgie),
                "config-file-path", ctx->config.model_config_file ? ctx->config.model_config_file : "configs/multi_inference_pgie_config.txt",
                "batch-size", MAX_SOURCES,
                NULL);
    
    /* Add elements to pipeline */
    gst_bin_add_many(GST_BIN(ctx->pipeline), ctx->streammux, queue1, ctx->pgie, queue2, NULL);
    
    if (ctx->config.enable_display) {
        /* Create display pipeline elements */
        ctx->tee = gst_element_factory_make("tee", "tee");
        ctx->tiler = gst_element_factory_make("nvmultistreamtiler", "nvtiler");
        ctx->nvvidconv = gst_element_factory_make("nvvideoconvert", "nvvideo-converter");
        ctx->nvosd = gst_element_factory_make("nvdsosd", "nv-onscreendisplay");
        
        /* Determine appropriate sink */
        int current_device = -1;
        cudaGetDevice(&current_device);
        struct cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, current_device);
        
        if (prop.integrated) {
            ctx->sink = gst_element_factory_make("nv3dsink", "nv3d-sink");
        } else {
            ctx->sink = gst_element_factory_make("nveglglessink", "nvvideo-renderer");
        }
        
        ctx->queue_tensor = gst_element_factory_make("queue", "queue-tensor");
        ctx->fakesink_tensor = gst_element_factory_make("fakesink", "fakesink-tensor");
        
        if (!ctx->tee || !ctx->tiler || !ctx->nvvidconv || !ctx->nvosd || !ctx->sink ||
            !ctx->queue_tensor || !ctx->fakesink_tensor) {
            g_printerr("Failed to create display pipeline elements\n");
            return FALSE;
        }
        
        /* Configure tiler for 2x2 layout */
        g_object_set(G_OBJECT(ctx->tiler),
                    "rows", 2,
                    "columns", 2,
                    "width", TILED_OUTPUT_WIDTH,
                    "height", TILED_OUTPUT_HEIGHT,
                    NULL);
        
        /* Configure OSD */
        g_object_set(G_OBJECT(ctx->nvosd),
                    "process-mode", 1, /* GPU mode */
                    "display-text", 1,
                    NULL);
        
        /* Configure sink */
        g_object_set(G_OBJECT(ctx->sink), "qos", 0, NULL);
        
        /* Add display elements to pipeline */
        gst_bin_add_many(GST_BIN(ctx->pipeline), ctx->tee, ctx->tiler, queue3, ctx->nvvidconv,
                        queue4, ctx->nvosd, queue5, ctx->sink, ctx->queue_tensor, 
                        ctx->fakesink_tensor, NULL);
        
        /* Link elements */
        if (!gst_element_link_many(ctx->streammux, queue1, ctx->pgie, queue2, ctx->tee, NULL)) {
            g_printerr("Failed to link basic pipeline elements\n");
            return FALSE;
        }
        
        /* Link display branch */
        if (!gst_element_link_many(ctx->tee, ctx->tiler, queue3, ctx->nvvidconv, queue4,
                                  ctx->nvosd, queue5, ctx->sink, NULL)) {
            g_printerr("Failed to link display pipeline elements\n");
            return FALSE;
        }
        
        /* Link tensor extraction branch */
        if (!gst_element_link_many(ctx->tee, ctx->queue_tensor, ctx->fakesink_tensor, NULL)) {
            g_printerr("Failed to link tensor extraction branch\n");
            return FALSE;
        }
        
        /* Add OSD probe for display overlay */
        osd_sink_pad = gst_element_get_static_pad(ctx->nvosd, "sink");
        if (osd_sink_pad) {
            gst_pad_add_probe(osd_sink_pad, GST_PAD_PROBE_TYPE_BUFFER,
                             osd_sink_pad_buffer_probe, NULL, NULL);
            gst_object_unref(osd_sink_pad);
        }
        
    } else {
        /* Headless mode - direct to fakesink */
        ctx->fakesink_tensor = gst_element_factory_make("fakesink", "fakesink-tensor");
        
        if (!ctx->fakesink_tensor) {
            g_printerr("Failed to create tensor sink\n");
            return FALSE;
        }
        
        gst_bin_add(GST_BIN(ctx->pipeline), ctx->fakesink_tensor);
        
        /* Link elements */
        if (!gst_element_link_many(ctx->streammux, queue1, ctx->pgie, queue2, ctx->fakesink_tensor, NULL)) {
            g_printerr("Failed to link headless pipeline elements\n");
            return FALSE;
        }
    }
    
    /* Add tensor extraction probe */
    tensor_probe_pad = gst_element_get_static_pad(ctx->pgie, "src");
    if (tensor_probe_pad) {
        gst_pad_add_probe(tensor_probe_pad, GST_PAD_PROBE_TYPE_BUFFER,
                         tensor_extract_probe, ctx, NULL);
        gst_object_unref(tensor_probe_pad);
    } else {
        g_printerr("Failed to get PGIE src pad for tensor extraction\n");
        return FALSE;
    }
    
    /* Create and link source bins */
    for (guint i = 0; i < ctx->config.num_sources; i++) {
        GstElement *source_bin = create_source_bin(i, ctx->config.source_uris[i]);
        if (!source_bin) {
            g_printerr("Failed to create source bin %d\n", i);
            return FALSE;
        }
        
        gst_bin_add(GST_BIN(ctx->pipeline), source_bin);
        
        /* Link to streammux */
        gchar pad_name[16] = {};
        g_snprintf(pad_name, 15, "sink_%u", i);
        
        GstPad *sinkpad = gst_element_request_pad_simple(ctx->streammux, pad_name);
        GstPad *srcpad = gst_element_get_static_pad(source_bin, "src");
        
        if (gst_pad_link(srcpad, sinkpad) != GST_PAD_LINK_OK) {
            g_printerr("Failed to link source bin %d to stream muxer\n", i);
            gst_object_unref(srcpad);
            gst_object_unref(sinkpad);
            return FALSE;
        }
        
        gst_object_unref(srcpad);
        gst_object_unref(sinkpad);
    }
    
    return TRUE;
}

/**
 * @brief Print application usage
 */
static void
print_usage(const char *prog_name)
{
    g_print("Usage: %s [OPTIONS] <uri1> <uri2> <uri3> <uri4>\n", prog_name);
    g_print("Multi-Source Batched Inference Application for DeepStream\n\n");
    g_print("This application processes exactly 4 video sources simultaneously\n");
    g_print("through a batched inference pipeline and outputs tensor data.\n\n");
    g_print("Required Arguments:\n");
    g_print("  uri1-uri4          Four video source URIs (files, RTSP streams, etc.)\n\n");
    g_print("Options:\n");
    g_print("  --enable-display   Enable visual output (2x2 tiled display)\n");
    g_print("  --no-display      Disable display (headless mode, default)\n");
    g_print("  --config FILE     Use custom configuration file\n");
    g_print("  --model FILE      Use custom model configuration file\n");
    g_print("  --gpu-id ID       GPU device ID (default: 0)\n");
    g_print("  --perf            Enable performance measurement\n");
    g_print("  --help            Show this help message\n\n");
    g_print("Examples:\n");
    g_print("  %s video1.mp4 video2.mp4 video3.mp4 video4.mp4\n", prog_name);
    g_print("  %s --enable-display rtsp://cam1 rtsp://cam2 rtsp://cam3 rtsp://cam4\n", prog_name);
    g_print("  %s --config custom.txt --model custom_model.txt vid1.mp4 vid2.mp4 vid3.mp4 vid4.mp4\n", prog_name);
}

/**
 * @brief Parse command line arguments
 */
static gboolean
parse_args(int argc, char *argv[], AppConfig *config)
{
    int i = 1;
    int uri_count = 0;
    
    /* Initialize config with defaults */
    memset(config, 0, sizeof(AppConfig));
    config->enable_display = FALSE;
    config->enable_perf_measurement = FALSE;
    config->gpu_id = 0;
    
    while (i < argc) {
        if (g_str_equal(argv[i], "--enable-display")) {
            config->enable_display = TRUE;
        } else if (g_str_equal(argv[i], "--no-display")) {
            config->enable_display = FALSE;
        } else if (g_str_equal(argv[i], "--config")) {
            if (++i >= argc) {
                g_printerr("Error: --config requires a file argument\n");
                return FALSE;
            }
            config->config_file = g_strdup(argv[i]);
        } else if (g_str_equal(argv[i], "--model")) {
            if (++i >= argc) {
                g_printerr("Error: --model requires a file argument\n");
                return FALSE;
            }
            config->model_config_file = g_strdup(argv[i]);
        } else if (g_str_equal(argv[i], "--gpu-id")) {
            if (++i >= argc) {
                g_printerr("Error: --gpu-id requires an ID argument\n");
                return FALSE;
            }
            config->gpu_id = (guint)atoi(argv[i]);
        } else if (g_str_equal(argv[i], "--perf")) {
            config->enable_perf_measurement = TRUE;
        } else if (g_str_equal(argv[i], "--help") || g_str_equal(argv[i], "-h")) {
            print_usage(argv[0]);
            return FALSE;
        } else {
            /* Assume this is a URI */
            if (uri_count < MAX_SOURCES) {
                config->source_uris[uri_count] = g_strdup(argv[i]);
                uri_count++;
            } else {
                g_printerr("Error: Too many URIs provided. Maximum is %d\n", MAX_SOURCES);
                return FALSE;
            }
        }
        i++;
    }
    
    /* Validate we have exactly 4 sources */
    if (uri_count != MAX_SOURCES) {
        g_printerr("Error: Exactly %d video sources are required, got %d\n", MAX_SOURCES, uri_count);
        return FALSE;
    }
    
    config->num_sources = uri_count;
    return TRUE;
}

/**
 * @brief Cleanup and exit application
 */
static void
cleanup_and_exit(AppContext *ctx)
{
    if (ctx->timer) {
        g_timer_destroy(ctx->timer);
    }
    
    if (ctx->tensor_output_file) {
        fclose(ctx->tensor_output_file);
    }
    
    if (ctx->pipeline) {
        gst_element_set_state(ctx->pipeline, GST_STATE_NULL);
        gst_object_unref(GST_OBJECT(ctx->pipeline));
    }
    
    if (ctx->bus_watch_id) {
        g_source_remove(ctx->bus_watch_id);
    }
    
    if (ctx->loop) {
        g_main_loop_unref(ctx->loop);
    }
    
    /* Free config strings */
    for (guint i = 0; i < MAX_SOURCES; i++) {
        if (ctx->config.source_uris[i]) {
            g_free(ctx->config.source_uris[i]);
        }
    }
    
    if (ctx->config.config_file) {
        g_free(ctx->config.config_file);
    }
    
    if (ctx->config.model_config_file) {
        g_free(ctx->config.model_config_file);
    }
    
    g_free(ctx);
}

/**
 * @brief Signal handler for clean shutdown
 */
static void
signal_handler(int signum)
{
    g_print("\nReceived signal %d, shutting down gracefully...\n", signum);
    if (g_app_ctx && g_app_ctx->loop) {
        g_main_loop_quit(g_app_ctx->loop);
    }
}

/**
 * @brief Main application entry point
 */
int
main(int argc, char *argv[])
{
    AppContext *ctx;
    
    /* Initialize GStreamer */
    gst_init(&argc, &argv);
    
    /* Parse command line arguments */
    ctx = g_new0(AppContext, 1);
    g_app_ctx = ctx;
    
    if (!parse_args(argc, argv, &ctx->config)) {
        cleanup_and_exit(ctx);
        return -1;
    }
    
    g_print("=== DeepStream Multi-Source Batched Inference Application ===\n");
    g_print("Processing %d sources with %s mode\n", 
            ctx->config.num_sources, 
            ctx->config.enable_display ? "display" : "headless");
    
    /* Set up signal handlers */
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    /* Initialize performance tracking */
    ctx->timer = g_timer_new();
    ctx->batch_num = 0;
    for (guint i = 0; i < MAX_SOURCES; i++) {
        ctx->frame_count[i] = 0;
    }
    
    /* Open tensor output file */
    ctx->tensor_output_file = fopen("tensor_output.csv", "w");
    if (ctx->tensor_output_file) {
        fprintf(ctx->tensor_output_file, 
               "Source,Batch,Frame,Layer,LayerName,NumDims,Dimensions\n");
    }
    
    /* Create main loop */
    ctx->loop = g_main_loop_new(NULL, FALSE);
    
    /* Set up pipeline */
    if (!setup_pipeline(ctx)) {
        g_printerr("Failed to setup pipeline\n");
        cleanup_and_exit(ctx);
        return -1;
    }
    
    /* Set up bus */
    ctx->bus = gst_pipeline_get_bus(GST_PIPELINE(ctx->pipeline));
    ctx->bus_watch_id = gst_bus_add_watch(ctx->bus, bus_call, ctx->loop);
    gst_object_unref(ctx->bus);
    
    /* Print source information */
    g_print("\nInput Sources:\n");
    for (guint i = 0; i < ctx->config.num_sources; i++) {
        g_print("  Source %d: %s\n", i, ctx->config.source_uris[i]);
    }
    
    /* Start pipeline */
    g_print("\nStarting pipeline...\n");
    if (gst_element_set_state(ctx->pipeline, GST_STATE_PLAYING) == GST_STATE_CHANGE_FAILURE) {
        g_printerr("Failed to set pipeline to playing state\n");
        cleanup_and_exit(ctx);
        return -1;
    }
    
    g_print("Pipeline started. Processing batches...\n");
    g_print("Press Ctrl+C to stop.\n\n");
    
    /* Start timer */
    g_timer_start(ctx->timer);
    
    /* Run main loop */
    g_main_loop_run(ctx->loop);
    
    /* Cleanup */
    g_print("Stopping pipeline...\n");
    cleanup_and_exit(ctx);
    
    g_print("Application finished successfully.\n");
    return 0;
}