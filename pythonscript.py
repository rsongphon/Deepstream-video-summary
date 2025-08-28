
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
import os
os.environ['GRPC_ENABLE_FORK_SUPPORT'] = '0'
os.environ['GRPC_POLL_STRATEGY'] = 'epoll1'

if 'XDG_RUNTIME_DIR' not in os.environ:
    os.environ['XDG_RUNTIME_DIR'] = '/tmp/runtime-root'

import sys
import time
import gi
import ctypes
import asyncio
import threading
import signal
import gc
import logging
from datetime import datetime

gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import cv2
import numpy as np
import pyds
import urllib
from ainuvision_processor import AinuVisionProcessor
# from alerts.video_manager import vid_manager
from config import SEARCH_INDEX_DATA_DIR, VIDEO_RESOLUTION_CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

global_pipeline = None
global_rtsp_urls_dict = {}
global_ainuvision_client = None

# Add this global variable at the top of your file
source_id_to_camera_id = {}

def signal_handler(sig, frame):
    print(f"Signal {sig} received, shutting down gracefully...")
    if global_pipeline:
        print("Setting pipeline to NULL state for clean shutdown...")
        global_pipeline.set_state(Gst.State.NULL)
        global_pipeline.get_state(Gst.CLOCK_TIME_NONE)
        print("Pipeline successfully set to NULL state")
    else:
        print("No active pipeline to shut down")
    if global_ainuvision_client:
        if loop.is_running():
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            new_loop.run_until_complete(global_ainuvision_client.close())
            new_loop.close()
        else:
            loop.run_until_complete(global_ainuvision_client.close())
        # Cancel all tasks and clean up
        pending = asyncio.all_tasks(loop)
        for task in pending:
            task.cancel()
        loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        loop.close()
    sys.exit(0)

signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

# --- EXACT SAME as reference: Asyncio event loop (global) ---
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)
for task in asyncio.all_tasks(loop):
    task.cancel()
    try:
        loop.run_until_complete(task)
    except asyncio.CancelledError:
        pass

frame_counters = {}
is_manual_rtsp = False

from brands.factory import BrandFactory
async def init_nvr(nvr_brand: str, ip: str, username: str, password: str, port: int) -> bool:
    """Initialize NVR connection and sync time"""
    global nvr_system
    global is_manual_rtsp

    if nvr_brand == 'manual_rtsp':
        is_manual_rtsp = True
        return True

    try:
        components = await BrandFactory.initialize_brand(
            brand=nvr_brand,
            ip_addresses=[ip],
            username=username,
            password=password,
            port=port
        )
        
        nvr_system = components['system']
        
        if not await nvr_system.sync_time():
            print("Time sync failed, but continuing with initialization")
        
        return True
            
    except Exception as e:
        print(f"Error initializing NVR: {e}")
        return False

# --- Helper: Load last frame number ---
def load_last_frame_number(camera_id, date):
    metadata_file = os.path.join(SEARCH_INDEX_DATA_DIR, str(date), str(camera_id), f"metadata_{date}_{camera_id}.txt")
    if os.path.exists(metadata_file):
        try:
            with open(metadata_file, 'r') as f:
                lines = f.readlines()
                if lines:
                    last_line = lines[-1].strip()
                    _, last_frame, _ = last_line.split(',')
                    return int(last_frame)
        except Exception as e:
            print(f"Error reading metadata for camera {camera_id}: {e}")
    return 0

# --- Async: Frame feature sending ---
async def process_frame(camera_id, tensor_data, frame_number, datetime_str, is_new_day, frame_data=None):
    global is_manual_rtsp
    if is_manual_rtsp:
        frame_number = -1  # To inform gRPC server
    return await global_ainuvision_client.process_feature(
        camera_id, tensor_data, frame_number, datetime_str, is_new_day, frame_data
    )
# --- Updated frame callback to handle NV12 format ---
def postprocess_frame_callback(pad, info, user_data):
    global source_id_to_camera_id

    buffer = info.get_buffer()
    if not buffer:
        print("Unable to get GstBuffer")
        return Gst.PadProbeReturn.OK

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buffer))
    l_frame = batch_meta.frame_meta_list

    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        mux_source_id = frame_meta.source_id
        camera_id = source_id_to_camera_id.get(mux_source_id)

        if camera_id is None:
            print(f"ERROR: No mapping found for source_id {mux_source_id}")
            try:
                l_frame = l_frame.next
            except StopIteration:
                break
            continue

        # print(f"Processing frame: mux_source_id={mux_source_id} -> camera_id={camera_id}")

        current_datetime = datetime.now()
        current_date = current_datetime.date()

        # Day/frame counting logic (UNCHANGED)
        if camera_id not in user_data:
            user_data[camera_id] = {
                'current_date': current_date,
                'frame_count': frame_counters.get(camera_id, 0)
            }
            is_new_day = False
        else:
            is_new_day = current_date != user_data[camera_id]['current_date']

        if is_new_day:
            user_data[camera_id]['current_date'] = current_date
            user_data[camera_id]['frame_count'] = 0
            print(f"New day started for camera {camera_id}, resetting frame count")

        frame_number = user_data[camera_id]['frame_count']
        user_data[camera_id]['frame_count'] += 1

        # --- Frame Data Extraction (UNCHANGED - Keep 1280x720 for high-quality crops) ---
        frame_data = None
        try:
            n_frame = pyds.get_nvds_buf_surface(hash(buffer), frame_meta.batch_id)
            if n_frame is not None:
                frame_copy = np.array(n_frame, copy=True, order='C')
                # print(f"Frame shape from pyds: {frame_copy.shape}")
                if len(frame_copy.shape) == 3 and frame_copy.shape[2] == 4:  # RGBA
                    frame_bgr = cv2.cvtColor(frame_copy, cv2.COLOR_RGBA2BGR)
                    frame_data = frame_bgr.tobytes()
                # elif len(frame_copy.shape) == 3 and frame_copy.shape[2] == 3:  # BGR
                #     frame_data = frame_copy.tobytes()
                # else:
                #     print(f"Unexpected frame shape: {frame_copy.shape}")
                #     frame_data = None
            else:
                print("pyds.get_nvds_buf_surface returned None")
                frame_data = None
        except Exception as e:
            print(f"Frame extraction failed for camera {camera_id}: {e}")
            frame_data = None

        # MODIFIED: Add frame to gRPC client circular buffer for every frame
        current_datetime_str = current_datetime.strftime("%d_%m_%Y-%H:%M:%S")
        # if frame_data is not None and len(frame_data) == VIDEO_RESOLUTION_CONFIG['RESOLUTION_W'] * VIDEO_RESOLUTION_CONFIG['RESOLUTION_H'] * 3:
        #     # Original video manager (UNCHANGED)
        #     # vid_manager.add_frame(camera_id, frame_data)
            
        #     # NEW: Add every frame to gRPC client circular buffer (regardless of inference interval)
        #     global_ainuvision_client.add_frame_to_buffer(camera_id, frame_data, frame_number, current_datetime_str)
        # else:
        #     print(f"Invalid frame data size for video manager: {len(frame_data) if frame_data else 0}")

        # --- Handle all model outputs ---
        l_user = frame_meta.frame_user_meta_list
        while l_user is not None:
            try:
                user_meta = pyds.NvDsUserMeta.cast(l_user.data)
            except StopIteration:
                break

            if user_meta.base_meta.meta_type == pyds.NVDSINFER_TENSOR_OUTPUT_META:
                tensor_meta = pyds.NvDsInferTensorMeta.cast(user_meta.user_meta_data)

                # OMD branch (MODIFIED - Direct processing in gRPC client)
                if tensor_meta.unique_id == 1:  # OMD branch
                    feature_list = []
                    try:
                        if tensor_meta.num_output_layers != 3:
                            print(f"ERROR: Expected 3 output layers from OMD model, got {tensor_meta.num_output_layers}")
                            continue
                        expected_shapes = [
                            (1, 192, 80, 80),
                            (1, 384, 40, 40),
                            (1, 768, 20, 20),
                        ]
                        expected_elements = [np.prod(shape) for shape in expected_shapes]
                        features = []
                        for i, num_e in enumerate(expected_elements):
                            layer = pyds.get_nvds_LayerInfo(tensor_meta, i)
                            ptr = ctypes.cast(pyds.get_ptr(layer.buffer), ctypes.POINTER(ctypes.c_float))
                            # features += [ptr[j] for j in range(num_e)] 
                            # Extract directly as numpy array - MUCH faster than [ptr[j] for j in range(num_e)]
                            layer_array = np.ctypeslib.as_array(ptr, shape=(num_e,)).copy()
                            features.append(layer_array)
                        if len(features) > 0:
                            # MODIFIED: Direct processing - no more shared memory writes
                            # Send to gRPC client for direct alert and search processing
                            future = asyncio.run_coroutine_threadsafe(
                                process_frame(camera_id, features, frame_number, current_datetime_str, is_new_day, frame_data),
                                loop
                            )
                            def handle_future(fut):
                                try:
                                    fut.result()
                                except Exception as e:
                                    print(f"ERROR in process_frame for camera {camera_id}: {e}")
                            future.add_done_callback(handle_future)
                    except Exception as e:
                        print(f"Error extracting OMD tensor data for camera {camera_id}: {e}")


            try:
                l_user = l_user.next
            except StopIteration:
                break

        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK

def create_pipeline_programmatic_fixed(rtsp_urls_dict, config_file_path, interval=25):
    """
    Fixed version - INFERENCE ONLY, WebRTC moved to recording server.
    """
    global source_id_to_camera_id
    
    # Get resolution from config
    inference_width = VIDEO_RESOLUTION_CONFIG['RESOLUTION_W']  # 1280
    inference_height = VIDEO_RESOLUTION_CONFIG['RESOLUTION_H']  # 720
    
    pipeline = Gst.Pipeline.new("main-pipeline")

    # --- NVStreamMux for inference branch ---
    nvstreammux = Gst.ElementFactory.make("nvstreammux", "mux")
    nvstreammux.set_property("width", inference_width)
    nvstreammux.set_property("height", inference_height)
    nvstreammux.set_property("batch-size", len(rtsp_urls_dict))
    nvstreammux.set_property("batched-push-timeout", 10000) # reduce from 33000 for lower latency
    nvstreammux.set_property("live-source", 1)
    nvstreammux.set_property("nvbuf-memory-type", 3) # NVBUF_MEM_CUDA_UNIFIED
    pipeline.add(nvstreammux)

    # Build source_id mapping for the probe
    source_id_to_camera_id = {i: cam_id for i, cam_id in enumerate(sorted(rtsp_urls_dict.keys()))}
    print(f"[DEBUG] Source ID mapping: {source_id_to_camera_id}")

    # --- For each RTSP camera - INFERENCE ONLY ---
    for idx, (camera_id, rtsp_url) in enumerate(sorted(rtsp_urls_dict.items())):
        print(f"Adding camera {camera_id}: {rtsp_url}")

        # Validate URL
        try:
            parsed_url = urllib.parse.urlparse(rtsp_url)
            if not (parsed_url.scheme == 'rtsp' and parsed_url.hostname and parsed_url.path):
                print(f"Skipping invalid RTSP URL for camera {camera_id}: {rtsp_url}")
                continue
        except Exception as e:
            print(f"Error parsing RTSP URL for camera {camera_id}: {e}")
            continue

        # RTSP source
        src = Gst.ElementFactory.make("rtspsrc", f"src{camera_id}")
        src.set_property("location", rtsp_url)
        src.set_property("latency", 100)
        src.set_property("protocols", "tcp")
        src.set_property("drop-on-latency", True)
        src.set_property("do-rtsp-keep-alive", True)

        # ADD JITTER BUFFER
        jitterbuf = Gst.ElementFactory.make("rtpjitterbuffer", f"jitterbuffer_{camera_id}")
        jitterbuf.set_property("latency", 100)
        jitterbuf.set_property("drop-on-latency", True)

        # Queue after rtspsrc
        queue_src = Gst.ElementFactory.make("queue", f"queue_src_{camera_id}")
        queue_src.set_property("max-size-buffers", 2)
        queue_src.set_property("max-size-bytes", 0)
        queue_src.set_property("max-size-time", 0)

        # Use decodebin for automatic handling
        decodebin = Gst.ElementFactory.make("decodebin", f"decodebin_{camera_id}")
        
        # nvvideoconvert for inference
        nvvidconv_inference = Gst.ElementFactory.make("nvvideoconvert", f"nvvidconv_inference_{camera_id}")
        nvvidconv_inference.set_property("nvbuf-memory-type", 3)
        
        # Add caps filter for inference
        caps_inference = Gst.ElementFactory.make("capsfilter", f"caps_inference_{camera_id}")
        caps_inference.set_property("caps", Gst.Caps.from_string(
            f"video/x-raw(memory:NVMM),format=RGBA,width={inference_width},height={inference_height}"
        ))
        
        # Queue for inference
        queue_inference = Gst.ElementFactory.make("queue", f"queue_inference_{camera_id}")
        queue_inference.set_property("max-size-buffers", 2)
        queue_inference.set_property("max-size-time", 0)
        queue_inference.set_property("max-size-bytes", 0)

        # Add elements to pipeline
        elements = [src, jitterbuf, queue_src, decodebin, nvvidconv_inference, caps_inference, queue_inference]
        
        for e in elements:
            if not e:
                raise RuntimeError(f"Failed to create element: {e}")
            pipeline.add(e)

        # Dynamic linking for rtspsrc
        def on_rtsp_pad(src, pad, jitter=jitterbuf):
            sink_pad = jitter.get_static_pad("sink")
            if not sink_pad.is_linked():
                res = pad.link(sink_pad)
                if res == Gst.PadLinkReturn.OK:
                    print(f"Linked {src.get_name()} to {jitter.get_name()}")
                else:
                    print(f"Failed to link {src.get_name()} to {jitter.get_name()}: {res}")

        src.connect("pad-added", on_rtsp_pad)

        # Link static elements
        jitterbuf.link(queue_src)
        queue_src.link(decodebin)

        # Dynamic linking for decodebin
        def on_decodebin_pad(decodebin, pad, nvvc=nvvidconv_inference):
            caps = pad.get_current_caps() or pad.query_caps(None)
            if not caps or not caps.is_fixed():
                print(f"Skipping unfixed caps for {decodebin.get_name()}")
                return

            s = caps.get_structure(0)
            media_type = s.get_name()
            print(f"[DECODEBIN] {decodebin.get_name()} pad added with media type: {media_type}")
            
            if media_type.startswith("video/"):
                sink_pad = nvvc.get_static_pad("sink")
                if not sink_pad.is_linked():
                    res = pad.link(sink_pad)
                    if res == Gst.PadLinkReturn.OK:
                        print(f"Linked {decodebin.get_name()} to {nvvc.get_name()}")
                    else:
                        print(f"Failed to link {decodebin.get_name()} to {nvvc.get_name()}: {res}")

        decodebin.connect("pad-added", on_decodebin_pad)

        # Link inference chain
        nvvidconv_inference.link(caps_inference)
        caps_inference.link(queue_inference)
        
        # Connect to muxer for inference
        sinkpad = nvstreammux.request_pad_simple(f"sink_{idx}")
        srcpad = queue_inference.get_static_pad("src")
        srcpad.link(sinkpad)

        print(f"[DEBUG] Camera {camera_id} inference pipeline setup complete")

    # --- Downstream for inference branch ---
    nvvidconv_infer = Gst.ElementFactory.make("nvvideoconvert", "nvvideoconvert_infer")
    nvvidconv_infer.set_property("nvbuf-memory-type", 3)

    nvinfer = Gst.ElementFactory.make("nvinfer", "nvinfer")
    nvinfer.set_property("config-file-path", config_file_path)
    nvinfer.set_property("interval", interval)

    fakesink = Gst.ElementFactory.make("fakesink", "sink")

    for e in [nvvidconv_infer, nvinfer, fakesink]:
        if not e:
            raise RuntimeError("Failed to create downstream element")
        pipeline.add(e)
    
    nvstreammux.link(nvvidconv_infer)
    nvvidconv_infer.link(nvinfer)
    nvinfer.link(fakesink)

    # Attach probe to nvinfer src pad
    src_pad = nvinfer.get_static_pad("src")
    if src_pad:
        src_pad.add_probe(Gst.PadProbeType.BUFFER, postprocess_frame_callback, {})
        print("[DEBUG] Added probe to nvinfer src pad")
    else:
        print("[ERROR] Could not get src pad from nvinfer")

    return pipeline
