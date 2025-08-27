#!/bin/bash

echo "=== GStreamer DeepStream Pipeline Test ==="

# Test basic GStreamer functionality
echo "1. Testing basic GStreamer..."
gst-launch-1.0 --version

echo -e "\n2. Testing available DeepStream plugins..."
gst-inspect-1.0 nvstreammux
gst-inspect-1.0 nvinfer
gst-inspect-1.0 nvurisrcbin

echo -e "\n3. Testing simple file playback..."
timeout 5s gst-launch-1.0 filesrc location=/opt/nvidia/deepstream/deepstream/samples/streams/sample_720p.mp4 ! qtdemux ! h264parse ! avdec_h264 ! videoconvert ! fakesink sync=false

echo -e "\n4. Testing nvurisrcbin with single file..."
timeout 5s gst-launch-1.0 nvurisrcbin uri=file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_720p.mp4 ! fakesink sync=false

echo -e "\n5. Testing DeepStream pipeline with single source..."
timeout 10s gst-launch-1.0 \
    nvurisrcbin uri=file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_720p.mp4 ! \
    nvstreammux batch-size=1 width=1920 height=1080 ! \
    nvinfer config-file-path=configs/multi_inference_pgie_config.txt batch-size=1 ! \
    fakesink sync=false

echo -e "\n6. Testing with uridecodebin fallback..."
timeout 10s gst-launch-1.0 \
    uridecodebin uri=file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_720p.mp4 ! \
    nvstreammux batch-size=1 width=1920 height=1080 ! \
    nvinfer config-file-path=configs/multi_inference_pgie_config.txt batch-size=1 ! \
    fakesink sync=false

echo -e "\nTest complete!"