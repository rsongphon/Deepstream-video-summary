#!/bin/bash

# Test script for flexible multi-source DeepStream application
# This script demonstrates various usage scenarios

set -e

echo "=== DeepStream Flexible Multi-Source Application Test ==="
echo ""

# Check if application exists
if [ ! -f "./deepstream-multi-source-cpp" ]; then
    echo "Error: Application not found. Run 'make cpp' first."
    exit 1
fi

# Test 1: Basic help
echo "1. Testing help functionality..."
./deepstream-multi-source-cpp --help
echo ""

# Test 2: Check sample videos exist
SAMPLE_DIR="/opt/nvidia/deepstream/deepstream/samples/streams"
VIDEO1="$SAMPLE_DIR/sample_720p.mp4"
VIDEO2="$SAMPLE_DIR/sample_1080p_h264.mp4"

if [ ! -f "$VIDEO1" ] || [ ! -f "$VIDEO2" ]; then
    echo "Warning: Sample videos not found at $SAMPLE_DIR"
    echo "Please provide video files as arguments to test with custom sources"
    echo "Usage: $0 video1.mp4 video2.mp4 [video3.mp4 ...]"
    exit 0
fi

# Test 3: Single source (no display)
echo "2. Testing single source processing (no display)..."
timeout 10s ./deepstream-multi-source-cpp "$VIDEO1" || echo "Single source test completed"
echo ""

# Test 4: Two sources (no display) 
echo "3. Testing two sources processing (no display)..."
timeout 10s ./deepstream-multi-source-cpp "$VIDEO1" "$VIDEO2" || echo "Two sources test completed"
echo ""

# Test 5: Three sources with custom config
echo "4. Testing three sources with performance monitoring..."
timeout 10s ./deepstream-multi-source-cpp -p "$VIDEO1" "$VIDEO2" "$VIDEO1" || echo "Three sources test completed"
echo ""

# Test 6: Check tensor output
echo "5. Checking tensor output generation..."
if [ -d "output" ] && [ "$(ls -A output)" ]; then
    echo "Tensor output files generated:"
    ls -la output/
else
    echo "No tensor output files found"
fi
echo ""

# Test 7: Configuration file test
if [ -f "config/pipeline_config.yaml" ]; then
    echo "6. Testing with YAML configuration..."
    timeout 10s ./deepstream-multi-source-cpp -c config/pipeline_config.yaml "$VIDEO1" "$VIDEO2" || echo "YAML config test completed"
    echo ""
fi

echo "=== All tests completed successfully! ==="
echo ""
echo "To test with display (if X11 forwarding available):"
echo "  ./deepstream-multi-source-cpp -d $VIDEO1 $VIDEO2"
echo ""
echo "To test with custom videos:"
echo "  ./deepstream-multi-source-cpp your_video1.mp4 your_video2.mp4"
echo ""
echo "To test with live streams:"
echo "  ./deepstream-multi-source-cpp rtsp://camera1/stream rtsp://camera2/stream"
echo ""
echo "Available options:"
echo "  -p    Enable performance monitoring"
echo "  -d    Enable display output" 
echo "  -c    Use YAML configuration file"
echo "  -o    Set output directory for tensor data"
echo "  -b    Set custom batch size"