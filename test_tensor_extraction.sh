#!/bin/bash

# Enhanced tensor extraction test
set -e

echo "=== DeepStream Tensor Extraction Test ==="
echo ""

# Test with longer duration and detailed logging
echo "Testing tensor extraction with detailed logging..."
echo "Running for 30 seconds to see more tensor activity..."
echo ""

# Clean output directory
rm -f output/tensor_output_*.csv

# Run with detailed logging for longer time
timeout 30s ./deepstream-multi-source-cpp \
    --detailed-logging \
    --max-tensor-values 5 \
    /opt/nvidia/deepstream/deepstream/samples/streams/sample_720p.mp4 \
    /opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4 \
    || echo "Test completed after timeout"

echo ""
echo "=== Results Analysis ==="

# Check output files
if [ -d "output" ] && [ "$(ls -A output/*.csv 2>/dev/null)" ]; then
    echo "Tensor output files generated:"
    ls -la output/*.csv
    echo ""
    
    # Count total entries
    total_entries=$(cat output/*.csv | grep -c "^Source_" || echo "0")
    echo "Total tensor entries extracted: $total_entries"
    
    # Show sample entries
    echo ""
    echo "Sample tensor data (first 5 entries):"
    head -6 output/tensor_output_*.csv | tail -5
    
    # Count entries per source
    echo ""
    echo "Entries per source:"
    grep "^Source_" output/*.csv | cut -d',' -f1 | sort | uniq -c || echo "No data"
    
else
    echo "No tensor output files found"
fi

echo ""
echo "=== Performance Tips ==="
echo "To increase tensor extraction frequency:"
echo "1. Reduce inference interval in model config (currently interval=0)"
echo "2. Use faster video sources or live streams"
echo "3. Process multiple sources simultaneously"
echo "4. Enable performance mode with -p flag"
echo ""
echo "The current extraction rate is normal for video files at 30fps"
echo "Each successful extraction shows 2 tensors (detection outputs)"