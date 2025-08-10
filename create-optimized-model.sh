#!/bin/bash

echo "=== Creating optimized gpt-oss model for GPU ==="
echo ""

# Create optimized model
echo "Creating gpt-oss:gpu model..."
ollama create gpt-oss:gpu -f /home/takedais/gpt-oss-gpu.modelfile

echo ""
echo "Model created. Testing..."
echo ""

# Test the optimized model
echo "Running optimized model..."
echo "What is 2+2?" | ollama run gpt-oss:gpu &
MODEL_PID=$!

# Monitor GPU for 10 seconds
echo ""
echo "Monitoring GPU usage..."
for i in {1..10}; do
    echo -n "Second $i: "
    nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
    sleep 1
done

# Kill if still running
kill $MODEL_PID 2>/dev/null

echo ""
echo "=== Checking offload status ==="
journalctl -u ollama --since "1 minute ago" --no-pager | grep -E "(gpt-oss|offload|layer)" | tail -10

echo ""
echo "Optimization complete. Use 'ollama run gpt-oss:gpu' for the optimized version."