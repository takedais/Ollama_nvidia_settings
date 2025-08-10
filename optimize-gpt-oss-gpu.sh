#!/bin/bash

echo "=== Optimizing gpt-oss:20b for GPU usage ==="
echo ""

# Create optimized Ollama configuration
echo "Creating optimized configuration..."
cat << 'EOF' | sudo tee /etc/systemd/system/ollama.service.d/gpu-optimize.conf
[Service]
# Clear all environment first
Environment=

# Force NVIDIA GPU only
Environment="CUDA_VISIBLE_DEVICES=0"
Environment="OLLAMA_HOST=0.0.0.0"
Environment="OLLAMA_ORIGINS=*"

# Disable AMD/ROCm completely
Environment="ROCR_VISIBLE_DEVICES="
Environment="HIP_VISIBLE_DEVICES="
Environment="HSA_OVERRIDE_GFX_VERSION="
Environment="OLLAMA_ROCM_DISABLE=1"

# CUDA settings
Environment="LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64"
Environment="OLLAMA_LLM_LIBRARY=cuda"

# Memory optimization settings
Environment="OLLAMA_NUM_GPU=1"
Environment="OLLAMA_GPU_OVERHEAD=0"
Environment="OLLAMA_MAX_LOADED_MODELS=1"
Environment="OLLAMA_KEEP_ALIVE=1m"

# Context and batch size optimization
Environment="OLLAMA_CONTEXT_LENGTH=2048"
Environment="OLLAMA_NUM_PARALLEL=1"

# Debug
Environment="OLLAMA_DEBUG=INFO"
EOF

# Restart Ollama
echo ""
echo "Restarting Ollama service..."
sudo systemctl daemon-reload
sudo systemctl restart ollama
sleep 5

# Check service status
echo ""
echo "=== Service Status ==="
sudo systemctl status ollama --no-pager | head -10

# Try to load model with specific layer count
echo ""
echo "=== Testing gpt-oss:20b with partial GPU offload ==="
echo ""
echo "Attempting to run model..."

# First, try with environment variable for layer offload
export OLLAMA_NUM_GPU_LAYERS=15

# Run a quick test
echo "What is 2+2?" | timeout 30 ollama run gpt-oss:20b || echo "Test completed or timed out"

echo ""
echo "=== Checking GPU usage logs ==="
journalctl -u ollama --since "1 minute ago" --no-pager | grep -E "(offload|layer|GPU|memory)" | tail -10

echo ""
echo "Configuration applied. If the model still doesn't use GPU, try:"
echo "1. Reduce context length further: export OLLAMA_CONTEXT_LENGTH=1024"
echo "2. Use a quantized version of the model if available"
echo "3. Consider using smaller models like llama3.1:8b or gemma2:9b"