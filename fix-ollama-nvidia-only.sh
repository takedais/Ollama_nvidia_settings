#!/bin/bash

echo "Configuring Ollama to use NVIDIA GPU only..."

# Create comprehensive configuration
cat << 'EOF' | sudo tee /etc/systemd/system/ollama.service.d/nvidia-only.conf
[Service]
# Clear all environment first
Environment=

# Force NVIDIA GPU only
Environment="CUDA_VISIBLE_DEVICES=0"
Environment="OLLAMA_HOST=0.0.0.0"
Environment="OLLAMA_ORIGINS=*"

# Completely disable AMD/ROCm
Environment="ROCR_VISIBLE_DEVICES="
Environment="HIP_VISIBLE_DEVICES="
Environment="HSA_OVERRIDE_GFX_VERSION="
Environment="OLLAMA_ROCM_DISABLE=1"

# CUDA library paths
Environment="LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64:$LD_LIBRARY_PATH"

# Force CUDA backend
Environment="OLLAMA_LLM_LIBRARY=cuda"

# Debug
Environment="OLLAMA_DEBUG=INFO"
EOF

echo "Removing old configurations..."
sudo rm -f /etc/systemd/system/ollama.service.d/nvidia-gpu.conf
sudo rm -f /etc/systemd/system/ollama.service.d/override.conf
sudo rm -f /etc/systemd/system/ollama.service.d/gfx1150-gpu.conf

echo "Reloading systemd and restarting Ollama..."
sudo systemctl daemon-reload
sudo systemctl restart ollama

sleep 5

echo ""
echo "=== Service Status ==="
sudo systemctl status ollama --no-pager | head -20

echo ""
echo "=== GPU Detection ==="
sudo journalctl -u ollama --since "10 seconds ago" --no-pager | grep -E "(GPU|CUDA|nvidia|looking for)" | head -10

echo ""
echo "Testing with a model..."
echo "2+2?" | timeout 10 ollama run llama3.2:latest || echo "Model test completed or timed out"

echo ""
echo "Setup complete. Check nvidia-smi while running a model to verify GPU usage."