#!/bin/bash

# Create override configuration for Ollama to use NVIDIA GPU only
cat << 'EOF' | sudo tee /etc/systemd/system/ollama.service.d/nvidia-gpu.conf
[Service]
# Force NVIDIA GPU usage and disable ROCm
Environment="CUDA_VISIBLE_DEVICES=0"
Environment="OLLAMA_HOST=0.0.0.0"
Environment="OLLAMA_ORIGINS=*"
Environment="HSA_OVERRIDE_GFX_VERSION="
Environment="ROCR_VISIBLE_DEVICES="
Environment="HIP_VISIBLE_DEVICES="
EOF

echo "Configuration created. Now reloading systemd and restarting Ollama..."

# Reload systemd configuration
sudo systemctl daemon-reload

# Restart Ollama service
sudo systemctl restart ollama

echo "Waiting for Ollama to start..."
sleep 5

# Check service status
sudo systemctl status ollama --no-pager -n 20

echo ""
echo "Testing NVIDIA GPU usage..."
# Test with a simple query
ollama run llama3.2:latest "Say hello in one word"