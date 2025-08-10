#!/bin/bash

echo "Fixing Ollama CUDA configuration..."

# Create updated configuration for NVIDIA GPU
cat << 'EOF' | sudo tee /etc/systemd/system/ollama.service.d/nvidia-gpu.conf
[Service]
# Force NVIDIA GPU usage
Environment="CUDA_VISIBLE_DEVICES=0"
Environment="OLLAMA_HOST=0.0.0.0"
Environment="OLLAMA_ORIGINS=*"
# Disable ROCm/AMD GPU
Environment="HSA_OVERRIDE_GFX_VERSION="
Environment="ROCR_VISIBLE_DEVICES="
Environment="HIP_VISIBLE_DEVICES="
# Add CUDA library path
Environment="LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"
EOF

echo "Configuration updated. Reloading and restarting Ollama..."

# Reload systemd
sudo systemctl daemon-reload

# Restart Ollama
sudo systemctl restart ollama

echo "Waiting for Ollama to start..."
sleep 5

# Check status
echo "=== Ollama Service Status ==="
sudo systemctl status ollama --no-pager -n 10

echo ""
echo "=== Testing NVIDIA GPU ==="
# Test with model
ollama run llama3.1:8b "Hello, what GPU am I using?" &

# Wait and check GPU usage
sleep 5
echo ""
echo "=== GPU Usage ==="
nvidia-smi | grep -E "(ollama|GeForce|MiB)"

echo ""
echo "Setup complete! Check if ollama appears in nvidia-smi output above."