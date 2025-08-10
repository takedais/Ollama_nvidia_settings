#!/bin/bash

echo "=== Testing Ollama GPU Usage ==="
echo ""

# Kill any existing ollama processes
echo "Cleaning up existing processes..."
pkill -f "ollama run"
sleep 2

# Restart service
echo "Restarting Ollama service..."
sudo systemctl restart ollama
sleep 5

# Test with small model first
echo ""
echo "Testing with llama3.2 (2GB model)..."
echo "Running model in background and checking GPU..."

# Run model in background
ollama run llama3.2:latest "What is artificial intelligence in 50 words?" &
MODEL_PID=$!

# Wait for model to load
sleep 8

# Check GPU usage
echo ""
echo "=== NVIDIA GPU Status ==="
nvidia-smi | grep -E "(GeForce|MiB.*C|ollama|runner)" || nvidia-smi

# Wait for completion
wait $MODEL_PID 2>/dev/null

echo ""
echo "=== Testing with llama3.1:8b (4.9GB model) ==="
ollama run llama3.1:8b "Hello world" &
MODEL_PID=$!

sleep 8
nvidia-smi | grep -E "(GeForce|MiB.*C|ollama|runner)" || echo "No ollama process in GPU"

wait $MODEL_PID 2>/dev/null

echo ""
echo "Test complete. If you don't see ollama in nvidia-smi, the model may be too small or running on CPU."
echo ""
echo "Check logs with: journalctl -u ollama --since '5 minutes ago' | grep -E 'offload|GPU|layer'"