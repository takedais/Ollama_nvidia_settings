#!/bin/bash

# Ollama System Information Checker
# This script collects comprehensive system and model information

echo "========================================"
echo "  Ollama System Information Report"
echo "========================================"
echo ""
echo "Date: $(date)"
echo ""

echo "=== System Information ==="
echo "OS: $(lsb_release -d | cut -f2)"
echo "Kernel: $(uname -r)"
echo "Architecture: $(uname -m)"
echo ""

echo "=== Memory Information ==="
free -h
echo ""

echo "=== GPU Information ==="
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free,memory.used,utilization.gpu --format=csv
echo ""
echo "CUDA Version:"
nvidia-smi | grep "CUDA Version" | awk '{print $9}'
echo ""

echo "=== Ollama Information ==="
echo "Ollama Version: $(ollama --version)"
echo "Service Status: $(systemctl is-active ollama)"
echo ""

echo "=== Installed Models ==="
ollama list
echo ""

echo "=== Model Storage Usage ==="
echo "Total model storage:"
du -sh ~/.ollama/models 2>/dev/null || echo "Models directory not found in user home"
echo ""

echo "=== Current Ollama Configuration ==="
echo "Environment Variables:"
systemctl show ollama | grep -E "^Environment=" | sed 's/Environment=/  /'
echo ""

echo "=== GPU Process Check ==="
nvidia-smi pmon -c 1
echo ""

echo "=== Recent Ollama Activity ==="
echo "Last 10 log entries with GPU/memory info:"
journalctl -u ollama --since "1 hour ago" --no-pager | grep -E "(GPU|memory|offload|layer)" | tail -10
echo ""

echo "========================================"
echo "  Report Complete"
echo "========================================"