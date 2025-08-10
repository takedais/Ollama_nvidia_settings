#!/bin/bash
# Ollama GPU加速起動スクリプト

# 環境変数読み込み
source ~/.ollama_gfx1150_env

# 既存プロセス停止
sudo pkill -f ollama 2>/dev/null || true
sleep 1

# GPU設定確認
echo "=== GPU設定確認 ==="
echo "HSA_OVERRIDE_GFX_VERSION: $HSA_OVERRIDE_GFX_VERSION"
echo "OLLAMA_GPU_RUNNER: $OLLAMA_GPU_RUNNER"
echo "ROCM_PATH: $ROCM_PATH"
echo ""

# GPU加速でOllama起動
echo "=== Ollama起動（GPU加速） ==="
echo "ログを確認して 'library=rocm' が表示されることを確認してください"
echo ""

exec ollama serve
