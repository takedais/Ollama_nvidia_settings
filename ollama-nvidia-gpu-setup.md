# Ollama NVIDIA GPU セットアップガイド

## 環境情報
- **GPU**: NVIDIA GeForce RTX 5060 Ti (16GB VRAM)
- **Driver**: 575.64.03
- **CUDA Version**: 12.9
- **Ollama Version**: 0.11.3
- **OS**: Linux (Ubuntu/Debian系)

## 実施した設定

### 1. NVIDIA GPU認識の問題と解決

#### 問題点
- OllamaがAMD GPU (内蔵GPU gfx1150) とNVIDIA GPUの両方を検出し、混乱していた
- rocBLASエラーによるクラッシュが発生
- CUDAライブラリが正しく認識されない

#### 解決策
NVIDIA GPU専用の設定ファイルを作成：

```bash
# /etc/systemd/system/ollama.service.d/nvidia-only.conf
[Service]
Environment=
Environment="CUDA_VISIBLE_DEVICES=0"
Environment="OLLAMA_HOST=0.0.0.0"
Environment="OLLAMA_ORIGINS=*"
Environment="ROCR_VISIBLE_DEVICES="
Environment="HIP_VISIBLE_DEVICES="
Environment="HSA_OVERRIDE_GFX_VERSION="
Environment="OLLAMA_ROCM_DISABLE=1"
Environment="LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
Environment="OLLAMA_LLM_LIBRARY=cuda"
Environment="OLLAMA_DEBUG=INFO"
```

### 2. GPUメモリ最適化設定

大きなモデル（gpt-oss:20b）をGPUで実行するための最適化：

```bash
# /etc/systemd/system/ollama.service.d/gpu-optimize.conf
[Service]
Environment=
Environment="CUDA_VISIBLE_DEVICES=0"
Environment="OLLAMA_HOST=0.0.0.0"
Environment="OLLAMA_ORIGINS=*"
Environment="ROCR_VISIBLE_DEVICES="
Environment="HIP_VISIBLE_DEVICES="
Environment="HSA_OVERRIDE_GFX_VERSION="
Environment="OLLAMA_ROCM_DISABLE=1"
Environment="LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64"
Environment="OLLAMA_LLM_LIBRARY=cuda"
Environment="OLLAMA_NUM_GPU=1"
Environment="OLLAMA_GPU_OVERHEAD=0"
Environment="OLLAMA_MAX_LOADED_MODELS=1"
Environment="OLLAMA_KEEP_ALIVE=1m"
Environment="OLLAMA_CONTEXT_LENGTH=2048"
Environment="OLLAMA_NUM_PARALLEL=1"
Environment="OLLAMA_DEBUG=INFO"
```

### 3. カスタムモデルの作成（gpt-oss:gpu）

大きなモデルを部分的にGPUにオフロードするためのカスタムモデル作成：

#### Modelfile (`/home/takedais/gpt-oss-gpu.modelfile`)
```dockerfile
# Optimized Modelfile for gpt-oss:20b GPU usage
FROM gpt-oss:20b

# Reduce context to save GPU memory
PARAMETER num_ctx 2048

# Batch size optimization
PARAMETER num_batch 128

# Number of threads (adjust based on your CPU)
PARAMETER num_thread 12

# Temperature for consistent output
PARAMETER temperature 0.7

# GPU layers - force partial offload (adjust this value)
PARAMETER num_gpu 15
```

#### モデル作成コマンド
```bash
ollama create gpt-oss:gpu -f /home/takedais/gpt-oss-gpu.modelfile
```

## 結果

### 各モデルのGPU使用状況

| モデル | サイズ | GPU使用状況 | 備考 |
|--------|--------|------------|------|
| llama3.2:latest | 2.0 GB | 全29レイヤーGPUオフロード | 完全にGPUで実行 |
| llama3.1:8b | 4.9 GB | 全33レイヤーGPUオフロード | 完全にGPUで実行 |
| gemma2:27b | 15 GB | CPUで実行 | GPUメモリ不足 |
| gpt-oss:20b (オリジナル) | 13 GB | CPUで実行 | 実行時14.9GB必要でGPUメモリ不足 |
| **gpt-oss:gpu (カスタム)** | 13 GB | **15/25レイヤーGPUオフロード** | **9.7GB GPUメモリ使用** |

### パフォーマンス
- GPU使用率: 推論実行時に5-14%
- GPUメモリ: 動的に管理され、使用後すぐに解放
- nvidia-smiのプロセス欄には表示されない（動的メモリ管理のため）

## トラブルシューティング

### 1. nvidia-smiにOllamaプロセスが表示されない
**原因**: Ollamaは動的にGPUメモリを管理するため  
**確認方法**: 
```bash
# GPU使用率を監視
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader --loop=1
```

### 2. モデルがCPUで実行される
**原因**: GPUメモリ不足  
**解決策**: 
- コンテキストサイズを削減: `OLLAMA_CONTEXT_LENGTH=1024`
- カスタムモデルで部分的にGPUオフロード
- より小さいモデルを使用

### 3. rocBLASエラー
**原因**: AMD GPUライブラリとの競合  
**解決策**: `OLLAMA_ROCM_DISABLE=1`を設定

## 便利なコマンド

### サービス管理
```bash
# Ollama再起動
sudo systemctl restart ollama

# ログ確認
journalctl -u ollama -f

# GPU offload状況確認
journalctl -u ollama --since "5 minutes ago" | grep -E "offload|layer|GPU"
```

### モデル管理
```bash
# インストール済みモデル一覧
ollama list

# モデル実行
ollama run llama3.1:8b
ollama run gpt-oss:gpu  # カスタマイズ版

# モデル削除
ollama rm model_name
```

### GPU監視
```bash
# GPU使用状況
nvidia-smi

# 継続的な監視
watch -n 1 nvidia-smi

# CSV形式で監視
nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv --loop=1
```

## 推奨事項

1. **小〜中規模モデル** (< 8GB): そのままGPUで実行可能
2. **大規模モデル** (10-15GB): カスタムModelfileで部分GPUオフロード
3. **超大規模モデル** (> 15GB): CPUで実行するか、量子化版を使用

## 設定ファイル一覧

- `/etc/systemd/system/ollama.service.d/nvidia-only.conf` - NVIDIA GPU専用設定
- `/etc/systemd/system/ollama.service.d/gpu-optimize.conf` - GPUメモリ最適化設定
- `/home/takedais/gpt-oss-gpu.modelfile` - カスタムモデル定義
- `/home/takedais/fix-ollama-cuda.sh` - CUDA設定修正スクリプト
- `/home/takedais/optimize-gpt-oss-gpu.sh` - GPU最適化スクリプト
- `/home/takedais/create-optimized-model.sh` - カスタムモデル作成スクリプト
- `/home/takedais/test-ollama-gpu.sh` - GPU動作テストスクリプト

## 参考情報

- Ollama公式ドキュメント: https://github.com/ollama/ollama
- NVIDIA CUDA: https://developer.nvidia.com/cuda-downloads
- モデルライブラリ: https://ollama.com/library

---
*作成日: 2025年8月9日*  
*環境: NVIDIA GeForce RTX 5060 Ti / Ollama 0.11.3*