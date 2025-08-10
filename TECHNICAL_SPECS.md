# Ollama NVIDIA GPU 技術仕様書

## システム環境

### ハードウェア仕様

#### GPU
```
GPU Model:        NVIDIA GeForce RTX 5060 Ti
VRAM:            16,311 MiB (約16GB)
Driver Version:   575.64.03
CUDA Version:     12.9
Compute Capability: 12.0
```

#### システムメモリ
```
RAM Total:        27 GiB
Available:        23 GiB
Swap:            8.0 GiB
```

#### CPU
```
Threads:         12 (Ollamaで使用設定)
```

### ソフトウェア環境
```
OS:              Linux (Ubuntu/Debian系)
Kernel:          6.14.0-27-generic
Ollama Version:  0.11.3
Platform:        linux/amd64
```

## インストール済みモデル一覧

| モデル名 | ID | サイズ | 最終更新 | 用途 |
|---------|-----|-------|---------|------|
| **gpt-oss:gpu** | c401b3da026b | 13 GB | カスタム作成 | GPU最適化版 |
| gpt-oss:20b | f2b8351c629c | 13 GB | オリジナル | 大規模言語モデル |
| llama3.1:8b | 46e0c10c039e | 4.9 GB | 標準 | 中規模言語モデル |
| llama3.2:latest | a80c4f17acd5 | 2.0 GB | 標準 | 小規模言語モデル |
| gemma2:27b | 53261bc9c192 | 15 GB | 標準 | 超大規模モデル |
| phi3:mini | 4f2222927938 | 2.2 GB | 標準 | 軽量モデル |
| mxbai-embed-large | 468836162de7 | 669 MB | 標準 | エンベディング用 |

## モデル詳細仕様

### gpt-oss:gpu (カスタマイズ版)
```yaml
Architecture:     gptoss
Parameters:       20.9B
Context Length:   131,072 (設定で2,048に制限)
Embedding Length: 2,880
Quantization:     MXFP4

カスタム設定:
- num_ctx:        2,048  # コンテキストサイズ削減
- num_batch:      128    # バッチサイズ最適化
- num_gpu:        15     # GPUオフロードレイヤー数
- num_thread:     12     # CPUスレッド数
- temperature:    0.7    # 生成温度
```

## GPU使用状況分析

### レイヤーオフロード状況

| モデル | 総レイヤー数 | GPUオフロード | メモリ要件 | 実際のGPU使用 | 状態 |
|--------|-------------|--------------|-----------|--------------|------|
| llama3.2:latest | 29 | 29 (100%) | 3.1 GiB | 3.1 GiB | ✅ 完全GPU |
| llama3.1:8b | 33 | 33 (100%) | 5.7 GiB | 5.7 GiB | ✅ 完全GPU |
| gpt-oss:20b | 25 | 0 (0%) | 14.9 GiB | 0 GiB | ❌ CPU実行 |
| **gpt-oss:gpu** | 25 | **15 (60%)** | **9.7 GiB** | **9.7 GiB** | ⚡ ハイブリッド |
| gemma2:27b | - | 0 (0%) | >16 GiB | 0 GiB | ❌ CPU実行 |

### メモリ計算式
```
必要メモリ = モデルウェイト + KVキャッシュ + グラフメモリ + オーバーヘッド

例: gpt-oss:gpu
- モデルウェイト: 11.7 GiB
- KVキャッシュ:   291 MiB
- グラフメモリ:   2.0 GiB
- 合計要件:       約14.9 GiB (フル)
- 部分オフロード: 9.7 GiB (15レイヤー)
```

## パフォーマンス特性

### GPU使用率
```
アイドル時:       0%
推論実行時:       5-14%
メモリ使用:       動的管理（使用後即解放）
```

### 推論速度比較（相対値）
```
CPU only:         1.0x (基準)
Hybrid (60% GPU): 2-3x
Full GPU:         5-10x
```

## Ollamaサービス設定

### 環境変数設定 (`/etc/systemd/system/ollama.service.d/`)

#### nvidia-only.conf
```bash
[Service]
Environment="CUDA_VISIBLE_DEVICES=0"
Environment="OLLAMA_LLM_LIBRARY=cuda"
Environment="OLLAMA_ROCM_DISABLE=1"
Environment="LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64"
```

#### gpu-optimize.conf
```bash
[Service]
Environment="OLLAMA_NUM_GPU=1"
Environment="OLLAMA_GPU_OVERHEAD=0"
Environment="OLLAMA_MAX_LOADED_MODELS=1"
Environment="OLLAMA_KEEP_ALIVE=1m"
Environment="OLLAMA_CONTEXT_LENGTH=2048"
Environment="OLLAMA_NUM_PARALLEL=1"
```

## トラブルシューティング用コマンド

### GPUステータス確認
```bash
# GPU基本情報
nvidia-smi

# GPU使用率モニタリング
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv --loop=1

# Ollamaログ確認
journalctl -u ollama -f | grep -E "offload|GPU|layer"
```

### モデル管理
```bash
# モデルリスト
ollama list

# モデル詳細
ollama show <model_name>

# GPU最適化版実行
ollama run gpt-oss:gpu
```

## 最適化のポイント

### 1. コンテキストサイズの調整
- デフォルト: 131,072 トークン
- 最適化後: 2,048 トークン
- 効果: メモリ使用量を大幅削減

### 2. レイヤーオフロード戦略
- 小規模モデル（<8GB）: 全レイヤーGPU
- 中規模モデル（8-12GB）: 70-80% GPU
- 大規模モデル（>12GB）: 50-60% GPU

### 3. バッチサイズ最適化
- デフォルト: 512
- 最適化後: 128
- 効果: メモリ効率向上

## 推奨設定

### 用途別モデル選択

| 用途 | 推奨モデル | 理由 |
|-----|-----------|------|
| 高速応答 | llama3.2:latest | 完全GPU実行、低レイテンシ |
| バランス | llama3.1:8b | 完全GPU実行、高精度 |
| 高精度 | gpt-oss:gpu | ハイブリッド実行、大規模モデル |
| エンベディング | mxbai-embed-large | 軽量、高速 |

## 制限事項

1. **動的メモリ管理**: nvidia-smiのプロセス欄に表示されない
2. **VRAMリミット**: 16GB以上のモデルは部分オフロードまたはCPU実行
3. **ROCm競合**: AMD GPU無効化が必要

## 今後の改善案

1. **量子化モデルの活用**: 4bit/8bit量子化でメモリ削減
2. **Flash Attention**: メモリ効率の改善
3. **マルチGPU対応**: 複数GPUでの分散処理
4. **動的レイヤー調整**: 負荷に応じた自動調整

---
*最終更新: 2025年8月10日*  
*作成者: takedais with Claude*