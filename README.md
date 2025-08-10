# Ollama NVIDIA GPU Settings

NVIDIA GPU（GeForce RTX 5060 Ti）でOllamaを動作させるための設定集です。

## 概要

このリポジトリには、OllamaをNVIDIA GPUで最適に動作させるための設定ファイル、スクリプト、ドキュメントが含まれています。

## 環境

- **GPU**: NVIDIA GeForce RTX 5060 Ti (16GB VRAM)
- **CUDA Version**: 12.9
- **Ollama Version**: 0.11.3
- **OS**: Linux (Ubuntu/Debian系)

## ファイル構成

### ドキュメント
- `ollama-nvidia-gpu-setup.md` - 詳細な設定ガイド

### セットアップスクリプト
- `ollama-nvidia-setup.sh` - 初期NVIDIA GPU設定
- `fix-ollama-cuda.sh` - CUDA設定修正
- `fix-ollama-nvidia-only.sh` - NVIDIA専用設定（AMD GPU無効化）
- `optimize-gpt-oss-gpu.sh` - GPUメモリ最適化

### モデル関連
- `gpt-oss-gpu.modelfile` - カスタムモデル定義ファイル
- `create-optimized-model.sh` - 最適化モデル作成スクリプト

### テスト・ユーティリティ
- `test-ollama-gpu.sh` - GPU動作確認スクリプト
- `start_ollama_gpu.sh` - Ollama GPU起動スクリプト

## クイックスタート

1. NVIDIA専用設定を適用：
```bash
./fix-ollama-nvidia-only.sh
```

2. 大きなモデル用の最適化を適用：
```bash
./optimize-gpt-oss-gpu.sh
```

3. カスタムモデルを作成（オプション）：
```bash
./create-optimized-model.sh
```

## 主な成果

- 小〜中規模モデル（llama3.2, llama3.1:8b）は完全にGPUで実行
- 大規模モデル（gpt-oss:20b）は部分的にGPUオフロード（15/25レイヤー）
- rocBLASエラーの解決
- 動的GPUメモリ管理の実現

## トラブルシューティング

詳細は `ollama-nvidia-gpu-setup.md` を参照してください。

## ライセンス

MIT License

## 作成者

takedais

## 作成日

2025年8月10日