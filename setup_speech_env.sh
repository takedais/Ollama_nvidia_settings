#!/bin/bash

# 音声認識環境セットアップスクリプト
# AMD GPU (ROCm) 対応版

set -e

PROJECT_NAME="speech-to-text-project"
PYTHON_VERSION="python3"

echo "=== 音声認識環境セットアップ開始 ==="
echo "日時: $(date)"
echo "Python: $($PYTHON_VERSION --version)"
echo ""

# 色付きメッセージ用関数
print_success() {
    echo -e "\033[32m✓ $1\033[0m"
}

print_warning() {
    echo -e "\033[33m⚠️ $1\033[0m"
}

print_error() {
    echo -e "\033[31m✗ $1\033[0m"
}

print_info() {
    echo -e "\033[34mℹ️ $1\033[0m"
}

# 1. プロジェクトディレクトリ作成
echo "1. プロジェクトディレクトリ作成"
if [ ! -d "$PROJECT_NAME" ]; then
    mkdir -p $PROJECT_NAME
    print_success "プロジェクトディレクトリ作成: $PROJECT_NAME"
else
    print_info "プロジェクトディレクトリ既存: $PROJECT_NAME"
fi

cd $PROJECT_NAME

# 2. 仮想環境作成
echo ""
echo "2. Python仮想環境作成"
if [ ! -d "venv" ]; then
    $PYTHON_VERSION -m venv venv
    print_success "仮想環境作成完了"
else
    print_info "仮想環境既存"
fi

# 3. 仮想環境有効化
echo ""
echo "3. 仮想環境有効化"
source venv/bin/activate
print_success "仮想環境有効化: $(which python)"

# 4. pipアップグレード
echo ""
echo "4. pipアップグレード"
pip install --upgrade pip wheel setuptools
print_success "pip更新完了"

# 5. ROCm環境変数設定
echo ""
echo "5. ROCm環境変数設定"
cat > .env << 'EOF'
# ROCm環境変数
export ROCM_PATH=/opt/rocm
export HIP_PATH=/opt/rocm
export HSA_PATH=/opt/rocm
export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH
export PATH=/opt/rocm/bin:$PATH

# GPU設定
export HSA_OVERRIDE_GFX_VERSION=11.5.1
export HIP_VISIBLE_DEVICES=0

# Ollama設定
export OLLAMA_GPU_RUNNER=rocm
export OLLAMA_MAX_VRAM=3GB
EOF

source .env
print_success "環境変数設定完了"

# 6. PyTorch (ROCm版)インストール
echo ""
echo "6. PyTorch (ROCm版) インストール"
print_info "ROCm対応PyTorchをインストール中..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.6
print_success "PyTorch インストール完了"

# 7. 音声関連ライブラリインストール
echo ""
echo "7. 音声関連ライブラリインストール"

# OpenAI Whisper
print_info "OpenAI Whisperインストール中..."
pip install openai-whisper
print_success "OpenAI Whisper インストール完了"

# 音声処理ライブラリ
print_info "音声処理ライブラリインストール中..."
pip install librosa soundfile pydub
print_success "音声処理ライブラリ インストール完了"

# Web API用
print_info "Web APIライブラリインストール中..."
pip install flask flask-cors requests python-dotenv
print_success "Web APIライブラリ インストール完了"

# 開発用ツール
print_info "開発用ツールインストール中..."
pip install jupyter notebook ipython
print_success "開発用ツール インストール完了"

# 8. requirements.txt作成
echo ""
echo "8. requirements.txt作成"
pip freeze > requirements.txt
print_success "requirements.txt作成完了"

# 9. プロジェクト構造作成
echo ""
echo "9. プロジェクト構造作成"
mkdir -p {src,tests,data/audio,notebooks,docs}

# サンプルファイル作成
cat > src/__init__.py << 'EOF'
# Speech-to-Text Project
__version__ = "0.1.0"
EOF

cat > src/speech_processor.py << 'EOF'
"""
音声認識処理クラス
"""
import whisper
import requests
import logging

class SpeechProcessor:
    def __init__(self, whisper_model="tiny", ollama_model="phi3:mini"):
        self.whisper_model = whisper.load_model(whisper_model)
        self.ollama_model = ollama_model
        self.ollama_url = "http://localhost:11434"
        
    def transcribe_audio(self, audio_path):
        """音声ファイルをテキストに変換"""
        result = self.whisper_model.transcribe(audio_path)
        return result["text"]
    
    def process_with_llm(self, text, prompt_template=None):
        """LLMでテキスト後処理"""
        if prompt_template is None:
            prompt_template = "以下の音声認識結果を整理してください: {text}"
        
        prompt = prompt_template.format(text=text)
        
        response = requests.post(
            f"{self.ollama_url}/api/generate",
            json={
                'model': self.ollama_model,
                'prompt': prompt,
                'stream': False
            },
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json().get('response', '')
        else:
            raise Exception(f"LLM処理エラー: {response.status_code}")
    
    def full_pipeline(self, audio_path, use_llm=True):
        """完全なパイプライン処理"""
        # 音声認識
        transcribed_text = self.transcribe_audio(audio_path)
        
        if use_llm:
            # LLM後処理
            processed_text = self.process_with_llm(transcribed_text)
            return {
                'original': transcribed_text,
                'processed': processed_text
            }
        else:
            return {
                'original': transcribed_text,
                'processed': transcribed_text
            }
EOF

cat > src/api_server.py << 'EOF'
"""
Flask API サーバー
"""
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import tempfile
from speech_processor import SpeechProcessor

app = Flask(__name__)
CORS(app)

# 音声処理初期化
processor = SpeechProcessor()

@app.route('/')
def index():
    return jsonify({
        'message': 'Speech-to-Text API Server',
        'version': '0.1.0',
        'endpoints': ['/health', '/transcribe']
    })

@app.route('/health')
def health():
    return jsonify({'status': 'healthy'})

@app.route('/transcribe', methods=['POST'])
def transcribe():
    try:
        # 音声ファイル取得
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        use_llm = request.form.get('use_llm', 'true').lower() == 'true'
        
        # 一時ファイルに保存
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            audio_file.save(tmp_file.name)
            
            # 音声処理実行
            result = processor.full_pipeline(tmp_file.name, use_llm=use_llm)
            
            # 一時ファイル削除
            os.unlink(tmp_file.name)
            
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
EOF

cat > tests/test_speech.py << 'EOF'
"""
音声認識テスト
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from speech_processor import SpeechProcessor

def test_whisper_availability():
    """Whisper利用可能性テスト"""
    try:
        import whisper
        models = whisper.available_models()
        print(f"✓ Whisper利用可能: {models}")
        return True
    except ImportError:
        print("✗ Whisper未インストール")
        return False

def test_model_loading():
    """モデル読み込みテスト"""
    try:
        processor = SpeechProcessor()
        print("✓ SpeechProcessor初期化成功")
        return True
    except Exception as e:
        print(f"✗ 初期化エラー: {e}")
        return False

if __name__ == "__main__":
    print("=== 音声認識テスト ===")
    test_whisper_availability()
    test_model_loading()
EOF

cat > notebooks/speech_demo.ipynb << 'EOF'
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 音声認識デモ\n",
    "OpenAI Whisper + Ollama LLM"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import sys\n",
    "sys.path.append('../src')\n",
    "\n",
    "from speech_processor import SpeechProcessor\n",
    "import whisper"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Whisperモデル確認\n",
    "print(\"利用可能なWhisperモデル:\")\n",
    "print(whisper.available_models())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 音声処理初期化\n",
    "processor = SpeechProcessor(whisper_model=\"tiny\")\n",
    "print(\"音声処理初期化完了\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
EOF

print_success "プロジェクト構造作成完了"

# 10. GPU動作確認
echo ""
echo "10. GPU動作確認"
print_info "PyTorch GPU認識確認中..."
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU count: {torch.cuda.device_count()}')
    print(f'Current GPU: {torch.cuda.get_device_name()}')
else:
    print('GPU認識なし（CPUモード）')
"
print_success "GPU確認完了"

# 11. Whisper動作確認
echo ""
echo "11. Whisper動作確認"
print_info "Whisperモデル確認中..."
python3 -c "
import whisper
models = whisper.available_models()
print(f'利用可能なWhisperモデル: {models}')

# 軽量モデル読み込みテスト
try:
    model = whisper.load_model('tiny')
    print('✓ Whisper tinyモデル読み込み成功')
except Exception as e:
    print(f'✗ Whisperモデル読み込みエラー: {e}')
"
print_success "Whisper確認完了"

# 12. セットアップ完了
echo ""
echo "=========================================="
echo "🎉 音声認識環境セットアップ完了！"
echo "=========================================="
echo ""
echo "📁 プロジェクト構造:"
echo "$(pwd)"
tree -L 2 2>/dev/null || find . -type d -maxdepth 2 | sed 's/^/  /'
echo ""
echo "🚀 次のステップ:"
echo "1. 仮想環境有効化: source venv/bin/activate"
echo "2. 環境変数読み込み: source .env"
echo "3. テスト実行: python tests/test_speech.py"
echo "4. APIサーバー起動: python src/api_server.py"
echo "5. Jupyter起動: jupyter notebook notebooks/"
echo ""
echo "📋 利用可能なコマンド:"
echo "- テスト: python tests/test_speech.py"
echo "- API: python src/api_server.py"
echo "- デモ: jupyter notebook"
echo ""
echo "⚙️ 設定ファイル:"
echo "- 依存関係: requirements.txt"
echo "- 環境変数: .env"
echo "- プロジェクト: src/"
echo ""
print_success "セットアップスクリプト完了！"