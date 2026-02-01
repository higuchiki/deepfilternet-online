# Python 3.11のスリム版を使用
FROM python:3.11-slim

# システムの依存パッケージ（ffmpeg, git, libsox-devなど）をインストール
RUN apt-get update && apt-get install -y \
    ffmpeg \
    curl \
    build-essential \
    git \
    libsox-dev \
    && rm -rf /var/lib/apt/lists/*

# 作業ディレクトリの設定
WORKDIR /app

# 依存ライブラリのコピーとインストール
COPY requirements_cloud.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# アプリケーションコードのコピー
COPY . .

# Streamlitのポート番号（デフォルト8080）を設定
ENV PORT=8080
EXPOSE 8080

# 実行コマンド
CMD streamlit run web_enhance.py --server.port=${PORT} --server.address=0.0.0.0
