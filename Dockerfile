# 1. 依存関係のインストール層 (キャッシュ用)
FROM python:3.11-slim AS builder

# システムの依存パッケージをインストール
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    libsox-dev \
    sox \
    libsox-fmt-all \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# requirements.txt のみを先にコピーしてインストール (ここがキャッシュされる)
COPY requirements_cloud.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 2. 実行層
FROM python:3.11-slim

# 実行に必要なシステムパッケージのみをインストール
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    libsox-dev \
    sox \
    libsox-fmt-all \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# builder層でインストールしたライブラリをコピー
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# アプリケーションコードをコピー (コードの修正時はここから下が実行されるため高速)
COPY . .

ENV PORT=8080
EXPOSE 8080

CMD streamlit run web_enhance.py --server.port=${PORT} --server.address=0.0.0.0
