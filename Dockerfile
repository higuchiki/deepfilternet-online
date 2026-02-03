# 1. 依存関係のインストール層
FROM python:3.11-slim AS builder

# システムの依存パッケージをインストール
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    libsox-dev \
    sox \
    libsox-fmt-all \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# requirements_cloud.txt をコピーしてインストール
COPY requirements_cloud.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 2. 実行層
FROM python:3.11-slim

# 実行に必要なシステムパッケージのみをインストール
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsox-dev \
    sox \
    libsox-fmt-all \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# builder層からインストール済みパッケージをコピー
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# プログラムをコピー
COPY . .

ENV PORT=8080
ENV PYTHONUNBUFFERED=1
EXPOSE 8080

CMD ["streamlit", "run", "web_enhance.py", "--server.port=8080", "--server.address=0.0.0.0"]
