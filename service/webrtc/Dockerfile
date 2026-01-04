FROM python:3.12-slim
# 安装系统依赖（包含 CA 证书）
# 清除缓存 - 2026-01-04 18:30
RUN apt-get update && \
    apt-get install -y ca-certificates build-essential pkg-config clang git ffmpeg && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    python -c "import gradio; print('GRADIO=', gradio.__version__)"

COPY . .

EXPOSE 8080
CMD ["python", "server.py"]
