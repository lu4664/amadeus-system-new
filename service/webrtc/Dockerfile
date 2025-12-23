FROM python:3.11-slim

RUN apt-get update && \
    apt-get install -y build-essential pkg-config clang git ffmpeg && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    python -c "import gradio; print('GRADIO=', gradio.__version__)"

COPY . .

EXPOSE 8001
CMD ["python", "server.py"]
