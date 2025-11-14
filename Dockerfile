FROM nvidia/cuda:13.0.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip git \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    ffmpeg libx264-dev libopenh264-dev \
    libjpeg-dev libpng-dev libtiff-dev libwebp-dev libopenjp2-7-dev \
    libavcodec-dev libavformat-dev libswscale-dev libv4l-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf python3 /usr/bin/python

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV PORT=8080
ENV PYTHONUNBUFFERED=1

RUN python3 -m pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu130 --break-system-packages

COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir -r requirements.txt --break-system-packages

COPY app/ ./app

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]