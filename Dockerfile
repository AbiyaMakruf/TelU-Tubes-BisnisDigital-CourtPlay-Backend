FROM nvidia/cuda:13.0.1-cudnn-runtime-ubuntu24.04

WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3 python3-pip \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    ffmpeg \
    libjpeg8 libpng-dev libtiff6 libwebp-dev libopenjp2-7-dev \
    libavcodec-dev libavformat-dev libswscale-dev libv4l-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf python3 /usr/bin/python

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV PORT=8080
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir --break-system-packages -r requirements.txt

RUN pip install --break-system-packages torch torchvision --index-url https://download.pytorch.org/whl/cu126

COPY app/ ./app

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
