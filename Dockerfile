# Gunakan base image dengan CUDA runtime + Ubuntu 22.04
FROM nvidia/cuda:13.0.1-cudnn-runtime-ubuntu24.04

# Tentukan working directory
WORKDIR /app

# Install dependensi sistem umum
RUN apt-get update && apt-get install -y \
    python3 python3-pip \
    libgl1 \
    libsm6 \
    libxext6 \
    ffmpeg \
    libjpeg-dev \
    libpng-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Pastikan Python default ke versi 3
RUN ln -sf python3 /usr/bin/python

# Set environment agar CUDA bisa diakses
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV PORT=8080
ENV PYTHONUNBUFFERED=1

# Copy dan install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir --break-system-packages -r requirements.txt

# (Opsional) Pastikan torch GPU terinstall
RUN pip install --break-system-packages torch torchvision --index-url https://download.pytorch.org/whl/cu126

# Copy seluruh kode aplikasi
COPY app/ ./app

# Jalankan aplikasi FastAPI
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
