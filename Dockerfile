FROM python:3.10-slim

WORKDIR /app

# 1. INSTALL DEPENDENSI SISTEM UNTUK OPENCV/YOLO
# libglib2.0-0 menyediakan libgthread-2.0.so.0
# libgl1, libsm6, libxext6 adalah dependensi visualisasi umum untuk OpenCV
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libgl1 \
    libsm6 \
    libxext6 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

ENV PORT 8080
ENV PYTHONUNBUFFERED 1

# 2. INSTALL DEPENDENSI PYTHON
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. SALIN APLIKASI
COPY app/ ./app

# 4. START SERVER
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]