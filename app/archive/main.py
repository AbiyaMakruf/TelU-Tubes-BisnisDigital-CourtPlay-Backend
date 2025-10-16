import time
import os
import shutil
import json
import threading
import torch
from concurrent.futures import TimeoutError
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, UUID4
from google.cloud import pubsub_v1
import logging
from dotenv import load_dotenv

load_dotenv()

# --- Konfigurasi Log ---
# Atur log level untuk melihat aktivitas Pub/Sub dan error
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Variabel Lingkungan ---
PROJECT_ID = os.getenv("GCP_PROJECT_ID") 
SUBSCRIPTION_ID = os.getenv("SUBSCRIPTION_ID") 
# PUBSUB_TIMEOUT tidak digunakan dalam mode streaming, tetapi dipertahankan untuk referensi
PUBSUB_TIMEOUT = 5.0 
# Jumlah pesan yang dapat diproses secara bersamaan (paralel) oleh worker:
# Nilai ini mendefinisikan batas atas jumlah thread paralel yang akan dibuat.
# Default adalah 1 (mode serial aman).
MAX_MESSAGES_PER_PULL = int(os.getenv("MAX_CONCURRENT_MESSAGES", 1)) 
bucket_name = "courtplay-storage"

# Impor utilitas (diasumsikan sudah ada di direktori yang sama)
from .gcs_utils import download_model, download_original_video, upload_video, upload_thumbnail
from .yolo_utils import inference_objectDetection, inference_playerKeyPoint
from .supabase_utils import get_link_original_video, update_project_details, get_user_info, get_project_info, update_projects, insert_inference_log
from .mailtrap_utils import send_success_email
from .common_utils import get_video_duration, extract_first_frame_as_thumbnail, get_hardware_info

app = FastAPI(title="YOLO Pub/Sub Video Inference Worker (PARALLEL - Streaming)")

# --- Model Data untuk Validasi Pesan Masuk ---
class InferencePayload(BaseModel):
    user_id: UUID4
    project_id: UUID4
    project_details_id: UUID4

class InferencePayload(BaseModel):
    id: UUID4
    user_id: UUID4
    project_details_id: UUID4

# --- Logika Proses Inferensi Inti ---

def process_inference_task(payload_data: dict):
    """
    Fungsi inti yang menangani seluruh alur kerja inferensi video, 
    dipanggil oleh setiap thread yang dibuat.
    """
    try:
        validated_payload = InferencePayload(**payload_data)
    except Exception as e:
        logger.error(f"Payload validation failed: {e}. Data received: {payload_data}")
        return # Abaikan pesan jika payload tidak valid

    # 1. SETUP: Variabel dan Inisialisasi Waktu
    user_id = str(validated_payload.user_id)
    project_id = str(validated_payload.id)
    project_details_id = str(validated_payload.project_details_id)
    hw_info = get_hardware_info()

    local_project_root_dir = f"inference/{user_id}/{project_id}"
    local_video_input_path = os.path.join(local_project_root_dir, "original_video.mp4")

    # Inisialisasi variabel waktu dan tautan untuk finally/error block
    link_thumbnail = None
    detection_inference_time = 0
    keypoint_inference_time = 0
    total_inference_time = 0
    inference_successful = False

    try:
        logger.info(f"Memulai pemrosesan (PARALEL) untuk Project ID: {project_id}")
        
        # 2. Ambil Data Proyek dan User
        username, email = get_user_info(user_id)
        project_name, project_details_id_actual = get_project_info(project_id)
        link_original_video = get_link_original_video(project_details_id)['link_video_original']
        
        # 3. Download dan Ekstraksi
        download_original_video(bucket_name, user_id, project_id, link_original_video)
        
        path_thumbnail = extract_first_frame_as_thumbnail(local_video_input_path, os.path.join(local_project_root_dir, "thumbnails"))
        link_thumbnail = upload_thumbnail(bucket_name, user_id, project_id, path_thumbnail)
        
        # 4. Inferensi dan Penghitungan Waktu
        detection_time_start = time.time()
        path_video_object_detection = inference_objectDetection(user_id, project_id)
        detection_time_end = time.time()
        detection_inference_time = detection_time_end - detection_time_start
        
        keypoint_time_start = time.time()
        keypoint_result = inference_playerKeyPoint(user_id, project_id)
        keypoint_time_end = time.time()
        keypoint_inference_time = keypoint_time_end - keypoint_time_start
        total_inference_time = keypoint_time_end - detection_time_start
        
        # Ekstrak hasil
        path_video_player_key_point = keypoint_result['path']
        stroke_count = keypoint_result['counts']
        
        # 5. Upload Video Hasil
        link_video_object_detection = upload_video(bucket_name, "objectDetection", user_id, project_id, path_video_object_detection)
        link_video_keypoints = upload_video(bucket_name, "playerKeyPoint", user_id, project_id, path_video_player_key_point)

        # Placeholder untuk To Do
        video_courtKeyPoint = "http://test"
        image_heatmap_player = "http://test"

        # 6. Update Supabase dan Notifikasi
        if update_project_details(
            project_details_id, 
            link_video_object_detection, 
            link_video_keypoints, 
            link_thumbnail,
            int(stroke_count['Forehand']), 
            int(stroke_count['Backhand']), 
            int(stroke_count['Serve']),
            int(stroke_count['Ready_Position']), 
            int(get_video_duration(local_video_input_path)),
            int(total_inference_time),
            video_courtKeyPoint,
            image_heatmap_player
        ):
            send_success_email(
                username=username,
                project_name=project_name,
                project_id=project_id,
                receiver_email=email,
                video_duration=get_video_duration(local_video_input_path)
            )
            
            update_projects(project_id, link_thumbnail, True)
            inference_successful = True
            logger.info(f"Pemrosesan Project ID {project_id} selesai dan Supabase diperbarui.")
            
    except Exception as e:
        logger.error(f"FATAL ERROR during processing Project ID {project_id}: {e}")
        inference_successful = False

    finally:
        # 7. Pembersihan (Cleanup)
        if os.path.exists(local_project_root_dir):
            shutil.rmtree(local_project_root_dir)
            logger.info(f"Direktori lokal {local_project_root_dir} dibersihkan.")
        
        # 8. Logging ke Supabase (berjalan terlepas dari hasil sukses/gagal)
        insert_inference_log(
            project_id,
            user_id,
            int(detection_inference_time),
            int(keypoint_inference_time),
            int(total_inference_time),
            hw_info['gpu_name'],
            hw_info['vram_mb'],
            hw_info['cpu_name'],
            hw_info['cpu_threads'],
            hw_info['ram_mb'],
            hw_info['os_info'],
            inference_successful
        )

# --- Fungsi Callback untuk Streaming Pull ---

def pubsub_callback(message):
    """
    Dipanggil oleh Pub/Sub SDK saat pesan diterima. Berjalan di thread pool.
    """
    message_id = message.message_id
    
    try:
        # Pesan Pub/Sub adalah bytes, harus didecode dan diubah ke JSON
        payload_data = json.loads(message.data.decode("utf-8"))
        
        # Panggil fungsi inti pemrosesan
        process_inference_task(payload_data)

        # Kirim konfirmasi (ACK) setelah proses selesai
        message.ack()
        logger.info(f"Pesan ID {message_id} berhasil diproses dan dikonfirmasi (ACK).")

    except Exception as e:
        logger.error(f"Gagal memproses pesan {message_id}. Error: {e}")
        # Pesan akan dikirim ulang setelah ack deadline jika tidak ada NACK eksplisit.
        # Untuk kasus fatal, biarkan pesan NACK (dikirim ulang)
        message.nack()
        logger.info(f"Pesan ID {message_id} di NACK untuk pengiriman ulang.")


# --- Fungsi Listener Pub/Sub (Diubah ke Streaming Pull) ---

def pull_messages():
    """
    Memulai proses mendengarkan pesan dari Pub/Sub menggunakan metode Streaming Pull 
    yang lebih stabil dan menggunakan flow control untuk paralelisme.
    """
    subscriber = pubsub_v1.SubscriberClient()
    subscription_path = subscriber.subscription_path(PROJECT_ID, SUBSCRIPTION_ID)

    # Menggunakan flow control untuk membatasi jumlah pesan (thread) yang diproses secara paralel
    flow_control = pubsub_v1.types.FlowControl(max_messages=MAX_MESSAGES_PER_PULL)
    
    logger.info(f"Mulai mendengarkan pesan secara PARALEL (Streaming Pull) di {subscription_path}")
    logger.info(f"Batas thread/pesan paralel (Flow Control): {MAX_MESSAGES_PER_PULL}")

    # subscriber.subscribe menggunakan thread pool bawaan untuk memanggil callback
    streaming_pull_future = subscriber.subscribe(
        subscription_path, 
        callback=pubsub_callback,
        flow_control=flow_control # Mengatur batas paralelisme
    )
    
    # Blokir thread ini untuk menjaga koneksi tetap terbuka selamanya
    try:
        streaming_pull_future.result() 
    except TimeoutError:  # Ini seharusnya tidak terjadi pada result() tanpa argumen
        streaming_pull_future.cancel()
        logger.warning("Streaming Pull Timeout (Unexpected).")
    except Exception as e:
        logger.error(f"FATAL ERROR di Pub/Sub Streaming Worker: {e}")
        streaming_pull_future.cancel()
        
    subscriber.close()
    logger.info("Pub/Sub listener dimatikan.")

# --- FastAPI Hooks dan Endpoint yang Tersisa ---

@app.on_event("startup")
def startup_event():
    # 1. Unduh Model (Logika Asli)
    try:
        logger.info("Mengunduh model YOLO dari GCS...")
        check_and_download_model(bucket_name, "objectDetection")
        check_and_download_model(bucket_name, "playerKeyPoint")
        check_and_download_model(bucket_name, "courtKeyPoint")
        logger.info("Semua model berhasil dimuat.")
    except Exception as e:
        logger.error(f"ERROR: Gagal memuat atau mengunduh model: {e}")

    # 2. Mulai Listener Pub/Sub di Background Thread
    # Background Thread ini sekarang menjalankan loop serial pull
    listener_thread = threading.Thread(target=pull_messages, daemon=True)
    listener_thread.start()
    logger.info("Background thread Pub/Sub listener PARALEL telah dimulai.")

# Endpoint /infer yang lama dihapus. Sekarang hanya ada Health Check.
@app.get("/")
def health_check():
    """Endpoint untuk memeriksa apakah layanan berjalan dan status hardware."""
    hardware_info = get_hardware_info()
    
    return {
        "status": "ok", 
        "message": f"YOLO Inference Worker Running (PARALLEL Mode, Max Threads: {MAX_MESSAGES_PER_PULL})",
        "hardware": hardware_info
    }

# --- Fungsi Model (Dipindahkan dari Logic Asli) ---

def check_and_download_model(bucket_name: str, model_type: str):
    """Fungsi pembantu untuk memeriksa dan mengunduh model."""
    local_model_path = f"models/{model_type}/{model_type}.pt"
    
    if os.path.exists(local_model_path):
        logger.info(f"Model {model_type} sudah ada. Melewati pengunduhan.")
        return
    else:
        logger.info(f"Mengunduh model {model_type} dari GCS...")
        download_model(bucket_name=bucket_name, model_type=model_type) 
        logger.info(f"Model {model_type} berhasil diunduh.")
