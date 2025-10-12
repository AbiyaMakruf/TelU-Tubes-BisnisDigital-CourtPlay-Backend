import time
import os
import shutil
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, UUID4

# Impor utilitas
from .gcs_utils import download_model, download_original_video, upload_video, upload_thumbnail
from .yolo_utils import inference_objectDetection, inference_playerKeyPoint
from .supabase_utils import get_link_original_video, update_project_details, get_user_info, get_project_info, update_projects
from .mailtrap_utils import send_success_email
from .common_utils import get_video_duration, extract_first_frame_as_thumbnail

app = FastAPI(title="YOLO Video Object Detection API")

def check_and_download_model(bucket_name: str, model_type: str):
    local_model_path = f"models/{model_type}/{model_type}.pt"
    
    if os.path.exists(local_model_path):
        print(f"Model {model_type} sudah ada di {local_model_path}. Melewati pengunduhan.")
        return
    else:
        print(f"Mengunduh model {model_type} dari GCS...")
        download_model(bucket_name=bucket_name, model_type=model_type)
        print(f"Model {model_type} berhasil diunduh.")

@app.on_event("startup")
def startup_event():
    try:
        print("Mengunduh model YOLO dari GCS...")

        bucket = "courtplay-storage"
        check_and_download_model(bucket, "objectDetection")
        check_and_download_model(bucket, "playerKeyPoint")
        check_and_download_model(bucket, "courtKeyPoint")

    except Exception as e:
        print(f"ERROR: Gagal memuat atau mengunduh model: {e}")


class InferenceRequest(BaseModel):
    user_id: UUID4
    project_id: UUID4
    project_details_id: UUID4

@app.post("/infer/")
def infer_video(payload: InferenceRequest):
    
    # 1. SETUP: Variabel Statis dan Path
    bucket_name = "courtplay-storage"
    user_id = str(payload.user_id)
    project_id = str(payload.project_id)
    project_details_id = str(payload.project_details_id)
    
    local_project_root_dir = f"inference/{user_id}/{project_id}"
    local_video_input_path = os.path.join(local_project_root_dir, "original_video.mp4")
    
    # Inisialisasi variabel untuk penggunaan finally block
    link_thumbnail = None
    
    try:
        # 2. Ambil Data Proyek dan User
        username, email = get_user_info(user_id)
        project_name, project_details_id_actual = get_project_info(project_id)
        # Ambil link video asli dari Supabase
        link_original_video = get_link_original_video(project_details_id)['link_original_video']
        
        # 3. Download dan Ekstraksi
        download_original_video(bucket_name, user_id, project_id, link_original_video)
        
        path_thumbnail = extract_first_frame_as_thumbnail(local_video_input_path, os.path.join(local_project_root_dir, "thumbnails"))
        link_thumbnail = upload_thumbnail(bucket_name, user_id, project_id, path_thumbnail)
        
        # 4. Inferensi dan Penghitungan Waktu
        time_start = time.time()
        path_video_object_detection = inference_objectDetection(user_id, project_id)
        keypoint_result = inference_playerKeyPoint(user_id, project_id)
        time_end = time.time()
        
        # Ekstrak hasil
        path_video_player_key_point = keypoint_result['path']
        stroke_count = keypoint_result['counts']
        
        # Penghitungan akhir
        duration_seconds = get_video_duration(local_video_input_path)
        inference_time = time_end - time_start

        # 5. Upload Video Hasil
        link_video_object_detection = upload_video(bucket_name, "objectDetection", user_id, project_id, path_video_object_detection)
        link_video_keypoints = upload_video(bucket_name, "playerKeyPoint", user_id, project_id, path_video_player_key_point)

        # 6. Update Supabase dan Notifikasi
        if update_project_details(
            project_details_id, 
            link_video_object_detection, 
            link_video_keypoints, 
            link_thumbnail, # Menggunakan link_thumbnail yang sudah di-upload
            int(stroke_count['forehand']), 
            int(stroke_count['backhand']), 
            int(stroke_count['serve']), 
            int(duration_seconds), 
            int(inference_time)
        ):
            send_success_email(
                username=username,
                project_name=project_name,
                project_id=project_id,
                receiver_email=email,
                video_duration=duration_seconds
            )
            update_projects(project_id, link_thumbnail, True)

        return {"status": "success", "message": "Inference and update complete."}
        
    except Exception as e:
        # Menangkap error dan memberikan respon HTTP yang benar
        print(f"FATAL ERROR during processing: {e}")
        # Lakukan pembaruan status Supabase ke FAILED di sini (opsional tapi dianjurkan)
        raise HTTPException(status_code=500, detail=f"Inference processing failed: {e}")

    finally:
        # 7. Pembersihan (Cleanup)
        if os.path.exists(local_project_root_dir):
            shutil.rmtree(local_project_root_dir)


def get_gpu_info():
    """Mendeteksi ketersediaan dan tipe GPU."""
    try:
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            
            # Mendapatkan nama GPU pertama
            gpu_name = torch.cuda.get_device_name(0) 
            
            return {
                "using_gpu": True,
                "gpu_count": device_count,
                "gpu_name": gpu_name
            }
        else:
            return {
                "using_gpu": False,
                "detail": "CUDA device not found. Running on CPU."
            }
    except Exception as e:
        # Menangani jika PyTorch tidak terinstal atau gagal inisialisasi CUDA
        return {
            "using_gpu": False,
            "detail": f"GPU check failed (PyTorch not functional): {e}"
        }

@app.get("/")
def health_check():
    gpu_status = get_gpu_info()
    
    return {
        "status": "ok", 
        "message": "Object Detection Service Running",
        "hardware": gpu_status
    }