import time
import os
import shutil
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, UUID4

# Impor utilitas
from .gcs_utils import download_model, download_original_video, upload_video, upload_thumbnail
from .yolo_utils import inference_objectDetection, inference_playerKeyPoint
from .supabase_utils import get_link_original_video, update_project_details, get_user_info, get_project_info, update_projects, insert_inference_log
from .mailtrap_utils import send_success_email
from .common_utils import get_video_duration, extract_first_frame_as_thumbnail, get_hardware_info

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
    hw_info = get_hardware_info()
    
    local_project_root_dir = f"inference/{user_id}/{project_id}"
    local_video_input_path = os.path.join(local_project_root_dir, "original_video.mp4")
    
    # Inisialisasi variabel untuk penggunaan finally block
    link_thumbnail = None
    
    try:
        # 2. Ambil Data Proyek dan User
        username, email = get_user_info(user_id)
        project_name, project_details_id_actual = get_project_info(project_id)
        # Ambil link video asli dari Supabase
        link_original_video = get_link_original_video(project_details_id)['link_video_original']
        
        # 3. Download dan Ekstraksi
        download_original_video(bucket_name, user_id, project_id, link_original_video)
        
        path_thumbnail = extract_first_frame_as_thumbnail(local_video_input_path, os.path.join(local_project_root_dir, "thumbnails"))
        link_thumbnail = upload_thumbnail(bucket_name, user_id, project_id, path_thumbnail)
        
        # 4. Inferensi dan Penghitungan Waktu
        detection_time_start = time.time()
        path_video_object_detection = inference_objectDetection(user_id, project_id)
        detection_time_end = time.time()
        
        keypoint_time_start = time.time()
        keypoint_result = inference_playerKeyPoint(user_id, project_id)
        keypoint_time_end = time.time()
        
        # Ekstrak hasil
        path_video_player_key_point = keypoint_result['path']
        stroke_count = keypoint_result['counts']
        
        # Penghitungan akhir
        duration_seconds = get_video_duration(local_video_input_path)
        detection_inference_time = detection_time_end - detection_time_start
        keypoint_inference_time = keypoint_time_end - keypoint_time_start
        total_inference_time = keypoint_time_end - detection_time_start

        # 5. Upload Video Hasil
        link_video_object_detection = upload_video(bucket_name, "objectDetection", user_id, project_id, path_video_object_detection)
        link_video_keypoints = upload_video(bucket_name, "playerKeyPoint", user_id, project_id, path_video_player_key_point)


        # To Do
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
            int(duration_seconds), 
            int(total_inference_time),
            video_courtKeyPoint,
            image_heatmap_player
        ):
            send_success_email(
                username=username,
                project_name=project_name,
                project_id=project_id,
                receiver_email=email,
                video_duration=duration_seconds
            )
            
            update_projects(project_id, link_thumbnail, True)
            
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
                "success"
            )

        return {"status": "success", "message": "Inference and update complete."}
        
    except Exception as e:
        # Menangkap error dan memberikan respon HTTP yang benar
        print(f"FATAL ERROR during processing: {e}")
        
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
                "failed"
            )
        
        # Lakukan pembaruan status Supabase ke FAILED di sini (opsional tapi dianjurkan)
        raise HTTPException(status_code=500, detail=f"Inference processing failed: {e}")

    finally:
        # 7. Pembersihan (Cleanup)
        if os.path.exists(local_project_root_dir):
            shutil.rmtree(local_project_root_dir)

@app.get("/")
def health_check():
    hardware_info = get_hardware_info()
    
    return {
        "status": "ok", 
        "message": "Object Detection Service Running",
        "hardware": hardware_info
    }