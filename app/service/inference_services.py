import json
import logging
import time
import shutil
from pydantic import BaseModel, UUID4
from google.cloud import pubsub_v1

# Import from /utils
from ..utils.common_utils import get_hardware_inference_info, get_original_video_name, extract_first_frame_as_thumbnail, get_video_duration
from ..utils.supabase_utils import *
from ..utils.gcs_utils import download, upload
from ..utils.yolo_utils import inference_objectDetection, inference_playerKeyPoint
from ..utils.mailtrap_utils import send_success_analysis_video

# Konfigurasi Log
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_payload_data(payload_data):
    class InferencePayload(BaseModel):
        id: UUID4
        user_id: UUID4
        project_details_id: UUID4

    try:
        validated_payload = InferencePayload(**payload_data)
        project_id = str(validated_payload.id)
        user_id = str(validated_payload.user_id)
        project_details_id = str(validated_payload.project_details_id)
        return {
            "project_id": project_id,
            "user_id": user_id,
            "project_details_id": project_details_id
        }
    except Exception as e:
        logger.error(f"Payload validation failed: {e}. Data received: {payload_data}")


def process_inference_task(payload: dict):
    """
    Fungsi inti yang menangani seluruh alur kerja inferensi video, 
    dipanggil oleh setiap thread yang dibuat.
    """
    hw_info = get_hardware_inference_info()

    try:
        payload_data = get_payload_data(payload)
        logger.info(f'Starting inference project {payload_data["project_id"]}')
        
        # 1. Get user and project data
        user_data = get("users", "username, email", {'id':payload_data["user_id"]}, False)
        project_data = get("projects", "project_name, project_details_id, upload_date", {'id':payload_data["project_id"]}, False)
        link_original_video = get("project_details","link_video_original", {'id':project_data[0]["project_details_id"]})
        
        # 2. Download original video from GCS        
        download("courtplay-storage",
                 f"uploads/videos/{payload_data['user_id']}/{payload_data['project_id']}/{get_original_video_name(link_original_video)}",
                 f"inference/{payload_data['user_id']}/{payload_data['project_id']}/original_video.mp4"
                 )
        video_duration = get_video_duration(f"inference/{payload_data['user_id']}/{payload_data['project_id']}/original_video.mp4")

        # 3. Extract thumbnail and upload to GCS
        thumbnail_path = extract_first_frame_as_thumbnail(f"inference/{payload_data['user_id']}/{payload_data['project_id']}/original_video.mp4", f"inference/{payload_data['user_id']}/{payload_data['project_id']}")
        thumbnail_public_link = upload("courtplay-storage", thumbnail_path, f"uploads/videos/{payload_data['user_id']}/{payload_data['project_details_id']}/thumbnail.jpg")

        # 4. Inference ObjectDetection
        detection_time_start = time.time()
        video_object_detection_path = inference_objectDetection(payload_data["user_id"], payload_data["project_id"])
        detection_time_end = time.time()
        detection_inference_time = detection_time_end - detection_time_start
        
        # 5. Inference PlayerKeyPoint 
        keypoint_time_start = time.time()
        keypoint_result = inference_playerKeyPoint(payload_data["user_id"], payload_data["project_id"])
        keypoint_time_end = time.time()
        keypoint_inference_time = keypoint_time_end - keypoint_time_start
        
        # 6. Extract inference playerKeyPoint results
        total_inference_time = detection_inference_time + keypoint_inference_time
        video_player_keypoint_path = keypoint_result['video_mp4_format_path']
        stroke_count = keypoint_result['stroke_counts']
        
        # 7. Upload video to GCS
        video_object_detection_public_link = upload("courtplay-storage",video_object_detection_path,f"uploads/videos/{payload_data['user_id']}/{payload_data['project_details_id']}/objectDetection.mp4")
        video_player_keypoint_public_link = upload("courtplay-storage",video_player_keypoint_path,f"uploads/videos/{payload_data['user_id']}/{payload_data['project_details_id']}/playerKeyPoint.mp4")

        # Placeholder untuk To Do
        video_court_keypoint_public_link = "IN DEVELOPMENT"
        image_heatmap_player_public_link = "IN DEVELOPMENT"
        image_ball_droppings_public_link = "IN DEVELOPMENT"

        # 8. Update data in Supabase
        data_update_project_details = {
            "link_video_object_detections":video_object_detection_public_link,
            "link_video_player_keypoints": video_player_keypoint_public_link,
            "link_video_court_keypoints": video_court_keypoint_public_link,
            "link_image_ball_droppings": image_ball_droppings_public_link,
            "link_image_heatmap_player": image_heatmap_player_public_link,
            "ready_position_count":stroke_count["Ready_Position"],
            "forehand_count":stroke_count["Forehand"],
            "backhand_count":stroke_count["Backhand"],
            "serve_count":stroke_count["Serve"],
            "video_duration":int(video_duration),
            "video_processing_time":int(total_inference_time),
            "updated_at":"now()"
        }

        data_update_projets = {
            "link_image_thumbnail": thumbnail_public_link,
            "is_mailed": True,
            "updated_at":"now()"
        }

        data_update_hwinfo = {
            "detection_inference_time":int(detection_inference_time),
            "keypoint_inference_time":int(keypoint_inference_time),
            "total_inference_time":int(total_inference_time),
            "gpu_name":hw_info['gpu_name'],
            "vram_mb":hw_info['vram_mb'],
            "cpu_name":hw_info['cpu_name'],
            "cpu_threads":hw_info['cpu_threads'],
            "ram_mb":hw_info['ram_mb'],
            "os_info":hw_info['os_info'],
            "is_success":True
        }

        update("project_details",data_update_project_details,{"id":project_data[0]["project_details_id"]} )
        update("hwinfo",data_update_hwinfo,{"project_id":payload_data["project_id"]} )
        

        context_email = {
            "username": user_data[0]["username"],
            "project_name": project_data[0]["project_name"],
            "video_duration": video_duration,
            "upload_date": project_data[0]["upload_date"],
            "report_url": f"courtplay.my.id/analytics/{payload_data['project_id']}"
        }

        if send_success_analysis_video(user_data[0]["email"], context_email):
            update("projects",data_update_projets,{"id":payload_data["project_id"]} )
            
            logger.info(f'Processing Project ID {payload_data["project_id"]} completed and email sent to {user_data[0]["email"]}')

        update("hwinfo",data_update_hwinfo,{"project_id":payload_data["project_id"]})
            
    except Exception as e:
        logger.error(f"Error during processing Project ID {payload_data['project_id']}: {e}")
        raise e

    finally:
        if os.path.exists(f"inference/{payload_data['user_id']}/{payload_data['project_id']}"):
            shutil.rmtree(f"inference/{payload_data['user_id']}/{payload_data['project_id']}")
        

def pubsub_callback(message):
    """
    Dipanggil oleh Pub/Sub SDK saat pesan diterima. Berjalan di thread pool.
    """
    message_id = message.message_id
    
    try:
        payload_data = json.loads(message.data.decode("utf-8"))
        process_inference_task(payload_data)
        message.ack()
        logger.info(f"Message ID {message_id} processed and ACKed.")

    except Exception as e:
        logger.error(f"Error processing message ID {message_id}: {e}")
        message.nack()
        logger.info(f"Message ID {message_id} NACKed for redelivery.")

def pull_messages(PROJECT_ID, SUBSCRIPTION_ID, MAX_MESSAGES_PER_PULL):
    """
    Memulai proses mendengarkan pesan dari Pub/Sub menggunakan metode Streaming Pull 
    """
    subscriber = pubsub_v1.SubscriberClient()
    subscription_path = subscriber.subscription_path(PROJECT_ID, SUBSCRIPTION_ID)

    # Menggunakan flow control untuk membatasi jumlah pesan (thread) yang diproses secara paralel
    flow_control = pubsub_v1.types.FlowControl(max_messages=MAX_MESSAGES_PER_PULL)

    streaming_pull_future = subscriber.subscribe(
        subscription_path, 
        callback=pubsub_callback,
        flow_control=flow_control
    )
    
    try:
        streaming_pull_future.result() 
    except TimeoutError:
        streaming_pull_future.cancel()
        logger.warning("Streaming Pull Timeout (Unexpected).")
    except Exception as e:
        logger.error(f"Error in Pub/Sub Streaming Worker: {e}")
        streaming_pull_future.cancel()
        
    subscriber.close()