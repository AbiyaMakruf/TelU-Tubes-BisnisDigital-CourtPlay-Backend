import os
import shutil
import logging
import threading
import google.cloud.logging

from .common_utils import convert_avi_to_mp4

# Konfigurasi lock
CONVERSION_LOCK = threading.Lock()
OD_LOCK = threading.Lock()
PKP_LOCK = threading.Lock()

# Konfigurasi Log
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def inference_objectDetection(user_id, project_id, model):
    project_path = os.path.join("inference",f"{user_id}", f"{project_id}")
    temporary_path = "temporary_objectDetection"
    video_original_path = f"{project_path}/original_video.mp4"
    video_avi_format_path = os.path.join(project_path, temporary_path, "original_video.avi")
    video_mp4_format_path = os.path.join(project_path, "objectDetection_video.mp4")

    # model = YOLO("models/objectDetection/objectDetection.pt")

    try:
        logger.info(f"Starting inference object detection. {project_id}")
        
        with OD_LOCK:
            logger.info(f"🔒Acquired lock for object detection inference. {project_id}")
            results = model(
                source=video_original_path,
                stream=True,
                save=True,
                project=project_path,
                name=temporary_path,
                # verbose=False,
                exist_ok=True
            )

            for result in results:
                result.save(filename=f"OD-{project_id}.png")

            logger.info(f"🔓Released lock for object detection inference. {project_id}")

        logger.info(f"Converting avi to mp4 format. {project_id}")
        with CONVERSION_LOCK:
            logger.info(f"🔒Acquired lock for video conversion OD. {project_id}")
            convert_avi_to_mp4(video_avi_format_path, video_mp4_format_path)
            logger.info(f"🔓Released lock for video conversion OD. {project_id}")
        os.remove(f"OD-{project_id}.png")

        logger.info(f"Success inference object detection. {project_id}")
        return video_mp4_format_path
    
    except Exception as e:
        logger.error(f"Error inference object detection. {e}")

    finally:
        delete_path = os.path.join(project_path,temporary_path)
        if os.path.exists(delete_path):
            shutil.rmtree(delete_path)

def inference_playerKeyPoint(user_id, project_id, model):
    project_path = os.path.join("inference",f"{user_id}", f"{project_id}")
    temporary_path = "temporary_playerKeyPoint"
    video_original_path = f"{project_path}/original_video.mp4"
    video_avi_format_path = os.path.join(project_path, temporary_path, "original_video.avi")
    video_mp4_format_path = os.path.join(project_path, "playerKeyPoint_video.mp4")

    # model = YOLO("models/playerKeyPoint/playerKeyPoint.pt")
    
    stroke_counts = {
        'Backhand': 0,
        'Forehand': 0,
        'Ready_Position': 0,
        'Serve': 0
    }

    try:
        logger.info(f"Starting inference player keypoint. {project_id}")

        with PKP_LOCK:
            logger.info(f"🔒Acquired lock for player keypoint inference. {project_id}")
            results = model(
                source=video_original_path,
                stream=True,
                save=True,
                project=project_path,
                name=temporary_path,
                # verbose=False,
                exist_ok=True
            )

            for result in results:
                detected_class_ids = result.boxes.cls.tolist()

                class_names = model.names

                for class_id in detected_class_ids:
                    class_name = class_names[int(class_id)]
                    
                    if class_name in stroke_counts:
                        stroke_counts[class_name] += 1

                result.save(filename=f"PKP-{project_id}.png")
            logger.info(f"🔓Released lock for player keypoint inference. {project_id}")

        logger.info(f"Converting avi to mp4 format. {project_id}")
        with CONVERSION_LOCK:
            logger.info(f"🔒Acquired lock for video conversion PKP. {project_id}")
            convert_avi_to_mp4(video_avi_format_path, video_mp4_format_path)
            logger.info(f"🔓Released lock for video conversion PKP. {project_id}")
        os.remove(f"PKP-{project_id}.png")

        logger.info(f"Success inference player keypoint. {project_id}")

        return {
            'video_mp4_format_path':video_mp4_format_path,
            'stroke_counts': stroke_counts
        }
    
    except Exception as e:
        logger.error(f"Error inference player keypoint. {e}")

    finally:
        delete_path = os.path.join(project_path,temporary_path)
        if os.path.exists(delete_path):
            shutil.rmtree(delete_path)
            
# # To Do
# - Heatmap player
# - Ball drop heatmap