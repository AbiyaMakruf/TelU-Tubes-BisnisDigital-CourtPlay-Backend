import os
import shutil
import logging
from ultralytics import YOLO

from .common_utils import convert_avi_to_mp4

# Konfigurasi Log
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def inference_objectDetection(user_id, project_id):
    project_path = f"inference/{user_id}/{project_id}"
    temporary_path = "temporary_objectDetection"
    video_original_path = f"{project_path}/original_video.mp4"
    video_avi_format_path = os.path.join(project_path, temporary_path, "original_video.avi")
    video_mp4_format_path = os.path.join(project_path, "objectDetection_video.mp4")

    # model = YOLO("models/objectDetection/objectDetection.pt")
    model = None

    try:
        results = model(
            source=video_original_path,
            stream=True,
            save=True,
            project=project_path,
            name=temporary_path,
            exist_ok=True
        )

        for result in results:
            result.save(filename="image.png")

        convert_avi_to_mp4(video_avi_format_path, video_mp4_format_path)

        logger.info("Success inference object detection")
        return video_mp4_format_path
    
    except Exception as e:
        logger.error("Error inference object detection")

    finally:
        delete_path = os.path.join(project_path,temporary_path)
        if os.path.exists(delete_path):
            shutil.rmtree(delete_path)

def inference_playerKeyPoint(user_id, project_id):
    project_path = f"inference/{user_id}/{project_id}"
    temporary_path = "temporary_playerKeyPoint"
    video_original_path = f"{project_path}/original_video.mp4"
    video_avi_format_path = os.path.join(project_path, temporary_path, "original_video.avi")
    video_mp4_format_path = os.path.join(project_path, "playerKeyPoint_video.mp4")

    # model = YOLO("models/playerKeyPoint/playerKeyPoint.pt")
    model = None

    stroke_counts = {
        'Backhand': 0,
        'Forehand': 0,
        'Ready_Position': 0,
        'Serve': 0
    }

    try:
        results = model(
            source=video_original_path,
            stream=True,
            save=True,
            project=project_path,
            name=temporary_path,
            exist_ok=True
        )

        for result in results:
            detected_class_ids = result.boxes.cls.tolist()

            class_names = model.names

            for class_id in detected_class_ids:
                class_name = class_names[int(class_id)]
                
                if class_name in stroke_counts:
                    stroke_counts[class_name] += 1

            result.save(filename="image.png")

        convert_avi_to_mp4(video_avi_format_path, video_mp4_format_path)

        logger.info("Success inference player keypoint")

        return {
            'video_mp4_format_path':video_mp4_format_path,
            'stroke_counts': stroke_counts
        }
    
    except Exception as e:
        logger.error("Error inference object detection")

    finally:
        delete_path = os.path.join(project_path,temporary_path)
        if os.path.exists(delete_path):
            shutil.rmtree(delete_path)


# # To Do
# - Heatmap player
# - Ball drop heatmap