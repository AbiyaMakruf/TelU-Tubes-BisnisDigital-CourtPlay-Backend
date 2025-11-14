import os
import logging
import base64
import threading
import google.cloud.logging
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from ultralytics import YOLO
from pydantic import BaseModel

# Import from /utils
from .utils.common_utils import download_if_model_not_exists, get_hardware_inference_info
from .utils.pull_process import pull_and_process
from .utils_v2.pull_process import pull_and_process as pull_and_process_v2

# Import from /service
from .service.inference_services import  set_global_models, process_inference_task

# Konfigurasi GCS
load_dotenv()
PROJECT_ID = os.getenv("GCP_PROJECT_ID") 
SUBSCRIPTION_ID = os.getenv("SUBSCRIPTION_ID")

# Konfigurasi Log
client = google.cloud.logging.Client()
client.setup_logging()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Konfigurasi Global
bucket_name = "courtplay-storage"
GLOBAL_MODELS = {
    "objectDetection": None,
    "playerKeyPoint": None,
    "courtKeyPoint": None,
}

# 1. Skema wrapper Pub/Sub
class PubSubMessage(BaseModel):
    message: dict
    subscription: str = None


# Start FastAPI app
app = FastAPI(title="Backend CourtPlay - Video Inference Worker")

@app.on_event("startup")
def startup_event():
    try:
        download_if_model_not_exists(bucket_name,"objectDetection")
        download_if_model_not_exists(bucket_name,"playerKeyPoint")
        download_if_model_not_exists(bucket_name,"courtKeyPoint")
        download_if_model_not_exists(bucket_name,"ballTracker")
        download_if_model_not_exists(bucket_name,"courtDetector")
        download_if_model_not_exists(bucket_name,"bounceDetector")

    except Exception as e:
        logger.error(f"Failed to download model. {e}")

    global GLOBAL_MODELS
    try:
        GLOBAL_MODELS["objectDetection"] = YOLO("models/objectDetection/objectDetection.pt")
        logger.info("Object Detection Model Loaded.")

        GLOBAL_MODELS["playerKeyPoint"] = YOLO("models/playerKeyPoint/playerKeyPoint.pt")
        logger.info("Player KeyPoint Model Loaded.")
        
        GLOBAL_MODELS["courtKeyPoint"] = YOLO("models/courtKeyPoint/courtKeyPoint.pt")
        logger.info("Court KeyPoint Model Loaded.")
        
        set_global_models(GLOBAL_MODELS)
        logger.info("All models successfully loaded to memory/VRAM.")

        thread = threading.Thread(
            # target=pull_and_process, 
            target=pull_and_process_v2, 
            args=("courtplay-analytics-474615","inference-pull",), 
            daemon=True)
        
        thread.start()

    except Exception as e:
        logger.error(f"FATAL ERROR: Failed to load models to VRAM. Check GPU resources/memory. {e}")
        raise RuntimeError(f"Model loading failed: {e}") from e


@app.get("/")
def health_check():
    hardware_info = get_hardware_inference_info()
    
    return {
        "status": "ok", 
        "message": f"YOLO Inference Worker Running",
        "hardware": hardware_info
    }