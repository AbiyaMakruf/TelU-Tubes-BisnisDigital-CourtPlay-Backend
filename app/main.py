import os
import logging
import threading
from dotenv import load_dotenv
from fastapi import FastAPI
from ultralytics import YOLO

# Import from /utils
from .utils.common_utils import download_if_model_not_exists, get_hardware_inference_info

# Import from /service
from .service.inference_services import pull_messages, set_global_models

# Konfigurasi GCS
load_dotenv()
PROJECT_ID = os.getenv("GCP_PROJECT_ID") 
SUBSCRIPTION_ID = os.getenv("SUBSCRIPTION_ID")
MAX_MESSAGES_PER_PULL = int(os.getenv("MAX_CONCURRENT_MESSAGES"))

# Konfigurasi Log
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Konfigurasi Global
bucket_name = "courtplay-storage"
GLOBAL_MODELS = {
    "objectDetection": None,
    "playerKeyPoint": None,
    "courtKeyPoint": None,
}

# Start FastAPI app
app = FastAPI(title="Backend CourtPlay - Video Inference Worker")

@app.on_event("startup")
def startup_event():
    try:
        download_if_model_not_exists(bucket_name,"objectDetection")
        download_if_model_not_exists(bucket_name,"playerKeyPoint")
        download_if_model_not_exists(bucket_name,"courtKeyPoint")

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

    except Exception as e:
        logger.error(f"FATAL ERROR: Failed to load models to VRAM. Check GPU resources/memory. {e}")
        raise RuntimeError(f"Model loading failed: {e}") from e

    listener_thread = threading.Thread(target=pull_messages, args=(PROJECT_ID, SUBSCRIPTION_ID, MAX_MESSAGES_PER_PULL), daemon=True)
    logger.info(f"Starting Pub/Sub Listener Thread. Max Concurrent Messages: {MAX_MESSAGES_PER_PULL}")
    listener_thread.start()

@app.get("/")
def health_check():
    hardware_info = get_hardware_inference_info()
    
    # Menambahkan line code untuk menguji coba seberapa cepat hasil caching github action
    return {
        "status": "ok", 
        "message": f"YOLO Inference Worker Running (PARALLEL Mode, Max Threads: {MAX_MESSAGES_PER_PULL})",
        "hardware": hardware_info
    }