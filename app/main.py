import os
import logging
import threading
from dotenv import load_dotenv
from fastapi import FastAPI


# Import from /utils
from .utils.common_utils import download_if_model_not_exists, get_hardware_inference_info

# Import from /service
from .service.inference_services import pull_messages

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

    listener_thread = threading.Thread(target=pull_messages(PROJECT_ID, SUBSCRIPTION_ID, MAX_MESSAGES_PER_PULL), daemon=True)
    listener_thread.start()

@app.get("/")
def health_check():
    hardware_info = get_hardware_inference_info()
    
    return {
        "status": "ok", 
        "message": f"YOLO Inference Worker Running (PARALLEL Mode, Max Threads: {MAX_MESSAGES_PER_PULL})",
        "hardware": hardware_info
    }