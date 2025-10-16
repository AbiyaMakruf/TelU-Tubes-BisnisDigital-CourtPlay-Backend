import subprocess
import os
import logging
import psutil
import cpuinfo
import platform
import cv2
from torch import cuda

from .gcs_utils import download

# Konfigurasi Log
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def convert_avi_to_mp4(input_path, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Perintah FFmpeg
    command = [
        "ffmpeg",
        "-y",                  # overwrite file kalau sudah ada
        "-i", input_path,      # input file
        "-c:v", "libx264",     # codec video H.264 (web compatible)
        "-preset", "fast",     # kecepatan encoding (fast/balanced)
        "-crf", "23",          # quality (0=lossless, 23=default)
        "-pix_fmt", "yuv420p", # pixel format yang didukung HTML5 video
        output_path
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if result.returncode != 0:
        logger.error(f"❌ FFmpeg conversion failed!. {result.stderr.decode()}")
        raise RuntimeError("FFmpeg failed to convert video")

    logger.info("✅ Conversion avi to mp4 complete")

def get_hardware_inference_info():
    hw_info = {
        'gpu_name': 'N/A',
        'vram_mb': 0,
        'cpu_name': 'N/A',
        'cpu_threads': 0,
        'ram_mb': 0,
        'os_info': 'N/A'
    }

    try:
        if cuda.is_available():
            hw_info['gpu_name'] = cuda.get_device_name(0)
            hw_info['vram_mb'] = int(cuda.get_device_properties(0).total_memory / (1024 * 1024))

        cpu = cpuinfo.get_cpu_info()
        hw_info['cpu_name'] = cpu['brand_raw']
        hw_info['cpu_threads'] = psutil.cpu_count(logical=True)
        hw_info['ram_mb'] = int(psutil.virtual_memory().total / (1024 * 1024))

        os_info_str = platform.platform()
        if os.path.exists('/.dockerenv'):
             hw_info["os_info"] = f"Docker ({os_info_str})"
        else:
             hw_info["os_info"] = os_info_str         

        return hw_info
    
    except Exception as e:
        logger.error(f"Fail to get hwinfo. {e}")

def get_video_duration(original_video_path):
    try:
        cap = cv2.VideoCapture(original_video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        
        if fps > 0 and frame_count > 0:
            duration_seconds = frame_count / fps
            return int(round(duration_seconds)) 
        else:
            logger.error(f"FPS or Frame Count not valid for {original_video_path}")
            return 0
            
    except Exception as e:
        logger.error(f"ERROR while reading video duration: {e}")
        return 0
    
def extract_first_frame_as_thumbnail(video_path, thumbnail_dir):  
    os.makedirs(thumbnail_dir, exist_ok=True)
    video_filename_base = os.path.basename(video_path).rsplit('.', 1)[0]
    thumbnail_filename = f"{video_filename_base}_thumbnail.jpg"
    thumbnail_path = os.path.join(thumbnail_dir, thumbnail_filename)

    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise IOError(f"Failed to open video file at {video_path}")

    try:
        ret, frame = cap.read()
        
        if ret:
            success = cv2.imwrite(thumbnail_path, frame)
            
            if not success:
                 raise RuntimeError("cv2.imwrite failed to save the thumbnail image.")
                 
            return thumbnail_path
        else:
            raise ValueError("Could not read the first frame of the video.")
        
    except Exception as e:
        logger.error(f"Error extracting thumbnail: {e}")    

    finally:
        cap.release()

def download_if_model_not_exists(bucket_name,model_type):
    if os.path.exists(f"models/{model_type}/{model_type}.pt"):
        logger.info(f"Model {model_type} already exists")
    else:
        download(bucket_name,f"models/{model_type}/{model_type}.pt",f"models/{model_type}/{model_type}.pt")
        logger.info(f"Model {model_type} downloaded")

def get_original_video_name(link_video):
    return link_video[0]["link_video_original"].split("/")[-1]