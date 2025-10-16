import cv2
import os
from torch import cuda
import psutil
import cpuinfo
import platform

def get_hardware_info():
    hw_info = {
        'gpu_name': 'N/A',
        'vram_mb': 0,
        'cpu_name': 'N/A',
        'cpu_threads': 0,
        'ram_mb': 0,
        'os_info': 'N/A'
    }
    
    # GPU Name & VRAM
    try:
        if cuda.is_available():
            hw_info['gpu_name'] = cuda.get_device_name(0)
            hw_info['vram_mb'] = int(cuda.get_device_properties(0).total_memory / (1024 * 1024))
    except Exception as e:
        hw_info['gpu_name'] = f"Cannot get GPU name"
        hw_info['vram_mb'] = 0
    
    # CPU Name, Threads, RAM
    try:
        cpu = cpuinfo.get_cpu_info()
        hw_info['cpu_name'] = cpu['brand_raw']
        hw_info['cpu_threads'] = psutil.cpu_count(logical=True)
        hw_info['ram_mb'] = int(psutil.virtual_memory().total / (1024 * 1024))
    except Exception as e:
        hw_info['cpu_name'] = "Cannot get CPU name"
        hw_info['cpu_threads'] = 0
        hw_info['ram_mb'] = 0

    # OS Info
    try:
        os_info_str = platform.platform()
        if os.path.exists('/.dockerenv'):
             hw_info["os_info"] = f"Docker ({os_info_str})"
        else:
             hw_info["os_info"] = os_info_str
    except Exception as e:
        hw_info['os_info'] = "Cannot get OS info"
    
    return hw_info

def get_video_duration(video_path: str) -> int:
    try:
        cap = cv2.VideoCapture(video_path)
        
        # Ambil FPS (Frame Per Second)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Ambil jumlah frame total
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        
        cap.release()
        
        if fps > 0 and frame_count > 0:
            # Durasi = Total Frames / FPS
            duration_seconds = frame_count / fps
            # Bulatkan ke 2 desimal
            return int(round(duration_seconds)) 
        else:
            print(f"ERROR: FPS atau Frame Count tidak valid untuk {video_path}")
            return 0.0
            
    except Exception as e:
        print(f"ERROR saat membaca durasi video: {e}")
        return 0.0
    

def extract_first_frame_as_thumbnail(video_path: str, thumbnail_dir: str) -> str:    
    # 1. Tentukan Path Output Thumbnail
    os.makedirs(thumbnail_dir, exist_ok=True)
    video_filename_base = os.path.basename(video_path).rsplit('.', 1)[0]
    thumbnail_filename = f"{video_filename_base}_thumb.png"
    thumbnail_path = os.path.join(thumbnail_dir, thumbnail_filename)

    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise IOError(f"Failed to open video file at {video_path}")

    try:
        # 2. Baca Frame Pertama (Frame index 0)
        ret, frame = cap.read()
        
        if ret:
            # 3. Simpan Frame sebagai File Gambar (JPEG)
            # cv2.imwrite mengembalikan True jika berhasil disimpan
            success = cv2.imwrite(thumbnail_path, frame)
            
            if not success:
                 raise RuntimeError("cv2.imwrite failed to save the thumbnail image.")
                 
            return thumbnail_path
        else:
            raise ValueError("Could not read the first frame of the video.")
            
    finally:
        # Selalu lepaskan objek capture
        cap.release()