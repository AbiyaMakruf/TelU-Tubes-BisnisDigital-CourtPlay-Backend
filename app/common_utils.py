import cv2
import os

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