from ultralytics import YOLO
import os
import cv2
import shutil

def convert_avi_to_mp4(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    
    # Get the video's properties (frame width, height, and FPS)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create a VideoWriter object for MP4 output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 format
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Read and write each frame to the new video file
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
    
    # Release resources
    cap.release()
    out.release()

def inference_objectDetection(user_id, project_id):
    project_dir = f"inference/{user_id}/{project_id}"
    temp_result_name = "temp-objectDetection"
    
    # Path Input dan Output
    video_path = f"{project_dir}/original_video.mp4"
    temp_yolo_output_dir = os.path.join(project_dir, temp_result_name)
    original_video_filename_base = os.path.basename(video_path).rsplit('.', 1)[0]
    temp_avi_path = os.path.join(temp_yolo_output_dir, f"{original_video_filename_base}.avi")
    final_mp4_path = os.path.join(project_dir, "objectDetection_video.mp4")

    model = YOLO("models/objectDetection/objectDetection.pt")

    try:
        results = model(
            source=video_path,
            stream=True, 
            save=True, 
            project=project_dir, 
            name=temp_result_name,
            exist_ok=True
        )

        for result in results:
            result.save()
        
        convert_avi_to_mp4(temp_avi_path, final_mp4_path)
        
        print(final_mp4_path)
        return final_mp4_path

    finally:
        if os.path.exists(temp_yolo_output_dir):
            shutil.rmtree(temp_yolo_output_dir)

def inference_playerKeyPoint(user_id, project_id):
    project_dir = f"inference/{user_id}/{project_id}"
    temp_result_name = "temp-playerKeyPoint"
    
    # Path Input dan Output
    video_path = f"{project_dir}/original_video.mp4"
    temp_yolo_output_dir = os.path.join(project_dir, temp_result_name)
    original_video_filename_base = os.path.basename(video_path).rsplit('.', 1)[0]
    temp_avi_path = os.path.join(temp_yolo_output_dir, f"{original_video_filename_base}.avi")
    final_mp4_path = os.path.join(project_dir, "playerKeyPoint_video.mp4")

    model = YOLO("models/playerKeyPoint/playerKeyPoint.pt")

    stroke_counts = {
        'forehand': 0,
        'backhand': 0,
        'serve': 0
    }

    class_names = model.names

    try:
        results = model(
            source=video_path,
            stream=True, 
            save=True, 
            project=project_dir, 
            name=temp_result_name,
            exist_ok=True
        )

        for result in results:
            detected_class_ids = result.boxes.cls.tolist()
            
            # Hitung setiap class ID
            for class_id in detected_class_ids:
                class_name = class_names[int(class_id)]
                
                if class_name in stroke_counts:
                    stroke_counts[class_name] += 1

            result.save()
        
        convert_avi_to_mp4(temp_avi_path, final_mp4_path)
        
        print(final_mp4_path)
        return {
            'path': final_mp4_path,
            'counts': stroke_counts
        }

    finally:
        if os.path.exists(temp_yolo_output_dir):
            shutil.rmtree(temp_yolo_output_dir)

# inference_objectDetection("a01760a7-6ca7-458a-88ec-369dfd0e8154","a017675f-ab61-4e2d-b105-25dc10684cb3")
# inference_playerKeyPoint("a01760a7-6ca7-458a-88ec-369dfd0e8154","a017675f-ab61-4e2d-b105-25dc10684cb3")