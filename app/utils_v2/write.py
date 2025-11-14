import cv2
import os
import subprocess
from typing import Optional
def write_video(imgs_res, fps, path_output_video, convert_mp4=True):
    path_avi = f"{path_output_video}.avi"
    path_mp4 = f"{path_output_video}.mp4"
    height, width = imgs_res[0].shape[:2]
    out = cv2.VideoWriter(path_avi, cv2.VideoWriter_fourcc(*'DIVX'), fps, (width, height))
    for num in range(len(imgs_res)):
        frame = imgs_res[num]
        out.write(frame)
    out.release()

    if convert_mp4:
        command = [
            "ffmpeg",
            "-y",                  # overwrite file kalau sudah ada
            "-i", path_avi,      # input file
            "-c:v", "libx264",     # codec video H.264 (web compatible)
            "-preset", "fast",     # kecepatan encoding (fast/balanced)
            "-crf", "23",          # quality (0=lossless, 23=default)
            "-pix_fmt", "yuv420p", # pixel format yang didukung HTML5 video
            path_mp4
        ]
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def write_image(imgs_res, path_output_folder, array=True,img_name="frame"):
    
    path_image = f"{path_output_folder}/{img_name}.png"

    if array:
        frame = imgs_res[-1]
        cv2.imwrite(path_image, frame)
    else:
        cv2.imwrite(path_image, imgs_res)


class VideoStreamWriter:
    """
    Incrementally writes frames to disk to avoid buffering the entire video in memory.
    """
    def __init__(
        self,
        fps: float,
        path_output_video: str,
        convert_mp4: bool = True,
        codec: str = "H264",
        use_hw_accel: bool = True,
    ):
        self.fps = fps
        self.path_output_video = path_output_video
        self.convert_mp4 = convert_mp4
        self.codec = codec
        self.use_hw_accel = use_hw_accel
        self._writer: Optional[cv2.VideoWriter] = None
        self._mp4_path = f"{path_output_video}.mp4"

    def _fourcc(self):
        if self.codec.upper() in {"H264", "AVC1"}:
            return cv2.VideoWriter_fourcc(*"avc1")
        if self.codec.upper() == "MP4V":
            return cv2.VideoWriter_fourcc(*"mp4v")
        return cv2.VideoWriter_fourcc(*"avc1")

    def write(self, frame):
        if self._writer is None:
            height, width = frame.shape[:2]
            self._writer = cv2.VideoWriter(
                self._mp4_path,
                self._fourcc(),
                self.fps,
                (width, height),
            )
        self._writer.write(frame)

    def close(self):
        if self._writer is None:
            return
        self._writer.release()
        if not self.convert_mp4:
            return
        command = [
            "ffmpeg",
            "-y",
            "-i", self._mp4_path,
        ]
        if self.use_hw_accel:
            command += ["-c:v", "h264_nvenc", "-preset", "p3", "-cq", "23"]
        else:
            command += ["-c:v", "libx264", "-preset", "fast", "-crf", "23"]
        command += ["-pix_fmt", "yuv420p", f"{self._mp4_path}.tmp.mp4"]
        subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        os.replace(f"{self._mp4_path}.tmp.mp4", self._mp4_path)
