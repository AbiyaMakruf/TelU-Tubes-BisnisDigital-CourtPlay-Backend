import cv2
import logging
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
    Incrementally writes frames to disk (AVI) and optionally converts to MP4 via ffmpeg.
    """

    def __init__(
        self,
        fps: float,
        path_output_video: str,
        convert_mp4: bool = True,
        codec: str = "MJPG",
        use_hw_accel: bool = True,
    ):
        self.fps = fps
        self.path_output_video = path_output_video
        self.convert_mp4 = convert_mp4
        self.codec = codec
        self.use_hw_accel = False
        self._writer: Optional[cv2.VideoWriter] = None
        self._avi_path = f"{path_output_video}.avi"
        self._mp4_path = f"{path_output_video}.mp4"
        self._logger = logging.getLogger(__name__)

    def _fourcc(self):
        if self.codec.upper() == "XVID":
            return cv2.VideoWriter_fourcc(*"XVID")
        return cv2.VideoWriter_fourcc(*"MJPG")

    def write(self, frame):
        if self._writer is None:
            height, width = frame.shape[:2]
            self._writer = cv2.VideoWriter(
                self._avi_path,
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
        if not os.path.exists(self._avi_path):
            self._logger.error(f"AVI source missing for conversion: {self._avi_path}")
            return
        command = [
            "ffmpeg",
            "-y",
            "-i",
            self._avi_path,
        ]
        if self.use_hw_accel:
            command += ["-c:v", "h264_nvenc", "-preset", "p3", "-cq", "23"]
        else:
            command += ["-c:v", "libx264", "-preset", "fast", "-crf", "23"]
        command += ["-pix_fmt", "yuv420p", self._mp4_path]
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            self._logger.error(
                "FFmpeg conversion failed: %s",
                result.stderr.decode("utf-8", errors="ignore"),
            )
            return
        os.remove(self._avi_path)
