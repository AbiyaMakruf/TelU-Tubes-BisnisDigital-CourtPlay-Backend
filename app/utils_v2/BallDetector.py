import torch
import cv2
import numpy as np
from collections import deque
from .tracknet import BallTrackerNet
from scipy.spatial import distance
from tqdm import tqdm
from .read_video import frame_generator
class BallDetector:
    def __init__(self, path_model, original_width, original_height):
        self.model = BallTrackerNet(input_channels=9, out_channels=256)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.original_width = original_width
        self.original_height = original_height
        self.width = 640
        self.height = 360
        self.scale_factor = self.original_width / self.width
        if path_model:
            self.model.load_state_dict(torch.load(path_model, map_location=self.device))
            self.model = self.model.to(self.device)
            self.model.eval()

    def _infer_from_iterator(self, iterator):
        buffer = deque(maxlen=3)
        ball_track = []
        prev_pred = [None, None]

        with torch.no_grad():
            for _, frame in iterator:
                resized = cv2.resize(frame, (self.width, self.height))
                buffer.append(resized)
                if len(buffer) < 3:
                    ball_track.append((None, None))
                    continue

                imgs = np.concatenate((buffer[2], buffer[1], buffer[0]), axis=2)
                imgs = imgs.astype(np.float32) / 255.0
                imgs = np.transpose(imgs, (2, 0, 1))
                inp = np.expand_dims(imgs, axis=0)

                out = self.model(torch.from_numpy(inp).float().to(self.device))
                output = out.argmax(dim=1).detach().cpu().numpy()
                x_pred, y_pred = self.postprocess(output, prev_pred)
                prev_pred = [x_pred, y_pred]
                ball_track.append((x_pred, y_pred))
        return ball_track

    def infer_model(self, frames):
        iterator = ((idx, frame) for idx, frame in enumerate(frames))
        iterator = tqdm(iterator, total=len(frames), desc="[Ball]", unit="frame")
        return self._infer_from_iterator(iterator)

    def infer_video(self, path_video, total_frames=None):
        iterator = frame_generator(path_video)
        iterator = tqdm(iterator, total=total_frames, desc="[Ball]", unit="frame")
        return self._infer_from_iterator(iterator)

    def postprocess(self, feature_map, prev_pred, max_dist=80):
        scale = self.scale_factor
        feature_map *= 255
        feature_map = feature_map.reshape((self.height, self.width))
        feature_map = feature_map.astype(np.uint8)
        ret, heatmap = cv2.threshold(feature_map, 127, 255, cv2.THRESH_BINARY)
        circles = cv2.HoughCircles(heatmap, cv2.HOUGH_GRADIENT, dp=1, minDist=1, param1=50, param2=2, minRadius=2, maxRadius=7)

        x,y = None, None
        if circles is not None:
            if prev_pred[0]:
                for i in range(len(circles[0])):
                    x_temp = circles[0][i][0]*scale
                    y_temp = circles[0][i][1]*scale
                    dist = distance.euclidean((x_temp, y_temp), prev_pred)
                    if dist < max_dist:
                        x, y = x_temp, y_temp
                        break
            else:
                x = circles[0][0][0]*scale
                y = circles[0][0][1]*scale
        return x, y
