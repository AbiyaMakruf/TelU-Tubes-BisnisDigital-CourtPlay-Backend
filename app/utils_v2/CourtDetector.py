import torch
import cv2
import numpy as np
from tqdm import tqdm
from .tracknet import BallTrackerNet
from .homography import get_trans_matrix, refer_kps
from .postprocess_court import refine_kps
from .read_video import frame_generator
class CourtDetector():
    def __init__(self, path_model, original_width, original_height):
        self.model = BallTrackerNet(out_channels=15)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.original_width = original_width
        self.original_height = original_height
        if path_model:
            self.model.load_state_dict(torch.load(path_model, map_location=self.device))
            self.model = self.model.to(self.device)
            self.model.eval()

    def _infer_from_iterator(self, iterator, total_frames=None, stride=1):
        output_width = 640
        output_height = 360
        scale = self.original_width / output_width

        kps_res = []
        matrixes_res = []
        last_points = None
        last_matrix = None
        stride = max(1, stride)
        iterator = tqdm(iterator, total=total_frames, desc="[Court]", unit="frame")
        for idx, image_frame in iterator:
            run_detection = (idx % stride == 0) or last_matrix is None
            if not run_detection:
                kps_res.append(last_points)
                matrixes_res.append(last_matrix)
                continue
            img = cv2.resize(image_frame, (output_width, output_height))
            inp = (img.astype(np.float32) / 255.0).transpose(2, 0, 1)
            inp_tensor = torch.from_numpy(inp).unsqueeze(0).to(self.device)

            with torch.no_grad():
                out = self.model(inp_tensor)[0]
            pred = torch.sigmoid(out).detach().cpu().numpy()

            points = []
            for kps_num in range(14):
                heatmap = (pred[kps_num]*255).astype(np.uint8)
                ret, heatmap = cv2.threshold(heatmap, 170, 255, cv2.THRESH_BINARY)
                circles = cv2.HoughCircles(heatmap, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=2,
                                           minRadius=10, maxRadius=25)
                if circles is not None:
                    x_pred = circles[0][0][0]*scale
                    y_pred = circles[0][0][1]*scale
                    if kps_num not in [8, 12, 9]:
                        x_pred, y_pred = refine_kps(image_frame, int(y_pred), int(x_pred), crop_size=40)
                    points.append((x_pred, y_pred))
                else:
                    points.append(None)

            matrix_trans = get_trans_matrix(points)
            points = None
            if matrix_trans is not None:
                points = cv2.perspectiveTransform(refer_kps, matrix_trans)
                matrix_trans = cv2.invert(matrix_trans)[1]
            last_points = points
            last_matrix = matrix_trans
            kps_res.append(last_points)
            matrixes_res.append(last_matrix)
        if total_frames is not None:
            while len(kps_res) < total_frames:
                kps_res.append(last_points)
                matrixes_res.append(last_matrix)
        return matrixes_res, kps_res

    def infer_model(self, frames, stride=1):
        iterator = ((idx, frame) for idx, frame in enumerate(frames))
        return self._infer_from_iterator(iterator, total_frames=len(frames), stride=stride)

    def infer_video(self, path_video, total_frames=None, stride=1):
        iterator = frame_generator(path_video)
        return self._infer_from_iterator(iterator, total_frames=total_frames, stride=stride)



