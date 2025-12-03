import cv2
import torch
import numpy as np
from tqdm import tqdm
from .court_reference import CourtReference
from scipy.spatial import distance
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, fasterrcnn_mobilenet_v3_large_fpn,  FasterRCNN_ResNet50_FPN_V2_Weights, FasterRCNN_MobileNet_V3_Large_FPN_Weights
from .read_video import frame_generator

class PersonDetector():
    def __init__(self, dtype=torch.FloatTensor):
        # self.detection_model = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
        self.detection_model = fasterrcnn_mobilenet_v3_large_fpn(weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT)
        self.detection_model = self.detection_model.to(dtype)
        self.detection_model.eval()
        self.dtype = dtype
        self.court_ref = CourtReference()
        self.ref_top_court = self.court_ref.get_court_mask(2)
        self.ref_bottom_court = self.court_ref.get_court_mask(1)
        self.point_person_top = None
        self.point_person_bottom = None
        self.counter_top = 0
        self.counter_bottom = 0

        
    def detect(self, image, person_min_score=0.85): 
        PERSON_LABEL = 1
        frame_tensor = image.transpose((2, 0, 1)) / 255
        frame_tensor = torch.from_numpy(frame_tensor).unsqueeze(0).float().to(self.dtype)
        
        with torch.no_grad():
            preds = self.detection_model(frame_tensor)
            
        persons_boxes = []
        probs = []
        for box, label, score in zip(preds[0]['boxes'][:], preds[0]['labels'], preds[0]['scores']):
            if label == PERSON_LABEL and score > person_min_score:    
                persons_boxes.append(box.detach().cpu().numpy())
                probs.append(score.detach().cpu().numpy())
        return persons_boxes, probs
    
    def detect_top_and_bottom_players(self, image, inv_matrix, filter_players=False):
        matrix = cv2.invert(inv_matrix)[1]
        mask_top_court = cv2.warpPerspective(self.ref_top_court, matrix, image.shape[1::-1])
        mask_bottom_court = cv2.warpPerspective(self.ref_bottom_court, matrix, image.shape[1::-1])

        if mask_top_court.ndim == 3:
            mask_top_court = mask_top_court[..., 0]
        if mask_bottom_court.ndim == 3:
            mask_bottom_court = mask_bottom_court[..., 0]

        mask_top_court = (mask_top_court > 0).astype(np.uint8)
        mask_bottom_court = (mask_bottom_court > 0).astype(np.uint8)
        person_bboxes_top, person_bboxes_bottom = [], []

        bboxes, probs = self.detect(image, person_min_score=0.85)
        if len(bboxes) > 0:
            person_points = [[int((bbox[2] + bbox[0]) / 2), int(bbox[3])] for bbox in bboxes]
            person_bboxes = list(zip(bboxes, person_points))

            person_bboxes_top = [
                pt for pt in person_bboxes
                if pt[1][1] > 0 and mask_top_court[pt[1][1] - 1, pt[1][0]] == 1
            ]
            person_bboxes_bottom = [
                pt for pt in person_bboxes
                if pt[1][1] > 0 and mask_bottom_court[pt[1][1] - 1, pt[1][0]] == 1
            ]

            if filter_players:
                person_bboxes_top, person_bboxes_bottom = self.filter_players(
                    person_bboxes_top, person_bboxes_bottom, matrix)

        return person_bboxes_top, person_bboxes_bottom

    def filter_players(self, person_bboxes_top, person_bboxes_bottom, matrix):
        """
        Leave one person at the top and bottom of the tennis court
        """
        refer_kps = np.array(self.court_ref.key_points[12:], dtype=np.float32).reshape((-1, 1, 2))
        trans_kps = cv2.perspectiveTransform(refer_kps, matrix)
        center_top_court = trans_kps[0][0]
        center_bottom_court = trans_kps[1][0]
        if len(person_bboxes_top) > 1:
            dists = [distance.euclidean(x[1], center_top_court) for x in person_bboxes_top]
            ind = dists.index(min(dists))
            person_bboxes_top = [person_bboxes_top[ind]]
        if len(person_bboxes_bottom) > 1:
            dists = [distance.euclidean(x[1], center_bottom_court) for x in person_bboxes_bottom]
            ind = dists.index(min(dists))
            person_bboxes_bottom = [person_bboxes_bottom[ind]]
        return person_bboxes_top, person_bboxes_bottom
    
    def _clone_person_list(self, persons):
        return [(bbox.copy(), (int(point[0]), int(point[1]))) for bbox, point in persons]

    def track_players(self, frames, matrix_all, filter_players=False, stride=1):
        persons_top = []
        persons_bottom = []
        stride = max(1, stride)
        last_top, last_bottom = [], []
        min_len = min(len(frames), len(matrix_all))
        for num_frame in tqdm(range(min_len)):
            img = frames[num_frame]
            matrix = matrix_all[num_frame]
            detect_now = matrix is not None and (num_frame % stride == 0 or not last_top and not last_bottom)
            if matrix is not None and detect_now:
                inv_matrix = matrix
                person_top, person_bottom = self.detect_top_and_bottom_players(img, inv_matrix, filter_players)
                last_top = self._clone_person_list(person_top)
                last_bottom = self._clone_person_list(person_bottom)
            elif matrix is None:
                last_top, last_bottom = [], []
            persons_top.append(self._clone_person_list(last_top))
            persons_bottom.append(self._clone_person_list(last_bottom))
        return persons_top, persons_bottom    

    def track_players_video(self, path_video, matrix_all, filter_players=False, stride=1, total_frames=None):
        persons_top = []
        persons_bottom = []
        stride = max(1, stride)
        last_top, last_bottom = [], []
        max_frames = total_frames if total_frames is not None else len(matrix_all)

        iterator = frame_generator(path_video)
        iterator = tqdm(iterator, total=max_frames, desc="[Player]", unit="frame")
        for idx, frame in iterator:
            if idx >= max_frames:
                break
            matrix = matrix_all[idx]
            detect_now = matrix is not None and (idx % stride == 0 or not last_top and not last_bottom)
            if matrix is not None and detect_now:
                inv_matrix = matrix
                person_top, person_bottom = self.detect_top_and_bottom_players(frame, inv_matrix, filter_players)
                last_top = self._clone_person_list(person_top)
                last_bottom = self._clone_person_list(person_bottom)
            elif matrix is None:
                last_top, last_bottom = [], []
            persons_top.append(self._clone_person_list(last_top))
            persons_bottom.append(self._clone_person_list(last_bottom))

        while len(persons_top) < max_frames:
            persons_top.append([])
            persons_bottom.append([])

        return persons_top, persons_bottom
