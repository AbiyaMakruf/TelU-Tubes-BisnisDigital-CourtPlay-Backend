import torch
import cv2
import numpy as np
from .tracknet import BallTrackerNet
from tqdm import tqdm
from scipy.spatial import distance
import os # Tambahkan impor os

class BallDetector:
    # ... (Bagian __init__ TIDAK BERUBAH) ...
    def __init__(self, path_model, original_width, original_height):
        self.model = BallTrackerNet(input_channels=9, out_channels=256)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.original_width = original_width
        self.original_height = original_height
        self.width = 640
        self.height = 360
        self.scale_factor = self.original_width / self.width # Pindahkan ini ke setelah self.width/height didefinisikan

        if path_model:
            self.model.load_state_dict(torch.load(path_model, map_location=self.device))
            self.model = self.model.to(self.device)
            self.model.eval()

    def infer_model(self, frames, save_visual_frames=True, output_folder='VISUAL_TRACKING'):
        # 1. Inisialisasi folder output
        if save_visual_frames:
            os.makedirs(output_folder, exist_ok=True)
            print(f"Frame visualisasi akan disimpan di folder: {output_folder}/")

        ball_track = [(None, None)]*2
        prev_pred = [None, None]

        for num in tqdm(range(2, len(frames))):
            # Ambil frame saat ini dalam resolusi asli untuk visualisasi
            current_frame_vis = frames[num].copy()
            
            # --- Persiapan Input Model (Resolusi Rendah) ---
            img = cv2.resize(frames[num], (self.width, self.height))
            img_prev = cv2.resize(frames[num-1], (self.width, self.height))
            img_preprev = cv2.resize(frames[num-2], (self.width, self.height))
            imgs = np.concatenate((img, img_prev, img_preprev), axis=2)
            imgs = imgs.astype(np.float32)/255.0
            imgs = np.rollaxis(imgs, 2, 0)
            inp = np.expand_dims(imgs, axis=0)

            # --- Inferensi Model ---
            out = self.model(torch.from_numpy(inp).to(self.device))
            feature_map = out[0, 0, :, :].detach().cpu().numpy()
            
            # --- Post-processing ---
            x_pred, y_pred = self.postprocess(feature_map, prev_pred)
            ball_track.append((x_pred, y_pred))

            # 2. Visualisasi dan Penyimpanan Frame
            if save_visual_frames and x_pred is not None and y_pred is not None:
                
                # Konversi koordinat float ke integer
                center_x = int(x_pred)
                center_y = int(y_pred)
                
                # Gambar lingkaran di lokasi bola yang terdeteksi pada frame asli
                # (1920x1080 atau berapapun self.original_width/height nya)
                cv2.circle(
                    img=current_frame_vis, 
                    center=(center_x, center_y), 
                    radius=10, # Ukuran lingkaran visualisasi
                    color=(0, 255, 0), # Warna Hijau (BGR)
                    thickness=3
                )
                
                # Simpan frame yang sudah ditandai
                filename = os.path.join(output_folder, f"tracked_frame_{num:05d}.jpg")
                cv2.imwrite(filename, current_frame_vis)
            
            # 3. Perbarui prev_pred
            if x_pred is not None:
                prev_pred = [x_pred, y_pred]

        return ball_track

    # ... (Fungsi postprocess TIDAK BERUBAH, tapi pastikan out_channels=1 jika Anda hanya menggunakan 1 channel) ...
    def postprocess(self, feature_map, prev_pred, max_dist=80):
        # ... (Kode postprocess yang sudah ada) ...
        scale = self.scale_factor
        # Catatan: feature_map di sini adalah hasil dari out[0, 0, :, :]
        # ...
        feature_map *= 255
        feature_map = feature_map.reshape((self.height, self.width))
        feature_map = feature_map.astype(np.uint8)
        ret, heatmap = cv2.threshold(feature_map, 127, 255, cv2.THRESH_BINARY)
        # ... (Lanjutkan hingga return x, y)
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