import cv2
import supervision as sv
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional

@dataclass
class TennisCourtConfiguration:
    """
    Konfigurasi final untuk lapangan tenis dalam mode POTRAIT.
    Urutan `keypoints` telah disesuaikan agar cocok dengan urutan
    deteksi spesifik dari model.
    """
    length: int = 2377  # Dimensi vertikal (tinggi)
    width: int = 1097   # Dimensi horizontal (lebar)
    singles_width: int = 823
    service_line_to_net: int = 640

    @property
    def doubles_alley_width(self) -> float:
        return (self.width - self.singles_width) / 2

    @property
    def keypoints(self) -> List[Tuple[float, float]]:
        """
        Menghasilkan 15 keypoints dalam urutan yang cocok dengan model deteksi.
        Koordinat tetap dalam mode potrait.
        """
        top_service_y = self.length / 2 - self.service_line_to_net
        bottom_service_y = self.length / 2 + self.service_line_to_net
        
        # Mendefinisikan semua titik dalam mode potrait
        p_kiri_bawah_luar = (0, self.length)
        p_kiri_bawah_dalam = (self.doubles_alley_width, self.length)
        p_kanan_bawah_dalam = (self.width - self.doubles_alley_width, self.length)
        p_kanan_bawah_luar = (self.width, self.length)
        
        p_kanan_atas_luar = (self.width, 0)
        p_kanan_atas_dalam = (self.width - self.doubles_alley_width, 0)
        p_kiri_atas_dalam = (self.doubles_alley_width, 0)
        p_kiri_atas_luar = (0, 0)
        
        p_servis_kiri_atas = (self.doubles_alley_width, top_service_y)
        p_t_atas = (self.width / 2, top_service_y)
        p_servis_kanan_atas = (self.width - self.doubles_alley_width, top_service_y)
        
        p_servis_kanan_bawah = (self.width - self.doubles_alley_width, bottom_service_y)
        p_t_bawah = (self.width / 2, bottom_service_y)
        p_servis_kiri_bawah = (self.doubles_alley_width, bottom_service_y)
        
        p_tengah_lapangan = (self.width / 2, self.length / 2)

        # Menyusun ulang titik sesuai urutan yang Anda berikan
        return [
            p_kiri_bawah_luar,      # no1. pojok kiri bawah
            p_kiri_bawah_dalam,     # no2. garis dalam sebelah kanan no1
            p_kanan_bawah_dalam,    # no3. garis dalam sebelah kiri, sebelum pojok kanan bawah
            p_kanan_bawah_luar,     # no4. pojok kanan bawah
            p_kanan_atas_luar,      # no5. pojok kanan atas
            p_kanan_atas_dalam,     # no6. garis dalam sebelah kiri pojok kanan atas
            p_kiri_atas_dalam,      # no7. garis dalam sebelah kanan pojok kiri atas
            p_kiri_atas_luar,       # no8. pojok kiri atas
            p_servis_kiri_atas,     # no9. titik dibawah no7
            p_t_atas,               # no10. titek tengah antara no 9 dan 11
            p_servis_kanan_atas,    # no11. dibawah titik no6
            p_servis_kanan_bawah,   # no12. diatas titik no3
            p_t_bawah,              # no13. titi tengah antara no12 dan no14
            p_servis_kiri_bawah,    # no14. diatas titik 2
            p_tengah_lapangan,      # no15. titik tengah lapangan
        ]

    @property
    def vertices(self) -> List[Tuple[float, float]]:
        """
        Menghasilkan semua 17 titik yang dibutuhkan untuk menggambar lapangan lengkap.
        """
        all_vertices = self.keypoints
        # Tambah 2 titik ekstra untuk ujung net
        all_vertices.append((0, self.length / 2))      # Vertex 16 (sisi kiri net)
        all_vertices.append((self.width, self.length / 2)) # Vertex 17 (sisi kanan net)
        return all_vertices

    @property
    def edges(self) -> List[Tuple[int, int]]:
        """
        Menyambungkan 17 titik untuk menggambar garis lapangan.
        Urutan indeks di sini harus diperbarui sesuai urutan keypoints yang baru.
        """
        return [
            # Garis baseline (bawah dan atas)
            (1, 4), (8, 5),
            # Garis sidelines (kiri dan kanan)
            (1, 8), (4, 5),
            # Garis singles sidelines
            (2, 7), (3, 6),
            # Garis servis (atas dan bawah)
            (9, 11), (14, 12),
            # Garis servis tengah
            (10, 13),
            # Garis net penuh
            (16, 17),
        ]
    
def draw_court(
    config: TennisCourtConfiguration,
    background_color: sv.Color = sv.Color(58, 83, 164), # Biru lapangan tenis
    line_color: sv.Color = sv.Color.WHITE,
    padding: int = 50,
    line_thickness: int = 2,
    scale: float = 0.4
) -> np.ndarray:
    """
    Menggambar lapangan tenis dengan dimensi, warna, dan skala tertentu.
    """
    scaled_length = int(config.length * scale)
    scaled_width = int(config.width * scale)

    # Membuat kanvas potrait (tinggi, lebar)
    court_image = np.ones(
        (scaled_length + 2 * padding,
         scaled_width + 2 * padding, 3),
        dtype=np.uint8
    ) * np.array(background_color.as_bgr(), dtype=np.uint8)

    # Menggunakan config.vertices (17 titik) untuk menggambar
    drawing_vertices = config.vertices
    for start, end in config.edges:
        point1 = (int(drawing_vertices[start - 1][0] * scale) + padding,
                  int(drawing_vertices[start - 1][1] * scale) + padding)
        point2 = (int(drawing_vertices[end - 1][0] * scale) + padding,
                  int(drawing_vertices[end - 1][1] * scale) + padding)
        cv2.line(
            img=court_image,
            pt1=point1,
            pt2=point2,
            color=line_color.as_bgr(),
            thickness=line_thickness
        )
    
    return court_image


def draw_points_on_court(
    config: TennisCourtConfiguration,
    xy: np.ndarray,
    face_color: sv.Color = sv.Color.RED,
    edge_color: sv.Color = sv.Color.BLACK,
    radius: int = 10,
    thickness: int = 2,
    padding: int = 50,
    scale: float = 0.4,
    court: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Menggambar titik-titik di atas lapangan tenis.
    """
    if court is None:
        court = draw_court(
            config=config,
            padding=padding,
            scale=scale
        )

    for point in xy:
        # Koordinat (x,y) dari xy sudah dalam mode potrait jika berasal dari model
        scaled_point = (
            int(point[0] * scale) + padding,
            int(point[1] * scale) + padding
        )
        cv2.circle(
            img=court, center=scaled_point, radius=radius,
            color=face_color.as_bgr(), thickness=-1
        )
        cv2.circle(
            img=court, center=scaled_point, radius=radius,
            color=edge_color.as_bgr(), thickness=thickness
        )

    return court


def draw_paths_on_court(
    config: TennisCourtConfiguration,
    paths: List[np.ndarray],
    color: sv.Color = sv.Color.WHITE,
    thickness: int = 2,
    padding: int = 50,
    scale: float = 0.4,
    court: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Menggambar jejak (paths) di atas lapangan tenis.
    """
    if court is None:
        court = draw_court(
            config=config,
            padding=padding,
            scale=scale
        )

    for path in paths:
        scaled_path = [
            (
                int(point[0] * scale) + padding,
                int(point[1] * scale) + padding
            )
            for point in path if point.size > 0
        ]

        if len(scaled_path) < 2:
            continue

        for i in range(len(scaled_path) - 1):
            cv2.line(
                img=court, pt1=scaled_path[i], pt2=scaled_path[i + 1],
                color=color.as_bgr(), thickness=thickness
            )

    return court