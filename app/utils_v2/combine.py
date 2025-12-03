import cv2
import numpy as np
from dataclasses import dataclass
from time import perf_counter
from typing import Generator, Optional
from .court_reference import CourtReference

try:
    import torch
    from torchvision.transforms import functional as TVF
except Exception:  # pragma: no cover - optional dependency
    torch = None
    TVF = None


MINIMAP_WIDTH = 166
MINIMAP_HEIGHT = 350
PLAYER_HEAT_INCREMENT = 5
PLAYER_HEAT_PERCENTILE = 98  # percentile that should map near red
PLAYER_HEAT_TARGET_DECAY = 0.985  # smoothing for dynamic scaling
PLAYER_HEAT_GAUSSIAN_SIGMA = 25
PLAYER_HEAT_RADIUS = 10
BALL_HEAT_INCREMENT = 12
BALL_HEAT_PERCENTILE = 96
BALL_HEAT_TARGET_DECAY = 0.99
BALL_HEAT_GAUSSIAN_SIGMA = 20
BALL_HEAT_RADIUS = 14
HEAT_ALPHA = 0.65
CONTOUR_LEVELS = [40, 90, 140, 190, 230]
PERCENTILE_TARGET_SAMPLES = 250_000
HEATMAP_DOWNSCALE = 2
USE_TORCH_HEAT = torch is not None and TVF is not None and torch.cuda.is_available()


@dataclass
class CombineFrameOutputs:
    combined: Optional[np.ndarray] = None
    minimap_ball: Optional[np.ndarray] = None
    minimap_player: Optional[np.ndarray] = None
    heatmap_player: Optional[np.ndarray] = None
    heatmap_ball: Optional[np.ndarray] = None
    profiling: Optional[dict] = None


@dataclass
class CombineRenderOptions:
    combined: bool = True
    minimap_ball: bool = True
    minimap_player: bool = True
    heatmap_player: bool = True
    heatmap_ball: bool = True


def get_court_img():
    court_reference = CourtReference()
    court_img = court_reference.court.copy()
    court_img = cv2.dilate(court_img, np.ones((10, 10), dtype=np.uint8))
    return court_img


def _draw_ball(frame, ball_track, index, draw_trace, trace_len):
    if not ball_track[index][0]:
        return frame
    if draw_trace:
        for offset in range(trace_len):
            prev = index - offset
            if prev < 0 or not ball_track[prev][0]:
                continue
            draw_x, draw_y = map(int, ball_track[prev])
            frame = cv2.circle(frame, (draw_x, draw_y), radius=3, color=(0, 255, 0), thickness=2)
    else:
        bx, by = map(int, ball_track[index])
        frame = cv2.circle(frame, (bx, by), radius=5, color=(0, 255, 0), thickness=2)
        frame = cv2.putText(frame, "ball", (bx + 8, by + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 255, 0), 2)
    return frame


def _draw_court_keypoints(frame, kps):
    if kps is None:
        return frame
    for kp in kps:
        x, y = int(kp[0, 0]), int(kp[0, 1])
        frame = cv2.circle(frame, (x, y), radius=0, color=(0, 0, 255), thickness=10)
    return frame


def _project_point(point, inv_mat, shape):
    pt = np.array(point, dtype=np.float32).reshape(1, 1, 2)
    pt = cv2.perspectiveTransform(pt, inv_mat)
    x = int(np.clip(pt[0, 0, 0], 0, shape[1] - 1))
    y = int(np.clip(pt[0, 0, 1], 0, shape[0] - 1))
    return x, y


def _blend_contour_map(base, heat_norm):
    if not np.any(heat_norm):
        return base.copy()
    heat_color = cv2.applyColorMap(heat_norm, cv2.COLORMAP_JET)
    overlay = base.copy()
    mask = heat_norm > 0
    overlay[mask] = cv2.addWeighted(heat_color[mask], HEAT_ALPHA, overlay[mask], 1 - HEAT_ALPHA, 0)
    for level in CONTOUR_LEVELS:
        _, thresh = cv2.threshold(heat_norm, level, 255, cv2.THRESH_BINARY)
        contours_info = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours_info[0] if len(contours_info) == 2 else contours_info[1]
        if not contours:
            continue
        level_color = cv2.applyColorMap(np.full((1, 1), level, dtype=np.uint8), cv2.COLORMAP_JET)[0, 0]
        cv2.drawContours(overlay, contours, -1, tuple(int(c) for c in level_color), 2)
    return overlay


def _sample_positive_percentile(arr, percentile):
    """
    Estimate percentile from a strided subset to avoid allocating huge helper arrays.
    The stride is chosen so that the sampled view stays close to PERCENTILE_TARGET_SAMPLES.
    """
    if not np.any(arr > 0):
        return 0.0
    total = arr.shape[0] * arr.shape[1]
    stride = int(max(1, np.sqrt(total / (PERCENTILE_TARGET_SAMPLES + 1e-6))))
    sampled = arr[::stride, ::stride]
    positives = sampled[sampled > 0]
    if positives.size == 0:
        # Fallback to the entire array when the strided sample missed positives.
        positives = arr[arr > 0]
    return float(np.percentile(positives, percentile))


def _fetch_frame(frames, idx):
    frame = frames[idx]
    return frame.copy()


def _alloc_heatmap(template):
    height, width = template.shape[:2]
    return np.zeros(
        (
            max(1, height // HEATMAP_DOWNSCALE),
            max(1, width // HEATMAP_DOWNSCALE),
        ),
        dtype=np.float32,
    )


def _project_heat_point(point, shape):
    x = int(np.clip(point[0] / HEATMAP_DOWNSCALE, 0, shape[1] - 1))
    y = int(np.clip(point[1] / HEATMAP_DOWNSCALE, 0, shape[0] - 1))
    return x, y


def _scaled_radius(radius):
    return max(1, int(radius / HEATMAP_DOWNSCALE))


def _render_heatmap_overlay(accum, base_img, sigma, percentile, target, increment, decay):
    heat_blurred = cv2.GaussianBlur(accum, (0, 0), sigmaX=sigma, sigmaY=sigma)
    heat_percentile = _sample_positive_percentile(heat_blurred, percentile)
    if heat_percentile == 0.0:
        return base_img.copy(), target
    target = max(target * decay, heat_percentile, increment)
    heat_norm = np.clip((heat_blurred / (target + 1e-6)) * 255.0, 0, 255).astype(np.uint8)
    heat_norm = cv2.resize(
        heat_norm,
        (base_img.shape[1], base_img.shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )
    return _blend_contour_map(base_img, heat_norm), target


class TorchHeatAccumulator:
    def __init__(self, base_img, radius, increment, sigma, percentile, decay, initial_target):
        if not USE_TORCH_HEAT:
            raise RuntimeError("Torch heat accumulator requires CUDA.")
        self.device = torch.device("cuda")
        height = max(1, base_img.shape[0] // HEATMAP_DOWNSCALE)
        width = max(1, base_img.shape[1] // HEATMAP_DOWNSCALE)
        self.accum = torch.zeros((1, 1, height, width), device=self.device, dtype=torch.float32)
        self.radius = _scaled_radius(radius)
        kernel_size = self.radius * 2 + 1
        yy, xx = torch.meshgrid(
            torch.arange(kernel_size, device=self.device),
            torch.arange(kernel_size, device=self.device),
            indexing="ij",
        )
        center = self.radius
        disk = ((yy - center) ** 2 + (xx - center) ** 2) <= (self.radius ** 2)
        self.disk_kernel = disk.float()
        self.increment = increment
        self.scale = HEATMAP_DOWNSCALE
        self.base_img = base_img
        self.percentile = percentile
        self.decay = decay
        self.target = initial_target
        self.sigma = max(0.1, sigma / self.scale)
        kernel_extent = max(3, int(6 * self.sigma) | 1)
        if kernel_extent % 2 == 0:
            kernel_extent += 1
        self.kernel_size = [kernel_extent, kernel_extent]
        self.dirty = False
        self.last_image = base_img.copy()

    def add_point(self, x, y):
        hx = int(x / self.scale)
        hy = int(y / self.scale)
        self._add_disk(hx, hy)
        self.dirty = True

    def _add_disk(self, x, y):
        h = self.accum.shape[-2]
        w = self.accum.shape[-1]
        r = self.radius
        if h <= 0 or w <= 0:
            return
        x0 = max(0, x - r)
        x1 = min(w, x + r + 1)
        y0 = max(0, y - r)
        y1 = min(h, y + r + 1)
        if x0 >= x1 or y0 >= y1:
            return
        kx0 = x0 - (x - r)
        kx1 = kx0 + (x1 - x0)
        ky0 = y0 - (y - r)
        ky1 = ky0 + (y1 - y0)
        self.accum[..., y0:y1, x0:x1] += self.disk_kernel[ky0:ky1, kx0:kx1] * self.increment

    def render(self):
        if not self.dirty:
            return self.last_image.copy()
        blurred = TVF.gaussian_blur(
            self.accum,
            kernel_size=self.kernel_size,
            sigma=(self.sigma, self.sigma),
        )
        positives = blurred[blurred > 0]
        if positives.numel() == 0:
            self.last_image = self.base_img.copy()
            self.dirty = False
            return self.last_image.copy()
        percentile = torch.quantile(positives, self.percentile / 100.0).item()
        self.target = max(self.target * self.decay, percentile, self.increment)
        heat_norm = torch.clamp((blurred / (self.target + 1e-6)) * 255.0, 0, 255).to(torch.uint8)
        heat_norm = heat_norm.squeeze().detach().cpu().numpy()
        heat_norm = cv2.resize(
            heat_norm,
            (self.base_img.shape[1], self.base_img.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
        self.last_image = _blend_contour_map(self.base_img, heat_norm)
        self.dirty = False
        return self.last_image.copy()


def combine_stream(frames, scenes, bounces, ball_track, homography_matrices, kps_court,
                   persons_top, persons_bottom, draw_trace=False, trace=10,
                   render_options: Optional[CombineRenderOptions] = None) -> Generator[CombineFrameOutputs, None, None]:
    """
    Generator variant that yields CombineFrameOutputs per frame to keep memory usage flat.
    `render_options` allows enabling/disabling individual visualizations to save time/memory.
    """
    if render_options is None:
        render_options = CombineRenderOptions()

    want_combined = render_options.combined
    want_minimap_ball = render_options.minimap_ball
    want_minimap_player = render_options.minimap_player
    want_heatmap_player = render_options.heatmap_player
    want_heatmap_ball = render_options.heatmap_ball
    need_any_court = any([want_combined, want_minimap_ball, want_minimap_player,
                          want_heatmap_player, want_heatmap_ball])
    if not (want_combined or need_any_court):
        raise ValueError("At least one visualization must be enabled in CombineRenderOptions.")
    need_player_projection = any([want_combined, want_minimap_player, want_heatmap_player])

    is_track = [mat is not None for mat in homography_matrices]

    for scene_start, scene_end in scenes:
        tracked = is_track[scene_start:scene_end]
        if not tracked or sum(tracked) / (len(tracked) + 1e-15) <= 0.5:
            neutral_court_color = None
            if need_any_court:
                neutral = get_court_img()
                if neutral.ndim == 2:
                    neutral = cv2.cvtColor(neutral, cv2.COLOR_GRAY2BGR)
                neutral_court_color = neutral
            for idx in range(scene_start, scene_end):
                frame_profile = {}
                frame_start = perf_counter()
                frame_copy = _fetch_frame(frames, idx) if want_combined else None
                elapsed = perf_counter() - frame_start
                frame_profile["scene_passthrough"] = elapsed
                frame_profile["total"] = elapsed
                yield CombineFrameOutputs(
                    combined=frame_copy if want_combined else None,
                    minimap_ball=neutral_court_color.copy() if (want_minimap_ball and neutral_court_color is not None) else None,
                    minimap_player=neutral_court_color.copy() if (want_minimap_player and neutral_court_color is not None) else None,
                    heatmap_player=neutral_court_color.copy() if (want_heatmap_player and neutral_court_color is not None) else None,
                    heatmap_ball=neutral_court_color.copy() if (want_heatmap_ball and neutral_court_color is not None) else None,
                    profiling=frame_profile,
                )
            continue

        court_template = get_court_img() if need_any_court else None
        if court_template is not None and court_template.ndim == 2:
            court_template = cv2.cvtColor(court_template, cv2.COLOR_GRAY2BGR)
        court_base = court_template.copy() if want_combined else None
        court_ball = (court_template.copy() if want_minimap_ball else None)
        court_player_heat = court_template.copy() if want_heatmap_player else None
        court_ball_heat = court_template.copy() if want_heatmap_ball else None
        player_heat_torch = None
        ball_heat_torch = None
        if want_heatmap_player and USE_TORCH_HEAT:
            player_heat_torch = TorchHeatAccumulator(
                court_player_heat,
                PLAYER_HEAT_RADIUS,
                PLAYER_HEAT_INCREMENT,
                PLAYER_HEAT_GAUSSIAN_SIGMA,
                PLAYER_HEAT_PERCENTILE,
                PLAYER_HEAT_TARGET_DECAY,
                PLAYER_HEAT_INCREMENT * 15,
            )
            heatmap_accum = None
            player_heat_image = court_player_heat.copy()
            heat_target = PLAYER_HEAT_INCREMENT * 15
            player_heat_dirty = False
            player_heat_sigma = None
        else:
            heatmap_accum = _alloc_heatmap(court_template) if want_heatmap_player else None
            player_heat_image = court_player_heat.copy() if want_heatmap_player else None
            heat_target = PLAYER_HEAT_INCREMENT * 15
            player_heat_dirty = False
            player_heat_sigma = max(0.1, PLAYER_HEAT_GAUSSIAN_SIGMA / HEATMAP_DOWNSCALE)

        if want_heatmap_ball and USE_TORCH_HEAT:
            ball_heat_torch = TorchHeatAccumulator(
                court_ball_heat,
                BALL_HEAT_RADIUS,
                BALL_HEAT_INCREMENT,
                BALL_HEAT_GAUSSIAN_SIGMA,
                BALL_HEAT_PERCENTILE,
                BALL_HEAT_TARGET_DECAY,
                BALL_HEAT_INCREMENT * 8,
            )
            ball_heatmap_accum = None
            ball_heat_image = court_ball_heat.copy()
            ball_heat_target = BALL_HEAT_INCREMENT * 8
            ball_heat_dirty = False
            ball_heat_sigma = None
        else:
            ball_heatmap_accum = _alloc_heatmap(court_template) if want_heatmap_ball else None
            ball_heat_image = court_ball_heat.copy() if want_heatmap_ball else None
            ball_heat_target = BALL_HEAT_INCREMENT * 8
            ball_heat_dirty = False
            ball_heat_sigma = max(0.1, BALL_HEAT_GAUSSIAN_SIGMA / HEATMAP_DOWNSCALE)

        bounce_history_court = []
        for idx in range(scene_start, scene_end):
            frame_profile = {}
            frame_start = perf_counter()
            inv_mat = homography_matrices[idx]

            frame = None
            if want_combined:
                section_start = perf_counter()
                frame = _fetch_frame(frames, idx)

                # Draw past bounces on the original frame
                if inv_mat is not None and bounce_history_court:
                    try:
                        mat_inv = np.linalg.inv(inv_mat)
                        pts_court = np.array(bounce_history_court, dtype=np.float32).reshape(-1, 1, 2)
                        pts_frame = cv2.perspectiveTransform(pts_court, mat_inv)
                        for pt in pts_frame:
                            bx, by = int(pt[0, 0]), int(pt[0, 1])
                            # Draw a marker for the bounce (e.g., a small yellow circle)
                            cv2.circle(frame, (bx, by), radius=5, color=(0, 255, 255), thickness=-1)
                            cv2.circle(frame, (bx, by), radius=5, color=(0, 0, 0), thickness=1)
                    except np.linalg.LinAlgError:
                        pass

                frame = _draw_ball(frame, ball_track, idx, draw_trace, trace)
                frame = _draw_court_keypoints(frame, kps_court[idx] if idx < len(kps_court) else None)
                frame_profile["frame_draw"] = perf_counter() - section_start

            if idx in bounces and inv_mat is not None and ball_track[idx][0]:
                section_start = perf_counter()
                ball_point = _project_point(ball_track[idx], inv_mat, court_template.shape[:2])
                bounce_history_court.append(ball_point)
                if want_combined and court_base is not None:
                    cv2.circle(court_base, ball_point, radius=0, color=(0, 255, 255), thickness=50)
                if want_minimap_ball and court_ball is not None:
                    cv2.circle(court_ball, ball_point, radius=0, color=(0, 255, 255), thickness=50)
                if ball_heat_torch is not None:
                    ball_heat_torch.add_point(*ball_point)
                elif want_heatmap_ball and ball_heatmap_accum is not None:
                    hx, hy = _project_heat_point(ball_point, ball_heatmap_accum.shape)
                    cv2.circle(
                        ball_heatmap_accum,
                        (hx, hy),
                        _scaled_radius(BALL_HEAT_RADIUS),
                        BALL_HEAT_INCREMENT,
                        -1,
                    )
                    ball_heat_dirty = True
                frame_profile["bounce_projection"] = frame_profile.get("bounce_projection", 0.0) + (
                    perf_counter() - section_start
                )

            section_start = perf_counter()
            minimap = court_base.copy() if want_combined else None
            imgs_ball_frame = court_ball.copy() if want_minimap_ball else None
            persons = persons_top[idx] + persons_bottom[idx] if need_player_projection else []
            court_person = court_template.copy() if want_minimap_player else None
            frame_profile["court_copy"] = frame_profile.get("court_copy", 0.0) + (perf_counter() - section_start)
            if persons:
                section_start = perf_counter()
                for bbox, person_point in persons:
                    if len(bbox) == 0 or inv_mat is None:
                        continue
                    px, py = _project_point(person_point, inv_mat, court_template.shape[:2])
                    if want_combined and frame is not None:
                        x1, y1, x2, y2 = map(int, bbox)
                        frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    if minimap is not None:
                        cv2.circle(minimap, (px, py), radius=0, color=(255, 0, 0), thickness=80)
                    if court_person is not None:
                        cv2.circle(court_person, (px, py), radius=0, color=(255, 0, 0), thickness=80)
                    if player_heat_torch is not None:
                        player_heat_torch.add_point(px, py)
                    elif heatmap_accum is not None:
                        hx, hy = _project_heat_point((px, py), heatmap_accum.shape)
                        cv2.circle(
                            heatmap_accum,
                            (hx, hy),
                            _scaled_radius(PLAYER_HEAT_RADIUS),
                            PLAYER_HEAT_INCREMENT,
                            -1,
                        )
                        player_heat_dirty = True
                frame_profile["player_projection"] = frame_profile.get("player_projection", 0.0) + (
                    perf_counter() - section_start
                )

            if want_combined and frame is not None and minimap is not None:
                section_start = perf_counter()
                minimap_resized = cv2.resize(minimap, (MINIMAP_WIDTH, MINIMAP_HEIGHT))
                frame[30:30 + MINIMAP_HEIGHT,
                      frame.shape[1] - 30 - MINIMAP_WIDTH:frame.shape[1] - 30] = minimap_resized
                frame_profile["minimap_embed"] = frame_profile.get("minimap_embed", 0.0) + (
                    perf_counter() - section_start
                )
            combined_frame = frame if want_combined else None

            imgs_person_frame = court_person if want_minimap_player else None

            imgs_heatmap_frame = None
            if want_heatmap_player:
                if player_heat_torch is not None:
                    if player_heat_torch.dirty:
                        section_start = perf_counter()
                        player_heat_image = player_heat_torch.render()
                        frame_profile["heatmap_player"] = frame_profile.get("heatmap_player", 0.0) + (
                            perf_counter() - section_start
                        )
                    imgs_heatmap_frame = player_heat_image.copy()
                elif heatmap_accum is not None:
                    if player_heat_dirty:
                        section_start = perf_counter()
                        player_heat_image, heat_target = _render_heatmap_overlay(
                            heatmap_accum,
                            court_player_heat,
                            player_heat_sigma,
                            PLAYER_HEAT_PERCENTILE,
                            heat_target,
                            PLAYER_HEAT_INCREMENT,
                            PLAYER_HEAT_TARGET_DECAY,
                        )
                        frame_profile["heatmap_player"] = frame_profile.get("heatmap_player", 0.0) + (
                            perf_counter() - section_start
                        )
                        player_heat_dirty = False
                    imgs_heatmap_frame = player_heat_image.copy()

            imgs_ball_heatmap_frame = None
            if want_heatmap_ball:
                if ball_heat_torch is not None:
                    if ball_heat_torch.dirty:
                        section_start = perf_counter()
                        ball_heat_image = ball_heat_torch.render()
                        frame_profile["heatmap_ball"] = frame_profile.get("heatmap_ball", 0.0) + (
                            perf_counter() - section_start
                        )
                    imgs_ball_heatmap_frame = ball_heat_image.copy()
                elif ball_heatmap_accum is not None:
                    if ball_heat_dirty:
                        section_start = perf_counter()
                        ball_heat_image, ball_heat_target = _render_heatmap_overlay(
                            ball_heatmap_accum,
                            court_ball_heat,
                            ball_heat_sigma,
                            BALL_HEAT_PERCENTILE,
                            ball_heat_target,
                            BALL_HEAT_INCREMENT,
                            BALL_HEAT_TARGET_DECAY,
                        )
                        frame_profile["heatmap_ball"] = frame_profile.get("heatmap_ball", 0.0) + (
                            perf_counter() - section_start
                        )
                        ball_heat_dirty = False
                    imgs_ball_heatmap_frame = ball_heat_image.copy()

            frame_profile["total"] = frame_profile.get("total", 0.0) + (perf_counter() - frame_start)
            yield CombineFrameOutputs(
                combined=combined_frame,
                minimap_ball=imgs_ball_frame,
                minimap_player=imgs_person_frame,
                heatmap_player=imgs_heatmap_frame,
                heatmap_ball=imgs_ball_heatmap_frame,
                profiling=frame_profile,
            )


def combine(*args, render_options: Optional[CombineRenderOptions] = None, **kwargs):
    if render_options is None:
        render_options = CombineRenderOptions()
    imgs_res, imgs_ball, imgs_person, imgs_heatmap, imgs_ball_heatmap = [], [], [], [], []
    for outputs in combine_stream(*args, render_options=render_options, **kwargs):
        imgs_res.append(outputs.combined)
        imgs_ball.append(outputs.minimap_ball)
        imgs_person.append(outputs.minimap_player)
        imgs_heatmap.append(outputs.heatmap_player)
        imgs_ball_heatmap.append(outputs.heatmap_ball)
    return imgs_res, imgs_ball, imgs_person, imgs_heatmap, imgs_ball_heatmap
