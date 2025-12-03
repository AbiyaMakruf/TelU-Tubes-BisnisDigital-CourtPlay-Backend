import os
import shutil
import logging
import threading
import cv2
from ultralytics import solutions, YOLO
from sports.common.view import ViewTransformer
from scipy.stats import gaussian_kde

from .common_utils import convert_avi_to_mp4
from .tennis_heatmap_utils import *

# Konfigurasi lock
CONVERSION_LOCK = threading.Lock()
OD_LOCK = threading.Lock()
PKP_LOCK = threading.Lock()
HM_LOCK =  threading.Lock()

# Konfigurasi Log
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def inference_objectDetection(user_id, project_id):
    project_path = os.path.join("inference",f"{user_id}", f"{project_id}")
    temporary_path = "temporary_objectDetection"
    video_original_path = f"{project_path}/original_video.mp4"
    video_avi_format_path = os.path.join(project_path, temporary_path, "original_video.avi")
    video_mp4_format_path = os.path.join(project_path, "objectDetection_video.mp4")

    model = YOLO("models/objectDetection/objectDetection.pt")

    try:
        logger.info(f"Starting inference object detection. {project_id}")
        
        with OD_LOCK:
            logger.info(f"🔒Acquired lock for object detection inference. {project_id}")
            results = model(
                source=video_original_path,
                stream=True,
                save=True,
                project=project_path,
                name=temporary_path,
                # verbose=False,
                exist_ok=True
            )

            for result in results:
                result.save(filename=f"OD-{project_id}.png")

            logger.info(f"🔓Released lock for object detection inference. {project_id}")

        logger.info(f"Converting avi to mp4 format. {project_id}")
        with CONVERSION_LOCK:
            logger.info(f"🔒Acquired lock for video conversion OD. {project_id}")
            convert_avi_to_mp4(video_avi_format_path, video_mp4_format_path)
            logger.info(f"🔓Released lock for video conversion OD. {project_id}")
        os.remove(f"OD-{project_id}.png")

        logger.info(f"Success inference object detection. {project_id}")
        return video_mp4_format_path
    
    except Exception as e:
        logger.error(f"Error inference object detection. {e}")

    finally:
        delete_path = os.path.join(project_path,temporary_path)
        if os.path.exists(delete_path):
            shutil.rmtree(delete_path)

def inference_playerKeyPoint(user_id, project_id):
    project_path = os.path.join("inference",f"{user_id}", f"{project_id}")
    temporary_path = "temporary_playerKeyPoint"
    video_original_path = f"{project_path}/original_video.mp4"
    video_avi_format_path = os.path.join(project_path, temporary_path, "original_video.avi")
    video_mp4_format_path = os.path.join(project_path, "playerKeyPoint_video.mp4")

    model = YOLO("models/playerKeyPoint/playerKeyPoint.pt")
    
    stroke_counts = {
        'Backhand': 0,
        'Forehand': 0,
        'Ready_Position': 0,
        'Serve': 0
    }

    try:
        logger.info(f"Starting inference player keypoint. {project_id}")

        with PKP_LOCK:
            logger.info(f"🔒Acquired lock for player keypoint inference. {project_id}")
            results = model(
                source=video_original_path,
                stream=True,
                save=True,
                project=project_path,
                name=temporary_path,
                # verbose=False,
                exist_ok=True
            )

            for result in results:
                detected_class_ids = result.boxes.cls.tolist()

                class_names = model.names

                for class_id in detected_class_ids:
                    class_name = class_names[int(class_id)]
                    
                    if class_name in stroke_counts:
                        stroke_counts[class_name] += 1

                result.save(filename=f"PKP-{project_id}.png")
            logger.info(f"🔓Released lock for player keypoint inference. {project_id}")

        logger.info(f"Converting avi to mp4 format. {project_id}")
        with CONVERSION_LOCK:
            logger.info(f"🔒Acquired lock for video conversion PKP. {project_id}")
            convert_avi_to_mp4(video_avi_format_path, video_mp4_format_path)
            logger.info(f"🔓Released lock for video conversion PKP. {project_id}")
        os.remove(f"PKP-{project_id}.png")

        logger.info(f"Success inference player keypoint. {project_id}")

        return {
            'video_mp4_format_path':video_mp4_format_path,
            'stroke_counts': stroke_counts
        }
    
    except Exception as e:
        logger.error(f"Error inference player keypoint. {e}")

    finally:
        delete_path = os.path.join(project_path,temporary_path)
        if os.path.exists(delete_path):
            shutil.rmtree(delete_path)

def inference_heatmapPlayer_v1(user_id, project_id, model):
    project_path = os.path.join("inference",f"{user_id}",f"{project_id}")
    temporary_path = "temporary_heatmapPlayer"
    video_original_path = f"{project_path}/original_video.mp4"
    image_heatmap_player_path = f"{project_path}/heatmapPlayer.png"
    
    try:
        with HM_LOCK:
            logger.info(f"🔒Acquired lock for heatmap player inference. {project_id}")
            cap = cv2.VideoCapture(video_original_path)

            heatmap = solutions.Heatmap(
                show=False, 
                model=model, 
                colormap=cv2.COLORMAP_PARULA, 
                # region=region_points, 
                # classes=[0, 2], 
                verbose=False
            )

            last_heatmap_frame = None

            while cap.isOpened():
                success, im0 = cap.read()

                if not success:
                    break

                results = heatmap(im0)

                if results.plot_im is not None:
                    last_heatmap_frame = results.plot_im.copy()
            cap.release()
            if last_heatmap_frame is not None:
                rotated_heatmap = cv2.rotate(last_heatmap_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                cv2.imwrite(image_heatmap_player_path, rotated_heatmap)
            cv2.destroyAllWindows()
            logger.info(f"🔓Released lock for heatmap player inference. {project_id}")
            return image_heatmap_player_path
    except Exception as e:
        logger.error(f"Error inference heatmap player. {e}")
    finally:
        delete_path = os.path.join(project_path,temporary_path)
        if os.path.exists(delete_path):
            shutil.rmtree(delete_path)

def inference_heatmapPlayer_v2(user_id, project_id, model_objectDetection, model_heatmap):
    # === init path & constants ===
    project_path = os.path.join("inference", f"{user_id}", f"{project_id}")
    temporary_path = "temporary_heatmapPlayer"
    video_original_path = f"{project_path}/original_video.mp4"
    image_heatmap_player_path = f"{project_path}/heatmapPlayer.png"

    CONFIG = TennisCourtConfiguration()
    PLAYER_ID = 1
    MIN_CONFIDENCE = 0.3
    KP_CONFIDENCE = 0.5
    REQUIRED_KP_POINTS = 4
    RADAR_SCALE = 0.4
    RADAR_PADDING = 50
    OVERLAY_OPACITY = 0.6
    LOG_EVERY = 200

    cap = cv2.VideoCapture(video_original_path)
    all_pitch_points = []
    transformer: Optional[ViewTransformer] = None

    # === homography (court keypoints -> 2D court) ===
    def get_homography_transformer(frame: np.ndarray) -> Optional[ViewTransformer]:
        try:
            result = model_heatmap(frame, conf=MIN_CONFIDENCE)[0]
            key_points = sv.KeyPoints.from_ultralytics(result)
            mask = key_points.confidence[0] > KP_CONFIDENCE
            if int(np.sum(mask)) >= REQUIRED_KP_POINTS:
                src = key_points.xy[0][mask]
                dst = np.array(CONFIG.keypoints)[mask]
                return ViewTransformer(source=src, target=dst)
        except Exception:
            return None
        return None

    try:
        with HM_LOCK:
            logger.info(f"🔒Acquired lock for heatmap player inference. {project_id}")
            frame_count = 0

            # === iterate frames: (1) homography, (2) detect players, (3) project anchors ===
            while cap.isOpened():
                ok, frame = cap.read()
                if not ok:
                    break

                if transformer is None:
                    transformer = get_homography_transformer(frame)
                    if transformer is None:
                        frame_count += 1
                        continue

                results = model_objectDetection(frame, conf=MIN_CONFIDENCE)[0]
                detections = sv.Detections.from_ultralytics(results)
                player_det = detections[detections.class_id == PLAYER_ID]

                if len(player_det) > 0:
                    anchors = player_det.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
                    proj = transformer.transform_points(points=anchors)
                    valid = proj[~np.isnan(proj).any(axis=1)]
                    if len(valid) > 0:
                        all_pitch_points.extend(valid)

                frame_count += 1
                if frame_count % LOG_EVERY == 0:
                    logger.info(f"Processed {frame_count} frames | collected {len(all_pitch_points)} coords")

            cap.release()

            # === render base court ===
            canvas = draw_court(CONFIG, scale=RADAR_SCALE, padding=RADAR_PADDING)
            h_canvas, w_canvas, _ = canvas.shape

            if len(all_pitch_points) == 0:
                rotated = cv2.rotate(canvas.astype(np.uint8), cv2.ROTATE_90_COUNTERCLOCKWISE)
                cv2.imwrite(image_heatmap_player_path, rotated)
                logger.info(f"🔓Released lock for heatmap player inference. {project_id}")
                return image_heatmap_player_path

            pts = np.array(all_pitch_points, dtype=float)
            scaled = (pts * RADAR_SCALE) + RADAR_PADDING

            # === heatmap (KDE) ===
            try:
                if scaled.shape[0] < 5:
                    raise ValueError("Insufficient points for KDE")

                X = scaled[:, 0]
                Y = scaled[:, 1]
                data = np.vstack([X, Y])

                xi, yi = np.mgrid[0:w_canvas:w_canvas*1j, 0:h_canvas:h_canvas*1j]
                coords = np.vstack([xi.ravel(), yi.ravel()])

                kde = gaussian_kde(data, bw_method=0.1)
                density = kde(coords).reshape(xi.shape)

                d_norm = (density - density.min()) / (density.max() - density.min() + 1e-12)
                heat_u8 = (d_norm * 255).astype(np.uint8).T

                heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)

                alpha = (heat_u8 > 0).astype(np.uint8)
                alpha3 = cv2.cvtColor(alpha * 255, cv2.COLOR_GRAY2BGR)
                blended = cv2.addWeighted(heat_color, OVERLAY_OPACITY, canvas, 1 - OVERLAY_OPACITY, 0)
                result = np.where(alpha3 > 0, blended, canvas).astype(np.uint8)

            except Exception as e:
                logger.warning(f"Heatmap KDE failed ({e}); fallback to raw points.")
                result = canvas.copy().astype(np.uint8)
                for x, y in scaled.astype(int):
                    cv2.circle(result, (int(x), int(y)), 3, sv.Color.RED.as_bgr(), -1)

            # === save ===
            rotated = cv2.rotate(result, cv2.ROTATE_90_COUNTERCLOCKWISE)
            cv2.imwrite(image_heatmap_player_path, rotated)
            logger.info(f"🔓Released lock for heatmap player inference. {project_id}")
            return image_heatmap_player_path

    except Exception as e:
        logger.error(f"Error inference heatmap player. {e}")
    finally:
        try:
            delete_path = os.path.join(project_path, temporary_path)
            if os.path.exists(delete_path):
                shutil.rmtree(delete_path)
        except Exception:
            pass

def inference_heatmapBall(user_id, project_id, model_objectDetection, model_heatmap):
    # === init path & constants ===
    project_path = os.path.join("inference", f"{user_id}", f"{project_id}")
    temporary_path = "temporary_heatmapBall"
    video_original_path = f"{project_path}/original_video.mp4"
    image_heatmap_ball_path = f"{project_path}/heatmapBall.png"

    CONFIG = TennisCourtConfiguration()
    PLAYER_ID = 0
    MIN_CONFIDENCE = 0.3
    KP_CONFIDENCE = 0.5
    REQUIRED_KP_POINTS = 4
    RADAR_SCALE = 0.4
    RADAR_PADDING = 50
    OVERLAY_OPACITY = 0.6
    LOG_EVERY = 200

    cap = cv2.VideoCapture(video_original_path)
    all_pitch_points = []
    transformer: Optional[ViewTransformer] = None

    # === homography (court keypoints -> 2D court) ===
    def get_homography_transformer(frame: np.ndarray) -> Optional[ViewTransformer]:
        try:
            result = model_heatmap(frame, conf=MIN_CONFIDENCE)[0]
            key_points = sv.KeyPoints.from_ultralytics(result)
            mask = key_points.confidence[0] > KP_CONFIDENCE
            if int(np.sum(mask)) >= REQUIRED_KP_POINTS:
                src = key_points.xy[0][mask]
                dst = np.array(CONFIG.keypoints)[mask]
                return ViewTransformer(source=src, target=dst)
        except Exception:
            return None
        return None

    try:
        with HM_LOCK:
            logger.info(f"🔒Acquired lock for heatmap ball inference. {project_id}")
            frame_count = 0

            # === iterate frames: (1) homography, (2) detect ball, (3) project anchors ===
            while cap.isOpened():
                ok, frame = cap.read()
                if not ok:
                    break

                if transformer is None:
                    transformer = get_homography_transformer(frame)
                    if transformer is None:
                        frame_count += 1
                        continue

                results = model_objectDetection(frame, conf=MIN_CONFIDENCE)[0]
                detections = sv.Detections.from_ultralytics(results)
                player_det = detections[detections.class_id == PLAYER_ID]

                if len(player_det) > 0:
                    anchors = player_det.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
                    proj = transformer.transform_points(points=anchors)
                    valid = proj[~np.isnan(proj).any(axis=1)]
                    if len(valid) > 0:
                        all_pitch_points.extend(valid)

                frame_count += 1
                if frame_count % LOG_EVERY == 0:
                    logger.info(f"Processed {frame_count} frames | collected {len(all_pitch_points)} coords")

            cap.release()

            # === render base court ===
            canvas = draw_court(CONFIG, scale=RADAR_SCALE, padding=RADAR_PADDING)
            h_canvas, w_canvas, _ = canvas.shape

            if len(all_pitch_points) == 0:
                rotated = cv2.rotate(canvas.astype(np.uint8), cv2.ROTATE_90_COUNTERCLOCKWISE)
                cv2.imwrite(image_heatmap_ball_path, rotated)
                return image_heatmap_ball_path

            pts = np.array(all_pitch_points, dtype=float)
            scaled = (pts * RADAR_SCALE) + RADAR_PADDING

            # === heatmap (KDE) ===
            try:
                if scaled.shape[0] < 5:
                    raise ValueError("Insufficient points for KDE")

                X = scaled[:, 0]
                Y = scaled[:, 1]
                data = np.vstack([X, Y])

                xi, yi = np.mgrid[0:w_canvas:w_canvas*1j, 0:h_canvas:h_canvas*1j]
                coords = np.vstack([xi.ravel(), yi.ravel()])

                kde = gaussian_kde(data, bw_method=0.1)
                density = kde(coords).reshape(xi.shape)

                d_norm = (density - density.min()) / (density.max() - density.min() + 1e-12)
                heat_u8 = (d_norm * 255).astype(np.uint8).T

                heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)

                alpha = (heat_u8 > 0).astype(np.uint8)
                alpha3 = cv2.cvtColor(alpha * 255, cv2.COLOR_GRAY2BGR)
                blended = cv2.addWeighted(heat_color, OVERLAY_OPACITY, canvas, 1 - OVERLAY_OPACITY, 0)
                result = np.where(alpha3 > 0, blended, canvas).astype(np.uint8)

            except Exception as e:
                logger.warning(f"Heatmap KDE failed ({e}); fallback to raw points.")
                result = canvas.copy().astype(np.uint8)
                for x, y in scaled.astype(int):
                    cv2.circle(result, (int(x), int(y)), 3, sv.Color.RED.as_bgr(), -1)

            # === save ===
            rotated = cv2.rotate(result, cv2.ROTATE_90_COUNTERCLOCKWISE)
            cv2.imwrite(image_heatmap_ball_path, rotated)
            logger.info(f"🔓Released lock for heatmap ball inference. {project_id}")
            return image_heatmap_ball_path

    except Exception as e:
        logger.error(f"Error inference heatmap ball. {e}")
    finally:
        try:
            delete_path = os.path.join(project_path, temporary_path)
            if os.path.exists(delete_path):
                shutil.rmtree(delete_path)
        except Exception:
            pass

# # To Do
# - Ball drop heatmap