import logging
import os
import time
import shutil
import requests
from typing import Dict, Any
from pydantic import BaseModel, UUID4

# Import from /utils
from ..utils.common_utils import (
    get_hardware_inference_info,
    get_original_video_name,
    extract_first_frame_as_thumbnail,
    get_video_duration,
)
from ..utils_v2.supabase_utils import get, update
from ..utils_v2.gcs_utils import download, upload
from ..utils_v2.yolo_utils import inference_playerKeyPoint
from ..utils_v2.combine_pipeline import start_inference_pipeline
from ..utils_v2.mailtrap_utils import send_success_analysis_video
from .genai_services import image_understanding

# ---------------------------------------------------------------------
# Konfigurasi Log
# ---------------------------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


API_NOTIFY_URL = "https://courtplay.my.id/api/project/update"
GCS_BUCKET = "courtplay-storage"

def _notify_status(project_id: str, status: str = "done") -> None:
    """Kirim notifikasi status ke endpoint eksternal setelah update DB."""
    try:
        resp = requests.post(
            API_NOTIFY_URL,
            json={"project_id": project_id, "status": status, "x-api-key": "606730771dcfb011dbc96f9e154294eacd7933bb47d1bc348e926a53be8d34c5"},
            timeout=10,
        )
        resp.raise_for_status()
        logger.info(f"Notifikasi terkirim: project_id={project_id}, status={status}, response={resp.text}")
    except Exception as e:
        logger.warning(f"Gagal mengirim notifikasi ke {API_NOTIFY_URL}: {e}")


def _update_and_notify(
    table: str, data: Dict[str, Any], where: Dict[str, Any], project_id: str
) -> None:
    """Helper: update DB lalu kirim notifikasi."""
    update(table, data, where)
    _notify_status(project_id, "done")


def get_payload_data(payload_data: dict) -> Dict[str, str]:
    """Validasi dan normalisasi payload."""
    class InferencePayload(BaseModel):
        id: UUID4
        user_id: UUID4
        project_details_id: UUID4

    try:
        validated = InferencePayload(**payload_data)
        return {
            "project_id": str(validated.id),
            "user_id": str(validated.user_id),
            "project_details_id": str(validated.project_details_id),
        }
    except Exception as e:
        logger.error(f"Payload validation failed: {e}. Data received: {payload_data}")
        raise


def process_inference_task(payload: dict) -> None:
    """
    Pipeline inferensi video.
    Kini setiap selesai 1 proses:
      - langsung upload hasilnya (jika ada file)
      - langsung update DB (parsial/inkremental)
      - langsung kirim notifikasi POST
    """
    hw_info = get_hardware_inference_info()
    logger.info("Getting hardware info success")

    payload_data = get_payload_data(payload)
    pid = payload_data["project_id"]
    uid = payload_data["user_id"]
    pdetail_id = payload_data["project_details_id"]

    workdir = f"inference/{uid}/{pid}"
    original_mp4 = f"{workdir}/original_video.mp4"

    try:
        logger.info(f"Starting inference project {pid}")

        # 1) Ambil data user & project
        user_data = get("users", "username, email", {"id": uid}, False)
        project_data = get("projects", "project_name, project_details_id, upload_date", {"id": pid}, False)
        link_original_video = get("project_details", "link_video_original", {"id": project_data[0]["project_details_id"]})

        # 2) Download original video dari GCS
        download(
            GCS_BUCKET,
            f"uploads/videos/{uid}/{pid}/{get_original_video_name(link_original_video)}",
            original_mp4,
        )
        video_duration = int(get_video_duration(original_mp4))

        # -> Update durasi video (parsial)
        _update_and_notify(
            "project_details",
            {"video_duration": video_duration, "updated_at": "now()"},
            {"id": pdetail_id},
            pid,
        )

        # 3) Ekstrak thumbnail, upload, lalu update projects
        thumbnail_path = extract_first_frame_as_thumbnail(original_mp4, workdir)
        thumbnail_public_link = upload(
            GCS_BUCKET,
            thumbnail_path,
            f"uploads/videos/{uid}/{pdetail_id}/thumbnail.jpg",
        )
        _update_and_notify(
            "projects",
            {"link_image_thumbnail": thumbnail_public_link, "updated_at": "now()"},
            {"id": pid},
            pid,
        )

        # 4) Inference: Player KeyPoint -> upload -> update parsial (link + stroke count)
        kp_t0 = time.time()
        keypoint_result = inference_playerKeyPoint(uid, pid)
        kp_time = int(time.time() - kp_t0)

        video_player_keypoint_path = keypoint_result["video_mp4_format_path"]
        stroke_count = keypoint_result["stroke_counts"]

        video_player_keypoint_public_link = upload(
            GCS_BUCKET,
            video_player_keypoint_path,
            f"uploads/videos/{uid}/{pdetail_id}/playerKeyPoint.mp4",
        )
        _update_and_notify(
            "project_details",
            {
                "link_video_player_keypoints": video_player_keypoint_public_link,
                "ready_position_count": stroke_count.get("Ready_Position", 0),
                "forehand_count": stroke_count.get("Forehand", 0),
                "backhand_count": stroke_count.get("Backhand", 0),
                "serve_count": stroke_count.get("Serve", 0),
                "updated_at": "now()",
            },
            {"id": pdetail_id},
            pid,
        )
        # 5) Inference: Analytics combine (output video & gambar terbaru)
        analytics_dir = os.path.join(workdir, "analytics_v2")
        analytics_t0 = time.time()
        analytics_artifacts, analytics_metrics = start_inference_pipeline(original_mp4, analytics_dir)
        analytics_time = int(time.time() - analytics_t0)
        analytics_metrics = analytics_metrics or {}

        def _upload_analytics(local_path: str, remote_path: str) -> str:
            if not local_path or not os.path.exists(local_path):
                logger.warning(f"Analytics artifact missing, skip upload: {local_path}")
                return None
            return upload(GCS_BUCKET, local_path, remote_path)

        analytics_uploads = {
            "combined_video": _upload_analytics(
                analytics_artifacts.get("combined_video"),
                f"uploads/videos/{uid}/{pdetail_id}/output.mp4",
            ),
            "minimap_player_video": _upload_analytics(
                analytics_artifacts.get("minimap_player_video"),
                f"uploads/videos/{uid}/{pdetail_id}/minimap_player.mp4",
            ),
            "heatmap_player_video": _upload_analytics(
                analytics_artifacts.get("heatmap_player_video"),
                f"uploads/videos/{uid}/{pdetail_id}/heatmap_player.mp4",
            ),
            "minimap_ball_video": _upload_analytics(
                analytics_artifacts.get("minimap_ball_video"),
                f"uploads/videos/{uid}/{pdetail_id}/minimap_ball.mp4",
            ),
            "heatmap_player_image": _upload_analytics(
                analytics_artifacts.get("heatmap_player_image"),
                f"uploads/images/{uid}/{pdetail_id}/heatmap_player.png",
            ),
            "minimap_ball_image": _upload_analytics(
                analytics_artifacts.get("minimap_ball_image"),
                f"uploads/images/{uid}/{pdetail_id}/minimap_ball.png",
            ),
            "heatmap_ball_image": _upload_analytics(
                analytics_artifacts.get("heatmap_ball_image"),
                f"uploads/images/{uid}/{pdetail_id}/heatmap_ball.png",
            ),
        }

        prompt_player = (
            "You are a tennis instructor providing feedback on my match performance against my opponent. My opponent is positioned at the top of the court, and I am positioned at the bottom of the court. Analyze my heatmap as if you are reviewing the match, and provide your analysis in two sentences, not in bullet points."
        )
        prompt_ball = (
            "You are a tennis instructor providing feedback on my match performance against my opponent. My opponent is positioned at the top of the court, and I am positioned at the bottom of the court. Analyze the ball drop heatmap, focusing on where the ball makes contact with the court, and provide your analysis in two sentences, not in bullet points."
        )

        heatmap_player_local = analytics_artifacts.get("heatmap_player_image")
        heatmap_ball_local = analytics_artifacts.get("heatmap_ball_image")

        try:
            genai_heatmapPlayer_understanding = (
                image_understanding(heatmap_player_local, prompt_player)
                if heatmap_player_local and os.path.exists(heatmap_player_local)
                else None
            )
            genai_ballDroppings_understanding = (
                image_understanding(heatmap_ball_local, prompt_ball)
                if heatmap_ball_local and os.path.exists(heatmap_ball_local)
                else None
            )
        except Exception as e:
            logger.error(f"GenAI image understanding failed: {e}")
            genai_heatmapPlayer_understanding = "Model failed to generate insights."
            genai_ballDroppings_understanding = "Model failed to generate insights."

        analytics_update_payload = {"updated_at": "now()"}
        if analytics_uploads["combined_video"]:
            analytics_update_payload["link_video_object_detections"] = analytics_uploads["combined_video"]
        if analytics_uploads["heatmap_player_video"]:
            analytics_update_payload["link_video_heatmap_player"] = analytics_uploads["heatmap_player_video"]
        if analytics_uploads["minimap_player_video"]:
            analytics_update_payload["link_video_minimap_player"] = analytics_uploads["minimap_player_video"]
        if analytics_uploads["minimap_ball_video"]:
            analytics_update_payload["link_video_ball_droppings"] = analytics_uploads["minimap_ball_video"]
        if analytics_uploads["heatmap_player_image"]:
            analytics_update_payload["link_image_heatmap_player"] = analytics_uploads["heatmap_player_image"]
        if analytics_uploads["minimap_ball_image"]:
            analytics_update_payload["link_image_ball_droppings"] = analytics_uploads["minimap_ball_image"]
        if analytics_uploads["heatmap_ball_image"]:
            analytics_update_payload["link_image_heatmap_ball_droppings"] = analytics_uploads["heatmap_ball_image"]
        if genai_heatmapPlayer_understanding:
            analytics_update_payload["genai_heatmap_player_understanding"] = genai_heatmapPlayer_understanding
        if genai_ballDroppings_understanding:
            analytics_update_payload["genai_ball_droppings_understanding"] = genai_ballDroppings_understanding

        _update_and_notify(
            "project_details",
            analytics_update_payload,
            {"id": pdetail_id},
            pid,
        )

        # 6) Total processing time (setelah keypoint + analytics selesai)
        total_time = int(kp_time + analytics_time)
        _update_and_notify(
            "project_details",
            {"video_processing_time": total_time, "updated_at": "now()"},
            {"id": pdetail_id},
            pid,
        )
        hwinfo_payload = {
            "project_id": pid,
            "user_id": uid,
            "gpu_name": hw_info["gpu_name"],
            "vram_mb": hw_info["vram_mb"],
            "cpu_name": hw_info["cpu_name"],
            "cpu_threads": hw_info["cpu_threads"],
            "ram_mb": hw_info["ram_mb"],
            "os_info": hw_info["os_info"],
            "read_video_metadata": analytics_metrics.get("read_video_metadata", 0.0),
            "scene_detect": analytics_metrics.get("scene_detect", 0.0),
            "ball_detection": analytics_metrics.get("ball_detection", 0.0),
            "court_detection": analytics_metrics.get("court_detection", 0.0),
            "player_detection": analytics_metrics.get("player_detection", 0.0),
            "bounce_detection": analytics_metrics.get("bounce_detection", 0.0),
            "combine_render": analytics_metrics.get("combine_render", 0.0),
            "render_heatmap_player": analytics_metrics.get("render_heatmap_player", 0.0),
            "render_heatmap_ball": analytics_metrics.get("render_heatmap_ball", 0.0),
            "player_keypoint": kp_time,
            "is_success": True,
            "created_at": "now()",
            "updated_at": "now()",
        }
        _update_and_notify(
            "hwinfo",
            hwinfo_payload,
            {"project_id": pid},
            pid,
        )

        # 10) Kirim email, lalu update projects.is_mailed (tetap seperti semula)
        context_email = {
            "username": user_data[0]["username"],
            "project_name": project_data[0]["project_name"],
            "video_duration": video_duration,
            "upload_date": project_data[0]["upload_date"],
            "report_url": f"courtplay.my.id/analytics/{pid}",
            "heatmap_player_image_url": analytics_uploads.get("heatmap_player_image"),
            "heatmap_player_description": genai_heatmapPlayer_understanding,
            "ball_drop_image_url": analytics_uploads.get("minimap_ball_image"),
            "ball_drop_description": genai_ballDroppings_understanding,
        }

        if send_success_analysis_video(user_data[0]["email"], context_email):
            _update_and_notify(
                "projects",
                {"is_mailed": True, "updated_at": "now()"},
                {"id": pid},
                pid,
            )
            logger.info(f"Processing Project ID {pid} completed and email sent to {user_data[0]['email']}")

    except Exception as e:
        logger.error(f"Error during processing Project ID {pid}: {e}")
        raise
    finally:
        # Bersihkan workdir
        if os.path.exists(workdir):
            shutil.rmtree(workdir)
