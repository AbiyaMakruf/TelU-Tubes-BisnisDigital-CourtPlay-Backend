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
from ..utils.supabase_utils import get, update
from ..utils.gcs_utils import download, upload
from ..utils.yolo_utils import (
    inference_objectDetection,
    inference_playerKeyPoint,
    inference_heatmapPlayer_v1,
    inference_heatmapPlayer_v2,
    inference_heatmapBall,
)
from ..utils.mailtrap_utils import send_success_analysis_video
from .genai_services import image_understanding

# ---------------------------------------------------------------------
# Konfigurasi Log
# ---------------------------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------
# Konfigurasi Global + Constant
# ---------------------------------------------------------------------
GLOBAL_MODELS = {
    "objectDetection": None,
    "playerKeyPoint": None,
    "courtKeyPoint": None,
}

API_NOTIFY_URL = "https://courtplay.my.id/api/project/update"
GCS_BUCKET = "courtplay-storage"


def set_global_models(models: Dict[str, Any]) -> None:
    """Setter untuk model global (tetap)."""
    global GLOBAL_MODELS
    GLOBAL_MODELS = models
    logger.info("Global models set for inference service.")


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

        # 4) Inference: Object Detection -> upload -> update parsial
        det_t0 = time.time()
        video_object_detection_path = inference_objectDetection(uid, pid, GLOBAL_MODELS["objectDetection"])
        det_time = int(time.time() - det_t0)

        video_object_detection_public_link = upload(
            GCS_BUCKET,
            video_object_detection_path,
            f"uploads/videos/{uid}/{pdetail_id}/objectDetection.mp4",
        )
        _update_and_notify(
            "project_details",
            {
                "link_video_object_detections": video_object_detection_public_link,
                "updated_at": "now()",
            },
            {"id": pdetail_id},
            pid,
        )
        # -> simpan waktu deteksi di hwinfo
        _update_and_notify(
            "hwinfo",
            {
                "detection_inference_time": det_time,
                "gpu_name": hw_info["gpu_name"],
                "vram_mb": hw_info["vram_mb"],
                "cpu_name": hw_info["cpu_name"],
                "cpu_threads": hw_info["cpu_threads"],
                "ram_mb": hw_info["ram_mb"],
                "os_info": hw_info["os_info"],
            },
            {"project_id": pid},
            pid,
        )

        # 5) Inference: Player KeyPoint -> upload -> update parsial (link + stroke count)
        kp_t0 = time.time()
        keypoint_result = inference_playerKeyPoint(uid, pid, GLOBAL_MODELS["playerKeyPoint"])
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
        # -> simpan waktu keypoint di hwinfo
        _update_and_notify(
            "hwinfo",
            {"keypoint_inference_time": kp_time},
            {"project_id": pid},
            pid,
        )

        # 6) (Opsional) Court KeyPoint placeholder tetap seperti semula
        video_court_keypoint_public_link = "IN DEVELOPMENT"
        # -> Tidak di-upload & tidak mengubah fungsionalitas; namun simpan placeholder jika ingin
        _update_and_notify(
            "project_details",
            {"link_video_court_keypoints": video_court_keypoint_public_link, "updated_at": "now()"},
            {"id": pdetail_id},
            pid,
        )

        # 7) Inference: Heatmap Player -> upload -> update parsial (+ GenAI understanding)
        image_heatmapPlayer_path = inference_heatmapPlayer_v2(
            uid, pid, GLOBAL_MODELS["objectDetection"], GLOBAL_MODELS["courtKeyPoint"]
        )
        prompt_player = (
            "You are a tennis instructor providing feedback on my match performance against my opponent. "
            "My opponent is positioned on the left side of the court, and I am on the right side. "
            "Analyze my heatmap as if you are reviewing the match — write your analysis in two sentences, not in bullet points."
        )
        genai_heatmapPlayer_understanding = image_understanding(image_heatmapPlayer_path, prompt_player)

        image_heatmapPlayer_public_link = upload(
            GCS_BUCKET,
            image_heatmapPlayer_path,
            f"uploads/images/{uid}/{pdetail_id}/heatmapPlayer.png",
        )
        _update_and_notify(
            "project_details",
            {
                "link_image_heatmap_player": image_heatmapPlayer_public_link,
                "genai_heatmap_player_understanding": genai_heatmapPlayer_understanding,
                "updated_at": "now()",
            },
            {"id": pdetail_id},
            pid,
        )

        # 8) Inference: Heatmap Ball -> upload -> update parsial (+ GenAI understanding)
        image_heatmapBall_path = inference_heatmapBall(
            uid, pid, GLOBAL_MODELS["objectDetection"], GLOBAL_MODELS["courtKeyPoint"]
        )
        prompt_ball = (
            "You are a tennis instructor providing feedback on my match performance against my opponent. "
            "My opponent is positioned on the left side of the court, and I am on the right side. "
            "Analyze my ball dropping heatmap as if you are reviewing the match — write your analysis in two sentences, not in bullet points."
        )
        genai_ballDroppings_understanding = image_understanding(image_heatmapBall_path, prompt_ball)

        image_ball_droppings_public_link = upload(
            GCS_BUCKET,
            image_heatmapBall_path,
            f"uploads/images/{uid}/{pdetail_id}/ballDroppings.png",
        )
        _update_and_notify(
            "project_details",
            {
                "link_image_ball_droppings": image_ball_droppings_public_link,
                "genai_ball_droppings_understanding": genai_ballDroppings_understanding,
                "updated_at": "now()",
            },
            {"id": pdetail_id},
            pid,
        )

        # 9) Total processing time (baru bisa dihitung setelah det + kp selesai)
        total_time = int(det_time + kp_time)
        _update_and_notify(
            "project_details",
            {"video_processing_time": total_time, "updated_at": "now()"},
            {"id": pdetail_id},
            pid,
        )
        _update_and_notify(
            "hwinfo",
            {"total_inference_time": total_time, "is_success": True},
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
