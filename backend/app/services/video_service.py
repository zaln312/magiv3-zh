import io
import os
import uuid
import json
import base64
import copy
import threading
import time

from PIL import Image
from app.utils.state import state
from app.utils.template_utils import (
    deep_get,
    replace_placeholders,
    check_images_marker,
    execute_user_code,
    http_request,
)
from app.config import UPLOAD_DIR


def collect_character_images_base64():
    images = []
    for entry in state.global_character_library:
        gid = entry["global_id"]
        refs = state.character_references.get(gid, [])
        for ref in refs:
            b64 = ref.get("image_base64", "")
            if b64:
                images.append(b64)

    design_upload_dir = os.path.join(
        UPLOAD_DIR, "design", state.project_id or "default"
    )
    if os.path.isdir(design_upload_dir):
        for gid_dir in sorted(os.listdir(design_upload_dir)):
            gid_path = os.path.join(design_upload_dir, gid_dir)
            if not os.path.isdir(gid_path):
                continue
            for fname in sorted(os.listdir(gid_path)):
                fpath = os.path.join(gid_path, fname)
                ext = os.path.splitext(fname)[1].lower()
                if ext not in (".png", ".jpg", ".jpeg", ".webp"):
                    continue
                try:
                    img = Image.open(fpath).convert("RGB")
                    buf = io.BytesIO()
                    img.save(buf, format="PNG")
                    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                    images.append(b64)
                except Exception:
                    continue
    return images


def submit_via_json_config(video_config: dict, prose: str, images_base64: list[str]):
    submit_cfg = video_config["submit"]
    submit_url = submit_cfg.get("url", "")
    if not submit_url:
        raise RuntimeError("未配置提交 URL (submit.url)")

    submit_method = submit_cfg.get("method", "POST").upper()
    headers = copy.deepcopy(submit_cfg.get("headers", {}))
    body_template = copy.deepcopy(submit_cfg.get("body_template", {}))
    task_id_path = submit_cfg.get("task_id_path", "id")

    body = replace_placeholders(body_template, prose, images_base64)
    body = check_images_marker(body, images_base64)

    response = http_request(submit_method, submit_url, headers, body)
    task_id = deep_get(response, task_id_path)
    if not task_id:
        raise RuntimeError(
            f"无法从响应中提取 task_id (路径: {task_id_path})，响应: {json.dumps(response, ensure_ascii=False)[:500]}"
        )

    return task_id, response


def poll_via_json_config(video_config: dict, task_id: str):
    poll_cfg = video_config["poll"]
    url_template = poll_cfg.get("url_template", "")
    if not url_template:
        raise RuntimeError("未配置轮询 URL (poll.url_template)")

    poll_method = poll_cfg.get("method", "GET").upper()
    headers = copy.deepcopy(poll_cfg.get("headers", {}))

    url = url_template.replace("{task_id}", str(task_id))
    response = http_request(poll_method, url, headers)

    state_path = poll_cfg.get("state_path", "state")
    raw_state = deep_get(response, state_path, "")

    state_values = poll_cfg.get("state_values", {})
    mapped_state = ""
    for key, val in state_values.items():
        if str(raw_state) == str(val):
            mapped_state = key
            break
    if not mapped_state:
        mapped_state = str(raw_state)

    creations_path = poll_cfg.get("creations_path", "creations")
    creations_raw = deep_get(response, creations_path, [])
    if not isinstance(creations_raw, list):
        creations_raw = []

    url_path = poll_cfg.get("url_path", "url")
    cover_url_path = poll_cfg.get("cover_url_path", "cover_url")

    creations = []
    for item in creations_raw:
        creation = {
            "url": deep_get(item, url_path, ""),
            "cover_url": deep_get(item, cover_url_path, ""),
        }
        creations.append(creation)

    return mapped_state, creations


class VideoPollingManager:
    def __init__(self):
        self._stop_events: dict[str, threading.Event] = {}

    def polling_worker(self, project_id: str, task_id: str, video_config: dict):
        poll_cfg = video_config["poll"]
        interval = poll_cfg.get("interval_seconds", 5)
        max_attempts = poll_cfg.get("max_attempts", 120)
        poll_code = video_config.get("poll_code", "")

        stop_event = threading.Event()
        self._stop_events[project_id] = stop_event

        try:
            for attempt in range(max_attempts):
                if stop_event.is_set():
                    return

                try:
                    if poll_code.strip():
                        result = execute_user_code(
                            poll_code,
                            "user_poll_video",
                            task_id=task_id,
                            config=video_config,
                        )
                        task_state = result.get("state", "")
                        creations = result.get("creations", [])
                    else:
                        task_state, creations = poll_via_json_config(
                            video_config, task_id
                        )
                except Exception as e:
                    state.video_task_state = "failed"
                    state.video_creations = []
                    state.video_result = {"error": str(e)}
                    state.persist_video()
                    return

                state.video_task_state = task_state
                state.video_creations = creations
                state.persist_video()

                if task_state in ("success", "failed"):
                    if task_state == "success" and creations:
                        video_output_dir = os.path.join(
                            UPLOAD_DIR, "video", project_id or "default"
                        )
                        os.makedirs(video_output_dir, exist_ok=True)

                        video_url = creations[0].get("url", "")
                        if video_url and video_url.startswith("data:"):
                            try:
                                _, b64_part = video_url.split(";base64,", 1)
                                video_bytes = base64.b64decode(b64_part)
                                video_filename = f"{uuid.uuid4().hex}.mp4"
                                video_path = os.path.join(
                                    video_output_dir, video_filename
                                )
                                with open(video_path, "wb") as f:
                                    f.write(video_bytes)
                                state.video_result = {
                                    "filename": video_filename,
                                    "video_base64": b64_part,
                                    "prose": state.prose,
                                    "creations": creations,
                                }
                                state.persist_video()
                            except Exception:
                                pass
                        else:
                            state.video_result = {
                                "prose": state.prose,
                                "creations": creations,
                            }
                            state.persist_video()
                    return

                time.sleep(interval)

            state.video_task_state = "failed"
            state.video_creations = []
            state.video_result = {"error": "轮询超时，超过最大尝试次数"}
            state.persist_video()

        finally:
            self._stop_events.pop(project_id, None)

    def cancel(self, project_id: str):
        if project_id in self._stop_events:
            self._stop_events[project_id].set()

    def is_running(self, project_id: str) -> bool:
        return project_id in self._stop_events


video_polling_manager = VideoPollingManager()
