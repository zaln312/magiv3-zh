import io
import json
import os
import uuid
import base64
import threading
import time
import urllib.request
import urllib.error
import copy

from fastapi import APIRouter, HTTPException
from PIL import Image
from app.utils.state import state
from app.services.app_config import get_video_config
from app.config import UPLOAD_DIR

router = APIRouter()

_polling_stop_events: dict[str, threading.Event] = {}


def _deep_get(obj: dict, path: str, default=None):
    keys = path.split(".")
    current = obj
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        elif isinstance(current, list) and key.isdigit():
            idx = int(key)
            if idx < len(current):
                current = current[idx]
            else:
                return default
        else:
            return default
    return current


def _replace_placeholders(template, prose, images_base64):
    if isinstance(template, str):
        return template.replace("{prose}", prose)
    if isinstance(template, dict):
        result = {}
        for k, v in template.items():
            result[k] = _replace_placeholders(v, prose, images_base64)
        return result
    if isinstance(template, list):
        return [_replace_placeholders(item, prose, images_base64) for item in template]
    return template


def _check_images_marker(val, images_base64):
    if isinstance(val, str) and val == "{images_base64}":
        return images_base64
    if isinstance(val, dict):
        result = {}
        for k, v in val.items():
            result[k] = _check_images_marker(v, images_base64)
        return result
    if isinstance(val, list):
        return [_check_images_marker(item, images_base64) for item in val]
    return val


def _execute_user_code(code: str, func_name: str, **kwargs):
    namespace = {}
    exec(code, namespace)
    if func_name not in namespace:
        raise RuntimeError(f"用户代码中未定义 {func_name} 函数")
    return namespace[func_name](**kwargs)


def _http_request(method: str, url: str, headers: dict, body: dict | None = None):
    data_bytes = None
    if body is not None:
        data_bytes = json.dumps(body, ensure_ascii=False).encode("utf-8")
        if "Content-Type" not in headers:
            headers["Content-Type"] = "application/json"

    req = urllib.request.Request(url, data=data_bytes, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            resp_data = resp.read().decode("utf-8")
            return json.loads(resp_data)
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace") if e.fp else ""
        raise RuntimeError(f"HTTP {e.code}: {err_body[:500]}")
    except Exception as e:
        raise RuntimeError(f"请求失败: {str(e)}")


def _collect_character_images_base64():
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


def _submit_via_json_config(video_config: dict, prose: str, images_base64: list[str]):
    submit_cfg = video_config["submit"]
    submit_url = submit_cfg.get("url", "")
    if not submit_url:
        raise RuntimeError("未配置提交 URL (submit.url)")

    submit_method = submit_cfg.get("method", "POST").upper()
    headers = copy.deepcopy(submit_cfg.get("headers", {}))
    body_template = copy.deepcopy(submit_cfg.get("body_template", {}))
    task_id_path = submit_cfg.get("task_id_path", "id")

    body = _replace_placeholders(body_template, prose, images_base64)
    body = _check_images_marker(body, images_base64)

    response = _http_request(submit_method, submit_url, headers, body)
    task_id = _deep_get(response, task_id_path)
    if not task_id:
        raise RuntimeError(
            f"无法从响应中提取 task_id (路径: {task_id_path})，响应: {json.dumps(response, ensure_ascii=False)[:500]}"
        )

    return task_id, response


def _poll_via_json_config(video_config: dict, task_id: str):
    poll_cfg = video_config["poll"]
    url_template = poll_cfg.get("url_template", "")
    if not url_template:
        raise RuntimeError("未配置轮询 URL (poll.url_template)")

    poll_method = poll_cfg.get("method", "GET").upper()
    headers = copy.deepcopy(poll_cfg.get("headers", {}))

    url = url_template.replace("{task_id}", str(task_id))
    response = _http_request(poll_method, url, headers)

    state_path = poll_cfg.get("state_path", "state")
    raw_state = _deep_get(response, state_path, "")

    state_values = poll_cfg.get("state_values", {})
    mapped_state = ""
    for key, val in state_values.items():
        if str(raw_state) == str(val):
            mapped_state = key
            break
    if not mapped_state:
        mapped_state = str(raw_state)

    creations_path = poll_cfg.get("creations_path", "creations")
    creations_raw = _deep_get(response, creations_path, [])
    if not isinstance(creations_raw, list):
        creations_raw = []

    url_path = poll_cfg.get("url_path", "url")
    cover_url_path = poll_cfg.get("cover_url_path", "cover_url")

    creations = []
    for item in creations_raw:
        creation = {
            "url": _deep_get(item, url_path, ""),
            "cover_url": _deep_get(item, cover_url_path, ""),
        }
        creations.append(creation)

    return mapped_state, creations


def _polling_worker(project_id: str, task_id: str, video_config: dict):
    poll_cfg = video_config["poll"]
    interval = poll_cfg.get("interval_seconds", 5)
    max_attempts = poll_cfg.get("max_attempts", 120)
    poll_code = video_config.get("poll_code", "")

    stop_event = threading.Event()
    _polling_stop_events[project_id] = stop_event

    try:
        for attempt in range(max_attempts):
            if stop_event.is_set():
                return

            try:
                if poll_code.strip():
                    result = _execute_user_code(
                        poll_code,
                        "user_poll_video",
                        task_id=task_id,
                        config=video_config,
                    )
                    task_state = result.get("state", "")
                    creations = result.get("creations", [])
                else:
                    task_state, creations = _poll_via_json_config(video_config, task_id)
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
                            video_path = os.path.join(video_output_dir, video_filename)
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
        _polling_stop_events.pop(project_id, None)


@router.get("/video/prepare")
async def prepare_video_data():
    if not state.prose:
        raise HTTPException(status_code=400, detail="请先生成 Prose 叙述")

    characters = []
    for entry in state.global_character_library:
        gid = entry["global_id"]
        name = state.character_name_map.get(gid, f"角色{gid}")
        refs = state.character_references.get(gid, [])
        ref_images = []
        for ref in refs:
            ref_images.append(
                {
                    "image_base64": ref.get("image_base64", ""),
                    "view": ref.get("view", ref.get("mode", "")),
                }
            )
        design_images = []
        design_dir = os.path.join(
            UPLOAD_DIR, "design", state.project_id or "default", str(gid)
        )
        if os.path.isdir(design_dir):
            for fname in sorted(os.listdir(design_dir)):
                fpath = os.path.join(design_dir, fname)
                ext = os.path.splitext(fname)[1].lower()
                if ext not in (".png", ".jpg", ".jpeg", ".webp", ".gif"):
                    continue
                if not os.path.isfile(fpath):
                    continue
                try:
                    img = Image.open(fpath).convert("RGB")
                    buf = io.BytesIO()
                    img.save(buf, format="PNG")
                    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                    design_images.append(
                        {
                            "filename": fname,
                            "image_base64": b64,
                        }
                    )
                except Exception:
                    continue

        characters.append(
            {
                "global_id": gid,
                "name": name,
                "ref_images": ref_images,
                "design_images": design_images,
            }
        )

    return {
        "success": True,
        "prose": state.prose,
        "characters": characters,
    }


@router.post("/video/submit")
async def submit_video_task():
    if not state.prose:
        raise HTTPException(status_code=400, detail="请先生成 Prose 叙述")

    video_config = get_video_config()
    if not video_config.get("enabled"):
        raise HTTPException(status_code=400, detail="视频生成功能未启用")

    project_id = state.project_id or "default"

    if state.video_task_id and state.video_task_state not in ("success", "failed", ""):
        if project_id in _polling_stop_events:
            return {
                "success": True,
                "task_id": state.video_task_id,
                "state": state.video_task_state,
            }

    prose = state.prose
    images_base64 = _collect_character_images_base64()

    try:
        submit_code = video_config.get("submit_code", "").strip()
        if submit_code:
            result = _execute_user_code(
                submit_code,
                "user_submit_video",
                prose=prose,
                images_base64=images_base64,
                config=video_config,
            )
            task_id = result.get("task_id", "")
            if not task_id:
                raise RuntimeError("user_submit_video 未返回 task_id")
        else:
            task_id, _ = _submit_via_json_config(video_config, prose, images_base64)

        state.video_task_id = task_id
        state.video_task_state = "created"
        state.video_creations = []
        state.persist_step("video")
        state.persist_video()

        thread = threading.Thread(
            target=_polling_worker,
            args=(project_id, task_id, video_config),
            daemon=True,
        )
        thread.start()

        return {
            "success": True,
            "task_id": task_id,
            "state": "created",
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"视频任务提交失败: {str(e)}")


@router.get("/video/status")
async def get_video_status():
    return {
        "success": True,
        "task_id": state.video_task_id,
        "state": state.video_task_state,
        "creations": state.video_creations,
    }


@router.post("/video/cancel")
async def cancel_video_task():
    project_id = state.project_id or "default"
    if project_id in _polling_stop_events:
        _polling_stop_events[project_id].set()

    state.video_task_id = None
    state.video_task_state = ""
    state.video_creations = []
    state.video_result = None
    state.persist_video()

    return {"success": True}


@router.get("/video/result")
async def get_video_result():
    result = state.video_result
    return {
        "success": True,
        "video_result": result,
        "task_id": state.video_task_id,
        "state": state.video_task_state,
        "creations": state.video_creations,
    }
