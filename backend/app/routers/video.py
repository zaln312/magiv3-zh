import io
import os
import base64
import threading

from fastapi import APIRouter, HTTPException
from PIL import Image
from app.utils.state import state
from app.services.app_config import get_video_config
from app.services.video_service import (
    collect_character_images_base64,
    submit_via_json_config,
    video_polling_manager,
)
from app.config import UPLOAD_DIR

router = APIRouter()


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
        if video_polling_manager.is_running(project_id):
            return {
                "success": True,
                "task_id": state.video_task_id,
                "state": state.video_task_state,
            }

    prose = state.prose
    images_base64 = collect_character_images_base64()

    try:
        submit_code = video_config.get("submit_code", "").strip()
        if submit_code:
            from app.utils.template_utils import execute_user_code

            result = execute_user_code(
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
            task_id, _ = submit_via_json_config(video_config, prose, images_base64)

        state.video_task_id = task_id
        state.video_task_state = "created"
        state.video_creations = []
        state.persist_step("video")
        state.persist_video()

        thread = threading.Thread(
            target=video_polling_manager.polling_worker,
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
    video_polling_manager.cancel(project_id)

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
