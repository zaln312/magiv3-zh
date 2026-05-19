import os
import shutil
import logging

from fastapi import APIRouter, HTTPException
from app.utils.state import state
from app.services.database import (
    create_project,
    get_project,
    list_projects,
    update_project_name,
    delete_project,
    load_full_state,
    load_video,
)
from app.services.model_manager import model_manager
from app.services.app_config import get_magi_v3_mode

logger = logging.getLogger(__name__)

router = APIRouter()


def _enter_project(project_id: str):
    """Common entry logic: load project state and optionally preload magi v3 model."""
    state.load_project(project_id)
    mode = get_magi_v3_mode()
    if mode == "persistent_project":
        model_manager.set_persistent_mode(True)
        model_manager.maybe_load()
        logger.info(f"Persistent mode: model loaded for project {project_id}")
    else:
        model_manager.set_persistent_mode(False)


def _exit_project():
    """Common exit logic: unload magi v3 if persistent, reset state."""
    if model_manager.persistent_mode and model_manager.is_loaded:
        model_manager.unload()
        logger.info("Persistent mode: model unloaded on project exit")
    model_manager.set_persistent_mode(False)
    state._project_id = None
    state.reset_pipeline()


@router.get("/project/list")
async def api_list_projects():
    projects = list_projects()
    for p in projects:
        p["data_exists"] = bool(load_full_state(p["id"]))
        video_data = load_video(p["id"])
        if video_data:
            p["video_task_id"] = video_data.get("task_id")
            p["video_task_state"] = video_data.get("state", "")
            p["video_creations"] = video_data.get("creations", [])
        else:
            p["video_task_id"] = None
            p["video_task_state"] = ""
            p["video_creations"] = []
    return {"success": True, "projects": projects}


@router.post("/project/create")
async def api_create_project(data: dict = {}):
    name = data.get("name", "").strip() or "未命名项目"
    project = create_project(name)
    _enter_project(project["id"])
    return {"success": True, "project": project}


@router.post("/project/{project_id}/load")
async def api_load_project(project_id: str):
    try:
        _enter_project(project_id)
        return {"success": True, "project": get_project(project_id)}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/project/current")
async def api_current_project():
    if not state.project_id:
        return {"success": False, "project": None}
    return {"success": True, "project": get_project(state.project_id)}


@router.get("/project/{project_id}")
async def api_get_project(project_id: str):
    project = get_project(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="项目不存在")
    return {"success": True, "project": project}


@router.post("/project/{project_id}/rename")
async def api_rename_project(project_id: str, data: dict):
    name = data.get("name", "")
    update_project_name(project_id, name)
    return {"success": True}


@router.delete("/project/{project_id}")
async def api_delete_project(project_id: str):
    delete_project(project_id)
    from app.config import UPLOAD_DIR

    project_upload_dir = os.path.join(UPLOAD_DIR, project_id)
    if os.path.exists(project_upload_dir):
        shutil.rmtree(project_upload_dir)

    if state.project_id == project_id:
        _exit_project()

    return {"success": True}


@router.post("/project/exit")
async def api_exit_project():
    _exit_project()
    return {"success": True}


@router.post("/project/apply_magi_mode")
async def api_apply_magi_mode():
    """
    Apply magi v3 mode setting to current project:
    - If persistent mode selected and model not loaded → load model
    - If dynamic mode selected and model loaded → unload model
    Call after changing mode in settings to take effect immediately.
    """
    from app.services.app_config import get_magi_v3_mode

    mode = get_magi_v3_mode()

    if mode == "persistent_project":
        if not model_manager.is_loaded:
            model_manager.set_persistent_mode(True)
            model_manager.maybe_load()
            logger.info("Persistent mode activated: model loaded")
        else:
            model_manager.set_persistent_mode(True)
    else:
        if model_manager.persistent_mode and model_manager.is_loaded:
            model_manager.unload()
            logger.info("Dynamic mode activated: model unloaded")
        model_manager.set_persistent_mode(False)

    return {"success": True, "mode": mode, "model_loaded": model_manager.is_loaded}
