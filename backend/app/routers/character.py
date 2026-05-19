import os
import uuid
import base64
import io

from fastapi import APIRouter, HTTPException, UploadFile, File
from PIL import Image
from app.config import UPLOAD_DIR
from app.utils.state import state

router = APIRouter()


def _design_dir(global_id: int) -> str:
    d = os.path.join(
        UPLOAD_DIR, "design", state.project_id or "default", str(global_id)
    )
    os.makedirs(d, exist_ok=True)
    return d


@router.get("/character/library")
async def get_character_library():
    return {
        "global_character_library": state.global_character_library,
        "character_name_map": state.character_name_map,
    }


@router.post("/character/update_name")
async def update_character_name(data: dict):
    global_id = data.get("global_id")
    name = data.get("name", "")

    if global_id is None:
        raise HTTPException(status_code=400, detail="缺少 global_id")

    state.character_name_map[global_id] = name
    state.persist_global_characters()
    return {"success": True, "character_name_map": state.character_name_map}


@router.post("/character/update_global_id")
async def update_character_global_id(data: dict):
    img_idx = data.get("img_idx")
    char_idx = data.get("char_idx")
    new_global_id = data.get("new_global_id")

    if img_idx is None or char_idx is None or new_global_id is None:
        raise HTTPException(status_code=400, detail="缺少必要参数")
    if img_idx < 0 or img_idx >= len(state.results):
        raise HTTPException(status_code=400, detail="img_idx 无效")

    result = state.results[img_idx]
    gids = result.get("global_character_ids", [])
    if char_idx < 0 or char_idx >= len(gids):
        raise HTTPException(status_code=400, detail="char_idx 无效")

    gids[char_idx] = new_global_id
    state.persist_predict_single(img_idx)
    state.persist_global_characters()
    return {"success": True}


@router.get("/character/panel_characters/{img_idx}")
async def get_panel_characters(img_idx: int):
    if img_idx < 0 or img_idx >= len(state.results):
        raise HTTPException(status_code=400, detail="img_idx 无效")

    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
    from ocr_utils import preprocess_panel_characters

    panel_chars = preprocess_panel_characters(state.results[img_idx])
    return {"panel_characters": panel_chars}


@router.post("/character/library/add")
async def add_character_to_library():
    existing_ids = {entry["global_id"] for entry in state.global_character_library}
    assigned_gid = 0
    while assigned_gid in existing_ids:
        assigned_gid += 1
    state.global_character_library.append({"global_id": assigned_gid})
    state.persist_global_characters()
    return {"success": True, "global_id": assigned_gid}


@router.delete("/character/library/{global_id}")
async def delete_character_from_library(global_id: int):
    for img_idx, r in enumerate(state.results):
        gids = r.get("global_character_ids", [])
        if global_id in gids:
            raise HTTPException(
                status_code=400,
                detail=f"第 {img_idx + 1} 张图片中存在该角色的框",
            )

    state.global_character_library = [
        entry
        for entry in state.global_character_library
        if entry.get("global_id") != global_id
    ]
    state.character_name_map.pop(global_id, None)
    state.persist_global_characters()

    return {"success": True}


@router.get("/character/{global_id}/design_images")
async def get_design_images(global_id: int):
    d = _design_dir(global_id)
    images = []
    if not os.path.isdir(d):
        return {"success": True, "global_id": global_id, "images": images}
    for fname in sorted(os.listdir(d)):
        fpath = os.path.join(d, fname)
        if not os.path.isfile(fpath):
            continue
        ext = os.path.splitext(fname)[1].lower()
        if ext not in (".png", ".jpg", ".jpeg", ".webp", ".gif"):
            continue
        try:
            img = Image.open(fpath).convert("RGB")
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            images.append({"filename": fname, "image_base64": b64})
        except Exception:
            continue
    return {"success": True, "global_id": global_id, "images": images}


@router.post("/character/{global_id}/design_upload")
async def upload_design_image(global_id: int, file: UploadFile = File(...)):
    d = _design_dir(global_id)
    ext = os.path.splitext(file.filename or "design.jpg")[1] or ".jpg"
    filename = f"{uuid.uuid4().hex}{ext}"
    filepath = os.path.join(d, filename)
    content = await file.read()
    with open(filepath, "wb") as f:
        f.write(content)
    return {"success": True, "global_id": global_id, "filename": filename}


@router.delete("/character/{global_id}/design_image/{filename}")
async def delete_design_image(global_id: int, filename: str):
    d = _design_dir(global_id)
    filepath = os.path.join(d, filename)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="文件不存在")
    os.remove(filepath)
    return {"success": True}
