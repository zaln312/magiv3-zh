from fastapi import APIRouter, HTTPException
from app.utils.state import state

router = APIRouter()


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
    return {"success": True}


@router.get("/character/panel_characters/{img_idx}")
async def get_panel_characters(img_idx: int):
    if img_idx < 0 or img_idx >= len(state.results):
        raise HTTPException(status_code=400, detail="img_idx 无效")

    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
    from ocr_utils import preprocess_panel_characters

    panel_chars = preprocess_panel_characters(state.results[img_idx])
    return {"panel_characters": panel_chars}