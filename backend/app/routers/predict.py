import sys
import os

from fastapi import APIRouter, HTTPException
from app.utils.state import state
from app.services.model_manager import model_manager

router = APIRouter()


@router.post("/predict/run")
async def run_predict():
    if not state.unordered_ocr_res:
        raise HTTPException(status_code=400, detail="请先执行 OCR 识别")

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

    try:
        model_manager.load()
        from ocr_utils import run_detection, predict_with_injected_ocr_and_global_id

        images, batch_inputs, generated_ids, results = run_detection(
            model_manager.model,
            model_manager.processor,
            state.img_paths,
        )

        state.results = predict_with_injected_ocr_and_global_id(
            model_manager.model,
            model_manager.processor,
            images,
            batch_inputs,
            generated_ids,
            results,
            state.unordered_ocr_res,
            global_character_library=state.global_character_library,
            debug=False,
        )
        state.reset_from_predict()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Predict 失败: {str(e)}")
    finally:
        model_manager.unload()

    return {
        "success": True,
        "results": _serialize_results(state.results),
        "count": len(state.results),
    }


@router.get("/predict/results")
async def get_predict_results():
    return {
        "results": _serialize_results(state.results),
        "count": len(state.results),
    }


@router.post("/predict/update_character_box")
async def update_character_box(data: dict):
    img_idx = data.get("img_idx")
    char_idx = data.get("char_idx")
    box = data.get("box")

    if img_idx is None or char_idx is None or box is None:
        raise HTTPException(status_code=400, detail="缺少必要参数")
    if img_idx < 0 or img_idx >= len(state.results):
        raise HTTPException(status_code=400, detail="img_idx 无效")

    chars = state.results[img_idx].get("characters", [])
    if char_idx < 0 or char_idx >= len(chars):
        raise HTTPException(status_code=400, detail="char_idx 无效")
    if len(box) != 4:
        raise HTTPException(status_code=400, detail="box 格式应为 [x1, y1, x2, y2]")

    chars[char_idx] = box
    return {"success": True}


@router.post("/predict/delete_character")
async def delete_character(data: dict):
    img_idx = data.get("img_idx")
    char_idx = data.get("char_idx")

    if img_idx is None or char_idx is None:
        raise HTTPException(status_code=400, detail="缺少必要参数")
    if img_idx < 0 or img_idx >= len(state.results):
        raise HTTPException(status_code=400, detail="img_idx 无效")

    result = state.results[img_idx]
    chars = result.get("characters", [])
    if char_idx < 0 or char_idx >= len(chars):
        raise HTTPException(status_code=400, detail="char_idx 无效")

    chars.pop(char_idx)

    labels = result.get("global_character_ids", [])
    if char_idx < len(labels):
        labels.pop(char_idx)

    old_associations = result.get("text_character_associations", [])
    new_associations = []
    for t_idx, c_idx in old_associations:
        if c_idx == char_idx:
            continue
        if c_idx > char_idx:
            c_idx -= 1
        new_associations.append([t_idx, c_idx])
    result["text_character_associations"] = new_associations

    return {"success": True}


@router.post("/predict/add_character")
async def add_character(data: dict):
    img_idx = data.get("img_idx")
    box = data.get("box")

    if img_idx is None or box is None:
        raise HTTPException(status_code=400, detail="缺少必要参数")
    if img_idx < 0 or img_idx >= len(state.results):
        raise HTTPException(status_code=400, detail="img_idx 无效")
    if len(box) != 4:
        raise HTTPException(status_code=400, detail="box 格式应为 [x1, y1, x2, y2]")

    result = state.results[img_idx]
    result.setdefault("characters", []).append(box)

    if state.global_character_library:
        existing_ids = {entry["global_id"] for entry in state.global_character_library}
        assigned_gid = min(existing_ids)
    else:
        used_ids = set()
        for r in state.results:
            for gid in r.get("global_character_ids", []):
                if isinstance(gid, (int, float)):
                    used_ids.add(int(gid))
        for entry in state.global_character_library:
            used_ids.add(entry["global_id"])
        assigned_gid = 0
        while assigned_gid in used_ids:
            assigned_gid += 1
        state.global_character_library.append({"global_id": assigned_gid})

    result.setdefault("global_character_ids", []).append(assigned_gid)

    return {"success": True, "char_idx": len(result["characters"]) - 1, "global_id": assigned_gid}


@router.post("/predict/update_text_char_association")
async def update_text_char_association(data: dict):
    img_idx = data.get("img_idx")
    text_idx = data.get("text_idx")
    char_idx = data.get("char_idx")

    if img_idx is None or text_idx is None:
        raise HTTPException(status_code=400, detail="缺少必要参数")
    if img_idx < 0 or img_idx >= len(state.results):
        raise HTTPException(status_code=400, detail="img_idx 无效")

    result = state.results[img_idx]
    associations = result.get("text_character_associations", [])

    associations = [[t, c] for t, c in associations if t != text_idx]

    if char_idx is not None:
        associations.append([text_idx, char_idx])

    result["text_character_associations"] = associations
    return {"success": True}


def _serialize_results(results: list) -> list:
    serialized = []
    for r in results:
        s = {}
        for k, v in r.items():
            if hasattr(v, "tolist"):
                s[k] = v.tolist()
            elif isinstance(v, list):
                s[k] = [x.tolist() if hasattr(x, "tolist") else x for x in v]
            else:
                s[k] = v
        serialized.append(s)
    return serialized
