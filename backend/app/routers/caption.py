import sys
import os

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.utils.state import state

router = APIRouter()


class CaptionRunRequest(BaseModel):
    think: bool = False


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


@router.post("/caption/run")
async def run_caption(req: CaptionRunRequest = CaptionRunRequest()):
    if not state.results:
        raise HTTPException(status_code=400, detail="请先执行 Predict")

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
    from ocr_utils import get_captions

    try:
        state.captions = get_captions(state.img_paths, state.results, think=req.think)
        print(
            f"[DEBUG] caption/run: state.captions 类型={type(state.captions)}, 长度={len(state.captions)}"
        )
        for i, caps in enumerate(state.captions):
            print(f"[DEBUG]   img[{i}]: {len(caps)} 个 panel captions")
            for j, c in enumerate(caps):
                print(f"[DEBUG]     panel[{j}]: {c[:80]}...")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Caption 失败: {str(e)}")

    return {
        "success": True,
        "captions": state.captions,
        "count": len(state.captions),
    }


@router.get("/caption/results")
async def get_caption_results():
    serialized_results = _serialize_results(state.results)

    for i, (res, caps) in enumerate(zip(serialized_results, state.captions)):
        res["caption"] = "\n\n".join(caps) if caps else ""

    if state.panel_scripts:
        for i, (res, scripts) in enumerate(
            zip(serialized_results, state.panel_scripts)
        ):
            flat_scripts = []
            for panel_lines in scripts:
                flat_scripts.extend(panel_lines)
            res["panel_script"] = "\n".join(flat_scripts) if flat_scripts else ""

    print(f"[DEBUG] caption/results: 返回 {len(serialized_results)} 个 results")
    for i, r in enumerate(serialized_results):
        print(f"[DEBUG]   result[{i}] keys: {list(r.keys())}")
        print(f"[DEBUG]   result[{i}].caption 前80字: {r.get('caption', '')[:80]}...")
        print(
            f"[DEBUG]   result[{i}].panel_script 前80字: {r.get('panel_script', '')[:80]}..."
        )

    return {
        "results": serialized_results,
        "count": len(serialized_results),
    }


@router.post("/caption/update_caption")
async def update_caption(data: dict):
    img_idx = data.get("img_idx")
    caption = data.get("caption")

    if img_idx is None or caption is None:
        raise HTTPException(status_code=400, detail="缺少必要参数")
    if img_idx < 0 or img_idx >= len(state.captions):
        raise HTTPException(status_code=400, detail="img_idx 无效")

    state.captions[img_idx] = [caption]
    print(f"[DEBUG] caption/update_caption: img[{img_idx}] 更新为: {caption[:80]}...")
    return {"success": True}


@router.post("/caption/update_panel_script")
async def update_panel_script(data: dict):
    img_idx = data.get("img_idx")
    panel_script = data.get("panel_script")

    if img_idx is None or panel_script is None:
        raise HTTPException(status_code=400, detail="缺少必要参数")
    if img_idx < 0 or img_idx >= len(state.results):
        raise HTTPException(status_code=400, detail="img_idx 无效")

    if not state.panel_scripts:
        state.panel_scripts = [[""] for _ in state.results]
    while len(state.panel_scripts) <= img_idx:
        state.panel_scripts.append([""])

    state.panel_scripts[img_idx] = [panel_script]
    print(
        f"[DEBUG] caption/update_panel_script: img[{img_idx}] 更新为: {panel_script[:80]}..."
    )
    return {"success": True}
