from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.utils.state import state
from app.utils.serialization import serialize_results
from app.services.app_config import get_caption_openai_config

import logging

logger = logging.getLogger(__name__)

router = APIRouter()


class CaptionRunRequest(BaseModel):
    think: bool = False
    style_prompt: str | None = None


@router.post("/caption/run")
async def run_caption(req: CaptionRunRequest = CaptionRunRequest()):
    if not state.results:
        raise HTTPException(status_code=400, detail="请先执行 Predict")

    from app.utils.ocr_utils import get_captions

    caption_cfg = get_caption_openai_config()

    try:
        state.captions = get_captions(
            state.img_paths,
            state.results,
            think=req.think,
            style_prompt=req.style_prompt,
            caption_config=caption_cfg,
        )
        logger.debug(
            f"caption/run: state.captions 类型={type(state.captions)}, 长度={len(state.captions)}"
        )
        for i, caps in enumerate(state.captions):
            logger.debug(f"  img[{i}]: {len(caps)} 个 panel captions")
            for j, c in enumerate(caps):
                logger.debug(f"    panel[{j}]: {c[:80]}...")

        state.persist_step("caption")
        state.persist_captions()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Caption 失败: {str(e)}")

    return {
        "success": True,
        "captions": state.captions,
        "count": len(state.captions),
    }


@router.get("/caption/results")
async def get_caption_results():
    serialized_results = serialize_results(state.results)

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

    logger.debug(f"caption/results: 返回 {len(serialized_results)} 个 results")
    for i, r in enumerate(serialized_results):
        logger.debug(f"  result[{i}] keys: {list(r.keys())}")
        logger.debug(f"  result[{i}].caption 前80字: {r.get('caption', '')[:80]}...")
        logger.debug(
            f"  result[{i}].panel_script 前80字: {r.get('panel_script', '')[:80]}..."
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
    state.persist_captions()
    logger.debug(f"caption/update_caption: img[{img_idx}] 更新为: {caption[:80]}...")
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
    state.persist_panel_scripts()
    logger.debug(
        f"caption/update_panel_script: img[{img_idx}] 更新为: {panel_script[:80]}..."
    )
    return {"success": True}
