import sys
import os

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.utils.state import state
from app.services.model_manager import model_manager
from app.services.app_config import get_caption_openai_config

router = APIRouter()


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


class GroundingRunRequest(BaseModel):
    style_prompt: str | None = None


@router.post("/grounding/run")
async def run_grounding(req: GroundingRunRequest = GroundingRunRequest()):
    if not state.results:
        raise HTTPException(status_code=400, detail="请先执行 Predict")

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

    try:
        from PIL import Image
        from ocr_utils import get_captions, preprocess_panel_characters, get_grounding

        if not state.captions or req.style_prompt:
            caption_cfg = get_caption_openai_config()
            state.captions = get_captions(
                state.img_paths,
                state.results,
                think=False,
                style_prompt=req.style_prompt,
                caption_config=caption_cfg,
            )
            state.persist_captions()
            print(
                f"[DEBUG] grounding/run: 自动执行 Caption 完成, {len(state.captions)} 张图片"
            )

        model_manager.maybe_load()

        grounded_results = []
        for i in range(len(state.img_paths)):
            grounded_result = []
            img = Image.open(state.img_paths[i])
            panel_imgs = [img.crop(panel) for panel in state.results[i]["panels"]]
            for panel_img, cap in zip(panel_imgs, state.captions[i]):
                res = model_manager.model.predict_character_grounding(
                    [panel_img],
                    [cap],
                    model_manager.processor,
                    strict=False,
                )
                grounded_result.extend(res)
            grounded_results.append(grounded_result)
        state.grounded_results = grounded_results

        grounded_captions = []
        for img_idx in range(len(state.img_paths)):
            panel_characters_list = preprocess_panel_characters(state.results[img_idx])
            grounded_result = grounded_results[img_idx]
            grounded_caption = []
            for panel_idx, res in enumerate(grounded_result):
                panel_characters = panel_characters_list[panel_idx]
                cap = get_grounding(res, panel_characters, None)
                grounded_caption.append(cap)
            grounded_captions.append(grounded_caption)
        state.grounded_captions = grounded_captions

        state.persist_step("grounding")
        state.persist_grounded_captions()

        print(f"[DEBUG] grounding/run: 完成, {len(state.grounded_captions)} 张图片")
        for i, caps in enumerate(state.grounded_captions):
            print(f"[DEBUG]   img[{i}]: {len(caps)} 个 panel grounded_captions")
            for j, c in enumerate(caps):
                print(f"[DEBUG]     panel[{j}]: {c[:80]}...")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Grounding 失败: {str(e)}")
    finally:
        model_manager.maybe_unload()

    return {
        "success": True,
        "grounded_captions": state.grounded_captions,
        "count": len(state.grounded_captions),
    }


@router.get("/grounding/results")
async def get_grounding_results():
    serialized_results = _serialize_results(state.results)

    for i, (res, caps) in enumerate(zip(serialized_results, state.grounded_captions)):
        res["grounded_caption"] = "\n\n".join(caps) if caps else ""
        res["grounded_captions_per_panel"] = caps

    print(f"[DEBUG] grounding/results: 返回 {len(serialized_results)} 个 results")
    for i, r in enumerate(serialized_results):
        print(f"[DEBUG]   result[{i}] keys: {list(r.keys())}")
        print(
            f"[DEBUG]   result[{i}].grounded_caption 前80字: {r.get('grounded_caption', '')[:80]}..."
        )

    return {
        "results": serialized_results,
        "count": len(serialized_results),
        "style_prompt": state.style_prompt,
    }


class SaveStylePromptRequest(BaseModel):
    style_prompt: str = ""


@router.post("/grounding/save_style_prompt")
async def save_style_prompt_api(req: SaveStylePromptRequest):
    state.style_prompt = req.style_prompt
    state.persist_prose()
    return {"success": True, "style_prompt": state.style_prompt}


class UpdateGroundedCaptionRequest(BaseModel):
    img_idx: int
    panel_idx: int
    grounded_caption: str


@router.post("/grounding/update_caption")
async def update_grounded_caption(data: UpdateGroundedCaptionRequest):
    img_idx = data.img_idx
    panel_idx = data.panel_idx
    caption = data.grounded_caption

    if img_idx < 0 or img_idx >= len(state.grounded_captions):
        raise HTTPException(status_code=400, detail="img_idx 无效")
    if panel_idx < 0 or panel_idx >= len(state.grounded_captions[img_idx]):
        raise HTTPException(status_code=400, detail="panel_idx 无效")

    state.grounded_captions[img_idx][panel_idx] = caption
    state.persist_grounded_caption_single(img_idx, panel_idx)
    return {"success": True}
