import sys
import os

from fastapi import APIRouter, HTTPException
from app.utils.state import state
from app.services.model_manager import model_manager

router = APIRouter()


@router.post("/grounding/run")
async def run_grounding():
    if not state.captions:
        raise HTTPException(status_code=400, detail="请先执行 Caption")

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

    try:
        from PIL import Image
        from ocr_utils import preprocess_panel_characters, get_grounding

        model_manager.load()

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

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Grounding 失败: {str(e)}")

    return {
        "success": True,
        "grounded_captions": state.grounded_captions,
        "count": len(state.grounded_captions),
    }


@router.get("/grounding/results")
async def get_grounding_results():
    return {
        "grounded_captions": state.grounded_captions,
        "count": len(state.grounded_captions),
    }
