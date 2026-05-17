import sys
import os

from fastapi import APIRouter, HTTPException
from app.utils.state import state

router = APIRouter()


@router.post("/prose/build_scripts")
async def build_panel_scripts_api():
    if not state.results:
        raise HTTPException(status_code=400, detail="请先执行 Predict")

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
    from ocr_utils import build_panel_scripts

    try:
        state.panel_scripts = []
        for result in state.results:
            state.panel_scripts.append(
                build_panel_scripts(
                    result,
                    essential_only=False,
                    include_narrator=True,
                    label="narrator",
                    label_char_name="character",
                    character_name_map=state.character_name_map,
                )
            )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"构建 Panel Scripts 失败: {str(e)}"
        )

    return {
        "success": True,
        "panel_scripts": state.panel_scripts,
        "count": len(state.panel_scripts),
    }


@router.post("/prose/build_prompt")
async def build_prose_prompt_api():
    if not state.grounded_captions:
        raise HTTPException(status_code=400, detail="请先执行 Grounding")
    if not state.panel_scripts:
        raise HTTPException(status_code=400, detail="请先构建 Panel Scripts")

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
    from ocr_utils import get_prose_prompt

    try:
        state.prose_prompt = get_prose_prompt(
            state.grounded_captions, state.panel_scripts
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"构建 Prose Prompt 失败: {str(e)}")

    return {
        "success": True,
        "prose_prompt": state.prose_prompt,
    }


@router.post("/prose/run")
async def run_prose():
    if not state.results:
        raise HTTPException(status_code=400, detail="请先执行 Predict")
    if not state.grounded_captions:
        raise HTTPException(status_code=400, detail="请先执行 Grounding")

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
    from ocr_utils import build_panel_scripts, get_prose_prompt, get_prose

    try:
        state.panel_scripts = []
        for img_idx, result in enumerate(state.results):
            try:
                state.panel_scripts.append(
                    build_panel_scripts(
                        result,
                        essential_only=False,
                        include_narrator=True,
                        label="narrator",
                        label_char_name="character",
                        character_name_map=state.character_name_map,
                    )
                )
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"构建 Panel Scripts 失败 (img {img_idx}): {str(e)}",
                )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"构建 Panel Scripts 失败: {str(e)}"
        )

    try:
        state.prose_prompt = get_prose_prompt(
            state.grounded_captions, state.panel_scripts
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"构建 Prose Prompt 失败: {str(e)}"
        )

    try:
        state.prose = get_prose(state.prose_prompt)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prose LLM 调用失败: {str(e)}")

    return {
        "success": True,
        "prose": state.prose,
    }


@router.get("/prose/results")
async def get_prose_results():
    return {
        "panel_scripts": state.panel_scripts,
        "prose_prompt": state.prose_prompt,
        "prose": state.prose,
    }
