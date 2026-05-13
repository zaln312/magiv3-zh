import sys
import os

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.utils.state import state

router = APIRouter()


class CaptionRunRequest(BaseModel):
    think: bool = False


@router.post("/caption/run")
async def run_caption(req: CaptionRunRequest = CaptionRunRequest()):
    if not state.results:
        raise HTTPException(status_code=400, detail="请先执行 Predict")

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
    from ocr_utils import get_captions

    try:
        state.captions = get_captions(state.img_paths, state.results, think=req.think)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Caption 失败: {str(e)}")

    return {
        "success": True,
        "captions": state.captions,
        "count": len(state.captions),
    }


@router.get("/caption/results")
async def get_caption_results():
    return {
        "captions": state.captions,
        "count": len(state.captions),
    }
