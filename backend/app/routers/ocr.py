import sys
import os

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.utils.state import state

router = APIRouter()


class OcrRunRequest(BaseModel):
    only_white_bg: bool = False
    zh_texts: bool = True


@router.post("/ocr/run")
async def run_ocr(req: OcrRunRequest = OcrRunRequest()):
    if not state.img_paths:
        raise HTTPException(status_code=400, detail="请先上传图片")

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
    from ocr_utils import get_ocr_results

    try:
        state.unordered_ocr_res = get_ocr_results(
            state.img_paths,
            only_white_bg=req.only_white_bg,
            zh_texts=req.zh_texts,
        )
        state.reset_from_ocr()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR 识别失败: {str(e)}")

    return {
        "success": True,
        "ocr_results": state.unordered_ocr_res,
        "count": len(state.unordered_ocr_res),
    }


@router.get("/ocr/results")
async def get_ocr_results_api():
    return {
        "ocr_results": state.unordered_ocr_res,
        "count": len(state.unordered_ocr_res),
    }


@router.post("/ocr/update_text")
async def update_ocr_text(data: dict):
    img_idx = data.get("img_idx")
    text_idx = data.get("text_idx")
    new_text = data.get("text")

    if img_idx is None or text_idx is None or new_text is None:
        raise HTTPException(status_code=400, detail="缺少必要参数")

    if img_idx < 0 or img_idx >= len(state.unordered_ocr_res):
        raise HTTPException(status_code=400, detail="img_idx 无效")

    ocr_res = state.unordered_ocr_res[img_idx]
    if text_idx < 0 or text_idx >= len(ocr_res.get("texts", [])):
        raise HTTPException(status_code=400, detail="text_idx 无效")

    ocr_res["texts"][text_idx] = new_text
    return {"success": True}


@router.post("/ocr/update_box")
async def update_ocr_box(data: dict):
    img_idx = data.get("img_idx")
    box_idx = data.get("box_idx")
    box = data.get("box")

    if img_idx is None or box_idx is None or box is None:
        raise HTTPException(status_code=400, detail="缺少必要参数")

    if img_idx < 0 or img_idx >= len(state.unordered_ocr_res):
        raise HTTPException(status_code=400, detail="img_idx 无效")

    ocr_res = state.unordered_ocr_res[img_idx]
    if box_idx < 0 or box_idx >= len(ocr_res.get("boxes", [])):
        raise HTTPException(status_code=400, detail="box_idx 无效")

    if len(box) != 4:
        raise HTTPException(status_code=400, detail="box 格式应为 [x1, y1, x2, y2]")

    ocr_res["boxes"][box_idx] = box
    return {"success": True}


@router.post("/ocr/delete_box")
async def delete_ocr_box(data: dict):
    img_idx = data.get("img_idx")
    box_idx = data.get("box_idx")

    if img_idx is None or box_idx is None:
        raise HTTPException(status_code=400, detail="缺少必要参数")

    if img_idx < 0 or img_idx >= len(state.unordered_ocr_res):
        raise HTTPException(status_code=400, detail="img_idx 无效")

    ocr_res = state.unordered_ocr_res[img_idx]
    if box_idx < 0 or box_idx >= len(ocr_res.get("boxes", [])):
        raise HTTPException(status_code=400, detail="box_idx 无效")

    ocr_res["boxes"].pop(box_idx)
    ocr_res["texts"].pop(box_idx)
    return {"success": True}
