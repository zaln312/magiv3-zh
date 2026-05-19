from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.utils.state import state
from app.services.model_manager import model_manager
from app.services.app_config import get_ocr_api_url, get_ocr_format_code

router = APIRouter()


class OcrRunRequest(BaseModel):
    only_white_bg: bool = False
    zh_texts: bool = True


@router.post("/ocr/run")
async def run_ocr(req: OcrRunRequest = OcrRunRequest()):
    if not state.img_paths:
        raise HTTPException(status_code=400, detail="请先上传图片")

    from app.utils.ocr_utils import get_ocr_results

    api_url = get_ocr_api_url()
    format_code = get_ocr_format_code()

    try:
        state.unordered_ocr_res = get_ocr_results(
            state.img_paths,
            only_white_bg=req.only_white_bg,
            zh_texts=req.zh_texts,
            api_url=api_url,
            format_code=format_code,
        )
        state.reset_from_ocr()

        model_manager.maybe_load()
        from app.utils.ocr_utils import prepare_ordered_ocr_and_detect

        _, _, _, _, ordered_ocr_results = prepare_ordered_ocr_and_detect(
            model_manager.model,
            model_manager.processor,
            state.img_paths,
            state.unordered_ocr_res,
        )
        state.unordered_ocr_res = ordered_ocr_results

        state.persist_step("ocr")
        state.persist_ocr_results()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR 识别失败: {str(e)}")
    finally:
        model_manager.maybe_unload()

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
    state.persist_ocr_text(img_idx)
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
    state.persist_ocr_boxes(img_idx)
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
    state.persist_ocr_results()
    return {"success": True}


@router.post("/ocr/reorder")
async def reorder_ocr_entries(data: dict):
    img_idx = data.get("img_idx")
    order = data.get("order")

    if img_idx is None or order is None:
        raise HTTPException(status_code=400, detail="缺少必要参数")

    if img_idx < 0 or img_idx >= len(state.unordered_ocr_res):
        raise HTTPException(status_code=400, detail="img_idx 无效")

    ocr_res = state.unordered_ocr_res[img_idx]
    boxes = ocr_res.get("boxes", [])
    texts = ocr_res.get("texts", [])

    if len(order) != len(boxes) or len(order) != len(texts):
        raise HTTPException(status_code=400, detail="order 长度不匹配")

    new_boxes = [boxes[i] for i in order]
    new_texts = [texts[i] for i in order]
    ocr_res["boxes"] = new_boxes
    ocr_res["texts"] = new_texts
    state.persist_ocr_results()
    return {"success": True}


@router.post("/ocr/add_box")
async def add_ocr_box(data: dict):
    img_idx = data.get("img_idx")
    box = data.get("box")
    text = data.get("text", "")

    if img_idx is None or box is None:
        raise HTTPException(status_code=400, detail="缺少必要参数")

    if img_idx < 0 or img_idx >= len(state.unordered_ocr_res):
        raise HTTPException(status_code=400, detail="img_idx 无效")

    if len(box) != 4:
        raise HTTPException(status_code=400, detail="box 格式应为 [x1, y1, x2, y2]")

    ocr_res = state.unordered_ocr_res[img_idx]
    ocr_res["boxes"].append(box)
    ocr_res["texts"].append(text)
    state.persist_ocr_results()
    return {"success": True, "idx": len(ocr_res["boxes"]) - 1}
