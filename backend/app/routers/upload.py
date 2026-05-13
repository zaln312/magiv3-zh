import os
import uuid
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from app.config import UPLOAD_DIR
from app.utils.state import state

router = APIRouter()


@router.post("/upload")
async def upload_images(files: list[UploadFile] = File(...)):
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    saved_paths = []
    for file in files:
        ext = os.path.splitext(file.filename or "image.jpg")[1] or ".jpg"
        filename = f"{uuid.uuid4().hex}{ext}"
        filepath = os.path.join(UPLOAD_DIR, filename)
        content = await file.read()
        with open(filepath, "wb") as f:
            f.write(content)
        saved_paths.append(filepath)

    state.img_paths.extend(saved_paths)
    state.reset_pipeline()

    return JSONResponse(
        content={
            "success": True,
            "img_paths": state.img_paths,
            "count": len(state.img_paths),
        }
    )


@router.get("/images")
async def list_images():
    return {"img_paths": state.img_paths, "count": len(state.img_paths)}


@router.post("/images/reorder")
async def reorder_images(data: dict):
    order = data.get("order", [])
    if len(order) != len(state.img_paths):
        return JSONResponse(
            content={"success": False, "message": "order 长度与图片数量不一致"},
            status_code=400,
        )
    state.img_paths = [state.img_paths[i] for i in order]
    state.reset_pipeline()
    return {"success": True, "img_paths": state.img_paths}


@router.post("/images/delete")
async def delete_image(data: dict):
    path = data.get("path")
    if not path or path not in state.img_paths:
        return JSONResponse(
            content={"success": False, "message": "无效的图片路径"},
            status_code=400,
        )
    state.img_paths.remove(path)
    if os.path.exists(path):
        os.remove(path)
    state.reset_pipeline()
    return {"success": True, "img_paths": state.img_paths, "removed": path}


@router.get("/images/serve/{img_idx}")
async def serve_image(img_idx: int):
    if img_idx < 0 or img_idx >= len(state.img_paths):
        raise HTTPException(status_code=404, detail="图片索引无效")
    filepath = state.img_paths[img_idx]
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="图片文件不存在")
    return FileResponse(filepath)


@router.get("/images/serve-by-name/{filename}")
async def serve_image_by_name(filename: str):
    for filepath in state.img_paths:
        if os.path.basename(filepath) == filename:
            if not os.path.exists(filepath):
                raise HTTPException(status_code=404, detail="图片文件不存在")
            return FileResponse(filepath)
    raise HTTPException(status_code=404, detail="图片未找到")
