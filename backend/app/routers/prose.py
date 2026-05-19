import sys
import os
import io
import base64

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from PIL import Image
from app.utils.state import state
from app.services.app_config import get_prose_openai_config, get_reference_config

router = APIRouter()


class StoryBackgroundRequest(BaseModel):
    story_background: str = ""


@router.post("/prose/save_story_background")
async def save_story_background_api(req: StoryBackgroundRequest):
    state.story_background = req.story_background
    state.persist_prose()
    return {"success": True, "story_background": state.story_background}


class SaveProsePromptRequest(BaseModel):
    prose_prompt_text: str = ""


@router.post("/prose/save_prose_prompt")
async def save_prose_prompt_api(req: SaveProsePromptRequest):
    state.prose_prompt = req.prose_prompt_text.split("\n") if req.prose_prompt_text else []
    state.persist_prose()
    return {"success": True}


class SaveProseTextRequest(BaseModel):
    prose_text: str = ""


@router.post("/prose/save_prose_text")
async def save_prose_text_api(req: SaveProseTextRequest):
    state.prose = req.prose_text
    state.persist_prose()
    return {"success": True}


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
        state.persist_panel_scripts()
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
            state.grounded_captions,
            state.panel_scripts,
            state.character_name_map,
            story_background=state.story_background,
        )
        state.persist_prose()
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
        state.persist_panel_scripts()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"构建 Panel Scripts 失败: {str(e)}"
        )

    try:
        state.prose_prompt = get_prose_prompt(
            state.grounded_captions,
            state.panel_scripts,
            state.character_name_map,
            story_background=state.story_background,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"构建 Prose Prompt 失败: {str(e)}")

    try:
        state.prose = get_prose(
            state.prose_prompt, prose_config=get_prose_openai_config()
        )

        state.persist_step("prose")
        state.persist_prose()
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
        "story_background": state.story_background,
    }


def _collect_best_crop(global_id: int) -> dict:
    best_area = 0
    best_b64 = None
    best_img_idx = None
    best_box = None

    for img_idx, result in enumerate(state.results):
        gids = result.get("global_character_ids", [])
        boxes = result.get("characters", [])

        for char_idx, gid in enumerate(gids):
            if gid != global_id or char_idx >= len(boxes):
                continue

            box = boxes[char_idx]
            x1, y1, x2, y2 = box
            area = (x2 - x1) * (y2 - y1)

            if area > best_area:
                img_path = state.img_paths[img_idx]
                if not os.path.exists(img_path):
                    continue
                img = Image.open(img_path).convert("RGB")
                w, h = img.size
                x1c = max(0, int(x1))
                y1c = max(0, int(y1))
                x2c = min(w, int(x2))
                y2c = min(h, int(y2))
                if x2c <= x1c or y2c <= y1c:
                    continue
                crop = img.crop((x1c, y1c, x2c, y2c))
                buf = io.BytesIO()
                crop.save(buf, format="PNG")
                best_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                best_area = area
                best_img_idx = img_idx
                best_box = box

    return {
        "global_id": global_id,
        "image_base64": best_b64,
        "img_idx": best_img_idx,
        "box": best_box,
        "crop_area": best_area,
    }


def _collect_all_crops(global_id: int) -> list[dict]:
    crops = []

    for img_idx, result in enumerate(state.results):
        gids = result.get("global_character_ids", [])
        boxes = result.get("characters", [])

        for char_idx, gid in enumerate(gids):
            if gid != global_id or char_idx >= len(boxes):
                continue

            box = boxes[char_idx]
            x1, y1, x2, y2 = box
            area = (x2 - x1) * (y2 - y1)

            img_path = state.img_paths[img_idx]
            if not os.path.exists(img_path):
                continue
            img = Image.open(img_path).convert("RGB")
            w, h = img.size
            x1c = max(0, int(x1))
            y1c = max(0, int(y1))
            x2c = min(w, int(x2))
            y2c = min(h, int(y2))
            if x2c <= x1c or y2c <= y1c:
                continue
            crop = img.crop((x1c, y1c, x2c, y2c))
            buf = io.BytesIO()
            crop.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

            crops.append(
                {
                    "img_idx": img_idx,
                    "char_idx": char_idx,
                    "box": box,
                    "area": area,
                    "image_base64": b64,
                    "crop_key": f"{img_idx}_{char_idx}",
                }
            )

    crops.sort(key=lambda c: c["area"], reverse=True)
    return crops


@router.get("/prose/character_crops/{global_id}")
async def get_character_crops(global_id: int):
    info = _collect_best_crop(global_id)
    if info["image_base64"] is None:
        raise HTTPException(
            status_code=404, detail=f"未找到角色 {global_id} 的 crop 图片"
        )
    return {
        "success": True,
        "global_id": global_id,
        "image_base64": info["image_base64"],
        "img_idx": info["img_idx"],
        "box": info["box"],
        "crop_area": info["crop_area"],
    }


@router.get("/prose/character_all_crops/{global_id}")
async def get_character_all_crops(global_id: int):
    crops = _collect_all_crops(global_id)
    if not crops:
        raise HTTPException(
            status_code=404, detail=f"未找到角色 {global_id} 的 crop 图片"
        )
    return {
        "success": True,
        "global_id": global_id,
        "crops": [{k: v for k, v in c.items() if k != "image_base64"} for c in crops],
        "crop_keys": [c["crop_key"] for c in crops],
    }


@router.get("/prose/character_crop_image/{global_id}/{crop_key}")
async def get_character_crop_image(global_id: int, crop_key: str):
    crops = _collect_all_crops(global_id)
    for c in crops:
        if c["crop_key"] == crop_key:
            return {
                "success": True,
                "global_id": global_id,
                "crop_key": crop_key,
                "image_base64": c["image_base64"],
            }
    raise HTTPException(status_code=404, detail=f"未找到 crop: {crop_key}")


@router.get("/prose/reference_results")
async def get_reference_results():
    result = {}
    for gid, refs in state.character_references.items():
        name = state.character_name_map.get(gid, f"角色{gid}")
        result[str(gid)] = {
            "global_id": gid,
            "name": name,
            "references": refs,
        }
    return {
        "success": True,
        "character_references": result,
        "character_library": [
            {
                "global_id": e["global_id"],
                "name": state.character_name_map.get(e["global_id"], ""),
            }
            for e in state.global_character_library
        ],
    }


def _collect_crops_by_keys(global_id: int, crop_keys: list[str]) -> list[Image.Image]:
    all_crops = _collect_all_crops(global_id)
    crop_map = {c["crop_key"]: c for c in all_crops}
    images = []
    for key in crop_keys:
        if key in crop_map:
            b64 = crop_map[key].get("image_base64")
            if b64:
                images.append(
                    Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
                )
    if not images:
        best = _collect_best_crop(global_id)
        if best["image_base64"]:
            images.append(
                Image.open(io.BytesIO(base64.b64decode(best["image_base64"]))).convert(
                    "RGB"
                )
            )
    return images


def _load_design_images(global_id: int, filenames: list[str]) -> list[Image.Image]:
    from app.routers.character import _design_dir

    d = _design_dir(global_id)
    images = []
    for fname in filenames:
        fpath = os.path.join(d, fname)
        if not os.path.isfile(fpath):
            continue
        ext = os.path.splitext(fname)[1].lower()
        if ext not in (".png", ".jpg", ".jpeg", ".webp", ".gif"):
            continue
        try:
            img = Image.open(fpath).convert("RGB")
            images.append(img)
        except Exception:
            continue
    return images


@router.post("/prose/generate_references/{global_id}")
async def generate_character_references(global_id: int, data: dict = {}):
    if not state.prose:
        raise HTTPException(status_code=400, detail="请先生成 Prose")

    views = data.get("views", ["front", "back", "side"])
    if not isinstance(views, list) or len(views) == 0:
        raise HTTPException(status_code=400, detail="views 必须是非空列表")

    for v in views:
        if v not in ("front", "back", "side"):
            raise HTTPException(
                status_code=400,
                detail=f"无效的视图 '{v}'，支持: front, back, side",
            )

    crop_keys = data.get("crop_keys", [])
    if not isinstance(crop_keys, list):
        crop_keys = []

    design_image_filenames = data.get("design_image_filenames", [])
    if not isinstance(design_image_filenames, list):
        design_image_filenames = []

    num_per_view = data.get("num_per_view", 1)

    ref_config = get_reference_config()

    char_name = state.character_name_map.get(global_id, f"角色{global_id}")
    crop_images = _collect_crops_by_keys(global_id, crop_keys)

    if design_image_filenames:
        design_imgs = _load_design_images(global_id, design_image_filenames)
        crop_images.extend(design_imgs)

    from app.services.reference_generator import ip_adapter_generator

    negative = ref_config.get("negative_prompt") or (
        "blurry, low quality, distorted face, bad anatomy, extra limbs, "
        "missing limbs, deformed hands, watermark, text, signature"
    )

    try:
        ip_adapter_generator.load()

        all_results = []
        for view in views:
            view_prompt = ref_config.get("prompt_template", "").format(
                view=view,
                character_name=char_name,
            )
            prompt = (
                view_prompt
                or f"{char_name}, character reference sheet, {view} view, full body standing pose, clean white background, anime manga style, detailed character design"
            )

            gen_results = ip_adapter_generator.generate_and_encode(
                crop_images=crop_images,
                prompt=prompt,
                negative_prompt=negative,
                num_images=num_per_view,
            )

            for result in gen_results:
                ref_entry = {
                    "view": view,
                    "prompt": prompt,
                    "image_base64": result["image_base64"],
                }
                if global_id not in state.character_references:
                    state.character_references[global_id] = []
                state.character_references[global_id].append(ref_entry)
                all_results.append(ref_entry)

        state.persist_character_references()
        ip_adapter_generator.unload()

    except Exception as e:
        try:
            ip_adapter_generator.unload()
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=f"参考图批量生成失败: {str(e)}")

    return {
        "success": True,
        "global_id": global_id,
        "results": all_results,
        "count": len(all_results),
    }
