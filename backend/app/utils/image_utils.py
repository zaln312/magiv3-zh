import os
import io
import base64

from PIL import Image
from app.config import UPLOAD_DIR
from app.utils.state import state


def design_dir(global_id: int) -> str:
    d = os.path.join(
        UPLOAD_DIR, "design", state.project_id or "default", str(global_id)
    )
    os.makedirs(d, exist_ok=True)
    return d


def collect_best_crop(global_id: int) -> dict:
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


def collect_all_crops(global_id: int) -> list[dict]:
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


def collect_crops_by_keys(
    global_id: int, crop_keys: list[str]
) -> list[Image.Image]:
    all_crops = collect_all_crops(global_id)
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
        best = collect_best_crop(global_id)
        if best["image_base64"]:
            images.append(
                Image.open(
                    io.BytesIO(base64.b64decode(best["image_base64"]))
                ).convert("RGB")
            )
    return images


def load_design_images(
    global_id: int, filenames: list[str]
) -> list[Image.Image]:
    d = design_dir(global_id)
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