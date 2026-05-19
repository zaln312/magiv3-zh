"""
Fake Server: 人物参考图生成模型 (Text + Image → Image)
模拟图片生成 API（兼容 OpenAI /v1/images/generations 格式）

启动方式: python fake_reference_server.py
默认端口: 8102
"""

import json
import time
import base64
import io
import os
import sys
import random
import mimetypes
from datetime import datetime

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}

app = FastAPI(title="Fake Reference Image Server", version="1.0.0")


def fmt_json(obj, max_str_len: int = 200) -> str:
    s = json.dumps(obj, indent=2, ensure_ascii=False, default=str)
    lines = s.split("\n")
    result = []
    for line in lines:
        if len(line) > max_str_len + 20:
            stripped = line.strip()
            if stripped.startswith('"') and len(stripped) > max_str_len:
                indent = len(line) - len(line.lstrip())
                short = stripped[:max_str_len] + '..."'
                line = " " * indent + short
        result.append(line)
    return "\n".join(result)


def _list_images():
    if not os.path.isdir(OUTPUT_DIR):
        return []
    files = []
    for f in os.listdir(OUTPUT_DIR):
        ext = os.path.splitext(f)[1].lower()
        if ext in SUPPORTED_EXTS:
            files.append(os.path.join(OUTPUT_DIR, f))
    return files


def _read_as_base64(filepath: str) -> str:
    with open(filepath, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def extract_image_info(image_data_b64: str) -> dict:
    info = {"base64_length": len(image_data_b64)}
    try:
        img_data = base64.b64decode(image_data_b64)
        img = Image.open(io.BytesIO(img_data))
        info["width"] = img.width
        info["height"] = img.height
        info["format"] = img.format
    except Exception:
        info["decode_status"] = "failed"
    return info





@app.api_route("/v1/images/generations", methods=["POST"])
async def image_generations(request: Request):
    body = await request.json()

    print("\n" + "=" * 72)
    print(
        f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] "
        f"=== Reference Server: 收到请求 ==="
    )
    print("-" * 72)
    print("[请求体]")
    print(fmt_json(body))
    print("-" * 72)

    model = body.get("model", "unknown")
    prompt = body.get("prompt", "")
    negative_prompt = body.get("negative_prompt", "")
    n = body.get("n", 1)
    size = body.get("size", "512x768")
    response_format = body.get("response_format", "b64_json")

    # 解析尺寸
    if "x" in str(size):
        w_str, h_str = str(size).split("x")
    else:
        w_str, h_str = "512", "768"
    width = int(w_str)
    height = int(h_str)

    print(f"[模型] {model}")
    print(f"[prompt] ({len(prompt)} 字符) {prompt[:150]}...")
    if negative_prompt:
        print(
            f"[negative_prompt] ({len(negative_prompt)} 字符) {negative_prompt[:100]}..."
        )
    print(f"[数量] {n}")
    print(f"[尺寸] {width}x{height}")

    # 提取参考图信息
    if "ref_images" in body:
        ref_images = body.get("ref_images", [])
        print(f"[参考图数量] {len(ref_images)}")
    if "image" in body:
        img_data = body.get("image", "")
        if img_data:
            info = extract_image_info(img_data)
            print(
                f"[输入图片] {info.get('width')}x{info.get('height')} "
                f"{info.get('format', '?')} ({info.get('base64_length', 0)} bytes)"
            )

    available = _list_images()
    if available:
        chosen = random.choices(available, k=min(n, len(available)))
        images_b64 = [_read_as_base64(p) for p in chosen]
        print(f"[随机选取] 从 {len(available)} 张图片中选了 {len(chosen)} 张")
    else:
        images_b64 = []
        print("[随机选取] output 目录无可用图片")

    response = {
        "created": int(time.time()),
        "data": [
            {
                "index": i,
                "b64_json": b64,
                "revised_prompt": prompt,
            }
            for i, b64 in enumerate(images_b64)
        ],
    }

    print("-" * 72)
    print("[响应体]")
    print(f"  created: {response['created']}")
    for i, d in enumerate(response["data"]):
        print(
            f"  data[{i}]: b64_json 长度={len(d['b64_json'])}, "
            f"revised_prompt 前80字={d['revised_prompt'][:80]}..."
        )
    print("=" * 72 + "\n")

    return JSONResponse(content=response)


@app.api_route("/v1/chat/completions", methods=["POST"])
async def chat_completions(request: Request):
    """兼容 OpenAI chat completions 格式的图片生成（用于某些多模态模型）"""
    body = await request.json()

    print("\n" + "=" * 72)
    print(
        f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] "
        f"=== Reference Server (chat模式): 收到请求 ==="
    )
    print("-" * 72)
    print("[请求体]")
    print(fmt_json(body))
    print("-" * 72)

    messages = body.get("messages", [])
    model = body.get("model", "unknown")
    extra_body = body.get("extra_body", {})

    prompt = ""
    img_count = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, list):
            for block in content:
                if block.get("type") == "image_url":
                    img_count += 1
                    url = block.get("image_url", {}).get("url", "")
                    if url.startswith("data:"):
                        parts = url.split(";base64,")
                        b64 = parts[1] if len(parts) > 1 else ""
                        info = extract_image_info(b64)
                        print(
                            f"[图片 #{img_count}] {info.get('width')}x{info.get('height')}"
                        )
                elif block.get("type") == "text":
                    prompt += block.get("text", "")
        elif isinstance(content, str):
            prompt += content

    width = extra_body.get("width", 512)
    height = extra_body.get("height", 768)
    n = extra_body.get("num_images_per_view", 1)

    print(f"[模型] {model}")
    print(f"[prompt] ({len(prompt)} 字符) {prompt[:150]}...")
    print(f"[图片数量] {img_count}")
    print(f"[尺寸] {width}x{height}")

    available = _list_images()
    if available:
        chosen = random.choices(available, k=min(n, len(available)))
        images_b64 = [_read_as_base64(p) for p in chosen]
        print(f"[随机选取] 从 {len(available)} 张图片中选了 {len(chosen)} 张")
    else:
        images_b64 = []
        print("[随机选取] output 目录无可用图片")

    response = {
        "id": f"img-fake-ref-{int(time.time()*1000)}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": json.dumps(images_b64),
                    "images": images_b64,
                },
                "finish_reason": "stop",
            }
        ],
    }

    print("-" * 72)
    print("[响应体]")
    for i, img in enumerate(images_b64):
        print(f"  images[{i}]: base64 长度={len(img)}")
    print("=" * 72 + "\n")

    return JSONResponse(content=response)


@app.get("/health")
async def health():
    return {"status": "ok", "server": "Fake Reference Image Server"}


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8102
    print(f"Fake Reference Image Server 启动在 http://localhost:{port}")
    print(f"图片生成端点: http://localhost:{port}/v1/images/generations")
    print(f"聊天兼容端点: http://localhost:{port}/v1/chat/completions")
    uvicorn.run(app, host="0.0.0.0", port=port)
