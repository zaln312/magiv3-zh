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
import sys
from datetime import datetime

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn
from PIL import Image, ImageDraw, ImageFont

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


def _wrap_text(text: str, max_chars: int) -> list[str]:
    lines = []
    while len(text) > max_chars:
        split_at = text.rfind(" ", 0, max_chars)
        if split_at == -1:
            split_at = max_chars
        lines.append(text[:split_at])
        text = text[split_at:].lstrip()
    if text:
        lines.append(text)
    return lines


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


def generate_fake_reference_image(prompt: str, width: int, height: int) -> str:
    """生成一张假的参考图（纯色+人物轮廓+文字标注，返回 base64 PNG）"""
    img = Image.new("RGB", (width, height), color=(220, 230, 240))
    draw = ImageDraw.Draw(img)

    # 绘制简单的人物轮廓示意
    cx, cy = width // 2, height // 2
    head_r = min(width, height) // 8

    # 头部
    draw.ellipse(
        [
            cx - head_r,
            cy - height // 3 - head_r,
            cx + head_r,
            cy - height // 3 + head_r,
        ],
        fill=(255, 220, 200),
        outline=(100, 80, 60),
        width=2,
    )
    # 身体
    body_top = cy - height // 3 + head_r
    body_bottom = cy + height // 4
    draw.rectangle(
        [cx - head_r, body_top, cx + head_r, body_bottom],
        fill=(100, 150, 200),
        outline=(60, 100, 140),
        width=2,
    )
    # 腿
    draw.rectangle(
        [cx - head_r // 2, body_bottom, cx, body_bottom + height // 5],
        fill=(60, 60, 80),
        outline=(40, 40, 60),
        width=2,
    )
    draw.rectangle(
        [cx, body_bottom, cx + head_r // 2, body_bottom + height // 5],
        fill=(60, 60, 80),
        outline=(40, 40, 60),
        width=2,
    )

    # 文字标注
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    label = prompt[:80] + "..." if len(prompt) > 80 else prompt
    draw.text((10, 10), "[Fake Reference Image]", fill=(0, 0, 0), font=font)
    for i, line in enumerate(_wrap_text(label, 50)):
        draw.text((10, 30 + i * 16), line, fill=(50, 50, 50), font=font)
    draw.text(
        (10, height - 20), f"{width}x{height} PNG", fill=(100, 100, 100), font=font
    )

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


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

    # 生成假图片
    images_b64 = []
    for i in range(n):
        b64 = generate_fake_reference_image(prompt, width, height)
        images_b64.append(b64)

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

    images_b64 = []
    for i in range(n):
        b64 = generate_fake_reference_image(prompt, width, height)
        images_b64.append(b64)

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
