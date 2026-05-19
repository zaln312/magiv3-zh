"""
Fake Server: Caption 生成模型 (Image+Text → Text)
模拟 OpenAI-compatible 多模态 API

启动方式: python fake_caption_server.py
默认端口: 8100
OpenAI client: base_url="http://localhost:8100/v1"
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
from PIL import Image

app = FastAPI(title="Fake Caption Server", version="1.0.0")


def fmt_json(obj, max_str_len: int = 200) -> str:
    """格式化 JSON 并截断过长的字符串"""
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


def extract_image_info(content_block: dict) -> dict:
    """从 OpenAI content 中提取图片信息"""
    info = {"type": "image"}
    image_url = content_block.get("image_url", {})
    url = image_url.get("url", "")
    if url.startswith("data:"):
        parts = url.split(";base64,")
        info["mime"] = parts[0].replace("data:", "") if len(parts) > 1 else "unknown"
        data_part = parts[1] if len(parts) > 1 else url
        info["base64_length"] = len(data_part)
        try:
            img_data = base64.b64decode(data_part)
            img = Image.open(io.BytesIO(img_data))
            info["width"] = img.width
            info["height"] = img.height
            info["format"] = img.format
        except Exception:
            info["decode_status"] = "failed"
    else:
        info["url"] = url
    return info


def generate_fake_caption(prompt: str, img_info: dict) -> str:
    """生成假 caption 文本"""
    w = img_info.get("width", "?")
    h = img_info.get("height", "?")
    fmt = img_info.get("format", "?")

    return (
        f"[Fake Caption] This is a {w}x{h} {fmt} image. "
        "The panel depicts a scene with two characters in conversation. "
        "On the left, a young woman with long dark hair and a white blouse "
        "gestures toward the right while speaking. On the right, a tall man "
        "with short brown hair wearing a dark jacket listens attentively. "
        "The background shows a quiet city street at dusk with soft streetlamps "
        "casting warm light. The atmosphere is calm and contemplative."
    )


@app.api_route("/v1/chat/completions", methods=["POST"])
async def chat_completions(request: Request):
    body = await request.json()

    print("\n" + "=" * 72)
    print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] "
          f"=== Caption Server: 收到请求 ===")
    print("-" * 72)
    print("[请求体]")
    print(fmt_json(body))
    print("-" * 72)

    messages = body.get("messages", [])
    model = body.get("model", "unknown")
    temperature = body.get("temperature", "N/A")
    max_tokens = body.get("max_tokens", "N/A")
    top_p = body.get("top_p", "N/A")
    extra_body = body.get("extra_body", {})

    print(f"[模型] {model}")
    print(f"[temperature] {temperature}")
    print(f"[max_tokens] {max_tokens}")
    print(f"[top_p] {top_p}")
    if extra_body:
        print(f"[extra_body] {json.dumps(extra_body, ensure_ascii=False)}")

    img_count = 0
    text_prompts = []
    for msg in messages:
        role = msg.get("role", "?")
        content = msg.get("content", "")
        if isinstance(content, list):
            for block in content:
                t = block.get("type", "")
                if t == "image_url":
                    img_count += 1
                    img_info = extract_image_info(block)
                    print(f"[图片 #{img_count}] {img_info.get('width')}x{img_info.get('height')} "
                          f"{img_info.get('format', '?')}")
                elif t == "text":
                    text_prompts.append(block.get("text", ""))
        elif isinstance(content, str):
            text_prompts.append(content)

    print(f"[消息角色] {', '.join(set(m.get('role','?') for m in messages))}")
    print(f"[图片数量] {img_count}")
    print(f"[文本段数] {len(text_prompts)}")

    # 构造假响应
    prompt_text = "\n".join(text_prompts) if text_prompts else "no text prompt"
    fake_img_info = {"width": 800, "height": 600, "format": "PNG"}
    fake_text = generate_fake_caption(prompt_text, fake_img_info)

    response = {
        "id": f"chatcmpl-fake-caption-{int(time.time()*1000)}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": fake_text,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 120,
            "completion_tokens": 85,
            "total_tokens": 205,
        },
    }

    print("-" * 72)
    print("[响应体]")
    print(fmt_json(response))
    print("=" * 72 + "\n")

    return JSONResponse(content=response)


@app.get("/v1/models")
async def list_models():
    return JSONResponse({
        "object": "list",
        "data": [{"id": "fake-caption-model", "object": "model"}],
    })


@app.get("/health")
async def health():
    return {"status": "ok", "server": "Fake Caption Server"}


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8100
    print(f"Fake Caption Server 启动在 http://localhost:{port}")
    print(f"OpenAI base_url: http://localhost:{port}/v1")
    uvicorn.run(app, host="0.0.0.0", port=port)