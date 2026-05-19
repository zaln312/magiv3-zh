"""
Fake Server: Prose 生成模型 (Text → Text)
模拟 OpenAI-compatible Chat API

启动方式: python fake_prose_server.py
默认端口: 8101
OpenAI client: base_url="http://localhost:8101/v1"
"""

import json
import time
import sys
from datetime import datetime

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn

app = FastAPI(title="Fake Prose Server", version="1.0.0")


def fmt_json(obj, max_str_len: int = 300) -> str:
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


def generate_fake_prose(user_text: str) -> str:
    preview = user_text[:200].replace("\n", " ")
    return (
        f"[Fake Prose] 这是一个基于以下面板描述生成的中文故事叙述。\n\n"
        f"在一个温暖的午后，两位角色相遇在城市的街角。左侧的年轻女子名为小樱，"
        f"她穿着白色的衬衫，长发在微风中轻轻飘动。她向右侧的男子微笑着，眼中带着一丝期待。"
        f"右侧的男子名为阿杰，他穿着深色的夹克，神情专注地听着小樱说话。"
        f"他们身后的街道在夕阳下泛着金色的光芒，路灯刚刚亮起，营造出一种宁静而浪漫的氛围。"
        f"\n\n随着对话的展开，两人的关系逐渐变得微妙。"
        f"小樱提到了一个许久以前的约定，阿杰陷入了沉思。"
        f"画面在这温馨与感伤之间流转，仿佛时间在这一刻凝固。"
        f"\n\n[输入摘要] {preview}..."
    )


@app.api_route("/v1/chat/completions", methods=["POST"])
async def chat_completions(request: Request):
    body = await request.json()

    print("\n" + "=" * 72)
    print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] "
          f"=== Prose Server: 收到请求 ===")
    print("-" * 72)
    print("[请求体]")
    print(fmt_json(body))
    print("-" * 72)

    messages = body.get("messages", [])
    model = body.get("model", "unknown")
    temperature = body.get("temperature", "N/A")
    max_tokens = body.get("max_tokens", "N/A")
    top_p = body.get("top_p", "N/A")
    presence_penalty = body.get("presence_penalty", "N/A")
    extra_body = body.get("extra_body", {})

    print(f"[模型] {model}")
    print(f"[temperature] {temperature}")
    print(f"[max_tokens] {max_tokens}")
    print(f"[top_p] {top_p}")
    print(f"[presence_penalty] {presence_penalty}")
    if extra_body:
        print(f"[extra_body] {json.dumps(extra_body, ensure_ascii=False)}")

    user_text = ""
    for msg in messages:
        role = msg.get("role", "?")
        content = msg.get("content", "")
        print(f"[消息角色] {role}")
        if isinstance(content, str):
            user_text = content
            preview = content[:200] + ("..." if len(content) > 200 else "")
            print(f"[文本长度] {len(content)} 字符")
            print(f"[文本预览] {preview}")
        elif isinstance(content, list):
            for block in content:
                if block.get("type") == "text":
                    t = block.get("text", "")
                    user_text += t
                    print(f"[文本块] {len(t)} 字符")

    fake_prose = generate_fake_prose(user_text)

    response = {
        "id": f"chatcmpl-fake-prose-{int(time.time()*1000)}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": fake_prose,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": len(user_text) // 4,
            "completion_tokens": len(fake_prose) // 4,
            "total_tokens": (len(user_text) + len(fake_prose)) // 4,
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
        "data": [{"id": "fake-prose-model", "object": "model"}],
    })


@app.get("/health")
async def health():
    return {"status": "ok", "server": "Fake Prose Server"}


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8101
    print(f"Fake Prose Server 启动在 http://localhost:{port}")
    print(f"OpenAI base_url: http://localhost:{port}/v1")
    uvicorn.run(app, host="0.0.0.0", port=port)