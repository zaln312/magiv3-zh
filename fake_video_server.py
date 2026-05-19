"""
Fake Video Server: 模拟 Vidu Q3 视频生成 API（异步任务模式）

支持两种接口：
1. Vidu 风格异步模式（新）:
   POST /ent/v2/creations  → 提交任务，返回 task_id
   GET  /ent/v2/tasks/{id}/creations → 轮询状态

2. OpenAI 兼容模式（保留）:
   POST /v1/chat/completions → 同步生成

启动方式: python fake_video_server.py [port]
默认端口: 8103
"""

import json
import time
import base64
import io
import uuid
import sys
import threading
from datetime import datetime

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn
from PIL import Image, ImageDraw

app = FastAPI(title="Fake Video Server", version="2.0.0")

tasks: dict[str, dict] = {}
tasks_lock = threading.Lock()

STATE_SEQUENCE = ["created", "queueing", "processing", "success"]
STATE_DURATIONS = {
    "created": 2,
    "queueing": 3,
    "processing": 5,
}


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


def generate_fake_gif(
    width: int = 640, height: int = 360, num_frames: int = 24, fps: int = 8
) -> str:
    frames: list[Image.Image] = []
    colors = [
        (40, 40, 80),
        (80, 40, 40),
        (40, 80, 40),
        (80, 80, 40),
        (40, 80, 80),
        (80, 40, 80),
    ]

    for i in range(num_frames):
        color = colors[i % len(colors)]
        img = Image.new("RGB", (width, height), color=color)
        draw = ImageDraw.Draw(img)

        draw.text((20, 20), "[Fake Video - Vidu Style]", fill=(255, 255, 255))
        draw.text((20, 50), f"Frame {i + 1}/{num_frames}", fill=(255, 255, 255))
        draw.text((20, 80), f"{fps}fps  {width}x{height}", fill=(200, 200, 200))
        draw.text(
            (20, height - 30),
            f"Task Mode  {datetime.now().strftime('%H:%M:%S')}",
            fill=(180, 180, 180),
        )

        square_x = int((i / max(num_frames - 1, 1)) * (width - 100)) + 50
        draw.rectangle(
            [square_x - 30, height // 2 - 30, square_x + 30, height // 2 + 30],
            fill=(255, 100, 100),
            outline=(255, 255, 255),
            width=2,
        )

        frames.append(img)

    buf = io.BytesIO()
    frames[0].save(
        buf,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=int(1000 / fps),
        loop=0,
    )
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _advance_task(task_id: str):
    for state in STATE_SEQUENCE:
        time.sleep(STATE_DURATIONS.get(state, 2))
        with tasks_lock:
            if task_id in tasks:
                tasks[task_id]["state"] = state
                ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                print(f"  [{ts}] Task {task_id[:8]}... → {state}")

    with tasks_lock:
        if task_id in tasks:
            gif_b64 = generate_fake_gif()
            tasks[task_id]["creations"] = [
                {
                    "id": f"cre_{task_id[:8]}",
                    "url": f"data:video/mp4;base64,{gif_b64}",
                    "cover_url": "",
                }
            ]


# ===================== Vidu 风格异步接口 =====================


@app.post("/ent/v2/creations")
async def submit_creation(request: Request):
    body = await request.json()

    print("\n" + "=" * 72)
    print(
        f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] === Vidu Submit: 收到任务提交 ==="
    )
    print("-" * 72)
    print("[请求体]")
    print(fmt_json(body))
    print("-" * 72)

    task_id = f"vidu_{uuid.uuid4().hex[:16]}"

    with tasks_lock:
        tasks[task_id] = {
            "id": task_id,
            "state": "created",
            "err_code": "",
            "credits": 0,
            "payload": json.dumps(body, ensure_ascii=False),
            "creations": [],
        }

    thread = threading.Thread(target=_advance_task, args=(task_id,), daemon=True)
    thread.start()

    response = {"id": task_id, "state": "created"}
    print(f"[响应] task_id={task_id}, state=created")
    print("=" * 72 + "\n")

    return JSONResponse(content=response)


@app.get("/ent/v2/tasks/{task_id}/creations")
async def query_creation(task_id: str):
    with tasks_lock:
        task = tasks.get(task_id)

    if not task:
        return JSONResponse(
            content={
                "id": task_id,
                "state": "failed",
                "err_code": "NOT_FOUND",
                "creations": [],
            },
            status_code=404,
        )

    print(
        f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] Poll: {task_id[:8]}... → {task['state']}"
    )

    return JSONResponse(
        content={
            "id": task["id"],
            "state": task["state"],
            "err_code": task.get("err_code", ""),
            "credits": task.get("credits", 0),
            "payload": task.get("payload", ""),
            "creations": task.get("creations", []),
        }
    )


# ===================== OpenAI 兼容接口（保留） =====================


@app.api_route("/v1/chat/completions", methods=["POST"])
async def chat_completions(request: Request):
    body = await request.json()
    model = body.get("model", "unknown")
    extra_body = body.get("extra_body", {})
    video_params = extra_body.get("video", {})
    width = video_params.get("width", 1024)
    height = video_params.get("height", 576)
    num_frames = video_params.get("num_frames", 48)
    fps = video_params.get("fps", 8)

    print(
        f"\n[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] === OpenAI Compat: 收到请求 ==="
    )
    print(f"  model={model}, {width}x{height}, {num_frames}frames, {fps}fps")

    video_b64 = generate_fake_gif(width, height, num_frames, fps)

    response = {
        "id": f"chatcmpl-fake-{int(time.time()*1000)}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": f"[Fake Video] {num_frames}-frame, {width}x{height}",
                    "video": {
                        "data": video_b64,
                        "video_base64": video_b64,
                        "format": "gif",
                        "width": width,
                        "height": height,
                        "num_frames": num_frames,
                        "fps": fps,
                    },
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 200, "completion_tokens": 500, "total_tokens": 700},
    }

    return JSONResponse(content=response)


@app.get("/v1/models")
async def list_models():
    return JSONResponse(
        {
            "object": "list",
            "data": [{"id": "fake-video-model", "object": "model"}],
        }
    )


@app.get("/health")
async def health():
    with tasks_lock:
        task_count = len(tasks)
    return {
        "status": "ok",
        "server": "Fake Video Server v2.0",
        "active_tasks": task_count,
    }


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8103
    print(f"Fake Video Server v2.0 (Vidu-style async + OpenAI compat)")
    print(f"  Vidu submit:  POST http://localhost:{port}/ent/v2/creations")
    print(f"  Vidu poll:    GET  http://localhost:{port}/ent/v2/tasks/{{id}}/creations")
    print(f"  OpenAI:       POST http://localhost:{port}/v1/chat/completions")
    uvicorn.run(app, host="0.0.0.0", port=port)
