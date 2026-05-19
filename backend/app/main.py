from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import CORS_ORIGINS
from app.routers import (
    upload,
    ocr,
    predict,
    caption,
    grounding,
    prose,
    character,
    project,
    video,
    config_router,
)
from app.services.database import init_db

init_db()

app = FastAPI(title="Magi Studio API", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(project.router, prefix="/api", tags=["project"])
app.include_router(upload.router, prefix="/api", tags=["upload"])
app.include_router(ocr.router, prefix="/api", tags=["ocr"])
app.include_router(predict.router, prefix="/api", tags=["predict"])
app.include_router(caption.router, prefix="/api", tags=["caption"])
app.include_router(grounding.router, prefix="/api", tags=["grounding"])
app.include_router(prose.router, prefix="/api", tags=["prose"])
app.include_router(character.router, prefix="/api", tags=["character"])
app.include_router(video.router, prefix="/api", tags=["video"])
app.include_router(config_router.router, prefix="/api", tags=["config"])


@app.get("/api/health")
async def health_check():
    return {"status": "ok", "message": "Magi Studio API is running"}
