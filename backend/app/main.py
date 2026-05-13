from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import CORS_ORIGINS
from app.routers import upload, ocr, predict, caption, grounding, prose, character

app = FastAPI(title="Magi Studio API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload.router, prefix="/api", tags=["upload"])
app.include_router(ocr.router, prefix="/api", tags=["ocr"])
app.include_router(predict.router, prefix="/api", tags=["predict"])
app.include_router(caption.router, prefix="/api", tags=["caption"])
app.include_router(grounding.router, prefix="/api", tags=["grounding"])
app.include_router(prose.router, prefix="/api", tags=["prose"])
app.include_router(character.router, prefix="/api", tags=["character"])


@app.get("/api/health")
async def health_check():
    return {"status": "ok", "message": "Magi Studio API is running"}
