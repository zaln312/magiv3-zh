from fastapi import APIRouter, HTTPException
from app.services.app_config import load_config, save_config, reset_config, get_magi_v3_mode

router = APIRouter()


@router.get("/config")
async def get_config():
    return {"success": True, "config": load_config()}


@router.post("/config")
async def update_config(data: dict):
    try:
        save_config(data)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return {"success": True, "config": load_config()}


@router.post("/config/reset")
async def reset_config_api():
    reset_config()
    return {"success": True, "config": load_config()}


@router.get("/config/magi_v3_mode")
async def get_magi_mode():
    return {"success": True, "mode": get_magi_v3_mode()}