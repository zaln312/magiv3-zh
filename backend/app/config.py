import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

sys.path.insert(0, PROJECT_ROOT)

UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

OCR_SERVER_URL = "http://127.0.0.1:8000/ocr"
LLAMA_SERVER_URL = "http://localhost:8001/v1"

CORS_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]