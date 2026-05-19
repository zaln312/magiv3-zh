"""
Application-level configuration (NOT per-project).
Stored in the same SQLite database.

Settings:
  - magi_v3_mode: 'dynamic' (load/unload per operation) or 'persistent_project' (load on enter, unload on exit)
  - ocr:       { api_url, enabled } — user provides call/format via app/utils/ocr_utils.py, URL is the only configurable field
  - caption:   OpenAI-compatible API  { base_url, api_key, model, prompt_template, temperature, max_tokens, top_p, extra_body }
  - prose:     OpenAI-compatible API  { base_url, api_key, model, prompt_template, temperature, max_tokens, top_p, extra_body }
"""

import json
import os
import sqlite3
from typing import Any

from app.services.database import _get_conn, DB_PATH


_DEFAULT_CONFIG = {
    "magi_v3_mode": "dynamic",
    "ocr": {
        "api_url": "http://127.0.0.1:8000/ocr",
        "format_code": (
            "def user_format_ocr_results(results):\n"
            '    """\n'
            "    用户提供的 ocr results 格式化\n"
            '    """\n'
            "    return [\n"
            "        {\n"
            '            "polys": res["rec_polys"],\n'
            '            "texts": res["rec_texts"],\n'
            "        }\n"
            "        for res in results\n"
            "    ]\n"
        ),
    },
    "caption": {
        "base_url": "http://localhost:8001/v1",
        "api_key": "EMPTY",
        "model": "Qwen3.5-4B",
        "prompt_template": (
            "Describe this image in a single prose paragraph. "
            "For each character, start by clearly stating their relative position "
            "(e.g., 'the character on the left', 'in the foreground', 'the girl on the right'), "
            "then describe their appearance (hair, clothing), and finally their actions or emotions. "
            "Do not use specific names. Ignore all embedded text, speech bubbles, and dialogue. "
            "Focus purely on visual elements."
        ),
        "temperature": 0.7,
        "top_p": 0.8,
        "max_tokens": 1024,
        "presence_penalty": 1.5,
        "extra_body": {
            "top_k": 20,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    },
    "prose": {
        "base_url": "http://localhost:8001/v1",
        "api_key": "EMPTY",
        "model": "Qwen3.5-4B",
        "prompt_template": (
            "{prompt}\n\n"
            "I want you to write a summary in Chinese so that a blind or visually impaired person can understand the story. "
            "Make sure to stick to the provided details. All these panels belong to the same page so make sure your narrative is coherent. "
            "The format of the narrative should be a prose."
        ),
        "temperature": 0.7,
        "top_p": 0.8,
        "max_tokens": 4096,
        "presence_penalty": 1.5,
        "extra_body": {
            "top_k": 20,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    },
    "reference": {
        "enabled": True,
        "base_url": "http://localhost:8001/v1",
        "api_key": "EMPTY",
        "model": "",
        "prompt_template": (
            "character reference sheet, {view} view, full body standing pose, "
            "{character_name}, clean white background, anime manga style, "
            "detailed character design, high quality, professional illustration"
        ),
        "negative_prompt": (
            "blurry, low quality, distorted face, bad anatomy, extra limbs, "
            "missing limbs, deformed hands, watermark, text, signature"
        ),
        "num_images_per_view": 1,
        "width": 512,
        "height": 768,
    },
    "video": {
        "enabled": True,
        "submit": {
            "url": "",
            "method": "POST",
            "headers": {},
            "body_template": {},
            "task_id_path": "id",
        },
        "poll": {
            "url_template": "",
            "method": "GET",
            "headers": {},
            "interval_seconds": 5,
            "max_attempts": 120,
            "state_path": "state",
            "state_values": {
                "created": "created",
                "queueing": "queueing",
                "processing": "processing",
                "success": "success",
                "failed": "failed",
            },
            "creations_path": "creations",
            "url_path": "url",
            "cover_url_path": "cover_url",
        },
        "submit_code": "",
        "poll_code": "",
        "parse_code": "",
    },
}


def _ensure_config_table():
    conn = _get_conn()
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS app_config (
            key   TEXT PRIMARY KEY,
            value TEXT NOT NULL DEFAULT ''
        );
    """
    )
    conn.commit()
    conn.close()


def _load_raw() -> dict[str, str]:
    _ensure_config_table()
    conn = _get_conn()
    rows = conn.execute("SELECT key, value FROM app_config").fetchall()
    conn.close()
    return {r["key"]: r["value"] for r in rows}


def load_config() -> dict[str, Any]:
    raw = _load_raw()
    merged = _deep_merge(_DEFAULT_CONFIG, {})
    for key, json_str in raw.items():
        try:
            merged[key] = json.loads(json_str)
        except (json.JSONDecodeError, TypeError):
            pass
    return merged


def _deep_merge(base: dict, override: dict) -> dict:
    result = {}
    for k, v in base.items():
        if k in override and isinstance(v, dict) and isinstance(override[k], dict):
            result[k] = _deep_merge(v, override[k])
        elif k in override:
            result[k] = override[k]
        else:
            result[k] = v
    for k, v in override.items():
        if k not in result:
            result[k] = v
    return result


_REQUIRED_FIELDS = {
    "caption": {
        "base_url": {"type": str, "non_empty": True},
        "api_key": {"type": str},
        "model": {"type": str, "non_empty": True},
        "prompt_template": {"type": str, "non_empty": True},
    },
    "prose": {
        "base_url": {"type": str, "non_empty": True},
        "api_key": {"type": str},
        "model": {"type": str, "non_empty": True},
        "prompt_template": {"type": str, "non_empty": True},
    },
    "reference": {
        "enabled": {"type": bool},
    },
    "video": {
        "enabled": {"type": bool},
        "submit": {"type": dict},
        "poll": {"type": dict},
        "submit_code": {"type": str},
        "poll_code": {"type": str},
        "parse_code": {"type": str},
    },
    "ocr": {
        "api_url": {"type": str, "non_empty": True},
        "format_code": {"type": str, "non_empty": True},
    },
}

_PLACEHOLDER_RULES = {
    "prose": {"prompt_template": "{prompt}"},
}


def _validate_config_section(section_key: str, data: dict) -> list[str]:
    errors = []
    rules = _REQUIRED_FIELDS.get(section_key, {})

    # 对于简单字符串类型的配置项（如 magi_v3_mode），直接校验类型
    if not isinstance(data, dict):
        if section_key in _REQUIRED_FIELDS:
            rule = rules.get("type")
            if rule and not isinstance(data, rule):
                errors.append(
                    f"[{section_key}] 类型错误: 期望 {rule.__name__}, 实际 {type(data).__name__}"
                )
        return errors

    for field, rule in rules.items():
        if field not in data:
            errors.append(f"[{section_key}] 缺少必填字段: {field}")
            continue
        val = data[field]
        expected_type = rule.get("type")
        if expected_type and not isinstance(val, expected_type):
            errors.append(
                f"[{section_key}] {field} 类型错误: 期望 {expected_type.__name__}, "
                f"实际 {type(val).__name__}"
            )
        if rule.get("non_empty") and isinstance(val, str) and not val.strip():
            errors.append(f"[{section_key}] {field} 不能为空字符串")

    placeholder_rules = _PLACEHOLDER_RULES.get(section_key, {})
    for field, placeholder in placeholder_rules.items():
        if field in data and isinstance(data[field], str):
            if placeholder not in data[field]:
                errors.append(f"[{section_key}] {field} 必须包含占位符 {placeholder}")

    return errors


def save_config(updates: dict[str, Any]):
    _ensure_config_table()
    current = load_config()

    for top_key, top_val in updates.items():
        if top_key not in _DEFAULT_CONFIG:
            continue
        current[top_key] = top_val

    all_errors = []
    for key in current:
        if key in _REQUIRED_FIELDS:
            all_errors.extend(_validate_config_section(key, current[key]))

    if all_errors:
        raise ValueError("配置校验失败:\n" + "\n".join(all_errors))

    conn = _get_conn()
    for key in _DEFAULT_CONFIG:
        val = current.get(key, _DEFAULT_CONFIG[key])
        conn.execute(
            "INSERT OR REPLACE INTO app_config (key, value) VALUES (?, ?)",
            (key, json.dumps(val, ensure_ascii=False)),
        )
    conn.commit()
    conn.close()


def reset_config():
    _ensure_config_table()
    conn = _get_conn()
    conn.execute("DELETE FROM app_config")
    conn.commit()
    conn.close()


# --- helpers for building OpenAI client from config ---


def get_caption_openai_config() -> dict:
    cfg = load_config().get("caption", {})
    return {
        "base_url": cfg.get("base_url", _DEFAULT_CONFIG["caption"]["base_url"]),
        "api_key": cfg.get("api_key", "EMPTY"),
        "model": cfg.get("model", _DEFAULT_CONFIG["caption"]["model"]),
        "temperature": cfg.get("temperature", 0.7),
        "top_p": cfg.get("top_p", 0.8),
        "max_tokens": cfg.get("max_tokens", 1024),
        "presence_penalty": cfg.get("presence_penalty", 1.5),
        "extra_body": cfg.get("extra_body", _DEFAULT_CONFIG["caption"]["extra_body"]),
        "prompt_template": cfg.get(
            "prompt_template", _DEFAULT_CONFIG["caption"]["prompt_template"]
        ),
    }


def get_prose_openai_config() -> dict:
    prose_cfg = load_config().get("prose", {})
    return {
        "base_url": prose_cfg.get("base_url", _DEFAULT_CONFIG["prose"]["base_url"]),
        "api_key": prose_cfg.get("api_key", "EMPTY"),
        "model": prose_cfg.get("model", _DEFAULT_CONFIG["prose"]["model"]),
        "temperature": prose_cfg.get("temperature", 0.7),
        "top_p": prose_cfg.get("top_p", 0.8),
        "max_tokens": prose_cfg.get("max_tokens", 4096),
        "presence_penalty": prose_cfg.get("presence_penalty", 1.5),
        "extra_body": prose_cfg.get(
            "extra_body", _DEFAULT_CONFIG["prose"]["extra_body"]
        ),
        "prompt_template": prose_cfg.get(
            "prompt_template", _DEFAULT_CONFIG["prose"]["prompt_template"]
        ),
    }


def get_ocr_api_url() -> str:
    cfg = load_config().get("ocr", {})
    return cfg.get("api_url", _DEFAULT_CONFIG["ocr"]["api_url"])


def get_ocr_format_code() -> str:
    cfg = load_config().get("ocr", {})
    return cfg.get("format_code", _DEFAULT_CONFIG["ocr"]["format_code"])


def get_magi_v3_mode() -> str:
    return load_config().get("magi_v3_mode", "dynamic")


def get_reference_config() -> dict:
    cfg = load_config().get("reference", {})
    defaults = _DEFAULT_CONFIG.get("reference", {})
    return {
        "enabled": cfg.get("enabled", defaults.get("enabled", True)),
        "base_url": cfg.get("base_url", defaults.get("base_url", "")),
        "api_key": cfg.get("api_key", defaults.get("api_key", "EMPTY")),
        "model": cfg.get("model", defaults.get("model", "")),
        "prompt_template": cfg.get(
            "prompt_template", defaults.get("prompt_template", "")
        ),
        "negative_prompt": cfg.get(
            "negative_prompt", defaults.get("negative_prompt", "")
        ),
        "num_images_per_view": cfg.get(
            "num_images_per_view", defaults.get("num_images_per_view", 1)
        ),
        "width": cfg.get("width", defaults.get("width", 512)),
        "height": cfg.get("height", defaults.get("height", 768)),
    }


def get_video_config() -> dict:
    cfg = load_config().get("video", {})
    defaults = _DEFAULT_CONFIG.get("video", {})
    return {
        "enabled": cfg.get("enabled", defaults.get("enabled", True)),
        "submit": _deep_merge(defaults.get("submit", {}), cfg.get("submit", {})),
        "poll": _deep_merge(defaults.get("poll", {}), cfg.get("poll", {})),
        "submit_code": cfg.get("submit_code", defaults.get("submit_code", "")),
        "poll_code": cfg.get("poll_code", defaults.get("poll_code", "")),
        "parse_code": cfg.get("parse_code", defaults.get("parse_code", "")),
    }
