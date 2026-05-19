import json
import os
import sqlite3
from datetime import datetime

from app.config import BASE_DIR

DB_PATH = os.path.join(BASE_DIR, "data", "magi.db")


def _get_conn() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db():
    conn = _get_conn()
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS projects (
            id          TEXT PRIMARY KEY,
            name        TEXT NOT NULL DEFAULT '',
            current_step TEXT NOT NULL DEFAULT 'upload',
            created_at  TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at  TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS project_images (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id  TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
            img_idx     INTEGER NOT NULL,
            file_path   TEXT NOT NULL,
            original_name TEXT NOT NULL DEFAULT '',
            UNIQUE(project_id, img_idx)
        );

        CREATE TABLE IF NOT EXISTS project_ocr_results (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id  TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
            img_idx     INTEGER NOT NULL,
            boxes       TEXT NOT NULL DEFAULT '[]',
            texts       TEXT NOT NULL DEFAULT '[]',
            UNIQUE(project_id, img_idx)
        );

        CREATE TABLE IF NOT EXISTS project_predict_results (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id  TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
            img_idx     INTEGER NOT NULL,
            result_json TEXT NOT NULL DEFAULT '{}',
            UNIQUE(project_id, img_idx)
        );

        CREATE TABLE IF NOT EXISTS project_global_characters (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id  TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
            global_id   INTEGER NOT NULL,
            name        TEXT NOT NULL DEFAULT '',
            UNIQUE(project_id, global_id)
        );

        CREATE TABLE IF NOT EXISTS project_captions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id  TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
            img_idx     INTEGER NOT NULL,
            panel_idx   INTEGER NOT NULL,
            caption     TEXT NOT NULL DEFAULT '',
            UNIQUE(project_id, img_idx, panel_idx)
        );

        CREATE TABLE IF NOT EXISTS project_grounded_captions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id  TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
            img_idx     INTEGER NOT NULL,
            panel_idx   INTEGER NOT NULL,
            grounded_caption TEXT NOT NULL DEFAULT '',
            UNIQUE(project_id, img_idx, panel_idx)
        );

        CREATE TABLE IF NOT EXISTS project_panel_scripts (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id  TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
            img_idx     INTEGER NOT NULL,
            panel_idx   INTEGER NOT NULL,
            script      TEXT NOT NULL DEFAULT '',
            UNIQUE(project_id, img_idx, panel_idx)
        );

        CREATE TABLE IF NOT EXISTS project_prose (
            project_id  TEXT PRIMARY KEY REFERENCES projects(id) ON DELETE CASCADE,
            prose_prompt TEXT NOT NULL DEFAULT '[]',
            prose       TEXT NOT NULL DEFAULT '',
            story_background TEXT NOT NULL DEFAULT ''
        );

        CREATE TABLE IF NOT EXISTS project_character_references (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id  TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
            global_id   INTEGER NOT NULL,
            ref_json    TEXT NOT NULL DEFAULT '[]'
        );
    """
    )
    conn.commit()

    try:
        conn.execute(
            "ALTER TABLE project_prose ADD COLUMN story_background TEXT NOT NULL DEFAULT ''"
        )
    except Exception:
        pass

    try:
        conn.execute(
            "ALTER TABLE project_prose ADD COLUMN style_prompt TEXT NOT NULL DEFAULT ''"
        )
    except Exception:
        pass

    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS project_video (
                project_id  TEXT PRIMARY KEY REFERENCES projects(id) ON DELETE CASCADE,
                video_json  TEXT NOT NULL DEFAULT '{}'
            )
        """
        )
    except Exception:
        pass

    try:
        conn.execute(
            "ALTER TABLE project_global_characters ADD COLUMN features TEXT NOT NULL DEFAULT ''"
        )
    except Exception:
        pass
    conn.commit()

    conn.close()


def _json_dumps(obj):
    return json.dumps(obj, ensure_ascii=False)


def _json_loads(text):
    return json.loads(text) if text else None


# ---- Project CRUD ----


def create_project(name: str = "") -> dict:
    import uuid

    project_id = uuid.uuid4().hex[:12]
    now = datetime.now().isoformat()
    conn = _get_conn()
    conn.execute(
        "INSERT INTO projects (id, name, current_step, created_at, updated_at) VALUES (?, ?, 'upload', ?, ?)",
        (project_id, name, now, now),
    )
    conn.commit()
    conn.close()
    return get_project(project_id)


def get_project(project_id: str) -> dict | None:
    conn = _get_conn()
    row = conn.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()
    conn.close()
    if row is None:
        return None
    return dict(row)


def list_projects() -> list[dict]:
    conn = _get_conn()
    rows = conn.execute("SELECT * FROM projects ORDER BY updated_at DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def update_project_step(project_id: str, step: str):
    conn = _get_conn()
    conn.execute(
        "UPDATE projects SET current_step = ?, updated_at = datetime('now') WHERE id = ?",
        (step, project_id),
    )
    conn.commit()
    conn.close()


def update_project_name(project_id: str, name: str):
    conn = _get_conn()
    conn.execute(
        "UPDATE projects SET name = ?, updated_at = datetime('now') WHERE id = ?",
        (name, project_id),
    )
    conn.commit()
    conn.close()


def delete_project(project_id: str):
    conn = _get_conn()
    conn.execute("DELETE FROM projects WHERE id = ?", (project_id,))
    conn.commit()
    conn.close()


# ---- Images ----


def save_images(
    project_id: str, img_paths: list[str], original_names: list[str] | None = None
):
    conn = _get_conn()
    conn.execute("DELETE FROM project_images WHERE project_id = ?", (project_id,))
    if original_names is None:
        original_names = [""] * len(img_paths)
    for idx, (path, name) in enumerate(zip(img_paths, original_names)):
        conn.execute(
            "INSERT INTO project_images (project_id, img_idx, file_path, original_name) VALUES (?, ?, ?, ?)",
            (project_id, idx, path, name),
        )
    conn.commit()
    conn.close()


def load_images(project_id: str) -> list[str]:
    conn = _get_conn()
    rows = conn.execute(
        "SELECT file_path FROM project_images WHERE project_id = ? ORDER BY img_idx",
        (project_id,),
    ).fetchall()
    conn.close()
    return [r["file_path"] for r in rows]


def load_image_names(project_id: str) -> dict[str, str]:
    conn = _get_conn()
    rows = conn.execute(
        "SELECT file_path, original_name FROM project_images WHERE project_id = ? ORDER BY img_idx",
        (project_id,),
    ).fetchall()
    conn.close()
    return {r["file_path"]: r["original_name"] for r in rows}


# ---- OCR Results ----


def save_ocr_results(project_id: str, ocr_results: list[dict]):
    conn = _get_conn()
    conn.execute("DELETE FROM project_ocr_results WHERE project_id = ?", (project_id,))
    for idx, res in enumerate(ocr_results):
        boxes = _json_dumps(res.get("boxes", []))
        texts = _json_dumps(res.get("texts", []))
        conn.execute(
            "INSERT INTO project_ocr_results (project_id, img_idx, boxes, texts) VALUES (?, ?, ?, ?)",
            (project_id, idx, boxes, texts),
        )
    conn.commit()
    conn.close()


def load_ocr_results(project_id: str) -> list[dict]:
    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM project_ocr_results WHERE project_id = ? ORDER BY img_idx",
        (project_id,),
    ).fetchall()
    conn.close()
    return [
        {"boxes": _json_loads(r["boxes"]), "texts": _json_loads(r["texts"])}
        for r in rows
    ]


def update_ocr_text(project_id: str, img_idx: int, texts: list[str]):
    conn = _get_conn()
    conn.execute(
        "UPDATE project_ocr_results SET texts = ? WHERE project_id = ? AND img_idx = ?",
        (_json_dumps(texts), project_id, img_idx),
    )
    conn.commit()
    conn.close()


def update_ocr_boxes(project_id: str, img_idx: int, boxes: list):
    conn = _get_conn()
    conn.execute(
        "UPDATE project_ocr_results SET boxes = ? WHERE project_id = ? AND img_idx = ?",
        (_json_dumps(boxes), project_id, img_idx),
    )
    conn.commit()
    conn.close()


# ---- Predict Results ----


def save_predict_results(project_id: str, results: list[dict]):
    conn = _get_conn()
    conn.execute(
        "DELETE FROM project_predict_results WHERE project_id = ?", (project_id,)
    )
    for idx, res in enumerate(results):
        serialized = {}
        for k, v in res.items():
            if hasattr(v, "tolist"):
                serialized[k] = v.tolist()
            elif isinstance(v, list):
                serialized[k] = [x.tolist() if hasattr(x, "tolist") else x for x in v]
            else:
                serialized[k] = v
        conn.execute(
            "INSERT INTO project_predict_results (project_id, img_idx, result_json) VALUES (?, ?, ?)",
            (project_id, idx, _json_dumps(serialized)),
        )
    conn.commit()
    conn.close()


def load_predict_results(project_id: str) -> list[dict]:
    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM project_predict_results WHERE project_id = ? ORDER BY img_idx",
        (project_id,),
    ).fetchall()
    conn.close()
    return [_json_loads(r["result_json"]) for r in rows]


def update_predict_result(project_id: str, img_idx: int, result: dict):
    conn = _get_conn()
    serialized = {}
    for k, v in result.items():
        if hasattr(v, "tolist"):
            serialized[k] = v.tolist()
        elif isinstance(v, list):
            serialized[k] = [x.tolist() if hasattr(x, "tolist") else x for x in v]
        else:
            serialized[k] = v
    conn.execute(
        "INSERT OR REPLACE INTO project_predict_results (project_id, img_idx, result_json) VALUES (?, ?, ?)",
        (project_id, img_idx, _json_dumps(serialized)),
    )
    conn.commit()
    conn.close()


# ---- Global Characters ----


def save_global_characters(
    project_id: str, character_library: list[dict], name_map: dict[int, str]
):
    conn = _get_conn()
    conn.execute(
        "DELETE FROM project_global_characters WHERE project_id = ?", (project_id,)
    )
    for entry in character_library:
        gid = entry["global_id"]
        name = name_map.get(gid, "")
        features = entry.get("features")
        if features is not None and hasattr(features, "tolist"):
            features = _json_dumps(features.tolist())
        elif features is not None and isinstance(features, list):
            features = _json_dumps(features)
        else:
            features = ""
        conn.execute(
            "INSERT INTO project_global_characters (project_id, global_id, name, features) VALUES (?, ?, ?, ?)",
            (project_id, gid, name, features),
        )
    conn.commit()
    conn.close()


def load_global_characters(project_id: str) -> tuple[list[dict], dict[int, str]]:
    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM project_global_characters WHERE project_id = ? ORDER BY global_id",
        (project_id,),
    ).fetchall()
    conn.close()
    library = []
    for r in rows:
        entry = {"global_id": r["global_id"]}
        features_str = r["features"] if "features" in r.keys() else ""
        if features_str:
            entry["features"] = _json_loads(features_str)
        library.append(entry)
    name_map = {r["global_id"]: r["name"] for r in rows}
    return library, name_map


# ---- Captions ----


def save_captions(project_id: str, captions: list[list[str]]):
    conn = _get_conn()
    conn.execute("DELETE FROM project_captions WHERE project_id = ?", (project_id,))
    for img_idx, caps in enumerate(captions):
        for panel_idx, cap in enumerate(caps):
            conn.execute(
                "INSERT INTO project_captions (project_id, img_idx, panel_idx, caption) VALUES (?, ?, ?, ?)",
                (project_id, img_idx, panel_idx, cap),
            )
    conn.commit()
    conn.close()


def load_captions(
    project_id: str, panel_count_per_img: list[int] | None = None
) -> list[list[str]]:
    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM project_captions WHERE project_id = ? ORDER BY img_idx, panel_idx",
        (project_id,),
    ).fetchall()
    conn.close()
    if not rows:
        return []
    max_img = max(r["img_idx"] for r in rows) + 1
    captions = [[] for _ in range(max_img)]
    for r in rows:
        while len(captions[r["img_idx"]]) <= r["panel_idx"]:
            captions[r["img_idx"]].append("")
        captions[r["img_idx"]][r["panel_idx"]] = r["caption"]
    return captions


def update_caption(project_id: str, img_idx: int, panel_idx: int, caption: str):
    conn = _get_conn()
    conn.execute(
        "INSERT OR REPLACE INTO project_captions (project_id, img_idx, panel_idx, caption) VALUES (?, ?, ?, ?)",
        (project_id, img_idx, panel_idx, caption),
    )
    conn.commit()
    conn.close()


# ---- Grounded Captions ----


def save_grounded_captions(project_id: str, grounded_captions: list[list[str]]):
    conn = _get_conn()
    conn.execute(
        "DELETE FROM project_grounded_captions WHERE project_id = ?", (project_id,)
    )
    for img_idx, caps in enumerate(grounded_captions):
        for panel_idx, cap in enumerate(caps):
            conn.execute(
                "INSERT INTO project_grounded_captions (project_id, img_idx, panel_idx, grounded_caption) VALUES (?, ?, ?, ?)",
                (project_id, img_idx, panel_idx, cap),
            )
    conn.commit()
    conn.close()


def load_grounded_captions(project_id: str) -> list[list[str]]:
    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM project_grounded_captions WHERE project_id = ? ORDER BY img_idx, panel_idx",
        (project_id,),
    ).fetchall()
    conn.close()
    if not rows:
        return []
    max_img = max(r["img_idx"] for r in rows) + 1
    captions = [[] for _ in range(max_img)]
    for r in rows:
        while len(captions[r["img_idx"]]) <= r["panel_idx"]:
            captions[r["img_idx"]].append("")
        captions[r["img_idx"]][r["panel_idx"]] = r["grounded_caption"]
    return captions


def update_grounded_caption(
    project_id: str, img_idx: int, panel_idx: int, caption: str
):
    conn = _get_conn()
    conn.execute(
        "INSERT OR REPLACE INTO project_grounded_captions (project_id, img_idx, panel_idx, grounded_caption) VALUES (?, ?, ?, ?)",
        (project_id, img_idx, panel_idx, caption),
    )
    conn.commit()
    conn.close()


# ---- Panel Scripts ----


def save_panel_scripts(project_id: str, panel_scripts: list[list[list[str]]]):
    conn = _get_conn()
    conn.execute(
        "DELETE FROM project_panel_scripts WHERE project_id = ?", (project_id,)
    )
    for img_idx, panels in enumerate(panel_scripts):
        for panel_idx, script_lines in enumerate(panels):
            script = (
                "\n".join(script_lines)
                if isinstance(script_lines, list)
                else str(script_lines)
            )
            conn.execute(
                "INSERT INTO project_panel_scripts (project_id, img_idx, panel_idx, script) VALUES (?, ?, ?, ?)",
                (project_id, img_idx, panel_idx, script),
            )
    conn.commit()
    conn.close()


def load_panel_scripts(project_id: str) -> list[list[list[str]]]:
    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM project_panel_scripts WHERE project_id = ? ORDER BY img_idx, panel_idx",
        (project_id,),
    ).fetchall()
    conn.close()
    if not rows:
        return []
    max_img = max(r["img_idx"] for r in rows) + 1
    scripts = [[] for _ in range(max_img)]
    for r in rows:
        while len(scripts[r["img_idx"]]) <= r["panel_idx"]:
            scripts[r["img_idx"]].append([])
        scripts[r["img_idx"]][r["panel_idx"]] = (
            r["script"].split("\n") if r["script"] else []
        )
    return scripts


# ---- Prose ----


def save_prose(
    project_id: str,
    prose_prompt: list[str],
    prose: str,
    story_background: str = "",
    style_prompt: str = "",
):
    conn = _get_conn()
    conn.execute(
        "INSERT OR REPLACE INTO project_prose (project_id, prose_prompt, prose, story_background, style_prompt) VALUES (?, ?, ?, ?, ?)",
        (project_id, _json_dumps(prose_prompt), prose, story_background, style_prompt),
    )
    conn.commit()
    conn.close()


def load_prose(project_id: str) -> tuple[list[str], str, str, str]:
    conn = _get_conn()
    row = conn.execute(
        "SELECT * FROM project_prose WHERE project_id = ?", (project_id,)
    ).fetchone()
    conn.close()
    if row is None:
        return [], "", "", ""
    return (
        _json_loads(row["prose_prompt"]) or [],
        row["prose"] or "",
        row["story_background"] or "",
        row["style_prompt"] or "",
    )


# ---- Character References ----


def save_character_references(project_id: str, refs: dict[int, list[dict]]):
    conn = _get_conn()
    conn.execute(
        "DELETE FROM project_character_references WHERE project_id = ?", (project_id,)
    )
    for global_id, ref_list in refs.items():
        refs_serializable = []
        for ref in ref_list:
            serializable_ref = {}
            for k, v in ref.items():
                if isinstance(v, bytes):
                    serializable_ref[k] = (
                        v.decode("utf-8") if isinstance(v, bytes) else str(v)
                    )
                else:
                    serializable_ref[k] = v
            refs_serializable.append(serializable_ref)
        conn.execute(
            "INSERT INTO project_character_references (project_id, global_id, ref_json) VALUES (?, ?, ?)",
            (project_id, global_id, _json_dumps(refs_serializable)),
        )
    conn.commit()
    conn.close()


def load_character_references(project_id: str) -> dict[int, list[dict]]:
    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM project_character_references WHERE project_id = ?",
        (project_id,),
    ).fetchall()
    conn.close()
    return {r["global_id"]: _json_loads(r["ref_json"]) for r in rows}


def save_video(project_id: str, video_result: dict):
    conn = _get_conn()
    serializable = {}
    for k, v in video_result.items():
        if isinstance(v, bytes):
            serializable[k] = v.decode("utf-8") if isinstance(v, bytes) else str(v)
        else:
            serializable[k] = v
    conn.execute(
        "INSERT OR REPLACE INTO project_video (project_id, video_json) VALUES (?, ?)",
        (project_id, _json_dumps(serializable)),
    )
    conn.commit()
    conn.close()


def load_video(project_id: str) -> dict | None:
    conn = _get_conn()
    row = conn.execute(
        "SELECT * FROM project_video WHERE project_id = ?", (project_id,)
    ).fetchone()
    conn.close()
    if row is None or not row["video_json"]:
        return None
    return _json_loads(row["video_json"])


# ---- Full state load ----


def load_full_state(project_id: str) -> dict:
    project = get_project(project_id)
    if project is None:
        return {}

    img_paths = load_images(project_id)
    ocr_results = load_ocr_results(project_id)
    predict_results = load_predict_results(project_id)
    captions = load_captions(project_id)
    grounded_captions = load_grounded_captions(project_id)
    panel_scripts = load_panel_scripts(project_id)
    prose_prompt, prose, story_background, style_prompt = load_prose(project_id)
    character_library, character_name_map = load_global_characters(project_id)
    character_references = load_character_references(project_id)
    video_result = load_video(project_id)

    return {
        "project": project,
        "img_paths": img_paths,
        "unordered_ocr_res": ocr_results,
        "results": predict_results,
        "captions": captions,
        "grounded_captions": grounded_captions,
        "panel_scripts": panel_scripts,
        "prose_prompt": prose_prompt,
        "prose": prose,
        "story_background": story_background,
        "style_prompt": style_prompt,
        "global_character_library": character_library,
        "character_name_map": character_name_map,
        "character_references": character_references,
        "video_result": video_result,
    }
