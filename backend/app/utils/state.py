from typing import Optional
from pydantic import BaseModel

from app.services.database import (
    update_project_step,
    save_images,
    save_ocr_results,
    update_ocr_text as db_update_ocr_text,
    update_ocr_boxes as db_update_ocr_boxes,
    save_predict_results,
    update_predict_result,
    save_global_characters,
    save_captions,
    save_grounded_captions,
    update_grounded_caption as db_update_grounded_caption,
    save_panel_scripts,
    save_prose,
    save_character_references,
    save_video,
    load_full_state,
)


class GlobalState:
    def __init__(self):
        self._project_id: str | None = None

        self.img_paths: list[str] = []
        self.unordered_ocr_res: list[dict] = []
        self.results: list[dict] = []
        self.captions: list[list[str]] = []
        self.grounded_results: list[list[dict]] = []
        self.grounded_captions: list[list[str]] = []
        self.panel_scripts: list[list[list[str]]] = []
        self.prose_prompt: list[str] = []
        self.prose: str = ""
        self.story_background: str = ""
        self.style_prompt: str = ""
        self.global_character_library: list[dict] = []
        self.character_name_map: dict[int, str] = {}
        self.character_references: dict[int, list[dict]] = {}
        self.video_result: dict | None = None
        self.video_task_id: str | None = None
        self.video_task_state: str = ""
        self.video_creations: list[dict] = []

    @property
    def project_id(self) -> str | None:
        return self._project_id

    def load_project(self, project_id: str):
        data = load_full_state(project_id)
        if not data:
            raise ValueError(f"项目 {project_id} 不存在")

        self._project_id = project_id
        self.img_paths = data["img_paths"]
        self.unordered_ocr_res = data["unordered_ocr_res"]
        self.results = data["results"]
        self.captions = data["captions"]
        self.grounded_results = []
        self.grounded_captions = data["grounded_captions"]
        self.panel_scripts = data["panel_scripts"]
        self.prose_prompt = data["prose_prompt"]
        self.prose = data["prose"]
        self.story_background = data.get("story_background", "")
        self.style_prompt = data.get("style_prompt", "")
        self.global_character_library = data["global_character_library"]
        self.character_name_map = data["character_name_map"]
        self.character_references = data["character_references"]
        self.video_result = data.get("video_result")
        video_data = data.get("video_result") or {}
        self.video_task_id = video_data.get("task_id")
        self.video_task_state = video_data.get("state", "")
        self.video_creations = video_data.get("creations", [])

    def set_project_id(self, project_id: str):
        self._project_id = project_id

    def _ensure_project(self):
        if not self._project_id:
            raise RuntimeError("未设置项目 ID")

    def persist_step(self, step: str):
        if self._project_id:
            update_project_step(self._project_id, step)

    def persist_images(self, original_names: list[str] | None = None):
        if self._project_id:
            save_images(self._project_id, self.img_paths, original_names)

    def persist_ocr_results(self):
        if self._project_id and self.unordered_ocr_res:
            save_ocr_results(self._project_id, self.unordered_ocr_res)

    def persist_ocr_text(self, img_idx: int):
        if self._project_id and img_idx < len(self.unordered_ocr_res):
            db_update_ocr_text(
                self._project_id,
                img_idx,
                self.unordered_ocr_res[img_idx].get("texts", []),
            )

    def persist_ocr_boxes(self, img_idx: int):
        if self._project_id and img_idx < len(self.unordered_ocr_res):
            db_update_ocr_boxes(
                self._project_id,
                img_idx,
                self.unordered_ocr_res[img_idx].get("boxes", []),
            )

    def persist_predict_results(self):
        if self._project_id and self.results:
            save_predict_results(self._project_id, self.results)

    def persist_predict_single(self, img_idx: int):
        if self._project_id and img_idx < len(self.results):
            update_predict_result(self._project_id, img_idx, self.results[img_idx])

    def persist_global_characters(self):
        if self._project_id:
            save_global_characters(
                self._project_id,
                self.global_character_library,
                self.character_name_map,
            )

    def persist_captions(self):
        if self._project_id and self.captions:
            save_captions(self._project_id, self.captions)

    def persist_grounded_captions(self):
        if self._project_id and self.grounded_captions:
            save_grounded_captions(self._project_id, self.grounded_captions)

    def persist_grounded_caption_single(self, img_idx: int, panel_idx: int):
        if (
            self._project_id
            and img_idx < len(self.grounded_captions)
            and panel_idx < len(self.grounded_captions[img_idx])
        ):
            db_update_grounded_caption(
                self._project_id,
                img_idx,
                panel_idx,
                self.grounded_captions[img_idx][panel_idx],
            )

    def persist_panel_scripts(self):
        if self._project_id and self.panel_scripts:
            save_panel_scripts(self._project_id, self.panel_scripts)

    def persist_prose(self):
        if self._project_id:
            save_prose(
                self._project_id,
                self.prose_prompt,
                self.prose,
                self.story_background,
                self.style_prompt,
            )

    def persist_character_references(self):
        if self._project_id and self.character_references:
            save_character_references(self._project_id, self.character_references)

    def persist_video(self):
        if self._project_id:
            video_data = self.video_result or {}
            video_data["task_id"] = self.video_task_id
            video_data["state"] = self.video_task_state
            video_data["creations"] = self.video_creations
            save_video(self._project_id, video_data)

    def reset_pipeline(self):
        self.unordered_ocr_res = []
        self.results = []
        self.captions = []
        self.grounded_results = []
        self.grounded_captions = []
        self.panel_scripts = []
        self.prose_prompt = []
        self.prose = ""
        self.story_background = ""
        self.global_character_library = []
        self.character_name_map = {}
        self.character_references = {}
        self.video_result = None
        self.video_task_id = None
        self.video_task_state = ""
        self.video_creations = []

    def reset_from_ocr(self):
        self.results = []
        self.captions = []
        self.grounded_results = []
        self.grounded_captions = []
        self.panel_scripts = []
        self.prose_prompt = []
        self.prose = ""
        self.story_background = ""
        self.global_character_library = []
        self.character_name_map = {}
        self.character_references = {}

    def reset_from_predict(self):
        self.captions = []
        self.grounded_results = []
        self.grounded_captions = []
        self.panel_scripts = []
        self.prose_prompt = []
        self.prose = ""
        self.story_background = ""
        self.character_references = {}
        self.video_result = None


state = GlobalState()
