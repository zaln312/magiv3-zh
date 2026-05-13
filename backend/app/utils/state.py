from typing import Optional
from pydantic import BaseModel


class GlobalState:
    def __init__(self):
        self.img_paths: list[str] = []
        self.unordered_ocr_res: list[dict] = []
        self.results: list[dict] = []
        self.captions: list[list[str]] = []
        self.grounded_results: list[list[dict]] = []
        self.grounded_captions: list[list[str]] = []
        self.panel_scripts: list[list[list[str]]] = []
        self.prose_prompt: list[str] = []
        self.prose: str = ""
        self.global_character_library: list[dict] = []
        self.character_name_map: dict[int, str] = {}

    def reset_pipeline(self):
        self.unordered_ocr_res = []
        self.results = []
        self.captions = []
        self.grounded_results = []
        self.grounded_captions = []
        self.panel_scripts = []
        self.prose_prompt = []
        self.prose = ""
        self.global_character_library = []
        self.character_name_map = {}

    def reset_from_ocr(self):
        self.results = []
        self.captions = []
        self.grounded_results = []
        self.grounded_captions = []
        self.panel_scripts = []
        self.prose_prompt = []
        self.prose = ""
        self.global_character_library = []
        self.character_name_map = {}

    def reset_from_predict(self):
        self.captions = []
        self.grounded_results = []
        self.grounded_captions = []
        self.panel_scripts = []
        self.prose_prompt = []
        self.prose = ""


state = GlobalState()