import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))


class ModelManager:
    def __init__(self):
        self._model = None
        self._processor = None
        self._model_path = "model/florence2"

    @property
    def model(self):
        return self._model

    @property
    def processor(self):
        return self._processor

    @property
    def is_loaded(self):
        return self._model is not None

    def load(self):
        if self._model is not None:
            return

        import torch
        from transformers import AutoProcessor, AutoModelForCausalLM

        project_root = os.path.join(os.path.dirname(__file__), "../../..")
        model_path = os.path.join(project_root, self._model_path)

        self._model = (
            AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                trust_remote_code=True,
            )
            .cuda()
            .eval()
        )
        self._processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True
        )

    def unload(self):
        if self._model is None:
            return

        import torch

        del self._model
        del self._processor
        self._model = None
        self._processor = None
        torch.cuda.empty_cache()


model_manager = ModelManager()