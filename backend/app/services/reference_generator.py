import io
import base64
from PIL import Image


class IPAdapterSD15Generator:

    def __init__(self):
        self._pipe = None
        self._model_id = "runwayml/stable-diffusion-v1-5"
        self._ip_adapter_scale = 0.7

    @property
    def is_loaded(self) -> bool:
        return self._pipe is not None

    def load(self):
        if self._pipe is not None:
            return
        import torch
        from diffusers import StableDiffusionPipeline

        self._pipe = StableDiffusionPipeline.from_pretrained(
            self._model_id,
            torch_dtype=torch.float16,
            safety_checker=None,
        ).to("cuda")

        self._pipe.load_ip_adapter(
            "h94/IP-Adapter",
            subfolder="models",
            weight_name="ip-adapter_sd15.bin",
        )
        self._pipe.enable_attention_slicing()

    def unload(self):
        if self._pipe is None:
            return
        import torch

        del self._pipe
        self._pipe = None
        torch.cuda.empty_cache()

    def generate(
        self,
        crop_images: list[Image.Image],
        prompt: str,
        negative_prompt: str = "",
        num_images: int = 1,
        width: int = 512,
        height: int = 768,
        ip_adapter_scale: float | None = None,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 30,
        **kwargs,
    ) -> list[Image.Image]:
        if not self.is_loaded:
            raise RuntimeError("IP-Adapter model not loaded")

        scale = (
            ip_adapter_scale if ip_adapter_scale is not None else self._ip_adapter_scale
        )
        self._pipe.set_ip_adapter_scale(scale)

        all_results = []
        for crop_img in crop_images:
            crop_resized = crop_img.resize((224, 224), Image.LANCZOS)

            result = self._pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                ip_adapter_image=crop_resized,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                num_images_per_prompt=num_images,
                width=width,
                height=height,
            ).images
            all_results.extend(result)

        return all_results

    def generate_and_encode(
        self,
        crop_images: list[Image.Image],
        prompt: str,
        negative_prompt: str = "",
        num_images: int = 1,
        width: int = 512,
        height: int = 768,
        **kwargs,
    ) -> list[dict]:
        results = self.generate(
            crop_images=crop_images,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_images=num_images,
            width=width,
            height=height,
            **kwargs,
        )
        encoded = []
        for img in results:
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            encoded.append({"image_base64": b64, "format": "png"})
        return encoded


ip_adapter_generator = IPAdapterSD15Generator()
