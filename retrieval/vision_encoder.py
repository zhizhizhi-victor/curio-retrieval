"""Unified CLIP / SigLIP image-text encoders with optional LoRA image tower."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor, CLIPModel, CLIPProcessor


CLIP_ID = "openai/clip-vit-base-patch32"
SIGLIP_ID = "google/siglip-base-patch16-224"  # widely available; SigLIP2 if fails fallback


class VisionTextEncoder:
    def __init__(
        self,
        model_name: str,
        local_dir: Path | None = None,
        lora_dir: Path | None = None,
        device: str | None = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        src = str(local_dir) if local_dir and (Path(local_dir) / "config.json").exists() else model_name
        self.is_clip = "clip" in model_name.lower() and "siglip" not in model_name.lower()

        if self.is_clip:
            self.processor = CLIPProcessor.from_pretrained(src)
            self.model = CLIPModel.from_pretrained(src)
        else:
            self.processor = AutoProcessor.from_pretrained(src)
            self.model = AutoModel.from_pretrained(src)

        if lora_dir and Path(lora_dir).exists():
            from peft import PeftModel

            # Attach LoRA to vision tower when available
            vision = getattr(self.model, "vision_model", None) or getattr(
                self.model, "vision_tower", None
            )
            if vision is not None:
                try:
                    self.model.vision_model = PeftModel.from_pretrained(self.model.vision_model, str(lora_dir))
                except Exception:
                    # Some checkpoints wrap whole model
                    self.model = PeftModel.from_pretrained(self.model, str(lora_dir))

        self.model.to(self.device)
        self.model.eval()
        self._text_cache: dict[str, np.ndarray] = {}

    @torch.inference_mode()
    def encode_text(self, text: str) -> np.ndarray:
        if text in self._text_cache:
            return self._text_cache[text]
        if self.is_clip:
            inputs = self.processor(text=[text], return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            feat = self.model.get_text_features(**inputs)
        else:
            inputs = self.processor(text=[text], return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            out = self.model.get_text_features(**inputs)
            feat = out
        feat = feat / feat.norm(dim=-1, keepdim=True)
        arr = feat.detach().cpu().numpy()[0].astype(np.float32)
        self._text_cache[text] = arr
        return arr

    @torch.inference_mode()
    def encode_image(self, image: Image.Image) -> np.ndarray:
        if self.is_clip:
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            feat = self.model.get_image_features(**inputs)
        else:
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            feat = self.model.get_image_features(**inputs)
        feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat.detach().cpu().numpy()[0].astype(np.float32)
