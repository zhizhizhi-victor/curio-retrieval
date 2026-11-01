"""CLIP zero-shot text-to-image retrieval."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


DEFAULT_MODEL = "openai/clip-vit-base-patch32"


class CLIPRetriever:
    def __init__(
        self,
        items: list[dict],
        index_path: Path,
        model_name: str = DEFAULT_MODEL,
        local_model_dir: Path | None = None,
        device: str | None = None,
    ):
        self.items = items
        self.ids = [it["id"] for it in items]
        self.id_to_idx = {i: n for n, i in enumerate(self.ids)}
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # Prefer local weights when present; otherwise Hugging Face hub id.
        model_src = model_name
        if local_model_dir and (local_model_dir / "config.json").exists():
            has_weights = any(
                (local_model_dir / name).exists()
                for name in ("model.safetensors", "pytorch_model.bin")
            )
            if has_weights:
                model_src = str(local_model_dir)
        self.processor = CLIPProcessor.from_pretrained(model_src)
        self.model = CLIPModel.from_pretrained(model_src).to(self.device)
        self.model.eval()

        data = np.load(index_path)
        self.image_embeds = data["embeddings"].astype(np.float32)
        index_ids = list(data["ids"])
        if list(index_ids) != self.ids:
            # reorder embeddings to match current gallery order
            pos = {i: n for n, i in enumerate(index_ids)}
            order = [pos[i] for i in self.ids]
            self.image_embeds = self.image_embeds[order]
        # L2 normalize
        norms = np.linalg.norm(self.image_embeds, axis=1, keepdims=True) + 1e-8
        self.image_embeds = self.image_embeds / norms
        self._text_cache: dict[str, np.ndarray] = {}

    @torch.inference_mode()
    def encode_text(self, text: str) -> np.ndarray:
        if text in self._text_cache:
            return self._text_cache[text]
        inputs = self.processor(text=[text], return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        feat = self.model.get_text_features(**inputs)
        feat = feat / feat.norm(dim=-1, keepdim=True)
        arr = feat.cpu().numpy()[0].astype(np.float32)
        self._text_cache[text] = arr
        return arr

    @torch.inference_mode()
    def encode_image(self, image: Image.Image) -> np.ndarray:
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        feat = self.model.get_image_features(**inputs)
        feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat.cpu().numpy()[0].astype(np.float32)

    def search(self, query: str, top_k: int | None = None) -> list[tuple[str, float]]:
        q = self.encode_text(query)
        scores = self.image_embeds @ q
        order = np.argsort(-scores)
        if top_k is not None:
            order = order[:top_k]
        return [(self.ids[i], float(scores[i])) for i in order]
