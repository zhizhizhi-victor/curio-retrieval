#!/usr/bin/env python3
"""Encode gallery images with CLIP and save index.npz."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODELS = ROOT / "models" / "clip-vit-base-patch32"
MODEL_ID = "openai/clip-vit-base-patch32"


def main() -> None:
    gallery = json.loads((DATA / "gallery.json").read_text())
    items = gallery["items"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_src = str(MODELS) if (MODELS / "config.json").exists() else MODEL_ID
    print(f"device={device} model={model_src}")

    processor = CLIPProcessor.from_pretrained(model_src)
    model = CLIPModel.from_pretrained(model_src).to(device)
    model.eval()

    embeds = []
    ids = []
    with torch.inference_mode():
        for it in tqdm(items, desc="encode"):
            path = DATA / it["image_path"]
            img = Image.open(path).convert("RGB")
            inputs = processor(images=img, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            feat = model.get_image_features(**inputs)
            feat = feat / feat.norm(dim=-1, keepdim=True)
            embeds.append(feat.cpu().numpy()[0].astype(np.float32))
            ids.append(it["id"])

    arr = np.stack(embeds, axis=0)
    fp = gallery["meta"]["fingerprint"]
    out = DATA / "index.npz"
    np.savez_compressed(out, embeddings=arr, ids=np.array(ids), fingerprint=fp)
    # save model locally for offline app use
    if not (MODELS / "config.json").exists():
        MODELS.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(MODELS)
        processor.save_pretrained(MODELS)
        print(f"saved local model -> {MODELS}")
    print(f"index {arr.shape} -> {out}")


if __name__ == "__main__":
    main()
