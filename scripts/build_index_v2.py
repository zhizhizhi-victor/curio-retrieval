#!/usr/bin/env python3
"""Build CLIP + SigLIP image indexes for the expanded gallery."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from retrieval.vision_encoder import CLIP_ID, SIGLIP_ID, VisionTextEncoder

DATA = ROOT / "data"
MODELS = ROOT / "models"


def encode_gallery(name: str, model_id: str, local_subdir: str, out_name: str, lora_dir: Path | None = None):
    gallery = json.loads((DATA / "gallery.json").read_text())
    items = gallery["items"]
    local = MODELS / local_subdir
    enc = VisionTextEncoder(model_id, local_dir=local if local.exists() else None, lora_dir=lora_dir)
    embeds, ids = [], []
    for it in tqdm(items, desc=name):
        img = Image.open(DATA / it["image_path"]).convert("RGB")
        embeds.append(enc.encode_image(img))
        ids.append(it["id"])
    arr = np.stack(embeds, axis=0)
    np.savez_compressed(
        DATA / out_name,
        embeddings=arr,
        ids=np.array(ids),
        fingerprint=gallery["meta"]["fingerprint"],
        model=model_id,
    )
    # cache processor/model locally if not present
    if not (local / "config.json").exists():
        local.mkdir(parents=True, exist_ok=True)
        enc.model.save_pretrained(local)
        enc.processor.save_pretrained(local)
    print(f"{name}: {arr.shape} -> {out_name}")


def main():
    encode_gallery("clip", CLIP_ID, "clip-vit-base-patch32", "index_clip.npz")
    # Prefer SigLIP; fall back handled by caller if download fails
    try:
        encode_gallery("siglip", SIGLIP_ID, "siglip-base-patch16-224", "index_siglip.npz")
    except Exception as e:
        print(f"SigLIP failed ({e}); trying siglip2...")
        encode_gallery(
            "siglip2",
            "google/siglip2-base-patch16-224",
            "siglip2-base-patch16-224",
            "index_siglip.npz",
        )
    # LoRA index if adapter exists
    lora = MODELS / "siglip_lora_image"
    if lora.exists():
        encode_gallery(
            "siglip_lora",
            SIGLIP_ID,
            "siglip-base-patch16-224",
            "index_siglip_lora.npz",
            lora_dir=lora,
        )


if __name__ == "__main__":
    main()
