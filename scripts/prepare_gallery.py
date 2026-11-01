#!/usr/bin/env python3
"""Build a China-only gallery JSON and download images."""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path

import requests
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SAMPLES = DATA / "samples"
FILTERED = DATA / "filtered"
IMAGES = DATA / "images"
IMAGES.mkdir(parents=True, exist_ok=True)

CHINA_RE = re.compile(r"\b(?:china|chinese)\b", re.I)
BAD_CULTURE = re.compile(
    r"\b(?:iran|japan|japanese|thailand|vietnam|vietnamese|korea|korean|"
    r"hacilar|persia|persian|india|indian)\b",
    re.I,
)


def is_china_item(r: dict) -> bool:
    culture = r.get("culture") or ""
    dept = r.get("department") or ""
    title = r.get("title") or ""
    blob = f"{culture} {dept} {title}"
    if BAD_CULTURE.search(culture) and not CHINA_RE.search(culture):
        return False
    if "Ancient West Asian" in dept:
        return False
    if CHINA_RE.search(blob) or dept.strip() == "Chinese Art":
        return True
    return False


def prefer_score(r: dict) -> int:
    score = 0
    if r.get("period") not in {"", "Unknown", None}:
        score += 2
    if r.get("material") not in {"", "Other", None}:
        score += 2
    if r.get("object_type") not in {"Painting/Scroll", "Print", "Miscellaneous"}:
        score += 1
    if r.get("object_type") == "Snuff bottle":
        score -= 1  # avoid over-representing
    return score


def download_image(url: str, dest: Path, session: requests.Session) -> bool:
    if dest.exists() and dest.stat().st_size > 1000:
        return True
    try:
        r = session.get(url, timeout=45)
        if r.status_code != 200 or len(r.content) < 500:
            return False
        dest.write_bytes(r.content)
        return True
    except Exception:
        return False


def main(target: int = 750) -> None:
    sample = json.loads((SAMPLES / "gallery_candidate_800.json").read_text())
    pool = json.loads((FILTERED / "merged_china_openaccess.json").read_text())

    clean_sample = [r for r in sample if is_china_item(r)]
    print(f"sample after china filter: {len(clean_sample)}/{len(sample)}")

    # backfill from pool if needed
    have = {r["id"] for r in clean_sample}
    extras = [r for r in pool if r["id"] not in have and is_china_item(r)]
    extras.sort(key=prefer_score, reverse=True)

    gallery = list(clean_sample)
    for r in extras:
        if len(gallery) >= target:
            break
        gallery.append(r)

    # Cap snuff bottles to keep diversity
    snuff = [r for r in gallery if r.get("object_type") == "Snuff bottle"]
    other = [r for r in gallery if r.get("object_type") != "Snuff bottle"]
    if len(snuff) > 80:
        snuff = snuff[:80]
        need = target - len(snuff) - len(other)
        # already have other; trim gallery rebuild
        gallery = other + snuff
        if len(gallery) < target:
            for r in extras:
                if r["id"] in {x["id"] for x in gallery}:
                    continue
                if r.get("object_type") == "Snuff bottle":
                    continue
                gallery.append(r)
                if len(gallery) >= target:
                    break

    gallery = gallery[: max(target, min(len(gallery), 800))]
    print(f"gallery candidates before download: {len(gallery)}")

    session = requests.Session()
    session.headers.update({"User-Agent": "COMP5523-curio-retrieval/1.0"})
    kept = []
    for r in tqdm(gallery, desc="download"):
        url = r.get("image_url") or r.get("image_url_hires")
        if not url:
            continue
        ext = ".jpg"
        fname = f"{r['id']}{ext}"
        dest = IMAGES / fname
        ok = download_image(url, dest, session)
        if not ok and r.get("image_url_hires") and r.get("image_url_hires") != url:
            ok = download_image(r["image_url_hires"], dest, session)
        if not ok:
            continue
        item = dict(r)
        item["image_path"] = f"images/{fname}"
        item["local_image"] = str(dest.relative_to(DATA))
        kept.append(item)
        time.sleep(0.02)

    # fingerprint
    blob = json.dumps([x["id"] for x in kept], ensure_ascii=False).encode()
    fp = hashlib.sha256(blob).hexdigest()[:16]
    meta = {
        "count": len(kept),
        "fingerprint": fp,
        "sources": sorted({x["source"] for x in kept}),
        "domain": "Chinese museum collections (Met + Cleveland Open Access)",
        "license_note": "Public domain / CC0 museum open access images",
        "relevance_dimensions": ["object_type", "material", "period"],
    }
    out = {"meta": meta, "items": kept}
    (DATA / "gallery.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))
    # also flat list for convenience
    (DATA / "gallery_items.json").write_text(json.dumps(kept, ensure_ascii=False, indent=2))
    print(f"saved {len(kept)} items -> data/gallery.json fingerprint={fp}")


if __name__ == "__main__":
    main()
