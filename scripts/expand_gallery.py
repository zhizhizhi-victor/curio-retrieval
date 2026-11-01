#!/usr/bin/env python3
"""Expand China gallery toward a larger presentation set (~2200)."""

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
    return bool(CHINA_RE.search(blob) or dept.strip() == "Chinese Art")


def score(r: dict) -> int:
    s = 0
    if r.get("period") not in {"", "Unknown", None}:
        s += 2
    if r.get("material") not in {"", "Other", None}:
        s += 2
    if r.get("object_type") not in {"Painting/Scroll", "Print", "Miscellaneous", None}:
        s += 1
    if r.get("object_type") == "Snuff bottle":
        s -= 1
    return s


def download(url: str, dest: Path, session: requests.Session) -> bool:
    if dest.exists() and dest.stat().st_size > 1000:
        return True
    try:
        r = session.get(url, timeout=40)
        if r.status_code != 200 or len(r.content) < 500:
            return False
        dest.write_bytes(r.content)
        return True
    except Exception:
        return False


def main(target: int = 2200) -> None:
    pool = json.loads((FILTERED / "merged_china_openaccess.json").read_text())
    existing = {}
    gpath = DATA / "gallery.json"
    if gpath.exists():
        old = json.loads(gpath.read_text())
        for it in old.get("items", []):
            existing[it["id"]] = it

    candidates = [r for r in pool if is_china_item(r)]
    candidates.sort(key=score, reverse=True)

    # Cap snuff bottles
    snuff, other = [], []
    for r in candidates:
        (snuff if r.get("object_type") == "Snuff bottle" else other).append(r)
    snuff = snuff[:120]
    ordered = other + snuff

    session = requests.Session()
    session.headers.update({"User-Agent": "COMP5523-curio-retrieval/2.0"})
    kept = []
    for r in tqdm(ordered, desc="expand"):
        if len(kept) >= target:
            break
        rid = r["id"]
        if rid in existing and (IMAGES / f"{rid}.jpg").exists():
            item = dict(existing[rid])
            item.update({k: r.get(k, item.get(k)) for k in r})
            item["image_path"] = f"images/{rid}.jpg"
            kept.append(item)
            continue
        url = r.get("image_url") or r.get("image_url_hires")
        if not url:
            continue
        dest = IMAGES / f"{rid}.jpg"
        ok = download(url, dest, session)
        if not ok and r.get("image_url_hires") and r["image_url_hires"] != url:
            ok = download(r["image_url_hires"], dest, session)
        if not ok:
            continue
        item = dict(r)
        item["image_path"] = f"images/{rid}.jpg"
        kept.append(item)
        time.sleep(0.015)

    fp = hashlib.sha256(json.dumps([x["id"] for x in kept]).encode()).hexdigest()[:16]
    meta = {
        "count": len(kept),
        "fingerprint": fp,
        "sources": sorted({x["source"] for x in kept}),
        "domain": "Chinese museum collections (Met + Cleveland Open Access)",
        "license_note": "Public domain / CC0 museum open access images",
        "relevance_dimensions": ["object_type", "material", "period"],
        "version": "v2-expanded",
    }
    (DATA / "gallery.json").write_text(
        json.dumps({"meta": meta, "items": kept}, ensure_ascii=False, indent=2)
    )
    (DATA / "gallery_items.json").write_text(json.dumps(kept, ensure_ascii=False, indent=2))
    print(f"gallery={len(kept)} fingerprint={fp}")


if __name__ == "__main__":
    main()
