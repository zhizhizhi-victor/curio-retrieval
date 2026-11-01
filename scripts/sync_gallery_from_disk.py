#!/usr/bin/env python3
"""Rebuild gallery.json from already-downloaded images (cap ~1200, not teammate-scale)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
IMAGES = DATA / "images"
FILTERED = DATA / "filtered"


def main(target: int = 1200) -> None:
    pool = {r["id"]: r for r in json.loads((FILTERED / "merged_china_openaccess.json").read_text())}
    old = {}
    if (DATA / "gallery.json").exists():
        for it in json.loads((DATA / "gallery.json").read_text()).get("items", []):
            old[it["id"]] = it

    files = sorted(IMAGES.glob("*.jpg"))
    kept = []
    for f in files:
        rid = f.stem
        base = old.get(rid) or pool.get(rid)
        if not base:
            continue
        item = dict(base)
        item["id"] = rid
        item["image_path"] = f"images/{f.name}"
        kept.append(item)
        if len(kept) >= target:
            break

    fp = hashlib.sha256(json.dumps([x["id"] for x in kept]).encode()).hexdigest()[:16]
    meta = {
        "count": len(kept),
        "fingerprint": fp,
        "sources": sorted({x.get("source", "?") for x in kept}),
        "domain": "Chinese museum collections (Met + Cleveland Open Access)",
        "license_note": "Public domain / CC0",
        "relevance_dimensions": ["object_type", "material", "period"],
        "version": "v2-attribute-compositional",
        "note": "Sized for attribute-balanced experiments; not competing on raw gallery scale.",
    }
    (DATA / "gallery.json").write_text(
        json.dumps({"meta": meta, "items": kept}, ensure_ascii=False, indent=2)
    )
    (DATA / "gallery_items.json").write_text(json.dumps(kept, ensure_ascii=False, indent=2))
    print(f"synced gallery={len(kept)} fingerprint={fp}")


if __name__ == "__main__":
    main()
