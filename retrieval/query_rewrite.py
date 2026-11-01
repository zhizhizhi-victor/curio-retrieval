"""Bilingual query rewriting: keep original + expand appearance / English aliases."""

from __future__ import annotations

import re

from .attributes import (
    MATERIAL_ALIASES,
    OBJECT_ALIASES,
    PERIOD_ALIASES,
    parse_query,
)

# Chinese cue -> English expansion tokens
ZH_MAP = {
    "青花": "blue and white porcelain",
    "粉彩": "famille rose enamel porcelain",
    "斗彩": "doucai porcelain",
    "梅瓶": "meiping plum vase narrow neck",
    "玉壶春": "yuhuchun pear shaped vase",
    "蒜头瓶": "garlic mouth vase",
    "琮": "cong jade ritual tube",
    "玉璧": "jade bi disk",
    "鼻烟壶": "snuff bottle",
    "青铜": "bronze",
    "瓷器": "porcelain",
    "陶瓷": "ceramic pottery",
    "玉": "jade",
    "漆器": "lacquer",
    "丝织": "silk textile",
    "刺绣": "embroidery textile",
    "康熙": "Kangxi Qing dynasty",
    "乾隆": "Qianlong Qing dynasty",
    "雍正": "Yongzheng Qing dynasty",
    "明代": "Ming dynasty",
    "清代": "Qing dynasty",
    "唐代": "Tang dynasty",
    "宋代": "Song dynasty",
    "元代": "Yuan dynasty",
    "汉代": "Han dynasty",
    "新石器": "Neolithic",
    "瓶": "vase",
    "碗": "bowl",
    "盘": "dish plate",
    "罐": "jar",
    "杯": "cup",
    "壶": "ewer pot",
    "炉": "censer incense burner",
    "镜": "mirror",
    "俑": "figure figurine",
    "绿色": "green",
    "蓝色": "blue",
    "红色": "red",
    "白色": "white",
    "黑色": "black",
    "黄色": "yellow",
    "窄口": "narrow mouth",
    "敞口": "wide mouth",
    "长颈": "long neck",
    "龙纹": "dragon motif",
    "花卉": "floral flower motif",
}


def _alias_terms(parsed) -> list[str]:
    terms = []
    for lab in parsed.materials:
        terms.extend(MATERIAL_ALIASES.get(lab, [lab.lower()])[:2])
    for lab in parsed.object_types:
        terms.extend(OBJECT_ALIASES.get(lab, [lab.lower()])[:2])
    for lab in parsed.periods:
        # prefer dynasty phrasing for vision models
        terms.append(f"{lab} dynasty")
        terms.extend(PERIOD_ALIASES.get(lab, [])[:1])
    return terms


def rewrite_query(text: str) -> dict:
    """Return original, rewritten English-leaning query, and parsed attrs."""
    raw = (text or "").strip()
    parsed = parse_query(raw)
    extras: list[str] = []

    # Chinese lexical expansion
    for zh, en in sorted(ZH_MAP.items(), key=lambda x: -len(x[0])):
        if zh in raw:
            extras.append(en)

    extras.extend(_alias_terms(parsed))

    # Mild English appearance cues already in query stay; add museum-photo phrasing
    extras.append("Chinese artifact museum photograph")

    # Deduplicate while preserving order
    seen = set()
    parts = [raw] if raw else []
    for t in extras:
        key = t.lower().strip()
        if not key or key in seen:
            continue
        # skip if already substring of raw lower
        if key in raw.lower():
            continue
        seen.add(key)
        parts.append(t)

    rewritten = ", ".join(parts)
    return {
        "original": raw,
        "rewritten": rewritten,
        "parsed": {
            "materials": parsed.materials,
            "object_types": parsed.object_types,
            "periods": parsed.periods,
        },
    }


def caption_from_item(item: dict) -> str:
    """Weak supervision caption from metadata (no dynasty leakage into image-only claims)."""
    bits = [
        item.get("object_type") or "",
        item.get("material") or "",
        "Chinese",
        "museum artifact",
    ]
    # Include period as textual supervision for contrastive adaptation
    if item.get("period") and item["period"] not in {"Unknown", "Other"}:
        bits.insert(0, f"{item['period']} dynasty")
    title = (item.get("title") or "").strip()
    if title and len(title) < 80:
        bits.append(title)
    return " ".join(b for b in bits if b)
