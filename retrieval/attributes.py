"""English attribute lexicon for Chinese museum artifact queries."""

from __future__ import annotations

import re
from dataclasses import dataclass


MATERIAL_ALIASES: dict[str, list[str]] = {
    "Porcelain": ["porcelain", "china ware", "chinaware", "blue and white", "瓷器", "瓷", "青花"],
    "Jade": ["jade", "nephrite", "jadeite", "玉", "玉器"],
    "Bronze": ["bronze", "bronze ware", "bronzes", "青铜", "铜器"],
    "Stoneware": ["stoneware", "炻器"],
    "Earthenware": ["earthenware", "terracotta", "terra cotta", "陶器", "陶"],
    "Ceramics": ["ceramic", "ceramics", "pottery", "陶瓷"],
    "Silk/Textile": ["silk", "textile", "tapestry", "embroidery", "brocade", "robe", "丝绸", "丝织", "刺绣"],
    "Enamel": ["enamel", "cloisonne", "cloisonné", "珐琅", "掐丝珐琅"],
    "Lacquer": ["lacquer", "lacquerware", "漆器", "漆"],
    "Ivory": ["ivory", "象牙"],
    "Metal": ["metal", "gold", "silver", "copper", "iron", "金", "银"],
    "Wood": ["wood", "wooden", "bamboo", "木", "竹"],
    "Paper/Painting": ["painting", "scroll", "ink painting", "album leaf", "书画", "绘画"],
}

OBJECT_ALIASES: dict[str, list[str]] = {
    "Vase": ["vase", "vases", "瓶", "梅瓶"],
    "Bowl": ["bowl", "bowls", "碗"],
    "Dish": ["dish", "dishes", "plate", "plates", "盘", "碟"],
    "Jar": ["jar", "jars", "urn", "罐"],
    "Cup": ["cup", "cups", "beaker", "wine cup", "杯"],
    "Bottle": ["bottle", "bottles", "瓶"],
    "Figure": ["figure", "figurine", "sculpture", "statue", "俑", "人像"],
    "Snuff bottle": ["snuff bottle", "snuff bottles", "鼻烟壶"],
    "Censer": ["censer", "incense burner", "炉", "香炉"],
    "Mirror": ["mirror", "mirrors", "镜", "铜镜"],
    "Box": ["box", "boxes", "盒"],
    "Ewer": ["ewer", "ewers", "壶"],
    "Teapot": ["teapot", "tea pot", "茶壶"],
    "Vessel": ["vessel", "vessels", "pot", "器"],
    "Textile/Panel": ["textile", "panel", "silk panel", "robe", "garment"],
    "Cong/Jade ritual": ["cong", "jade cong", "玉琮", "琮"],
    "Amulet": ["amulet", "pendant", "talisman", "佩"],
    "Scholar object": ["brush rest", "inkstone", "scholar object", "brush pot"],
    "Plaque": ["plaque"],
    "Painting/Scroll": ["painting", "scroll", "hanging scroll"],
}

PERIOD_ALIASES: dict[str, list[str]] = {
    "Neolithic": ["neolithic", "liangzhu", "新石器", "良渚"],
    "Shang": ["shang", "shang dynasty", "商", "商代"],
    "Zhou": ["zhou", "zhou dynasty", "eastern zhou", "western zhou", "周", "周代"],
    "Han": ["han", "han dynasty", "eastern han", "western han", "汉", "汉代"],
    "Tang": ["tang", "tang dynasty", "唐", "唐代"],
    "Song": ["song", "song dynasty", "northern song", "southern song", "宋", "宋代"],
    "Yuan": ["yuan", "yuan dynasty", "元", "元代"],
    "Ming": ["ming", "ming dynasty", "明", "明代"],
    "Qing": ["qing", "qing dynasty", "kangxi", "qianlong", "yongzheng", "清", "清代", "康熙", "乾隆", "雍正"],
    "Jin": ["jin dynasty", "金代"],
    "Liao": ["liao", "liao dynasty", "辽"],
    "Sui": ["sui", "sui dynasty", "隋"],
    "Republic": ["republic period", "republican period", "民国"],
}


@dataclass
class ParsedQuery:
    materials: list[str]
    object_types: list[str]
    periods: list[str]
    raw: str


def _find_labels(text: str, alias_map: dict[str, list[str]]) -> list[str]:
    t = text.lower()
    found = []
    for label, aliases in alias_map.items():
        for alias in sorted(aliases, key=len, reverse=True):
            # Chinese aliases have no word boundaries; English uses \b
            if re.search(r"[\u4e00-\u9fff]", alias):
                hit = alias in text
            else:
                hit = bool(re.search(rf"\b{re.escape(alias.lower())}\b", t))
            if hit:
                found.append(label)
                break
    out = []
    for x in found:
        if x not in out:
            out.append(x)
    return out


def parse_query(text: str) -> ParsedQuery:
    return ParsedQuery(
        materials=_find_labels(text, MATERIAL_ALIASES),
        object_types=_find_labels(text, OBJECT_ALIASES),
        periods=_find_labels(text, PERIOD_ALIASES),
        raw=text,
    )


def attribute_score(item: dict, parsed: ParsedQuery) -> float | None:
    """Average of +1 match / -1 conflict over recognized constraints. None if none parsed."""
    dims = []
    if parsed.materials:
        dims.append(1.0 if item.get("material") in parsed.materials else -1.0)
    if parsed.object_types:
        dims.append(1.0 if item.get("object_type") in parsed.object_types else -1.0)
    if parsed.periods:
        dims.append(1.0 if item.get("period") in parsed.periods else -1.0)
    if not dims:
        return None
    return sum(dims) / len(dims)


def item_matches_constraints(item: dict, constraints: dict) -> bool:
    """Strict relevance: all declared constraints must match."""
    for key in ("material", "object_type", "period"):
        want = constraints.get(key)
        if want and item.get(key) != want:
            return False
    return True
