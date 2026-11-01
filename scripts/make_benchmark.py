#!/usr/bin/env python3
"""Create English validation/test queries with automatic relevance labels."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

TEMPLATES = [
    "{period} {material} {object}",
    "{period} {object} made of {material}",
    "a {material} {object} from the {period} dynasty",
    "Find {period} {material} {object}s",
    "I am looking for a {period} {object} in {material}",
    "Show me {material} {object}s from the {period} period",
]

# Bilingual lexicon queries (our differentiation: EN+ZH attribute parsing)
ZH_PERIOD = {
    "Qing": "清代",
    "Ming": "明代",
    "Tang": "唐代",
    "Song": "宋代",
    "Han": "汉代",
    "Yuan": "元代",
}
ZH_MATERIAL = {
    "Porcelain": "瓷",
    "Jade": "玉",
    "Bronze": "青铜",
    "Lacquer": "漆",
    "Stoneware": "炻器",
}
ZH_OBJECT = {
    "Vase": "瓶",
    "Bowl": "碗",
    "Jar": "罐",
    "Figure": "俑",
    "Mirror": "镜",
    "Cup": "杯",
    "Dish": "盘",
}

SINGLE_TEMPLATES = {
    "period": [
        "{period} dynasty artifacts",
        "works from the {period} period",
        "Show me {period} pieces",
    ],
    "material": [
        "{material} artifacts",
        "objects made of {material}",
        "Find {material} works in the collection",
    ],
    "object_type": [
        "{object}s in the collection",
        "Show me {object}s",
        "I am looking for a {object}",
    ],
}


def relevant_ids(items: list[dict], constraints: dict) -> list[str]:
    out = []
    for it in items:
        ok = True
        for k, v in constraints.items():
            if v and it.get(k) != v:
                ok = False
                break
        if ok:
            out.append(it["id"])
    return out


def main(seed: int = 5523) -> None:
    rng = random.Random(seed)
    gallery = json.loads((DATA / "gallery.json").read_text())
    items = gallery["items"]

    # Count combinations with enough positives
    combos = defaultdict(list)
    for it in items:
        key = (it.get("period"), it.get("material"), it.get("object_type"))
        if all(key) and key[0] not in {"Unknown", "Other"} and key[1] != "Other":
            combos[key].append(it["id"])

    viable = [(k, ids) for k, ids in combos.items() if len(ids) >= 2]
    viable.sort(key=lambda x: (-len(x[1]), x[0]))
    rng.shuffle(viable)

    queries = []
    used_triple = set()

    def add_query(qid: str, split: str, text: str, constraints: dict, qtype: str) -> bool:
        rel = relevant_ids(items, constraints)
        if len(rel) < 2:
            return False
        queries.append(
            {
                "id": qid,
                "split": split,
                "type": qtype,
                "text": text,
                "constraints": constraints,
                "relevant_ids": rel,
            }
        )
        return True

    # Triple constraints: 8 val + 24 test
    triples = [k for k, _ in viable]
    rng.shuffle(triples)
    for i, key in enumerate(triples):
        if key in used_triple:
            continue
        period, material, obj = key
        # readable material/object
        mat_word = material.split("/")[0].lower()
        obj_word = obj.split("/")[0].lower()
        if obj_word.endswith("s"):
            obj_word = obj_word  # keep
        tmpl = TEMPLATES[len(used_triple) % len(TEMPLATES)]
        text = tmpl.format(period=period, material=mat_word, object=obj_word)
        split = "validation" if len([q for q in queries if q["split"] == "validation" and q["type"] == "triple"]) < 8 else "test"
        if split == "test" and len([q for q in queries if q["split"] == "test" and q["type"] == "triple"]) >= 24:
            continue
        if split == "validation" and len([q for q in queries if q["split"] == "validation" and q["type"] == "triple"]) >= 8:
            split = "test"
            if len([q for q in queries if q["split"] == "test" and q["type"] == "triple"]) >= 24:
                continue
        qid = f"q_triple_{len(used_triple):03d}"
        if add_query(
            qid,
            split,
            text,
            {"period": period, "material": material, "object_type": obj},
            "triple",
        ):
            used_triple.add(key)

    # Single-dimension queries
    def collect_single(field: str, n_val: int, n_test: int) -> None:
        counts = defaultdict(list)
        for it in items:
            v = it.get(field)
            if v and v not in {"Unknown", "Other"}:
                counts[v].append(it["id"])
        labels = [k for k, ids in counts.items() if len(ids) >= 5]
        rng.shuffle(labels)
        for i, lab in enumerate(labels):
            split = "validation" if i < n_val else "test"
            if split == "test" and i >= n_val + n_test:
                break
            if field == "period":
                text = SINGLE_TEMPLATES["period"][i % 3].format(period=lab)
                constraints = {"period": lab}
                word = lab
            elif field == "material":
                word = lab.split("/")[0]
                text = SINGLE_TEMPLATES["material"][i % 3].format(material=word.lower())
                constraints = {"material": lab}
            else:
                word = lab.split("/")[0]
                text = SINGLE_TEMPLATES["object_type"][i % 3].format(object=word.lower())
                constraints = {"object_type": lab}
            add_query(f"q_{field}_{i:03d}", split, text, constraints, f"single_{field}")

    collect_single("period", 4, 12)
    collect_single("material", 4, 12)
    collect_single("object_type", 4, 12)

    # Chinese lexicon queries (test only) — attribute parser must handle ZH aliases
    zh_n = 0
    for key, ids in viable:
        if zh_n >= 8:
            break
        period, material, obj = key
        zp = ZH_PERIOD.get(period)
        zm = ZH_MATERIAL.get(material.split("/")[0])
        zo = ZH_OBJECT.get(obj.split("/")[0])
        if not (zp and zm and zo):
            continue
        text = f"{zp}{zm}{zo}"
        if add_query(
            f"q_zh_{zh_n:03d}",
            "test",
            text,
            {"period": period, "material": material, "object_type": obj},
            "zh_triple",
        ):
            zh_n += 1

    # Ensure counts
    val = [q for q in queries if q["split"] == "validation"]
    test = [q for q in queries if q["split"] == "test"]
    print(f"queries: total={len(queries)} val={len(val)} test={len(test)} zh={zh_n}")

    payload = {
        "meta": {
            "gallery_fingerprint": gallery["meta"]["fingerprint"],
            "seed": seed,
            "language": "en+zh-lexicon",
            "relevance": "An item is relevant iff all listed constraints match its metadata exactly.",
            "method_family": "attribute-compositional",
        },
        "queries": queries,
    }
    (DATA / "queries.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2))

    # CSV for human review
    import csv

    with (DATA / "queries_for_review.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "split",
                "type",
                "text",
                "constraints",
                "n_relevant",
                "relevant_ids",
                "reviewed",
                "corrected_relevant_ids",
                "reviewer",
                "notes",
            ],
        )
        w.writeheader()
        for q in queries:
            w.writerow(
                {
                    "id": q["id"],
                    "split": q["split"],
                    "type": q["type"],
                    "text": q["text"],
                    "constraints": json.dumps(q["constraints"]),
                    "n_relevant": len(q["relevant_ids"]),
                    "relevant_ids": "|".join(q["relevant_ids"]),
                    "reviewed": "no",
                    "corrected_relevant_ids": "",
                    "reviewer": "",
                    "notes": "",
                }
            )
    print("wrote data/queries.json and data/queries_for_review.csv")


if __name__ == "__main__":
    main()
