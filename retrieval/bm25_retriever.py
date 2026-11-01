"""BM25 keyword baseline over artifact metadata text."""

from __future__ import annotations

import re

from rank_bm25 import BM25Okapi


def tokenize(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s\-]", " ", text)
    toks = [t for t in text.split() if t]
    # light plural normalization
    out = []
    for t in toks:
        if t.endswith("ies") and len(t) > 4:
            out.append(t[:-3] + "y")
        elif t.endswith("ses") and len(t) > 4:
            out.append(t[:-2])
        elif t.endswith("s") and not t.endswith("ss") and len(t) > 3:
            out.append(t[:-1])
        else:
            out.append(t)
    return out


def item_document(item: dict) -> str:
    parts = [
        item.get("title") or "",
        item.get("object_type") or "",
        item.get("object_type_raw") or "",
        item.get("material") or "",
        item.get("material_raw") or "",
        item.get("period") or "",
        item.get("period_raw") or "",
        item.get("culture") or "",
        item.get("classification") or "",
        item.get("department") or "",
        item.get("artist") or "",
    ]
    return " ".join(p for p in parts if p)


class BM25Retriever:
    def __init__(self, items: list[dict]):
        self.items = items
        self.ids = [it["id"] for it in items]
        corpus = [tokenize(item_document(it)) for it in items]
        self.bm25 = BM25Okapi(corpus)

    def search(self, query: str, top_k: int | None = None) -> list[tuple[str, float]]:
        scores = self.bm25.get_scores(tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: float(scores[i]), reverse=True)
        if top_k is not None:
            order = order[:top_k]
        return [(self.ids[i], float(scores[i])) for i in order]
