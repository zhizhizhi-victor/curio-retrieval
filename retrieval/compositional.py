"""Compositional multi-prompt CLIP: our distinct method vs pure visual systems.

Innovation
----------
Instead of one text embedding (CLIP/SigLIP) or metadata-only BM25, we
*decompose* a natural-language query into attribute-specific prompts
(period / material / object type) plus the full query, score the gallery
against each prompt, then fuse scores. Soft attribute gates further
penalize hard conflicts with parsed constraints.

This is intentionally different from:
- teammate default: single SigLIP2 vector, attribute head weight 0
- teammate reference-image pipeline / DINOv2 / handcrafted shape-color
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .attributes import ParsedQuery, attribute_score, parse_query


@dataclass
class PromptBundle:
    full: str
    period: str | None = None
    material: str | None = None
    object_type: str | None = None


def build_prompts(query: str, parsed: ParsedQuery | None = None) -> PromptBundle:
    parsed = parsed or parse_query(query)
    period = None
    material = None
    obj = None
    if parsed.periods:
        p = parsed.periods[0]
        period = f"a Chinese museum artifact from the {p} dynasty"
    if parsed.materials:
        m = parsed.materials[0].split("/")[0]
        material = f"a Chinese {m.lower()} artifact photograph"
    if parsed.object_types:
        o = parsed.object_types[0].split("/")[0]
        object_type = f"a Chinese {o.lower()} in a museum collection"
        obj = object_type
    return PromptBundle(full=query.strip(), period=period, material=material, object_type=obj)


def fuse_scores(
    score_map: dict[str, dict[str, float]],
    weights: dict[str, float] | None = None,
) -> list[tuple[str, float]]:
    """score_map: channel -> {id: score}; returns ranked (id, fused)."""
    weights = weights or {"full": 0.45, "period": 0.2, "material": 0.2, "object_type": 0.15}
    ids = set()
    for ch in score_map.values():
        ids.update(ch.keys())
    fused = []
    for doc_id in ids:
        total = 0.0
        wsum = 0.0
        for ch, w in weights.items():
            if ch not in score_map:
                continue
            if doc_id in score_map[ch]:
                total += w * score_map[ch][doc_id]
                wsum += w
        if wsum > 0:
            fused.append((doc_id, total / wsum))
    fused.sort(key=lambda x: x[1], reverse=True)
    return fused


def soft_gate(
    ranked: list[tuple[str, float]],
    by_id: dict,
    parsed: ParsedQuery,
    gate_beta: float = 0.35,
) -> list[tuple[str, float]]:
    """Multiply visual/compositional score by a soft attribute agreement factor.

    gate = 1 + beta * attr_score, with attr_score in [-1, 1].
    Missing parsed attrs => no gating (return as-is).
    """
    out = []
    for doc_id, score in ranked:
        item = by_id[doc_id]
        attr = attribute_score(item, parsed)
        if attr is None:
            out.append((doc_id, score))
        else:
            out.append((doc_id, float(score) * (1.0 + gate_beta * attr)))
    out.sort(key=lambda x: x[1], reverse=True)
    return out


def evidence_for_item(item: dict, parsed: ParsedQuery) -> dict:
    """Explainable match / conflict evidence for the UI and talk demos."""

    def check(field: str, wanted: list[str]) -> dict:
        if not wanted:
            return {"status": "unspecified", "wanted": [], "got": item.get(field)}
        got = item.get(field)
        ok = got in wanted
        return {
            "status": "match" if ok else "conflict",
            "wanted": wanted,
            "got": got,
        }

    return {
        "period": check("period", parsed.periods),
        "material": check("material", parsed.materials),
        "object_type": check("object_type", parsed.object_types),
    }


def graded_relevance(item: dict, constraints: dict) -> float:
    """Graded label: fraction of declared constraints satisfied (0..1)."""
    keys = [k for k, v in constraints.items() if v]
    if not keys:
        return 0.0
    hit = sum(1 for k in keys if item.get(k) == constraints[k])
    return hit / len(keys)


def rrf_fuse(
    rankings: list[list[tuple[str, float]]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion across ranked lists (ids, scores ignored except order)."""
    scores: dict[str, float] = {}
    for ranked in rankings:
        for rank, (doc_id, _) in enumerate(ranked, 1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
