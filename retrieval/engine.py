"""Retrieval engine: BM25 / CLIP / attribute rerank / compositional CLIP / hybrid.

Distinct from the teammate visual-only SigLIP2+LoRA+reference-image stack:
our flagship method is *compositional multi-prompt CLIP with soft attribute gates*.
"""

from __future__ import annotations

import json
from pathlib import Path

from .attributes import attribute_score, parse_query
from .bm25_retriever import BM25Retriever
from .clip_retriever import CLIPRetriever
from .compositional import (
    build_prompts,
    evidence_for_item,
    fuse_scores,
    rrf_fuse,
    soft_gate,
)

METHODS = (
    "bm25",
    "clip",
    "clip_rerank",
    "compositional",  # flagship innovation
    "hybrid_rrf",  # BM25 ∪ compositional RRF
)


class RetrievalEngine:
    def __init__(
        self,
        gallery_path: Path,
        index_path: Path,
        model_dir: Path | None = None,
        rerank_alpha: float = 0.15,
        gate_beta: float = 0.35,
        candidate_k: int = 100,
        load_clip: bool = True,
    ):
        self.gallery_path = Path(gallery_path)
        raw = json.loads(self.gallery_path.read_text())
        self.items = raw["items"] if isinstance(raw, dict) and "items" in raw else raw
        self.by_id = {it["id"]: it for it in self.items}
        self.rerank_alpha = rerank_alpha
        self.gate_beta = gate_beta
        self.candidate_k = candidate_k
        self.bm25 = BM25Retriever(self.items)
        self.clip: CLIPRetriever | None = None
        if load_clip:
            self.clip = CLIPRetriever(
                self.items,
                index_path=Path(index_path),
                local_model_dir=model_dir,
            )

    def search(
        self,
        query: str,
        method: str = "compositional",
        top_k: int = 20,
        object_type: str | None = None,
        material: str | None = None,
        period: str | None = None,
        with_evidence: bool = True,
    ) -> list[dict]:
        method = method.lower().strip()
        if method not in METHODS:
            raise ValueError(f"Unknown method {method}; choose from {METHODS}")

        parsed = parse_query(query)
        if method == "bm25":
            ranked = self.bm25.search(query)
        elif method == "clip":
            if self.clip is None:
                raise RuntimeError("CLIP not loaded")
            ranked = self.clip.search(query)
        elif method == "clip_rerank":
            ranked = self._clip_rerank(query, parsed)
        elif method == "compositional":
            ranked = self._compositional(query, parsed)
        else:
            ranked = self._hybrid_rrf(query, parsed)

        results = []
        for doc_id, score in ranked:
            item = self.by_id[doc_id]
            if object_type and item.get("object_type") != object_type:
                continue
            if material and item.get("material") != material:
                continue
            if period and item.get("period") != period:
                continue
            row = dict(item)
            row["score"] = float(score)
            if with_evidence:
                row["evidence"] = evidence_for_item(item, parsed)
                row["parsed_query"] = {
                    "periods": parsed.periods,
                    "materials": parsed.materials,
                    "object_types": parsed.object_types,
                }
            results.append(row)
            if len(results) >= top_k:
                break
        return results

    def _clip_scores(self, text: str) -> dict[str, float]:
        assert self.clip is not None
        return {i: s for i, s in self.clip.search(text)}

    def _clip_rerank(self, query: str, parsed=None) -> list[tuple[str, float]]:
        assert self.clip is not None
        parsed = parsed or parse_query(query)
        base = self.clip.search(query)
        head = base[: self.candidate_k]
        tail = base[self.candidate_k :]
        rescored = []
        for doc_id, cos in head:
            item = self.by_id[doc_id]
            attr = attribute_score(item, parsed)
            score = cos if attr is None else cos + self.rerank_alpha * attr
            rescored.append((doc_id, float(score)))
        rescored.sort(key=lambda x: x[1], reverse=True)
        return rescored + tail

    def _compositional(self, query: str, parsed=None) -> list[tuple[str, float]]:
        """Multi-prompt CLIP fusion + soft attribute gate (flagship)."""
        assert self.clip is not None
        parsed = parsed or parse_query(query)
        prompts = build_prompts(query, parsed)
        channels: dict[str, dict[str, float]] = {"full": self._clip_scores(prompts.full)}
        if prompts.period:
            channels["period"] = self._clip_scores(prompts.period)
        if prompts.material:
            channels["material"] = self._clip_scores(prompts.material)
        if prompts.object_type:
            channels["object_type"] = self._clip_scores(prompts.object_type)
        fused = fuse_scores(channels)
        return soft_gate(fused, self.by_id, parsed, gate_beta=self.gate_beta)

    def _hybrid_rrf(self, query: str, parsed=None) -> list[tuple[str, float]]:
        """RRF of BM25 and compositional lists — metadata+vision without copying teammate stack."""
        parsed = parsed or parse_query(query)
        bm = self.bm25.search(query)
        comp = self._compositional(query, parsed)
        return rrf_fuse([bm, comp], k=60)

    def facets(self) -> dict:
        mats, types, periods = set(), set(), set()
        for it in self.items:
            if it.get("material"):
                mats.add(it["material"])
            if it.get("object_type"):
                types.add(it["object_type"])
            if it.get("period"):
                periods.add(it["period"])
        return {
            "materials": sorted(mats),
            "object_types": sorted(types),
            "periods": sorted(periods),
            "methods": list(METHODS),
            "flagship": "compositional",
        }
