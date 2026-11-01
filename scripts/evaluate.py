#!/usr/bin/env python3
"""Evaluate methods with strict mAP and graded nDCG@10."""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from retrieval.compositional import graded_relevance
from retrieval.engine import METHODS, RetrievalEngine
from retrieval.metrics import (
    average_precision,
    mean_average_precision,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)

DATA = ROOT / "data"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)


def eval_method(engine: RetrievalEngine, queries: list[dict], method: str, items_by_id: dict) -> dict:
    rows = []
    aps, p5s, p10s, r10s, ndcgs = [], [], [], [], []
    for q in queries:
        relevant = set(q["relevant_ids"])
        constraints = q.get("constraints") or {}
        graded = {
            iid: graded_relevance(items_by_id[iid], constraints)
            for iid in items_by_id
            if graded_relevance(items_by_id[iid], constraints) > 0
        }
        # ensure full matches are in graded
        for rid in relevant:
            graded[rid] = 1.0

        ranked = engine.search(q["text"], method=method, top_k=len(engine.items), with_evidence=False)
        ranked_ids = [r["id"] for r in ranked]
        ap = average_precision(ranked_ids, relevant)
        p5 = precision_at_k(ranked_ids, relevant, 5)
        p10 = precision_at_k(ranked_ids, relevant, 10)
        r10 = recall_at_k(ranked_ids, relevant, 10)
        ndcg = ndcg_at_k(ranked_ids, graded, 10)
        aps.append(ap)
        p5s.append(p5)
        p10s.append(p10)
        r10s.append(r10)
        ndcgs.append(ndcg)
        rows.append(
            {
                "id": q["id"],
                "text": q["text"],
                "n_relevant": len(relevant),
                "AP": ap,
                "P@5": p5,
                "P@10": p10,
                "R@10": r10,
                "nDCG@10": ndcg,
                "top5": "|".join(ranked_ids[:5]),
            }
        )
    summary = {
        "method": method,
        "n_queries": len(queries),
        "mAP": mean_average_precision(aps),
        "P@5": float(np.mean(p5s)),
        "P@10": float(np.mean(p10s)),
        "R@10": float(np.mean(r10s)),
        "nDCG@10": float(np.mean(ndcgs)),
    }
    return {"summary": summary, "per_query": rows}


def main() -> None:
    payload = json.loads((DATA / "queries.json").read_text())
    queries = payload["queries"]
    val = [q for q in queries if q["split"] == "validation"]
    test = [q for q in queries if q["split"] == "test"]

    engine = RetrievalEngine(
        gallery_path=DATA / "gallery.json",
        index_path=DATA / "index.npz",
        model_dir=ROOT / "models" / "clip-vit-base-patch32",
        load_clip=True,
    )
    items_by_id = engine.by_id

    # tune alpha / beta on validation using compositional nDCG
    best = {"alpha": 0.15, "beta": 0.35, "score": -1.0}
    grid = []
    for alpha in (0.1, 0.15, 0.2):
        for beta in (0.25, 0.35, 0.5):
            engine.rerank_alpha = alpha
            engine.gate_beta = beta
            res = eval_method(engine, val, "compositional", items_by_id)
            s = res["summary"]["nDCG@10"]
            grid.append({"alpha": alpha, "beta": beta, "val_nDCG@10": s})
            print(f"val compositional α={alpha} β={beta} nDCG@10={s:.4f}")
            if s > best["score"]:
                best = {"alpha": alpha, "beta": beta, "score": s}

    engine.rerank_alpha = best["alpha"]
    engine.gate_beta = best["beta"]
    (RESULTS / "selected_config.json").write_text(
        json.dumps({"flagship": "compositional", **best, "grid": grid}, indent=2)
    )

    methods = list(METHODS)
    summaries = []
    all_per_query = []
    full = {"selected": best, "validation_grid": grid, "test": {}}
    for method in methods:
        res = eval_method(engine, test, method, items_by_id)
        summaries.append(res["summary"])
        for row in res["per_query"]:
            row = dict(row)
            row["method"] = method
            all_per_query.append(row)
        full["test"][method] = res
        s = res["summary"]
        print(
            f"TEST {method}: mAP={s['mAP']:.4f} P@5={s['P@5']:.4f} "
            f"nDCG@10={s['nDCG@10']:.4f} R@10={s['R@10']:.4f}"
        )

    with (RESULTS / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f, fieldnames=["method", "n_queries", "mAP", "P@5", "P@10", "R@10", "nDCG@10"]
        )
        w.writeheader()
        w.writerows(summaries)

    with (RESULTS / "per_query.csv").open("w", newline="", encoding="utf-8") as f:
        fields = [
            "method",
            "id",
            "text",
            "n_relevant",
            "AP",
            "P@5",
            "P@10",
            "R@10",
            "nDCG@10",
            "top5",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(all_per_query)

    full["metrics"] = summaries
    full["gallery_fingerprint"] = json.loads((DATA / "gallery.json").read_text())["meta"][
        "fingerprint"
    ]
    full["n_gallery"] = len(engine.items)
    full["n_val"] = len(val)
    full["n_test"] = len(test)
    full["differentiation"] = {
        "ours": "compositional multi-prompt CLIP + soft attribute gates + graded nDCG",
        "not_copied": [
            "SigLIP2 + image LoRA as default",
            "external reference-image pipeline",
            "DINOv2 / handcrafted shape-color-texture",
            "attribute head weight forced to 0",
        ],
    }
    (RESULTS / "evaluation.json").write_text(json.dumps(full, ensure_ascii=False, indent=2))

    md = [
        "# Curio Retrieval Experiment Report (Attribute-Compositional)",
        "",
        f"Gallery: {full['n_gallery']}; val: {full['n_val']}; test: {full['n_test']}.",
        f"Selected compositional α={best['alpha']}, gate β={best['beta']} (val nDCG@10={best['score']:.4f}).",
        "",
        "| Method | P@5 | P@10 | R@10 | mAP | nDCG@10 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for s in summaries:
        md.append(
            f"| {s['method']} | {s['P@5']:.4f} | {s['P@10']:.4f} | {s['R@10']:.4f} | "
            f"{s['mAP']:.4f} | {s['nDCG@10']:.4f} |"
        )
    md += [
        "",
        "## Differentiation",
        "- Flagship: **compositional multi-prompt CLIP** (period/material/type prompts fused) + soft gates.",
        "- Graded nDCG rewards partial attribute matches; strict mAP still uses full-constraint positives.",
        "- We do **not** copy teammate SigLIP2-LoRA / reference-image / DINO pipelines.",
        "",
    ]
    (RESULTS / "experiment_report.md").write_text("\n".join(md), encoding="utf-8")
    print("wrote results/evaluation.json")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"elapsed {time.time() - t0:.1f}s")
