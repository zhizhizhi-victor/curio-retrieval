#!/usr/bin/env python3
"""Web UI for attribute-compositional Chinese museum retrieval (EN + ZH queries)."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"

app = Flask(__name__, static_folder="static", static_url_path="/static")
ENGINE = None


def get_engine():
    global ENGINE
    if ENGINE is None:
        from retrieval.engine import RetrievalEngine

        alpha, beta = 0.15, 0.35
        cfg = RESULTS / "selected_config.json"
        if cfg.exists():
            c = json.loads(cfg.read_text())
            alpha = c.get("alpha", alpha)
            beta = c.get("beta", beta)
        ENGINE = RetrievalEngine(
            gallery_path=DATA / "gallery.json",
            index_path=DATA / "index.npz",
            model_dir=ROOT / "models" / "clip-vit-base-patch32",
            rerank_alpha=alpha,
            gate_beta=beta,
            load_clip=True,
        )
    return ENGINE


@app.get("/")
def index():
    return send_from_directory(ROOT / "static", "index.html")


@app.get("/api/health")
def health():
    g = json.loads((DATA / "gallery.json").read_text())
    return jsonify(
        {
            "ok": True,
            "gallery_size": g["meta"]["count"],
            "fingerprint": g["meta"]["fingerprint"],
            "version": g["meta"].get("version"),
            "flagship": "compositional",
            "language": "en+zh-lexicon",
        }
    )


@app.get("/api/facets")
def facets():
    return jsonify(get_engine().facets())


@app.post("/api/search")
def search():
    body = request.get_json(force=True) or {}
    query = (body.get("query") or "").strip()
    if not query:
        return jsonify({"error": "query is required"}), 400
    method = (body.get("method") or "compositional").strip()
    top_k = int(body.get("top_k") or 20)
    t0 = time.time()
    engine = get_engine()
    results = engine.search(
        query,
        method=method,
        top_k=top_k,
        object_type=body.get("object_type") or None,
        material=body.get("material") or None,
        period=body.get("period") or None,
        with_evidence=True,
    )
    for r in results:
        r["thumb_url"] = "/media/" + Path(r["image_path"]).name
    return jsonify(
        {
            "query": query,
            "method": method,
            "latency_ms": round((time.time() - t0) * 1000, 1),
            "results": results,
        }
    )


@app.get("/media/<path:filename>")
def media(filename: str):
    return send_from_directory(DATA / "images", filename)


@app.get("/api/metrics")
def metrics():
    path = RESULTS / "evaluation.json"
    if not path.exists():
        return jsonify({"error": "Run scripts/evaluate.py first"}), 404
    return jsonify(json.loads(path.read_text()))


@app.get("/api/queries")
def queries():
    path = DATA / "queries.json"
    if not path.exists():
        return jsonify({"queries": []})
    return jsonify(json.loads(path.read_text()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--preload", action="store_true")
    args = parser.parse_args()
    if args.preload:
        get_engine()
    print(f"Curio Compositional Retrieval  http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
