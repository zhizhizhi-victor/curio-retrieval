# Methodology

## Domain

Chinese museum collections from Met + Cleveland open-access / CC0 records.

## Relevance

**Strict:** every declared `period` / `material` / `object_type` constraint must match metadata.  
**Graded (nDCG):** fraction of declared constraints satisfied ∈ [0, 1].

## Methods

1. **BM25** — English metadata keywords  
2. **CLIP** — ViT-B/32 cosine  
3. **CLIP + attribute rerank** — `score = cos + α · attr_score`  
4. **Compositional (flagship)** — encode full + attribute-specific prompts; fuse; soft-gate with β  
5. **Hybrid RRF** — fuse BM25 and compositional rankings  

Differentiation: we do **not** use SigLIP2-LoRA, external reference-image pipelines, or DINOv2 as the primary story.

## Metrics

P@5, P@10, R@10, mAP, nDCG@10. Hyperparameters (α, β) chosen on validation only.
