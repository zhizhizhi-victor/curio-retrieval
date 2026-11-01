# Curio Retrieval Experiment Report (Attribute-Compositional)

Gallery: 1052; val: 20; test: 64.
Selected compositional α=0.1, gate β=0.5 (val nDCG@10=0.7750).

| Method | P@5 | P@10 | R@10 | mAP | nDCG@10 |
|---|---:|---:|---:|---:|---:|
| bm25 | 0.5594 | 0.4719 | 0.5032 | 0.6617 | 0.6986 |
| clip | 0.1750 | 0.1625 | 0.0899 | 0.1411 | 0.3151 |
| clip_rerank | 0.4688 | 0.3656 | 0.2905 | 0.3791 | 0.6740 |
| compositional | 0.6469 | 0.5500 | 0.5807 | 0.8149 | 0.8551 |
| hybrid_rrf | 0.6594 | 0.5734 | 0.5641 | 0.7834 | 0.8748 |

## Differentiation
- Flagship: **compositional multi-prompt CLIP** (period/material/type prompts fused) + soft gates.
- Graded nDCG rewards partial attribute matches; strict mAP still uses full-constraint positives.
- We do **not** copy teammate SigLIP2-LoRA / reference-image / DINO pipelines.
