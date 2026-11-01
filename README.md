# Curio · Attribute-Compositional Museum Retrieval

Text-to-image retrieval over **Chinese museum collections** (Met + Cleveland open access / CC0).

**Flagship method:** compositional multi-prompt CLIP + soft attribute gates.

Gallery images and the CLIP index are included so teammates can run the demo without re-downloading the collection. Large CLIP weight files are **not** in git (auto-downloaded from Hugging Face on first run).

## Setup (teammates)

```bash
git clone <THIS_REPO_URL> curio-retrieval
cd curio-retrieval

# Python 3.10+ recommended
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# First CLIP load needs network once (downloads openai/clip-vit-base-patch32)
python app.py --port 8765 --preload
```

Open **http://127.0.0.1:8765**.

### Optional: rebuild / re-evaluate

```bash
python scripts/build_index.py      # only if gallery images change
python scripts/make_benchmark.py
python scripts/evaluate.py
```

## Methods

1. **BM25** — metadata keyword baseline  
2. **CLIP** — ViT-B/32 cosine  
3. **CLIP + attribute rerank** — `cos + α · attr`  
4. **Compositional (flagship)** — multi-prompt fusion + soft gate β  
5. **Hybrid RRF** — BM25 ∪ compositional  

Queries: English + Chinese lexicon (e.g. `清代瓷瓶`).  

**完整说明书：** [`docs/系统说明书.md`](docs/系统说明书.md)（模型、数据、效果、安装与界面）  
其他：`docs/PROJECT_REPORT.md`、`docs/汇报稿.md`、`docs/METHODOLOGY.md`。

## Repo layout

```text
app.py                 Flask API + UI
static/                Front-end
retrieval/             BM25 / CLIP / compositional / metrics
data/gallery.json      Gallery metadata
data/images/           Local CC0 images
data/index.npz         Precomputed CLIP image embeddings
data/queries.json      Val/test queries
results/               Evaluation outputs
docs/                  Report + talk notes
```

## License note

Images/metadata: Met / Cleveland open access (Public Domain / CC0), for coursework / non-commercial research. Code: for the COMP5523 group project.
