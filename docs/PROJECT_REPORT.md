# Curio：中国博物馆馆藏自然语言图像检索系统

## 项目报告（COMP5523）

**系统名称：** Curio · Attribute-Compositional Museum Retrieval  
**任务类型：** Domain-Specific Text-to-Image Retrieval  
**查询语言：** English + 中文属性词表（lexicon）  
**日期：** 2026-11-01  

---

## 1. 摘要

本项目实现面向**中国博物馆馆藏**的文字→图片检索：Query → 固定 Gallery → Ranked results。相关性按 **时期 × 材质 × 器型** 多属性同时约束定义。

相对纯视觉路线（如组员侧常见的 SigLIP2+LoRA / 参考图 / DINOv2），本系统的**差异化贡献**是：

1. **属性组合式多提示 CLIP（Compositional）**：把查询拆成 full + period + material + object_type 多路文本嵌入，再融合；  
2. **软属性门控（soft gate）**：对冲突属性降权、对匹配升权；  
3. **可解释 evidence**：每条结果标出 period/material/type 的 match / conflict；  
4. **分级 nDCG**：除严格 mAP 外，奖励部分属性命中；  
5. **中英词表解析**：支持如「清代瓷瓶」的属性抽取。

在更新后的图库（1052）与测试集上，**compositional** 的 mAP / nDCG@10 明显高于纯 CLIP 与 CLIP+重排，并超过 BM25 基线。

---

## 2. 与组员方案的差异（汇报重点）

| 维度 | 本组（Curio） | 组员常见路线（对照，不复制） |
|---|---|---|
| 主模型故事 | CLIP ViT-B/32 + **多提示属性组合** | SigLIP2 + 图像 LoRA 等 |
| 属性用法 | **显式多通道提示 + soft gate + 可解释 evidence** | 属性头权重常置 0 / 弱使用 |
| 参考图 | 不做外部分类参考图管线 | 参考图 / 手写形状颜色纹理等 |
| 视觉骨干 | 不引入 DINOv2 作为主卖点 | DINOv2 等额外视觉特征 |
| 图库策略 | ~1k 级、属性均衡实验 | 冲更大 gallery 规模 |
| 元数据 | BM25 为**显式基线**；hybrid RRF 可选融合 | 元数据常不作为独立可对比轴 |
| 评价 | 严格 mAP + **graded nDCG@10** | 以视觉召回指标为主时叙事不同 |

**一句话：** 我们讲「属性如何被分解、融合、解释」，不讲「换更大视觉模型 / 微调 / 冲规模」。

---

## 3. 问题定义

| 部分 | 含义 |
|---|---|
| **Query** | 英文或中文属性词，如 `Qing porcelain vase` / `清代瓷瓶` |
| **Gallery** | 1052 张中国相关开放获取馆藏图（Met + Cleveland CC0） |
| **Ranked results** | 按相关程度排序的列表，并附 evidence |

**严格相关：** 查询解析出的每个约束（period / material / object_type）都必须与条目元数据精确相等。  
**分级相关：** 命中约束比例 ∈ [0,1]，用于 nDCG。

---

## 4. 数据

- 来源：Met Open Access、Cleveland Open Access（CC0 / Public Domain）。  
- 过滤：中国相关 + 有图 + 许可可用；材质/器型/时期规范化。  
- 规模：磁盘已下载图像同步入库 **1052**（指纹见 `gallery.json`），不做与组员拼规模竞赛。  
- 评价集：`scripts/make_benchmark.py` 生成；含英文模板查询 + **8 条中文三元组查询**；val/test 划分。

---

## 5. 方法

### 5.1 BM25（元数据基线）

标题与属性字段拼文档，关键词排序。不看像素。

### 5.2 CLIP

`openai/clip-vit-base-patch32`，图离线索引，查询余弦相似度。

### 5.3 CLIP + attribute rerank

Top-100 CLIP 候选上：`score = cos + α · attr_score`。

### 5.4 Compositional CLIP（旗舰）

1. 解析属性 → 构造多提示：`full`、`period`、`material`、`object_type`；  
2. 各通道与全库做 CLIP 相似度；  
3. 加权融合；  
4. soft gate：`score' = score · (1 + β · attr_score)`。  

验证集上按 compositional 的 nDCG@10 选 β（本轮 **β=0.5**；α 主要影响 rerank）。

### 5.5 Hybrid RRF

BM25 列表 ∪ compositional 列表做 Reciprocal Rank Fusion（k=60）。

---

## 6. 实验结果

图库 **1052**；验证 **20**；测试 **64**（含中文查询）；选定 gate **β=0.5**。

| Method | P@5 | P@10 | R@10 | mAP | nDCG@10 |
|---|---:|---:|---:|---:|---:|
| BM25 | 0.5594 | 0.4719 | 0.5032 | 0.6617 | 0.6986 |
| CLIP | 0.1750 | 0.1625 | 0.0899 | 0.1411 | 0.3151 |
| CLIP + rerank | 0.4688 | 0.3656 | 0.2905 | 0.3791 | 0.6740 |
| **compositional** | **0.6469** | **0.5500** | **0.5807** | **0.8149** | 0.8551 |
| hybrid_rrf | 0.6594 | 0.5734 | 0.5641 | 0.7834 | **0.8748** |

（完整小数见 `results/metrics.csv` / `evaluation.json`。）

### 解读

1. **Compositional 明显优于纯 CLIP / 重排**，说明多提示分解 + 门控比「单向量 + 事后加一点属性」更贴合多属性查询。  
2. **Compositional 超过 BM25（mAP）**，hybrid 在 nDCG 上略高——视觉通道与元数据通道可互补。  
3. **诚实边界：** 评价标签仍来自元数据属性；属性门控与 GT 同源，会抬高属性方法。演示时应强调「可解释约束满足」，并抽查失败 case。  
4. 与组员对比时强调：**路线不同、指标叙事不同**，不是简单复现对方的 SigLIP/LoRA/参考图。

---

## 7. 系统与演示

```bash
cd curio-retrieval
../.venv/bin/python app.py --port 8765 --preload
# http://127.0.0.1:8765
```

- Search：默认 **Compositional**；可切换 hybrid / rerank / CLIP / BM25；结果带 match/conflict chips。  
- Experiments：展示含 nDCG 的指标表。  
- About：写明与纯视觉路线的差异。

---

## 8. 局限

弱标签；属性同源偏差；细粒度纹饰未进硬标签；未做 SigLIP/LoRA（有意不做，以保持差异化）。

---

## 9. 结论

Curio 的个人汇报主线是 **attribute-compositional retrieval**：多提示 CLIP、软门控、分级 nDCG、中英词表与可解释证据。在同一领域任务上与组员拉开方法差异，而不是复制其视觉微调与规模叙事。

---

## 附录：关键路径

| 路径 | 内容 |
|---|---|
| `retrieval/compositional.py` | 多提示融合 / soft gate / evidence / RRF |
| `retrieval/engine.py` | 五种方法调度 |
| `retrieval/attributes.py` | 中英属性词表 |
| `results/evaluation.json` | 完整实验 |
| `results/experiment_report.md` | 自动摘要 |
