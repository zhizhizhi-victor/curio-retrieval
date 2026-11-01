#!/usr/bin/env python3
"""Merge already-filtered Met/CMA (and optional AIC) into gallery sample + report."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from filter_datasets import (  # noqa: E402
    FILTERED,
    SAMPLES,
    dedupe,
    fetch_aic,
    stratified_sample,
    summarize,
    write_csv,
)


def load(name: str) -> list:
    path = FILTERED / name
    if not path.exists():
        print(f"missing {path}")
        return []
    return json.loads(path.read_text())


def main() -> None:
    met = load("met_china_pd.json")
    cma = load("cma_china_cc0.json")
    print(f"loaded met={len(met)} cma={len(cma)}", flush=True)

    aic: list = []
    try:
        aic = fetch_aic(max_pages=30)
    except Exception as e:  # noqa: BLE001
        print(f"[AIC] skipped: {e}", flush=True)
    if aic:
        write_csv(FILTERED / "aic_china_pd.csv", aic)
        (FILTERED / "aic_china_pd.json").write_text(json.dumps(aic, ensure_ascii=False, indent=2))
    else:
        (FILTERED / "aic_china_pd.json").write_text("[]")
        write_csv(FILTERED / "aic_china_pd.csv", [])

    merged = dedupe(met + cma + aic)
    write_csv(FILTERED / "merged_china_openaccess.csv", merged)
    (FILTERED / "merged_china_openaccess.json").write_text(
        json.dumps(merged, ensure_ascii=False, indent=2)
    )

    sample = stratified_sample(merged, target=800, seed=5523)
    write_csv(SAMPLES / "gallery_candidate_800.csv", sample)
    (SAMPLES / "gallery_candidate_800.json").write_text(
        json.dumps(sample, ensure_ascii=False, indent=2)
    )

    # Also keep a preview of 20 examples for the report
    preview = sample[:20]

    report = {
        "met": summarize(met),
        "cleveland": summarize(cma),
        "aic": summarize(aic),
        "merged_deduped": summarize(merged),
        "gallery_candidate_800": summarize(sample),
        "preview_20": [
            {
                "id": r["id"],
                "source": r["source"],
                "title": r["title"],
                "object_type": r["object_type"],
                "material": r["material"],
                "period": r["period"],
                "image_url": r["image_url"],
            }
            for r in preview
        ],
        "filter_rules": {
            "must_have": [
                "China/Chinese related culture or Chinese Art department",
                "public domain / CC0",
                "non-empty image URL",
            ],
            "normalized_labels": ["object_type", "material", "period"],
            "sample_seed": 5523,
            "sample_size": len(sample),
            "note_aic": "AIC API returned 403 from this network; Met+Cleveland used as primary pool.",
        },
    }
    (FILTERED / "filter_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))

    md = f"""# 中国博物馆馆藏 · 数据集筛选报告

## 筛选条件
- 与中国相关（Culture/Country 含 China/Chinese，或 Asian Art + 中国朝代，或 Cleveland Chinese Art 部门）
- 公有领域 / CC0
- 有图片 URL
- 规范化标签：器型 `object_type`、材质 `material`、时期 `period`

## 各源结果

| 来源 | 筛选后(有图) | 有时期标签% | 有材质标签% |
|---|---:|---:|---:|
| Met | {report['met']['total']} | {report['met']['labeled_period_pct']} | {report['met']['labeled_material_pct']} |
| Cleveland | {report['cleveland']['total']} | {report['cleveland']['labeled_period_pct']} | {report['cleveland']['labeled_material_pct']} |
| AIC | {report['aic']['total']} | {report['aic']['labeled_period_pct']} | {report['aic']['labeled_material_pct']} |
| 合并去重 | {report['merged_deduped']['total']} | {report['merged_deduped']['labeled_period_pct']} | {report['merged_deduped']['labeled_material_pct']} |
| **建议图库抽样** | **{report['gallery_candidate_800']['total']}** | {report['gallery_candidate_800']['labeled_period_pct']} | {report['gallery_candidate_800']['labeled_material_pct']} |

> 说明：当前网络下 AIC API 返回 403，故主池为 Met + Cleveland（与取样图中的两家一致）。AIC 可后续在可访问网络补入。

### 合并后按来源

```json
{json.dumps(report['merged_deduped']['by_source'], ensure_ascii=False, indent=2)}
```

### 建议图库 · 材质 Top

```json
{json.dumps(report['gallery_candidate_800']['by_material'], ensure_ascii=False, indent=2)}
```

### 建议图库 · 器型 Top

```json
{json.dumps(report['gallery_candidate_800']['by_object_type'], ensure_ascii=False, indent=2)}
```

### 建议图库 · 时期 Top

```json
{json.dumps(report['gallery_candidate_800']['by_period'], ensure_ascii=False, indent=2)}
```

## 输出文件
- `data/raw/MetObjects.csv`
- `data/filtered/met_china_pd.csv` / `.json`
- `data/filtered/cma_china_cc0.csv` / `.json`
- `data/filtered/merged_china_openaccess.csv`
- `data/samples/gallery_candidate_800.csv` ← **建议用作项目图库**
- `data/filtered/filter_report.json`
- `data/filtered/筛选报告.md`
"""
    (FILTERED / "筛选报告.md").write_text(md, encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("met", "cleveland", "aic", "merged_deduped", "gallery_candidate_800")}, ensure_ascii=False, indent=2))
    print("Done -> data/filtered/筛选报告.md", flush=True)


if __name__ == "__main__":
    main()
