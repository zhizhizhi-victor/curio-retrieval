#!/usr/bin/env python3
"""Filter Chinese museum open-access objects from Met CSV / Cleveland / AIC APIs."""

from __future__ import annotations

import csv
import json
import random
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import quote

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
FILTERED = ROOT / "data" / "filtered"
SAMPLES = ROOT / "data" / "samples"
for p in (RAW, FILTERED, SAMPLES):
    p.mkdir(parents=True, exist_ok=True)

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "COMP5523-curio-retrieval/0.1 (course project)"})

STRICT_CHINA_RE = re.compile(r"\b(?:china|chinese)\b", re.I)
CHINA_LOOSE_RE = re.compile(
    r"\b(?:china|chinese)\b|"
    r"(?:ming|qing|tang|song|yuan|han|zhou|neolithic|liangzhu|"
    r"warring states|northern wei|sui|liao)\s*(?:dynasty|period)?",
    re.I,
)
PREFERRED_OBJECT_HINTS = re.compile(
    r"vase|bowl|dish|jar|cup|bottle|plate|ewer|pot|figure|sculpture|"
    r"jade|cong|amulet|panel|textile|silk|porcelain|stoneware|bronze|"
    r"mirror|snuff|brush|censer|incense|plaque|box|vessel|beaker|"
    r"teapot|saucer|basin|tray|robe|cloak",
    re.I,
)


def get_json(url: str, *, sleep: float = 0.05, retries: int = 4) -> Any:
    last_err: Exception | None = None
    for i in range(retries):
        try:
            r = SESSION.get(url, timeout=60)
            if r.status_code == 429:
                time.sleep(1.5 * (i + 1))
                continue
            r.raise_for_status()
            time.sleep(sleep)
            return r.json()
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(0.6 * (i + 1))
    raise RuntimeError(f"GET failed: {url} ({last_err})")


def norm(s: Any) -> str:
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return ""
    if isinstance(s, list):
        return "; ".join(norm(x) for x in s if x)
    return str(s).strip()


def simplify_period(period: str, dynasty: str = "", date_text: str = "") -> str:
    blob = f"{period} {dynasty} {date_text}".lower()
    mapping = [
        ("neolithic", "Neolithic"),
        ("liangzhu", "Neolithic"),
        ("shang", "Shang"),
        ("zhou", "Zhou"),
        ("warring states", "Warring States"),
        ("han", "Han"),
        ("tang", "Tang"),
        ("song", "Song"),
        ("yuan", "Yuan"),
        ("ming", "Ming"),
        ("qing", "Qing"),
        ("republic", "Republic"),
        ("liao", "Liao"),
        ("jin dynasty", "Jin"),
        ("northern wei", "Northern Wei"),
        ("sui", "Sui"),
    ]
    for key, label in mapping:
        if key in blob:
            return label
    if period:
        return period.split("(")[0].strip()[:40] or "Unknown"
    if dynasty:
        return dynasty.split("(")[0].strip()[:40] or "Unknown"
    return "Unknown"


def simplify_material(medium: str, classification: str = "") -> str:
    blob = f"{medium} {classification}".lower()
    mapping = [
        ("jade", "Jade"),
        ("nephrite", "Jade"),
        ("porcelain", "Porcelain"),
        ("stoneware", "Stoneware"),
        ("earthenware", "Earthenware"),
        ("ceramics", "Ceramics"),
        ("ceramic", "Ceramics"),
        ("bronze", "Bronze"),
        ("silk", "Silk/Textile"),
        ("textile", "Silk/Textile"),
        ("tapestry", "Silk/Textile"),
        ("enamel", "Enamel"),
        ("cloisonn", "Enamel"),
        ("lacquer", "Lacquer"),
        ("ivory", "Ivory"),
        ("wood", "Wood"),
        ("gold", "Metal"),
        ("silver", "Metal"),
        ("copper", "Metal"),
        ("iron", "Metal"),
        ("paper", "Paper/Painting"),
        ("ink", "Paper/Painting"),
        ("hanging scroll", "Paper/Painting"),
        ("handscroll", "Paper/Painting"),
        ("album", "Paper/Painting"),
    ]
    for key, label in mapping:
        if key in blob:
            return label
    if medium:
        return medium.split(",")[0].strip()[:40] or "Other"
    return "Other"


def simplify_object_type(object_name: str, title: str = "", classification: str = "") -> str:
    blob = f"{object_name} {title} {classification}".lower()
    mapping = [
        ("snuff bottle", "Snuff bottle"),
        ("vase", "Vase"),
        ("bottle", "Bottle"),
        ("bowl", "Bowl"),
        ("dish", "Dish"),
        ("plate", "Dish"),
        ("jar", "Jar"),
        ("cup", "Cup"),
        ("beaker", "Cup"),
        ("ewer", "Ewer"),
        ("teapot", "Teapot"),
        ("figure", "Figure"),
        ("sculpture", "Figure"),
        ("cong", "Cong/Jade ritual"),
        ("amulet", "Amulet"),
        ("pendant", "Amulet"),
        ("mirror", "Mirror"),
        ("censer", "Censer"),
        ("incense", "Censer"),
        ("brush", "Scholar object"),
        ("inkstone", "Scholar object"),
        ("panel", "Textile/Panel"),
        ("textile", "Textile/Panel"),
        ("robe", "Textile/Panel"),
        ("scroll", "Painting/Scroll"),
        ("painting", "Painting/Scroll"),
        ("album", "Painting/Scroll"),
        ("box", "Box"),
        ("plaque", "Plaque"),
        ("vessel", "Vessel"),
        ("urn", "Jar"),
    ]
    for key, label in mapping:
        if key in blob:
            return label
    if object_name:
        return object_name.split(",")[0].strip()[:40] or "Other"
    return "Other"


def china_mask(df: pd.DataFrame) -> pd.Series:
    cols = [
        "Culture",
        "Country",
        "Region",
        "Period",
        "Dynasty",
        "Title",
        "Department",
        "Classification",
        "Object Name",
        "Medium",
        "Tags",
    ]
    present = [c for c in cols if c in df.columns]
    blob = df[present].fillna("").astype(str).agg(" | ".join, axis=1)
    strict = blob.str.contains(r"\b(?:china|chinese)\b", case=False, na=False, regex=True)
    asian = df.get("Department", pd.Series([""] * len(df))).fillna("").str.contains(
        "Asian Art", case=False, na=False
    )
    dynastyish = blob.str.contains(
        r"\b(?:ming|qing|tang|song|yuan|han|zhou|neolithic|liangzhu)\b",
        case=False,
        na=False,
        regex=True,
    )
    return strict | (asian & dynastyish)


def score_row(r: dict[str, Any]) -> int:
    score = 0
    if r.get("period") and r["period"] != "Unknown":
        score += 2
    if r.get("material") and r["material"] != "Other":
        score += 2
    text = f"{r.get('object_type_raw','')} {r.get('title','')} {r.get('material_raw','')}"
    if PREFERRED_OBJECT_HINTS.search(text):
        score += 2
    if r.get("object_type") == "Painting/Scroll":
        score -= 1
    if r.get("material") == "Paper/Painting":
        score -= 1
    return score


def met_row_to_rec(row: dict[str, Any] | pd.Series) -> dict[str, Any]:
    get = row.get
    oid = int(get("Object ID"))
    title = norm(get("Title"))
    object_name = norm(get("Object Name"))
    medium = norm(get("Medium"))
    classification = norm(get("Classification"))
    period = norm(get("Period"))
    dynasty = norm(get("Dynasty"))
    date_text = norm(get("Object Date"))
    culture = norm(get("Culture")) or "China"
    return {
        "id": f"met_{oid}",
        "source": "Met",
        "source_id": str(oid),
        "title": title,
        "object_type_raw": object_name,
        "object_type": simplify_object_type(object_name, title, classification),
        "material_raw": medium,
        "material": simplify_material(medium, classification),
        "period_raw": period or dynasty,
        "period": simplify_period(period, dynasty, date_text),
        "culture": culture,
        "classification": classification,
        "department": norm(get("Department")),
        "artist": norm(get("Artist Display Name")),
        "date_text": date_text,
        "image_url": "",
        "image_url_hires": "",
        "page_url": norm(get("Link Resource")),
        "is_public_domain": True,
        "license": "CC0/Public Domain (Met Open Access)",
        "_score": 0,
    }


def attach_met_images(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    cache_path = RAW / "met_image_cache.jsonl"
    cached: dict[str, dict[str, str]] = {}
    if cache_path.exists():
        with cache_path.open() as f:
            for line in f:
                row = json.loads(line)
                cached[str(row["source_id"])] = row
        print(f"[Met] image cache {len(cached)}", flush=True)

    with cache_path.open("a") as cache_f:
        for i, rec in enumerate(records, 1):
            sid = rec["source_id"]
            if sid in cached:
                rec["image_url"] = cached[sid].get("image_url", "")
                rec["image_url_hires"] = cached[sid].get("image_url_hires", "")
            else:
                try:
                    obj = get_json(
                        f"https://collectionapi.metmuseum.org/public/collection/v1/objects/{sid}",
                        sleep=0.02,
                    )
                except Exception as e:  # noqa: BLE001
                    print(f"[Met] skip {sid}: {e}", flush=True)
                    continue
                img_small = norm(obj.get("primaryImageSmall") or obj.get("primaryImage"))
                img_hi = norm(obj.get("primaryImage"))
                rec["image_url"] = img_small
                rec["image_url_hires"] = img_hi
                cache_f.write(
                    json.dumps(
                        {
                            "source_id": sid,
                            "image_url": img_small,
                            "image_url_hires": img_hi,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                cache_f.flush()
            if rec["image_url"]:
                out.append(rec)
            if i % 100 == 0:
                print(f"[Met] images {i}/{len(records)} with_image={len(out)}", flush=True)
    print(f"[Met] with images: {len(out)}/{len(records)}", flush=True)
    return out


def fetch_met(candidate_limit: int = 2500) -> list[dict[str, Any]]:
    csv_path = RAW / "MetObjects.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing {csv_path}; download MetObjects.csv first")

    print("[Met] reading CSV...")
    df = pd.read_csv(csv_path, low_memory=False)
    # strip BOM-ish column
    df.columns = [c.replace("\ufeff", "") for c in df.columns]
    print(f"[Met] rows={len(df)}")

    pd_mask = df["Is Public Domain"].astype(str).str.lower().isin(["true", "1", "yes"])
    china = china_mask(df)
    cand = df[pd_mask & china].copy()
    print(f"[Met] China + Public Domain metadata: {len(cand)}")

    # Prefer Asian Art department when present
    if "Department" in cand.columns:
        asian = cand["Department"].fillna("").str.contains("Asian Art", case=False)
        # keep all china, but rank Asian Art higher later
        cand = cand.assign(_asian=asian.astype(int))
    else:
        cand = cand.assign(_asian=0)

    print("[Met] converting rows...", flush=True)
    records = [met_row_to_rec(row) for row in cand.to_dict(orient="records")]
    for r in records:
        r["_score"] = score_row(r) + (2 if "Asian Art" in r.get("department", "") else 0)

    records.sort(key=lambda r: (-r["_score"], r["source_id"]))
    shortlist = records[:candidate_limit]
    print(f"[Met] shortlist for image fetch: {len(shortlist)}", flush=True)
    with_images = attach_met_images(shortlist)
    for r in with_images:
        r.pop("_score", None)
    return with_images


def fetch_cleveland() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    queries = [
        "department=Chinese%20Art&cc0&has_image=1&limit=1000",
        "culture=China&cc0&has_image=1&limit=1000",
        "culture=Chinese&cc0&has_image=1&limit=1000",
    ]
    for q in queries:
        skip = 0
        while True:
            url = f"https://openaccess-api.clevelandart.org/api/artworks/?{q}&skip={skip}"
            data = get_json(url, sleep=0.12)
            info = data.get("info") or {}
            rows = data.get("data") or []
            if not rows:
                break
            for obj in rows:
                aid = str(obj.get("id") or obj.get("accession_number") or "")
                if not aid or aid in seen:
                    continue
                images = obj.get("images") or {}
                img = ""
                if isinstance(images, dict):
                    for key in ("web", "print", "full"):
                        block = images.get(key) or {}
                        if isinstance(block, dict) and block.get("url"):
                            img = block["url"]
                            break
                if not img:
                    continue
                license_status = norm(obj.get("share_license_status")).upper()
                if license_status == "COPYRIGHT":
                    continue
                culture = obj.get("culture")
                if isinstance(culture, list):
                    culture = "; ".join(norm(x) for x in culture)
                else:
                    culture = norm(culture)
                title = norm(obj.get("title"))
                technique = norm(obj.get("technique") or obj.get("medium"))
                typ = norm(obj.get("type"))
                creation_date = norm(obj.get("creation_date") or obj.get("date_text"))
                tombstone = norm(obj.get("tombstone"))
                dept = obj.get("department")
                if isinstance(dept, dict):
                    dept = norm(dept.get("name") or dept.get("title"))
                else:
                    dept = norm(dept)
                geo_blob = " ".join([culture, title, tombstone, creation_date, dept])
                if not (
                    STRICT_CHINA_RE.search(geo_blob)
                    or re.search(r"chinese art", dept, re.I)
                ):
                    continue
                artists = obj.get("artists") or []
                artist = ""
                if artists and isinstance(artists[0], dict):
                    artist = norm(artists[0].get("name"))
                rec = {
                    "id": f"cma_{aid}",
                    "source": "Cleveland",
                    "source_id": aid,
                    "title": title,
                    "object_type_raw": typ,
                    "object_type": simplify_object_type(typ, title, technique),
                    "material_raw": technique,
                    "material": simplify_material(technique, typ),
                    "period_raw": creation_date,
                    "period": simplify_period(creation_date, "", creation_date),
                    "culture": culture or "China",
                    "classification": typ,
                    "department": dept or "Chinese Art",
                    "artist": artist,
                    "date_text": creation_date,
                    "image_url": img,
                    "image_url_hires": img,
                    "page_url": norm(obj.get("url") or f"https://www.clevelandart.org/art/{aid}"),
                    "is_public_domain": True,
                    "license": "CC0 (Cleveland Open Access)",
                }
                seen.add(aid)
                out.append(rec)
            total = int(info.get("total") or 0)
            skip += len(rows)
            print(f"[CMA] {q[:35]}... {skip}/{total} kept={len(out)}")
            if not total or skip >= total or len(rows) < 1000:
                break
    print(f"[CMA] kept {len(out)}")
    return out


def fetch_aic(max_pages: int = 50) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    fields = (
        "id,title,artist_title,date_display,place_of_origin,medium_display,"
        "artwork_type_title,department_title,classification_title,is_public_domain,"
        "image_id,style_title,culture"
    )
    for q in ("China", "Chinese"):
        page = 1
        while page <= max_pages:
            url = (
                "https://api.artic.edu/api/v1/artworks/search"
                f"?q={quote(q)}&query[term][is_public_domain]=true"
                f"&limit=100&page={page}&fields={quote(fields)}"
            )
            data = get_json(url, sleep=0.2)
            rows = data.get("data") or []
            if not rows:
                break
            iiif = (data.get("config") or {}).get("iiif_url") or "https://www.artic.edu/iiif/2"
            for obj in rows:
                oid = str(obj.get("id"))
                if not oid or oid in seen or not obj.get("is_public_domain"):
                    continue
                image_id = obj.get("image_id")
                if not image_id:
                    continue
                place = norm(obj.get("place_of_origin"))
                culture = norm(obj.get("culture"))
                title = norm(obj.get("title"))
                dept = norm(obj.get("department_title"))
                blob = " ".join([place, culture, title, dept, norm(obj.get("date_display"))])
                if not STRICT_CHINA_RE.search(blob) and "china" not in place.lower():
                    continue
                medium = norm(obj.get("medium_display"))
                typ = norm(obj.get("artwork_type_title") or obj.get("classification_title"))
                date_text = norm(obj.get("date_display"))
                rec = {
                    "id": f"aic_{oid}",
                    "source": "AIC",
                    "source_id": oid,
                    "title": title,
                    "object_type_raw": typ,
                    "object_type": simplify_object_type(typ, title, medium),
                    "material_raw": medium,
                    "material": simplify_material(medium, typ),
                    "period_raw": date_text,
                    "period": simplify_period(date_text, "", date_text),
                    "culture": culture or place or "China",
                    "classification": typ,
                    "department": dept,
                    "artist": norm(obj.get("artist_title")),
                    "date_text": date_text,
                    "image_url": f"{iiif}/{image_id}/full/843,/0/default.jpg",
                    "image_url_hires": f"{iiif}/{image_id}/full/1686,/0/default.jpg",
                    "page_url": f"https://www.artic.edu/artworks/{oid}",
                    "is_public_domain": True,
                    "license": "CC0/Public Domain (AIC Open Access)",
                }
                seen.add(oid)
                out.append(rec)
            pag = data.get("pagination") or {}
            total_pages = int(pag.get("total_pages") or page)
            print(f"[AIC] q={q} page={page}/{total_pages} kept={len(out)}")
            if page >= total_pages:
                break
            page += 1
    print(f"[AIC] kept {len(out)}")
    return out


def dedupe(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out = []
    for r in records:
        key = "|".join(
            [
                re.sub(r"\W+", " ", r.get("title", "")).strip().lower(),
                r.get("object_type", ""),
                r.get("period", ""),
                r.get("material", ""),
                r.get("source", ""),
            ]
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def stratified_sample(rows: list[dict[str, Any]], target: int = 800, seed: int = 5523) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    pool = sorted(rows, key=score_row, reverse=True)
    # Prefer score >= 2
    preferred = [r for r in pool if score_row(r) >= 2] or pool
    buckets: dict[tuple[str, str, str], list] = {}
    for r in preferred:
        key = (r.get("material", "Other"), r.get("object_type", "Other"), r.get("period", "Unknown"))
        buckets.setdefault(key, []).append(r)
    for k in buckets:
        rng.shuffle(buckets[k])
    selected: list[dict] = []
    caps = {k: min(18, len(v)) for k, v in buckets.items()}
    progressed = True
    while len(selected) < target and progressed:
        progressed = False
        for k, items in buckets.items():
            if caps[k] <= 0 or not items:
                continue
            selected.append(items.pop())
            caps[k] -= 1
            progressed = True
            if len(selected) >= target:
                break
    leftover = [r for items in buckets.values() for r in items]
    # also allow non-preferred fill
    leftover += [r for r in pool if r not in selected and r not in leftover]
    rng.shuffle(leftover)
    for r in leftover:
        if len(selected) >= target:
            break
        selected.append(r)
    return selected[:target]


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = max(len(rows), 1)
    return {
        "total": len(rows),
        "by_source": dict(Counter(r["source"] for r in rows)),
        "by_material": dict(Counter(r["material"] for r in rows).most_common(20)),
        "by_object_type": dict(Counter(r["object_type"] for r in rows).most_common(20)),
        "by_period": dict(Counter(r["period"] for r in rows).most_common(20)),
        "labeled_period_pct": round(
            100 * sum(1 for r in rows if r.get("period") not in {"", "Unknown"}) / n, 1
        ),
        "labeled_material_pct": round(
            100 * sum(1 for r in rows if r.get("material") not in {"", "Other"}) / n, 1
        ),
    }


def main() -> None:
    print("=== Met (CSV + image hydrate) ===", flush=True)
    # 1500 is enough for an 800-image gallery after CMA/AIC merge; cache resumes.
    met = fetch_met(candidate_limit=1500)
    write_csv(FILTERED / "met_china_pd.csv", met)
    (FILTERED / "met_china_pd.json").write_text(json.dumps(met, ensure_ascii=False, indent=2))

    print("=== Cleveland ===")
    cma = fetch_cleveland()
    write_csv(FILTERED / "cma_china_cc0.csv", cma)
    (FILTERED / "cma_china_cc0.json").write_text(json.dumps(cma, ensure_ascii=False, indent=2))

    print("=== AIC ===")
    aic = fetch_aic()
    write_csv(FILTERED / "aic_china_pd.csv", aic)
    (FILTERED / "aic_china_pd.json").write_text(json.dumps(aic, ensure_ascii=False, indent=2))

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

    report = {
        "met": summarize(met),
        "cleveland": summarize(cma),
        "aic": summarize(aic),
        "merged_deduped": summarize(merged),
        "gallery_candidate_800": summarize(sample),
        "filter_rules": {
            "must_have": [
                "China/Chinese culture or Asian Art + Chinese dynasty terms",
                "public domain / CC0",
                "non-empty image URL",
            ],
            "normalized_labels": ["object_type", "material", "period"],
            "sample_seed": 5523,
            "sample_size": len(sample),
            "met_csv": str(RAW / "MetObjects.csv"),
        },
    }
    (FILTERED / "filter_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))

    md = [
        "# 中国博物馆馆藏 · 数据集筛选报告",
        "",
        "## 筛选条件",
        "- 与中国相关（Culture/Country 含 China/Chinese，或 Asian Art + 中国朝代词）",
        "- 公有领域 / CC0",
        "- 有图片 URL",
        "- 规范化标签：器型 `object_type`、材质 `material`、时期 `period`",
        "",
        "## 各源结果",
        "",
        "| 来源 | 筛选后(有图) | 有时期标签% | 有材质标签% |",
        "|---|---:|---:|---:|",
        f"| Met | {report['met']['total']} | {report['met']['labeled_period_pct']} | {report['met']['labeled_material_pct']} |",
        f"| Cleveland | {report['cleveland']['total']} | {report['cleveland']['labeled_period_pct']} | {report['cleveland']['labeled_material_pct']} |",
        f"| AIC | {report['aic']['total']} | {report['aic']['labeled_period_pct']} | {report['aic']['labeled_material_pct']} |",
        f"| 合并去重 | {report['merged_deduped']['total']} | {report['merged_deduped']['labeled_period_pct']} | {report['merged_deduped']['labeled_material_pct']} |",
        f"| **建议图库抽样** | **{report['gallery_candidate_800']['total']}** | {report['gallery_candidate_800']['labeled_period_pct']} | {report['gallery_candidate_800']['labeled_material_pct']} |",
        "",
        "### 合并后按来源",
        "",
        "```json",
        json.dumps(report["merged_deduped"]["by_source"], ensure_ascii=False, indent=2),
        "```",
        "",
        "### 建议图库 · 材质",
        "",
        "```json",
        json.dumps(report["gallery_candidate_800"]["by_material"], ensure_ascii=False, indent=2),
        "```",
        "",
        "### 建议图库 · 器型",
        "",
        "```json",
        json.dumps(report["gallery_candidate_800"]["by_object_type"], ensure_ascii=False, indent=2),
        "```",
        "",
        "### 建议图库 · 时期",
        "",
        "```json",
        json.dumps(report["gallery_candidate_800"]["by_period"], ensure_ascii=False, indent=2),
        "```",
        "",
        "## 输出文件",
        "",
        "- `data/raw/MetObjects.csv` — Met 全量开放元数据",
        "- `data/filtered/met_china_pd.csv`",
        "- `data/filtered/cma_china_cc0.csv`",
        "- `data/filtered/aic_china_pd.csv`",
        "- `data/filtered/merged_china_openaccess.csv`",
        "- `data/samples/gallery_candidate_800.csv` ← 建议项目图库",
        "- `data/filtered/filter_report.json`",
        "",
    ]
    (FILTERED / "筛选报告.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("Done -> data/filtered/筛选报告.md")


if __name__ == "__main__":
    main()
