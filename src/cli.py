# src/cli.py
"""
Batch runner for SHL GenAI recommender.
Generates train/test prediction CSVs without starting FastAPI.

Improvements:
- Sets HF env and warms indices to avoid cold starts
- De-duplicates identical queries (runs once, fans out)
- Always writes a strict two-column CSV with headers: Query, Assessment_url
- Deterministic URL post-processing (canonicalise + de-dup, stable order)
"""

from __future__ import annotations
import argparse
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from src.config import HF_ENV_VARS, TRAIN_PATH, TEST_PATH
from src.api import recommend_single_query
from src.utils.urls import canon_urls
from src.utils.text_clean import clean_query_text
from src.embed_index import load_bm25, load_dense_components


def _set_hf_env_once():
    for k, v in HF_ENV_VARS.items():
        os.environ.setdefault(k, v)


def _warmup_indices():
    try:
        _ = load_bm25()              # cache BM25
        _ = load_dense_components()  # cache dense components (model/index/map)
    except Exception:
        # not fatal; proceed
        pass


def load_queries(path: Path) -> List[str]:
    ext = path.suffix.lower()
    df = pd.read_excel(path) if ext in {".xlsx", ".xls"} else pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    qcol = cols.get("query")
    if not qcol:
        raise ValueError(f"Expected column 'Query' in {path}. Found: {list(df.columns)}")
    # normalise to one-line clean queries
    df["Query"] = df[qcol].astype(str).apply(clean_query_text)
    return df["Query"].tolist()


def _dedup_preserve_order(seq: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for s in seq:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _predict_for_query(q: str) -> List[str]:
    """
    Calls the in-process API once and returns canonicalised, deduped URLs
    in rank order (no alphabetical re-sort).
    """
    urls = recommend_single_query(q)
    urls = canon_urls(urls)

    # de-dup while preserving order
    unique: List[str] = []
    seen = set()
    for u in urls:
        if u and u not in seen:
            seen.add(u)
            unique.append(u)
    return unique


def write_two_column_csv(preds: Dict[str, List[str]], out_path: Path) -> None:
    """
    Write exactly two columns with required casing:
      - Query
      - Assessment_url

    Each (query, predicted_url) becomes a row.
    """
    rows: List[Tuple[str, str]] = []
    for q, urls in preds.items():
        for u in urls:
            rows.append((q, u))
    df = pd.DataFrame(rows, columns=["Query", "Assessment_url"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)


def main():
    _set_hf_env_once()
    _warmup_indices()

    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["train", "test"])
    ap.add_argument("--topk", type=int, default=10, help="max predictions per query (default 10)")
    ap.add_argument("--in", dest="inp", type=str, default=None, help="optional custom input file")
    ap.add_argument("--out", dest="out", type=str, default=None, help="optional custom output file")
    args = ap.parse_args()

    if args.inp:
        inp = Path(args.inp)
    else:
        inp = TRAIN_PATH if args.mode == "train" else TEST_PATH

    if args.out:
        out = Path(args.out)
    else:
        out = Path("artifacts/train_predictions.csv" if args.mode == "train" else "artifacts/test_predictions.csv")

    queries = load_queries(inp)
    print(f"Loaded {len(queries)} queries from {inp}")

    # De-duplicate identical queries to avoid re-running the same text
    unique_queries = _dedup_preserve_order(queries)
    print(f"Unique queries to evaluate: {len(unique_queries)}")

    # Predict once per unique query
    unique_preds: Dict[str, List[str]] = {}
    for i, uq in enumerate(unique_queries, 1):
        try:
            urls = _predict_for_query(uq)
            unique_preds[uq] = urls[:args.topk]
        except Exception as e:
            print(f"[WARN] {i}/{len(unique_queries)} failed: {e}")
            unique_preds[uq] = []
        if i % 10 == 0 or i == len(unique_queries):
            print(f"Processed {i}/{len(unique_queries)} unique queries")

    # Fan-out back to original order (each original query gets the same list)
    final_preds: Dict[str, List[str]] = {}
    for q in queries:
        final_preds[q] = unique_preds.get(q, [])

    write_two_column_csv(final_preds, out)
    total_rows = sum(len(v) for v in final_preds.values())
    print(f"✅ Wrote {total_rows} rows to {out}")


if __name__ == "__main__":
    main()
