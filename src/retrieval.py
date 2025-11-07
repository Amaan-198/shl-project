from __future__ import annotations

"""
Retrieval module for the SHL recommender.

This module wires together the lexical BM25 index and the dense FAISS
index to produce a list of candidate catalog items for a given query.  A
simple score fusion is performed to balance lexical precision with
semantic recall.  The resulting candidate list is later reranked and
diversified.

The implementation mirrors the original project code.  Where optional
dependencies (e.g. FAISS, sentence‑transformers) are unavailable this
module gracefully falls back to a BM25‑only workflow.  The dense
components are loaded via :func:`~src.embed_index.load_dense_components`,
which returns a ``dense_stale`` flag indicating whether dense search
should be skipped.

Example::

    from src.retrieval import retrieve_candidates
    raw_query, cleaned_query, candidates = retrieve_candidates("C++ developer")
    for item_id, fused_score in candidates:
        ...

"""

import numpy as np
try:
    from loguru import logger  # type: ignore
except Exception:
    import logging
    logging.basicConfig(level=logging.INFO)

    class _FallbackLogger:
        """Simple logger that supports loguru-style brace formatting."""

        def __init__(self, logger):
            self._logger = logger

        def info(self, msg: str, *args, **kwargs) -> None:
            self._logger.info(msg.format(*args))

        def warning(self, msg: str, *args, **kwargs) -> None:
            self._logger.warning(msg.format(*args))

        def error(self, msg: str, *args, **kwargs) -> None:
            self._logger.error(msg.format(*args))

        def exception(self, msg: str, *args, **kwargs) -> None:
            self._logger.exception(msg.format(*args))

    logger = _FallbackLogger(logging.getLogger(__name__))
from typing import List, Tuple

from .config import (
    BM25_INDEX_PATH,
    FAISS_INDEX_PATH,
    IDS_MAPPING_PATH,
    FUSION_TOP_K,
    FUSION_WINSORIZE_MIN,
    FUSION_WINSORIZE_MAX,
    FUSION_EPS,
)
from .embed_index import (
    load_bm25 as load_bm25_cache,
    load_dense_components,
    embed_query_with_chunking,
)

# Explicit fusion weights for reproducibility.  Adjusted to reduce
# lexical dominance and increase semantic influence based on offline
# evaluation.  These values override config.BM25_WEIGHT / DENSE_WEIGHT.
BM25_WEIGHT = 0.60  # emphasise lexical precision to anchor job-role keywords
DENSE_WEIGHT = 0.40  # reduce semantic dominance


def _winsorize(arr: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Clip values into the ``[lo, hi]`` range to reduce outliers."""
    return np.clip(arr, lo, hi)


def _to_z(arr: np.ndarray, eps: float = FUSION_EPS) -> np.ndarray:
    """Standardize an array by subtracting mean and dividing by std.

    A small epsilon avoids division by zero when variance is tiny.
    """
    m = arr.mean() if arr.size else 0.0
    s = arr.std() if arr.size else 1.0
    if s < eps:
        s = eps
    return (arr - m) / s


def _normalize_slice(pairs: List[Tuple[int, float]]) -> dict[int, float]:
    """Convert a list of (id, score) tuples into a normalized dict.

    Scores are winsorized and standardized before returning.  If the
    input list is empty an empty dict is returned.
    """
    if not pairs:
        return {}
    ids, scores = zip(*pairs)
    scores = np.asarray(scores, dtype="float32")
    scores = _to_z(_winsorize(scores, FUSION_WINSORIZE_MIN, FUSION_WINSORIZE_MAX))
    return {i: float(s) for i, s in zip(ids, scores)}


def fuse_scores(
    bm25: List[Tuple[int, float]],
    dense: List[Tuple[int, float]],
    top_k: int = FUSION_TOP_K,
) -> List[Tuple[int, float]]:
    """Combine BM25 and dense scores into a single ranking.

    Each score list is normalized independently then combined using
    fixed weights.  Only the top ``top_k`` candidates are returned.
    """
    nz_bm25 = _normalize_slice(bm25)
    nz_dense = _normalize_slice(dense)
    all_ids = set(nz_bm25) | set(nz_dense)
    fused: List[Tuple[int, float]] = []
    for i in all_ids:
        s = BM25_WEIGHT * nz_bm25.get(i, 0.0) + DENSE_WEIGHT * nz_dense.get(i, 0.0)
        fused.append((i, s))
    fused.sort(key=lambda x: (-x[1], x[0]))
    return fused[:top_k]


def _load_retrieval_components():
    """Internal helper to load the BM25 and dense indices.

    Returns a tuple ``(bm25, model, faiss_index, id_map, dense_stale)``.
    """
    bm25 = load_bm25_cache(BM25_INDEX_PATH)
    model, faiss_index, id_map, dense_stale = load_dense_components(
        FAISS_INDEX_PATH, IDS_MAPPING_PATH
    )
    return bm25, model, faiss_index, id_map, dense_stale


def retrieve_bm25(cleaned_query: str) -> List[Tuple[int, float]]:
    """Query only the BM25 index and return (id, score) pairs."""
    bm25 = load_bm25_cache(BM25_INDEX_PATH)
    return bm25.query(cleaned_query)


def retrieve_dense(model, faiss_index, id_map, query_vec) -> List[Tuple[int, float]]:
    """Query only the dense FAISS index and return (id, score) pairs."""
    # cosine similarity top‑N; index.search returns (D, I) arrays
    D, I = faiss_index.search(query_vec[np.newaxis, :], 200)
    out: List[Tuple[int, float]] = []
    for dist, idx in zip(D[0], I[0]):
        if idx < 0:
            continue
        item_id = id_map[str(idx)]
        out.append((int(item_id), float(dist)))
    return out


def retrieve_candidates(raw_query: str) -> Tuple[str, str, List[Tuple[int, float]]]:
    """High‑level function to produce fused candidates for a query.

    The input ``raw_query`` is normalized using the lexical cleaning
    pipeline.  The BM25 and dense indices are queried (if available)
    and their scores fused.  A list of top candidate IDs and fused
    scores is returned.
    """
    from .normalize import normalize_for_lexical_index as clean_text

    cleaned_query = clean_text(raw_query)
    bm25, model, faiss_index, id_map, dense_stale = _load_retrieval_components()

    bm25_results = retrieve_bm25(cleaned_query)

    if dense_stale:
        dense_results: List[Tuple[int, float]] = []
        logger.warning("Dense index stale or missing — falling back to BM25 only.")
    else:
        q_vec = embed_query_with_chunking(model, cleaned_query)
        dense_results = retrieve_dense(model, faiss_index, id_map, q_vec)

    fused = fuse_scores(bm25_results, dense_results)
    logger.info("Retrieved {} fused candidates for query", len(fused))
    return raw_query, cleaned_query, fused


if __name__ == "__main__":
    # Simple CLI for ad‑hoc testing
    q = input("Enter query or JD URL: ")
    _, cleaned, results = retrieve_candidates(q)
    print("CLEANED:", cleaned)
    for rid, score in results[:10]:
        print(f"{rid}: {score:.4f}")