from __future__ import annotations

"""
Reranking of candidate items using a cross‑encoder.

After retrieving a list of candidate items via fused lexical and
semantic search the recommender uses a cross‑encoder to compute
fine‑grained relevance scores between the query and each candidate's
metadata.  This module encapsulates the preparation of candidate
texts, loading of the cross‑encoder model, and application of
reranking.  When the ``sentence_transformers`` package is unavailable
or models cannot be loaded the reranking gracefully falls back to
preserving the original fused order.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
try:
    from loguru import logger  # type: ignore
except Exception:
    import logging
    logging.basicConfig(level=logging.INFO)

    class _FallbackLogger:
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

from .config import RERANK_CUTOFF, BGE_RERANKER_MODEL
from .catalog_build import load_catalog_snapshot as load_catalog_df


@dataclass
class Candidate:
    item_id: int
    fused_score: float
    rerank_score: float


def build_candidate_text(row: pd.Series) -> str:
    """Construct a rich text representation for the cross‑encoder.

    Concatenates the item name, description, test type labels and
    flags (adaptive/remote) into a single string.  Assumes the row
    has already been normalized by ``catalog_build``.
    """
    name = str(row.get("name", "")).strip()
    desc = str(row.get("description", "")).strip()
    raw_types = row.get("test_type", [])
    types_list: List[str] = []
    # Normalise the test_type field into a list of strings
    if raw_types is None or (isinstance(raw_types, float) and np.isnan(raw_types)):
        types_list = []
    elif isinstance(raw_types, str):
        # Comma‑separated or repr of list; strip brackets/quotes
        cleaned = raw_types.replace("[", "").replace("]", "").replace("'", "")
        types_list = [t.strip() for t in cleaned.split(",") if t.strip()]
    elif isinstance(raw_types, (list, tuple)):
        types_list = [str(t).strip() for t in raw_types if str(t).strip()]
    else:
        # Try to coerce other iterables (e.g. numpy array) to list
        try:
            types_list = [str(t).strip() for t in list(raw_types) if str(t).strip()]
        except Exception:
            if pd.notnull(raw_types):
                types_list = [str(raw_types).strip()]
            else:
                types_list = []
    flags = []
    if str(row.get("adaptive_support", "")).strip().lower() == "yes":
        flags.append("adaptive")
    if str(row.get("remote_support", "")).strip().lower() == "yes":
        flags.append("remote")
    parts: List[str] = []
    if name:
        parts.append(name)
    if desc:
        parts.append(desc)
    if types_list:
        parts.append(" ".join(types_list))
    if flags:
        parts.append(" ".join(flags))
    return ". ".join(p for p in parts if p)


def get_candidate_texts(item_ids: Sequence[int], catalog_df: Optional[pd.DataFrame] = None) -> List[str]:
    """Gather the candidate text for each item ID.

    If an item ID is not present in the catalog a warning is logged
    and an empty string is returned for that position.
    """
    if catalog_df is None:
        catalog_df = load_catalog_df()
    texts: List[str] = []
    for iid in item_ids:
        if iid not in catalog_df.index:
            logger.warning("Item id {} not found in catalog; using empty text.", iid)
            texts.append("")
            continue
        row = catalog_df.loc[iid]
        texts.append(build_candidate_text(row))
    return texts


# Attempt to import CrossEncoder.  If unavailable we define a dummy
# fallback that returns zeros and signals to the reranker that
# reranking should be skipped.
try:
    from sentence_transformers import CrossEncoder as _HF_CrossEncoder
    _CROSSENCODER_AVAILABLE = True
except Exception:
    _HF_CrossEncoder = None  # type: ignore
    _CROSSENCODER_AVAILABLE = False


class DummyCrossEncoder:
    """Minimal stand‑in for the HuggingFace CrossEncoder.

    Instances of this class expose a ``predict`` method returning
    zeros.  The reranking logic checks for this class to decide
    whether to use fused scores directly.
    """

    def __init__(self, *args, **kwargs) -> None:
        pass

    def predict(self, pairs: Sequence[Tuple[str, str]]) -> List[float]:
        return [0.0] * len(pairs)


def load_reranker() -> object:
    """Load and return a cross‑encoder model or a dummy fallback."""
    if _CROSSENCODER_AVAILABLE:
        try:
            return _HF_CrossEncoder(BGE_RERANKER_MODEL)
        except Exception as e:
            logger.warning(
                "Failed to load CrossEncoder model '{}': {}. Using dummy reranker.",
                BGE_RERANKER_MODEL,
                e,
            )
            return DummyCrossEncoder()
    else:
        logger.warning("sentence_transformers not available — using dummy reranker.")
        return DummyCrossEncoder()


def score_with_model(model: object, query_text: str, candidate_texts: Sequence[str]) -> np.ndarray:
    """Compute cross‑encoder scores for each candidate.

    If a dummy model is supplied an empty array is returned and the
    caller should fall back to fused scores.
    """
    if isinstance(model, DummyCrossEncoder):
        # Return an empty array to signal fallback to fused scores
        return np.asarray([], dtype="float32")
    pairs = [(query_text, c) for c in candidate_texts]
    scores = model.predict(pairs)
    return np.asarray(scores, dtype="float32")


def rerank_candidates(
    query_text: str,
    fused_candidates: Sequence[Tuple[int, float]],
    cutoff: Optional[int] = None,
    *,
    catalog_df: Optional[pd.DataFrame] = None,
    model: Optional[object] = None,
) -> List[Candidate]:
    """Apply cross‑encoder reranking to the top fused candidates.

    If no cross‑encoder is available the fused scores are used as
    rerank scores and the original order is preserved.  Otherwise
    candidates are sorted by cross‑encoder score in descending order.
    """
    if cutoff is None:
        cutoff = RERANK_CUTOFF
    if not fused_candidates:
        return []
    top = list(fused_candidates)[:cutoff]
    item_ids = [iid for iid, _ in top]
    fused_scores = [fs for _, fs in top]
    if catalog_df is None:
        catalog_df = load_catalog_df()
    if model is None:
        model = load_reranker()
    candidate_texts = get_candidate_texts(item_ids, catalog_df=catalog_df)
    logger.info("Running reranker on {} candidates (cutoff={})", len(candidate_texts), cutoff)
    scores = score_with_model(model, query_text, candidate_texts)
    candidates: List[Candidate] = []
    if scores.size == 0:
        # Fallback: use fused scores as rerank scores
        candidates = [
            Candidate(item_id=iid, fused_score=float(fused), rerank_score=float(fused))
            for iid, fused in zip(item_ids, fused_scores)
        ]
    else:
        candidates = [
            Candidate(item_id=iid, fused_score=float(fused), rerank_score=float(rscore))
            for iid, fused, rscore in zip(item_ids, fused_scores, scores)
        ]
    # Sort primarily by rerank score desc, secondarily by item id to stabilise ordering
    candidates.sort(key=lambda c: (-float(c.rerank_score), c.item_id))
    return candidates