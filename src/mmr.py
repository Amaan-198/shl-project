from __future__ import annotations

"""
Maximal Marginal Relevance (MMR) diversification for search results.

The MMR algorithm selects a subset of items that balances relevance
against diversity by penalising items that are very similar to ones
already chosen.  In the original project this requires an embedding
matrix saved alongside the dense FAISS index.  In environments where
the embeddings file is unavailable this module falls back to a simple
truncation of the input list.
"""

import json
import os
from typing import List, Sequence, Tuple

import numpy as np
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

from .config import MMR_LAMBDA, RESULT_DEFAULT_TARGET, EMBEDDINGS_PATH, IDS_MAPPING_PATH


def load_item_embeddings() -> tuple[np.ndarray | None, list[int]]:
    """Load the item embedding matrix and corresponding ID map.

    If the files referenced by ``EMBEDDINGS_PATH`` or ``IDS_MAPPING_PATH`` are
    missing a ``(None, [])`` tuple is returned.  This allows the rest of
    the pipeline to skip MMR diversification gracefully.
    """
    emb_path = str(EMBEDDINGS_PATH)
    ids_path = str(IDS_MAPPING_PATH)
    if not os.path.exists(emb_path) or not os.path.exists(ids_path):
        logger.warning(
            "Embedding files missing ({} or {}). Skipping MMR diversification.",
            emb_path,
            ids_path,
        )
        return None, []
    try:
        logger.info("Loading item embeddings and ID mapping")
        emb = np.load(emb_path, mmap_mode="r")
        with open(ids_path, "r", encoding="utf-8") as f:
            ids = json.load(f)
        if emb.shape[0] != len(ids):
            raise ValueError(
                f"Embeddings ({emb.shape[0]}) and id map ({len(ids)}) length mismatch"
            )
        logger.info("Loaded embeddings: shape={}, items={}", emb.shape, len(ids))
        return emb, ids
    except Exception as e:
        logger.exception("Error loading embeddings: {}", e)
        return None, []


def mmr_select(
    candidates: Sequence[Tuple[int, float]],
    embeddings: np.ndarray | None,
    ids: list[int],
    k: int = RESULT_DEFAULT_TARGET,
    lambda_: float = MMR_LAMBDA,
) -> List[int]:
    """Apply MMR to diversify a ranked list of candidates.

    If ``embeddings`` is None the input candidates are truncated to
    ``k`` items.  Otherwise the standard MMR algorithm is applied
    using cosine similarity of the provided embedding matrix.
    """
    if not candidates:
        return []
    # Fallback: if no embeddings available, return top-k by relevance
    if embeddings is None or len(ids) == 0:
        logger.warning("No embeddings available for MMR — returning top {} items only.", k)
        return [cid for cid, _ in candidates[:k]]
    # Map item_id -> embedding row index
    id_to_index = {id_: idx for idx, id_ in enumerate(ids)}
    # Precompute embedding vectors for these candidates
    chosen_ids = [cid for cid, _ in candidates]
    try:
        vecs = np.stack([embeddings[id_to_index[cid]] for cid in chosen_ids])
    except Exception as e:
        logger.exception("Error retrieving embeddings for MMR: {}", e)
        return [cid for cid, _ in candidates[:k]]
    relevance_scores = np.array([score for _, score in candidates], dtype="float32")
    # Cosine similarities between candidates (embeddings assumed L2 normalised)
    sim_matrix = np.dot(vecs, vecs.T)
    n = len(candidates)
    selected: list[int] = []
    unselected = list(range(n))
    # Select first item (highest relevance)
    selected.append(unselected.pop(0))
    # Iteratively select items balancing relevance and novelty
    while len(selected) < min(k, n) and unselected:
        mmr_scores = []
        for idx in unselected:
            rel = relevance_scores[idx]
            div = 0.0 if not selected else float(np.max(sim_matrix[idx, selected]))
            mmr = lambda_ * rel - (1.0 - lambda_) * div
            mmr_scores.append(mmr)
        next_index = unselected[int(np.argmax(mmr_scores))]
        selected.append(next_index)
        unselected.remove(next_index)
    final_ids = [chosen_ids[i] for i in selected]
    logger.info("MMR selected {} items (λ={})", len(final_ids), lambda_)
    return final_ids