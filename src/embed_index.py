from __future__ import annotations

"""
Index builders and loaders for both lexical (BM25) and dense retrieval.

This module offers utilities to create and persist two kinds of search
indices over the catalog's ``search_text`` field:

* A lexical index based on BM25 (Okapi) for bag‑of‑words retrieval.
* A dense embedding index based on BAAI/BGE and FAISS for semantic
  similarity.  In environments where the requisite third‑party
  dependencies (``rank_bm25``, ``sentence_transformers``, ``faiss``)
  are unavailable, the module transparently degrades to a simplified
  fallback that still exposes the same API surface.  In that case
  BM25 queries are executed with a basic token overlap heuristic and
  the dense index is marked stale so it will not be used by the
  retrieval layer.
"""

import json
import os
import pickle
from pathlib import Path
from typing import List, Sequence, Tuple, Dict, Optional

import numpy as np  # type: ignore
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

from .catalog_build import load_catalog_snapshot
from .config import (
    BGE_ENCODER_MODEL,
    BM25_INDEX_PATH,
    EMBEDDINGS_PATH,
    FAISS_INDEX_PATH,
    HF_ENV_VARS,
    IDS_MAPPING_PATH,
)
from .normalize import lexical_tokens_for_bm25

# Optional third‑party dependencies.  We attempt to import them at
# module load time and fall back to ``None`` if they are missing.
try:
    import faiss  # type: ignore[import-not-found]
except Exception:
    faiss = None  # type: ignore

try:
    from rank_bm25 import BM25Okapi  # type: ignore[import-not-found]
except Exception:
    BM25Okapi = None  # type: ignore

try:
    from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]
except Exception:
    SentenceTransformer = None  # type: ignore


def _ensure_hf_env() -> None:
    """
    Ensure key HuggingFace environment variables are set.

    When downloading models for the first time (online mode) or loading
    them from a pre‑existing cache (offline mode), certain environment
    variables influence behaviour.  We only set them if not already
    present to avoid overriding user preferences.
    """
    for key, val in HF_ENV_VARS.items():
        if key not in os.environ:
            os.environ[key] = val


# -----------------------------------------------------------------------------
# Fallback BM25 implementation
# -----------------------------------------------------------------------------

class SimpleBM25Index:
    """
    Minimal fallback for BM25 when ``rank_bm25`` is unavailable.

    This index stores tokenised documents and exposes a ``query``
    method that returns a list of (item_id, score) tuples sorted by
    decreasing score.  The scoring function counts the number of
    unique token overlaps between the query and each document.  While
    not a true BM25 implementation it preserves the essential API
    surface for the retrieval layer.
    """

    def __init__(self, corpus_tokens: List[List[str]], item_ids: List[int]):
        self.corpus_tokens = corpus_tokens
        self.item_ids = item_ids

    def query(self, text: str, top_n: Optional[int] = None) -> List[Tuple[int, float]]:
        q_tokens = set(lexical_tokens_for_bm25(text or ""))
        scores: List[Tuple[int, float]] = []
        for doc_tokens, iid in zip(self.corpus_tokens, self.item_ids):
            if not doc_tokens:
                overlap = 0
            else:
                overlap = len(q_tokens.intersection(doc_tokens))
            scores.append((iid, float(overlap)))
        scores.sort(key=lambda x: (-x[1], x[0]))
        if top_n is not None:
            return scores[:top_n]
        return scores


def prepare_corpus_for_bm25(search_texts: Sequence[str]) -> List[List[str]]:
    """
    Tokenise a list of search_text strings for BM25 or the fallback index.

    Uses the same lexical normalisation and synonym expansion as queries.
    Each document becomes a list of tokens.
    """
    corpus_tokens: List[List[str]] = []
    for text in search_texts:
        tokens = lexical_tokens_for_bm25(text or "")
        corpus_tokens.append(tokens)
    return corpus_tokens


def build_bm25_index(
    search_texts: Sequence[str],
    item_ids: Sequence[int],
    output_path: Path = BM25_INDEX_PATH,
) -> Path:
    """
    Build a BM25Okapi index over the given search_text corpus.

    Persisted object is a pickle containing either a ``BM25Okapi``
    instance (if the dependency is available) or a ``SimpleBM25Index``
    fallback, along with the list of item IDs.  The ``SimpleBM25Index``
    additionally stores the tokenised corpus so that queries can be
    evaluated later.
    """
    logger.info("Building BM25 index over {} documents", len(search_texts))

    # Tokenise the corpus once
    corpus_tokens = prepare_corpus_for_bm25(search_texts)
    item_ids_list = list(item_ids)

    if BM25Okapi is not None and corpus_tokens:
        try:
            bm25 = BM25Okapi(corpus_tokens)  # type: ignore[arg-type]
            index_obj = {
                "bm25": bm25,
                "item_ids": item_ids_list,
            }
            logger.info("Constructed rank_bm25 BM25Okapi index")
        except Exception as e:
            # Fall back if initialisation fails (e.g. due to input shape)
            logger.warning("rank_bm25 failed to build index: {}; falling back to simple", e)
            index_obj = {
                "bm25": None,
                "item_ids": item_ids_list,
                "corpus_tokens": corpus_tokens,
            }
    else:
        # Fallback index
        index_obj = {
            "bm25": None,
            "item_ids": item_ids_list,
            "corpus_tokens": corpus_tokens,
        }
        if BM25Okapi is None:
            logger.info("rank_bm25 not available; using SimpleBM25Index fallback")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        pickle.dump(index_obj, f, protocol=pickle.HIGHEST_PROTOCOL)

    logger.info("BM25 index written to {}", output_path)
    return output_path


# -----------------------------------------------------------------------------
# Dense embeddings + FAISS
# -----------------------------------------------------------------------------

def _load_bge_encoder() -> Optional[SentenceTransformer]:
    """
    Load the BGE encoder (base, en v1.5) with HuggingFace cache hints.

    Returns ``None`` if ``sentence_transformers`` is not available.
    """
    if SentenceTransformer is None:
        logger.warning("sentence_transformers is not available; dense encoder cannot be loaded")
        return None
    _ensure_hf_env()
    logger.info("Loading dense encoder model: {}", BGE_ENCODER_MODEL)
    try:
        model = SentenceTransformer(BGE_ENCODER_MODEL)
        return model
    except Exception as e:
        logger.warning("Failed to load SentenceTransformer model: {}", e)
        return None


def build_dense_index(
    search_texts: Sequence[str],
    item_ids: Sequence[int],
    embeddings_path: Path = EMBEDDINGS_PATH,
    faiss_index_path: Path = FAISS_INDEX_PATH,
    ids_mapping_path: Path = IDS_MAPPING_PATH,
) -> None:
    """
    Build dense embeddings with BGE and a FAISS IndexFlatIP over L2‑normalised vectors.

    Saves:
    - ``embeddings_path``: ``np.ndarray`` float32 (num_items, dim)
    - ``faiss_index_path``: FAISS index or omitted if FAISS unavailable
    - ``ids_mapping_path``: JSON mapping row index -> item_id (list of ints)

    If ``sentence_transformers`` or ``faiss`` is unavailable, this function
    creates placeholder files and logs a warning.  The dense index will
    subsequently be marked as stale and ignored during retrieval.
    """
    logger.info("Building dense embeddings for {} documents", len(search_texts))

    # Attempt to load the encoder; skip if unavailable
    model = _load_bge_encoder()
    if model is None or faiss is None:
        logger.warning(
            "Dense index cannot be built due to missing dependencies; writing placeholders"
        )
        # Create a dummy embeddings array of zeros with a single dimension
        emb = np.zeros((len(search_texts), 1), dtype="float32")
        embeddings_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(embeddings_path, emb, allow_pickle=False)
        # Write mapping
        mapping = list(item_ids)
        ids_mapping_path.parent.mkdir(parents=True, exist_ok=True)
        with ids_mapping_path.open("w", encoding="utf-8") as f:
            json.dump(mapping, f)
        # We do not create a FAISS index when faiss is missing
        return

    # Compute embeddings in batches; normalise vectors for cosine similarity
    try:
        embeddings = model.encode(
            list(search_texts),
            batch_size=64,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
    except Exception as e:
        logger.warning("Failed to compute embeddings: {}; writing placeholders", e)
        emb = np.zeros((len(search_texts), 1), dtype="float32")
        embeddings_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(embeddings_path, emb, allow_pickle=False)
        mapping = list(item_ids)
        ids_mapping_path.parent.mkdir(parents=True, exist_ok=True)
        with ids_mapping_path.open("w", encoding="utf-8") as f:
            json.dump(mapping, f)
        return

    # Ensure float32 dtype
    embeddings = embeddings.astype("float32", copy=False)

    # Save embeddings
    embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(embeddings_path, embeddings, allow_pickle=False)
    logger.info("Saved item embeddings to {}", embeddings_path)

    # Build FAISS index (cosine via inner product over normalised vectors)
    dim = embeddings.shape[1]
    logger.info("Creating FAISS IndexFlatIP with dim={}", dim)
    try:
        index = faiss.IndexFlatIP(dim)  # type: ignore[assignment]
        index.add(embeddings)
        faiss_index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(faiss_index_path))  # type: ignore[attr-defined]
        logger.info("FAISS index written to {}", faiss_index_path)
    except Exception as e:
        logger.warning("Failed to build or save FAISS index: {}", e)

    # Save ids mapping (row index -> item_id)
    mapping = list(item_ids)
    ids_mapping_path.parent.mkdir(parents=True, exist_ok=True)
    with ids_mapping_path.open("w", encoding="utf-8") as f:
        json.dump(mapping, f)
    logger.info("IDs mapping written to {}", ids_mapping_path)


# -----------------------------------------------------------------------------
# Orchestrator
# -----------------------------------------------------------------------------

def build_all_indices(catalog_path: Optional[Path] = None) -> None:
    """
    High-level entrypoint for building both BM25 and dense indices.

    Steps:
    1) Load the catalog snapshot (Parquet) into a DataFrame.
    2) Build the BM25 index.
    3) Build dense embeddings, FAISS index and ID mapping.
    If the catalog is empty the functions will still generate empty
    index artefacts.
    """
    if catalog_path is None:
        df = load_catalog_snapshot()
    else:
        df = load_catalog_snapshot(catalog_path)


    # Even if the snapshot is empty or missing columns, we still want to
    # produce valid index artefacts.  Some environments (e.g. limited
    # connectivity) may result in an empty crawl and thus empty snapshot.
    # Rather than failing outright, we synthesise missing columns and
    # proceed with empty lists.
    if "item_id" not in df.columns:
        logger.warning(
            "Catalog snapshot missing 'item_id' column; assigning sequential IDs"
        )
        # Use the DataFrame index as the item ID
        df = df.copy()
        df["item_id"] = list(range(len(df)))
    if "search_text" not in df.columns:
        logger.warning(
            "Catalog snapshot missing 'search_text' column; filling with empty strings"
        )
        df = df.copy()
        df["search_text"] = ""
    if df.empty:
        logger.warning("Catalog snapshot is empty; indices will be empty")

    search_texts = df["search_text"].astype(str).tolist()
    item_ids = df["item_id"].tolist()

    logger.info("Starting index build for {} catalog items", len(df))

    build_bm25_index(search_texts=search_texts, item_ids=item_ids)
    build_dense_index(search_texts=search_texts, item_ids=item_ids)

    logger.info("All indices built successfully.")


# -----------------------------------------------------------------------------
# BM25 + FAISS loading helpers
# -----------------------------------------------------------------------------

def load_bm25(index_path: Path = BM25_INDEX_PATH):
    """
    Load the BM25 index from disk and return an index object.

    This wrapper calls ``load_bm25_cache`` for compatibility with
    legacy callers that expect the function to be named ``load_bm25``.
    """
    return load_bm25_cache(index_path)


def load_bm25_cache(index_path: Path = BM25_INDEX_PATH):
    """
    Load the BM25 index and item_ids from the saved pickle file.

    Returns an object exposing a ``query`` method that accepts a
    string query and returns a list of (item_id, score) tuples.  If
    the saved object contains a ``BM25Okapi`` instance and the
    ``rank_bm25`` package is available, that instance is used
    directly.  Otherwise a ``SimpleBM25Index`` is constructed from
    the stored tokenised corpus.
    """
    if not index_path.exists():
        raise FileNotFoundError(f"BM25 index not found at {index_path}. Run the index build first.")
    logger.info("Loading BM25 index from {}", index_path)
    with index_path.open("rb") as f:
        data = pickle.load(f)
    bm25_obj = data.get("bm25")
    item_ids: List[int] = data.get("item_ids", [])
    corpus_tokens: List[List[str]] = data.get("corpus_tokens", [])
    if bm25_obj is not None and BM25Okapi is not None:
        logger.info("Loaded rank_bm25 BM25Okapi index with {} items", len(item_ids))
        # The bm25_obj already knows the corpus; we need to wrap it with
        # item_ids to emulate query return format
        class _Wrapper:
            def __init__(self, bm25_model, ids):
                self.bm25_model = bm25_model
                self.ids = ids

            def query(self, text: str, top_n: Optional[int] = None) -> List[Tuple[int, float]]:
                tokens = lexical_tokens_for_bm25(text or "")
                scores = self.bm25_model.get_scores(tokens)
                pairs = list(zip(self.ids, [float(s) for s in scores]))
                pairs.sort(key=lambda x: (-x[1], x[0]))
                if top_n is not None:
                    return pairs[:top_n]
                return pairs

        return _Wrapper(bm25_obj, item_ids)
    else:
        logger.info("Using SimpleBM25Index with {} items", len(item_ids))
        return SimpleBM25Index(corpus_tokens, item_ids)


def load_dense_components(
    faiss_index_path: Path = FAISS_INDEX_PATH,
    ids_mapping_path: Path = IDS_MAPPING_PATH,
) -> Tuple[Optional[object], Optional[object], Dict[str, int], bool]:
    """
    Load the dense index components.

    Returns a tuple ``(model, faiss_index, id_map, dense_stale)`` where

    * ``model`` is a ``SentenceTransformer`` instance or ``None`` if
      unavailable.
    * ``faiss_index`` is a FAISS index object or ``None`` if missing.
    * ``id_map`` maps FAISS row indices to ``item_id`` values.
    * ``dense_stale`` is a flag indicating whether the dense index is
      missing or should be considered stale (in which case the
      retrieval layer should ignore it).
    """
    # id_map must always be loaded to allow MMR and mapping functions to run
    id_map: Dict[str, int] = {}
    if ids_mapping_path.exists():
        try:
            with ids_mapping_path.open("r", encoding="utf-8") as f:
                raw = json.load(f)
            # raw may be a list of ints; convert to string keys
            if isinstance(raw, list):
                # Map the FAISS row index (list position) to the corresponding item_id
                # Use enumerate instead of identity mapping; this fixes the bug where
                # the mapping incorrectly used the ID as both key and value.
                id_map = {str(i): int(v) for i, v in enumerate(raw)}
            elif isinstance(raw, dict):
                id_map = {str(k): int(v) for k, v in raw.items()}
        except Exception as e:
            logger.warning("Failed to load ID mapping: {}", e)
    else:
        logger.warning("IDs mapping file {} does not exist", ids_mapping_path)

    dense_stale = True
    model: Optional[object] = None
    faiss_index_obj: Optional[object] = None

    if SentenceTransformer is not None and faiss is not None:
        # Check existence of required files
        if faiss_index_path.exists() and EMBEDDINGS_PATH.exists():
            try:
                _ensure_hf_env()
                # Load FAISS index
                faiss_index_obj = faiss.read_index(str(faiss_index_path))  # type: ignore[attr-defined]
                # Load embeddings for dimension sanity (but we don't return them)
                embeddings = np.load(EMBEDDINGS_PATH, mmap_mode="r")
                # Load model lazily
                model = _load_bge_encoder()
                if model is not None:
                    dense_stale = False
                logger.info(
                    "Loaded dense index components: faiss index rows={}, embedding dim={}"
                    , embeddings.shape[0], embeddings.shape[1]
                )
            except Exception as e:
                logger.warning("Failed to load dense index components: {}", e)
                dense_stale = True
    else:
        logger.warning("SentenceTransformer or faiss unavailable; dense index marked stale")
        dense_stale = True

    return model, faiss_index_obj, id_map, dense_stale


def embed_query_with_chunking(
    model: SentenceTransformer,
    text: str,
    max_tokens: int = 512,
    chunk_chars: int = 800,
) -> np.ndarray:
    """
    Embed a query string using the provided SentenceTransformer with optional
    chunking for long inputs.

    This function chops long texts into overlapping character chunks,
    encodes each chunk separately and averages the resulting vectors.
    If ``sentence_transformers`` is unavailable, a ``RuntimeError`` is
    raised.  Note that retrieval code only calls this when the dense
    index is not stale.
    """
    if SentenceTransformer is None or model is None:
        raise RuntimeError("Dense embedding model is unavailable; cannot embed queries")
    if not text:
        return np.zeros((model.get_sentence_embedding_dimension(),), dtype="float32")
    # For short texts just encode directly
    if len(text) <= chunk_chars:
        vec = model.encode([text], convert_to_numpy=True, normalize_embeddings=True)[0]
        return vec.astype("float32")
    # Otherwise chunk the text; we operate on characters as a proxy for tokens
    chunks: List[str] = []
    start = 0
    while start < len(text) and len(chunks) < 10:
        chunk = text[start : start + chunk_chars]
        chunks.append(chunk)
        start += chunk_chars // 2  # half overlap
    # Encode each chunk and average the embeddings
    vecs = model.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)
    vec = vecs.mean(axis=0)
    return vec.astype("float32")


# -----------------------------------------------------------------------------
# CLI entrypoint
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # Manual usage:
    # python -m src.embed_index
    build_all_indices()