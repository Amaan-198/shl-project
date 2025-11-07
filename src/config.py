from __future__ import annotations

"""
Centralized configuration constants for the SHL recommender project.

This module defines a variety of file paths, model names, and tuning
parameters used throughout the codebase.  Keeping these values in a
single place makes it easy to adjust defaults without hunting through
the rest of the source.  See comments for guidance on what each
setting controls.
"""

import os
from pathlib import Path
from typing import Dict, List

from pydantic import BaseModel, Field


# ---------------------------
# Paths
# ---------------------------

# Root of the project.  Resolve relative to this file so things work
# both when invoked as a module (python -m src.foo) and when run from
# the repository root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Directory for any data artifacts produced by the crawler or catalog
# build pipeline.  `catalog_raw` holds the original XLSX provided by
# SHL, `catalog_snapshot.parquet` stores a cleaned/normalized copy.
DATA_DIR = PROJECT_ROOT / "data"
CATALOG_RAW_DIR = DATA_DIR / "catalog_raw"
CATALOG_SNAPSHOT_PATH = DATA_DIR / "catalog_snapshot.parquet"
TRAIN_PATH = DATA_DIR / "gen_ai_train.xlsx"
TEST_PATH = DATA_DIR / "gen_ai_test.xlsx"

# Index artifacts.  The BM25 index is a pickle containing both the
# underlying BM25Okapi object and the list of item IDs.  The dense
# index consists of a FAISS index, a NumPy array of embeddings, and
# a JSON mapping from FAISS row indices back to item IDs.
INDICES_DIR = PROJECT_ROOT / "indices"
BM25_INDEX_PATH = INDICES_DIR / "bm25.pkl"
FAISS_INDEX_PATH = INDICES_DIR / "faiss.index"
EMBEDDINGS_PATH = INDICES_DIR / "item_embeddings.npy"
IDS_MAPPING_PATH = INDICES_DIR / "ids.json"

# Optional directory to mount pre‑downloaded HF models.  When running
# offline or in an environment with limited internet access, point
# TRANSFORMERS_CACHE here and set HF_HUB_OFFLINE=1.
MODELS_DIR = PROJECT_ROOT / "models"


# ---------------------------
# Model names (pinned)
# ---------------------------

# Dense encoder used for semantic search.  The BAAI bge‑base‑en
# family strikes a good balance between accuracy and speed for this
# use case.  If you change this, you should also update CHUNK_SIZE
# and CHUNK_STRIDE in config.py to reflect the encoder's maximum
# sequence length.
BGE_ENCODER_MODEL = "BAAI/bge-base-en-v1.5"

# Cross‑encoder reranker used for query/document re‑scoring.  A
# smaller model could be substituted here if runtime costs are a
# concern; the bge‑reranker-base model is sufficiently fast for our
# purposes.
BGE_RERANKER_MODEL = "BAAI/bge-reranker-base"

# Zero‑shot intent classifier used to determine whether a query is
# technical (knowledge & skills) or behavioral/personality.  See
# balance.py for how this value influences final result allocation.
ZERO_SHOT_MODEL = "facebook/bart-large-mnli"


# ---------------------------
# Retrieval & fusion settings
# ---------------------------

BM25_TOP_N = 200          # number of docs to retrieve from the BM25 index
DENSE_TOP_N = 200         # number of docs to retrieve from the dense index
FUSION_TOP_K = 60         # top‑K candidates retained after score fusion

# Relative weight of BM25 vs dense scores.  These values are chosen
# empirically to slightly favor BM25 for precision on short queries.
BM25_WEIGHT = 0.60
DENSE_WEIGHT = 0.40

# Winsorization for fusion scores.  We clip scores into a narrow
# range before computing z‑scores, which prevents extreme values
# from dominating.
FUSION_WINSORIZE_MIN = -3.0
FUSION_WINSORIZE_MAX = 3.0
FUSION_EPS = 1e-8

# MMR diversification.  A higher lambda places more emphasis on
# relevance and less on diversity.
MMR_LAMBDA = 0.60


# ---------------------------
# Result size policy
# ---------------------------

# Minimum and maximum number of results returned by the recommend API.
RESULT_MIN = 5
RESULT_MAX = 10
RESULT_DEFAULT_TARGET = 10


# ---------------------------
# Rerank settings & env toggles
# ---------------------------

DEFAULT_RERANK_CUTOFF = 60
RERANK_CUTOFF = int(os.getenv("RERANK_CUTOFF", str(DEFAULT_RERANK_CUTOFF)))

# HuggingFace environment hints.  HF_HUB_ENABLE_HF_TRANSFER speeds up
# downloads; TRANSFORMERS_CACHE points to a persistent cache; and
# HF_HUB_OFFLINE can be set externally to force offline mode after the
# first run.  See `_ensure_hf_env` in embed_index.py for usage.
HF_ENV_VARS = {
    "HF_HUB_ENABLE_HF_TRANSFER": "1",
    "TRANSFORMERS_CACHE": str(MODELS_DIR),
    # HF_HUB_OFFLINE may be set to "1" outside of this codebase to
    # prevent further network calls once models are cached.
}


# ---------------------------
# Text processing
# ---------------------------

# Maximum size of any text field we process.  Longer inputs will be
# truncated to this many characters to protect against runaway
# downloads or misconfigured endpoints.
MAX_INPUT_CHARS = 20_000

# Token budget for the encoder.  Some models can handle 768 tokens; we
# budget for 512 to leave room for special tokens and avoid OOM.
ENCODER_TOKEN_BUDGET = 512

# When breaking long job descriptions into chunks for dense embedding,
# we use a sliding window over tokens.  CHUNK_SIZE_TOKENS controls
# the window size and CHUNK_STRIDE_TOKENS controls the overlap.
CHUNK_SIZE_TOKENS = 220
CHUNK_STRIDE_TOKENS = 110
CHUNK_TOP_K = 3  # how many top scoring chunks to keep per JD


# ---------------------------
# Synonym map for lexical search
# ---------------------------

# A small, deterministic synonym map used to expand common abbreviations
# into canonical forms before indexing.  See normalize.py for usage.
SYNONYM_MAP: Dict[str, str] = {
    "js": "javascript",
    "ts": "typescript",
    "node": "nodejs",
    "node.js": "nodejs",
    "asp.net": "aspnet",
    "ml": "machine learning",
    "ai": "artificial intelligence",
    "pm": "project manager",
    "stakeholder mgmt": "stakeholder management",

    # --- Custom synonyms for improved lexical retrieval ---
    # Teamwork and collaboration synonyms
    "teamwork": "collaboration",
    "collaboration": "teamwork",
    "cross-functional": "collaboration",
    "stakeholder": "client",
    # Client-facing synonyms
    "client-facing": "customer",
    "customer": "client",
    "client": "stakeholder",
    # Communication and interpersonal skills
    "presentation": "communication",
    "communication": "interpersonal",
    "interpersonal": "communication",
    "verbal": "communication",
    "writing": "communication",
    "listening": "communication",
    # Java/Spring synonyms
    "spring boot": "springboot",
    "springboot": "spring boot",
    "spring": "spring boot",
    "java ee": "spring boot",
    "j2ee": "spring boot",
    "microservices": "spring boot",
}

# Whether to augment token lists with ESCO expansion.  Disabled for
# now; left here for future experimentation.
ENABLE_ESCO_AUGMENT = False


# ---------------------------
# JD fetch / HTTP hardening
# ---------------------------

# Timeout settings for httpx clients used throughout the codebase.
# Keeping these values low prevents the crawler from hanging on slow
# pages.
HTTP_CONNECT_TIMEOUT = 3.0
HTTP_READ_TIMEOUT = 7.0
HTTP_MAX_REDIRECTS = 2
HTTP_MAX_BYTES = 1_000_000  # 1 MB cap on downloaded pages

# User‑agent string presented during HTTP requests.  Identify our
# crawler politely and provide a contact in case of issues.
HTTP_USER_AGENT = (
    "shl-rag-recommender/1.0 (+https://example.com; contact=genai@placeholder.com)"
)


# ---------------------------
# Zero‑shot smoothing (balance calibration)
# ---------------------------

INTENT_TEMP = 1.5        # temperature for softmax over intent logits
INTENT_SMOOTH_EPS = 0.15  # mix with uniform prior
INTENT_CLIP_MIN = 0.20    # floor after smoothing
INTENT_CLIP_MAX = 0.80    # ceiling after smoothing


# ---------------------------
# K/P balance policy
# ---------------------------

INTENT_LABEL_TECHNICAL = "technical skills / knowledge"
INTENT_LABEL_PERSONALITY = "personality / behavior"
INTENT_LABELS: List[str] = [INTENT_LABEL_TECHNICAL, INTENT_LABEL_PERSONALITY]

# Soft bonus for items whose test_type matches dominant intent.  See
# balance.py for usage.
INTENT_SOFT_BONUS = 0.06


# ---------------------------
# Crawl controls
# ---------------------------

# Maximum number of pages to crawl in the catalog.  This protects
# against infinite pagination loops if the site changes unexpectedly.
MAX_CATALOG_PAGES = 100
HTTP_RETRY_READS = 1


# Thresholds for the result allocator (see balance.py).  Kept here
# because they are part of the public API of the recommender.
BOTH_HIGH_THRESHOLD = 0.45
DOMINANT_THRESHOLD = 0.60
SECONDARY_MIN_FOR_SPLIT = 0.30

BALANCE_5_5_SIZE = 10
BALANCE_7_3_SIZE = 10


# ---------------------------
# Logging / observability
# ---------------------------

# Directory where logs can be written.  The API server can be
# configured to use this, and ad‑hoc scripts will also write here.
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)


# ---------------------------
# Pydantic models shared around the app
# ---------------------------

class AssessmentItem(BaseModel):
    """
    Canonical schema for a single recommended assessment.

    This matches the API contract exactly; any changes here must be
    reflected in the OpenAPI schema exposed by src/api.py.
    """

    url: str
    name: str
    description: str
    duration: int = Field(ge=0)
    adaptive_support: str  # "Yes"/"No"
    remote_support: str    # "Yes"/"No"
    test_type: List[str]

    def ensure_flags_are_literal(self) -> None:
        """
        Safety check: normalize adaptive/remote flags strictly to
        uppercase literal 'Yes' or 'No'.  This can be used if
        upstream code accidentally introduces other values.
        """
        self.adaptive_support = "Yes" if self.adaptive_support == "Yes" else "No"
        self.remote_support = "Yes" if self.remote_support == "Yes" else "No"


class RecommendResponse(BaseModel):
    """
    Response body for POST /recommend.  A list of recommendations is
    nested under the `recommended_assessments` key so additional
    metadata could be added to the response without breaking clients.
    """

    recommended_assessments: List[AssessmentItem]


class HealthResponse(BaseModel):
    """
    Response body for GET /health.  Kept separate to allow future
    extension.
    """

    status: str
