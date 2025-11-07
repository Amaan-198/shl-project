from __future__ import annotations

"""
FastAPI application for the SHL recommender.

This module exposes two endpoints: a basic health check at ``/health``
and a ``POST /recommend`` endpoint that accepts a free‑form query and
returns a list of recommended assessment items.  The pipeline covers
retrieval, reranking, diversification via MMR, and allocation based on
intent classification.  Optional components fall back gracefully when
their dependencies or artifacts are missing.
"""

from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
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
from pydantic import BaseModel, Field

from .config import RESULT_MAX
from .catalog_build import load_catalog_snapshot
from .retrieval import retrieve_candidates
from .rerank import rerank_candidates
from .mmr import load_item_embeddings, mmr_select
from .balance import allocate
from .mapping import map_items_to_response


# -----------------------------------------------------------------------------
# Heuristics for domain filtering and generic item penalisation
#
# The retrieval pipeline sometimes returns assessments from unrelated
# domains or overly generic items (e.g. "AI Skills", "360 Feedback").
# To improve precision, the API applies a lightweight domain filter and
# a penalty for generic assessments before diversification and final
# allocation.  These heuristics can be tuned based on offline
# evaluation and are intentionally simple to avoid heavy dependencies.

# Keywords that indicate the query is targeting technical or software roles.
_TECH_KEYWORDS = [
    "software", "developer", "programmer", "coder", "engineer", "technician",
    "technology", "technical", "coding", "programming", "devops",
]

# Allowed test types for technical roles.  Only items with one of these
# types will be retained when the query matches _TECH_KEYWORDS.  If no
# candidates remain after filtering, the original list is used.
_TECH_ALLOWED_TYPES = {"Knowledge & Skills", "Ability & Aptitude"}

# Generic item name substrings to penalize.  These items often appear
# across diverse queries and are less specific to the query intent.
_GENERIC_PATTERNS = [
    "ai skills", "360", "verify", "inductive reasoning", "360 feedback",
]

# Additional language codes considered non‑English for filtering.  When a
# query does not explicitly mention the language, items containing
# these substrings in their name or description incur a penalty.
_NON_EN_LANGUAGES = [
    "spanish", "french", "german", "mandarin", "chinese", "arabic",
    "hindi", "japanese", "portuguese", "italian", "sv", "svenska",
]

# Allowed types for client‑facing roles
_CLIENT_ALLOWED_TYPES = {
    "Biodata & Situational Judgement",
    "Personality & Behaviour",
    "Knowledge & Skills",
}

# Keywords indicating entry‑level roles
_ENTRY_LEVEL_KEYWORDS = ["entry-level", "entry level", "graduate", "intern", "internship"]

# Positive and negative patterns for entry‑level bias adjustments
_ENTRY_LEVEL_POSITIVE = ["verify g+", "inductive", "numerical", "multitasking"]
_ENTRY_LEVEL_NEGATIVE = ["expert", "senior", "advanced"]

# Domain keywords for off-topic penalty.  If a candidate's name contains
# one of these and the query lacks it, the candidate is penalised.
_DOMAIN_KEYWORDS = [
    "food", "beverage", "hospitality", "accounting", "retail", "filing", "front office",
    "office management", "restaurants", "hotel", "pharmaceutical", "insurance",
    "sales", "marketing", "customer service", "support", "filling", "warehouse",
]

# AI/ML keywords for domain boosting and filtering
_AI_KEYWORDS = [
    "artificial intelligence", "ai", "machine learning", "ml", "deep learning",
    "data science", "neural network", "computer vision", "nlp", "natural language",
]

# Additional domain keyword groups for focused query matching.  These
# are used to detect the dominant subject of a query and to
# favour candidates that belong to that domain.  The keys are
# descriptive only and not used in logic directly; values are lists
# of lowercase substrings that signal the domain.
_DOMAIN_FOCUS_KEYWORDS = {
    "analytics": [
        "analytics", "data analysis", "business data", "analyze", "analyse",
        "data-driven", "reporting", "insight", "data insights", "data interpretation",
    ],
    "communication": [
        "communication", "writing", "presentation", "interpersonal", "client communication",
        "collaboration", "stakeholder management", "storytelling", "articulation",
    ],
    "sales": [
        "sales", "negotiation", "customer", "service orientation", "customer service",
        "client-facing", "selling", "retail", "marketing",
    ],
    # Note: AI/ML keywords are handled separately via _AI_KEYWORDS
}

# Names of assessments that frequently appear in off-target contexts.  If
# these names show up and the query does not explicitly mention the
# domain, they receive a small penalty to discourage them from
# occupying top positions.  Keep the list lowercase for case-insensitive
# matching.  This list should be curated based on observed noise.
_COMMON_IRRELEVANT_PATTERNS = [
    "filing - names", "filing - numbers", "food science", "food and beverage", "front office management",
    "following instructions", "written english", "filling", "office management", "office operations",
]

# Mapping of high-level categories to test type labels
_TYPE_CATEGORY_MAP = {
    "Knowledge & Skills": "technical",
    "Ability & Aptitude": "aptitude",
    "Personality & Behavior": "behaviour",
    "Biodata & Situational Judgement": "behaviour",
    "Simulations": "behaviour",
}

# Keyword hints for query intent categories
_INTENT_KEYWORDS = {
    "technical": _TECH_KEYWORDS,
    "behaviour": [
        "communication", "interpersonal", "presentation", "leadership",
        "teamwork", "collaboration", "stakeholder", "client", "customer",
        "soft skills", "relationship", "partner", "consultant", "empathy",
        "negotiation", "service", "orientation", "sales", "creative",
    ],
    "aptitude": [
        "analytical", "reasoning", "logic", "logical", "numerical", "inductive",
        "aptitude", "problem solving", "quantitative", "cognitive",
    ],
}


def _normalize_basename(name: str) -> str:
    """Return a simplified base name for duplicate detection.

    Lowercases, removes punctuation and spaces.  Used to penalize
    near‑duplicate items that differ only in small details (e.g. phone
    vs. solution).
    """
    import re
    base = name.lower()
    base = re.sub(r"[\s&/\-]+", "", base)
    base = re.sub(r"[^a-z0-9]", "", base)
    return base

def _post_rank_adjustments(
    ranked: List, query: str, catalog_df
) -> List:
    """Apply various heuristic adjustments to candidate scores.

    Adjustments include:
    - Duration penalty for items lacking a time estimate (duration==0)
    - Name penalty for meta reports/guides/profiles
    - Intent bucket bonuses based on query keywords and test type
    - Language filtering penalty for off‑language items
    - Entry‑level bias adjustments
    - Near‑duplicate dampening

    Returns a new list of Candidate objects sorted by updated
    ``rerank_score`` descending.
    """
    q_lower = query.lower()
    # Determine if entry‑level query
    is_entry = any(k in q_lower for k in _ENTRY_LEVEL_KEYWORDS)
    # Determine if query is client facing (non‑tech but interpersonal)
    is_client = any(k in q_lower for k in ["client", "customer", "stakeholder", "presentation", "communication", "teamwork", "collaboration"])
    adjusted: List = []
    seen_bases = {}
    for c in ranked:
        iid = c.item_id
        try:
            row = catalog_df.loc[iid]
        except Exception:
            row = {}
        score = c.rerank_score
        # Duration penalty
        try:
            duration = float(row.get("duration", 0))
        except Exception:
            duration = 0.0
        if duration == 0:
            score -= 0.06
        # Name penalty for reports/guides/profiles
        try:
            name = str(row.get("name", ""))
        except Exception:
            name = ""
        if any(word in name.lower() for word in ["report", "guide", "profile"]):
            score -= 0.03
        # Intent type bonus
        try:
            types = row.get("test_type", [])
        except Exception:
            types = []
        # Normalise types to list
        if isinstance(types, str):
            cleaned = types.replace("[", "").replace("]", "").replace("'", "")
            types_list = [t.strip() for t in cleaned.split(",") if t.strip()]
        elif isinstance(types, (list, tuple)):
            types_list = [str(t).strip() for t in types if str(t).strip()]
        else:
            try:
                types_list = [str(t).strip() for t in list(types) if str(t).strip()]
            except Exception:
                types_list = [str(types).strip()] if types else []
        # Tech bucket bonus
        if any(k in q_lower for k in _TECH_KEYWORDS):
            if any(t in _TECH_ALLOWED_TYPES for t in types_list):
                score += 0.08
        # Client‑facing bucket bonus
        if is_client:
            if any(t in _CLIENT_ALLOWED_TYPES for t in types_list):
                score += 0.08
        # Language penalty
        # Only penalize if the query does not mention the language
        if not any(lang in q_lower for lang in _NON_EN_LANGUAGES):
            # Check name and description for language indicators
            try:
                desc = str(row.get("description", ""))
            except Exception:
                desc = ""
            combined = (name + " " + desc).lower()
            if any(lang in combined for lang in _NON_EN_LANGUAGES):
                score -= 0.08
        # Entry‑level adjustments
        if is_entry:
            lname = name.lower()
            # Positive boosts for certain families
            if any(pat in lname for pat in _ENTRY_LEVEL_POSITIVE):
                score += 0.06
            # Small penalty for senior/niche tech batteries
            if any(pat in lname for pat in _ENTRY_LEVEL_NEGATIVE):
                score -= 0.03
        # Domain keyword penalty: if candidate name contains a domain keyword
        # and the query does not mention it, apply a small penalty
        if name:
            lname = name.lower()
            for kw in _DOMAIN_KEYWORDS:
                if kw in lname and kw not in q_lower:
                    score -= 0.05
                    break
        # Near‑duplicate dampening
        base = _normalize_basename(name)
        if base:
            if base in seen_bases:
                # penalise duplicates slightly
                score -= 0.05
            else:
                seen_bases[base] = True
        # AI/ML domain boosting and penalty
        ai_query = any(kw in q_lower for kw in _AI_KEYWORDS)
        if ai_query:
            name_desc = (name + " " + desc).lower()
            if any(kw in name_desc for kw in _AI_KEYWORDS):
                score += 0.08
            else:
                score -= 0.05

        # Generic domain focus boosting and penalty
        # Detect dominant domain(s) in the query (excluding AI/ML which is handled above)
        query_domains: set[str] = set()
        for dom, kws in _DOMAIN_FOCUS_KEYWORDS.items():
            if any(k in q_lower for k in kws):
                query_domains.add(dom)
        # If a domain is detected, prefer candidates whose name/description contains
        # keywords from that domain.  Apply a boost or penalty accordingly.
        if query_domains:
            name_desc_lc = (name + " " + desc).lower()
            # Determine if candidate matches any detected domain
            matches_domain = False
            for dom in query_domains:
                if any(kw in name_desc_lc for kw in _DOMAIN_FOCUS_KEYWORDS.get(dom, [])):
                    matches_domain = True
                    break
            # Apply modest adjustments to steer results
            if matches_domain:
                score += 0.05
            else:
                score -= 0.05

        # Penalise known common irrelevant items when off-domain
        # Only apply when the query does not explicitly mention the pattern
        name_lc = name.lower()
        if not any(pat in q_lower for pat in _COMMON_IRRELEVANT_PATTERNS):
            for pat in _COMMON_IRRELEVANT_PATTERNS:
                if pat in name_lc:
                    score -= 0.05
                    break
        adjusted.append(type(c)(item_id=c.item_id, fused_score=c.fused_score, rerank_score=score))
    # Sort adjusted list by score desc then id
    # Tie-break by shorter duration, then shorter name length, then item ID
    adjusted.sort(
        key=lambda c: (
            -float(c.rerank_score),
            catalog_df.loc[c.item_id].get("duration", 0) if c.item_id in catalog_df.index else 0,
            len(str(catalog_df.loc[c.item_id].get("name", ""))) if c.item_id in catalog_df.index else 0,
            c.item_id,
        )
    )
    return adjusted


def _get_query_intent_categories(query: str) -> set[str]:
    """Infer high-level intent categories (technical, behaviour, aptitude) from the query."""
    q_lower = query.lower()
    cats: set[str] = set()
    for cat, keywords in _INTENT_KEYWORDS.items():
        if any(k in q_lower for k in keywords):
            cats.add(cat)
    return cats


def _categories_for_item(row) -> set[str]:
    """Map an item's test_type list to one or more high-level categories."""
    cats: set[str] = set()
    types = row.get("test_type", [])
    # normalise types to list of strings
    if isinstance(types, str):
        cleaned = types.replace("[", "").replace("]", "").replace("'", "")
        types_list = [t.strip() for t in cleaned.split(",") if t.strip()]
    elif isinstance(types, (list, tuple)):
        types_list = [str(t).strip() for t in types if str(t).strip()]
    else:
        try:
            types_list = [str(t).strip() for t in list(types) if str(t).strip()]
        except Exception:
            types_list = [str(types).strip()] if types else []
    for t in types_list:
        cat = _TYPE_CATEGORY_MAP.get(t)
        if cat:
            cats.add(cat)
    return cats


def _apply_category_balance(ranked: List, query: str, catalog_df) -> List:
    """Ensure the ranked list includes items from each intent category present in the query.

    When the query mentions multiple intent categories (e.g. technical and behaviour),
    this function ensures that at least one item from each category appears in the top
    results by selecting the first occurrence of the missing category from the ranked list.
    The returned list maintains the original order as far as possible.
    """
    needed_cats = _get_query_intent_categories(query)
    if not needed_cats or len(needed_cats) == 1:
        return ranked
    # Determine categories already present in the ranked list
    present: set[str] = set()
    for c in ranked[:RESULT_MAX]:
        try:
            row = catalog_df.loc[c.item_id]
        except Exception:
            continue
        present |= _categories_for_item(row)
    missing = needed_cats - present
    if not missing:
        return ranked
    # For each missing category, find the first candidate in the ranked list belonging to it
    to_promote: List = []
    for cat in missing:
        for c in ranked:
            try:
                row = catalog_df.loc[c.item_id]
            except Exception:
                continue
            if cat in _categories_for_item(row):
                to_promote.append(c)
                break
    # Prepend promoted items if not already in top list
    new_ranked: List = []
    promoted_ids = {c.item_id for c in to_promote}
    # Add promoted items first in their original relative order
    for c in ranked:
        if c.item_id in promoted_ids and c not in new_ranked:
            new_ranked.append(c)
    # Then add remaining ranked items
    for c in ranked:
        if c not in new_ranked:
            new_ranked.append(c)
    return new_ranked


def _apply_category_filter(ranked: List, query: str, catalog_df) -> List:
    """Drop items that are only technical when behavioural or aptitude cues are present.

    If the query indicates a need for behavioural or aptitude assessments, this
    filter removes candidates whose category is solely 'technical' (i.e.
    Knowledge & Skills) unless there are insufficient non-technical items.
    """
    query_cats = _get_query_intent_categories(query)
    # If the query includes behavioural or aptitude cues, filter out technical-only items
    if any(cat in query_cats for cat in ["behaviour", "aptitude"]):
        filtered: List = []
        for c in ranked:
            try:
                row = catalog_df.loc[c.item_id]
            except Exception:
                filtered.append(c)
                continue
            cats = _categories_for_item(row)
            # Keep if candidate has non-technical category or multiple categories
            if "technical" not in cats or len(cats) > 1:
                filtered.append(c)
        # Ensure we don't return an empty list
        if filtered:
            return filtered
    return ranked


def _apply_dynamic_cutoff(final_ids: List[int], ranked_scores: dict[int, float]) -> List[int]:
    """Adaptive result count: keep items with normalized score ≥ 0.55 while ensuring 5–10 results.

    ``final_ids`` is the current list of recommended item IDs in order, and
    ``ranked_scores`` maps item_id to its final rerank score.

    The function normalizes scores, filters by a threshold of 0.55 and caps the
    result list between 5 and 10 items.  If fewer than 5 items remain after
    thresholding, the top 5 are retained.  If more than 10 remain, the top 10
    are returned.
    """
    scores = [ranked_scores.get(i, 0.0) for i in final_ids]
    if not scores:
        return final_ids
    mn, mx = min(scores), max(scores)
    # Avoid division by zero
    rng = mx - mn if mx > mn else 1.0
    normalized = [(s - mn) / rng for s in scores]
    # Apply cutoff
    # Increase threshold slightly to filter weaker matches; if too many are filtered,
    # fallback to the min count.  A threshold of 0.60 empirically balances
    # relevance and diversity without eliminating too many candidates.
    kept: List[int] = [iid for iid, n in zip(final_ids, normalized) if n >= 0.60]
    if len(kept) < 5:
        kept = final_ids[:5]
    elif len(kept) > 10:
        kept = kept[:10]
    return kept


def _ensure_min_category_diversity(
    final_ids: List[int], ranked: List, catalog_df, min_categories: int = 2
) -> List[int]:
    """Ensure that the final list includes assessments from at least `min_categories` distinct categories.

    If fewer categories are present, the function searches the ranked candidate list
    for the first item whose category is not yet represented and appends it.  The
    list is capped to RESULT_MAX length.
    """
    # Determine categories present in final_ids
    present: set[str] = set()
    for iid in final_ids:
        try:
            row = catalog_df.loc[iid]
        except Exception:
            continue
        present |= _categories_for_item(row)
    # If enough categories already, return as is
    if len(present) >= min_categories:
        return final_ids
    # Search ranked candidates for missing categories
    for c in ranked:
        if len(present) >= min_categories:
            break
        if c.item_id in final_ids:
            continue
        try:
            row = catalog_df.loc[c.item_id]
        except Exception:
            continue
        cats = _categories_for_item(row)
        # Add if introduces a new category
        if not cats:
            continue
        new_cats = cats - present
        if new_cats:
            final_ids.append(c.item_id)
            present |= new_cats
    # Trim to RESULT_MAX
    if len(final_ids) > RESULT_MAX:
        final_ids = final_ids[:RESULT_MAX]
    return final_ids

def _filter_domain_candidates(
    query: str, ranked: List, catalog_df
) -> List:
    """Filter ranked candidates by domain heuristics.

    If the query contains technical keywords, only retain items whose
    ``test_type`` intersects ``_TECH_ALLOWED_TYPES``.  If the query
    does not match technical keywords or the filter removes all
    candidates, the original list is returned unchanged.
    """
    q_lower = query.lower()
    if not any(k in q_lower for k in _TECH_KEYWORDS):
        return ranked
    filtered: List = []
    for c in ranked:
        try:
            types = catalog_df.loc[c.item_id, "test_type"]
        except Exception:
            types = []
        # Normalise to list
        if isinstance(types, str):
            cleaned = types.replace("[", "").replace("]", "").replace("'", "")
            types_list = [t.strip() for t in cleaned.split(",") if t.strip()]
        elif isinstance(types, (list, tuple)):
            types_list = [str(t).strip() for t in types if str(t).strip()]
        else:
            # Fallback: attempt to iterate or coerce to single string
            try:
                types_list = [str(t).strip() for t in list(types) if str(t).strip()]
            except Exception:
                types_list = [str(types).strip()] if types else []
        if any(t in _TECH_ALLOWED_TYPES for t in types_list):
            filtered.append(c)
    return filtered or ranked


def _apply_generic_penalty(ranked: List, catalog_df) -> List:
    """Apply a downweighting penalty to generic items in the ranked list.

    Items whose names contain substrings from ``_GENERIC_PATTERNS`` have
    their ``rerank_score`` multiplied by 0.7.  The list is then
    resorted by the updated rerank scores.  If an item has no name
    entry it is left unchanged.
    """
    penalised: List = []
    for c in ranked:
        try:
            name = str(catalog_df.loc[c.item_id, "name"]).lower()
        except Exception:
            name = ""
        score = c.rerank_score
        if any(pat in name for pat in _GENERIC_PATTERNS):
            score *= 0.7
        penalised.append(type(c)(item_id=c.item_id, fused_score=c.fused_score, rerank_score=score))
    penalised.sort(key=lambda c: (-float(c.rerank_score), c.item_id))
    return penalised


class QueryRequest(BaseModel):
    """Request model for the recommend endpoint."""

    query: str = Field(..., min_length=1)


class HealthResponse(BaseModel):
    """Simple health check response."""

    status: str


app = FastAPI()

# Open CORS for development; tighten in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cache catalog snapshot in process to avoid repeated disk I/O
_catalog_df = None  # type: ignore


@app.on_event("startup")
def startup_event() -> None:
    """Load the catalog and warm up indices on startup."""
    global _catalog_df
    logger.info("Starting app warmup...")
    _catalog_df = load_catalog_snapshot()
    logger.info("Loaded catalog snapshot with {} rows", len(_catalog_df))
    # Warm‑load indices/models so first request is fast
    try:
        from .retrieval import _load_retrieval_components
        _load_retrieval_components()
        from .mmr import load_item_embeddings as _load_emb
        _load_emb()
    except Exception as e:
        logger.warning("Warmup partial failure: {}", e)
    logger.info("Warmup complete.")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Return a basic health status."""
    return HealthResponse(status="healthy")


@app.post("/recommend")
def recommend(req: QueryRequest):
    """Endpoint to get assessment recommendations for a query."""
    query = req.query.strip()
    if not query:
        raise HTTPException(status_code=422, detail="Query must be non‑empty")
    raw_query, cleaned_query, fused = retrieve_candidates(query)
    ranked = rerank_candidates(cleaned_query, fused)
    # Apply domain filtering heuristics
    ranked = _filter_domain_candidates(cleaned_query, ranked, _catalog_df)
    # Downrank generic items
    ranked = _apply_generic_penalty(ranked, _catalog_df)
    # Apply post‑rank heuristic adjustments (duration/name/language/entry-level)
    ranked = _post_rank_adjustments(ranked, cleaned_query, _catalog_df)
    # Apply category balance to ensure representation of all intent categories
    ranked = _apply_category_balance(ranked, cleaned_query, _catalog_df)
    # Drop purely technical items when behavioural or aptitude cues present
    ranked = _apply_category_filter(ranked, cleaned_query, _catalog_df)
    # Diversify with MMR
    embeddings, ids = load_item_embeddings()
    mmr_ids = mmr_select(
        candidates=[(c.item_id, c.rerank_score) for c in ranked],
        embeddings=embeddings,
        ids=ids,
        k=RESULT_MAX,
        lambda_=0.7,
    )
    # Detect intent (K vs P) and allocate.  Here we stub out the intent
    # classifier probabilities with uniform values (pt=pb=0.5) since the
    # zero‑shot model is not loaded in this environment.  Replace with
    # real intent scores if available.
    pt = pb = 0.5
    # Log the deny substrings configured in balance.py.  Use a direct f-string
    # rather than brace formatting to avoid confusion with our fallback logger.
    from .balance import DENY_SUBSTR
    logger.info(f"Domain deny substrings: {DENY_SUBSTR}")
    # Build class map for allocator from the snapshot
    classes: dict[int, List[str]] = {}
    for iid in mmr_ids:
        row = _catalog_df.loc[iid]
        raw_types = row.get("test_type")
        types_list: List[str] = []
        # Normalise the test_type field into a list of strings without relying on truthiness
        try:
            import numpy as _np  # local import to avoid global dependency
        except Exception:
            _np = None  # type: ignore
        # Determine emptiness without using pandas.isna to avoid elementwise ambiguity
        if raw_types is None or (
            isinstance(raw_types, float) and (raw_types != raw_types)
        ):
            types_list = []
        elif isinstance(raw_types, str):
            cleaned = raw_types.replace("[", "").replace("]", "").replace("'", "")
            types_list = [t.strip() for t in cleaned.split(",") if t.strip()]
        elif isinstance(raw_types, (list, tuple)):
            types_list = [str(t).strip() for t in raw_types if str(t).strip()]
        else:
            # Try to coerce other iterables (e.g. numpy array) to list
            try:
                types_list = [str(t).strip() for t in list(raw_types) if str(t).strip()]
            except Exception:
                # Fallback: if raw_types isn't iterable, coerce to string unless NaN
                if raw_types is not None and not (
                    isinstance(raw_types, float) and (raw_types != raw_types)
                ):
                    types_list = [str(raw_types).strip()]
                else:
                    types_list = []
        classes[iid] = types_list
    final_ids = allocate(
        mmr_ids,
        classes,
        RESULT_MAX,
        pt=pt,
        pb=pb,
        catalog_df=_catalog_df,
    )
    # Build score lookup for dynamic cutoff
    score_lookup = {c.item_id: c.rerank_score for c in ranked}
    # Apply dynamic cutoff to final_ids based on normalized scores
    final_ids = _apply_dynamic_cutoff(final_ids, score_lookup)
    # Ensure we have at least two distinct categories represented
    final_ids = _ensure_min_category_diversity(final_ids, ranked, _catalog_df, min_categories=2)
    response = map_items_to_response(final_ids, _catalog_df)
    # If more than 10 items remain after cutoff/balancing, trim to RESULT_MAX
    if len(response.recommended_assessments) > RESULT_MAX:
        response.recommended_assessments = response.recommended_assessments[:RESULT_MAX]
    return response