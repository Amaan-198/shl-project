from __future__ import annotations

"""
Simple allocation logic for balancing final recommendation sets.

After the reranking and diversification steps the recommender needs to
assemble a short list of assessment IDs that respects the user's
inferred intent (knowledge & skills versus personality & behavior).
This module encodes a handful of heuristics for splitting the list
into K/P buckets and then filling any remaining slots while filtering
out obviously off‑domain items.

The logic here closely follows the original project.  We expose a
single :func:`allocate` function that takes a ranked list of item IDs
with associated class labels and returns a final list of IDs sized
between ``RESULT_MIN`` and ``RESULT_MAX``.  Denied substrings can be
configured via the ``DENY_SUBSTR`` list.
"""

from typing import Dict, List
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

from .config import RESULT_MIN, RESULT_MAX

# Very small denylist for SWE‑ish queries; keep conservative
DENY_SUBSTR = [
    "accounts payable",
    "accounts receivable",
    "pharmaceutical",
    "svar",
    "spoken english",
    "spoken french",
    "spoken spanish",
]


def _looks_offdomain(name: str, desc: str) -> bool:
    """Return True if a name/description appears to be off‑domain."""
    text = f"{name} {desc}".lower()
    return any(s in text for s in DENY_SUBSTR)


def allocate(
    item_ids: List[int],
    item_classes: Dict[int, List[str]],
    target_size: int,
    *,
    pt: float,
    pb: float,
    catalog_df=None,  # optional: for name/desc filter
) -> List[int]:
    """Allocate items into a final result list based on inferred intent.

    ``item_ids`` should be in relevance order and may contain IDs
    belonging to multiple classes.  ``item_classes`` maps each ID to
    a list of test type labels.  The caller provides ``pt`` and
    ``pb``, which are probabilities for the knowledge/skills and
    personality/behavior intents respectively.  The algorithm
    computes K/P targets and then fills the final list accordingly.

    An off‑domain filter is applied during backfill to weed out
    obviously irrelevant items.  If too few items remain the list is
    padded up to ``RESULT_MIN``.  The final result is capped at
    ``RESULT_MAX``.
    """
    BOTH_HIGH_THRESHOLD = 0.7
    DOMINANT_THRESHOLD = 0.75
    SECONDARY_MIN_FOR_SPLIT = 0.35
    # Partition IDs by class
    k_items = [i for i in item_ids if "Knowledge & Skills" in item_classes.get(i, [])]
    p_items = [i for i in item_ids if "Personality & Behavior" in item_classes.get(i, [])]
    both_items = [i for i in item_ids if i in k_items and i in p_items]
    # Dedup preserve order
    k_items = list(dict.fromkeys(k_items))
    p_items = list(dict.fromkeys(p_items))
    # Determine targets based on intent probabilities
    if pt >= BOTH_HIGH_THRESHOLD and pb >= BOTH_HIGH_THRESHOLD:
        k_target = p_target = target_size // 2
    elif (pt >= DOMINANT_THRESHOLD and pb >= SECONDARY_MIN_FOR_SPLIT) or (
        pb >= DOMINANT_THRESHOLD and pt >= SECONDARY_MIN_FOR_SPLIT
    ):
        k_target, p_target = (7, 3) if pt > pb else (3, 7)
    else:
        k_target = p_target = target_size // 2
    logger.info(
        "Alloc targets: k_target=%d, p_target=%d, (pt=%.2f, pb=%.2f)",
        k_target,
        p_target,
        pt,
        pb,
    )
    selected: List[int] = []
    # Prefer BOTH early when filling buckets (it supports both intents)
    for i in both_items:
        if len([x for x in selected if x in k_items]) >= k_target and len(
            [x for x in selected if x in p_items]
        ) >= p_target:
            break
        if i not in selected:
            selected.append(i)
    # Fill K then P
    for i in k_items:
        if len([x for x in selected if x in k_items]) >= k_target:
            break
        if i not in selected:
            selected.append(i)
    for i in p_items:
        if len([x for x in selected if x in p_items]) >= p_target:
            break
        if i not in selected:
            selected.append(i)
    # Backfill from remaining, but skip obvious off‑domain if we have catalog_df
    def _ok(i: int) -> bool:
        if catalog_df is None:
            return True
        if i not in catalog_df.index:
            return True
        row = catalog_df.loc[i]
        return not _looks_offdomain(str(row.get("name", "")), str(row.get("description", "")))
    for i in item_ids:
        if len(selected) >= target_size:
            break
        if i not in selected and _ok(i):
            selected.append(i)
    # Still short? allow anything to meet minimum
    if len(selected) < RESULT_MIN:
        for i in item_ids:
            if len(selected) >= RESULT_MIN:
                break
            if i not in selected:
                selected.append(i)
    # Trim to maximum
    if len(selected) > RESULT_MAX:
        selected = selected[:RESULT_MAX]
    return selected