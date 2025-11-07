from __future__ import annotations

"""
Mapping utilities for the SHL recommender API.

This module converts rows from the catalog DataFrame into strict
Pydantic objects defined in :mod:`src.config`.  It also provides a
convenient wrapper to assemble a full API response from a list of
item IDs.  All transformation logic is encapsulated here to keep
``api.py`` simple.
"""

from typing import Dict, List, Sequence
import re
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

from .config import AssessmentItem, RecommendResponse
from .catalog_build import load_catalog_snapshot


def _normalize_test_type_field(raw) -> List[str]:
    """Robustly convert arbitrary test_type values into a list of strings.

    Handles sequences, numpy arrays, stringified lists (e.g. "['A' 'B']"),
    and comma/semicolon separated strings.  Duplicates are removed while
    preserving order.
    """
    labels: List[str] = []
    # 1) Sequence types (list/tuple/set/ndarray)
    if isinstance(raw, (list, tuple, set)):
        labels = [str(x).strip() for x in raw if str(x).strip()]
    elif isinstance(raw, np.ndarray):
        labels = [str(x).strip() for x in raw.tolist() if str(x).strip()]
    # 2) String types
    elif isinstance(raw, str):
        s = raw.strip()
        if not s:
            labels = []
        else:
            # Try to parse patterns like "['A' 'B']"
            matches = re.findall(r"'([^']+)'", s)
            if matches:
                labels = [m.strip() for m in matches if m.strip()]
            else:
                # Fallback: split on common separators
                parts = re.split(r"[;,/|]+", s)
                labels = [p.strip() for p in parts if p.strip()]
    # 3) Other single values
    else:
        s = str(raw).strip()
        labels = [s] if s else []
    # Deduplicate while preserving order
    seen = set()
    out: List[str] = []
    for lbl in labels:
        if lbl not in seen:
            seen.add(lbl)
            out.append(lbl)
    return out


def to_api_item(row: pd.Series) -> AssessmentItem:
    """Convert one catalog row into an :class:`AssessmentItem`.

    Enforces types and literal flags.  If any required field cannot be
    coerced into the expected type an exception is raised and logged.
    """
    try:
        url = str(row.get("url", "")).strip()
        name = str(row.get("name", "")).strip()
        description = str(row.get("description", "")).strip()
        duration = int(row.get("duration", 0) or 0)
        adaptive_support = str(row.get("adaptive_support", "No")).strip()
        remote_support = str(row.get("remote_support", "No")).strip()
        tt_raw = row.get("test_type", [])
        tt = _normalize_test_type_field(tt_raw)
        item = AssessmentItem(
            url=url,
            name=name,
            description=description,
            duration=max(0, duration),
            adaptive_support="Yes" if adaptive_support == "Yes" else "No",
            remote_support="Yes" if remote_support == "Yes" else "No",
            test_type=tt,
        )
        item.ensure_flags_are_literal()
        return item
    except Exception as e:
        logger.exception("Error mapping row to API item: {}", e)
        raise


def map_items_to_response(
    item_ids: Sequence[int],
    catalog_df: pd.DataFrame | None = None,
) -> RecommendResponse:
    """Convert a list of item IDs into a full RecommendResponse object."""
    if catalog_df is None:
        catalog_df = load_catalog_snapshot()
    id_to_row: Dict[int, pd.Series] = {int(r.item_id): r for _, r in catalog_df.iterrows()}
    items: List[AssessmentItem] = []
    for iid in item_ids:
        if iid not in id_to_row:
            logger.warning("Item id {} not in catalog; skipping", iid)
            continue
        row = id_to_row[iid]
        item = to_api_item(row)
        items.append(item)
    logger.info("Mapped {} items into API schema", len(items))
    return RecommendResponse(recommended_assessments=items)