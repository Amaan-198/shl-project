from __future__ import annotations

"""
Text normalization utilities used across the SHL recommender project.

These helpers perform basic cleaning (HTML stripping, unicode
normalization, whitespace collapsing), lexical tokenization for
BM25 indexing, and synonym expansion.  Keeping normalization logic
centralized here ensures consistent treatment of user queries and
catalog content.
"""

import re
import unicodedata
from typing import Iterable, List

from bs4 import BeautifulSoup

from .config import MAX_INPUT_CHARS, SYNONYM_MAP


# ---------------------------
# Basic helpers
# ---------------------------

def clamp_text_length(text: str, max_chars: int = MAX_INPUT_CHARS) -> str:
    """
    Hard cap on input size so we don't accidentally feed huge strings
    into models.
    """
    if not isinstance(text, str):
        text = str(text)
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


def strip_html(raw: str) -> str:
    """
    Strip HTML tags using BeautifulSoup, then clean up whitespace and
    spacing around punctuation.  If parsing fails, the input is
    returned unchanged to fail open rather than drop text.
    """
    if not raw:
        return ""
    # Fast path: if there's no '<', it's almost certainly not HTML
    if "<" not in raw:
        return raw

    try:
        soup = BeautifulSoup(raw, "lxml")
        text = soup.get_text(" ", strip=True)
        # Normalize whitespace first
        text = normalize_whitespace(text)
        # Remove spaces before common punctuation marks
        text = re.sub(r"\s+([.,!?;:])", r"\1", text)
        return text
    except Exception:
        # Fail open: return the input as-is
        return raw


def normalize_unicode(text: str) -> str:
    """
    Normalize weird unicode (fancy quotes, etc.) into a more stable
    form.  Using NFC keeps things mostly intact but canonicalized.
    """
    if not text:
        return ""
    return unicodedata.normalize("NFC", text)


def normalize_whitespace(text: str) -> str:
    """
    Collapse all whitespace runs into a single space and strip edges.
    """
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------
# Tokenization & synonyms
# ---------------------------

# Allow alphanumerics, underscore, +, # (for things like C#, C++, etc.)
WORD_SPLIT_RE = re.compile(r"[^\w#+]+")


def simple_tokenize(text: str) -> List[str]:
    """
    Simple, fast tokenization for BM25 / lexical use.  Lowercase and
    split on non-word separators, keeping C#, C++ etc roughly intact.
    Returns a list of tokens with empty strings removed.
    """
    if not text:
        return []
    text = text.lower()
    tokens = [t for t in WORD_SPLIT_RE.split(text) if t]
    return tokens


def apply_synonyms(tokens: Iterable[str]) -> List[str]:
    """
    Apply a small, deterministic synonym map.  E.g. 'js' ->
    'javascript', 'node.js' -> 'nodejs', etc.  If a synonym expands to
    multiple words, we split them and inline into the token list.
    """
    normalized: List[str] = []
    for tok in tokens:
        key = tok.lower()
        replacement = SYNONYM_MAP.get(key)
        if replacement:
            # Allow multi-word expansions like "machine learning"
            for sub in replacement.split():
                sub = sub.strip()
                if sub:
                    normalized.append(sub)
        else:
            normalized.append(tok)
    return normalized


# ---------------------------
# High-level normalization pipelines
# ---------------------------

def basic_clean(text: str) -> str:
    """
    End-to-end basic cleaning used for both catalog content and
    queries:

    - clamp length
    - strip HTML
    - normalize unicode
    - normalize whitespace
    """
    if text is None:
        return ""
    text = clamp_text_length(str(text))
    text = strip_html(text)
    text = normalize_unicode(text)
    text = normalize_whitespace(text)
    return text


def normalize_for_lexical_index(text: str) -> str:
    """
    Pipeline used when preparing fields for BM25 / lexical indexing.

    - basic_clean
    - lowercase tokenization
    - apply synonyms
    - join back with single spaces
    """
    clean = basic_clean(text)
    tokens = simple_tokenize(clean)
    tokens = apply_synonyms(tokens)
    return " ".join(tokens)


def normalize_query(text: str) -> str:
    """
    Dedicated pipeline for user queries.

    For now it's identical to normalize_for_lexical_index but split
    out so we can tweak behavior for queries later (e.g., special
    handling of question marks, leading phrases, etc.).
    """
    return normalize_for_lexical_index(text)


def lexical_tokens_for_bm25(text: str) -> List[str]:
    """
    Return the token list specifically for BM25, after cleaning and
    synonyms.  This simply splits the normalized string on spaces.
    """
    normalized = normalize_for_lexical_index(text)
    if not normalized:
        return []
    return normalized.split()


# ---------------------------
# Debug / CLI usage
# ---------------------------

if __name__ == "__main__":
    # Tiny manual debug helper.  Running this module directly will
    # demonstrate what the normalizers do on a sample string.  This
    # section is intentionally kept trivial so it doesn't interfere
    # with the primary function of the module.
    sample = "Senior JS/TS Engineer (Node.js) – strong AI/ML, stakeholder mgmt.<br>Remote-friendly."
    print("RAW:", sample)
    print("BASIC CLEAN:", basic_clean(sample))
    print("LEXICAL:", normalize_for_lexical_index(sample))
    print("TOKENS:", lexical_tokens_for_bm25(sample))
