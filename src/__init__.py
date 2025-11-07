"""
Top-level package for the SHL assessment recommender system.

This package contains modules for crawling the SHL product catalog,
building a normalized catalog snapshot, constructing search indices
for both lexical (BM25) and dense retrieval, and serving a simple
recommendation API.  The package is intentionally lightweight: there
are no side‑effects on import and each module can be executed as a
script for ad‑hoc debugging or manual workflows.
"""
