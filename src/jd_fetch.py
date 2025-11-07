from __future__ import annotations

"""
Helper for downloading and extracting job description (JD) pages or product
pages from the SHL catalog.

The original implementation uses ``httpx`` for robust HTTP fetching and
optionally ``trafilatura`` to extract main page content.  In this minimal
environment we preserve the same API surface while gracefully degrading
functionality when optional dependencies are missing.  If ``trafilatura``
cannot be imported the raw page text is returned instead.  Likewise, if
network access is unavailable the caller should handle ``None`` being
returned.
"""

import httpx
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

try:
    # ``trafilatura`` is an optional dependency for more accurate content
    # extraction.  It may not be installed in constrained environments.
    import trafilatura  # type: ignore[import-not-found]
except Exception:  # pragma: no cover
    trafilatura = None

from .config import (
    HTTP_CONNECT_TIMEOUT,
    HTTP_READ_TIMEOUT,
    HTTP_MAX_REDIRECTS,
    HTTP_MAX_BYTES,
    HTTP_USER_AGENT,
)
from .normalize import basic_clean


def fetch_and_extract(url: str) -> str | None:
    """
    Fetch a page and attempt to extract its main textual content.

    Parameters
    ----------
    url : str
        The URL to fetch.

    Returns
    -------
    str | None
        Cleaned text extracted from the page or ``None`` on failure.

    Notes
    -----
    This function implements several hardened behaviours:

    * Uses ``httpx`` with timeouts and limited redirects to avoid hanging.
    * Enforces a maximum download size (configured via ``HTTP_MAX_BYTES``).
    * If ``trafilatura`` is available it is used to strip boilerplate and
      retain only the main content; otherwise the raw HTML is passed through
      the normaliser.
    * All text is normalised via ``basic_clean`` before being returned.
    * Any exception or non‑200 HTTP status results in a warning and a
      ``None`` return value, allowing callers to continue gracefully.
    """
    headers = {"User-Agent": HTTP_USER_AGENT}
    try:
        with httpx.Client(
            follow_redirects=True,
            timeout=httpx.Timeout(HTTP_READ_TIMEOUT, connect=HTTP_CONNECT_TIMEOUT),
            max_redirects=HTTP_MAX_REDIRECTS,
        ) as client:
            r = client.get(url, headers=headers)
            if r.status_code >= 400:
                logger.warning("JD fetch: HTTP {} for {}", r.status_code, url)
                return None

            if len(r.content) > HTTP_MAX_BYTES:
                logger.warning(
                    "JD fetch aborted: {} bytes > {} limit",
                    len(r.content),
                    HTTP_MAX_BYTES,
                )
                return None

            text: str | None = None
            if trafilatura is not None:
                try:
                    # Extract only the main content; this may fail on some pages.
                    text = trafilatura.extract(r.text)
                except Exception as e:
                    logger.warning("Trafilatura failed: {}", e)

            if not text:
                text = r.text

            cleaned = basic_clean(text)
            return cleaned if cleaned else None
    except httpx.ReadTimeout:
        logger.warning("JD fetch timeout for {}", url)
        return None
    except Exception as e:  # pragma: no cover
        logger.warning("JD fetch exception for {}: {}", url, e)
        return None