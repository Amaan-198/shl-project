from __future__ import annotations

"""
Crawler for the SHL product catalog.

This module encapsulates the logic to iterate through the paginated
``Individual Test Solutions`` table on the SHL product catalog and
produce a structured data frame.  Each row in the resulting DataFrame
contains the assessment name, URL, humanised test type labels, remote
and adaptive support flags, a cleaned description and an estimated
duration in minutes.

Compared to the upstream implementation this version includes a
number of robustness improvements:

* ``_extract_duration_minutes`` now recognises hyphenated durations
  (e.g. ``30-minute``) as well as ranges (e.g. ``20-30 minutes``)
  and alternative spellings such as ``mins`` or ``min``.  When a
  range is detected the upper bound is used.
* ``_strip_boilerplate`` has been extended to remove common "noise"
  lines that appear in product descriptions, such as ``Keywords: …``
  and promotional calls to action.  Unwanted patterns can be added
  easily to the ``kill`` list.
* The crawler limits itself to a reasonable number of pages (see
  ``MAX_CATALOG_PAGES`` in ``config.py``) and logs progress via
  ``loguru``.
* Errors encountered when fetching pages or product details are
  logged but do not cause the entire crawl to abort; instead the
  crawler continues with the next page or row.
"""

import re
from typing import Dict, List
from urllib.parse import urljoin

import httpx
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup  # type: ignore[import-not-found]
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

from .config import (
    DATA_DIR,
    HTTP_CONNECT_TIMEOUT,
    HTTP_READ_TIMEOUT,
    HTTP_MAX_REDIRECTS,
    HTTP_MAX_BYTES,
    HTTP_USER_AGENT,
    MAX_CATALOG_PAGES,
)
from .normalize import basic_clean
from .jd_fetch import fetch_and_extract


# Base URLs for the catalog and detail pages
BASE_CATALOG_URL = "https://www.shl.com/products/product-catalog/"
BASE_SHL_URL = "https://www.shl.com"


# Mapping from legend codes used in the table to human readable labels.
LEGEND_MAP: Dict[str, str] = {
    "A": "Ability & Aptitude",
    "B": "Biodata & Situational Judgement",
    "C": "Competencies",
    "D": "Development & 360",
    "E": "Assessment Exercises",
    "K": "Knowledge & Skills",
    "P": "Personality & Behavior",
    "S": "Simulations",
}


def _http_client() -> httpx.Client:
    """
    Construct a configured HTTP client for crawling pages.

    Uses the global timeout, redirect and user agent settings from ``config.py``.
    """
    # Explicitly set trust_env=False so httpx ignores proxy environment
    # variables such as ALL_PROXY/HTTP_PROXY.  This avoids attempts to
    # use SOCKS proxies without the optional socksio dependency.
    return httpx.Client(
        headers={"User-Agent": HTTP_USER_AGENT},
        follow_redirects=True,
        timeout=httpx.Timeout(HTTP_READ_TIMEOUT, connect=HTTP_CONNECT_TIMEOUT),
        max_redirects=HTTP_MAX_REDIRECTS,
        trust_env=False,
    )


def _fetch_html(client: httpx.Client, url: str) -> str:
    """
    Retrieve the HTML content of a catalog page.

    Raises ``RuntimeError`` on HTTP errors or overly large responses.  The
    caller is expected to catch exceptions and decide whether to continue.
    """
    logger.info("Fetching catalog page: {}", url)
    r = client.get(url)
    if r.status_code >= 400:
        raise RuntimeError(f"HTTP {r.status_code} for {url}")
    if len(r.content) > HTTP_MAX_BYTES:
        raise RuntimeError(
            f"Page too large ({len(r.content)} bytes) for {url}",
        )
    return r.text


def _parse_individual_table(html: str) -> List[Dict]:
    """
    Parse the HTML of a catalog page to extract rows from the
    ``Individual Test Solutions`` table.

    The function returns a list of dictionaries with keys ``name``,
    ``url``, ``remote_support_raw``, ``adaptive_support_raw`` and
    ``legend_codes``.  The latter still needs to be mapped to human
    readable test type labels via ``_legend_codes_to_test_types``.
    """
    soup = BeautifulSoup(html, "lxml")

    target_table = None
    for tbl in soup.find_all("table"):
        headers = [th.get_text(" ", strip=True) for th in tbl.find_all("th")]
        joined = " ".join(headers)
        if "Individual Test Solutions" in joined:
            target_table = tbl
            break

    if target_table is None:
        logger.warning("No 'Individual Test Solutions' table found on page")
        return []

    body = target_table.find("tbody") or target_table

    rows: List[Dict] = []
    for tr in body.find_all("tr"):
        tds = tr.find_all("td")
        if not tds:
            continue

        link = tds[0].find("a")
        if not link:
            continue

        name = link.get_text(strip=True)
        href = link.get("href", "")
        url = urljoin(BASE_SHL_URL, href)

        # Some tables omit columns; guard against missing indexes
        remote_text = tds[1].get_text(" ", strip=True) if len(tds) > 1 else ""
        adaptive_text = tds[2].get_text(" ", strip=True) if len(tds) > 2 else ""
        type_text = tds[-1].get_text(" ", strip=True) if len(tds) >= 1 else ""

        rows.append(
            {
                "name": name,
                "url": url,
                "remote_support_raw": remote_text,
                "adaptive_support_raw": adaptive_text,
                "legend_codes": type_text,
            }
        )

    logger.info("Parsed {} rows from Individual Test Solutions table", len(rows))
    return rows


def _legend_codes_to_test_types(codes: str) -> List[str]:
    """
    Convert a string of legend codes into a list of human readable test types.

    Multiple codes may be separated by commas or whitespace.  Duplicate
    labels are removed while preserving order.
    """
    parts = [c.strip() for c in codes.replace(",", " ").split() if c.strip()]
    labels: List[str] = []
    for c in parts:
        lbl = LEGEND_MAP.get(c.upper())
        if lbl and lbl not in labels:
            labels.append(lbl)
    return labels


def _extract_duration_minutes(text: str) -> int:
    """
    Extract an estimated duration in minutes from arbitrary text.

    The function searches for patterns indicating time durations and
    returns the maximum value found (clamped between 1 and 240).  It
    recognises both explicit assignments (``minutes = 30``), simple
    mentions (``30 minutes``), hyphenated forms (``30-minute``),
    shorthand (``30 min`` or ``30 mins``) and ranges (``20-30 minutes``).
    When a range is found the upper bound is used.  If no valid
    duration is detected the function returns 0.
    """
    if not text:
        return 0

    candidates: list[int] = []
    lower = text.lower()

    # Explicit assignments like "minutes = 30"
    for m in re.findall(r"minutes?\s*=\s*(\d+)", lower):
        try:
            val = int(m)
        except Exception:
            continue
        if 1 <= val <= 240:
            candidates.append(val)

    # Ranges like "20-30 minutes" or "15 – 25 mins"; use the upper bound
    for m1, m2 in re.findall(r"(\d+)\s*[\-–to]+\s*(\d+)\s*(?:minutes?|mins?|min)\b", lower):
        try:
            hi = int(m2)
        except Exception:
            continue
        if 1 <= hi <= 240:
            candidates.append(hi)

    # Simple mentions e.g. "30 minutes", "45 mins", "15 min"
    for m in re.findall(r"(\d+)\s*(?:minutes?|mins?|min)\b", lower):
        try:
            val = int(m)
        except Exception:
            continue
        if 1 <= val <= 240:
            candidates.append(val)

    # Hyphenated forms e.g. "30-minute"
    for m in re.findall(r"(\d+)\s*\-\s*minute\b", lower):
        try:
            val = int(m)
        except Exception:
            continue
        if 1 <= val <= 240:
            candidates.append(val)

    return max(candidates) if candidates else 0


def _strip_boilerplate(text: str) -> str:
    """
    Remove common boilerplate and extraneous lines from product descriptions.

    The SHL product pages sometimes embed repeated headings or navigation
    content inside the extracted text.  This helper trims those
    artefacts using regular expressions.  The default kill list
    removes lines about test types, remote testing, legend bullets,
    cookie banners and known promotional sections.  Additional
    patterns can be added as needed.
    """
    # Remove duplicated "Title Description" prefix
    text = re.sub(r"^([\w\s\-\(\)&]+)\s+Description\s+\1\s+", "", text, flags=re.I)
    text = re.sub(r"^([\w\s\-\(\)&]+)\s+Description\s+", "", text, flags=re.I)
    
    # Patterns to strip entire lines or trailing content.  Each pattern
    # should match as broadly as possible without removing actual
    # description content.
    kill = [
        r"Test Type:.*$",
        r"Remote Testing:.*$",
        r"Accelerate Your Talent Strategy.*$",
        r"-\s*[ABCDKPS]\b.*$",  # legend bullets
        r"Your use of this assessment product may be subject to .*Law 144.*$",
        r"Keywords?:.*$",  # remove keyword tags
        r"Related Products?:.*$",  # remove related product sections
        r"Click here.*$",  # remove calls to action
        r"Book (a|your) demo.*$",  # promotional CTA
    ]
    for pat in kill:
        text = re.sub(pat, "", text, flags=re.I | re.M)

    # Collapse whitespace to single spaces and trim
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _shorten_description(text: str, max_chars: int = 500) -> str:
    """
    Shorten a long description to a maximum number of characters.

    This helper first strips boilerplate, then truncates the text
    without splitting words.  It uses ``max_chars`` as a hard cap
    and returns the entire string if it is already short enough.
    """
    if not text:
        return ""
    text = _strip_boilerplate(text)
    if len(text) > max_chars:
        trunc = text[:max_chars]
        last_space = trunc.rfind(" ")
        if last_space > 0:
            trunc = trunc[:last_space]
        text = trunc.strip()
    return text


def _enrich_with_product_details(rows: List[Dict]) -> None:
    """
    For each row in ``rows`` fetch the product detail page and
    populate the ``description`` and ``duration`` fields.

    Failures in fetching or extraction result in empty strings and
    zero durations.  Normalised text is cleaned and truncated
    appropriately.
    """
    for row in rows:
        url = row.get("url", "")
        try:
            text = fetch_and_extract(url)
        except Exception as e:
            logger.warning("Failed to fetch product details for {}: {}", url, e)
            text = None

        if not text:
            row["description"] = ""
            row["duration"] = 0
        else:
            cleaned = basic_clean(text)
            row["description"] = _shorten_description(cleaned, max_chars=500)
            row["duration"] = _extract_duration_minutes(cleaned)


def _normalize_flags(rows: List[Dict]) -> None:
    """
    Normalise the raw remote/adaptive flags to literal "Yes" or "No".

    Accepts freeform strings and symbols such as "✓" and various
    capitalisations.  Missing or unknown values default to ``"No"``.
    """
    def flag(text: str) -> str:
        t = (text or "").lower()
        if not t:
            return "No"
        if "yes" in t or t.startswith("y"):
            return "Yes"
        if "✓" in text or "check" in t:
            return "Yes"
        return "No"

    for row in rows:
        row["remote_support"] = flag(row.get("remote_support_raw", ""))
        row["adaptive_support"] = flag(row.get("adaptive_support_raw", ""))


def _build_search_text(
    name: str, desc: str, test_type: List[str], adaptive: str, remote: str
) -> str:
    """
    Construct the ``search_text`` field used for indexing and retrieval.

    Concatenates the name, description, test type labels and flag hints
    ("adaptive", "remote"), lowercases everything and normalises
    whitespace.  Empty segments are ignored.
    """
    parts: List[str] = []
    if name:
        parts.append(name)
    if desc:
        parts.append(desc)
    if test_type:
        parts.append(" ".join(test_type))
    flags: List[str] = []
    if adaptive == "Yes":
        flags.append("adaptive")
    if remote == "Yes":
        flags.append("remote")
    if flags:
        parts.append(" ".join(flags))
    out = ". ".join(p for p in parts if p)
    out = re.sub(r"\s+", " ", out.lower()).strip()
    return out


def crawl_individual_test_solutions() -> pd.DataFrame:
    """
    Crawl the Individual Test Solutions catalog and return a DataFrame.

    Pagination continues until either no new rows are returned or the
    ``MAX_CATALOG_PAGES`` limit is reached.  Each row is enriched
    with product details and normalised fields before being
    assembled into the final DataFrame.
    """
    client = _http_client()
    all_rows: List[Dict] = []
    seen_urls = set()
    page_size = 12  # observed default; used to detect end of pagination
    start = 0
    pages_crawled = 0

    while True:
        url = BASE_CATALOG_URL if start == 0 else f"{BASE_CATALOG_URL}?start={start}&type=1"
        try:
            html = _fetch_html(client, url)
        except Exception as e:
            logger.warning("Stopping crawl at start={} due to error: {}", start, e)
            break

        page_rows = _parse_individual_table(html)
        fresh = [r for r in page_rows if r["url"] not in seen_urls]
        for r in fresh:
            seen_urls.add(r["url"])
            all_rows.append(r)

        logger.info(
            "Crawl page start={}, got {} rows ({} unique so far)",
            start,
            len(page_rows),
            len(all_rows),
        )
        pages_crawled += 1
        if not page_rows or len(page_rows) < page_size or pages_crawled >= MAX_CATALOG_PAGES:
            break
        start += page_size

    if not all_rows:
        logger.warning("No catalog rows crawled from Individual Test Solutions.")
        return pd.DataFrame(columns=["item_id", "name", "url"])

    # Fetch details for each product
    _enrich_with_product_details(all_rows)
    _normalize_flags(all_rows)
    for row in all_rows:
        row["test_type"] = _legend_codes_to_test_types(row.get("legend_codes", ""))

    # Assign unique item_id and build search_text
    for idx, row in enumerate(all_rows):
        row["item_id"] = idx
        row["search_text"] = _build_search_text(
            name=row.get("name", ""),
            desc=row.get("description", ""),
            test_type=row.get("test_type", []),
            adaptive=row.get("adaptive_support", "No"),
            remote=row.get("remote_support", "No"),
        )

    df = pd.DataFrame(all_rows)
    cols = [
        "item_id",
        "name",
        "url",
        "description",
        "duration",
        "adaptive_support",
        "remote_support",
        "test_type",
        "search_text",
    ]
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    df = df[cols]
    logger.info("Crawl complete. Final catalog rows: {}", len(df))
    return df


def build_catalog_snapshot_from_crawl() -> None:
    """
    Execute a full crawl and write the resulting snapshot to disk.

    The snapshot is stored as a Parquet file under ``data/catalog_snapshot.parquet``.
    """
    df = crawl_individual_test_solutions()
    output = DATA_DIR / "catalog_snapshot.parquet"
    output.parent.mkdir(parents=True, exist_ok=True)
    # Attempt to write Parquet.  If pyarrow/fastparquet is unavailable,
    # fall back to CSV to avoid dependency issues.
    try:
        df.to_parquet(output, index=False)
        logger.info("Catalog snapshot written to {}", output)
    except Exception as e:
        logger.warning("Failed to write parquet snapshot ({}). Falling back to CSV.", e)
        csv_output = output.with_suffix(".csv")
        df.to_csv(csv_output, index=False)
        logger.info("Catalog snapshot written to {}", csv_output)


if __name__ == "__main__":
    build_catalog_snapshot_from_crawl()