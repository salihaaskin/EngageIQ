#!/usr/bin/env python3
"""
Collect Beginner Coding content from the Hacker News Algolia search API
(no authentication required) and export a clean CSV.

Queries multiple beginner-focused search terms to get diverse coverage of:
  - Learn-to-code discussions ("Ask HN: best way to learn Python")
  - Tutorial and resource posts
  - Beginner project showcases ("Show HN: my first app")
  - CS fundamentals (algorithms, data structures for beginners)
  - Career-switch-to-tech threads
  - Coding challenge / competitive programming discussions

Usage:
  python Data/scripts/collect_beginner_coding.py
  python Data/scripts/collect_beginner_coding.py --output Data/raw/beginner_coding_hn.csv --target 500
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

ALGOLIA_BASE = "https://hn.algolia.com/api/v1/search"
REQUEST_TIMEOUT = 20
SLEEP_BETWEEN_REQUESTS = 0.4   # stay well within Algolia rate limits

# Each tuple: (search query, hitsPerPage)
SEARCH_QUERIES: list[tuple[str, int]] = [
    # Learn-to-code Ask HN / discussions
    ("learn programming beginner",      80),
    ("learn python beginner",           80),
    ("learn javascript beginner",       60),
    ("learn to code",                   80),
    ("coding bootcamp",                 60),
    ("first programming language",      50),
    ("how to learn programming",        60),
    ("beginner programmer",             60),
    # Project showcases
    ("show hn first app",               50),
    ("show hn beginner project",        50),
    ("my first python project",         40),
    ("portfolio project beginner",      40),
    # Tutorials and resources
    ("programming tutorial beginner",   60),
    ("free coding resources beginners", 50),
    ("cs50 harvard",                    50),
    ("introduction to programming",     50),
    ("programming fundamentals",        40),
    # CS fundamentals
    ("data structures beginners",       50),
    ("algorithms beginners",            50),
    ("leetcode beginners",              40),
    ("coding challenges beginners",     40),
    ("competitive programming basics",  40),
    # Career / junior dev
    ("junior developer career",         50),
    ("career switch software engineer", 50),
    ("self taught programmer",          50),
    ("coding interview preparation",    60),
    # Python-specific beginner content
    ("python basics",                   50),
    ("python for beginners",            50),
    # Ask HN catch-all
    ("ask hn learn code",               60),
    ("ask hn beginner",                 60),
]

REQUIRED_COLUMNS = [
    "id", "source", "title", "text", "url",
    "score", "num_comments", "created_utc", "domain",
]


def algolia_search(query: str, hits_per_page: int = 50) -> list[dict[str, Any]]:
    """Fetch story hits from the HN Algolia API for a given query."""
    params = urlencode({
        "query": query,
        "tags": "story",
        "hitsPerPage": hits_per_page,
        "attributesToRetrieve": (
            "objectID,title,story_text,url,points,num_comments,"
            "created_at_i,author,_tags"
        ),
    })
    url = f"{ALGOLIA_BASE}?{params}"
    try:
        with urlopen(url, timeout=REQUEST_TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("hits", [])
    except (URLError, json.JSONDecodeError, OSError) as exc:
        logger.warning("Algolia request failed for %r: %s", query, exc)
        return []


def hit_to_record(hit: dict[str, Any]) -> dict[str, Any] | None:
    """Convert an Algolia hit to the EngageIQ schema."""
    obj_id = str(hit.get("objectID", "")).strip()
    if not obj_id:
        return None

    title = str(hit.get("title") or "").strip()
    if not title:
        return None

    text = str(hit.get("story_text") or "").strip()
    url = str(hit.get("url") or "").strip()
    if not url:
        url = f"https://news.ycombinator.com/item?id={obj_id}"

    points = hit.get("points")
    num_comments = hit.get("num_comments")

    created_ts = hit.get("created_at_i")
    if created_ts:
        try:
            created_utc = datetime.fromtimestamp(int(created_ts), tz=timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            )
        except (ValueError, OSError, OverflowError):
            created_utc = ""
    else:
        created_utc = ""

    if not created_utc:
        return None

    # Derive domain from URL
    domain: str | None = None
    if url.startswith("http"):
        try:
            from urllib.parse import urlparse
            host = urlparse(url).netloc.lower()
            if host.startswith("www."):
                host = host[4:]
            domain = host or None
        except Exception:
            domain = None

    return {
        "id": obj_id,
        "source": "hackernews",
        "title": title,
        "text": text,
        "url": url,
        "score": int(points) if isinstance(points, (int, float)) else 0,
        "num_comments": int(num_comments) if isinstance(num_comments, (int, float)) else 0,
        "created_utc": created_utc,
        "domain": domain,
    }


def collect_beginner_records(target: int) -> list[dict[str, Any]]:
    seen_ids: set[str] = set()
    records: list[dict[str, Any]] = []

    for query, hits_per_page in SEARCH_QUERIES:
        if len(records) >= target:
            break

        logger.info("Querying Algolia: %r (up to %d hits)", query, hits_per_page)
        hits = algolia_search(query, hits_per_page)
        added = 0

        for hit in hits:
            record = hit_to_record(hit)
            if record is None:
                continue
            if record["id"] in seen_ids:
                continue
            seen_ids.add(record["id"])
            records.append(record)
            added += 1

        logger.info("  +%d new records (total: %d)", added, len(records))
        time.sleep(SLEEP_BETWEEN_REQUESTS)

    return records


def export_csv(records: list[dict[str, Any]], output_path: Path) -> None:
    df = pd.DataFrame(records)[REQUIRED_COLUMNS]
    df["score"] = pd.to_numeric(df["score"], errors="coerce").fillna(0).astype(int)
    df["num_comments"] = pd.to_numeric(df["num_comments"], errors="coerce").fillna(0).astype(int)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Wrote %d records to %s", len(df), output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect Beginner Coding HN posts via Algolia search API."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("Data/raw/beginner_coding_hn.csv"),
    )
    parser.add_argument(
        "--target",
        type=int,
        default=500,
        help="Target number of unique records to collect (default: 500).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records = collect_beginner_records(args.target)
    if not records:
        logger.error("No records collected.")
        return 1
    export_csv(records, args.output)
    logger.info("Done. Collected %d unique Beginner Coding records.", len(records))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
