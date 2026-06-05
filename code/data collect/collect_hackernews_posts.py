#!/usr/bin/env python3
"""Collect Hacker News posts via Algolia + Firebase APIs and export to CSV."""

from __future__ import annotations

import argparse
import logging
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pandas as pd
import requests
from requests.adapters import HTTPAdapter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

TARGET_TOTAL = 7000
RANDOM_SEED = 42
ALGOLIA_SEARCH_URL = "https://hn.algolia.com/api/v1/search"
ALGOLIA_SEARCH_BY_DATE_URL = "https://hn.algolia.com/api/v1/search_by_date"
HN_ITEM_URL = "https://hacker-news.firebaseio.com/v0/item/{item_id}.json"
HN_POST_PAGE = "https://news.ycombinator.com/item?id={item_id}"
REQUEST_TIMEOUT = 30
MAX_ALGOLIA_PAGE = 50  # pages per query (100 hits/page => 6000 candidates max)

# Topic -> search queries for balanced, diverse sampling
TOPIC_QUERIES: dict[str, list[str]] = {
    "Machine Learning": [
        "machine learning",
        "ml",
        "deep learning",
        "neural network",
        "scikit-learn",
        "xgboost",
        "lightgbm",
        "catboost",
        "random forest",
        "gradient boosting",
        "classification",
        "regression",
        "clustering",
    ],

    "DevOps/K8s": [
        "kubernetes",
        "docker",
        "helm",
        "terraform",
        "devops",
        "container",
        "argo",
        "eks",
        "gke",
        "aks",
        "service mesh",
        "istio",
        "linkerd",
        "ci/cd",
        "continuous integration",
        "continuous deployment",
        "infrastructure as code",
    ],

    "Trending Open-Source": [
        "open source",
        "github",
        "foss",
        "oss",
        "apache foundation",
        "linux foundation",
        "trending repository",
        "popular repository",
        "rising repository",
    ],

    "Developer Tools": [
        "ide",
        "vscode",
        "jetbrains",
        "cli",
        "developer tool",
        "debugger",
        "build system",
        "linter",
        "formatter",
        "code quality",
        "code review tool",
        "testing framework",
        "profiling tool",
    ],

    "Cybersecurity": [
        "security",
        "cybersecurity",
        "vulnerability",
        "cve",
        "malware",
        "exploit",
        "encryption",
        "authentication",
        "authorization",
        "penetration testing",
        "red team",
        "blue team",
        "threat intelligence",
        "security research",
    ],

    "Frontend (React/Web)": [
        "react",
        "next.js",
        "javascript",
        "typescript",
        "css",
        "frontend",
        "web app",
        "ui",
        "user interface",
        "web development",
        "browser",
        "dom",
        "web performance",
        "accessibility",
    ],

    "B2B SaaS": [
        "b2b saas",
        "saas",
        "software as a service",
        "enterprise software",
        "cloud software",
        "saas startup",
        "arr",
        "mrr",
        "churn",
        "customer acquisition",
        "customer retention",
        "crm",
        "erp",
        "product led growth",
        "plg",
        "go to market",
        "gtm",
        "product market fit",
        "subscription business",
        "customer success",
        "workflow automation",
        "business software",
        "enterprise customer",
        "saas pricing",
        "lead generation",
        "sales automation",
    ],

    "Blockchain": [
        "blockchain",
        "web3",
        "ethereum",
        "solana",
        "crypto",
        "smart contract",
        "nft",
        "defi",
        "decentralized finance",
        "dapp",
        "metaverse",
        "token",
        "cryptocurrency",
        "bitcoin",
            "layer 2",
            "zk rollup",
            "optimistic rollup",
            "sidechain",
            "consensus algorithm",
            "proof of work",
            "proof of stake",
    ],

    "Python Data Eng": [
        "pandas",
        "spark",
        "airflow",
        "data engineering",
        "etl",
        "data pipeline",
        "duckdb",
        "polars",
        "dask",
        "data warehouse",
        "snowflake",
        "bigquery",
        "redshift",
        "databricks",
        "dbt",
        "data lake",
        "streaming data",
        "kafka",
        "flink",
        "beam",
    ],

    "GameDev (C++)": [
         "gamedev",
        "game development",
        "game programming",
        "c++",
        "unreal engine",
        "ue5",
        "unity",
        "godot",
        "graphics programming",
        "rendering",
        "opengl",
        "vulkan",
        "directx",
        "shader",
        "game engine",
        "physics engine",
        "collision detection",
        "game ai",
        "pathfinding",
        "multiplayer",
        "game networking",
        "indie game",
        "game jam",
        "gameplay programming",
        "optimization",
    ],

    "AI Research": [
        "llm",
        "transformer",
        "openai",
        "anthropic",
        "arxiv",
        "research paper",
        "foundation model",
        "large language model",
        "gpt",
        "bert",
        "t5",
        "computer vision",
        "natural language processing",
        "reinforcement learning",
        "self-supervised learning",
        "few-shot learning",
        "zero-shot learning",
        "multimodal model",
    ],

    "Embedded Systems (C/RTOS)": [
        "embedded systems",
        "embedded programming",
        "firmware",
        "embedded software",
        "embedded c",
        "rtos",
        "freertos",
        "zephyr",
        "threadx",
        "microcontroller",
        "stm32",
        "esp32",
        "arm cortex",
        "uart",
        "spi",
        "i2c",
        "can bus",
        "interrupt",
        "device driver",
        "embedded linux",
        "yocto",
        "buildroot",
        "jtag",
        "bare metal",
        "real time system",
    ],

    "Cloud APIs": [
        "api",
        "rest api",
        "graphql",
        "aws api",
        "cloud service",
        "webhook",
        "serverless",
        "lambda",
        "azure functions",
        "google cloud functions",
        "api gateway",
        "cloud api",
    ],

    "Mobile Dev (iOS/Flutter)": [
        "ios",
        "swift",
        "flutter",
        "android",
        "react native",
        "mobile app",
        "mobile development",
        "xamarin",
        "kotlin",
        "mobile ui",
        "mobile performance",
        "mobile testing",
        "mobile security",
    ],

    "Beginner Coding": [
        "learn programming",
        "coding tutorial",
        "beginner",
        "coding bootcamp",
        "python basics",
        "javascript basics",
        "introduction to programming",
        "entry-level programming",
        "beginner-friendly",
        "first app",
        "junior developer",
        "career switch to tech",
        "coding interview prep",
        "programming fundamentals",
        "portfolio project",
        "coding help",
        "homework help",
        "programming challenge",
        "competitive programming",
        "coding exercise",
        "algorithm practice",
        "data structure practice",
        "code katas",
        "coding problem",
        "coding constest",
    ],

}
   
REQUIRED_FIELDS = {
    "id",
    "source",
    "title",
    "text",
    "url",
    "score",
    "num_comments",
    "created_utc",
    "domain",
}

DELETED_MARKERS = {"[deleted]", "[dead]", ""}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect Hacker News posts and export to CSV."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("Data/hackernews_posts.csv"),
        help="Output CSV file path (default: Data/hackernews_posts.csv)",
    )
    parser.add_argument(
        "--target-total",
        type=int,
        default=TARGET_TOTAL,
        help=f"Approximate total posts to collect (default: {TARGET_TOTAL})",
    )
    parser.add_argument(
        "--years",
        type=float,
        default=2.0,
        help="Only include posts from the last N years (default: 2)",
    )
    return parser.parse_args()


def create_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {"User-Agent": "EngageIQ-HN-Collector/1.0 (research dataset)"}
    )
    adapter = HTTPAdapter(pool_connections=32, pool_maxsize=32)
    session.mount("https://", adapter)
    return session


def extract_domain(external_url: str | None) -> str | None:
    if not external_url:
        return None
    try:
        host = urlparse(external_url).netloc.lower()
        if not host:
            return None
        if host.startswith("www."):
            host = host[4:]
        return host or None
    except Exception:
        return None


def format_created_utc(created_at_i: int) -> str:
    return datetime.fromtimestamp(created_at_i, tz=timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def is_accessible_hit(hit: dict[str, Any]) -> bool:
    title = (hit.get("title") or "").strip()
    author = (hit.get("author") or "").strip()
    if title in DELETED_MARKERS or author in DELETED_MARKERS:
        return False
    if title.lower() in ("[deleted]", "[dead]"):
        return False
    return True


def hit_to_partial_record(hit: dict[str, Any], cutoff_ts: int) -> dict[str, Any] | None:
    if not is_accessible_hit(hit):
        return None

    created_at_i = hit.get("created_at_i")
    if created_at_i is None or created_at_i < cutoff_ts:
        return None

    post_id = str(hit.get("story_id") or hit.get("objectID") or "")
    if not post_id.isdigit():
        return None

    external_url = hit.get("url")
    return {
        "id": post_id,
        "source": "hackernews",
        "title": (hit.get("title") or "").strip(),
        "text": "",
        "url": HN_POST_PAGE.format(item_id=post_id),
        "score": int(hit.get("points") or 0),
        "num_comments": int(hit.get("num_comments") or 0),
        "created_utc": format_created_utc(int(created_at_i)),
        "domain": extract_domain(external_url),
        "_external_url": external_url,
    }


def search_algolia(
    session: requests.Session,
    query: str,
    cutoff_ts: int,
    max_pages: int,
    use_date_index: bool,
) -> list[dict[str, Any]]:
    """Paginate Algolia HN search for story hits within the date window."""
    base_url = ALGOLIA_SEARCH_BY_DATE_URL if use_date_index else ALGOLIA_SEARCH_URL
    hits: list[dict[str, Any]] = []
    numeric_filter = f"created_at_i>={cutoff_ts}"

    for page in range(max_pages):
        params = {
            "query": query,
            "tags": "story",
            "numericFilters": numeric_filter,
            "hitsPerPage": 100,
            "page": page,
        }
        try:
            response = session.get(base_url, params=params, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as exc:
            logger.warning("Algolia search failed for %r page %d: %s", query, page, exc)
            break

        batch = data.get("hits") or []
        if not batch:
            break

        hits.extend(batch)
        nb_pages = data.get("nbPages", 0)
        if page + 1 >= nb_pages:
            break

        time.sleep(0.15)

    return hits


def collect_candidates_for_topic(
    session: requests.Session,
    topic: str,
    queries: list[str],
    cutoff_ts: int,
    target_candidates: int,
) -> list[dict[str, Any]]:
    seen_ids: set[str] = set()
    records: list[dict[str, Any]] = []
    pages_per_query = max(3, MAX_ALGOLIA_PAGE // len(queries))

    for index, query in enumerate(queries):
        if len(records) >= target_candidates:
            break

        use_date_index = index % 2 == 0
        raw_hits = search_algolia(
            session,
            query,
            cutoff_ts,
            pages_per_query,
            use_date_index=use_date_index,
        )

        for hit in raw_hits:
            partial = hit_to_partial_record(hit, cutoff_ts)
            if partial is None:
                continue
            if partial["id"] in seen_ids:
                continue
            seen_ids.add(partial["id"])
            records.append(partial)

        logger.info(
            "Topic '%s' query %r: %d candidates so far",
            topic,
            query,
            len(records),
        )
        time.sleep(0.2)

    logger.info("Topic '%s': %d candidate posts", topic, len(records))
    return records


def fetch_item(session: requests.Session, item_id: str) -> dict[str, Any] | None:
    try:
        response = session.get(
            HN_ITEM_URL.format(item_id=item_id),
            timeout=REQUEST_TIMEOUT,
        )
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return None


def enrich_records(
    session: requests.Session,
    records: list[dict[str, Any]],
    max_workers: int = 16,
) -> list[dict[str, Any]]:
    """Fetch Firebase items to fill text, validate alive/deleted, refine domain."""
    enriched: list[dict[str, Any]] = []
    ids = [record["id"] for record in records]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(fetch_item, session, item_id): item_id
            for item_id in ids
        }
        item_by_id: dict[str, dict[str, Any] | None] = {}
        for future in as_completed(future_map):
            item_id = future_map[future]
            try:
                item_by_id[item_id] = future.result()
            except Exception:
                item_by_id[item_id] = None
            time.sleep(0.01)

    for record in records:
        item = item_by_id.get(record["id"])
        if item is None:
            continue
        if item.get("deleted") or item.get("dead"):
            continue

        title = (item.get("title") or record["title"] or "").strip()
        if title in DELETED_MARKERS:
            continue

        author = (item.get("by") or "").strip()
        if author in DELETED_MARKERS:
            continue

        if item.get("type") not in (None, "story"):
            continue

        text = (item.get("text") or "").strip()
        external_url = item.get("url") or record.pop("_external_url", None)
        record.pop("_external_url", None)

        record["title"] = title
        record["text"] = text
        record["score"] = int(item.get("score") or record["score"])
        if item.get("descendants") is not None:
            record["num_comments"] = int(item["descendants"])
        if item.get("time"):
            record["created_utc"] = format_created_utc(int(item["time"]))

        domain = extract_domain(external_url)
        record["domain"] = domain

        enriched.append(record)

    logger.info(
        "Enrichment: %d / %d records kept after Firebase validation",
        len(enriched),
        len(records),
    )
    return enriched


def balance_and_deduplicate(
    topic_records: dict[str, list[dict[str, Any]]],
    target_per_topic: int,
) -> list[dict[str, Any]]:
    rng = random.Random(RANDOM_SEED)
    final: list[dict[str, Any]] = []
    global_seen: set[str] = set()
    topic_order = list(topic_records.keys())
    rng.shuffle(topic_order)

    for topic in topic_order:
        records = topic_records[topic]
        unique: list[dict[str, Any]] = []
        seen: set[str] = set()
        for record in records:
            if record["id"] in seen:
                continue
            seen.add(record["id"])
            unique.append(record)

        rng.shuffle(unique)

        added = 0
        for record in unique:
            if added >= target_per_topic:
                break
            if record["id"] in global_seen:
                continue
            global_seen.add(record["id"])
            clean = {k: v for k, v in record.items() if not k.startswith("_")}
            final.append(clean)
            added += 1

        if added < target_per_topic:
            logger.warning(
                "Topic '%s' only has %d posts (target %d)",
                topic,
                added,
                target_per_topic,
            )

    return final


def fill_topic_shortfalls(
    session: requests.Session,
    topic_records: dict[str, list[dict[str, Any]]],
    cutoff_ts: int,
    target_per_topic: int,
    global_seen: set[str],
) -> None:
    for topic, records in topic_records.items():
        topic_ids = {record["id"] for record in records}
        needed = target_per_topic - len(topic_ids)
        if needed <= 0:
            continue

        logger.info("Topic '%s' short by %d; running fill pass", topic, needed)
        extra_candidates = target_per_topic + needed + 100
        batch = collect_candidates_for_topic(
            session,
            topic,
            TOPIC_QUERIES[topic],
            cutoff_ts,
            extra_candidates,
        )
        batch = enrich_records(session, batch)
        for record in batch:
            if record["id"] in topic_ids or record["id"] in global_seen:
                continue
            records.append(record)
            topic_ids.add(record["id"])
            global_seen.add(record["id"])
            needed -= 1
            if needed <= 0:
                break


def validate_records(records: list[dict[str, Any]]) -> None:
    if not records:
        raise ValueError("No records to validate.")

    for index, record in enumerate(records):
        missing = REQUIRED_FIELDS - record.keys()
        if missing:
            raise ValueError(f"Record {index} missing fields: {missing}")

        if record["source"] != "hackernews":
            raise ValueError(
                f"Record {index} has invalid source: {record['source']!r}"
            )

        if not isinstance(record["id"], str) or not record["id"]:
            raise ValueError(f"Record {index} has invalid id.")

        for field in ("title", "text", "url", "created_utc"):
            if not isinstance(record[field], str):
                raise ValueError(
                    f"Record {index} field '{field}' must be a string."
                )

        if not record["title"].strip():
            raise ValueError(f"Record {index} has empty title.")

        if record["url"] != HN_POST_PAGE.format(item_id=record["id"]):
            raise ValueError(
                f"Record {index} url must be the HN item page, got {record['url']!r}"
            )

        for field in ("score", "num_comments"):
            if not isinstance(record[field], int):
                raise ValueError(
                    f"Record {index} field '{field}' must be an integer."
                )

        if record["domain"] is not None and not isinstance(record["domain"], str):
            raise ValueError(
                f"Record {index} field 'domain' must be null or string."
            )

        try:
            datetime.strptime(record["created_utc"], "%Y-%m-%dT%H:%M:%SZ").replace(
                tzinfo=timezone.utc
            )
        except ValueError as exc:
            raise ValueError(
                f"Record {index} has invalid created_utc: {record['created_utc']!r}"
            ) from exc

    ids = [record["id"] for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate post IDs found after deduplication.")

    logger.info("Schema validation passed for %d records.", len(records))


def export_csv(records: list[dict[str, Any]], output_path: Path) -> None:
    df = pd.DataFrame(records)

    for column in REQUIRED_FIELDS:
        if column not in df.columns:
            raise ValueError(f"Missing column in DataFrame: {column}")

    df["domain"] = df["domain"].astype(object)
    df.loc[df["domain"].isna(), "domain"] = None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Wrote %d records to %s", len(records), output_path)


def main() -> int:
    args = parse_args()
    num_topics = len(TOPIC_QUERIES)
    target_per_topic = max(1, args.target_total // num_topics)
    cutoff = datetime.now(timezone.utc) - timedelta(days=int(args.years * 365.25))
    cutoff_ts = int(cutoff.timestamp())
    candidate_buffer = target_per_topic + 150

    logger.info(
        "Collecting ~%d posts per topic (%d topics, target total ~%d), "
        "created on or after %s",
        target_per_topic,
        num_topics,
        target_per_topic * num_topics,
        cutoff.strftime("%Y-%m-%dT%H:%M:%SZ"),
    )

    session = create_session()
    global_seen: set[str] = set()
    topic_records: dict[str, list[dict[str, Any]]] = {}

    for topic, queries in TOPIC_QUERIES.items():
        candidates = collect_candidates_for_topic(
            session,
            topic,
            queries,
            cutoff_ts,
            candidate_buffer,
        )
        enriched = enrich_records(session, candidates)
        topic_records[topic] = enriched
        for record in enriched:
            global_seen.add(record["id"])

    fill_topic_shortfalls(
        session,
        topic_records,
        cutoff_ts,
        target_per_topic,
        global_seen,
    )

    records = balance_and_deduplicate(topic_records, target_per_topic)
    validate_records(records)
    export_csv(records, args.output)

    logger.info(
        "Done. Final dataset: %d unique posts across %d topics.",
        len(records),
        num_topics,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        logger.error("Interrupted by user.")
        raise SystemExit(130)
    except Exception as exc:
        logger.error("Collection failed: %s", exc)
        raise SystemExit(1)
