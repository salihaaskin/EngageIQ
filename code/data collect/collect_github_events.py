#!/usr/bin/env python3
"""Collect public GitHub events from GH Archive and export to CSV."""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import random
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

TARGET_TOTAL = 7000
RANDOM_SEED = 42
GHARCHIVE_BASE = "https://data.gharchive.org"
USER_AGENT = "EngageIQ-GHArchive-Collector/1.0 (research dataset)"
ARCHIVE_TIMEOUT = 300
MAX_EVENTS_PER_HOUR = 120_000
MAX_COLLECTED_PER_HOUR = 450
DOWNLOAD_RETRIES = 3

# Display topic name -> keyword phrases for matching
TOPIC_KEYWORDS: dict[str, list[str]] = {
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
    "event_type",
    "repo_name",
    "actor_login",
    "title",
    "text",
    "url",
    "score",
    "num_comments",
    "created_utc",
    "domain",
    "topic",
}

DELETED_MARKERS = {"[deleted]", "[removed]", "[bot]", ""}

# Event types we can map to title/text
SUPPORTED_EVENT_TYPES = frozenset(
    {
        "PushEvent",
        "IssuesEvent",
        "PullRequestEvent",
        "IssueCommentEvent",
        "PullRequestReviewCommentEvent",
        "PullRequestReviewEvent",
        "CommitCommentEvent",
        "ReleaseEvent",
        "CreateEvent",
        "ForkEvent",
        "WatchEvent",
        "PublicEvent",
        "GollumEvent",
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect GitHub events from GH Archive and export to CSV."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("Data/github_events.csv"),
        help="Output CSV file path (default: Data/github_events.csv)",
    )
    parser.add_argument(
        "--target-total",
        type=int,
        default=TARGET_TOTAL,
        help=f"Approximate total events to collect (default: {TARGET_TOTAL})",
    )
    parser.add_argument(
        "--years",
        type=float,
        default=2.0,
        help="Only include events from the last N years (default: 2)",
    )
    parser.add_argument(
        "--max-hours",
        type=int,
        default=720,
        help="Maximum hourly archive files to scan (default: 720)",
    )
    return parser.parse_args()


def compile_topic_patterns() -> dict[str, list[re.Pattern[str]]]:
    patterns: dict[str, list[re.Pattern[str]]] = {}
    for topic, keywords in TOPIC_KEYWORDS.items():
        compiled: list[re.Pattern[str]] = []
        for keyword in keywords:
            escaped = re.escape(keyword.lower())
            if re.search(r"\w", keyword):
                compiled.append(re.compile(rf"(?<!\w){escaped}(?!\w)", re.IGNORECASE))
            else:
                compiled.append(re.compile(escaped, re.IGNORECASE))
        patterns[topic] = compiled
    return patterns


def format_created_utc(created_at: str) -> str | None:
    if not created_at:
        return None
    text = created_at.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def archive_url(dt: datetime) -> str:
    hour = dt.hour
    return f"{GHARCHIVE_BASE}/{dt:%Y-%m-%d}-{hour}.json.gz"


def iter_hour_slots(start: datetime, end: datetime) -> list[datetime]:
    slots: list[datetime] = []
    current = start.replace(minute=0, second=0, microsecond=0)
    end_floor = end.replace(minute=0, second=0, microsecond=0)
    while current <= end_floor:
        slots.append(current)
        current += timedelta(hours=1)
    return slots


def iter_events_from_archive(url: str) -> Iterator[dict[str, Any]]:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    last_error: Exception | None = None

    for attempt in range(1, DOWNLOAD_RETRIES + 1):
        try:
            with urlopen(request, timeout=ARCHIVE_TIMEOUT) as response:
                with gzip.GzipFile(fileobj=response) as gz:
                    for raw_line in gz:
                        if not raw_line.strip():
                            continue
                        try:
                            yield json.loads(raw_line)
                        except json.JSONDecodeError:
                            continue
            return
        except HTTPError as exc:
            if exc.code == 404:
                logger.debug("Archive not found: %s", url)
                return
            last_error = exc
        except (URLError, gzip.BadGzipFile, EOFError, OSError) as exc:
            last_error = exc
            logger.warning(
                "Archive read failed (attempt %d/%d) %s: %s",
                attempt,
                DOWNLOAD_RETRIES,
                url,
                exc,
            )
            time.sleep(2 * attempt)

    if last_error is not None:
        logger.warning("Skipping archive after retries: %s", url)


def safe_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def collect_payload_strings(payload: Any, depth: int = 0) -> list[str]:
    if depth > 6 or payload is None:
        return []
    parts: list[str] = []
    if isinstance(payload, str):
        parts.append(payload)
    elif isinstance(payload, dict):
        for key, value in payload.items():
            if key in ("avatar_url", "gravatar_id", "sha", "id", "node_id"):
                continue
            parts.extend(collect_payload_strings(value, depth + 1))
    elif isinstance(payload, list):
        for item in payload[:50]:
            parts.extend(collect_payload_strings(item, depth + 1))
    return parts


def extract_repo_description(event: dict[str, Any]) -> str:
    repo = event.get("repo") or {}
    if isinstance(repo, dict):
        for key in ("description", "repo_description"):
            value = repo.get(key)
            if value:
                return safe_str(value)
    org = event.get("org") or {}
    if isinstance(org, dict):
        value = org.get("description")
        if value:
            return safe_str(value)
    payload = event.get("payload") or {}
    if isinstance(payload, dict):
        for key in ("description", "repository", "repo"):
            nested = payload.get(key)
            if isinstance(nested, dict):
                desc = nested.get("description")
                if desc:
                    return safe_str(desc)
    return ""


def extract_title_and_text(
    event_type: str, payload: dict[str, Any], repo_name: str
) -> tuple[str, str]:
    title = ""
    text = ""

    if event_type == "PushEvent":
        commits = payload.get("commits") or []
        messages: list[str] = []
        for commit in commits[:20]:
            if not isinstance(commit, dict):
                continue
            message = safe_str(commit.get("message"))
            if message:
                messages.append(message)
        if messages:
            title = messages[0][:500]
            text = "\n".join(messages[1:])[:8000] if len(messages) > 1 else ""
            return title, text
        ref = safe_str(payload.get("ref"))
        if ref:
            title = f"Push to {ref}"
        return title, ""

    if event_type == "IssuesEvent":
        issue = payload.get("issue") or {}
        if isinstance(issue, dict):
            title = safe_str(issue.get("title"))[:500]
            text = safe_str(issue.get("body"))[:8000]
            return title, text

    if event_type == "PullRequestEvent":
        pr = payload.get("pull_request") or {}
        if isinstance(pr, dict):
            title = safe_str(pr.get("title"))[:500]
            text = safe_str(pr.get("body"))[:8000]
            return title, text

    if event_type in ("IssueCommentEvent", "CommitCommentEvent"):
        comment = payload.get("comment") or {}
        issue = payload.get("issue") or {}
        if isinstance(issue, dict):
            title = safe_str(issue.get("title"))[:500]
        if isinstance(comment, dict):
            body = safe_str(comment.get("body"))
            if not title and body:
                title = body[:200]
            text = body[:8000]
        return title, text

    if event_type in ("PullRequestReviewCommentEvent", "PullRequestReviewEvent"):
        comment = payload.get("comment") or {}
        pr = payload.get("pull_request") or {}
        if isinstance(pr, dict):
            title = safe_str(pr.get("title"))[:500]
        if isinstance(comment, dict):
            body = safe_str(comment.get("body"))
            if body:
                text = body[:8000]
            elif not title:
                title = body[:200] if body else title
        return title, text

    if event_type == "ReleaseEvent":
        release = payload.get("release") or {}
        if isinstance(release, dict):
            title = safe_str(release.get("name") or release.get("tag_name"))[:500]
            text = safe_str(release.get("body"))[:8000]
            return title, text

    if event_type == "CreateEvent":
        desc = safe_str(payload.get("description"))
        ref_type = safe_str(payload.get("ref_type"))
        ref = safe_str(payload.get("ref"))
        if ref_type and ref:
            title = f"Create {ref_type} {ref}"
        elif desc:
            title = desc[:500]
        return title, desc[:8000]

    if event_type == "GollumEvent":
        pages = payload.get("pages") or []
        page_titles: list[str] = []
        for page in pages[:10]:
            if isinstance(page, dict):
                page_title = safe_str(page.get("title"))
                if page_title:
                    page_titles.append(page_title)
        if page_titles:
            title = page_titles[0][:500]
            text = "\n".join(page_titles[1:])[:8000]
        return title, text

    if event_type == "WatchEvent":
        title = f"Starred {repo_name}"
        return title, ""

    if event_type == "ForkEvent":
        forkee = payload.get("forkee")
        forker = ""
        if isinstance(forkee, dict):
            forker = safe_str(forkee.get("full_name"))
        title = f"Forked {repo_name}" + (f" to {forker}" if forker else "")
        return title[:500], ""

    if event_type == "PublicEvent":
        title = f"Repository {repo_name} made public"
        return title, ""

    return title, ""


def build_event_url(
    event_type: str, repo_name: str, payload: dict[str, Any]
) -> str | None:
    if not repo_name or "/" not in repo_name:
        return None

    if event_type == "IssuesEvent":
        issue = payload.get("issue") or {}
        if isinstance(issue, dict):
            url = issue.get("html_url")
            if url:
                return safe_str(url)

    if event_type in ("PullRequestEvent", "PullRequestReviewCommentEvent"):
        pr = payload.get("pull_request") or {}
        if isinstance(pr, dict):
            url = pr.get("html_url")
            if url:
                return safe_str(url)

    if event_type == "PushEvent":
        commits = payload.get("commits") or []
        for commit in commits:
            if isinstance(commit, dict):
                url = commit.get("url")
                if url:
                    return safe_str(url)

    if event_type == "ReleaseEvent":
        release = payload.get("release") or {}
        if isinstance(release, dict):
            url = release.get("html_url")
            if url:
                return safe_str(url)

    if event_type in ("IssueCommentEvent", "CommitCommentEvent"):
        comment = payload.get("comment") or {}
        if isinstance(comment, dict):
            url = comment.get("html_url")
            if url:
                return safe_str(url)

    return f"https://github.com/{repo_name}"


def build_search_blob(event: dict[str, Any], title: str, text: str) -> str:
    repo = event.get("repo") or {}
    repo_name = safe_str(repo.get("name") if isinstance(repo, dict) else "")
    payload = event.get("payload") or {}
    payload_parts = collect_payload_strings(payload)
    description = extract_repo_description(event)
    parts = [repo_name, description, title, text, *payload_parts]
    return "\n".join(p for p in parts if p).lower()


def matching_topics(
    search_blob: str, topic_patterns: dict[str, list[re.Pattern[str]]]
) -> list[str]:
    matched: list[str] = []
    for topic, patterns in topic_patterns.items():
        if any(pattern.search(search_blob) for pattern in patterns):
            matched.append(topic)
    return matched


def pick_topic_for_balance(
    matched: list[str], topic_counts: dict[str, int]
) -> str:
    return min(matched, key=lambda topic: topic_counts.get(topic, 0))


def is_valid_event(event: dict[str, Any], cutoff: datetime) -> bool:
    if event.get("public") is not True:
        return False

    event_id = event.get("id")
    if event_id is None:
        return False

    event_type = safe_str(event.get("type"))
    if event_type not in SUPPORTED_EVENT_TYPES:
        return False

    actor = event.get("actor") or {}
    if not isinstance(actor, dict):
        return False
    actor_login = safe_str(actor.get("login"))
    if not actor_login or actor_login.lower() in DELETED_MARKERS:
        return False

    repo = event.get("repo") or {}
    if not isinstance(repo, dict):
        return False
    repo_name = safe_str(repo.get("name"))
    if not repo_name or "/" not in repo_name:
        return False

    created_utc = format_created_utc(safe_str(event.get("created_at")))
    if not created_utc:
        return False

    try:
        event_dt = datetime.strptime(created_utc, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
    except ValueError:
        return False

    if event_dt < cutoff:
        return False

    return True


def event_to_record(
    event: dict[str, Any], topic: str
) -> dict[str, Any] | None:
    event_type = safe_str(event.get("type"))
    payload = event.get("payload") or {}
    if not isinstance(payload, dict):
        payload = {}

    repo = event.get("repo") or {}
    actor = event.get("actor") or {}
    repo_name = safe_str(repo.get("name"))
    actor_login = safe_str(actor.get("login"))

    title, text = extract_title_and_text(event_type, payload, repo_name)
    if not title:
        action = safe_str(payload.get("action"))
        if action:
            title = f"{event_type}: {action}"
        else:
            title = f"{event_type} on {repo_name}"

    if title.lower() in DELETED_MARKERS:
        return None

    url = build_event_url(event_type, repo_name, payload)
    if not url:
        return None

    created_utc = format_created_utc(safe_str(event.get("created_at")))
    if not created_utc:
        return None

    return {
        "id": str(event["id"]),
        "source": "github",
        "event_type": event_type,
        "repo_name": repo_name,
        "actor_login": actor_login,
        "title": title,
        "text": text or "",
        "url": url,
        "score": None,
        "num_comments": None,
        "created_utc": created_utc,
        "domain": "github.com",
        "topic": topic,
    }


def balance_final_records(
    records: list[dict[str, Any]],
    target_per_topic: int,
) -> list[dict[str, Any]]:
    rng = random.Random(RANDOM_SEED)
    by_topic: dict[str, list[dict[str, Any]]] = {topic: [] for topic in TOPIC_KEYWORDS}
    global_seen: set[str] = set()

    for record in records:
        topic = record["topic"]
        if record["id"] in global_seen:
            continue
        if topic not in by_topic:
            continue
        global_seen.add(record["id"])
        by_topic[topic].append(record)

    final: list[dict[str, Any]] = []
    topic_order = list(TOPIC_KEYWORDS.keys())
    rng.shuffle(topic_order)

    for topic in topic_order:
        bucket = by_topic[topic]
        rng.shuffle(bucket)
        taken = 0
        for record in bucket:
            if taken >= target_per_topic:
                break
            final.append(record)
            taken += 1
        if taken < target_per_topic:
            logger.warning(
                "Topic '%s' only has %d events (target %d)",
                topic,
                taken,
                target_per_topic,
            )

    rng.shuffle(final)
    return final


def validate_records(records: list[dict[str, Any]]) -> None:
    if not records:
        raise ValueError("No records to validate.")

    for index, record in enumerate(records):
        missing = REQUIRED_FIELDS - record.keys()
        if missing:
            raise ValueError(f"Record {index} missing fields: {missing}")

        if record["source"] != "github":
            raise ValueError(
                f"Record {index} has invalid source: {record['source']!r}"
            )

        if record["domain"] != "github.com":
            raise ValueError(
                f"Record {index} has invalid domain: {record['domain']!r}"
            )

        if not isinstance(record["id"], str) or not record["id"]:
            raise ValueError(f"Record {index} has invalid id.")

        for field in (
            "event_type",
            "repo_name",
            "actor_login",
            "title",
            "text",
            "url",
            "created_utc",
            "topic",
        ):
            if not isinstance(record[field], str):
                raise ValueError(
                    f"Record {index} field '{field}' must be a string."
                )

        if not record["title"].strip():
            raise ValueError(f"Record {index} has empty title.")

        if not record["repo_name"].strip() or "/" not in record["repo_name"]:
            raise ValueError(f"Record {index} has invalid repo_name.")

        if record["score"] is not None or record["num_comments"] is not None:
            raise ValueError(
                f"Record {index} score/num_comments must be null for GitHub events."
            )

        try:
            datetime.strptime(record["created_utc"], "%Y-%m-%dT%H:%M:%SZ").replace(
                tzinfo=timezone.utc
            )
        except ValueError as exc:
            raise ValueError(
                f"Record {index} has invalid created_utc: {record['created_utc']!r}"
            ) from exc

        if record["topic"] not in TOPIC_KEYWORDS:
            raise ValueError(
                f"Record {index} has unknown topic: {record['topic']!r}"
            )

    ids = [record["id"] for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate event IDs found after deduplication.")

    logger.info("Schema validation passed for %d records.", len(records))


def export_csv(records: list[dict[str, Any]], output_path: Path) -> None:
    df = pd.DataFrame(records)

    for column in REQUIRED_FIELDS:
        if column not in df.columns:
            raise ValueError(f"Missing column in DataFrame: {column}")

    df["score"] = pd.array([None] * len(df), dtype=pd.Int64Dtype())
    df["num_comments"] = pd.array([None] * len(df), dtype=pd.Int64Dtype())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Wrote %d records to %s", len(records), output_path)


def collect_events(
    target_total: int,
    years: float,
    max_hours: int,
) -> list[dict[str, Any]]:
    end = datetime.now(timezone.utc)
    cutoff = end - timedelta(days=int(years * 365.25))
    num_topics = len(TOPIC_KEYWORDS)
    target_per_topic = max(1, target_total // num_topics)

    slots = iter_hour_slots(cutoff, end)
    rng = random.Random(RANDOM_SEED)
    rng.shuffle(slots)

    logger.info(
        "Collecting ~%d events per topic (%d topics, target ~%d), "
        "from %s to %s, up to %d hourly archives",
        target_per_topic,
        num_topics,
        target_per_topic * num_topics,
        cutoff.strftime("%Y-%m-%dT%H:%M:%SZ"),
        end.strftime("%Y-%m-%dT%H:%M:%SZ"),
        max_hours,
    )

    topic_patterns = compile_topic_patterns()
    topic_counts: dict[str, int] = {topic: 0 for topic in TOPIC_KEYWORDS}
    global_seen: set[str] = set()
    collected: list[dict[str, Any]] = []

    for hour_index, slot in enumerate(slots[:max_hours]):
        if all(count >= target_per_topic for count in topic_counts.values()):
            logger.info("All topic quotas met after %d hours.", hour_index)
            break

        url = archive_url(slot)
        hour_added = 0
        events_scanned = 0
        logger.info("Scanning archive %s", url)
        try:
            for event in iter_events_from_archive(url):
                events_scanned += 1
                if events_scanned > MAX_EVENTS_PER_HOUR:
                    logger.info(
                        "Reached %d event cap for hour %s, moving on",
                        MAX_EVENTS_PER_HOUR,
                        slot.strftime("%Y-%m-%d-%H"),
                    )
                    break

                if all(count >= target_per_topic for count in topic_counts.values()):
                    break

                if hour_added >= MAX_COLLECTED_PER_HOUR:
                    logger.info(
                        "Collected %d events for hour %s, moving to next archive",
                        hour_added,
                        slot.strftime("%Y-%m-%d-%H"),
                    )
                    break

                if not is_valid_event(event, cutoff):
                    continue

                event_id = str(event.get("id"))
                if event_id in global_seen:
                    continue

                repo = event.get("repo") or {}
                repo_name = safe_str(repo.get("name"))
                payload = event.get("payload") or {}
                if not isinstance(payload, dict):
                    payload = {}

                event_type = safe_str(event.get("type"))
                title, text = extract_title_and_text(event_type, payload, repo_name)
                if not title:
                    action = safe_str(payload.get("action"))
                    title = (
                        f"{event_type}: {action}"
                        if action
                        else f"{event_type} on {repo_name}"
                    )

                search_blob = build_search_blob(event, title, text)
                matched = matching_topics(search_blob, topic_patterns)
                if not matched:
                    continue

                if all(
                    topic_counts.get(topic, 0) >= target_per_topic for topic in matched
                ):
                    continue

                topic = pick_topic_for_balance(matched, topic_counts)
                if topic_counts.get(topic, 0) >= target_per_topic:
                    continue

                record = event_to_record(event, topic)
                if record is None:
                    continue

                global_seen.add(record["id"])
                topic_counts[topic] = topic_counts.get(topic, 0) + 1
                collected.append(record)
                hour_added += 1

        except HTTPError as exc:
            if exc.code != 404:
                logger.warning("HTTP error for %s: %s", url, exc)
            continue
        except (URLError, gzip.BadGzipFile, EOFError, OSError) as exc:
            logger.warning("Error processing %s: %s", url, exc)
            continue

        if hour_added:
            logger.info(
                "Hour %s: +%d events (total %d, min topic %d / target %d)",
                slot.strftime("%Y-%m-%d-%H"),
                hour_added,
                len(collected),
                min(topic_counts.values()),
                target_per_topic,
            )

        if (hour_index + 1) % 20 == 0:
            time.sleep(0.5)

    return balance_final_records(collected, target_per_topic)


def main() -> int:
    args = parse_args()
    records = collect_events(args.target_total, args.years, args.max_hours)

    if len(records) < args.target_total * 0.8:
        logger.warning(
            "Collected %d records (target ~%d). Increase --max-hours if needed.",
            len(records),
            args.target_total,
        )

    validate_records(records)
    export_csv(records, args.output)

    topic_dist: dict[str, int] = {}
    for record in records:
        topic_dist[record["topic"]] = topic_dist.get(record["topic"], 0) + 1

    logger.info("Done. Final dataset: %d unique events.", len(records))
    for topic, count in sorted(topic_dist.items()):
        logger.info("  %s: %d", topic, count)
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
