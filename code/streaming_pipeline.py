#!/usr/bin/env python3
"""
Real-time streaming pipeline for EngageIQ.

Simulates a Kafka-style event-driven ingestion pipeline using a thread-safe
in-process queue. Each collector runs as a producer thread that emits events;
a consumer thread normalizes, deduplicates, and routes them to the live feed.

Architecture:
  Producers (GitHub, HN, Reddit) → queue → Consumer (normalize + dedup)
                                                  ↓
                                          live_feed.jsonl  (append-only log)
                                                  ↓
                                    Streamlit reads tail for live preview

Run:
  python streaming_pipeline.py [--duration 60] [--sources github hn reddit]

To integrate with an actual Kafka broker, replace SimpleQueue with
a confluent_kafka.Producer / Consumer pair and set KAFKA_MODE=true.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import queue
import random
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(threadName)-20s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Event schema
# ---------------------------------------------------------------------------

@dataclass
class StreamEvent:
    event_id: str
    source: str          # github | hn | reddit
    title: str
    url: str
    score: int
    num_comments: int
    domain: str
    ingested_at: float = field(default_factory=time.time)

    def fingerprint(self) -> str:
        return hashlib.md5(f"{self.source}::{self.url}".encode()).hexdigest()


# ---------------------------------------------------------------------------
# Simulated producers (replace with real API calls in production)
# ---------------------------------------------------------------------------

_SAMPLE_TITLES = {
    "github": [
        "Add support for streaming embeddings in batch mode",
        "Fix FAISS index rebuild on large datasets",
        "good first issue: improve CLI error messages",
        "Implement MMR diversity re-ranking for recommendations",
        "good first issue: add unit tests for ranking.py",
        "Upgrade sentence-transformers to 2.x API",
        "good first issue: document environment variable setup",
    ],
    "hn": [
        "Ask HN: How are you handling real-time ML inference?",
        "Show HN: Open-source engagement analytics dashboard",
        "LLM fine-tuning on private codebases — lessons learned",
        "Why RAG systems fail in production",
        "Building developer communities at scale",
    ],
    "reddit": [
        "r/MachineLearning: Best practices for embedding freshness?",
        "r/datascience: Week-over-week trend detection with Spark",
        "r/learnmachinelearning: resources for RAG beginners",
        "r/devops: Kafka vs SQS for streaming ML pipelines",
        "r/startups: Measuring developer engagement ROI",
    ],
}

_DOMAINS = ["machine_learning", "developer_tools", "cloud_computing", "research", "devops", "AI Research"]


def _make_event(source: str) -> StreamEvent:
    titles = _SAMPLE_TITLES[source]
    title = random.choice(titles)
    return StreamEvent(
        event_id=hashlib.md5(f"{source}{title}{time.time()}".encode()).hexdigest()[:12],
        source=source,
        title=title,
        url=f"https://{source}.example.com/{random.randint(1000, 9999)}",
        score=random.randint(1, 500),
        num_comments=random.randint(0, 150),
        domain=random.choice(_DOMAINS),
    )


def producer(source: str, event_queue: queue.Queue, interval: float, stop_event: threading.Event) -> None:
    """Emit events from one source on a fixed interval."""
    logger.info("Producer started  source=%s  interval=%.1fs", source, interval)
    while not stop_event.is_set():
        evt = _make_event(source)
        event_queue.put(evt)
        logger.info("EMIT  %-8s  id=%-12s  score=%-4d  %s", source, evt.event_id, evt.score, evt.title[:60])
        time.sleep(interval + random.uniform(-interval * 0.2, interval * 0.2))
    logger.info("Producer stopped  source=%s", source)


# ---------------------------------------------------------------------------
# Consumer
# ---------------------------------------------------------------------------

def consumer(
    event_queue: queue.Queue,
    output_path: Path,
    stop_event: threading.Event,
    seen: set[str],
) -> None:
    """Consume events: deduplicate, normalize, append to live feed log."""
    logger.info("Consumer started  output=%s", output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed = 0
    duplicates = 0

    with output_path.open("a", encoding="utf-8") as fh:
        while not stop_event.is_set() or not event_queue.empty():
            try:
                evt = event_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            fp = evt.fingerprint()
            if fp in seen:
                duplicates += 1
                event_queue.task_done()
                continue

            seen.add(fp)
            fh.write(json.dumps(asdict(evt)) + "\n")
            fh.flush()
            processed += 1
            event_queue.task_done()
            logger.info(
                "STORED %-8s  id=%-12s  processed=%-4d  dupes=%-3d",
                evt.source, evt.event_id, processed, duplicates,
            )

    logger.info("Consumer stopped  processed=%d  duplicates=%d", processed, duplicates)


# ---------------------------------------------------------------------------
# Stats reporter
# ---------------------------------------------------------------------------

def stats_reporter(output_path: Path, stop_event: threading.Event, interval: float = 10.0) -> None:
    """Periodically report pipeline throughput."""
    while not stop_event.is_set():
        time.sleep(interval)
        if output_path.exists():
            lines = sum(1 for _ in output_path.open(encoding="utf-8"))
            logger.info("STATS  live_feed_records=%d  path=%s", lines, output_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(
    duration: int,
    sources: list[str],
    output_path: Path,
    intervals: dict[str, float] | None = None,
) -> None:
    if intervals is None:
        intervals = {"github": 2.0, "hn": 3.0, "reddit": 4.0}

    event_queue: queue.Queue = queue.Queue(maxsize=1000)
    stop_event = threading.Event()
    seen: set[str] = set()

    threads: list[threading.Thread] = []

    for source in sources:
        t = threading.Thread(
            target=producer,
            args=(source, event_queue, intervals.get(source, 3.0), stop_event),
            name=f"producer-{source}",
            daemon=True,
        )
        threads.append(t)

    consumer_thread = threading.Thread(
        target=consumer,
        args=(event_queue, output_path, stop_event, seen),
        name="consumer",
        daemon=True,
    )
    threads.append(consumer_thread)

    stats_thread = threading.Thread(
        target=stats_reporter,
        args=(output_path, stop_event),
        name="stats",
        daemon=True,
    )
    threads.append(stats_thread)

    logger.info("Pipeline starting  sources=%s  duration=%ds  output=%s", sources, duration, output_path)
    for t in threads:
        t.start()

    try:
        time.sleep(duration)
    except KeyboardInterrupt:
        logger.info("Interrupted — stopping pipeline.")
    finally:
        stop_event.set()
        for t in threads:
            t.join(timeout=5.0)

    total = sum(1 for _ in output_path.open(encoding="utf-8")) if output_path.exists() else 0
    logger.info("Pipeline finished  total_records=%d  output=%s", total, output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EngageIQ real-time streaming pipeline.")
    parser.add_argument("--duration", type=int, default=30, help="How long to run (seconds). Default: 30.")
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=["github", "hn", "reddit"],
        default=["github", "hn", "reddit"],
        help="Data sources to stream from.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("Data/live_feed.jsonl"),
        help="Append-only JSONL log for ingested events.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run(args.duration, args.sources, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
