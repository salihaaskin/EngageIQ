#!/usr/bin/env python3
"""
Batch analytics and trend detection for the EngageIQ offline opportunity snapshot.

Outputs dashboard-ready CSV files:
  analytics/active_communities.csv
  analytics/opportunity_volume_over_time.csv
  analytics/category_distribution.csv
  analytics/domain_distribution.csv
  analytics/trending_topics.csv

Example:
  spark-submit analytics_spark.py \
    --input Data/snapshots/engageiq_snapshot_v1.parquet \
    --output-dir analytics
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import (
    col,
    concat_ws,
    count,
    lit,
    to_date,
    trim,
    when,
)

DEFAULT_INPUT = Path("Data/snapshots/engageiq_snapshot_v1.parquet")
FALLBACK_INPUT = Path("Embeddings/snapshots/v1.parquet")
DEFAULT_OUTPUT_DIR = Path("analytics")

STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "new",
    "of",
    "on",
    "or",
    "show",
    "the",
    "to",
    "with",
    "you",
    "your",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Spark batch analytics over the EngageIQ offline snapshot."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Offline Parquet snapshot path (default: {DEFAULT_INPUT}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Dashboard CSV output directory (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--trending-max-features",
        type=int,
        default=30,
        help="Maximum number of trending title terms/ngrams to export.",
    )
    return parser.parse_args()


def resolve_input_path(input_path: Path) -> Path:
    if input_path.exists():
        return input_path
    if input_path == DEFAULT_INPUT and FALLBACK_INPUT.exists():
        return FALLBACK_INPUT
    raise FileNotFoundError(
        f"Snapshot not found at {input_path}. "
        f"Run Data/merge.py first or pass --input {FALLBACK_INPUT}."
    )


def ensure_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)


def write_spark_csv(df: DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.toPandas().to_csv(output_path, index=False)


def existing_column(df: DataFrame, candidates: Iterable[str]) -> str | None:
    available = set(df.columns)
    for candidate in candidates:
        if candidate in available:
            return candidate
    return None


def add_text_column(df: DataFrame, output_column: str, candidates: Iterable[str]) -> DataFrame:
    source = existing_column(df, candidates)
    if source is None:
        return df.withColumn(output_column, lit("unknown"))
    return df.withColumn(
        output_column,
        when(trim(col(source).cast("string")) == "", lit("unknown"))
        .otherwise(col(source).cast("string")),
    )


def normalize_snapshot(df: DataFrame) -> DataFrame:
    """
    Normalize expected dashboard columns across raw and EngageIQ snapshot schemas.

    Requested analytics columns:
      community, category, domain, created_at, engagement, comments, title

    Current snapshot aliases:
      source/domain -> community fallback
      topic_category/domain_category -> category/domain rollups
      created_utc -> created_at
      score/num_comments -> engagement/comments
    """
    out = df
    out = add_text_column(out, "analytics_title", ("title",))
    out = add_text_column(out, "analytics_source", ("source",))
    out = add_text_column(out, "analytics_category", ("category", "topic_category"))
    out = add_text_column(out, "analytics_domain", ("domain", "domain_category"))

    community_source = existing_column(out, ("community",))
    if community_source is not None:
        out = add_text_column(out, "analytics_community", ("community",))
    else:
        out = out.withColumn(
            "analytics_community",
            concat_ws(
                "::",
                col("analytics_source"),
                col("analytics_domain"),
            ),
        )

    created_source = existing_column(out, ("created_at", "created_utc", "created_at_utc"))
    if created_source is None:
        out = out.withColumn("analytics_created_at", lit(None).cast("string"))
    else:
        out = out.withColumn("analytics_created_at", col(created_source).cast("string"))

    engagement_source = existing_column(out, ("engagement", "score", "upvotes", "points"))
    comments_source = existing_column(out, ("comments", "num_comments", "comment_count"))
    out = out.withColumn(
        "analytics_engagement",
        col(engagement_source).cast("double") if engagement_source else lit(0.0),
    )
    out = out.withColumn(
        "analytics_comments",
        col(comments_source).cast("double") if comments_source else lit(0.0),
    )
    return out


def compute_active_communities(df: DataFrame) -> DataFrame:
    return (
        df.groupBy(col("analytics_community").alias("community"))
        .agg(count("*").alias("post_count"))
        .orderBy(col("post_count").desc(), col("community").asc())
    )


def compute_volume_over_time(df: DataFrame) -> DataFrame:
    return (
        df.withColumn("day", to_date("analytics_created_at"))
        .where(col("day").isNotNull())
        .groupBy("day")
        .agg(count("*").alias("count"))
        .orderBy("day")
    )


def compute_weekly_volume(volume_df: DataFrame) -> DataFrame:
    """Compute week-over-week volume change from the daily volume DataFrame."""
    from pyspark.sql.functions import date_trunc, lag, round as spark_round
    from pyspark.sql.window import Window

    weekly = (
        volume_df
        .withColumn("week", date_trunc("week", col("day")))
        .groupBy("week")
        .agg(count("*").alias("weekly_count"))
        .orderBy("week")
    )
    window_spec = Window.orderBy("week")
    weekly = (
        weekly
        .withColumn("prev_week_count", lag("weekly_count", 1).over(window_spec))
        .withColumn(
            "wow_delta",
            col("weekly_count") - col("prev_week_count"),
        )
        .withColumn(
            "wow_pct_change",
            spark_round(
                (col("wow_delta") / col("prev_week_count")) * 100,
                2,
            ),
        )
        .where(col("prev_week_count").isNotNull())
    )
    return weekly


def compute_category_distribution(df: DataFrame) -> DataFrame:
    return (
        df.groupBy(col("analytics_category").alias("category"))
        .agg(count("*").alias("count"))
        .orderBy(col("count").desc(), col("category").asc())
    )


def compute_domain_distribution(df: DataFrame) -> DataFrame:
    return (
        df.groupBy(col("analytics_domain").alias("domain"))
        .agg(count("*").alias("count"))
        .orderBy(col("count").desc(), col("domain").asc())
    )


def tokenize_title(title: str) -> list[str]:
    tokens = [
        token
        for token in re.findall(r"[a-zA-Z][a-zA-Z0-9_+#.-]{1,}", title.lower())
        if token not in STOP_WORDS
    ]
    bigrams = [f"{left} {right}" for left, right in zip(tokens, tokens[1:])]
    return tokens + bigrams


def fallback_tfidf_topics(titles: list[str], max_features: int) -> list[tuple[str, float]]:
    document_tokens = [tokenize_title(title) for title in titles if title.strip()]
    if not document_tokens:
        return []

    document_count = len(document_tokens)
    term_frequency: Counter[str] = Counter()
    document_frequency: Counter[str] = Counter()

    for tokens in document_tokens:
        counts = Counter(tokens)
        term_frequency.update(counts)
        document_frequency.update(counts.keys())

    scores: dict[str, float] = {}
    for term, tf in term_frequency.items():
        idf = math.log((1 + document_count) / (1 + document_frequency[term])) + 1.0
        scores[term] = float(tf * idf)

    return sorted(scores.items(), key=lambda item: item[1], reverse=True)[:max_features]


def compute_trending_topics(df: DataFrame, output_path: Path, max_features: int) -> None:
    titles_pdf = df.select(col("analytics_title").alias("title")).dropna().toPandas()
    titles = titles_pdf["title"].astype(str)

    try:
        import pandas as pd
        from sklearn.feature_extraction.text import TfidfVectorizer

        vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words="english",
            ngram_range=(1, 2),
        )
        tfidf_matrix = vectorizer.fit_transform(titles)
        trending_topics = pd.DataFrame(
            {
                "topic": vectorizer.get_feature_names_out(),
                "score": tfidf_matrix.sum(axis=0).A1,
            }
        ).sort_values("score", ascending=False)
        trending_topics.to_csv(output_path, index=False)
        return
    except Exception:
        topics = fallback_tfidf_topics(list(titles), max_features)

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["topic", "score"])
        writer.writeheader()
        for topic, score in topics:
            writer.writerow({"topic": topic, "score": score})


def run(input_path: Path, output_dir: Path, trending_max_features: int) -> None:
    snapshot_path = resolve_input_path(input_path)
    ensure_output_dir(output_dir)

    spark = (
        SparkSession.builder.appName("EngageIQ Analytics")
        .getOrCreate()
    )

    try:
        df = spark.read.parquet(str(snapshot_path))
        df = normalize_snapshot(df).cache()

        active_communities = compute_active_communities(df)
        volume_over_time = compute_volume_over_time(df)
        weekly_volume = compute_weekly_volume(volume_over_time)
        category_distribution = compute_category_distribution(df)
        domain_distribution = compute_domain_distribution(df)

        active_communities.show()
        volume_over_time.show()
        weekly_volume.show()
        category_distribution.show()
        domain_distribution.show()

        write_spark_csv(
            active_communities,
            output_dir / "active_communities.csv",
        )
        write_spark_csv(
            volume_over_time,
            output_dir / "opportunity_volume_over_time.csv",
        )
        write_spark_csv(
            weekly_volume,
            output_dir / "weekly_volume_wow.csv",
        )
        write_spark_csv(
            category_distribution,
            output_dir / "category_distribution.csv",
        )
        write_spark_csv(
            domain_distribution,
            output_dir / "domain_distribution.csv",
        )
        compute_trending_topics(
            df,
            output_dir / "trending_topics.csv",
            trending_max_features,
        )
    finally:
        spark.stop()


def main() -> int:
    args = parse_args()
    run(args.input, args.output_dir, args.trending_max_features)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
