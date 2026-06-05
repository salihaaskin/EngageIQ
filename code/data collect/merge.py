#!/usr/bin/env python3
# python merge.py \
#    --input-dir Data/raw \
#    --output Data/snapshots/finaldataset.csv
"""
Merge CSV datasets through a staged offline pipeline:

  Raw Data
    -> Schema Standardization
    -> Merge
    -> Deduplication
    -> Domain Classification
    -> Domain Coverage Check
    -> Feature Engineering
    -> Offline Snapshot Export (CSV)
    -> Snapshot Metadata
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "1.0.0"

REQUIRED_COLUMNS = [
    "id",
    "source",
    "title",
    "text",
    "url",
    "score",
    "num_comments",
    "created_utc",
    "domain",
]

COLUMN_ALIASES = {
    "created_at": "created_utc",
    "created_at_utc": "created_utc",
    "comments": "num_comments",
    "comment_count": "num_comments",
    "upvotes": "score",
    "points": "score",
}

DOMAIN_CATEGORY_MAP = {
    "github.com": "developer_tools",
    "gitlab.com": "developer_tools",
    "openai.com": "artificial_intelligence",
    "anthropic.com": "artificial_intelligence",
    "deepmind.google": "artificial_intelligence",
    "huggingface.co": "machine_learning",
    "arxiv.org": "research",
    "paperswithcode.com": "research",
    "aws.amazon.com": "cloud_computing",
    "azure.microsoft.com": "cloud_computing",
    "cloud.google.com": "cloud_computing",
    "kubernetes.io": "devops",
    "docker.com": "devops",
    "postgresql.org": "databases",
    "sqlite.org": "databases",
    "mongodb.com": "databases",
    "react.dev": "frontend_development",
    "nodejs.org": "backend_development",
}

TOPIC_KEYWORDS = {
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

FINAL_COLUMNS = [
    "id",
    "source",
    "title",
    "text",
    "url",
    "score",
    "num_comments",
    "created_utc",
    "domain",
    "domain_category",
    "topic_category",
    "title_length",
    "age_days",
    "engagement_score",
    "snapshot_utc",
]

MIN_DOMAIN_COVERAGE_FRACTION = 0.05
UNKNOWN_DOMAIN_WARN_FRACTION = 0.50


@dataclass
class SnapshotMetadata:
    schema_version: str
    created_at_utc: str
    source_files: list[str]
    output_path: str
    row_counts: dict[str, int]
    deduplication_count: int
    domain_coverage: dict[str, int]
    topic_coverage: dict[str, int] = field(default_factory=dict)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge CSV files through schema standardization, deduplication, "
            "domain classification, feature engineering, and snapshot export."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing source CSV files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("Data/snapshots/engageiq_snapshot_v1.csv"),
        help="Output CSV snapshot path.",
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=None,
        help=(
            "Snapshot metadata JSON path "
            "(default: <output>.metadata.json)."
        ),
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search for CSV files recursively under --input-dir.",
    )
    return parser.parse_args()


def extract_domain(url: str | None) -> str | None:
    if not isinstance(url, str) or not url.strip():
        return None

    try:
        host = urlparse(url).netloc.lower()
        if host.startswith("www."):
            host = host[4:]
        return host or None
    except Exception:
        return None


def normalize_text(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        source: target
        for source, target in COLUMN_ALIASES.items()
        if source in df.columns and target not in df.columns
    }
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def standardize_schema(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    """Normalize column names and data types for a single source file."""
    df = rename_columns(df.copy())

    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            df[col] = None

    df["id"] = df["id"].astype(str)
    df["source"] = df["source"].fillna(source_name).astype(str).str.strip()
    df["title"] = df["title"].apply(normalize_text)
    df["text"] = df["text"].apply(normalize_text)
    df["url"] = df["url"].apply(normalize_text)

    df["score"] = pd.to_numeric(df["score"], errors="coerce").fillna(0).astype(int)
    df["num_comments"] = (
        pd.to_numeric(df["num_comments"], errors="coerce")
        .fillna(0)
        .astype(int)
    )

    df["created_utc"] = pd.to_datetime(
        df["created_utc"], errors="coerce", utc=True
    )

    df["domain"] = df["domain"].where(df["domain"].notna(), None)
    df["domain"] = df.apply(
        lambda row: row["domain"] if row["domain"] else extract_domain(row["url"]),
        axis=1,
    )

    return df[REQUIRED_COLUMNS]


def load_raw_csv_files(
    input_dir: Path,
    *,
    recursive: bool,
) -> tuple[list[Path], list[pd.DataFrame]]:
    pattern = "**/*.csv" if recursive else "*.csv"
    files = sorted(input_dir.glob(pattern))

    if not files:
        raise FileNotFoundError(f"No CSV files found in {input_dir}")

    frames: list[pd.DataFrame] = []
    for file_path in files:
        raw_df = pd.read_csv(file_path)
        frames.append(raw_df)

    return files, frames


def merge_datasets(standardized_frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not standardized_frames:
        raise ValueError("No standardized frames to merge.")

    return pd.concat(standardized_frames, ignore_index=True)


def deduplicate(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Remove duplicate records, keeping the highest-engagement row."""
    before_count = len(df)
    df = df.copy()

    df["dedup_key"] = df["source"].astype(str) + "::" + df["id"].astype(str)
    df = df.sort_values(by=["score", "num_comments"], ascending=False)
    df = df.drop_duplicates(subset=["dedup_key"], keep="first")

    df["url_key"] = df["url"].str.lower().str.strip()
    df = df.drop_duplicates(subset=["url_key"], keep="first")
    df = df.drop(columns=["dedup_key", "url_key"])

    duplicates_removed = before_count - len(df)
    return df.reset_index(drop=True), duplicates_removed


def classify_domain(domain: str | None) -> str:
    if not isinstance(domain, str) or not domain:
        return "unknown"

    domain = domain.lower()

    if domain in DOMAIN_CATEGORY_MAP:
        return DOMAIN_CATEGORY_MAP[domain]

    for known_domain, category in DOMAIN_CATEGORY_MAP.items():
        if domain.endswith(known_domain):
            return category

    return "other"


def classify_topic(title: str, text: str) -> str:
    content = f"{title} {text}".lower()
    scores = {}

    for topic, keywords in TOPIC_KEYWORDS.items():
        score = sum(1 for keyword in keywords if keyword in content)
        scores[topic] = score

    best_topic = max(scores, key=scores.get)
    if scores[best_topic] == 0:
        return "unknown"

    return best_topic


def classify_domains(df: pd.DataFrame) -> pd.DataFrame:
    """Assign domain and topic categories to each record."""
    df = df.copy()
    df["domain_category"] = df["domain"].apply(classify_domain)
    df["topic_category"] = df.apply(
        lambda row: classify_topic(row["title"], row["text"]),
        axis=1,
    )
    return df


def check_domain_coverage(df: pd.DataFrame) -> dict[str, int]:
    """
    Summarize domain_category distribution and warn on poor coverage.
    """
    if df.empty:
        raise ValueError("Cannot check domain coverage on an empty dataset.")

    coverage = (
        df["domain_category"]
        .value_counts(dropna=False)
        .astype(int)
        .to_dict()
    )
    total = len(df)

    unknown_count = coverage.get("unknown", 0)
    unknown_fraction = unknown_count / total
    if unknown_fraction >= UNKNOWN_DOMAIN_WARN_FRACTION:
        logger.warning(
            "%.1f%% of records have unknown domain_category (%d / %d).",
            unknown_fraction * 100,
            unknown_count,
            total,
        )

    known_categories = {
        category: count
        for category, count in coverage.items()
        if category not in {"unknown", "other"}
    }
    if known_categories:
        min_category = min(known_categories, key=known_categories.get)
        min_fraction = known_categories[min_category] / total
        if min_fraction < MIN_DOMAIN_COVERAGE_FRACTION:
            logger.warning(
                "Domain category %r has low coverage: %.1f%% (%d / %d).",
                min_category,
                min_fraction * 100,
                known_categories[min_category],
                total,
            )

    logger.info("Domain coverage: %s", coverage)
    return coverage


def engineer_features(df: pd.DataFrame, snapshot_utc: datetime) -> pd.DataFrame:
    """
    Generate required features:
      - title_length: content quality proxy
      - age_days: recency
      - engagement_score: popularity
      - source: data source identifier (preserved from raw)
      - domain: content domain (preserved from raw)
    """
    df = df.copy()

    df["title_length"] = df["title"].str.len().astype(int)

    age_delta = snapshot_utc - df["created_utc"]
    df["age_days"] = age_delta.dt.days.fillna(0).clip(lower=0).astype(int)

    df["engagement_score"] = (
        df["score"].astype(float) + 2.0 * df["num_comments"].astype(float)
    )

    df["created_utc"] = df["created_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    df["snapshot_utc"] = snapshot_utc.strftime("%Y-%m-%dT%H:%M:%SZ")

    return df


def validate_final(df: pd.DataFrame) -> None:
    missing = set(FINAL_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing final columns: {missing}")

    if df["id"].isna().any():
        raise ValueError("Some records have missing id.")

    if df["title"].str.strip().eq("").any():
        raise ValueError("Some records have empty title.")

    if df.duplicated(subset=["source", "id"]).any():
        raise ValueError("Duplicate source/id pairs found after deduplication.")


def write_snapshot(df: pd.DataFrame, output_path: Path) -> None:
    df = df[FINAL_COLUMNS]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def write_snapshot_metadata(metadata: SnapshotMetadata, metadata_path: Path) -> None:
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(asdict(metadata), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run_pipeline(
    input_dir: Path,
    output_path: Path,
    metadata_path: Path,
    *,
    recursive: bool,
) -> SnapshotMetadata:
    snapshot_utc = datetime.now(timezone.utc)

    logger.info("Stage 1/9: Loading raw CSV files from %s", input_dir)
    source_files, raw_frames = load_raw_csv_files(
        input_dir,
        recursive=recursive,
    )
    raw_row_count = sum(len(frame) for frame in raw_frames)
    source_file_names = [str(path.name) for path in source_files]
    logger.info(
        "Loaded %d file(s), %d raw row(s).",
        len(source_files),
        raw_row_count,
    )

    logger.info("Stage 2/9: Schema standardization")
    standardized_frames = [
        standardize_schema(raw_df, file_path.stem)
        for raw_df, file_path in zip(raw_frames, source_files)
    ]

    logger.info("Stage 3/9: Merge")
    merged_df = merge_datasets(standardized_frames)
    merged_row_count = len(merged_df)
    logger.info("Merged row count: %d", merged_row_count)

    logger.info("Stage 4/9: Deduplication")
    deduplicated_df, deduplication_count = deduplicate(merged_df)
    logger.info(
        "Rows after deduplication: %d (removed %d duplicate(s)).",
        len(deduplicated_df),
        deduplication_count,
    )

    logger.info("Stage 5/9: Domain classification")
    classified_df = classify_domains(deduplicated_df)

    logger.info("Stage 6/9: Domain coverage check")
    domain_coverage = check_domain_coverage(classified_df)
    topic_coverage = (
        classified_df["topic_category"]
        .value_counts(dropna=False)
        .astype(int)
        .to_dict()
    )

    logger.info("Stage 7/9: Feature engineering")
    featured_df = engineer_features(classified_df, snapshot_utc)

    validate_final(featured_df)

    logger.info("Stage 8/9: Offline snapshot export")
    write_snapshot(featured_df, output_path)

    metadata = SnapshotMetadata(
        schema_version=SCHEMA_VERSION,
        created_at_utc=snapshot_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        source_files=source_file_names,
        output_path=str(output_path),
        row_counts={
            "raw_total": raw_row_count,
            "merged": merged_row_count,
            "after_deduplication": len(deduplicated_df),
            "final": len(featured_df),
        },
        deduplication_count=deduplication_count,
        domain_coverage=domain_coverage,
        topic_coverage=topic_coverage,
    )

    logger.info("Stage 9/9: Snapshot metadata")
    write_snapshot_metadata(metadata, metadata_path)

    return metadata


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    args = parse_args()
    metadata_path = args.metadata_output or args.output.with_suffix(
        args.output.suffix + ".metadata.json"
    )

    metadata = run_pipeline(
        args.input_dir,
        args.output,
        metadata_path,
        recursive=args.recursive,
    )

    logger.info("Final snapshot written to: %s", args.output)
    logger.info("Snapshot metadata written to: %s", metadata_path)
    logger.info("Final rows: %d", metadata.row_counts["final"])
    logger.info("Schema version: %s", metadata.schema_version)
    logger.info("Duplicates removed: %d", metadata.deduplication_count)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
