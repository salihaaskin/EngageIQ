#!/usr/bin/env python3
"""
Generate dense SBERT embeddings for opportunities (CSV) and user interest profiles (JSON).

Opportunity text template:
  Title: {title}
  Description: {description}
  Domain: {domain}
  Source: {source}

User profiles use the same model and preprocessing; see generate_user_embeddings.py.

Example:
  python generate_embeddings.py \\
    --opportunities Data/snapshots/engageiq_snapshot_v1.csv \\
    --users Data/examples/user_profiles.sample.json \\
    --output-dir Data/embeddings
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Iterable

import pandas as pd

from embedding_common import (
    DEFAULT_MODEL,
    encode_texts,
    embedding_dimension,
    load_sentence_transformer,
    normalize_text,
)
from user_profile_embeddings import (
    build_user_profile_texts,
    build_user_records,
    load_user_profiles,
)

logger = logging.getLogger(__name__)

OPPORTUNITY_COLUMN_ALIASES = {
    "description": "text",
    "body": "text",
    "engagement": "score",
    "upvotes": "score",
    "points": "score",
    "comments": "num_comments",
    "comment_count": "num_comments",
    "created_at": "created_utc",
    "created_at_utc": "created_utc",
}

REQUIRED_OPPORTUNITY_COLUMNS = ("source", "title")


@dataclass
class EmbeddingMetadata:
    schema_version: str
    created_at_utc: str
    model_name: str
    embedding_dimension: int
    opportunity_count: int
    user_count: int
    opportunities_path: str
    users_path: str | None
    opportunity_embeddings_path: str
    user_embeddings_path: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build semantic embeddings for opportunities (CSV) and "
            "user interest profiles (JSON) using Sentence-BERT."
        )
    )
    parser.add_argument(
        "--opportunities",
        type=Path,
        required=True,
        help="CSV file or directory of CSV files (opportunity dataset).",
    )
    parser.add_argument(
        "--users",
        type=Path,
        default=None,
        help="JSON file: array of user profiles or { \"users\": [...] }.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Data/embeddings"),
        help="Directory for embedding JSONL and metadata.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Sentence-Transformers model id (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Encoding batch size.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device (e.g. cpu, cuda). Default: model default.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="If --opportunities is a directory, load CSV files recursively.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of opportunities to embed (for testing).",
    )
    return parser.parse_args()


def rename_opportunity_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        source: target
        for source, target in OPPORTUNITY_COLUMN_ALIASES.items()
        if source in df.columns and target not in df.columns
    }
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def resolve_embedding_domain(row: pd.Series) -> str:
    """Pick the most informative domain label for embedding text."""
    for column in ("topic_category", "domain_category", "domain"):
        if column not in row.index:
            continue
        value = normalize_text(row[column])
        if not value or value.lower() in {"unknown", "other", "nan"}:
            continue
        return value
    return "unknown"


def build_opportunity_id(row: pd.Series) -> str:
    source = normalize_text(row.get("source", "")).lower()
    record_id = normalize_text(row.get("id", ""))
    if source and record_id:
        return f"{source}::{record_id}"

    url = normalize_text(row.get("url", ""))
    if url:
        digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
        return f"{source or 'unknown'}::{digest}"

    title = normalize_text(row.get("title", ""))
    digest = hashlib.sha256(f"{source}|{title}".encode("utf-8")).hexdigest()[:16]
    return f"{source or 'unknown'}::{digest}"


def format_opportunity_text(row: pd.Series) -> str:
    title = normalize_text(row.get("title", ""))
    description = normalize_text(
        row.get("text", row.get("description", ""))
    )
    domain = resolve_embedding_domain(row)
    source = normalize_text(row.get("source", ""))

    return (
        f"Title: {title}\n"
        f"Description: {description}\n"
        f"Domain: {domain}\n"
        f"Source: {source}"
    )


def load_opportunities_csv(
    path: Path,
    *,
    recursive: bool,
) -> pd.DataFrame:
    if path.is_dir():
        pattern = "**/*.csv" if recursive else "*.csv"
        files = sorted(path.glob(pattern))
        if not files:
            raise FileNotFoundError(f"No CSV files found in {path}")
        frames = [pd.read_csv(file_path) for file_path in files]
        df = pd.concat(frames, ignore_index=True)
    else:
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_csv(path)

    df = rename_opportunity_columns(df)
    missing = [col for col in REQUIRED_OPPORTUNITY_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Opportunity dataset missing required columns: {missing}")

    if "id" not in df.columns:
        df["id"] = None

    if "text" not in df.columns and "description" in df.columns:
        df["text"] = df["description"]

    if "text" not in df.columns:
        df["text"] = ""

    return df


def load_opportunities_parquet(
    path: Path,
    *,
    recursive: bool,
) -> pd.DataFrame:
    """Backward-compatible alias for callers that still import the old name."""
    return load_opportunities_csv(path, recursive=recursive)


def write_jsonl(records: Iterable[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_opportunity_records(
    df: pd.DataFrame,
    embeddings: list[list[float]],
) -> list[dict[str, Any]]:
    if len(df) != len(embeddings):
        raise ValueError(
            f"Row count ({len(df)}) does not match embedding count ({len(embeddings)})."
        )

    records: list[dict[str, Any]] = []
    for row_index, (_, row) in enumerate(df.iterrows()):
        domain = resolve_embedding_domain(row)
        records.append(
            {
                "opportunity_id": build_opportunity_id(row),
                "embedding": embeddings[row_index],
                "source": normalize_text(row.get("source", "")).lower(),
                "domain": domain,
            }
        )
    return records


def run(
    opportunities_path: Path,
    users_path: Path | None,
    output_dir: Path,
    *,
    model_name: str,
    batch_size: int,
    device: str | None,
    recursive: bool,
    limit: int | None,
) -> EmbeddingMetadata:
    created_at = datetime.now(timezone.utc)

    logger.info("Loading opportunities from %s", opportunities_path)
    df = load_opportunities_csv(opportunities_path, recursive=recursive)
    if limit is not None:
        df = df.head(limit).reset_index(drop=True)
    logger.info("Opportunities to embed: %d", len(df))

    opportunity_texts = [format_opportunity_text(row) for _, row in df.iterrows()]

    logger.info("Loading Sentence-BERT model: %s", model_name)
    model = load_sentence_transformer(model_name, device)
    dimension = embedding_dimension(model)

    logger.info("Encoding opportunity texts (batch_size=%d)", batch_size)
    opportunity_embeddings = encode_texts(
        model, opportunity_texts, batch_size=batch_size
    )
    opportunity_records = build_opportunity_records(df, opportunity_embeddings)

    opportunity_output = output_dir / "opportunity_embeddings.jsonl"
    write_jsonl(opportunity_records, opportunity_output)
    logger.info(
        "Wrote %d opportunity embeddings to %s",
        len(opportunity_records),
        opportunity_output,
    )

    user_output: Path | None = None
    user_count = 0
    if users_path is not None:
        logger.info("Loading user profiles from %s", users_path)
        profiles = load_user_profiles(users_path)

        profile_texts: list[str] = []
        encoding_texts: list[str] = []
        for profile in profiles:
            profile_text, encoding_text = build_user_profile_texts(profile)
            profile_texts.append(profile_text)
            encoding_texts.append(encoding_text)

        logger.info("Encoding %d user profile(s)", len(profiles))
        user_embeddings = encode_texts(model, encoding_texts, batch_size=batch_size)
        user_records = build_user_records(profiles, profile_texts, user_embeddings)
        user_output = output_dir / "user_embeddings.jsonl"
        write_jsonl(user_records, user_output)
        user_count = len(user_records)
        logger.info("Wrote %d user embeddings to %s", user_count, user_output)

    metadata = EmbeddingMetadata(
        schema_version="1.0.0",
        created_at_utc=created_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
        model_name=model_name,
        embedding_dimension=dimension,
        opportunity_count=len(opportunity_records),
        user_count=user_count,
        opportunities_path=str(opportunities_path),
        users_path=str(users_path) if users_path else None,
        opportunity_embeddings_path=str(opportunity_output),
        user_embeddings_path=str(user_output) if user_output else None,
    )

    metadata_path = output_dir / "embeddings.metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(asdict(metadata), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    logger.info("Metadata written to %s", metadata_path)

    return metadata


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    args = parse_args()
    run(
        args.opportunities,
        args.users,
        args.output_dir,
        model_name=args.model,
        batch_size=args.batch_size,
        device=args.device,
        recursive=args.recursive,
        limit=args.limit,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
