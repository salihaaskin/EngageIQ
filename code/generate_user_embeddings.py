#!/usr/bin/env python3
"""
Generate dense SBERT embeddings for user interest profiles.

Combines interests, skills, domains, and optional engagement history into a
single text profile, encodes with the same model and preprocessing as
opportunity embeddings, and writes JSONL records:

  {"user_id": "...", "profile_text": "...", "embedding": [...]}

Example:
  python generate_user_embeddings.py \\
    --users Data/examples/user_profiles.sample.json \\
    --output-dir Data/embeddings
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from embedding_common import (
    DEFAULT_MODEL,
    encode_texts,
    embedding_dimension,
    load_sentence_transformer,
)
from user_profile_embeddings import (
    build_user_profile_texts,
    build_user_records,
    load_user_profiles,
)

logger = logging.getLogger(__name__)


@dataclass
class UserEmbeddingMetadata:
    schema_version: str
    created_at_utc: str
    model_name: str
    embedding_dimension: int
    user_count: int
    users_path: str
    user_embeddings_path: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate user profile embeddings with Sentence-BERT for semantic "
            "matching against opportunity vectors."
        )
    )
    parser.add_argument(
        "--users",
        type=Path,
        required=True,
        help="JSON file: array of user profiles or { \"users\": [...] }.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Data/embeddings"),
        help="Directory for user_embeddings.jsonl and metadata.",
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
    return parser.parse_args()


def write_jsonl(records: Iterable[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def run(
    users_path: Path,
    output_dir: Path,
    *,
    model_name: str,
    batch_size: int,
    device: str | None,
) -> UserEmbeddingMetadata:
    created_at = datetime.now(timezone.utc)

    logger.info("Loading user profiles from %s", users_path)
    profiles = load_user_profiles(users_path)

    profile_texts: list[str] = []
    encoding_texts: list[str] = []
    for profile in profiles:
        profile_text, encoding_text = build_user_profile_texts(profile)
        profile_texts.append(profile_text)
        encoding_texts.append(encoding_text)

    logger.info("Loading Sentence-BERT model: %s", model_name)
    model = load_sentence_transformer(model_name, device)
    dimension = embedding_dimension(model)

    logger.info("Encoding %d user profile(s) (batch_size=%d)", len(profiles), batch_size)
    embeddings = encode_texts(model, encoding_texts, batch_size=batch_size)
    user_records = build_user_records(profiles, profile_texts, embeddings)

    output_path = output_dir / "user_embeddings.jsonl"
    write_jsonl(user_records, output_path)
    logger.info("Wrote %d user embeddings to %s", len(user_records), output_path)

    metadata = UserEmbeddingMetadata(
        schema_version="1.0.0",
        created_at_utc=created_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
        model_name=model_name,
        embedding_dimension=dimension,
        user_count=len(user_records),
        users_path=str(users_path),
        user_embeddings_path=str(output_path),
    )

    metadata_path = output_dir / "user_embeddings.metadata.json"
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
        args.users,
        args.output_dir,
        model_name=args.model,
        batch_size=args.batch_size,
        device=args.device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
