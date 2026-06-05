#!/usr/bin/env python3
"""
Build interaction-based relevance labels for offline ranking evaluation.

Labels are inferred from user profile preferences matched against retrieved
candidates (proxy for logged interactions when clickstream data is unavailable).

Example:
  python generate_evaluation_labels.py \\
    --user-profiles ../Data/examples/user_profiles.sample.json \\
    --opportunities snapshots/v1.parquet \\
    --opportunity-embeddings embeddings/opportunity_embeddings.jsonl \\
    --user-embeddings embeddings/user_embeddings.jsonl \\
    --top-k 100 \\
    --output ../Data/examples/ranking_evaluation_labels.parquet
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import json
import logging
import re

import pandas as pd

from embedding_common import normalize_text
from evaluation import interaction_to_relevance, normalize_labels_frame
from generate_embeddings import build_opportunity_id
from retrieve_candidates import load_embeddings_jsonl, run as retrieve_run
from user_profile_embeddings import load_user_profiles

logger = logging.getLogger(__name__)

PREFERENCE_ALIASES: dict[str, list[str]] = {
    "artificial_intelligence": ["ai", "artificial intelligence", "llm", "openai"],
    "devops": ["devops", "kubernetes", "k8s", "cloud native", "terraform"],
    "machine_learning": ["machine learning", "ml", "deep learning", "rag"],
    "frontend_development": ["frontend", "react", "typescript", "css", "web"],
    "developer_tools": ["developer tools", "ide", "cli", "tooling"],
    "cloud_computing": ["cloud", "aws", "azure", "gcp"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate ranking evaluation labels.")
    parser.add_argument("--user-profiles", type=Path, required=True)
    parser.add_argument("--opportunities", type=Path, required=True)
    parser.add_argument("--opportunity-embeddings", type=Path, required=True)
    parser.add_argument("--user-embeddings", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--recursive", action="store_true")
    return parser.parse_args()


def _normalize_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", normalize_text(value).lower()).strip()


def _preference_terms(preferred_domains: list[str], interests: list[str]) -> set[str]:
    terms: set[str] = set()
    for domain in preferred_domains:
        key = _normalize_token(domain).replace(" ", "_")
        terms.add(key)
        for alias in PREFERENCE_ALIASES.get(key, []):
            terms.add(_normalize_token(alias))
    for interest in interests:
        terms.add(_normalize_token(interest))
    return {term for term in terms if term}


def _opportunity_terms(row: pd.Series) -> set[str]:
    parts: list[str] = []
    for column in ("topic_category", "domain_category", "domain", "title", "text"):
        value = normalize_text(row.get(column, ""))
        if value:
            parts.append(_normalize_token(value))
    combined = " ".join(parts)
    return {token for token in combined.split() if token}


def _match_strength(opp_terms: set[str], user_terms: set[str]) -> float:
    if not opp_terms or not user_terms:
        return 0.0
    overlap = len(opp_terms & user_terms)
    if overlap == 0:
        return 0.0
    return overlap / max(len(user_terms), 1)


def _combined_relevance_signal(strength: float, similarity: float) -> float:
    return 0.65 * strength + 0.35 * similarity


def _interaction_from_percentile(percentile: float) -> str:
    if percentile >= 0.90:
        return "engaged"
    if percentile >= 0.70:
        return "bookmarked"
    if percentile >= 0.40:
        return "clicked"
    return "skipped"


def build_labels_for_user(
    candidates: pd.DataFrame,
    user_id: str,
    preferred_domains: list[str],
    interests: list[str],
) -> pd.DataFrame:
    user_terms = _preference_terms(preferred_domains, interests)
    scored_rows: list[tuple[str, float]] = []

    for _, row in candidates.iterrows():
        opportunity_id = build_opportunity_id(row)
        strength = _match_strength(_opportunity_terms(row), user_terms)
        similarity = float(row.get("similarity", 0.0))
        signal = _combined_relevance_signal(strength, similarity)
        scored_rows.append((opportunity_id, signal))

    if not scored_rows:
        return pd.DataFrame(
            columns=["user_id", "opportunity_id", "interaction", "relevance_label"]
        )

    scored_rows.sort(key=lambda item: item[1], reverse=True)
    n = len(scored_rows)
    rows: list[dict[str, object]] = []

    for rank, (opportunity_id, _) in enumerate(scored_rows):
        percentile = 1.0 - (rank / max(n - 1, 1))
        interaction = _interaction_from_percentile(percentile)
        rows.append(
            {
                "user_id": user_id,
                "opportunity_id": opportunity_id,
                "interaction": interaction,
                "relevance_label": interaction_to_relevance(interaction),
            }
        )

    return pd.DataFrame(rows)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    profiles = {profile["user_id"]: profile for profile in load_user_profiles(args.user_profiles)}
    embedded_users = {
        normalize_text(record.get("user_id", ""))
        for record in load_embeddings_jsonl(args.user_embeddings)
    }
    all_labels: list[pd.DataFrame] = []

    for user_id, profile in profiles.items():
        if user_id not in embedded_users:
            logger.warning("Skipping %s (no user embedding found).", user_id)
            continue
        logger.info("Retrieving top-%d candidates for %s", args.top_k, user_id)
        candidates = retrieve_run(
            args.opportunities,
            args.opportunity_embeddings,
            args.user_embeddings,
            user_id,
            top_k=args.top_k,
            output_path=None,
            recursive=args.recursive,
        )
        labels = build_labels_for_user(
            candidates,
            user_id,
            profile.get("preferred_domains", []),
            profile.get("interests", []),
        )
        logger.info(
            "User %s label distribution: %s",
            user_id,
            labels["interaction"].value_counts().to_dict(),
        )
        all_labels.append(labels)

    combined = normalize_labels_frame(pd.concat(all_labels, ignore_index=True))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    suffix = args.output.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        combined.to_parquet(args.output, index=False)
    elif suffix == ".csv":
        combined.to_csv(args.output, index=False)
    elif suffix == ".jsonl":
        with args.output.open("w", encoding="utf-8") as handle:
            for record in combined.to_dict(orient="records"):
                handle.write(json.dumps(record) + "\n")
    else:
        raise ValueError("Output must be .parquet, .csv, or .jsonl.")

    logger.info("Wrote %d label(s) to %s", len(combined), args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
