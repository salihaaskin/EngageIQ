#!/usr/bin/env python3
"""
Score and re-rank FAISS-retrieved opportunity candidates.

Input: Parquet or CSV with retrieval fields (similarity or retrieval_distance)
and opportunity metadata (title, text/description, score, num_comments, etc.).

Example:
  python rank_candidates.py \\
    --input Data/retrieval/user_123_candidates.parquet \\
    --output Data/recommendations/user_123_ranked.parquet \\
    --rerank simple
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import json
import logging

import pandas as pd

from explainability import format_recommendation_record
from ranking import rank_and_rerank_candidates

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rank retrieved candidates by engagement value and apply "
            "diversity-aware re-ranking."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Retrieved candidates (Parquet or CSV).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Ranked output path (.parquet or .csv).",
    )
    parser.add_argument(
        "--rerank",
        choices=("simple", "mmr", "none"),
        default="simple",
        help="Diversity re-ranking strategy (default: simple).",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=None,
        help="Optional cap on number of recommendations to return.",
    )
    parser.add_argument(
        "--diversity-bonus",
        type=float,
        default=0.05,
        help="Bonus for first candidate per domain (simple rerank).",
    )
    parser.add_argument(
        "--mmr-lambda",
        type=float,
        default=0.7,
        help="MMR relevance vs diversity trade-off (mmr rerank).",
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=None,
        help="Optional JSON file for ranking evaluation metrics.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Optional JSON array of {title, score, why, url} recommendation cards.",
    )
    return parser.parse_args()


def load_candidates(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported input format {suffix!r}; use .parquet or .csv.")


def write_output(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df.to_csv(path, index=False)
    elif suffix in {".parquet", ".pq"}:
        df.to_parquet(path, index=False)
    else:
        raise ValueError(f"Unsupported output format {suffix!r}; use .parquet or .csv.")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    logger.info("Loading candidates from %s", args.input)
    candidates = load_candidates(args.input)
    logger.info("Loaded %d candidate(s)", len(candidates))

    ranked, metrics = rank_and_rerank_candidates(
        candidates,
        rerank_method=args.rerank,
        diversity_bonus=args.diversity_bonus,
        mmr_lambda=args.mmr_lambda,
        top_n=args.top_n,
    )

    write_output(ranked, args.output)
    logger.info("Wrote %d ranked recommendation(s) to %s", len(ranked), args.output)

    logger.info(
        "NDCG@10 (post re-rank): %.4f | NDCG@10 (pre re-rank): %.4f",
        metrics["ndcg_at_10"],
        metrics["ndcg_at_10_pre_rerank"],
    )

    if args.metrics_output is not None:
        args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        logger.info("Wrote metrics to %s", args.metrics_output)

    if args.json_output is not None:
        cards = [
            format_recommendation_record(row)
            for _, row in ranked.iterrows()
        ]
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(cards, indent=2), encoding="utf-8")
        logger.info("Wrote %d recommendation card(s) to %s", len(cards), args.json_output)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
