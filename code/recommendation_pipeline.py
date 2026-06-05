#!/usr/bin/env python3
"""
End-to-end recommendation pipeline: FAISS retrieval → scoring → diversity re-ranking.

Example:
  python recommendation_pipeline.py \\
    --opportunities Embeddings/snapshots/finaldataset.csv \\
    --opportunity-embeddings Embeddings/embeddings/opportunity_embeddings.jsonl \\
    --user-embeddings Embeddings/embeddings/user_embeddings.jsonl \\
    --user-id user_123 \\
    --top-k 100 \\
    --rerank simple \\
    --output Embeddings/recommendations/user_123_recommendations.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
EMBEDDINGS_DIR = SCRIPT_DIR.parent

sys.path.insert(0, str(EMBEDDINGS_DIR))
sys.path.insert(0, str(EMBEDDINGS_DIR / "evaluation"))

import argparse
import json
import logging

import pandas as pd

from evaluation import (
    compare_ranking_models,
    format_markdown_report,
    load_interaction_labels,
)
from explainability import format_recommendation_record
from ranking import rank_and_rerank_candidates
from retrieve_candidates import run as retrieve_run

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Retrieve candidates with FAISS, score by engagement value, "
            "and apply diversity re-ranking."
        )
    )
    parser.add_argument(
        "--opportunities",
        type=Path,
        required=True,
        help="Opportunity CSV file or directory.",
    )
    parser.add_argument(
        "--opportunity-embeddings",
        type=Path,
        required=True,
        help="Opportunity embeddings JSONL.",
    )
    parser.add_argument(
        "--user-embeddings",
        type=Path,
        required=True,
        help="User embeddings JSONL.",
    )
    parser.add_argument(
        "--user-id",
        type=str,
        required=True,
        help="User id to recommend for.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=100,
        help="FAISS retrieval depth (default: 100).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Final ranked recommendations CSV.",
    )
    parser.add_argument(
        "--retrieval-output",
        type=Path,
        default=None,
        help="Optional path to save raw FAISS candidates before ranking.",
    )
    parser.add_argument(
        "--rerank",
        choices=("simple", "mmr", "none"),
        default="simple",
        help="Diversity re-ranking strategy (default: simple).",
    )
    parser.add_argument(
        "--recommendation-top-n",
        type=int,
        default=None,
        help="Optional cap on final recommendations after re-ranking.",
    )
    parser.add_argument(
        "--diversity-bonus",
        type=float,
        default=0.05,
        help="Domain diversity bonus (simple rerank).",
    )
    parser.add_argument(
        "--mmr-lambda",
        type=float,
        default=0.7,
        help="MMR lambda (mmr rerank).",
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=None,
        help="Optional JSON path for NDCG@10 metrics.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively load CSV files when --opportunities is a directory.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Optional JSON array of {title, score, why, url} recommendation cards.",
    )
    parser.add_argument(
        "--evaluate-labels",
        type=Path,
        default=None,
        help="Optional interaction labels for NDCG@10 evaluation after ranking.",
    )
    parser.add_argument(
        "--evaluation-report",
        type=Path,
        default=None,
        help="Markdown report path when --evaluate-labels is set.",
    )
    return parser.parse_args()


def write_dataframe(df: pd.DataFrame, path: Path) -> Path:
    if path.suffix.lower() != ".csv":
        path = path.with_suffix(".csv")

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    logger.info("Stage 1/4: FAISS candidate retrieval")
    candidates = retrieve_run(
        args.opportunities,
        args.opportunity_embeddings,
        args.user_embeddings,
        args.user_id,
        top_k=args.top_k,
        output_path=args.retrieval_output,
        recursive=args.recursive,
    )

    logger.info("Stage 2/4: Engagement-value scoring")
    logger.info("Stage 3/4: Diversity re-ranking (%s)", args.rerank)
    ranked, metrics = rank_and_rerank_candidates(
        candidates,
        rerank_method=args.rerank,
        diversity_bonus=args.diversity_bonus,
        mmr_lambda=args.mmr_lambda,
        top_n=args.recommendation_top_n,
    )

    logger.info("Stage 4/4: Explainability (why) + optional evaluation")
    output_path = write_dataframe(ranked, args.output)
    logger.info("Wrote %d recommendation(s) to %s", len(ranked), output_path)

    logger.info(
        "NDCG@10 (post re-rank): %.4f | NDCG@10 (pre re-rank): %.4f",
        metrics["ndcg_at_10"],
        metrics["ndcg_at_10_pre_rerank"],
    )

    if args.metrics_output is not None:
        payload = {
            "user_id": args.user_id,
            "top_k": args.top_k,
            "rerank_method": args.rerank,
            **metrics,
        }
        args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info("Wrote metrics to %s", args.metrics_output)

    if args.json_output is not None:
        cards = [
            format_recommendation_record(row)
            for _, row in ranked.iterrows()
        ]
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(cards, indent=2), encoding="utf-8")
        logger.info("Wrote %d recommendation card(s) to %s", len(cards), args.json_output)

    if args.evaluate_labels is not None:
        labels = load_interaction_labels(args.evaluate_labels)
        report = compare_ranking_models(
            {args.user_id: candidates},
            labels,
        )
        logger.info("NDCG@10 (baseline): %.4f", report.baseline_ndcg_at_10)
        for result in report.model_results:
            logger.info("NDCG@10 (%s): %.4f", result.display_name, result.ndcg_at_10)
        if args.evaluation_report is not None:
            args.evaluation_report.parent.mkdir(parents=True, exist_ok=True)
            args.evaluation_report.write_text(
                format_markdown_report(report),
                encoding="utf-8",
            )
            logger.info("Wrote evaluation report to %s", args.evaluation_report)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
