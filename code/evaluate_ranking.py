#!/usr/bin/env python3
"""
Evaluate ranking quality with NDCG@10 across model variants.

Example:
  python evaluate_ranking.py \\
    --labels ../Data/examples/ranking_evaluation_labels.parquet \\
    --opportunities snapshots/v1.parquet \\
    --opportunity-embeddings embeddings/opportunity_embeddings.jsonl \\
    --user-embeddings embeddings/user_embeddings.jsonl \\
    --top-k 100 \\
    --report evaluation/RANKING_EVALUATION.md \\
    --results-json evaluation/evaluation_results.json
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import json
import logging

import pandas as pd

from evaluation import (
    build_evaluation_export,
    compare_ranking_models,
    format_markdown_report,
    load_interaction_labels,
)
from embedding_common import normalize_text
from retrieve_candidates import load_embeddings_jsonl, run as retrieve_run

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute NDCG@10 and compare ranking model variants."
    )
    parser.add_argument("--labels", type=Path, required=True, help="Interaction labels file.")
    parser.add_argument("--opportunities", type=Path, required=True)
    parser.add_argument("--opportunity-embeddings", type=Path, required=True)
    parser.add_argument("--user-embeddings", type=Path, required=True)
    parser.add_argument(
        "--top-k",
        type=int,
        default=100,
        help="FAISS retrieval depth per user (default: 100).",
    )
    parser.add_argument(
        "--ndcg-k",
        type=int,
        default=10,
        help="NDCG cutoff (default: 10).",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("evaluation/RANKING_EVALUATION.md"),
        help="Markdown report output path.",
    )
    parser.add_argument(
        "--results-json",
        type=Path,
        default=Path("evaluation/evaluation_results.json"),
        help="JSON metrics output path.",
    )
    parser.add_argument(
        "--export-dataset",
        type=Path,
        default=None,
        help="Optional Parquet export: user_id, opportunity_id, predicted_score, relevance_label.",
    )
    parser.add_argument("--recursive", action="store_true")
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    labels = load_interaction_labels(args.labels)
    from retrieve_candidates import load_embeddings_jsonl

    embedded_users = {
        normalize_text(record.get("user_id", ""))
        for record in load_embeddings_jsonl(args.user_embeddings)
    }
    user_ids = sorted(
        uid for uid in labels["user_id"].unique() if uid in embedded_users
    )
    if not user_ids:
        raise ValueError("No labeled users have embeddings for retrieval.")
    logger.info("Evaluating %d user(s): %s", len(user_ids), ", ".join(user_ids))

    candidates_by_user: dict[str, pd.DataFrame] = {}
    for user_id in user_ids:
        logger.info("Retrieving top-%d candidates for %s", args.top_k, user_id)
        candidates_by_user[user_id] = retrieve_run(
            args.opportunities,
            args.opportunity_embeddings,
            args.user_embeddings,
            user_id,
            top_k=args.top_k,
            output_path=None,
            recursive=args.recursive,
        )

    report = compare_ranking_models(
        candidates_by_user,
        labels,
        k=args.ndcg_k,
    )

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(format_markdown_report(report), encoding="utf-8")
    logger.info("Wrote report to %s", args.report)

    args.results_json.parent.mkdir(parents=True, exist_ok=True)
    args.results_json.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    logger.info("Wrote JSON results to %s", args.results_json)

    logger.info("Baseline NDCG@10 = %.4f", report.baseline_ndcg_at_10)
    for result in report.model_results:
        logger.info("%s: NDCG@10 = %.4f", result.display_name, result.ndcg_at_10)

    if args.export_dataset is not None:
        export = build_evaluation_export(candidates_by_user, labels)
        args.export_dataset.parent.mkdir(parents=True, exist_ok=True)
        export.to_parquet(args.export_dataset, index=False)
        logger.info("Wrote evaluation dataset (%d rows) to %s", len(export), args.export_dataset)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
