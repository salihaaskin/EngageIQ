#!/usr/bin/env python3
"""
Adaptive feedback learning simulation for EngageIQ recommendations.

Runs repeated simulated feedback rounds (engage, bookmark, skip), updates the
user profile embedding, re-runs FAISS retrieval and ranking after each update,
and reports NDCG@10 improvement against a held-out simulated preference vector.

Example:
  python adaptive_learning.py \
    --opportunities snapshots/finaldataset.csv \
    --opportunity-embeddings embeddings/opportunity_embeddings.jsonl \
    --user-embeddings embeddings/user_embeddings.jsonl \
    --user-id user-alice \
    --rounds 50 \
    --output recommendations/user-alice_feedback_learning.json
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

from embedding_common import normalize_text
from generate_embeddings import load_opportunities_csv
from ranking import rank_and_rerank_candidates
from retrieve_candidates import (
    build_faiss_index,
    load_embeddings_jsonl,
    load_opportunity_embedding_matrix,
    load_user_embedding,
    search_index,
)

logger = logging.getLogger(__name__)

FeedbackType = Literal["engage", "bookmark", "skip"]

FEEDBACK_RELEVANCE: dict[FeedbackType, int] = {
    "engage": 3,
    "bookmark": 2,
    "skip": 0,
}

UPDATE_WEIGHTS: dict[FeedbackType, tuple[float, float]] = {
    "engage": (0.90, 0.10),
    "bookmark": (0.93, 0.07),
    "skip": (0.95, -0.05),
}


@dataclass
class FeedbackEvent:
    round: int
    feedback: FeedbackType
    opportunity_id: str
    relevance: int
    ndcg_at_10: float


@dataclass
class LearningResult:
    user_id: str
    initial_ndcg_at_10: float
    final_ndcg_at_10: float
    improvement_percent: float
    feedback_rounds: int
    learning_demonstrated: bool
    feedback_counts: dict[str, int]
    history: list[dict[str, Any]]


class NumpyL2Index:
    """Small exact-search fallback matching the FAISS IndexFlatL2 search API."""

    def __init__(self, vectors: np.ndarray):
        if vectors.ndim != 2:
            raise ValueError("Opportunity embeddings must be a 2-D matrix.")
        self.vectors = np.ascontiguousarray(vectors.astype("float32"))

    def search(self, query: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        query = np.ascontiguousarray(query.astype("float32"))
        if query.ndim == 1:
            query = query.reshape(1, -1)
        diff = self.vectors[None, :, :] - query[:, None, :]
        distances = np.sum(diff * diff, axis=2)
        k = min(top_k, self.vectors.shape[0])
        indices = np.argsort(distances, axis=1)[:, :k]
        ranked_distances = np.take_along_axis(distances, indices, axis=1)
        return ranked_distances.astype("float32"), indices.astype("int64")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulate feedback-based learning for EngageIQ recommendations."
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
    parser.add_argument("--user-id", type=str, required=True, help="User id to adapt.")
    parser.add_argument(
        "--rounds",
        type=int,
        default=50,
        help="Number of simulated feedback rounds (minimum: 50, default: 50).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=100,
        help="FAISS retrieval depth after each feedback round.",
    )
    parser.add_argument(
        "--recommendation-top-n",
        type=int,
        default=25,
        help="Final recommendation list size after re-ranking.",
    )
    parser.add_argument(
        "--rerank",
        choices=("simple", "mmr", "none"),
        default="simple",
        help="Diversity re-ranking strategy.",
    )
    parser.add_argument(
        "--diversity-bonus",
        type=float,
        default=0.05,
        help="Domain diversity bonus for simple re-ranking.",
    )
    parser.add_argument(
        "--mmr-lambda",
        type=float,
        default=0.7,
        help="MMR lambda for MMR diversity re-ranking.",
    )
    parser.add_argument(
        "--exploration-rate",
        type=float,
        default=0.20,
        help="Probability of sampling feedback beyond the top result.",
    )
    parser.add_argument(
        "--negative-feedback-rate",
        type=float,
        default=0.10,
        help="Probability of skipping the weakest visible recommendation.",
    )
    parser.add_argument(
        "--preference-shift",
        type=float,
        default=0.75,
        help=(
            "Blend toward a nearby hidden preference centroid for simulated labels; "
            "larger values create more room for learning."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible feedback simulation.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path for summary and per-round history.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively load CSV files when --opportunities is a directory.",
    )
    return parser.parse_args()


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        raise ValueError("Cannot normalize a zero-length embedding vector.")
    return (vector / norm).astype("float32")


def build_opportunity_lookup(
    opportunities: pd.DataFrame,
    opportunity_embeddings: np.ndarray,
    opportunity_ids: list[str],
) -> tuple[pd.DataFrame, dict[str, int]]:
    if len(opportunity_ids) != opportunity_embeddings.shape[0]:
        raise ValueError("opportunity_ids length must match embedding row count.")

    if "opportunity_id" in opportunities.columns:
        opportunities_with_ids = opportunities.copy()
        opportunities_with_ids["opportunity_id"] = opportunities_with_ids["opportunity_id"].map(normalize_text)
    else:
        from retrieve_candidates import attach_opportunity_ids

        opportunities_with_ids = attach_opportunity_ids(opportunities)
    by_id = opportunities_with_ids.drop_duplicates(subset=["opportunity_id"]).set_index(
        "opportunity_id",
        drop=False,
    )
    embedding_row_by_id = {opportunity_id: index for index, opportunity_id in enumerate(opportunity_ids)}
    return by_id, embedding_row_by_id


def load_opportunities_for_learning(
    opportunities_path: Path,
    opportunity_embeddings_path: Path,
    *,
    recursive: bool,
) -> pd.DataFrame:
    try:
        return load_opportunities_csv(opportunities_path, recursive=recursive)
    except ImportError as exc:
        logger.warning(
            "Could not read CSV opportunities (%s); using embedding metadata fallback.",
            exc,
        )

    records = load_embeddings_jsonl(opportunity_embeddings_path)
    rows: list[dict[str, Any]] = []
    for record in records:
        opportunity_id = normalize_text(record.get("opportunity_id", ""))
        if not opportunity_id:
            continue
        rows.append(
            {
                "opportunity_id": opportunity_id,
                "source": normalize_text(record.get("source", "")),
                "domain": normalize_text(record.get("domain", "unknown")) or "unknown",
                "score": 0,
                "num_comments": 0,
                "text": "",
            }
        )
    if not rows:
        raise ValueError(
            "Embedding metadata fallback could not build opportunity rows."
        )
    return pd.DataFrame(rows)


def build_feedback_index(opportunity_embeddings: np.ndarray):
    try:
        return build_faiss_index(opportunity_embeddings)
    except ImportError as exc:
        logger.warning(
            "FAISS is unavailable (%s); using exact NumPy L2 retrieval fallback.",
            exc,
        )
        return NumpyL2Index(opportunity_embeddings)


def retrieve_candidates_from_index(
    index,
    user_vector: np.ndarray,
    opportunities_by_id: pd.DataFrame,
    opportunity_ids: list[str],
    *,
    top_k: int,
) -> pd.DataFrame:
    distances, indices = search_index(index, user_vector, top_k=top_k)
    rows: list[pd.Series] = []

    for rank, row_index in enumerate(indices[0], start=1):
        if row_index < 0:
            continue
        opportunity_id = opportunity_ids[int(row_index)]
        if opportunity_id not in opportunities_by_id.index:
            continue
        row = opportunities_by_id.loc[opportunity_id].copy()
        distance = float(distances[0][rank - 1])
        row["retrieval_rank"] = rank
        row["retrieval_distance"] = distance
        row["similarity"] = 1.0 / (1.0 + max(distance, 0.0))
        rows.append(row)

    if not rows:
        raise ValueError("No retrieved candidates matched opportunity rows.")

    return pd.DataFrame(rows).reset_index(drop=True)


def run_ranking_pipeline(
    index,
    user_vector: np.ndarray,
    opportunities_by_id: pd.DataFrame,
    opportunity_ids: list[str],
    *,
    top_k: int,
    rerank_method: str,
    diversity_bonus: float,
    mmr_lambda: float,
    recommendation_top_n: int | None,
) -> pd.DataFrame:
    candidates = retrieve_candidates_from_index(
        index,
        user_vector,
        opportunities_by_id,
        opportunity_ids,
        top_k=top_k,
    )
    ranked, _ = rank_and_rerank_candidates(
        candidates,
        rerank_method=rerank_method,
        diversity_bonus=diversity_bonus,
        mmr_lambda=mmr_lambda,
        top_n=recommendation_top_n,
    )
    return ranked


def build_simulated_preference_vector(
    user_vector: np.ndarray,
    opportunity_embeddings: np.ndarray,
    *,
    preference_shift: float,
    seed: int,
) -> np.ndarray:
    """
    Create a reproducible hidden preference vector near the user profile.

    The centroid comes from items that are reasonably close to the current user
    while still representing a discoverable preference direction. This makes the
    simulation realistic: feedback can reveal a stronger sub-interest without
    making the user profile unrelated to its starting point.
    """
    rng = np.random.default_rng(seed)
    similarities = opportunity_embeddings @ user_vector
    ordered = np.argsort(similarities)[::-1]
    start = min(25, max(len(ordered) - 1, 0))
    stop = min(125, len(ordered))
    pool = ordered[start:stop]
    if len(pool) == 0:
        pool = ordered[: min(100, len(ordered))]

    sample_size = min(25, len(pool))
    selected = rng.choice(pool, size=sample_size, replace=False)
    centroid = normalize_vector(opportunity_embeddings[selected].mean(axis=0))
    blended = (1.0 - preference_shift) * user_vector + preference_shift * centroid
    return normalize_vector(blended)


def build_simulated_relevance_lookup(
    opportunity_embeddings: np.ndarray,
    opportunity_ids: list[str],
    preference_vector: np.ndarray,
) -> dict[str, int]:
    similarities = opportunity_embeddings @ preference_vector
    engage_cutoff = float(np.quantile(similarities, 0.95))
    bookmark_cutoff = float(np.quantile(similarities, 0.85))
    weak_cutoff = float(np.quantile(similarities, 0.70))

    relevance_by_id: dict[str, int] = {}
    for opportunity_id, similarity in zip(opportunity_ids, similarities):
        if similarity >= engage_cutoff:
            relevance = 3
        elif similarity >= bookmark_cutoff:
            relevance = 2
        elif similarity >= weak_cutoff:
            relevance = 1
        else:
            relevance = 0
        relevance_by_id[opportunity_id] = relevance
    return relevance_by_id


def attach_simulated_relevance(
    recommendations: pd.DataFrame,
    relevance_by_id: dict[str, int],
) -> pd.DataFrame:
    df = recommendations.copy()
    labels: list[int] = []
    for _, row in df.iterrows():
        opportunity_id = normalize_text(row.get("opportunity_id", ""))
        labels.append(relevance_by_id.get(opportunity_id, 0))
    df["simulated_relevance"] = labels
    return df


def evaluate_simulated_ndcg(recommendations: pd.DataFrame, *, k: int = 10) -> float:
    if recommendations.empty or recommendations["simulated_relevance"].max() == 0:
        return 0.0

    score_column = "reranked_score" if "reranked_score" in recommendations.columns else "final_score"
    ordered = recommendations.sort_values(score_column, ascending=False)
    relevance = ordered["simulated_relevance"].astype(float).to_numpy()
    k_eff = min(k, len(relevance))

    gains = np.power(2.0, relevance[:k_eff]) - 1.0
    discounts = 1.0 / np.log2(np.arange(2, k_eff + 2))
    dcg = float(np.sum(gains * discounts))

    ideal = np.sort(relevance)[::-1]
    ideal_gains = np.power(2.0, ideal[:k_eff]) - 1.0
    ideal_dcg = float(np.sum(ideal_gains * discounts))
    if ideal_dcg == 0.0:
        return 0.0
    return dcg / ideal_dcg


def feedback_type_from_relevance(relevance: int, rng: np.random.Generator) -> FeedbackType:
    if relevance >= 3:
        return "engage" if rng.random() < 0.80 else "bookmark"
    if relevance == 2:
        return "bookmark" if rng.random() < 0.75 else "engage"
    return "skip"


def simulate_feedback(
    recommendations: pd.DataFrame,
    *,
    rng: np.random.Generator,
    exploration_rate: float,
    negative_feedback_rate: float,
    seen_counts: dict[str, int],
) -> dict[str, Any]:
    if recommendations.empty:
        raise ValueError("Cannot simulate feedback without recommendations.")

    pool = recommendations.head(min(25, len(recommendations))).copy()
    if rng.random() < negative_feedback_rate and len(pool) > 1:
        row = pool.sort_values(
            ["simulated_relevance", "rank"],
            ascending=[True, False],
        ).iloc[0]
        return {
            "type": "skip",
            "opportunity_id": normalize_text(row["opportunity_id"]),
            "relevance": FEEDBACK_RELEVANCE["skip"],
            "simulated_relevance": int(row["simulated_relevance"]),
        }

    if rng.random() < exploration_rate and len(recommendations) > 1:
        gains = np.power(2.0, pool["simulated_relevance"].astype(float).to_numpy()) - 1.0
        repeat_penalty = np.array(
            [
                1.0 / (1.0 + seen_counts.get(normalize_text(opportunity_id), 0))
                for opportunity_id in pool["opportunity_id"]
            ],
            dtype=float,
        )
        weights = (gains + 0.25) * repeat_penalty
        weights = weights / weights.sum()
        selected_position = int(rng.choice(np.arange(len(pool)), p=weights))
        row = pool.iloc[selected_position]
    else:
        pool["_seen_count"] = pool["opportunity_id"].map(
            lambda opportunity_id: seen_counts.get(normalize_text(opportunity_id), 0)
        )
        row = pool.sort_values(
            ["simulated_relevance", "_seen_count", "rank"],
            ascending=[False, True, True],
        ).iloc[0]

    relevance = int(row["simulated_relevance"])
    feedback_type = feedback_type_from_relevance(relevance, rng)
    return {
        "type": feedback_type,
        "opportunity_id": normalize_text(row["opportunity_id"]),
        "relevance": FEEDBACK_RELEVANCE[feedback_type],
        "simulated_relevance": relevance,
    }


def update_user_vector(
    user_vector: np.ndarray,
    item_vector: np.ndarray,
    feedback_type: FeedbackType,
) -> np.ndarray:
    user_weight, item_weight = UPDATE_WEIGHTS[feedback_type]
    return normalize_vector(user_weight * user_vector + item_weight * item_vector)


def run_feedback_learning(
    *,
    opportunities_path: Path,
    opportunity_embeddings_path: Path,
    user_embeddings_path: Path,
    user_id: str,
    rounds: int,
    top_k: int,
    recommendation_top_n: int | None,
    rerank_method: str,
    diversity_bonus: float,
    mmr_lambda: float,
    exploration_rate: float,
    negative_feedback_rate: float,
    preference_shift: float,
    seed: int,
    recursive: bool,
) -> LearningResult:
    if rounds < 50:
        raise ValueError("--rounds must be at least 50 to demonstrate feedback learning.")

    rng = np.random.default_rng(seed)
    opportunity_embeddings, opportunity_ids = load_opportunity_embedding_matrix(
        opportunity_embeddings_path
    )
    opportunity_embeddings = np.asarray(
        [normalize_vector(vector) for vector in opportunity_embeddings],
        dtype="float32",
    )
    user_vector = normalize_vector(load_user_embedding(user_embeddings_path, user_id))
    opportunities = load_opportunities_for_learning(
        opportunities_path,
        opportunity_embeddings_path,
        recursive=recursive,
    )
    opportunities_by_id, embedding_row_by_id = build_opportunity_lookup(
        opportunities,
        opportunity_embeddings,
        opportunity_ids,
    )
    index = build_feedback_index(opportunity_embeddings)
    preference_vector = build_simulated_preference_vector(
        user_vector,
        opportunity_embeddings,
        preference_shift=preference_shift,
        seed=seed,
    )
    relevance_by_id = build_simulated_relevance_lookup(
        opportunity_embeddings,
        opportunity_ids,
        preference_vector,
    )

    recommendations = run_ranking_pipeline(
        index,
        user_vector,
        opportunities_by_id,
        opportunity_ids,
        top_k=top_k,
        rerank_method=rerank_method,
        diversity_bonus=diversity_bonus,
        mmr_lambda=mmr_lambda,
        recommendation_top_n=recommendation_top_n,
    )
    recommendations = attach_simulated_relevance(
        recommendations,
        relevance_by_id,
    )
    initial_ndcg = evaluate_simulated_ndcg(recommendations)

    history: list[FeedbackEvent] = []
    feedback_counts = {feedback_type: 0 for feedback_type in FEEDBACK_RELEVANCE}
    seen_counts: dict[str, int] = {}

    for round_id in range(1, rounds + 1):
        feedback = simulate_feedback(
            recommendations,
            rng=rng,
            exploration_rate=exploration_rate,
            negative_feedback_rate=negative_feedback_rate,
            seen_counts=seen_counts,
        )
        feedback_type = feedback["type"]
        feedback_counts[feedback_type] += 1
        seen_counts[feedback["opportunity_id"]] = seen_counts.get(feedback["opportunity_id"], 0) + 1
        item_index = embedding_row_by_id[feedback["opportunity_id"]]
        user_vector = update_user_vector(
            user_vector,
            opportunity_embeddings[item_index],
            feedback_type,
        )

        recommendations = run_ranking_pipeline(
            index,
            user_vector,
            opportunities_by_id,
            opportunity_ids,
            top_k=top_k,
            rerank_method=rerank_method,
            diversity_bonus=diversity_bonus,
            mmr_lambda=mmr_lambda,
            recommendation_top_n=recommendation_top_n,
        )
        recommendations = attach_simulated_relevance(
            recommendations,
            relevance_by_id,
        )
        ndcg_at_10 = evaluate_simulated_ndcg(recommendations)
        history.append(
            FeedbackEvent(
                round=round_id,
                feedback=feedback_type,
                opportunity_id=feedback["opportunity_id"],
                relevance=int(feedback["relevance"]),
                ndcg_at_10=ndcg_at_10,
            )
        )

    final_ndcg = history[-1].ndcg_at_10 if history else initial_ndcg
    improvement = (
        ((final_ndcg - initial_ndcg) / initial_ndcg) * 100.0
        if initial_ndcg > 0.0
        else 0.0
    )
    return LearningResult(
        user_id=user_id,
        initial_ndcg_at_10=round(initial_ndcg, 4),
        final_ndcg_at_10=round(final_ndcg, 4),
        improvement_percent=round(improvement, 2),
        feedback_rounds=rounds,
        learning_demonstrated=final_ndcg > initial_ndcg,
        feedback_counts=feedback_counts,
        history=[asdict(event) for event in history],
    )


def write_result(result: LearningResult, output_path: Path | None) -> None:
    payload = asdict(result)
    text = json.dumps(payload, indent=2)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")
        logger.info("Wrote adaptive learning result to %s", output_path)
    print(text)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    result = run_feedback_learning(
        opportunities_path=args.opportunities,
        opportunity_embeddings_path=args.opportunity_embeddings,
        user_embeddings_path=args.user_embeddings,
        user_id=args.user_id,
        rounds=args.rounds,
        top_k=args.top_k,
        recommendation_top_n=args.recommendation_top_n,
        rerank_method=args.rerank,
        diversity_bonus=args.diversity_bonus,
        mmr_lambda=args.mmr_lambda,
        exploration_rate=args.exploration_rate,
        negative_feedback_rate=args.negative_feedback_rate,
        preference_shift=args.preference_shift,
        seed=args.seed,
        recursive=args.recursive,
    )
    write_result(result, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
