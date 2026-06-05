"""
Ranking features, engagement-value scoring, diversity re-ranking, and evaluation.

Expects FAISS retrieval output (or equivalent) with semantic similarity and
opportunity metadata. Column names are normalized from common aliases
(description, engagement, comments, created_at) to the EngageIQ snapshot schema.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Literal

import numpy as np
import pandas as pd
try:
    from sklearn.metrics import ndcg_score
    from sklearn.preprocessing import MinMaxScaler
except ImportError:
    def ndcg_score(y_true, y_score, *, k=None) -> float:
        true = np.asarray(y_true, dtype=float)
        score = np.asarray(y_score, dtype=float)
        if true.ndim == 2:
            true = true[0]
        if score.ndim == 2:
            score = score[0]
        if true.size == 0 or true.max() == 0:
            return 0.0

        order = np.argsort(score)[::-1]
        k_eff = min(k or true.size, true.size)
        ranked_true = true[order][:k_eff]
        gains = np.power(2.0, ranked_true) - 1.0
        discounts = 1.0 / np.log2(np.arange(2, k_eff + 2))
        dcg = float(np.sum(gains * discounts))

        ideal = np.sort(true)[::-1][:k_eff]
        ideal_gains = np.power(2.0, ideal) - 1.0
        ideal_dcg = float(np.sum(ideal_gains * discounts))
        return dcg / ideal_dcg if ideal_dcg else 0.0

    class MinMaxScaler:
        def fit_transform(self, values):
            array = np.asarray(values, dtype=float)
            if array.ndim == 1:
                array = array.reshape(-1, 1)
            minimum = np.nanmin(array, axis=0)
            maximum = np.nanmax(array, axis=0)
            span = maximum - minimum
            span[span == 0] = 1.0
            return (array - minimum) / span

from embedding_common import normalize_text

from explainability import attach_explanations

COLUMN_ALIASES = {
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

EFFORT_SCORE_MAP = {
    "low": 1.0,
    "medium": 0.6,
    "high": 0.3,
}

DEFAULT_SCORE_WEIGHTS = {
    "similarity": 0.45,
    "engagement_score": 0.20,
    "freshness_score": 0.15,
    "trend_score": 0.10,
    "effort_score": 0.10,
}

FEATURES_TO_NORMALIZE = ("engagement_score", "freshness_score", "trend_score")

RerankMethod = Literal["simple", "mmr", "none"]
ScoringVariant = Literal[
    "similarity_only",
    "similarity_engagement",
    "full_scoring",
    "full_with_diversity",
]

SIMILARITY_ENGAGEMENT_WEIGHTS = {
    "similarity": 0.6,
    "engagement_score": 0.4,
}


def l2_distance_to_similarity(distance: float) -> float:
    """Map FAISS L2 distance to a bounded similarity in (0, 1]."""
    return 1.0 / (1.0 + max(float(distance), 0.0))


def rename_candidate_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        source: target
        for source, target in COLUMN_ALIASES.items()
        if source in df.columns and target not in df.columns
    }
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def resolve_diversity_domain(row: pd.Series) -> str:
    """Topic or domain label used for diversity re-ranking."""
    for column in ("topic_category", "domain_category", "domain"):
        if column not in row.index:
            continue
        value = normalize_text(row[column])
        if value and value.lower() not in {"unknown", "other", "nan", "none"}:
            return value
    return "unknown"


def parse_created_at(value: Any, current_date: datetime) -> datetime | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, pd.Timestamp):
        dt = value.to_pydatetime()
    else:
        text = normalize_text(value)
        if not text:
            return None
        text = text.replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(text)
        except ValueError:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def ensure_similarity_column(candidates: pd.DataFrame) -> pd.DataFrame:
    df = candidates.copy()
    if "similarity" in df.columns and df["similarity"].notna().any():
        df["similarity"] = pd.to_numeric(df["similarity"], errors="coerce").fillna(0.0)
        return df

    if "retrieval_distance" in df.columns:
        df["similarity"] = df["retrieval_distance"].apply(l2_distance_to_similarity)
        return df

    raise ValueError(
        "Candidates must include 'similarity' or 'retrieval_distance' for ranking."
    )


def estimate_effort(description: str) -> str:
    word_count = len(normalize_text(description).split())
    if word_count > 300:
        return "high"
    if word_count > 100:
        return "medium"
    return "low"


def compute_ranking_features(
    candidates: pd.DataFrame,
    *,
    current_date: datetime | None = None,
) -> pd.DataFrame:
    """
    Add raw ranking features: engagement_score, freshness_score, trend_score,
    effort, effort_score, and diversity_domain.
    """
    if candidates.empty:
        return candidates.copy()

    now = current_date or datetime.now(timezone.utc)
    df = rename_candidate_columns(ensure_similarity_column(candidates))

    engagement = pd.to_numeric(df.get("score", 0), errors="coerce").fillna(0)
    comments = pd.to_numeric(df.get("num_comments", 0), errors="coerce").fillna(0)

    df["engagement_raw"] = engagement
    df["comments_raw"] = comments
    df["engagement_score"] = np.log(engagement + comments + 1)

    days_old_list: list[int] = []
    freshness_list: list[float] = []
    trend_list: list[float] = []

    descriptions = df.get("text", df.get("description", pd.Series([""] * len(df))))

    for index, row in df.iterrows():
        created = parse_created_at(row.get("created_utc"), now)
        if created is None:
            days_old = 365
        else:
            days_old = max((now - created).days, 0)
        days_old_list.append(days_old)
        freshness_list.append(1.0 / (1.0 + days_old))
        comment_count = float(comments.loc[index])
        trend_list.append(comment_count / max(days_old, 1))

    df["days_old"] = days_old_list
    df["freshness_score"] = freshness_list
    df["trend_score"] = trend_list

    efforts: list[str] = []
    effort_scores: list[float] = []
    diversity_domains: list[str] = []

    for position, (_, row) in enumerate(df.iterrows()):
        desc = normalize_text(
            descriptions.iloc[position] if hasattr(descriptions, "iloc") else ""
        )
        effort = estimate_effort(desc)
        efforts.append(effort)
        effort_scores.append(EFFORT_SCORE_MAP[effort])
        diversity_domains.append(resolve_diversity_domain(row))

    df["effort"] = efforts
    df["effort_score"] = effort_scores
    df["diversity_domain"] = diversity_domains

    return df


def normalize_features(
    candidates: pd.DataFrame,
    feature_columns: tuple[str, ...] = FEATURES_TO_NORMALIZE,
) -> pd.DataFrame:
    df = candidates.copy()
    columns = [column for column in feature_columns if column in df.columns]
    if not columns:
        return df

    if len(df) == 1:
        for column in columns:
            df[column] = 0.0
        return df

    scaler = MinMaxScaler()
    df[columns] = scaler.fit_transform(df[columns].astype(float))
    return df


def compute_final_scores(
    candidates: pd.DataFrame,
    *,
    weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Compute weighted engagement-value score (final_score)."""
    weights = weights or DEFAULT_SCORE_WEIGHTS
    df = candidates.copy()

    required = ["similarity", "engagement_score", "freshness_score", "trend_score", "effort_score"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing ranking columns for final score: {missing}")

    df["final_score"] = (
        weights["similarity"] * df["similarity"].astype(float)
        + weights["engagement_score"] * df["engagement_score"].astype(float)
        + weights["freshness_score"] * df["freshness_score"].astype(float)
        + weights["trend_score"] * df["trend_score"].astype(float)
        + weights["effort_score"] * df["effort_score"].astype(float)
    )

    # Representation boost: the "beginner coding" domain is severely
    # underrepresented in the dataset (~0.2% of records).  Apply a small
    # additive boost so beginner-friendly items aren't systematically buried.
    _UNDERREPRESENTED_DOMAINS = {"beginner", "beginner_coding", "beginner coding", "learn", "learning"}
    _BEGINNER_KEYWORDS = ("good first issue", "good-first-issue", "beginner", "starter", "first issue")

    def _is_beginner_item(row: pd.Series) -> bool:
        domain_val = str(row.get("domain", "") or row.get("domain_category", "") or "").lower()
        if any(d in domain_val for d in _UNDERREPRESENTED_DOMAINS):
            return True
        text_val = " ".join(str(row.get(c, "")) for c in ("title", "text", "topic_category")).lower()
        return any(kw in text_val for kw in _BEGINNER_KEYWORDS)

    beginner_boost = df.apply(_is_beginner_item, axis=1).astype(float) * 0.10
    df["final_score"] = df["final_score"] + beginner_boost

    return df


def diversity_rerank_simple(
    candidates: pd.DataFrame,
    *,
    diversity_bonus: float = 0.05,
    score_column: str = "final_score",
) -> pd.DataFrame:
    """
    Apply a small bonus to the first candidate per diversity domain, then
    re-order by reranked_score descending.
    """
    if candidates.empty:
        return candidates.copy()

    domain_column = "diversity_domain" if "diversity_domain" in candidates.columns else "domain"
    selected_domains: set[str] = set()
    reranked_rows: list[pd.Series] = []

    ordered = candidates.sort_values(score_column, ascending=False)
    for _, row in ordered.iterrows():
        row = row.copy()
        score = float(row[score_column])
        domain_key = normalize_text(row.get(domain_column, "unknown")) or "unknown"
        if domain_key not in selected_domains:
            score += diversity_bonus
        row["reranked_score"] = score
        reranked_rows.append(row)
        selected_domains.add(domain_key)

    result = pd.DataFrame(reranked_rows)
    return result.sort_values("reranked_score", ascending=False).reset_index(drop=True)


def _domain_similarity(domain_a: str, domain_b: str) -> float:
    if not domain_a or not domain_b:
        return 0.0
    if domain_a == domain_b:
        return 1.0
    return 0.0


def diversity_rerank_mmr(
    candidates: pd.DataFrame,
    *,
    top_n: int | None = None,
    lambda_param: float = 0.7,
    score_column: str = "final_score",
) -> pd.DataFrame:
    """
    Greedy Maximal Marginal Relevance selection using diversity_domain overlap.
    """
    if candidates.empty:
        return candidates.copy()

    pool = candidates.copy()
    domain_column = "diversity_domain" if "diversity_domain" in pool.columns else "domain"
    n_select = top_n if top_n is not None else len(pool)
    n_select = min(n_select, len(pool))

    remaining = pool.index.tolist()
    selected_indices: list[Any] = []
    selected_domains: list[str] = []

    for _ in range(n_select):
        best_index: Any | None = None
        best_mmr = -math.inf

        for index in remaining:
            row = pool.loc[index]
            relevance = float(row[score_column])
            domain_key = normalize_text(row.get(domain_column, "unknown")) or "unknown"
            if selected_domains:
                max_sim = max(
                    _domain_similarity(domain_key, previous) for previous in selected_domains
                )
            else:
                max_sim = 0.0
            mmr = lambda_param * relevance - (1.0 - lambda_param) * max_sim
            if mmr > best_mmr:
                best_mmr = mmr
                best_index = index

        if best_index is None:
            break

        selected_indices.append(best_index)
        selected_domains.append(
            normalize_text(pool.loc[best_index].get(domain_column, "unknown")) or "unknown"
        )
        remaining.remove(best_index)

    selected = pool.loc[selected_indices].copy()
    selected["reranked_score"] = selected[score_column].astype(float)
    unselected = pool.drop(index=selected_indices, errors="ignore").copy()
    if not unselected.empty:
        unselected["reranked_score"] = unselected[score_column].astype(float) * 0.5
        selected = pd.concat([selected, unselected], ignore_index=False)

    return selected.reset_index(drop=True)


def score_candidates_variant(
    candidates: pd.DataFrame,
    variant: ScoringVariant,
    *,
    current_date: datetime | None = None,
    diversity_bonus: float = 0.05,
) -> pd.DataFrame:
    """
    Apply a ranking model variant and set ``predicted_score`` for evaluation.
    """
    featured = compute_ranking_features(candidates, current_date=current_date)
    normalized = normalize_features(featured)

    if variant == "similarity_only":
        result = normalized.copy()
        result["predicted_score"] = result["similarity"].astype(float)
        return result.sort_values("predicted_score", ascending=False).reset_index(drop=True)

    if variant == "similarity_engagement":
        result = normalized.copy()
        result["predicted_score"] = (
            SIMILARITY_ENGAGEMENT_WEIGHTS["similarity"] * result["similarity"].astype(float)
            + SIMILARITY_ENGAGEMENT_WEIGHTS["engagement_score"]
            * result["engagement_score"].astype(float)
        )
        return result.sort_values("predicted_score", ascending=False).reset_index(drop=True)

    scored = compute_final_scores(normalized)
    if variant == "full_scoring":
        result = scored.copy()
        result["predicted_score"] = result["final_score"].astype(float)
        return result.sort_values("predicted_score", ascending=False).reset_index(drop=True)

    if variant == "full_with_diversity":
        reranked = diversity_rerank_simple(scored, diversity_bonus=diversity_bonus)
        result = reranked.copy()
        result["predicted_score"] = result["reranked_score"].astype(float)
        return result

    raise ValueError(f"Unknown scoring variant: {variant!r}")


def compute_proxy_relevance(candidates: pd.DataFrame) -> pd.Series:
    """Proxy labels for NDCG when explicit relevance judgments are unavailable."""
    df = candidates.copy()
    similarity = df["similarity"].astype(float)

    engagement = pd.to_numeric(df.get("engagement_raw", df.get("score", 0)), errors="coerce")
    comments = pd.to_numeric(df.get("comments_raw", df.get("num_comments", 0)), errors="coerce")

    if len(df) == 1:
        norm_engagement = pd.Series([0.0], index=df.index)
        norm_comments = pd.Series([0.0], index=df.index)
    else:
        scaler = MinMaxScaler()
        norm_engagement = pd.Series(
            scaler.fit_transform(engagement.fillna(0).to_numpy().reshape(-1, 1)).ravel(),
            index=df.index,
        )
        norm_comments = pd.Series(
            scaler.fit_transform(comments.fillna(0).to_numpy().reshape(-1, 1)).ravel(),
            index=df.index,
        )

    return 0.5 * similarity + 0.3 * norm_engagement + 0.2 * norm_comments


def evaluate_ndcg_at_k(
    candidates: pd.DataFrame,
    *,
    score_column: str = "reranked_score",
    k: int = 10,
) -> float:
    """
    NDCG@k using proxy relevance labels and predicted ranking scores.
    """
    if candidates.empty:
        return 0.0

    proxy = compute_proxy_relevance(candidates)
    predicted = candidates[score_column].astype(float)

    k_eff = min(k, len(candidates))
    y_true = np.asarray([proxy.to_numpy()], dtype=float)
    y_score = np.asarray([predicted.to_numpy()], dtype=float)
    return float(ndcg_score(y_true, y_score, k=k_eff))


def rank_and_rerank_candidates(
    candidates: pd.DataFrame,
    *,
    rerank_method: RerankMethod = "simple",
    current_date: datetime | None = None,
    diversity_bonus: float = 0.05,
    mmr_lambda: float = 0.7,
    top_n: int | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """
    Full pipeline: feature engineering → normalization → final score → re-ranking.

    Returns ranked candidates and evaluation metrics.
    """
    if candidates.empty:
        return candidates.copy(), {"ndcg_at_10": 0.0}

    featured = compute_ranking_features(candidates, current_date=current_date)
    normalized = normalize_features(featured)
    scored = compute_final_scores(normalized)

    if rerank_method == "mmr":
        reranked = diversity_rerank_mmr(
            scored,
            top_n=top_n,
            lambda_param=mmr_lambda,
        )
    elif rerank_method == "simple":
        reranked = diversity_rerank_simple(scored, diversity_bonus=diversity_bonus)
    else:
        reranked = scored.copy()
        reranked["reranked_score"] = reranked["final_score"]

    if top_n is not None:
        reranked = reranked.head(top_n).reset_index(drop=True)

    reranked["rank"] = range(1, len(reranked) + 1)

    reranked = attach_explanations(reranked, score_column="reranked_score")

    metrics = {
        "ndcg_at_10": evaluate_ndcg_at_k(reranked, score_column="reranked_score", k=10),
        "ndcg_at_10_pre_rerank": evaluate_ndcg_at_k(
            scored.sort_values("final_score", ascending=False).reset_index(drop=True),
            score_column="final_score",
            k=10,
        ),
    }
    return reranked, metrics
