"""
Human-readable “why this recommendation?” explanations from ranking features.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from embedding_common import normalize_text

SCORE_WEIGHTS = {
    "similarity": 0.45,
    "engagement_score": 0.20,
    "freshness_score": 0.15,
    "trend_score": 0.10,
    "effort_score": 0.10,
}

DEFAULT_THRESHOLDS = {
    "similarity": 0.85,
    "engagement_score": 0.75,
    "freshness_score": 0.75,
    "trend_score": 0.70,
    "effort_score": 0.75,
}

MAX_REASONS = 3


def _float_value(row: pd.Series, key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _resolve_diversity_domain(row: pd.Series) -> str:
    for column in ("topic_category", "domain_category", "domain"):
        if column not in row.index:
            continue
        value = normalize_text(row[column])
        if value and value.lower() not in {"unknown", "other", "nan", "none"}:
            return value
    return "unknown"


def _format_domain_label(row: pd.Series) -> str:
    label = _resolve_diversity_domain(row)
    if label == "unknown":
        for column in ("domain", "topic_category", "domain_category"):
            value = normalize_text(row.get(column, ""))
            if value and value.lower() not in {"unknown", "other", "nan", "none"}:
                label = value
                break
    return label.replace("_", " ").strip()


def _freshness_reason(row: pd.Series, freshness_score: float, threshold: float) -> str | None:
    days_old = row.get("days_old")
    if days_old is not None and not (isinstance(days_old, float) and pd.isna(days_old)):
        try:
            days = int(days_old)
        except (TypeError, ValueError):
            days = None
        if days is not None:
            if days <= 1:
                return "active in the last 24h"
            if days <= 7:
                return "posted this week"
            if days <= 30:
                return "active recently"

    if freshness_score >= threshold:
        return "active recently"
    return None


def _effort_reason(effort_score: float, threshold: float) -> str | None:
    if effort_score >= threshold:
        return "low effort to engage"
    if effort_score >= 0.6:
        return "moderate effort to engage"
    return None


def _collect_threshold_reasons(
    row: pd.Series,
    *,
    thresholds: dict[str, float],
) -> list[tuple[float, str]]:
    """Return (priority, reason text) pairs for features that pass thresholds."""
    similarity = _float_value(row, "similarity")
    engagement = _float_value(row, "engagement_score")
    freshness = _float_value(row, "freshness_score")
    trend = _float_value(row, "trend_score")
    effort = _float_value(row, "effort_score")

    candidates: list[tuple[float, str]] = []

    if similarity >= thresholds["similarity"]:
        candidates.append(
            (similarity * SCORE_WEIGHTS["similarity"], f"{round(similarity * 100)}% semantic match")
        )

    if engagement >= thresholds["engagement_score"]:
        candidates.append(
            (
                engagement * SCORE_WEIGHTS["engagement_score"],
                "high community engagement",
            )
        )

    freshness_text = _freshness_reason(row, freshness, thresholds["freshness_score"])
    if freshness_text:
        candidates.append(
            (freshness * SCORE_WEIGHTS["freshness_score"], freshness_text)
        )

    if trend >= thresholds["trend_score"]:
        candidates.append(
            (
                trend * SCORE_WEIGHTS["trend_score"],
                "strong recent activity trend",
            )
        )

    effort_text = _effort_reason(effort, thresholds["effort_score"])
    if effort_text:
        weight_key = "effort_score"
        candidates.append((effort * SCORE_WEIGHTS[weight_key], effort_text))

    domain_label = _format_domain_label(row)
    if domain_label and domain_label.lower() != "unknown":
        domain_priority = similarity * 0.25
        candidates.append(
            (domain_priority, f"matches your interest in {domain_label}")
        )

    return candidates


def _backfill_reasons(
    row: pd.Series,
    existing: list[str],
    *,
    max_reasons: int,
) -> list[str]:
    """Add reasons from the strongest features when threshold rules produce fewer than max_reasons."""
    if len(existing) >= max_reasons:
        return existing

    similarity = _float_value(row, "similarity")
    engagement = _float_value(row, "engagement_score")
    freshness = _float_value(row, "freshness_score")
    trend = _float_value(row, "trend_score")
    effort = _float_value(row, "effort_score")

    signal_templates: list[tuple[float, str]] = [
        (
            similarity * SCORE_WEIGHTS["similarity"],
            f"{round(similarity * 100)}% semantic match",
        ),
        (
            engagement * SCORE_WEIGHTS["engagement_score"],
            "high community engagement",
        ),
        (
            freshness * SCORE_WEIGHTS["freshness_score"],
            _freshness_reason(row, freshness, 1.0) or "recent opportunity",
        ),
        (
            trend * SCORE_WEIGHTS["trend_score"],
            "strong recent activity trend",
        ),
        (
            effort * SCORE_WEIGHTS["effort_score"],
            _effort_reason(effort, 1.0) or "higher effort to engage",
        ),
    ]

    domain_label = _format_domain_label(row)
    if domain_label and domain_label.lower() != "unknown":
        signal_templates.append(
            (similarity * 0.2, f"matches your interest in {domain_label}")
        )

    signal_templates.sort(key=lambda item: item[0], reverse=True)

    reasons = list(existing)
    seen = set(reasons)
    for _, text in signal_templates:
        if len(reasons) >= max_reasons:
            break
        if text in seen:
            continue
        reasons.append(text)
        seen.add(text)

    return reasons[:max_reasons]


def generate_reasons(
    row: pd.Series,
    *,
    thresholds: dict[str, float] | None = None,
    max_reasons: int = MAX_REASONS,
) -> list[str]:
    """
    Build up to ``max_reasons`` short explanation strings for one recommendation.

    Uses threshold rules from the product spec, then backfills from the
    strongest weighted ranking signals when fewer than ``max_reasons`` match.
    """
    thresholds = thresholds or DEFAULT_THRESHOLDS

    candidates = _collect_threshold_reasons(row, thresholds=thresholds)
    candidates.sort(key=lambda item: item[0], reverse=True)

    reasons: list[str] = []
    seen: set[str] = set()
    for _, text in candidates:
        if text in seen:
            continue
        reasons.append(text)
        seen.add(text)
        if len(reasons) >= max_reasons:
            return reasons

    return _backfill_reasons(row, reasons, max_reasons=max_reasons)


def attach_explanations(
    recommendations: pd.DataFrame,
    *,
    thresholds: dict[str, float] | None = None,
    max_reasons: int = MAX_REASONS,
    score_column: str = "reranked_score",
) -> pd.DataFrame:
    """Add a ``why`` column (list of reason strings) to each recommendation row."""
    if recommendations.empty:
        out = recommendations.copy()
        out["why"] = pd.Series(dtype=object)
        return out

    df = recommendations.copy()
    why_lists = [
        generate_reasons(row, thresholds=thresholds, max_reasons=max_reasons)
        for _, row in df.iterrows()
    ]
    df["why"] = why_lists

    if score_column in df.columns and "score" not in df.columns:
        df["score"] = df[score_column].astype(float)

    return df


def format_recommendation_record(row: pd.Series, *, score_column: str = "reranked_score") -> dict[str, Any]:
    """Compact JSON-friendly view of a single recommendation."""
    score = _float_value(row, "score", _float_value(row, score_column))
    why = row.get("why", [])
    if not isinstance(why, list):
        why = list(why) if why is not None else []

    return {
        "title": normalize_text(row.get("title", "")),
        "score": round(score, 4),
        "why": why,
        "url": normalize_text(row.get("url", "")),
        "source": normalize_text(row.get("source", "")),
        "domain": _format_domain_label(row),
    }
