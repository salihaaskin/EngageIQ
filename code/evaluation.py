"""
Ranking evaluation with interaction-based relevance labels and NDCG@10.

Supports labeled evaluation datasets and model-variant comparison
(similarity-only → full scoring → diversity re-ranking).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score

from embedding_common import normalize_text
from generate_embeddings import build_opportunity_id
from ranking import ScoringVariant, score_candidates_variant

RELEVANCE_MAPPING: dict[str, int] = {
    "engaged": 3,
    "bookmarked": 2,
    "clicked": 1,
    "skipped": 0,
}

INTERACTION_FROM_RELEVANCE: dict[int, str] = {
    3: "engaged",
    2: "bookmarked",
    1: "clicked",
    0: "skipped",
}

MODEL_VARIANTS: tuple[ScoringVariant, ...] = (
    "similarity_only",
    "similarity_engagement",
    "full_scoring",
    "full_with_diversity",
)

MODEL_DISPLAY_NAMES: dict[str, str] = {
    "similarity_only": "Similarity Only",
    "similarity_engagement": "Similarity + Engagement",
    "full_scoring": "Full Scoring Model",
    "full_with_diversity": "Full Model + Diversity Re-ranking",
}

DEFAULT_NDCG_K = 10


def interaction_to_relevance(interaction: str) -> int:
    key = normalize_text(interaction).lower()
    if key not in RELEVANCE_MAPPING:
        raise ValueError(
            f"Unknown interaction {interaction!r}; "
            f"expected one of {sorted(RELEVANCE_MAPPING)}."
        )
    return RELEVANCE_MAPPING[key]


def relevance_to_interaction(relevance: int) -> str:
    if relevance not in INTERACTION_FROM_RELEVANCE:
        raise ValueError(f"Relevance must be 0–3, got {relevance}.")
    return INTERACTION_FROM_RELEVANCE[relevance]


def attach_opportunity_ids(df: pd.DataFrame) -> pd.DataFrame:
    if "opportunity_id" in df.columns and df["opportunity_id"].notna().all():
        return df.copy()
    out = df.copy()
    out["opportunity_id"] = out.apply(build_opportunity_id, axis=1)
    return out


def normalize_labels_frame(labels: pd.DataFrame) -> pd.DataFrame:
    """Ensure user_id, opportunity_id, interaction, relevance_label columns."""
    df = labels.copy()
    if "user_id" not in df.columns:
        raise ValueError("Labels must include user_id.")

    if "opportunity_id" not in df.columns:
        raise ValueError("Labels must include opportunity_id.")

    df["user_id"] = df["user_id"].map(normalize_text)
    df["opportunity_id"] = df["opportunity_id"].map(normalize_text)

    if "interaction" not in df.columns and "relevance_label" not in df.columns:
        raise ValueError("Labels need interaction or relevance_label.")

    if "relevance_label" not in df.columns:
        df["relevance_label"] = df["interaction"].map(interaction_to_relevance)
    else:
        df["relevance_label"] = pd.to_numeric(df["relevance_label"], errors="coerce").fillna(0).astype(int)

    if "interaction" not in df.columns:
        df["interaction"] = df["relevance_label"].map(relevance_to_interaction)

    return df[
        ["user_id", "opportunity_id", "interaction", "relevance_label"]
    ].drop_duplicates(subset=["user_id", "opportunity_id"])


def load_interaction_labels(path: Path) -> pd.DataFrame:
    """Load labels from Parquet, CSV, or JSON/JSONL."""
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        df = pd.read_parquet(path)
    elif suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix == ".jsonl":
        records = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        df = pd.DataFrame(records)
    elif suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            df = pd.DataFrame(payload)
        elif isinstance(payload, dict) and "interactions" in payload:
            df = pd.DataFrame(payload["interactions"])
        else:
            raise ValueError(f"Unsupported JSON structure in {path}.")
    else:
        raise ValueError(f"Unsupported labels format {suffix!r}.")

    return normalize_labels_frame(df)


def merge_candidates_with_labels(
    candidates: pd.DataFrame,
    labels: pd.DataFrame,
    user_id: str,
) -> pd.DataFrame:
    """
    Join retrieval pool with ground-truth labels for one user.

    Unlabeled candidates in the pool receive relevance_label 0 (skipped).
    """
    target_user = normalize_text(user_id)
    user_labels = labels[labels["user_id"] == target_user]

    pool = attach_opportunity_ids(candidates)
    merged = pool.merge(
        user_labels[["opportunity_id", "interaction", "relevance_label"]],
        on="opportunity_id",
        how="left",
    )
    merged["relevance_label"] = merged["relevance_label"].fillna(0).astype(int)
    merged["interaction"] = merged["interaction"].fillna("skipped")
    merged["user_id"] = target_user
    return merged


def compute_ndcg_at_k(
    true_relevance: Iterable[int] | Iterable[float],
    predicted_scores: Iterable[float],
    *,
    k: int = DEFAULT_NDCG_K,
) -> float:
    """
    NDCG@k for a single ranked list (one query / user).

    ``true_relevance`` and ``predicted_scores`` must already be aligned to the
    same candidate ordering (typically sort by ``predicted_scores`` descending
    before calling).
    """
    y_true = np.asarray(list(true_relevance), dtype=float).reshape(1, -1)
    y_score = np.asarray(list(predicted_scores), dtype=float).reshape(1, -1)

    if y_true.size == 0:
        return 0.0

    k_eff = min(k, y_true.shape[1])
    if y_true.max() == 0:
        return 0.0

    return float(ndcg_score(y_true, y_score, k=k_eff))


def evaluate_ranking_list(
    labeled_candidates: pd.DataFrame,
    *,
    score_column: str = "predicted_score",
    k: int = DEFAULT_NDCG_K,
) -> float:
    """NDCG@k on a labeled frame sorted by ``score_column``."""
    ordered = labeled_candidates.sort_values(score_column, ascending=False)
    return compute_ndcg_at_k(
        ordered["relevance_label"],
        ordered[score_column],
        k=k,
    )


def evaluate_variant_for_user(
    candidates: pd.DataFrame,
    labels: pd.DataFrame,
    user_id: str,
    variant: ScoringVariant,
    *,
    k: int = DEFAULT_NDCG_K,
) -> float:
    scored = score_candidates_variant(candidates, variant)
    labeled = merge_candidates_with_labels(scored, labels, user_id)
    if "predicted_score" not in labeled.columns:
        labeled = labeled.merge(
            scored[["opportunity_id", "predicted_score"]],
            on="opportunity_id",
            how="left",
        )
    return evaluate_ranking_list(labeled, score_column="predicted_score", k=k)


@dataclass
class ModelEvaluationResult:
    variant: str
    display_name: str
    ndcg_at_10: float
    per_user_ndcg: dict[str, float]


@dataclass
class RankingEvaluationReport:
    k: int
    user_ids: list[str]
    baseline_variant: str
    baseline_ndcg_at_10: float
    model_results: list[ModelEvaluationResult]

    def to_dict(self) -> dict[str, Any]:
        return {
            "k": self.k,
            "user_ids": self.user_ids,
            "baseline_variant": self.baseline_variant,
            "baseline_ndcg_at_10": round(self.baseline_ndcg_at_10, 4),
            "models": [
                {
                    "variant": result.variant,
                    "display_name": result.display_name,
                    "ndcg_at_10": round(result.ndcg_at_10, 4),
                    "per_user_ndcg": {
                        user: round(score, 4)
                        for user, score in result.per_user_ndcg.items()
                    },
                }
                for result in self.model_results
            ],
        }

    def comparison_table(self) -> list[tuple[str, float]]:
        return [(result.display_name, result.ndcg_at_10) for result in self.model_results]


def compare_ranking_models(
    candidates_by_user: dict[str, pd.DataFrame],
    labels: pd.DataFrame,
    *,
    variants: tuple[ScoringVariant, ...] = MODEL_VARIANTS,
    k: int = DEFAULT_NDCG_K,
    baseline_variant: ScoringVariant = "similarity_only",
) -> RankingEvaluationReport:
    """
    Macro-average NDCG@k across users for each ranking model variant.
    """
    labels = normalize_labels_frame(labels)
    user_ids = sorted(candidates_by_user.keys())

    model_results: list[ModelEvaluationResult] = []
    for variant in variants:
        per_user: dict[str, float] = {}
        for user_id in user_ids:
            candidates = candidates_by_user[user_id]
            per_user[user_id] = evaluate_variant_for_user(
                candidates,
                labels,
                user_id,
                variant,
                k=k,
            )
        mean_ndcg = float(np.mean(list(per_user.values()))) if per_user else 0.0
        model_results.append(
            ModelEvaluationResult(
                variant=variant,
                display_name=MODEL_DISPLAY_NAMES[variant],
                ndcg_at_10=mean_ndcg,
                per_user_ndcg=per_user,
            )
        )

    baseline_ndcg = next(
        (result.ndcg_at_10 for result in model_results if result.variant == baseline_variant),
        0.0,
    )

    return RankingEvaluationReport(
        k=k,
        user_ids=user_ids,
        baseline_variant=baseline_variant,
        baseline_ndcg_at_10=baseline_ndcg,
        model_results=model_results,
    )


def build_evaluation_export(
    candidates_by_user: dict[str, pd.DataFrame],
    labels: pd.DataFrame,
    variant: ScoringVariant = "full_with_diversity",
) -> pd.DataFrame:
    """
    Build evaluation dataset rows: user_id, opportunity_id, predicted_score,
    relevance_label, interaction.
    """
    labels = normalize_labels_frame(labels)
    rows: list[dict[str, Any]] = []

    for user_id, candidates in candidates_by_user.items():
        scored = score_candidates_variant(candidates, variant)
        labeled = merge_candidates_with_labels(scored, labels, user_id)
        if "predicted_score" not in labeled.columns:
            labeled = labeled.merge(
                scored[["opportunity_id", "predicted_score"]],
                on="opportunity_id",
                how="left",
            )
        for _, row in labeled.iterrows():
            rows.append(
                {
                    "user_id": user_id,
                    "opportunity_id": row["opportunity_id"],
                    "predicted_score": float(row["predicted_score"]),
                    "relevance_label": int(row["relevance_label"]),
                    "interaction": row["interaction"],
                    "title": normalize_text(row.get("title", "")),
                }
            )

    return pd.DataFrame(rows)


def format_markdown_report(report: RankingEvaluationReport) -> str:
    lines = [
        "# Ranking Evaluation (NDCG@10)",
        "",
        "Evaluation metric: **NDCG@10** — rewards placing high-relevance opportunities near the top.",
        "",
        "## Relevance labels",
        "",
        "| Interaction | Relevance |",
        "|-------------|-----------|",
    ]
    for interaction, relevance in sorted(
        RELEVANCE_MAPPING.items(), key=lambda item: item[1], reverse=True
    ):
        lines.append(f"| {interaction} | {relevance} |")

    lines.extend(
        [
            "",
            f"## Baseline ({MODEL_DISPLAY_NAMES[report.baseline_variant]})",
            "",
            f"**Baseline NDCG@10 = {report.baseline_ndcg_at_10:.4f}**",
            "",
            "## Model comparison",
            "",
            "| Model | NDCG@10 |",
            "|-------|---------|",
        ]
    )
    for display_name, ndcg in report.comparison_table():
        lines.append(f"| {display_name} | {ndcg:.4f} |")

    lines.extend(["", "## Per-user NDCG@10", ""])
    for result in report.model_results:
        lines.append(f"### {result.display_name}")
        lines.append("")
        for user_id, score in sorted(result.per_user_ndcg.items()):
            lines.append(f"- `{user_id}`: {score:.4f}")
        lines.append("")

    lines.extend(
        [
            "## Interpretation",
            "",
            "- **Similarity + Engagement** often improves NDCG@10 over similarity-only retrieval "
            "when community signals align with user interest.",
            "- **Full scoring** adds freshness, trend, and effort; gains depend on label quality.",
            "- **Diversity re-ranking** can lower NDCG@10 while improving topic spread — "
            "measure both ranking quality and diversity in production.",
            "",
            "## End-to-end pipeline",
            "",
            "```text",
            "Raw Data → Schema Standardization → Merge & Deduplication",
            "→ Domain Classification → Feature Engineering → Offline Snapshot",
            "→ Opportunity Embeddings → User Profile Embeddings → FAISS Retrieval",
            "→ Ranking & Engagement Scoring → Diversity Re-ranking → Explainability",
            "→ Evaluation (NDCG@10) → Final Ranked Recommendations",
            "```",
            "",
            "## How to reproduce",
            "",
            "```bash",
            "python generate_evaluation_labels.py \\",
            "  --user-profiles ../Data/examples/user_profiles.sample.json \\",
            "  --opportunities snapshots/v1.parquet \\",
            "  --opportunity-embeddings embeddings/opportunity_embeddings.jsonl \\",
            "  --user-embeddings embeddings/user_embeddings.jsonl \\",
            "  --output ../Data/examples/ranking_evaluation_labels.parquet",
            "",
            "python evaluate_ranking.py \\",
            "  --labels ../Data/examples/ranking_evaluation_labels.parquet \\",
            "  --opportunities snapshots/v1.parquet \\",
            "  --opportunity-embeddings embeddings/opportunity_embeddings.jsonl \\",
            "  --user-embeddings embeddings/user_embeddings.jsonl",
            "```",
        ]
    )
    return "\n".join(lines)
