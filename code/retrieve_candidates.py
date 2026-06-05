#!/usr/bin/env python3
"""
Retrieve top-k opportunity candidates for a user via FAISS nearest-neighbor search.

Inputs:
  - Opportunity embeddings (JSONL from generate_embeddings.py)
  - User profile embedding (JSONL from generate_user_embeddings.py)
  - Original opportunity dataset (CSV)

Example:
  python retrieve_candidates.py \\
    --opportunities Data/snapshots/engageiq_snapshot_v1.csv \\
    --opportunity-embeddings Data/embeddings/opportunity_embeddings.jsonl \\
    --user-embeddings Data/embeddings/user_embeddings.jsonl \\
    --user-id user_123 \\
    --top-k 100 \\
    --output Data/retrieval/user_123_candidates.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import json
import logging
from typing import Any

import numpy as np
import pandas as pd

from embedding_common import normalize_text
from generate_embeddings import build_opportunity_id, load_opportunities_csv
from ranking import l2_distance_to_similarity

logger = logging.getLogger(__name__)

DEFAULT_TOP_K = 100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Retrieve semantically relevant opportunities for a user embedding "
            "using FAISS L2 nearest-neighbor search."
        )
    )
    parser.add_argument(
        "--opportunities",
        type=Path,
        required=True,
        help="Original opportunity CSV file or directory.",
    )
    parser.add_argument(
        "--opportunity-embeddings",
        type=Path,
        required=True,
        help="Opportunity embeddings JSONL (opportunity_id, embedding, ...).",
    )
    parser.add_argument(
        "--user-embeddings",
        type=Path,
        required=True,
        help="User embeddings JSONL (user_id, profile_text, embedding).",
    )
    parser.add_argument(
        "--user-id",
        type=str,
        required=True,
        help="User id to retrieve candidates for.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Number of candidates to retrieve (default: {DEFAULT_TOP_K}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write candidate opportunities as CSV.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="If --opportunities is a directory, load CSV files recursively.",
    )
    return parser.parse_args()


def load_embeddings_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)

    records: list[dict[str, Any]] = []
    # utf-8-sig also accepts ordinary UTF-8 and strips a Windows-added BOM.
    with path.open(encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}") from exc

    if not records:
        raise ValueError(f"No embedding records found in {path}")

    return records


def load_opportunity_embedding_matrix(
    path: Path,
) -> tuple[np.ndarray, list[str]]:
    records = load_embeddings_jsonl(path)
    opportunity_ids: list[str] = []
    vectors: list[list[float]] = []

    for index, record in enumerate(records):
        opportunity_id = normalize_text(record.get("opportunity_id", ""))
        embedding = record.get("embedding")
        if not opportunity_id:
            raise ValueError(f"Record {index} in {path} is missing opportunity_id.")
        if not isinstance(embedding, list) or not embedding:
            raise ValueError(f"Record {index} in {path} has invalid embedding.")

        opportunity_ids.append(opportunity_id)
        vectors.append(embedding)

    matrix = np.array(vectors, dtype="float32")
    return matrix, opportunity_ids


def load_user_embedding(path: Path, user_id: str) -> np.ndarray:
    records = load_embeddings_jsonl(path)
    target = normalize_text(user_id)

    for record in records:
        if normalize_text(record.get("user_id", "")) != target:
            continue
        embedding = record.get("embedding")
        if not isinstance(embedding, list) or not embedding:
            break
        return np.array(embedding, dtype="float32")

    raise ValueError(f"User id {user_id!r} not found in {path}.")


def attach_opportunity_ids(opportunities: pd.DataFrame) -> pd.DataFrame:
    df = opportunities.copy()
    df["opportunity_id"] = df.apply(build_opportunity_id, axis=1)
    return df


def build_faiss_index(opportunity_embeddings: np.ndarray):
    try:
        import faiss
    except ImportError as exc:
        raise ImportError(
            "faiss is required. Install with:\n"
            "  pip install -r Data/requirements-embeddings.txt"
        ) from exc

    if opportunity_embeddings.ndim != 2:
        raise ValueError("Opportunity embeddings must be a 2-D matrix.")

    embedding_dim = opportunity_embeddings.shape[1]
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(opportunity_embeddings)
    return index


def search_index(
    index,
    user_embedding: np.ndarray,
    *,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    if user_embedding.ndim == 1:
        query = user_embedding.reshape(1, -1)
    else:
        query = user_embedding

    query = np.ascontiguousarray(query.astype("float32"))
    distances, indices = index.search(query, top_k)
    return distances, indices


def retrieve_candidate_opportunities(
    opportunities: pd.DataFrame,
    opportunity_embeddings: np.ndarray,
    opportunity_ids: list[str],
    user_embedding: np.ndarray,
    *,
    top_k: int = DEFAULT_TOP_K,
) -> pd.DataFrame:
    """
    Build a FAISS index over opportunity embeddings and return the top-k
    matching rows from the original opportunity dataset.
    """
    if len(opportunity_ids) != opportunity_embeddings.shape[0]:
        raise ValueError("opportunity_ids length must match embedding row count.")

    if user_embedding.shape[0] != opportunity_embeddings.shape[1]:
        raise ValueError(
            f"User embedding dimension ({user_embedding.shape[0]}) does not match "
            f"opportunity embeddings ({opportunity_embeddings.shape[1]})."
        )

    k = min(top_k, opportunity_embeddings.shape[0])
    index = build_faiss_index(opportunity_embeddings)
    distances, indices = search_index(index, user_embedding, top_k=k)

    ranked_ids: list[str] = []
    ranked_distances: list[float] = []
    for position, row_index in enumerate(indices[0]):
        if row_index < 0:
            continue
        ranked_ids.append(opportunity_ids[row_index])
        ranked_distances.append(float(distances[0][position]))

    opportunities = attach_opportunity_ids(opportunities)
    by_id = opportunities.drop_duplicates(subset=["opportunity_id"]).set_index(
        "opportunity_id",
        drop=False,
    )

    rows: list[pd.Series] = []
    for rank, (opportunity_id, distance) in enumerate(
        zip(ranked_ids, ranked_distances),
        start=1,
    ):
        if opportunity_id not in by_id.index:
            logger.warning(
                "Retrieved opportunity_id %r not found in dataset; skipping.",
                opportunity_id,
            )
            continue
        row = by_id.loc[opportunity_id].copy()
        row["retrieval_rank"] = rank
        row["retrieval_distance"] = distance
        row["similarity"] = l2_distance_to_similarity(distance)
        rows.append(row)

    if not rows:
        raise ValueError(
            "No retrieved candidates matched rows in the opportunity dataset. "
            "Ensure embeddings were generated from the same --opportunities file."
        )

    return pd.DataFrame(rows).reset_index(drop=True)


def write_candidates(candidates: pd.DataFrame, output_path: Path) -> Path:
    if output_path.suffix.lower() != ".csv":
        output_path = output_path.with_suffix(".csv")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(output_path, index=False)
    return output_path


def run(
    opportunities_path: Path,
    opportunity_embeddings_path: Path,
    user_embeddings_path: Path,
    user_id: str,
    *,
    top_k: int,
    output_path: Path | None,
    recursive: bool,
) -> pd.DataFrame:
    logger.info("Loading opportunity embeddings from %s", opportunity_embeddings_path)
    embedding_matrix, opportunity_ids = load_opportunity_embedding_matrix(
        opportunity_embeddings_path
    )
    logger.info("Indexed %d opportunity embedding(s)", embedding_matrix.shape[0])

    logger.info("Loading user embedding for %r", user_id)
    user_embedding = load_user_embedding(user_embeddings_path, user_id)

    logger.info("Loading opportunities from %s", opportunities_path)
    opportunities = load_opportunities_csv(
        opportunities_path,
        recursive=recursive,
    )

    logger.info("Searching FAISS index for top %d candidate(s)", top_k)
    candidate_opportunities = retrieve_candidate_opportunities(
        opportunities,
        embedding_matrix,
        opportunity_ids,
        user_embedding,
        top_k=top_k,
    )

    logger.info("Retrieved %d candidate opportunity(ies)", len(candidate_opportunities))

    if output_path is not None:
        csv_output_path = write_candidates(candidate_opportunities, output_path)
        logger.info("Wrote candidates to %s", csv_output_path)

    return candidate_opportunities


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    args = parse_args()
    run(
        args.opportunities,
        args.opportunity_embeddings,
        args.user_embeddings,
        args.user_id,
        top_k=args.top_k,
        output_path=args.output,
        recursive=args.recursive,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
