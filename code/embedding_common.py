"""Shared Sentence-BERT loading, preprocessing, and encoding."""

from __future__ import annotations

from typing import Any

import pandas as pd

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def normalize_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def preprocess_embedding_text(text: str) -> str:
    """
    Normalize text before encoding so opportunity and user vectors share
    the same preprocessing and vector space.
    """
    if not text:
        return ""

    lines = []
    for line in text.splitlines():
        collapsed = " ".join(line.split())
        if collapsed:
            lines.append(collapsed)

    if lines:
        return "\n".join(lines)

    return " ".join(text.split())


def load_sentence_transformer(model_name: str, device: str | None):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is required. Install with:\n"
            "  pip install -r Data/requirements-embeddings.txt"
        ) from exc

    kwargs: dict[str, Any] = {}
    if device:
        kwargs["device"] = device
    return SentenceTransformer(model_name, **kwargs)


def embedding_dimension(model) -> int:
    if hasattr(model, "get_embedding_dimension"):
        return int(model.get_embedding_dimension())
    return int(model.get_sentence_embedding_dimension())


def encode_texts(
    model,
    texts: list[str],
    *,
    batch_size: int,
) -> list[list[float]]:
    prepared = [preprocess_embedding_text(text) for text in texts]
    if not prepared:
        return []

    vectors = model.encode(
        prepared,
        batch_size=batch_size,
        show_progress_bar=len(prepared) > batch_size,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    return [vector.tolist() for vector in vectors]
