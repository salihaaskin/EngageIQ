"""User profile text assembly and embedding record builders."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import json
from typing import Any

from embedding_common import normalize_text, preprocess_embedding_text

PROFILE_TERM_FIELDS = (
    "interests",
    "technical_skills",
    "skills",
    "expertise",
    "preferred_domains",
    "domains",
    "topics_of_expertise",
    "expertise_topics",
    "topics",
)


def flatten_terms(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        term = normalize_text(value)
        return [term] if term else []
    if isinstance(value, list):
        terms: list[str] = []
        for item in value:
            terms.extend(flatten_terms(item))
        return terms
    return []


def terms_from_engagement_history(history: Any, *, max_entries: int = 20) -> list[str]:
    if not history:
        return []

    terms: list[str] = []
    for entry in history[:max_entries]:
        if isinstance(entry, str):
            terms.extend(flatten_terms(entry))
            continue
        if not isinstance(entry, dict):
            continue

        for key in ("title", "domain", "topic", "topic_category", "source"):
            terms.extend(flatten_terms(entry.get(key)))

    return terms


def collect_profile_terms(profile: dict[str, Any]) -> list[str]:
    """Collect unique profile terms in stable order."""
    seen: set[str] = set()
    ordered: list[str] = []

    def add(term: str) -> None:
        key = term.casefold()
        if not term or key in seen:
            return
        seen.add(key)
        ordered.append(term)

    for field in PROFILE_TERM_FIELDS:
        for term in flatten_terms(profile.get(field)):
            add(term)

    for term in terms_from_engagement_history(
        profile.get("engagement_history") or profile.get("history")
    ):
        add(term)

    explicit = normalize_text(profile.get("profile_text", ""))
    if explicit:
        for term in explicit.replace(",", " ").split():
            add(term.strip())

    return ordered


def build_user_profile_texts(profile: dict[str, Any]) -> tuple[str, str]:
    """Return (profile_text, encoding_text) for one user."""
    terms = collect_profile_terms(profile)
    if not terms:
        raise ValueError(
            f"User {profile.get('user_id', '?')!r} has no interests, skills, "
            "domains, or profile_text to embed."
        )

    profile_text = " ".join(terms)
    encoding_text = preprocess_embedding_text("\n".join(terms))
    return profile_text, encoding_text


def load_user_profiles(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        profiles = raw
    elif isinstance(raw, dict) and isinstance(raw.get("users"), list):
        profiles = raw["users"]
    else:
        raise ValueError(
            "User profiles JSON must be an array or an object with a 'users' array."
        )

    for index, profile in enumerate(profiles):
        if not isinstance(profile, dict):
            raise ValueError(f"User profile at index {index} must be an object.")
        if not normalize_text(profile.get("user_id", "")):
            raise ValueError(f"User profile at index {index} is missing user_id.")

    return profiles


def build_user_records(
    profiles: list[dict[str, Any]],
    profile_texts: list[str],
    embeddings: list[list[float]],
) -> list[dict[str, Any]]:
    if len(profiles) != len(embeddings) or len(profiles) != len(profile_texts):
        raise ValueError("Profile, profile_text, and embedding counts must match.")

    return [
        {
            "user_id": normalize_text(profile["user_id"]),
            "profile_text": profile_texts[index],
            "embedding": embeddings[index],
        }
        for index, profile in enumerate(profiles)
    ]
