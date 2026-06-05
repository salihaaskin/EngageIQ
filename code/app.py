from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from pathlib import Path
import sys

st.set_page_config(page_title="EngageIQ Dashboard", layout="wide")

BASE_DIR = Path(__file__).resolve().parent          # finalhw/code
PROJECT_ROOT = BASE_DIR.parent                      # finalhw

SNAPSHOT_PATH = PROJECT_ROOT / "data" / "main data" / "finaldataset.csv"
SCORED_PATH = PROJECT_ROOT / "data" / "scored_opportunities.csv"
TRENDS_PATH = PROJECT_ROOT / "data" / "trends.csv"

OPP_EMBEDDINGS_PATH = PROJECT_ROOT / "data" / "opportunity_embeddings.jsonl"
USER_EMBEDDINGS_PATH = PROJECT_ROOT / "data" / "user_embeddings.jsonl"

sys.path.insert(0, str(BASE_DIR))

STOP_WORDS = {
    "and", "are", "for", "from", "into", "the", "this", "with", "your",
}

PERSONAS: dict[str, dict[str, Any]] = {
    "Custom new user": {
        "user_id": "new-user",
        "interests": "",
        "expertise": "",
        "domains": "",
        "goal": "Find useful places to engage",
    },
    "Sofia ML Student": {
        "user_id": "sofia-ml",
        "interests": "large language models, RAG, MLOps, AI safety",
        "expertise": "python, notebooks, machine learning",
        "domains": "artificial_intelligence, machine_learning, AI Research",
        "goal": "Ask a thoughtful question",
    },
    "David DevOps Engineer": {
        "user_id": "david-devops",
        "interests": "kubernetes, observability, cloud native, automation",
        "expertise": "kubernetes, prometheus, terraform, CI/CD",
        "domains": "cloud_computing, devops, DevOps/K8s",
        "goal": "Find a page to comment on",
    },
    "Lina Data Journalist": {
        "user_id": "lina-data",
        "interests": "data tools, public datasets, AI policy, visualization",
        "expertise": "python, storytelling, data analysis",
        "domains": "research, Python Data Eng, AI Research",
        "goal": "Draft a short reply",
    },
    "Raj Startup Founder": {
        "user_id": "raj-founder",
        "interests": "developer tools, SaaS, AI agents, product analytics",
        "expertise": "go-to-market, product, APIs",
        "domains": "developer_tools, B2B SaaS, Cloud APIs",
        "goal": "Find useful places to engage",
    },
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_opportunities() -> pd.DataFrame:
    source_path = SNAPSHOT_PATH if SNAPSHOT_PATH.exists() else SCORED_PATH
    df = load_csv(source_path)
    if df.empty:
        return df

    df = df.copy()
    if "platform" not in df.columns and "source" in df.columns:
        df["platform"] = df["source"]
    if "source" not in df.columns and "platform" in df.columns:
        df["source"] = df["platform"]
    if "text" not in df.columns:
        df["text"] = ""
    if "url" not in df.columns:
        df["url"] = ""
    if "topic_category" not in df.columns:
        df["topic_category"] = df.get("domain_category", "General Tech")
    if "domain_category" not in df.columns:
        df["domain_category"] = df.get("topic_category", "other")
    if "id" not in df.columns:
        df["id"] = [f"opp_{i + 1}" for i in range(len(df))]

    for column in (
        "score", "num_comments", "age_days", "engagement_score",
        "final_score", "relevance_score", "effort_score",
    ):
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0)

    return df


@st.cache_data(show_spinner=False)
def load_trends() -> pd.DataFrame:
    trends = load_csv(TRENDS_PATH)
    if trends.empty:
        return trends
    trends = trends.copy()
    trends["count"] = pd.to_numeric(trends.get("count", 0), errors="coerce").fillna(0)
    return trends


# ---------------------------------------------------------------------------
# FAISS embedding pipeline (primary recommendation engine)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading FAISS embedding index...")
def load_faiss_engine():
    """
    Load opportunity embeddings and build a FAISS index.
    Returns (model, faiss_index, embedding_matrix, opportunity_ids)
    or (None, None, None, None) if unavailable.
    """
    try:
        from embedding_common import DEFAULT_MODEL, load_sentence_transformer
        from retrieve_candidates import (
            build_faiss_index,
            load_opportunity_embedding_matrix,
        )

        if not OPP_EMBEDDINGS_PATH.exists():
            return None, None, None, None

        model = load_sentence_transformer(DEFAULT_MODEL)
        matrix, opp_ids = load_opportunity_embedding_matrix(OPP_EMBEDDINGS_PATH)
        matrix = matrix.astype("float32")
        # L2-normalize for cosine-like similarity
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        matrix = matrix / norms
        faiss_index = build_faiss_index(matrix)
        return model, faiss_index, matrix, opp_ids
    except Exception:
        return None, None, None, None


def encode_profile(model, profile: dict[str, Any]) -> np.ndarray:
    """Encode the user's interest profile into a unit-norm embedding vector."""
    terms = profile["interests"] + profile["expertise"] + profile["domains"]
    profile_text = (
        f"Interests: {', '.join(profile['interests'])}. "
        f"Expertise: {', '.join(profile['expertise'])}. "
        f"Topics: {', '.join(profile['domains'])}."
    ) if terms else "general technology open source developer"
    vec = model.encode([profile_text])[0].astype("float32")
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec


def recommend_with_faiss(
    opportunities: pd.DataFrame,
    profile: dict[str, Any],
    *,
    top_n: int,
) -> tuple[pd.DataFrame, bool]:
    """
    FAISS-based recommendation:
      user profile → SBERT embedding → FAISS ANN search
      → engagement scoring → diversity re-ranking

    Returns (ranked_df, faiss_used).
    """
    model, faiss_index, matrix, opp_ids = load_faiss_engine()

    if model is None or faiss_index is None:
        return recommend_keyword_fallback(opportunities, profile, top_n=top_n), False

    try:
        from ranking import rank_and_rerank_candidates
        from retrieve_candidates import search_index

        user_vec = encode_profile(model, profile).reshape(1, -1)
        distances, indices = search_index(faiss_index, user_vec, top_k=min(top_n * 5, 300))

        # Build candidate DataFrame from retrieved indices
        rows = []
        for rank, (idx, dist) in enumerate(zip(indices[0], distances[0]), start=1):
            if idx < 0 or idx >= len(opp_ids):
                continue
            opp_id = opp_ids[int(idx)]
            # Match back to the opportunities DataFrame by source::id pattern
            source_part, _, id_part = opp_id.partition("::")
            match = opportunities[
                (opportunities["source"].str.lower() == source_part) &
                (opportunities["id"].astype(str) == id_part)
            ]
            if match.empty:
                continue
            row = match.iloc[0].copy()
            similarity = float(1.0 / (1.0 + max(float(dist), 0.0)))
            row["retrieval_rank"] = rank
            row["retrieval_distance"] = float(dist)
            row["similarity"] = similarity
            rows.append(row)

        if not rows:
            return recommend_keyword_fallback(opportunities, profile, top_n=top_n), False

        candidates = pd.DataFrame(rows).reset_index(drop=True)
        ranked, metrics = rank_and_rerank_candidates(
            candidates,
            rerank_method="simple",
            diversity_bonus=0.05,
            top_n=top_n,
        )

        ranked = ranked.head(top_n).copy()
        ranked["rank"] = range(1, len(ranked) + 1)
        ranked["recommended_page"] = ranked.apply(page_label, axis=1)
        ranked["recommended_topic"] = ranked.apply(topic_label, axis=1)
        terms = profile["interests"] + profile["expertise"] + profile["domains"]
        ranked["why"] = ranked.apply(lambda row: faiss_reasons(row, terms), axis=1)
        ranked["writing_prompt"] = ranked.apply(
            lambda row: writing_prompt(row, profile), axis=1
        )
        # Ensure id column exists for feedback
        if "id" not in ranked.columns:
            ranked["id"] = [f"opp_{i}" for i in range(len(ranked))]
        return ranked, True

    except Exception as exc:
        st.sidebar.caption(f"⚠️ FAISS ranking error, using keyword fallback: {exc}")
        return recommend_keyword_fallback(opportunities, profile, top_n=top_n), False


# ---------------------------------------------------------------------------
# Keyword fallback ranking (used when FAISS is unavailable)
# ---------------------------------------------------------------------------

def split_terms(value: str) -> list[str]:
    parts = re.split(r"[,;\n]+", value or "")
    return [part.strip() for part in parts if part.strip()]


def tokenize(value: Any) -> set[str]:
    words = re.findall(r"[a-z0-9+#.-]+", str(value).lower())
    return {word for word in words if len(word) > 2 and word not in STOP_WORDS}


def normalize_series(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").fillna(0)
    span = numeric.max() - numeric.min()
    if span == 0:
        return pd.Series([0.0] * len(numeric), index=numeric.index)
    return (numeric - numeric.min()) / span


def opportunity_text(row: pd.Series) -> str:
    fields = [
        row.get("title", ""), row.get("text", ""), row.get("platform", ""),
        row.get("domain", ""), row.get("domain_category", ""), row.get("topic_category", ""),
    ]
    return " ".join(str(field) for field in fields if pd.notna(field))


def profile_match(row: pd.Series, terms: list[str], profile_tokens: set[str]) -> float:
    text = opportunity_text(row).lower()
    row_tokens = tokenize(text)
    exact_matches = sum(1 for term in terms if term.lower() in text)
    exact_score = exact_matches / max(len(terms), 1)
    token_overlap = len(row_tokens & profile_tokens) / max(len(profile_tokens), 1)
    return min(1.0, 0.7 * exact_score + 0.3 * token_overlap)


def build_quality_score(df: pd.DataFrame) -> pd.Series:
    if "engagement_score" in df.columns:
        return normalize_series(df["engagement_score"])
    if {"score", "num_comments"}.issubset(df.columns):
        return normalize_series(df["score"] + df["num_comments"])
    if "final_score" in df.columns:
        return normalize_series(df["final_score"])
    return pd.Series([0.0] * len(df), index=df.index)


def build_freshness_score(df: pd.DataFrame) -> pd.Series:
    if "age_days" not in df.columns:
        return pd.Series([0.5] * len(df), index=df.index)
    age = pd.to_numeric(df["age_days"], errors="coerce").fillna(365)
    return 1.0 / (1.0 + age.clip(lower=0))


def build_effort_score(df: pd.DataFrame) -> pd.Series:
    if "effort_score" in df.columns:
        return normalize_series(df["effort_score"])
    lengths = df.apply(lambda row: len(str(row.get("text", "")).split()), axis=1)
    return 1.0 - normalize_series(lengths)


def recommend_keyword_fallback(
    opportunities: pd.DataFrame,
    profile: dict[str, Any],
    *,
    top_n: int,
) -> pd.DataFrame:
    if opportunities.empty:
        return opportunities

    terms = profile["interests"] + profile["expertise"] + profile["domains"]
    profile_tokens = tokenize(" ".join(terms))

    df = opportunities.copy()
    df["profile_match_score"] = df.apply(
        lambda row: profile_match(row, terms, profile_tokens), axis=1,
    )
    df["quality_score"] = build_quality_score(df)
    df["freshness_score"] = build_freshness_score(df)
    df["effort_score_normalized"] = build_effort_score(df)
    df["recommendation_score"] = (
        0.55 * df["profile_match_score"]
        + 0.25 * df["quality_score"]
        + 0.10 * df["freshness_score"]
        + 0.10 * df["effort_score_normalized"]
    )

    ranked = df.sort_values("recommendation_score", ascending=False).head(top_n).copy()
    ranked["rank"] = range(1, len(ranked) + 1)
    ranked["recommended_page"] = ranked.apply(page_label, axis=1)
    ranked["recommended_topic"] = ranked.apply(topic_label, axis=1)
    ranked["why"] = ranked.apply(lambda row: recommendation_reasons(row, terms), axis=1)
    ranked["writing_prompt"] = ranked.apply(lambda row: writing_prompt(row, profile), axis=1)
    return ranked


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def topic_label(row: pd.Series) -> str:
    for column in ("topic_category", "domain_category", "domain"):
        value = str(row.get(column, "")).strip()
        if value and value.lower() not in {"nan", "none", "unknown", "other"}:
            return value.replace("_", " ")
    return "General Tech"


def page_label(row: pd.Series) -> str:
    source = str(row.get("platform", row.get("source", ""))).lower()
    if "github" in source:
        return "GitHub issue or repository page"
    if "hacker" in source or source == "hn":
        return "Hacker News discussion"
    if "reddit" in source:
        return "Reddit community thread"
    return f"{source.title()} page" if source else "Community page"


def writing_prompt(row, profile):
    title = str(row.get("title", "Untitled"))
    source = str(row.get("source", "")).lower()
    topic = topic_label(row)
    goal = profile.get("goal", "")
    persona = profile.get("user_id", "")

    # Sofia ML Student
    if "sofia" in persona:
        prompts = [
            f"Write a beginner-friendly GitHub comment on '{title}' asking for clarification about the implementation.",
            f"Write a thoughtful question about '{title}' that could help you learn more about {topic}.",
            f"Draft a message introducing yourself as an ML student and asking how to contribute to '{title}'.",
            f"Write a short response highlighting what you learned from '{title}' and ask for recommended resources."
        ]
        return np.random.choice(prompts)

    # David DevOps
    if "david" in persona:
        prompts = [
            f"Write a DevOps-focused comment discussing scalability concerns in '{title}'.",
            f"Suggest a monitoring or observability improvement related to '{title}'.",
            f"Ask how '{title}' performs in production environments at scale.",
            f"Write a practical deployment question related to '{title}'."
        ]
        return np.random.choice(prompts)

    # Lina Journalist
    if "lina" in persona:
        prompts = [
            f"Write a data-journalism angle on '{title}' and identify a trend worth investigating.",
            f"Draft a question asking for evidence or data supporting the claims in '{title}'.",
            f"Summarize the key insight from '{title}' in two sentences for a broader audience.",
            f"Identify what makes '{title}' newsworthy and suggest a follow-up question."
        ]
        return np.random.choice(prompts)

    # Raj Founder
    if "raj" in persona:
        prompts = [
            f"Write a founder-focused comment exploring product-market fit for '{title}'.",
            f"Ask users what their biggest pain point is related to '{title}'.",
            f"Draft a response gathering feedback on business value and adoption challenges.",
            f"Write a question about monetization opportunities related to '{title}'."
        ]
        return np.random.choice(prompts)

    # Source-specific fallback
    if "github" in source:
        prompts = [
            f"Suggest a code improvement for '{title}'.",
            f"Write a GitHub issue comment proposing an enhancement for '{title}'.",
            f"Ask a technical implementation question about '{title}'.",
            f"Propose a new feature idea related to '{title}'."
        ]
    elif "reddit" in source:
        prompts = [
            f"Write a helpful Reddit reply for '{title}'.",
            f"Add a personal insight to the discussion around '{title}'.",
            f"Ask a follow-up question that encourages discussion on '{title}'.",
            f"Provide a constructive counterpoint to the discussion in '{title}'."
        ]
    elif "hacker" in source:
        prompts = [
            f"Write a concise Hacker News comment sharing your perspective on '{title}'.",
            f"Highlight a tradeoff mentioned in '{title}' and discuss its implications.",
            f"Ask a thoughtful technical question about '{title}'.",
            f"Relate '{title}' to a similar trend you've observed."
        ]
    else:
        prompts = [
            f"Create an engagement message for '{title}'.",
            f"Write a short response highlighting the most interesting aspect of '{title}'.",
            f"Suggest a discussion topic inspired by '{title}'.",
            f"Ask a meaningful follow-up question related to '{title}'."
        ]

    return np.random.choice(prompts)

def faiss_reasons(row: pd.Series, terms: list[str]) -> list[str]:
    """Generate 'Why this?' reasons for FAISS-ranked results."""
    reasons: list[str] = []
    similarity = float(row.get("similarity", 0.0))
    if similarity >= 0.85:
        reasons.append(f"high semantic similarity to your profile ({similarity:.2f})")
    elif similarity >= 0.70:
        reasons.append(f"strong semantic match to your interests ({similarity:.2f})")
    else:
        reasons.append(f"semantic relevance score: {similarity:.2f}")

    engagement = float(row.get("engagement_score", row.get("score", 0)))
    if engagement >= 50:
        reasons.append("high community engagement (comments + upvotes)")

    freshness = float(row.get("freshness_score", 0.0))
    if freshness >= 0.05:
        reasons.append("recent enough to engage meaningfully")

    topic = topic_label(row)
    if topic != "General Tech":
        reasons.append(f"domain: {topic}")

    return reasons[:3]


def recommendation_reasons(row: pd.Series, terms: list[str]) -> list[str]:
    text = opportunity_text(row).lower()
    matched = [term for term in terms if term.lower() in text][:3]
    reasons = [f"profile match: {term}" for term in matched]
    if float(row.get("quality_score", 0)) >= 0.7:
        reasons.append("strong community activity")
    if float(row.get("freshness_score", 0)) >= 0.05:
        reasons.append("recent enough to engage")
    if not reasons:
        reasons.append(f"closest available match in {topic_label(row)}")
    return reasons[:3]


def feedback_key(user_id: str, opportunity_id: Any, action: str) -> str:
    return f"{user_id}:{opportunity_id}:{action}"


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.title("EngageIQ - Smart Engagement Dashboard")

opportunities = load_opportunities()
trends = load_trends()

if opportunities.empty:
    st.error("No opportunity data found.")
    st.stop()

st.sidebar.header("User Profile")
persona_name = st.sidebar.selectbox("Start from persona", list(PERSONAS.keys()))
persona = PERSONAS[persona_name]

with st.sidebar.form("profile_form"):
    user_id = st.text_input("User id", value=persona["user_id"])
    interests_text = st.text_area("Interests", value=persona["interests"], height=80)
    expertise_text = st.text_area("Expertise", value=persona["expertise"], height=70)
    domains_text = st.text_area("Preferred domains/topics", value=persona["domains"], height=70)
    goal = st.selectbox(
        "Engagement goal",
        [
            "Find useful places to engage",
            "Ask a thoughtful question",
            "Draft a short reply",
            "Find a page to comment on",
        ],
        index=[
            "Find useful places to engage",
            "Ask a thoughtful question",
            "Draft a short reply",
            "Find a page to comment on",
        ].index(persona["goal"]),
    )
    top_n = st.slider("Number of recommendations", 5, 25, 10)
    # Sofia persona: surface GitHub "good first issue" items
    good_first_issue_only = st.checkbox(
        "Good first issues only (GitHub)",
        value=persona_name == "Sofia ML Student",
        help="Filter to GitHub issues tagged 'good first issue' — ideal for first-time contributors.",
    )
    submitted = st.form_submit_button("Update recommendations")

profile = {
    "user_id": user_id.strip() or "new-user",
    "interests": split_terms(interests_text),
    "expertise": split_terms(expertise_text),
    "domains": split_terms(domains_text),
    "goal": goal,
}

# Apply "good first issue" pre-filter for Sofia / any user who opts in
opportunities_filtered = opportunities.copy()
if good_first_issue_only:
    gfi_mask = (
        opportunities_filtered
        .apply(lambda r: any(
            kw in str(r.get(col, "")).lower()
            for kw in ("good first issue", "good-first-issue", "beginner", "starter", "first-issue")
            for col in ("title", "text", "topic_category", "domain_category", "domain")
        ), axis=1)
        | (opportunities_filtered.get("source", pd.Series(dtype=str)).str.lower() == "github")
    )
    gfi_df = opportunities_filtered[gfi_mask]
    # Fall back to all GitHub issues if the strict filter yields too few results
    if len(gfi_df) >= 5:
        opportunities_filtered = gfi_df
    else:
        github_mask = opportunities_filtered.get("source", pd.Series(dtype=str)).str.lower() == "github"
        opportunities_filtered = opportunities_filtered[github_mask] if github_mask.any() else opportunities_filtered

# Run the recommendation engine (FAISS primary, keyword fallback)
recommendations, faiss_used = recommend_with_faiss(opportunities_filtered, profile, top_n=top_n)

if "feedback" not in st.session_state:
    st.session_state.feedback = {}
if "skipped_ids" not in st.session_state:
    st.session_state.skipped_ids = set()
if "skipped_domains" not in st.session_state:
    st.session_state.skipped_domains = {}  # domain → skip count


def apply_skip_deprioritization(df: pd.DataFrame, skipped_ids: set, skipped_domains: dict) -> pd.DataFrame:
    """
    Adaptive re-ranking: penalise previously skipped items and their domains.
    Demonstrates Capability 6 (learning from skips) for Raj persona.
    """
    if df.empty or (not skipped_ids and not skipped_domains):
        return df
    df = df.copy()
    score_col = "reranked_score" if "reranked_score" in df.columns else "recommendation_score"
    if score_col not in df.columns:
        df[score_col] = 0.0

    def _penalty(row: pd.Series) -> float:
        penalty = 0.0
        item_id = str(row.get("id", ""))
        if item_id in skipped_ids:
            penalty += 0.30  # hard down-rank for explicitly skipped items
        domain_key = str(row.get("domain", row.get("domain_category", "unknown"))).lower()
        skip_count = skipped_domains.get(domain_key, 0)
        penalty += min(skip_count * 0.05, 0.20)  # up to -0.20 for repeat skips in same domain
        return penalty

    df[score_col] = df[score_col].astype(float) - df.apply(_penalty, axis=1)
    df = df.sort_values(score_col, ascending=False).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)
    return df


# Apply any accumulated in-session skip learning before rendering
recommendations = apply_skip_deprioritization(
    recommendations, st.session_state.skipped_ids, st.session_state.skipped_domains
)

# Engine indicator
engine_label = "FAISS + SBERT" if faiss_used else "Keyword Ranking (fallback)"
engine_color = "green" if faiss_used else "orange"
st.sidebar.markdown(
    f"**Ranking engine:** :{engine_color}[{engine_label}]",
    help=(
        "FAISS + SBERT uses dense vector embeddings (Sentence-BERT) with "
        "approximate nearest-neighbor retrieval for semantic matching. "
        "Keyword fallback uses token overlap scoring."
    ),
)

metric1, metric2, metric3, metric4 = st.columns(4)
opp_count = len(opportunities_filtered)
metric1.metric("Opportunities", f"{opp_count:,}", delta=f"{opp_count - len(opportunities):,}" if good_first_issue_only else None)
metric2.metric("Recommendations", f"{len(recommendations):,}")
metric3.metric("Topics found", f"{recommendations['recommended_topic'].nunique():,}")
metric4.metric("User", profile["user_id"])

profile_terms = profile["interests"] + profile["expertise"] + profile["domains"]
if not profile_terms:
    st.info("Add interests, expertise, or preferred domains to personalize the ranking for a new user.")

recommendation_tab, topics_tab, writing_tab, analytics_tab = st.tabs(
    ["Recommendations", "Topics", "Writing", "Analytics"]
)

with recommendation_tab:
    st.header("Recommended Pages")
    if faiss_used:
        st.caption(
            "Ranked by semantic similarity (SBERT embeddings + FAISS) "
            "→ engagement scoring → diversity re-ranking."
        )
    for _, row in recommendations.iterrows():
        with st.container(border=True):
            title = str(row.get("title", "Untitled opportunity"))
            score_col = "reranked_score" if "reranked_score" in row.index else "recommendation_score"
            score = round(float(row.get(score_col, 0)), 3)
            st.subheader(f"{int(row['rank'])}. {title}")
            st.write(
                f"**Page:** {row['recommended_page']}  |  "
                f"**Topic:** {row['recommended_topic']}  |  "
                f"**Score:** {score}"
            )
            url = str(row.get("url", "")).strip()
            if url:
                st.link_button("Open page", url)

            with st.expander("Why this recommendation?"):
                for reason in row["why"]:
                    st.write(f"- {reason}")

            with st.expander("Suggested writing"):
                st.write(row["writing_prompt"])

            col1, col2, col3 = st.columns(3)
            for label, column in (("Engage", col1), ("Skip", col2), ("Bookmark", col3)):
                key = feedback_key(profile["user_id"], row["id"], label.lower())
                if column.button(label, key=key):
                    st.session_state.feedback[key] = {
                        "user_id": profile["user_id"],
                        "opportunity_id": row["id"],
                        "action": label.lower(),
                    }
                    if label == "Skip":
                        # Adaptive learning: record skip so next render deprioritises
                        # this item and its domain (Raj founder use-case).
                        opp_id = str(row.get("id", ""))
                        st.session_state.skipped_ids.add(opp_id)
                        domain_key = str(row.get("domain", row.get("domain_category", "unknown"))).lower()
                        st.session_state.skipped_domains[domain_key] = (
                            st.session_state.skipped_domains.get(domain_key, 0) + 1
                        )
                        st.toast(f"Skipped — future recommendations will deprioritise this domain.")
                        st.rerun()
                    else:
                        st.toast(f"Saved feedback: {label}")

with topics_tab:
    st.header("Recommended Topics")
    topic_summary = (
        recommendations.groupby("recommended_topic")
        .agg(
            recommendations=("id", "count"),
            avg_score=("reranked_score" if "reranked_score" in recommendations.columns else "recommendation_score", "mean"),
            best_page=("title", "first"),
        )
        .sort_values(["recommendations", "avg_score"], ascending=False)
        .reset_index()
    )
    st.dataframe(
        topic_summary.rename(
            columns={
                "recommended_topic": "Topic",
                "recommendations": "Recommended pages",
                "avg_score": "Average score",
                "best_page": "Best page to start",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

with writing_tab:
    st.header("Writing Suggestions")
    score_col = "reranked_score" if "reranked_score" in recommendations.columns else "recommendation_score"
    writing_rows = recommendations[
        ["rank", "title", "recommended_topic", "writing_prompt"]
    ].rename(
        columns={
            "rank": "Rank",
            "title": "Page",
            "recommended_topic": "Topic",
            "writing_prompt": "Prompt",
        }
    )
    st.dataframe(writing_rows, use_container_width=True, hide_index=True)

with analytics_tab:
    st.header("Trend Analytics")
    col1, col2 = st.columns(2)

    score_col = "reranked_score" if "reranked_score" in recommendations.columns else "recommendation_score"

    with col1:
        st.subheader("Profile-Matched Topics")
        fig = px.bar(
            recommendations,
            x="recommended_topic",
            y=score_col,
            color="recommended_page",
            labels={"recommended_topic": "Topic", score_col: "Score"},
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Global Topic Trends")
        if trends.empty:
            st.info("No trend data available.")
        else:
            fig = px.pie(trends.head(12), names="category", values="count")
            st.plotly_chart(fig, use_container_width=True)

    # Week-over-week volume change — key metric for Lina (Data Journalist)
    st.subheader("Week-over-Week Volume Change")
    vol_path = PROJECT_ROOT / "Analytics" / "opportunity_volume_over_time.csv"
    if vol_path.exists():
        vol_df = pd.read_csv(vol_path, parse_dates=["day"])
        vol_df = vol_df.dropna(subset=["day"]).sort_values("day")
        if len(vol_df) >= 14:
            vol_df["week"] = vol_df["day"].dt.to_period("W").apply(lambda p: p.start_time)
            weekly = vol_df.groupby("week")["count"].sum().reset_index()
            weekly["prev_week"] = weekly["count"].shift(1)
            weekly["wow_delta"] = weekly["count"] - weekly["prev_week"]
            weekly["wow_pct"] = (weekly["wow_delta"] / weekly["prev_week"].replace(0, float("nan")) * 100).round(1)
            weekly = weekly.dropna(subset=["prev_week"]).tail(12)
            fig_wow = px.bar(
                weekly,
                x="week",
                y="wow_pct",
                labels={"week": "Week", "wow_pct": "WoW Change (%)"},
                color="wow_pct",
                color_continuous_scale=["#d62728", "#aec7e8", "#1f77b4"],
                title="Weekly Post Volume — Week-over-Week % Change",
            )
            fig_wow.add_hline(y=0, line_dash="dash", line_color="gray")
            st.plotly_chart(fig_wow, use_container_width=True)

            # Lina-specific summary callout
            if len(weekly) >= 2:
                latest = weekly.iloc[-1]
                direction = "up" if latest["wow_pct"] > 0 else "down"
                st.info(
                    f"**Last week:** {int(latest['count']):,} posts "
                    f"({direction} {abs(latest['wow_pct']):.1f}% vs the prior week). "
                    "Use this trend signal for your data story."
                )
        else:
            st.info("Not enough daily volume data to compute week-over-week changes (need ≥ 14 days).")
    else:
        st.info("Volume analytics not yet computed — run `spark-submit Embeddings/analytics_spark.py` first.")

    st.subheader("Weekly Engagement Brief")
    export_columns = [
        col for col in [
            "rank", "title", "recommended_page", "recommended_topic",
            score_col, "writing_prompt", "url",
        ] if col in recommendations.columns
    ]
    csv = recommendations[export_columns].to_csv(index=False)
    st.download_button(
        label="Download Weekly Brief CSV",
        data=csv,
        file_name=f"{profile['user_id']}_weekly_engagement_brief.csv",
        mime="text/csv",
    )
