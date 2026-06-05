# EngageIQ — AI Prompts Used (Expanded Submission Version)

This document records the major AI-assisted prompts used throughout the development of EngageIQ. It demonstrates how generative AI was used for architecture design, data engineering, recommendation modeling, ranking, evaluation, explainability, analytics, and dashboard development.

---

# Data Collection & Dataset Construction

## 1. Hacker News Data Collection

**Prompt**
> Using the Hacker News API (https://github.com/HackerNews/API), collect approximately 5,000 posts across AI, ML, Data Engineering, DevOps, Cloud, Cybersecurity, Databases, Developer Tools, Open Source, Blockchain, and related technical domains. Ensure balanced sampling, deduplication, UTC timestamps, schema validation, and export to a Snappy-compressed Parquet file.

**Used For**
- `Data/hackernews_ingestion.py`
- Historical opportunity collection
- Balanced domain coverage
- Parquet snapshot generation

---

## 2. GitHub Event Collection via GH Archive

**Prompt**
> Using GH Archive, collect approximately 5,000 public GitHub events from the last two years. Filter events using repository metadata, pull requests, commits, issue titles, and topic keywords. Deduplicate by event ID and export to Parquet.

**Used For**
- `Data/github_ingestion.py`
- GitHub opportunity discovery
- Open-source activity analysis
- Offline snapshot creation

---

## 3. Unified Dataset Schema Design

**Prompt**
> Design a unified schema for combining GitHub events, Reddit posts, and Hacker News stories into a single opportunity dataset that supports semantic retrieval, ranking, trend analytics, engagement scoring, and domain classification.

**Used For**
- Schema standardization
- Multi-source data integration
- Common feature definitions

---

# Data Engineering Pipeline

## 4. Merge & Snapshot Pipeline

**Prompt**
> Improve merge.py by implementing the following pipeline:
>
> Raw Data → Schema Standardization → Merge → Deduplication → Domain Classification → Domain Coverage Check → Feature Engineering → Offline Snapshot Export → Snapshot Metadata

**Used For**
- `Data/merge.py`
- Snapshot generation
- Metadata reporting
- Deduplication workflow

### Generated Features
- title_length
- age_days
- engagement_score
- source
- domain

---

# Embedding & Retrieval

## 5. Embedding Template Design

**Prompt**
> What is the best embedding template for developer engagement opportunities using sentence-transformers/all-MiniLM-L6-v2? The template should incorporate title, content, domain, and source information.

**Used For**
- `Embeddings/generate_embeddings.py`

### Template
```text
Title: {title}
Description: {text}
Domain: {domain}
Source: {source}
```

---

## 6. FAISS Candidate Retrieval

**Prompt**
> Implement approximate nearest-neighbor retrieval using FAISS. Given a user profile embedding and opportunity embeddings, retrieve the top 100 semantically relevant opportunities.

**Used For**
- `Embeddings/retrieval.py`
- Candidate generation layer

### Outcome
- Top-100 candidate opportunities
- ANN retrieval requirement

---

# Ranking & Recommendation

## 7. Multi-Signal Ranking Design

**Prompt**
> Rank engagement opportunities using semantic similarity, engagement, freshness, trend momentum, and effort. Prioritize relevance while still accounting for visibility and ease of participation.

**Used For**
- `Embeddings/ranking.py`

### Final Weighting
| Feature | Weight |
|----------|---------|
| Similarity | 0.45 |
| Engagement | 0.20 |
| Freshness | 0.15 |
| Trend | 0.10 |
| Effort | 0.10 |

---

## 8. Diversity Re-Ranking

**Prompt**
> Explain and implement both simple diversity re-ranking and Maximal Marginal Relevance (MMR). The goal is to avoid recommendation lists dominated by a single topic.

**Used For**
- `rerank_simple()`
- `rerank_mmr()`

### Purpose
Improve recommendation diversity while preserving relevance.

---

## 9. Explainability Layer

**Prompt**
> Generate human-readable "Why this recommendation?" explanations based on semantic similarity, engagement, freshness, trend activity, effort estimate, and domain match.

**Used For**
- `Embeddings/explainability.py`

### Example Output
- 92% semantic match
- High community engagement
- Active recently

---

# Evaluation

## 10. Ranking Evaluation Metric Selection

**Prompt**
> What ranking metric should be used when opportunities have graded relevance labels (engaged=3, bookmarked=2, clicked=1, skipped=0)?

**Used For**
- NDCG@10 evaluation

### Reason
Measures whether highly relevant opportunities appear near the top of recommendation lists.

---

## 11. NDCG Analysis & Debugging

**Prompt**
> In my evaluation, a similarity-only model outperforms the full ranking model. Is this expected if labels were generated from cosine similarity?

**Used For**
- Evaluation interpretation
- Label leakage analysis
- Experimental reporting

---

## 12. Full Ranking Evaluation Pipeline

**Prompt**
> Implement NDCG@10 evaluation for candidate retrieval, ranking, and re-ranking. Compare multiple ranking configurations and report improvements.

**Used For**
- `Embeddings/evaluation.py`
- Benchmark comparisons

### Compared Models
- Similarity Only
- Similarity + Engagement
- Full Ranking Model
- Full Ranking + Diversity Re-ranking

---

# Adaptive Learning

## 13. Adaptive Learning from Feedback

**Prompt**
> Implement a recommendation learning mechanism using Engage, Bookmark, and Skip feedback. Update user embeddings after each interaction and re-run retrieval, ranking, and evaluation.

**Used For**
- `Embeddings/adaptive_learning.py`

### Feedback Updates
- Engage → move closer
- Bookmark → move moderately closer
- Skip → move away

### Demonstration
- 50+ simulated feedback rounds
- NDCG improvement tracking

---

# Analytics & Trend Detection

## 14. Spark Analytics Pipeline

**Prompt**
> Write a PySpark analytics pipeline that computes active communities, opportunity volume over time, domain distributions, trending topics, and week-over-week changes.

**Used For**
- `analytics_spark.py`

### Outputs
- Community leaderboard
- Trend analytics
- Domain distributions
- Time-series visualizations

---

# Streaming Architecture

## 15. Streaming Pipeline Design

**Prompt**
> Design a producer-consumer streaming architecture for GitHub, Reddit, and Hacker News ingestion that can later be upgraded to Kafka.

**Used For**
- `streaming_pipeline.py`

### Outcome
- Queue-based architecture
- Kafka-compatible design

---

# User Experience & Dashboard

## 16. Persona Design

**Prompt**
> Design four user personas for an engagement recommendation system: portfolio builder, DevOps engineer, startup founder, and data journalist.

**Used For**
- Persona testing
- Recommendation filtering
- Demo scenarios

---

## 17. Streamlit Dashboard Design

**Prompt**
> Design a Streamlit dashboard containing recommendations, analytics, trend visualizations, user feedback controls, explanations, and downloadable engagement briefs.

**Used For**
- `App/app.py`

### Dashboard Components
- Persona selector
- Ranked recommendations
- Why-this explanations
- Engage / Skip / Bookmark buttons
- Trend analytics
- Weekly engagement brief export

---

## 18. Beginner-Coding Domain Handling

**Prompt**
> The beginner-coding category is severely underrepresented. How should I prevent recommendation quality degradation for beginner users?

**Used For**
- Domain-aware boosting
- Good-first-issue prioritization

---

# End-to-End Pipeline Prompt

## 19. Full Recommendation System Blueprint

**Prompt**
> Design an end-to-end developer engagement recommendation system including ingestion, normalization, embeddings, FAISS retrieval, ranking, diversity re-ranking, explainability, adaptive learning, analytics, evaluation, and dashboard delivery.

**Used For**
Overall project architecture and final implementation roadmap.

---

This prompt log documents the primary AI-assisted design decisions used throughout the EngageIQ project and satisfies the BAX-423 requirement to disclose significant AI-generated assistance during development.
