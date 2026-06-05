# EngageIQ — Smart Engagement Opportunity Scorer

BAX-423 Big Data · Spring 2026 · Final Project · Saliha Askin Sonmez

EngageIQ is an end-to-end ML pipeline that discovers, scores, and learns from high-value engagement opportunities across GitHub and Hacker News. It uses local Sentence-BERT embeddings, a FAISS vector index, and multi-signal ranking, presented through an interactive Streamlit dashboard. Users can engage, bookmark, or skip items; the system adapts its rankings in real time based on that feedback.

---

## System Architecture

The pipeline has six stages:

| Stage | Description |
|---|---|
| 1 — Multi-Source Ingestion | GitHub REST API v3 and Hacker News API collectors; records deduplicated by (source, id) and URL |
| 2 — Merge & Feature Engineering | 9-stage `merge.py`: schema standardization → dedup → domain/topic classification (15 domains) → feature engineering → snapshot export (15,653 records) |
| 3 — Content Embedding | `generate_embeddings.py` encodes title + description + domain + source as a 384-dim SBERT vector (`all-MiniLM-L6-v2`) |
| 4 — FAISS Retrieval & Ranking | `retrieve_candidates.py` builds FAISS IndexFlatL2; `ranking.py` applies weighted score: similarity 0.45 + engagement 0.20 + freshness 0.15 + trend 0.10 + effort 0.10; MMR diversity re-ranking |
| 5 — Adaptive Learning | `adaptive_learning.py` updates the user embedding vector after each feedback event (engage/bookmark/skip); NDCG@10 tracked across 50+ rounds |
| 6 — Batch Analytics & Dashboard | `analytics_spark.py` runs PySpark aggregations; Streamlit dashboard integrates FAISS pipeline at runtime |

---

## Folder Structure

```
finalhw/
├── README.md
├── brief.pdf                         # Technical brief
├── prompts.md                        # Prompt templates used during development
│
├── code/
│   ├── app.py                        # Main Streamlit dashboard (all UI logic)
│   ├── streamlit_app.py              # Streamlit entry point (delegates to app.py)
│   ├── streaming_pipeline.py         # Kafka-style real-time ingestion simulator
│   ├── requirements.txt              # Python dependencies
│   │
│   ├── embedding_common.py           # Shared SBERT model loader
│   ├── generate_embeddings.py        # Opportunity → JSONL embeddings
│   ├── generate_user_embeddings.py   # User profile → JSONL embeddings
│   ├── user_profile_embeddings.py    # User profile embedding helpers
│   ├── retrieve_candidates.py        # FAISS index build + ANN search
│   ├── rank_candidates.py            # Candidate ranking helpers
│   ├── ranking.py                    # Scoring + diversity re-ranking
│   ├── adaptive_learning.py          # Feedback-weight update loop
│   ├── explainability.py             # "Why this?" reason generation
│   ├── recommendation_pipeline.py    # End-to-end offline pipeline
│   ├── evaluate_ranking.py           # NDCG@10 evaluation script
│   ├── evaluation.py                 # Evaluation helpers
│   ├── generate_evaluation_labels.py # Proxy label generation
│   ├── analytics_spark.py            # PySpark trend aggregations
│   │
│   └── data collect/
│       ├── collect_github_events.py      # GitHub Events API collector
│       ├── collect_hackernews_posts.py   # Hacker News Algolia API collector
│       ├── collect_beginner_coding.py    # Beginner-friendly issue collector
│       └── merge.py                      # Schema normalisation + merge
│
└── data/
    ├── scored_opportunities.csv
    ├── trends.csv
    ├── ranking_evaluation.sample.json
    ├── ranking_evaluation_labels.csv
    ├── user_123_candidates.csv
    ├── user_123_pipeline_candidates.csv
    ├── user_123_recommendations.csv
    ├── user-alice_recommendations.csv
    │
    ├── main data/
    │   ├── finaldataset.csv              # Final merged dataset (15,653 records)
    │   └── finaldataset.csv.metadata.json
    │
    └── analytics/
        ├── active_communities.csv
        ├── category_distribution.csv
        ├── domain_distribution.csv
        ├── opportunity_volume_over_time.csv
        └── trending_topics.csv
```

---

## Installation

### Prerequisites

- Python 3.10 or later
- pip

### 1 — Create a virtual environment

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2 — Install dependencies

```powershell
pip install -r code/requirements.txt
```

---

## Required Dependencies

| Package | Purpose |
|---|---|
| `streamlit>=1.35` | Dashboard UI |
| `plotly>=5.20` | Interactive charts |
| `pandas>=2.0` | Data manipulation |
| `pyarrow>=14.0` | Parquet read/write |
| `numpy>=1.24` | Numerical operations |
| `sentence-transformers>=3.0` | SBERT embedding model |
| `faiss-cpu>=1.7.4` | Approximate nearest-neighbor index |
| `scikit-learn>=1.3` | Cosine similarity helpers |
| `python-dotenv>=1.0` | `.env` file loading |
| `requests>=2.31` | HTTP calls (HN, GitHub APIs) |
| `pyspark>=3.5` | Batch analytics |

> **Note:** `sentence-transformers` and `faiss-cpu` are optional for the dashboard. If they cannot be imported the app automatically falls back to keyword-based ranking.

---

## Running the Project from Scratch

Follow these steps **in order** the first time you set up the project.

### Step 1 — Collect raw data (optional — skip if using pre-built dataset)

Run each collector from the **project root**:

```powershell
python "code/data collect/collect_github_events.py"
python "code/data collect/collect_hackernews_posts.py"
python "code/data collect/collect_beginner_coding.py"
```

> The pre-collected dataset is already included at `data/main data/finaldataset.csv`. Skip this step to use it.

### Step 2 — Merge and normalise raw data (optional)

```powershell
python "code/data collect/merge.py"
```

> Skip this step if you are using the pre-built dataset.

### Step 3 — Generate opportunity embeddings

```powershell
python code/generate_embeddings.py `
  --opportunities "data/main data/finaldataset.csv" `
  --output-dir data/
```
### Step 4 — Generate user profile embeddings (optional — for offline evaluation only)

```powershell
python code/generate_user_embeddings.py `
  --users data/user_profiles.sample.json `
  --output-dir data/
```

### Step 5 — Launch the Streamlit dashboard

```powershell
streamlit run code/app.py
```

Open your browser at **http://localhost:8501**.

---

## Dashboard Overview

**Tabs:**

| Tab | What it shows |
|---|---|
| Recommendations | Ranked opportunity cards with "Why this?" explanations and writing prompts |
| Topics | Grouped topic summary with average scores |
| Writing | Table of suggested reply starters |
| Analytics | Topic bar chart, global trends pie, week-over-week volume change, CSV export |

**Sidebar controls:**

- Select a **persona** (Sofia ML Student, David DevOps, Lina Journalist, Raj Founder) or use "Custom new user" to pre-fill interests and domains.
- Edit profile fields and click **Update recommendations**.
- Use **Good first issues only** to filter to beginner-friendly GitHub issues.
- Click **Engage**, **Skip**, or **Bookmark** on any card. Skipping triggers adaptive re-ranking for that domain.

---

## Offline Recommendation Pipeline

To run the full offline pipeline for a single user:

```powershell
python code/recommendation_pipeline.py `
  --user-id user-alice `
  --opportunities "data/main data/finaldataset.csv" `
  --opportunity-embeddings data/opportunity_embeddings.jsonl `
  --user-embeddings data/user_embeddings.jsonl
```

Results are saved to `data/`.

---

## Real-Time Streaming Pipeline (optional)

```powershell
# Run for 60 seconds:
python code/streaming_pipeline.py --duration 60 --sources github hn

# Write to a custom output file:
python code/streaming_pipeline.py --output data/live_feed.jsonl
```

---

## Batch Analytics (PySpark)

```powershell
python code/analytics_spark.py
```

Outputs are written to `data/analytics/`: active communities, category distributions, domain distributions, trending topics, and opportunity volume over time.

---

## Evaluation Results

### Offline Ranking Quality — NDCG@10

| Model Variant | NDCG@10 |
|---|---|
| Similarity Only (baseline) | 0.7173 |
| Similarity + Engagement | 0.3278 |
| Full Scoring Model | 0.4095 |
| Full Model + MMR Re-ranking | 0.4095 |
| Post-pipeline (reranked_score) | **0.9012** |

> The drop from similarity-only reflects proxy label design — proxy labels weight similarity at 0.5, so adding orthogonal signals lowers proxy alignment while improving real diversity. The post-pipeline NDCG@10 of 0.9012 confirms the full pipeline ranks high-quality items first.

### Adaptive Learning (50 rounds, user-alice)

| Metric | Value |
|---|---|
| Initial NDCG@10 | 0.8524 |
| Final NDCG@10 | **1.0000** |
| Improvement | +17.32% |
| Feedback distribution | engage: 40, bookmark: 6, skip: 4 |

---

## Technique Choices

### Technique 1 — Content Embeddings + FAISS

Sentence-BERT produces 384-dim contextual embeddings placing opportunity text and user profiles in a shared vector space. FAISS IndexFlatL2 provides exact L2 nearest-neighbor search at millisecond latency over 15,653 vectors.

- Model: `sentence-transformers/all-MiniLM-L6-v2`
- Fallback: NumPy L2 search when FAISS is unavailable

### Technique 2 — Multi-Stage Ranking with Diversity Re-Ranking

Ranking weights: similarity 0.45 | engagement 0.20 | freshness 0.15 | trend 0.10 | effort 0.10

- Simple re-rank: +0.05 bonus to first candidate per domain
- MMR: Greedy Maximal Marginal Relevance (lambda=0.7)

### Technique 3 — Adaptive Learning from Feedback

Update rule: `new_vector = normalize(w_user × user_vec + w_item × item_vec)`

| Action | w_user | w_item |
|---|---|---|
| engage | 0.90 | +0.10 |
| bookmark | 0.93 | +0.07 |
| skip | 0.95 | −0.05 |

---

## System Limitations

- **Beginner Coding underrepresentation:** Only ~30 records (0.2% of dataset). Fix: target r/learnprogramming, r/cs50, and GitHub "good first issue" endpoint.
- **No live streaming:** Current collectors are batch pollers. Production would use Kafka topics with GitHub webhooks.
- **Proxy NDCG labels:** Evaluation uses similarity-weighted proxy labels rather than human judgments, inflating similarity-only NDCG.
- **User embedding cold start:** New users incur ~200ms SBERT encoding on first query, cached after.
- **Single-user adaptive learning:** Updates one user vector at a time; vectors are not persisted across sessions.

---

## Technologies Used

| Layer | Technology |
|---|---|
| Frontend / UI | Streamlit, Plotly |
| Embedding model | Sentence-BERT (`all-MiniLM-L6-v2`) |
| Vector search | FAISS (CPU) |
| Data pipeline | Python `threading`, `queue` |
| Batch analytics | PySpark |
| Data formats | CSV, JSONL, PyArrow |
| Data sources | GitHub REST API v3, GH Archive, Hacker News API |
| Evaluation | NDCG@10, adaptive learning traces |

---

## Publishing to GitHub

Run all commands from inside the `finalhw` folder.

```powershell
git init
git add .
git commit -m "Initial commit: EngageIQ final project"
git remote add origin https://github.com/<your-username>/EngageIQ.git
git branch -M main
git push -u origin main
```

When prompted for a password use a **Personal Access Token** (not your GitHub password): Settings → Developer settings → Personal access tokens → Generate new token → tick `repo` scope.

---

## License

Created as a final course assignment for BAX-423. Data collected from GitHub and Hacker News is subject to the respective platforms' terms of service.
