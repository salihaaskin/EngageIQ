"""
Streamlit Cloud / HuggingFace Spaces entry point.

Streamlit Cloud requires a file named streamlit_app.py (or app.py) at the
repo root. This file simply delegates to App/app.py so the repo structure
stays intact while cloud deployment works automatically.

Deploy steps (Streamlit Community Cloud):
  1. Push this repo to GitHub (make sure Embeddings/snapshots/finaldataset.csv
     is included — it is ~13 MB, well within GitHub's 100 MB limit).
  2. Go to https://share.streamlit.io → "New app"
  3. Repo: <your-github-username>/EngageIQ
  4. Main file path: streamlit_app.py
  5. Click Deploy.

The FAISS + SBERT engine requires ~600 MB RAM. On Streamlit's free tier the
app will automatically fall back to the keyword ranking engine if sentence-
transformers cannot be loaded. All persona features and analytics remain
fully functional on the keyword fallback path.
"""

import runpy
import sys
from pathlib import Path

# Make sure App/ and Embeddings/ are on the path
_root = Path(__file__).resolve().parent
for _sub in ("App", "Embeddings"):
    _p = str(_root / _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Run the main app module
runpy.run_path(str(_root / "app.py"), run_name="__main__")
