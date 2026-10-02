"""
bert_utils.py

Loads the pre-computed BERT embeddings produced by
dataset_preprocessing_bert.py and exposes:

    retrieve_relevant_fashion_data(query, top_k=5) -> str

Drop-in replacement for the equivalent function in clip_utils.py.
"""

import os
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

# ── Paths ─────────────────────────────────────────────────────────────────────
# bert_utils.py lives in  fashion_app/backend/
# Data/ lives in          fashion_app/Data/
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))   # .../fashion_app/backend
ROOT_DIR    = os.path.dirname(BACKEND_DIR)                 # .../fashion_app
DATA_DIR    = os.path.join(ROOT_DIR, "Data")
EMB_PATH    = os.path.join(DATA_DIR, "bert_embeddings.npy")
DESC_PATH   = os.path.join(DATA_DIR, "bert_descriptions.pkl")

# ── Model (loaded once at import time) ────────────────────────────────────────
MODEL_NAME = "all-MiniLM-L6-v2"

print(f"[BERT] Loading model '{MODEL_NAME}' ...")
_model = SentenceTransformer(MODEL_NAME)
print("[BERT] Model ready.")

# ── Embeddings (loaded once at import time) ───────────────────────────────────
def _load_embeddings():
    if not os.path.exists(EMB_PATH):
        print(f"[BERT] ERROR: embeddings file not found at {EMB_PATH}")
        print("[BERT] Run dataset_preprocessing_bert.py first.")
        return np.array([]), []

    if not os.path.exists(DESC_PATH):
        print(f"[BERT] ERROR: descriptions file not found at {DESC_PATH}")
        print("[BERT] Run dataset_preprocessing_bert.py first.")
        return np.array([]), []

    embeddings = np.load(EMB_PATH)                  # (N, 384) float32
    with open(DESC_PATH, "rb") as f:
        descriptions = pickle.load(f)               # list[str], length N

    print(f"[BERT] Loaded embeddings: shape={embeddings.shape} from {DATA_DIR}")
    print(f"[BERT] Loaded {len(descriptions)} descriptions.")
    return embeddings, descriptions


EMBEDDINGS_NORM, DESCRIPTIONS = _load_embeddings()


# ── Public API ─────────────────────────────────────────────────────────────────
def retrieve_relevant_fashion_data(query: str, top_k: int = 5) -> str:
    """
    Encodes `query` with all-MiniLM-L6-v2, L2-normalises it, and returns
    the top_k most similar fashion descriptions as a newline-separated string.

    Because both the stored embeddings and the query vector are L2-normalised,
    cosine similarity == dot product — so we just use np.dot.
    """
    if EMBEDDINGS_NORM.size == 0:
        return "No fashion data available."

    # Encode & normalise query vector — shape: (384,)
    query_vec = _model.encode(
        query,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    # Sanity checks
    if np.any(np.isnan(query_vec)):
        query_vec = np.nan_to_num(query_vec)

    norm = np.linalg.norm(query_vec)
    if norm < 1e-8:
        print("[BERT] WARNING: query embedding near-zero — returning top entries by default")
        return "\n".join(f"- {d}" for d in DESCRIPTIONS[:top_k])

    # Cosine similarity via dot product (both sides already normalised)
    similarities = np.dot(EMBEDDINGS_NORM, query_vec)   # (N,)

    if np.all(np.isnan(similarities)):
        print("[BERT] WARNING: all similarities NaN — returning first entries")
        return "\n".join(f"- {d}" for d in DESCRIPTIONS[:top_k])

    similarities = np.nan_to_num(similarities, nan=0.0)
    top_indices  = np.argsort(similarities)[::-1][:top_k]
    top_results  = [DESCRIPTIONS[i] for i in top_indices]
    top_scores   = similarities[top_indices]

    print(f"[BERT] Query: '{query}' → top {top_k} results:")
    for rank, (desc, sc) in enumerate(zip(top_results, top_scores), 1):
        print(f"  [{rank}] score={sc:.3f} | {desc[:90]}")

    return "\n".join(f"- {r}" for r in top_results)