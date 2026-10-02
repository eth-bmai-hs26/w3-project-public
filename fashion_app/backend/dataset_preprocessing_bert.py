"""
dataset_preprocessing_bert.py

Reads Data/fashion_combined.csv, encodes every description with
all-MiniLM-L6-v2, L2-normalises the vectors, and saves:
  Data/bert_embeddings.npy   – float32 array (N, 384)
  Data/bert_descriptions.pkl – list of description strings (length N)

Run once from fashion_app/backend/:
  python dataset_preprocessing_bert.py
"""

import os
import pickle
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# ── Paths ─────────────────────────────────────────────────────────────────────
# This script lives in  fashion_app/backend/
# Data/ lives in        fashion_app/Data/
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))   # .../fashion_app/backend
ROOT_DIR    = os.path.dirname(BACKEND_DIR)                 # .../fashion_app
DATA_DIR    = os.path.join(ROOT_DIR, "Data")
CSV_PATH    = os.path.join(DATA_DIR, "fashion_combined.csv")
EMB_PATH    = os.path.join(DATA_DIR, "bert_embeddings.npy")
DESC_PATH   = os.path.join(DATA_DIR, "bert_descriptions.pkl")

MODEL_NAME  = "all-MiniLM-L6-v2"

def main():
    # 1. Load CSV
    print(f"[BERT-PREP] Loading dataset from {CSV_PATH} ...")
    df = pd.read_csv(CSV_PATH)

    if "description" not in df.columns:
        raise ValueError("Expected a 'description' column in fashion_combined.csv")

    descriptions = df["description"].fillna("").tolist()
    print(f"[BERT-PREP] {len(descriptions)} descriptions found.")

    # 2. Load model
    print(f"[BERT-PREP] Loading model '{MODEL_NAME}' ...")
    model = SentenceTransformer(MODEL_NAME)

    # 3. Encode
    print("[BERT-PREP] Encoding descriptions ...")
    embeddings = model.encode(
        descriptions,
        batch_size=256,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,   # L2-normalise → cosine sim == dot product
    )
    # embeddings shape: (N, 384), float32

    # 4. Save
    np.save(EMB_PATH, embeddings.astype(np.float32))
    with open(DESC_PATH, "wb") as f:
        pickle.dump(descriptions, f)

    print(f"[BERT-PREP] Saved embeddings → {EMB_PATH}  shape={embeddings.shape}")
    print(f"[BERT-PREP] Saved descriptions → {DESC_PATH}  ({len(descriptions)} entries)")
    print("[BERT-PREP] Done.")

if __name__ == "__main__":
    main()