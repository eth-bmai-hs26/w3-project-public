"""
clip_utils.py
─────────────
CLIP model loading (once at startup), RAG retrieval, and image-text scoring.

Both retrieve_relevant_fashion_data() and score_image_text_alignment() use the
same model and the same projection path so they share a vector space — this is
intentional and must be preserved if you swap models.
"""

import math
import numpy as np
import torch
import pandas as pd
from PIL import Image as PILImage
from transformers import CLIPTokenizer, CLIPTextModel, CLIPModel, CLIPProcessor

from config import (
    FASHION_CLIP_NAME, CLIP_SCORE_THRESHOLD,
    COMBINED_CSV, EMBEDDINGS_PATH, device,
)

# ── Load models once at startup ───────────────────────────────────────────────
print(f"Loading fashion-CLIP model on {device} (MPS disabled for this model)...")

fashion_clip_tokenizer  = CLIPTokenizer.from_pretrained(FASHION_CLIP_NAME)
fashion_clip_text_model = CLIPTextModel.from_pretrained(FASHION_CLIP_NAME).to(device)
fashion_clip_full_model = CLIPModel.from_pretrained(FASHION_CLIP_NAME).to(device)
fashion_clip_processor  = CLIPProcessor.from_pretrained(FASHION_CLIP_NAME)

fashion_clip_text_model.eval()
fashion_clip_full_model.eval()
print("fashion-CLIP loaded.")

# ── Load dataset and embeddings once at startup ───────────────────────────────
print("Loading fashion dataset and embeddings...")
try:
    df_fashion      = pd.read_csv(COMBINED_CSV)
    EMBEDDINGS      = np.load(EMBEDDINGS_PATH).astype(np.float32)
    norms           = np.linalg.norm(EMBEDDINGS, axis=1, keepdims=True)
    norms           = np.where(norms < 1e-8, 1.0, norms)
    EMBEDDINGS_NORM = EMBEDDINGS / norms
    if np.any(np.isnan(EMBEDDINGS_NORM)) or np.any(np.isinf(EMBEDDINGS_NORM)):
        print("WARNING: embeddings contain nan/inf — re-run dataset_preprocessing.py!")
    print(f"Loaded {len(df_fashion)} fashion entries, embeddings shape: {EMBEDDINGS.shape}")
    print(f"Embeddings dtype: {EMBEDDINGS.dtype}, norm range: [{norms.min():.3f}, {norms.max():.3f}]")
except Exception as e:
    print(f"Warning: Could not load fashion data or embeddings: {e}")
    df_fashion      = pd.DataFrame({"description": []})
    EMBEDDINGS_NORM = np.array([])


# ── Public helpers ────────────────────────────────────────────────────────────

def sanitise_scores(scores: dict) -> dict:
    """
    Replaces nan/inf CLIP scores with 0.0 before JSON serialisation.
    Python's json.dumps produces bare NaN which is invalid JSON.
    """
    return {
        k: (0.0 if (v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v)))) else float(v))
        for k, v in scores.items()
    }


def retrieve_relevant_fashion_data(query: str, top_k: int = 5) -> str:
    """
    Embeds the query with the same projection path used in dataset_preprocessing.py:
      text_model → pooler_output → text_projection → 512-dim L2-normalised vector.

    Pre-normalised embeddings mean cosine similarity == dot product.
    """
    if EMBEDDINGS_NORM.size == 0:
        return "No fashion data available."

    inputs = fashion_clip_tokenizer(
        query, padding=True, truncation=True, max_length=77, return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        text_out  = fashion_clip_full_model.text_model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )
        projected = fashion_clip_full_model.text_projection(text_out.pooler_output)
        query_vec = projected.squeeze(0).float().cpu().numpy()  # (512,)

    if np.any(np.isnan(query_vec)):
        query_vec = np.nan_to_num(query_vec)

    norm = np.linalg.norm(query_vec)
    if norm < 1e-8 or np.isnan(norm):
        print("[RAG] WARNING: query embedding near-zero — returning top entries by default")
        return "\n".join(f"- {r}" for r in df_fashion["description"].head(top_k).tolist())

    query_vec    = query_vec / norm
    similarities = np.dot(EMBEDDINGS_NORM, query_vec)

    if np.all(np.isnan(similarities)):
        print("[RAG] WARNING: all similarities NaN — returning first entries")
        return "\n".join(f"- {r}" for r in df_fashion["description"].head(top_k).tolist())

    similarities = np.nan_to_num(similarities, nan=0.0)
    top_indices  = np.argsort(similarities)[::-1][:top_k]
    results      = df_fashion.iloc[top_indices]["description"].tolist()
    scores       = similarities[top_indices]

    print(f"[RAG] Query: '{query}' → top {top_k} results:")
    for rank, (desc, sc) in enumerate(zip(results, scores), 1):
        print(f"  [{rank}] score={sc:.3f} | {desc[:90]}")

    return "\n".join(f"- {r}" for r in results)


def score_image_text_alignment(image_path: str, prompt: str) -> float:
    """
    Scores image-text alignment using CLIP cosine similarity.

    Both image and text pass through CLIPModel in one forward pass.
    Score is cosine similarity rescaled from [-1, 1] to [0, 1].
    """
    try:
        image = PILImage.open(image_path).convert("RGB")

        pixel_inputs = fashion_clip_processor(images=image, return_tensors="pt")
        text_inputs  = fashion_clip_tokenizer(
            prompt, padding=True, truncation=True, max_length=77, return_tensors="pt"
        )

        pixel_values = pixel_inputs["pixel_values"].to(device)
        input_ids    = text_inputs["input_ids"].to(device)
        attn_mask    = text_inputs["attention_mask"].to(device)

        with torch.no_grad():
            # Vision path: vision_model → CLS token → visual_projection → 512-dim
            vision_out = fashion_clip_full_model.vision_model(pixel_values=pixel_values)
            img_feats  = fashion_clip_full_model.visual_projection(vision_out.pooler_output)

            # Text path: text_model → EOS token → text_projection → 512-dim
            text_out   = fashion_clip_full_model.text_model(
                input_ids=input_ids, attention_mask=attn_mask
            )
            txt_feats  = fashion_clip_full_model.text_projection(text_out.pooler_output)

        img_norm = img_feats / img_feats.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        txt_norm = txt_feats / txt_feats.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        cosine   = float((img_norm * txt_norm).sum().item())  # [-1, 1]
        score    = (cosine + 1) / 2                           # [0, 1]

        if np.isnan(score) or np.isinf(score):
            print("[CLIP score] NaN/Inf — defaulting to 0.5")
            return 0.5

        print(f"[CLIP score] '{prompt[:50]}' → {score:.3f}")
        return score

    except Exception as e:
        print(f"[CLIP score] Failed: {e}")
        return 1.0