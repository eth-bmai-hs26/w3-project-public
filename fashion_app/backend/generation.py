"""
generation.py
─────────────
Image generation (OpenRouter Image API + CLIP quality check) and article writing.
"""

import os
import base64
import requests
from datetime import datetime
import openai

from config import (
    OUTPUT_DIR, CLIP_SCORE_THRESHOLD,
    LLM_MODEL, IMAGE_MODEL, BASE_URL,
)
from clip_utils import score_image_text_alignment
from history import save_history
from prompts import build_refinement_prompt, build_article_prompt, build_image_prompt

# ── Image generation ──────────────────────────────────────────────────────────
def _request_image(client, prompt: str) -> str:
    """
    Calls OpenRouter's Image API (POST /images) and returns the base64 image.
    The OpenAI SDK's images.generate() targets /images/generations, which
    OpenRouter does not serve, so the request is made directly.
    """
    r = requests.post(
        f"{BASE_URL}/images",
        headers={"Authorization": f"Bearer {client.api_key}"},
        json={"model": IMAGE_MODEL, "prompt": prompt, "aspect_ratio": "1:1", "n": 1},
        timeout=180,
    )
    if not r.ok:
        raise RuntimeError(f"Image generation failed ({r.status_code}): {r.text[:300]}")
    return r.json()["data"][0]["b64_json"]


def generate_image(client, title: str, style: str, max_attempts: int = 2) -> tuple[str, float]:
    """
    Generates one image via OpenRouter, saves to disk.
    Scores alignment with CLIP. Retries once if score is below threshold.
    Returns (filename, clip_score).
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    last_filename = ""
    last_score    = 0.0

    for attempt in range(1, max_attempts + 1):
        prompt = build_image_prompt(title, style, attempt)

        b64 = _request_image(client, prompt)

        ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"image_{ts}_{style[:20].replace(' ', '_')}.png"
        path     = os.path.join(OUTPUT_DIR, filename)

        with open(path, "wb") as f:
            f.write(base64.b64decode(b64))

        clip_score    = score_image_text_alignment(path, style)
        last_filename = filename
        last_score    = clip_score

        if clip_score >= CLIP_SCORE_THRESHOLD or attempt == max_attempts:
            save_history("images", {
                "title": title, "style": style,
                "filename": filename, "clip_score": clip_score,
            })
            return filename, clip_score

        print(f"[generate_image] Score {clip_score:.3f} below threshold, retrying...")

    return last_filename, last_score

# ── Article generation ────────────────────────────────────────────────────────

def generate_article(client, query: str, relevant_data: str) -> str:
    """Generates a fashion article grounded in relevant fashion data."""
    prompt = build_article_prompt(query, relevant_data)
    resp   = client.chat.completions.create(
        model=LLM_MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    article_text = resp.choices[0].message.content.strip()
    save_history("text", {"query": query, "article_preview": article_text[:300]})
    return article_text

# ── Input refinement ──────────────────────────────────────────────────────────

def refine_user_inputs(client, article_query: str, image_styles: list[str]) -> dict:
    """
    Single LLM call that refines ALL user inputs before any generation.
    Returns dict with keys: clean_intent, article_query, image_styles.
    """
    if not image_styles:
        return {
            "clean_intent":  article_query,
            "article_query": article_query,
            "image_styles":  [],
        }

    prompt = build_refinement_prompt(article_query, image_styles)
    resp   = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=600,
    )

    raw   = resp.choices[0].message.content.strip()
    lines = raw.split("\n")

    result = {
        "clean_intent":  article_query,
        "article_query": article_query,
        "image_styles":  list(image_styles),
    }

    refined_styles = {}
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("INTENT:"):
            val = line.replace("INTENT:", "").strip()
            if val:
                result["clean_intent"] = val
        elif line.startswith("ARTICLE:"):
            val = line.replace("ARTICLE:", "").strip()
            if val:
                result["article_query"] = val
        elif line.startswith("IMAGE_"):
            try:
                tag, _, val = line.partition(":")
                idx         = int(tag.replace("IMAGE_", "").strip()) - 1
                if val.strip() and idx < len(image_styles):
                    refined_styles[idx] = val.strip()
            except (ValueError, IndexError):
                pass

    result["image_styles"] = [
        refined_styles.get(i, image_styles[i])
        for i in range(len(image_styles))
    ]

    print(f"[refine] intent:  \"{article_query}\" → \"{result['clean_intent']}\"")
    print(f"[refine] article: \"{result['article_query']}\"")
    for i, s in enumerate(result["image_styles"]):
        print(f"[refine] image {i+1}: \"{s}\"")

    return result
