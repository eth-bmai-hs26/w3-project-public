"""
pipelines.py
────────────
High-level orchestration: fresh generation pipeline and feedback/revision pipeline.
These functions coordinate all the lower-level modules and report progress via callback.
"""

import os
import json
import queue
import threading
from datetime import datetime

from flask import Response, stream_with_context

from config import OUTPUT_DIR, CLIP_SCORE_THRESHOLD, LLM_MODEL
# from clip_utils import retrieve_relevant_fashion_data
from bert_utils import retrieve_relevant_fashion_data
from generation import refine_user_inputs, generate_image, generate_article
from html_pipeline import (
    make_generate_fn, safe_html_pipeline, emergency_fallback_html,
)
from prompts import build_fresh_html_prompt, build_feedback_html_prompt
from history import save_history
from clip_utils import sanitise_scores


# ── Fresh Pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(client, title: str, image_styles: list, article_query: str,
                 progress_callback=None):
    def emit(step, total, msg):
        if progress_callback:
            progress_callback(step, total, msg)
        print(f"[{step}/{total}] {msg}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    total = 4 + len(image_styles)

    step = 1
    #emit(step, total, "Refining inputs…")
    refined = refine_user_inputs(client, article_query, image_styles)
    emit(step, total, f"✏️  Intent: '{refined['clean_intent']}'")

    step += 1
    emit(step, total, "Retrieving relevant fashion data…")
    relevant_data = retrieve_relevant_fashion_data(refined["clean_intent"], top_k=5)

    image_filenames = []
    clip_scores     = {}

    for i, style in enumerate(image_styles, 1):
        step += 1
        emit(step, total, f"Generating image {i}/{len(image_styles)}: {style[:50]}…")
        try:
            fname, score = generate_image(client, title, style)
            image_filenames.append(fname)
            clip_scores[fname] = score
            emit(step, total, f"📊 Image {i} CLIP score: {score:.2f} {'✅' if score >= CLIP_SCORE_THRESHOLD else '⚠️ low'}")
        except Exception as e:
            emit(step, total, f"⚠️ Image {i} failed: {e}")

    step += 1
    emit(step, total, "Writing article…")
    article_text = generate_article(client, article_query, relevant_data)  

    step += 1
    emit(step, total, "Assembling final HTML page…")
    base_prompt = build_fresh_html_prompt(title, article_query, article_text, image_filenames)  
    generate_fn  = make_generate_fn(client, base_prompt, LLM_MODEL)

    final_html, succeeded = safe_html_pipeline(
        generate_fn=generate_fn,
        image_filenames=image_filenames,
        emit=lambda msg: emit(step, total, msg),
    )
    if not succeeded:
        final_html = emergency_fallback_html(title, article_text, image_filenames)

    ts            = datetime.now().strftime("%Y%m%d_%H%M%S")
    html_filename = f"article_{ts}.html"
    with open(os.path.join(OUTPUT_DIR, html_filename), "w") as f:
        f.write(final_html)

    save_history("articles", {
        "title":           title,
        "article_query":   article_query,
        "image_styles":    image_styles,
        "html_filename":   html_filename,
        "image_filenames": image_filenames,
        "clip_scores":     clip_scores,
        "article_preview": article_text[:300],
    })

    emit(step, total, f"✅ Done! Article saved as {html_filename}")
    return {
        "html_filename":   html_filename,
        "image_filenames": image_filenames,
        "clip_scores":     clip_scores,
    }


# ── Feedback Pipeline ──────────────────────────────────────────────────────────

def run_feedback_pipeline(client, text_feedback: str, image_feedbacks: dict,
                           current_filename: str, current_image_filenames: list,
                           title: str, image_styles: list,
                           progress_callback=None):
    def emit(step, total, msg):
        if progress_callback:
            progress_callback(step, total, msg)
        print(f"[{step}/{total}] {msg}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    total = 2 + len(image_feedbacks) + 1

    step = 1
    emit(step, total, "Loading existing article…")
    html_path = os.path.join(OUTPUT_DIR, current_filename)
    with open(html_path, encoding="utf-8") as f:
        existing_html = f.read()

    updated_image_map     = {}
    final_image_filenames = list(current_image_filenames)
    clip_scores           = {}

    # Collect indices that have feedback
    feedbacks_to_refine = {
        int(idx): fb
        for idx, fb in image_feedbacks.items()
        if fb and fb.strip()
    }

    if feedbacks_to_refine:
        step += 1
        emit(step, total, f"Refining {len(feedbacks_to_refine)} image feedback(s)…")
        feedback_styles = [
            image_feedbacks.get(str(idx), image_styles[idx - 1])
            for idx in sorted(feedbacks_to_refine.keys())
        ]
        refined     = refine_user_inputs(client, article_query=title, image_styles=feedback_styles)
        refined_map = {
            idx: refined["image_styles"][i]
            for i, idx in enumerate(sorted(feedbacks_to_refine.keys()))
        }
    else:
        refined_map = {}

    for idx, img_feedback in image_feedbacks.items():
        step += 1
        idx = int(idx)

        if not img_feedback or not img_feedback.strip():
            emit(step, total, f"↩ Image {idx} — no feedback, keeping original.")
            continue

        new_style = refined_map.get(idx, img_feedback)
        emit(step, total, f"Regenerating image {idx}: {new_style[:50]}…")
        try:
            old_filename = current_image_filenames[idx - 1]
            new_filename, score = generate_image(client, title, new_style)
            updated_image_map[old_filename]      = new_filename
            final_image_filenames[idx - 1]       = new_filename
            clip_scores[new_filename]             = score
            emit(step, total, f"📊 Image {idx} CLIP score: {score:.2f} {'✅' if score >= CLIP_SCORE_THRESHOLD else '⚠️ low'}")
        except Exception as e:
            emit(step, total, f"⚠️ Image {idx} regeneration failed: {e}")

    step += 1
    emit(step, total, "Applying feedback to article…")
    base_prompt = build_feedback_html_prompt(existing_html, text_feedback, updated_image_map)
    generate_fn = make_generate_fn(client, base_prompt, LLM_MODEL)

    final_html, succeeded = safe_html_pipeline(
        generate_fn=generate_fn,
        image_filenames=final_image_filenames,
        emit=lambda msg: emit(step, total, msg),
    )
    if not succeeded:
        emit(step, total, "🚨 Max retries reached — keeping original article.")
        final_html = existing_html

    ts            = datetime.now().strftime("%Y%m%d_%H%M%S")
    html_filename = f"article_{ts}_revised.html"
    with open(os.path.join(OUTPUT_DIR, html_filename), "w") as f:
        f.write(final_html)

    save_history("articles", {
        "title":            title,
        "article_query":    f"[FEEDBACK] {text_feedback}",
        "image_styles":     image_styles,
        "html_filename":    html_filename,
        "image_filenames":  final_image_filenames,
        "clip_scores":      clip_scores,
        "article_preview":  text_feedback[:300],
        "revised_from":     current_filename,
    })

    emit(step, total, f"✅ Feedback applied! Saved as {html_filename}")
    return {
        "html_filename":   html_filename,
        "image_filenames": final_image_filenames,
        "clip_scores":     clip_scores,
    }


# ── SSE Stream Helper ──────────────────────────────────────────────────────────

def stream_pipeline(pipeline_fn):
    """
    Runs pipeline_fn in a background thread, streaming progress via SSE.
    pipeline_fn receives a progress_callback(step, total, message).
    """
    q = queue.Queue()

    def cb(step, total, message):
        pct = int((step / total) * 100)
        q.put(json.dumps({
            "type": "progress", "step": step,
            "total": total, "pct": pct, "message": message,
        }))

    def run():
        try:
            result = pipeline_fn(cb)
            q.put(json.dumps({
                "type":          "done",
                "html_filename": result["html_filename"],
                "images":        result["image_filenames"],
                "clip_scores":   sanitise_scores(result.get("clip_scores", {})),
            }))
        except Exception as ex:
            q.put(json.dumps({"type": "error", "message": str(ex)}))
        finally:
            q.put(None)

    threading.Thread(target=run, daemon=True).start()

    def generate():
        while True:
            item = q.get()
            if item is None:
                break
            yield f"data: {item}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )