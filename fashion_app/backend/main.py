"""
main.py
───────
Flask application — routes only.
All business logic lives in the other modules:
  config.py        → constants and device setup
  clip_utils.py    → CLIP model, RAG retrieval, image scoring
  generation.py    → image generation, article writing, input refinement
  html_pipeline.py → HTML validation, patching, retry loop
  pipelines.py     → run_pipeline, run_feedback_pipeline, SSE streaming
  prompts.py       → all LLM prompt strings
  history.py       → save/load session history
  cv_pipeline.py   → YOLOv8 object detection and image editing
"""

import os
import openai
import requests
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS

from config import API_KEY, BASE_URL, OUTPUT_DIR, HISTORY_DIR
from history import load_history
from pipelines import run_pipeline, run_feedback_pipeline, stream_pipeline
from cv_pipeline import detect_and_blur, list_detected_objects, detect_and_remove, detect_and_inpaint

app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), "..", "templates"))
CORS(app)

# Single shared client — stateless, safe to reuse across requests.
# Created from .env if a key is there, otherwise when the user enters one in the UI.
# The key is kept in memory only and never written to disk.
_state = {"client": openai.OpenAI(api_key=API_KEY, base_url=BASE_URL) if API_KEY else None}

NO_KEY_ERROR = "No OpenRouter API key set. Enter your key in the sidebar first."


# ── API key ───────────────────────────────────────────────────────────────────

@app.route("/api-key", methods=["GET"])
def api_key_status():
    return jsonify({"configured": _state["client"] is not None})


@app.route("/api-key", methods=["POST"])
def set_api_key():
    key = (request.json or {}).get("api_key", "").strip()
    if not key:
        return jsonify({"error": "Please enter an API key."}), 400

    # Validate against OpenRouter (free call, uses no credits)
    try:
        r = requests.get(f"{BASE_URL}/key", headers={"Authorization": f"Bearer {key}"}, timeout=15)
    except requests.RequestException:
        return jsonify({"error": "Could not reach OpenRouter. Check your internet connection."}), 502
    if r.status_code == 401:
        return jsonify({"error": "OpenRouter rejected this key. Please check it and try again."}), 401
    if not r.ok:
        return jsonify({"error": f"OpenRouter returned an error ({r.status_code})."}), 502

    _state["client"] = openai.OpenAI(api_key=key, base_url=BASE_URL)
    return jsonify({"configured": True})


# ── Pages ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("fashion_magazine.html")


@app.route("/article/<filename>")
def view_article(filename):
    html_path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(html_path):
        return render_template("error.html",
                               error_message=f"Article file '{filename}' not found."), 404
    with open(html_path, encoding="utf-8") as f:
        article_html = f.read()
    title = filename.replace(".html", "").replace("_", " ").title()
    return render_template("article.html", title=title, article_html=article_html)


@app.route("/article-html/<filename>")
def article_html_raw(filename):
    html_path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(html_path):
        return "Article not found", 404
    with open(html_path, encoding="utf-8") as f:
        return f.read(), 200, {"Content-Type": "text/html"}


# ── Generation ────────────────────────────────────────────────────────────────

@app.route("/generate", methods=["POST"])
def generate():
    data          = request.json
    title         = data.get("title", "").strip()
    image_styles  = [s.strip() for s in data.get("image_styles", []) if s.strip()]
    article_query = data.get("article_query", "").strip()

    if not title or not article_query:
        return jsonify({"error": "Title and article query are required."}), 400

    client = _state["client"]
    if client is None:
        return jsonify({"error": NO_KEY_ERROR}), 400

    return stream_pipeline(
        lambda cb: run_pipeline(client, title, image_styles, article_query, progress_callback=cb)
    )


@app.route("/feedback", methods=["POST"])
def feedback():
    data                    = request.json
    text_feedback           = data.get("text_feedback", "").strip()
    image_feedbacks         = data.get("image_feedbacks", {})
    current_filename        = data.get("current_filename", "").strip()
    current_image_filenames = data.get("current_image_filenames", [])
    title                   = data.get("title", "").strip()
    image_styles            = data.get("image_styles", [])

    if not current_filename:
        return jsonify({"error": "Current article filename is required."}), 400

    html_path = os.path.join(OUTPUT_DIR, current_filename)
    if not os.path.exists(html_path):
        return jsonify({"error": f"Article '{current_filename}' not found."}), 404

    client = _state["client"]
    if client is None:
        return jsonify({"error": NO_KEY_ERROR}), 400

    return stream_pipeline(
        lambda cb: run_feedback_pipeline(
            client, text_feedback, image_feedbacks, current_filename,
            current_image_filenames, title, image_styles,
            progress_callback=cb,
        )
    )


# ── History ───────────────────────────────────────────────────────────────────

@app.route("/history/<kind>")
def get_history(kind):
    if kind not in ("articles", "images", "text"):
        return jsonify({"error": "Invalid history type"}), 400
    return jsonify(load_history(kind))


@app.route("/delete", methods=["POST"])
def delete_history():
    deleted = []
    for kind in ["articles", "images", "text"]:
        path = os.path.join(HISTORY_DIR, f"{kind}.json")
        if os.path.exists(path):
            os.remove(path)
            deleted.append(f"{kind}.json")
    return jsonify({"success": True, "deleted": deleted})


# ── Static output ─────────────────────────────────────────────────────────────

@app.route("/output/<path:filename>")
def serve_output(filename):
    return send_from_directory(OUTPUT_DIR, filename)


# ── CV routes ─────────────────────────────────────────────────────────────────

@app.route("/detect", methods=["POST"])
def detect():
    data     = request.json
    filename = data.get("filename", "").strip()
    if not filename:
        return jsonify({"error": "filename required"}), 400

    image_path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(image_path):
        return jsonify({"error": f"Image '{filename}' not found"}), 404

    return jsonify({"objects": list_detected_objects(image_path)})


@app.route("/blur", methods=["POST"])
def blur():
    data         = request.json
    filename     = data.get("filename", "").strip()
    target_label = data.get("label", "").strip()
    blur_radius  = int(data.get("blur_radius", 20))

    if not filename or not target_label:
        return jsonify({"error": "filename and label are required"}), 400

    image_path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(image_path):
        return jsonify({"error": f"Image '{filename}' not found"}), 404

    output_path, success = detect_and_blur(image_path, target_label, blur_radius=blur_radius)

    if not success:
        return jsonify({"success": False, "message": f"'{target_label}' not detected in image."}), 200

    return jsonify({
        "success":  True,
        "filename": os.path.basename(output_path),
        "message":  f"'{target_label}' blurred successfully.",
    })


@app.route("/inpaint", methods=["POST"])
def inpaint():
    data         = request.json
    filename     = data.get("filename", "").strip()
    target_label = data.get("label", "").strip()

    if not filename or not target_label:
        return jsonify({"error": "filename and label are required"}), 400

    image_path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(image_path):
        return jsonify({"error": f"Image '{filename}' not found"}), 404

    output_path, success = detect_and_inpaint(image_path, target_label)
    if not success:
        return jsonify({"success": False, "message": f"'{target_label}' not detected in image."})

    return jsonify({
        "success":  True,
        "filename": os.path.basename(output_path),
        "message":  f"'{target_label}' removed via inpainting.",
    })


@app.route("/remove", methods=["POST"])
def remove():
    data         = request.json
    filename     = data.get("filename", "").strip()
    target_label = data.get("label", "").strip()

    if not filename or not target_label:
        return jsonify({"error": "filename and label are required"}), 400

    image_path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(image_path):
        return jsonify({"error": f"Image '{filename}' not found"}), 404

    output_path, success = detect_and_remove(image_path, target_label)
    if not success:
        return jsonify({"success": False, "message": f"'{target_label}' not detected in image."})

    return jsonify({
        "success":  True,
        "filename": os.path.basename(output_path),
        "message":  f"'{target_label}' filled with white.",
    })


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=True, port=5001)