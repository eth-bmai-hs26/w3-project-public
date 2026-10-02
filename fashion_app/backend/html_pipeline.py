"""
html_pipeline.py
────────────────
HTML validation, truncation patching, image path fixing, and the
safe retry loop that wraps every LLM HTML generation call.
"""

import re
from flask import Response, stream_with_context


# ── Validation ────────────────────────────────────────────────────────────────

def validate_html(html: str, image_filenames: list[str]) -> tuple[bool, list[str], str]:
    """
    Minimal validation — only checks things that genuinely break the page:
      1. Structural tags present (html, body, style)
      2. All generated images are referenced
      3. CSS brace balance
      4. Heading tags balanced (h1-h6) — most common LLM truncation symptom

    Deliberately skips div/section/article nesting — those mismatches are
    cosmetic and were causing false-positive retries on valid HTML.
    """
    errors = []

    # Strip markdown fences the LLM sometimes wraps output in
    s = html.strip()
    if s.startswith("```"):
        lines_h = s.split("\n")[1:]
        if lines_h and lines_h[-1].strip().startswith("```"):
            lines_h = lines_h[:-1]
        html = "\n".join(lines_h)

    # 1. Must-have structural tokens
    for token, msg in [
        ("<!DOCTYPE html>", "missing <!DOCTYPE html>"),
        ("<html",           "missing <html> tag"),
        ("</html>",         "missing </html> tag"),
        ("<body",           "missing <body> tag"),
        ("</body>",         "missing </body> tag"),
        ("<style",          "missing <style> tag"),
    ]:
        if token not in html:
            errors.append(msg)

    # 2. Every generated image must appear in the HTML
    for f in image_filenames:
        if f not in html:
            errors.append(f"image '{f}' not referenced in HTML")

    # 3. CSS brace balance
    style_blocks = re.findall(r"<style[^>]*>(.*?)</style>", html, re.DOTALL | re.IGNORECASE)
    if style_blocks:
        css = "\n".join(style_blocks)
        opens, closes = css.count("{"), css.count("}")
        if opens != closes:
            errors.append(f"CSS brace mismatch: {opens} open vs {closes} close")

    # 4. Heading balance (h1–h6) — LLMs often truncate mid-heading
    for n in range(1, 7):
        opens  = len(re.findall(rf"<h{n}[\s>]", html, re.IGNORECASE))
        closes = len(re.findall(rf"</h{n}>",     html, re.IGNORECASE))
        if opens != closes:
            errors.append(f"<h{n}> mismatch: {opens} open, {closes} closed")

    return (len(errors) == 0, errors, html)


def patch_truncated_html(html: str) -> str:
    """
    Closes tags left open by the LLM due to token-limit truncation.

    Counts open vs close occurrences per tag and appends the difference.
    Tags are closed in inside-out order for valid HTML.
    """
    html = html.strip()

    def open_count(tag):
        return len(re.findall(rf"<{tag}[\s>]", html, re.IGNORECASE))

    def close_count(tag):
        return len(re.findall(rf"</{tag}>", html, re.IGNORECASE))

    # Inner tags first, then outer
    for tag in ["li", "ul", "ol", "thead", "tbody", "tr", "td", "th",
                "article", "section", "nav", "header", "footer", "main"]:
        diff = open_count(tag) - close_count(tag)
        if diff > 0:
            html += f"\n</{tag}>" * diff

    div_diff = html.count("<div") - html.count("</div>")
    if div_diff > 0:
        html += "\n</div>" * div_diff

    if "</body>" not in html:
        html += "\n</body>"
    if "</html>" not in html:
        html += "\n</html>"

    return html


def fix_image_paths(html: str, image_filenames: list[str]) -> str:
    """Normalises all image src variants to /output/{filename}."""
    for filename in image_filenames:
        wrong_variants = [
            f'src="{filename}"',          f"src='{filename}'",
            f'src="output/{filename}"',   f"src='output/{filename}'",
            f'src="./output/{filename}"', f"src='./output/{filename}'",
            f'src="../output/{filename}"',f"src='../output/{filename}'",
        ]
        correct = f'src="/output/{filename}"'
        for wrong in wrong_variants:
            html = html.replace(wrong, correct)
    return html


def emergency_fallback_html(title: str, article_text: str, image_filenames: list[str]) -> str:
    """Minimal valid HTML page used when all LLM attempts fail validation."""
    imgs = "\n".join(
        f'<img src="/output/{f}" alt="Fashion image" style="max-width:100%;margin:1rem 0;">'
        for f in image_filenames
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{title}</title>
<style>
  body {{ font-family: Georgia, serif; max-width: 800px; margin: 40px auto; padding: 0 20px; color: #333; }}
  h1 {{ font-size: 2rem; border-bottom: 2px solid #ccc; padding-bottom: 0.5rem; }}
  p {{ line-height: 1.8; }}
</style>
</head>
<body>
<h1>{title}</h1>
{imgs}
<div>{article_text.replace(chr(10), '<br>')}</div>
</body>
</html>"""


# ── Safe retry loop ───────────────────────────────────────────────────────────

def make_generate_fn(client, base_prompt: str, model: str, max_tokens: int = 4096):
    """
    Returns a generate_fn(attempt, previous_errors) closure.

    Extracted to avoid duplicating this closure in both run_pipeline
    and run_feedback_pipeline.
    """
    from prompts import inject_validation_errors

    def generate_fn(attempt: int, previous_errors: list[str]) -> str:
        prompt = inject_validation_errors(base_prompt, previous_errors)
        return client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
        ).choices[0].message.content

    return generate_fn


def safe_html_pipeline(
    generate_fn,
    image_filenames: list[str],
    emit=None,
    max_attempts: int = 3,
) -> tuple[str, bool]:
    """
    Calls generate_fn up to max_attempts times, validating and patching
    HTML on each attempt. Returns (html, succeeded).
    """
    last_errors = []
    last_html   = ""

    for attempt in range(1, max_attempts + 1):
        if emit:
            emit(f"HTML generation attempt {attempt}/{max_attempts}…")

        raw   = generate_fn(attempt, last_errors)
        fixed = fix_image_paths(raw, image_filenames)
        fixed = patch_truncated_html(fixed)
        valid, errors, cleaned = validate_html(fixed, image_filenames)

        if valid:
            if emit:
                emit("✅ HTML validated successfully.")
            return cleaned, True

        last_errors = errors
        last_html   = cleaned
        if emit:
            emit(f"⚠️ Validation failed ({len(errors)} errors), retrying with corrections…")
        print(f"[Attempt {attempt}/{max_attempts}] Errors: {errors}")

    if emit:
        emit("🚨 Max retries reached — using emergency fallback HTML.")
    return last_html, False