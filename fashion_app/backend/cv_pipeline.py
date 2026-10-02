"""
cv_pipeline.py
──────────────
Computer Vision pipeline for object detection, segmentation, blurring and inpainting.

Three operations available:
    detect_and_blur     — Gaussian blur over the segmented object pixels
    detect_and_inpaint  — Classical OpenCV inpainting: removes object, fills with background
    detect_and_remove   — Fills masked region with white (useful to visualise the mask)

Pipeline for all three:
    1. YOLOv8-seg detects objects and produces pixel-level masks
    2. We find the mask matching the user's target label
    3. We apply the chosen operation only to the masked pixels
    4. The rest of the image is untouched
    5. The modified image is saved to disk and its path is returned
"""

import os
import json
import re
import cv2
import numpy as np
from PIL import Image, ImageFilter
from ultralytics import YOLO

# ── Load model once at module import ─────────────────────────────────────────
print("[cv_pipeline] Loading YOLOv8 segmentation model...")
_model = YOLO("yolov8n-seg.pt")
print("[cv_pipeline] Model ready.")

COCO_CLASSES = _model.names  # dict: {0: "person", 1: "bicycle", ...}

# ── Fashion-relevant label whitelist ──────────────────────────────────────────
_WHITELIST_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Data", "fashion_yolo_labels.json"
)
try:
    with open(_WHITELIST_PATH) as f:
        FASHION_LABELS = set(json.load(f))
    print(f"[cv_pipeline] Loaded {len(FASHION_LABELS)} fashion-relevant labels.")
except FileNotFoundError:
    print("[cv_pipeline] Warning: fashion_yolo_labels.json not found — using full COCO set.")
    FASHION_LABELS = set(COCO_CLASSES.values())


# ── Private helpers ───────────────────────────────────────────────────────────

def _build_mask(image_path: str, target_label: str) -> tuple[np.ndarray, Image.Image, bool]:
    """
    Runs YOLOv8 on image_path and builds a combined binary mask for all
    instances of target_label.

    Returns:
        combined_mask  — uint8 array (H, W), 1 = object pixel, 0 = background
        original       — PIL Image (RGB) at original dimensions
        found          — True if target_label was detected at least once
    """
    results  = _model(image_path, verbose=False)
    original = Image.open(image_path).convert("RGB")
    img_w, img_h = original.size

    combined_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    found = False

    for result in results:
        if result.boxes is None or result.masks is None:
            continue
        for box, cls, mask_tensor in zip(result.boxes, result.boxes.cls, result.masks.data):
            if COCO_CLASSES[int(cls)].lower() != target_label.lower():
                continue
            found = True
            mask_np   = mask_tensor.cpu().numpy()
            mask_img  = Image.fromarray((mask_np * 255).astype(np.uint8)).resize(
                (img_w, img_h), Image.NEAREST
            )
            binary    = (np.array(mask_img) > 127).astype(np.uint8)
            combined_mask = np.clip(combined_mask + binary, 0, 1)

    return combined_mask, original, found


def _save(image: Image.Image, source_path: str, suffix: str) -> str:
    """Saves a PIL image next to source_path with the given suffix before the extension."""
    base, ext   = os.path.splitext(source_path)
    output_path = f"{base}{suffix}{ext}"
    image.save(output_path)
    return output_path


# ── Public API ────────────────────────────────────────────────────────────────

def list_detected_objects(image_path: str, min_confidence: float = 0.4) -> list[dict]:
    """
    Runs YOLOv8 on the image and returns detected objects above min_confidence,
    filtered to the fashion-relevant whitelist.
    """
    results  = _model(image_path, verbose=False)
    detected = []

    for result in results:
        if result.boxes is None or result.masks is None:
            continue
        for i, (box, cls) in enumerate(zip(result.boxes, result.boxes.cls)):
            confidence = float(box.conf[0])
            if confidence < min_confidence:
                continue

            label = COCO_CLASSES[int(cls)]
            if label not in FASHION_LABELS:
                print(f"[cv_pipeline] Skipping '{label}' (not in fashion whitelist)")
                continue

            detected.append({
                "label":      label,
                "confidence": round(confidence, 3),
                "warning":    "might be misidentified" if confidence < 0.6 else None,
                "index":      i,
            })

    print(f"[cv_pipeline] Detected in '{os.path.basename(image_path)}': "
          f"{[d['label'] for d in detected]}")
    return detected


def detect_and_blur(image_path: str, target_label: str,
                    blur_radius: int = 20,
                    output_suffix: str = "_blurred") -> tuple[str, bool]:
    """
    Applies Gaussian blur only to the pixels belonging to target_label.
    Returns (output_path, success).
    """
    combined_mask, original, found = _build_mask(image_path, target_label)

    if not found:
        print(f"[cv_pipeline] '{target_label}' not detected — returning original.")
        return image_path, False

    blurred     = original.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    original_np = np.array(original)
    blurred_np  = np.array(blurred)
    mask_3ch    = np.stack([combined_mask] * 3, axis=-1)
    result_np   = np.where(mask_3ch == 1, blurred_np, original_np).astype(np.uint8)

    output_path = _save(Image.fromarray(result_np), image_path, output_suffix)
    print(f"[cv_pipeline] Blurred '{target_label}' → '{os.path.basename(output_path)}'")
    return output_path, True


def detect_and_inpaint(image_path: str, target_label: str,
                       inpaint_radius: int = 10,
                       output_suffix: str = "_inpainted") -> tuple[str, bool]:
    """
    Removes target_label and fills the region using OpenCV INPAINT_TELEA.
    Works best on simple/uniform backgrounds.
    Returns (output_path, success).
    """
    combined_mask, original, found = _build_mask(image_path, target_label)

    if not found:
        print(f"[cv_pipeline] '{target_label}' not detected — returning original.")
        return image_path, False

    img_bgr    = cv2.cvtColor(np.array(original), cv2.COLOR_RGB2BGR)
    mask_cv    = (combined_mask * 255).astype(np.uint8)
    result_bgr = cv2.inpaint(img_bgr, mask_cv, inpaint_radius, cv2.INPAINT_TELEA)
    result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)

    output_path = _save(Image.fromarray(result_rgb), image_path, output_suffix)
    print(f"[cv_pipeline] Inpainted '{target_label}' → '{os.path.basename(output_path)}'")
    return output_path, True


def detect_and_remove(image_path: str, target_label: str,
                      fill_color: tuple = (255, 255, 255),
                      output_suffix: str = "_removed") -> tuple[str, bool]:
    """
    Fills target_label's pixels with a solid color (default: white).
    Useful for visualising the mask. Returns (output_path, success).
    """
    combined_mask, original, found = _build_mask(image_path, target_label)

    if not found:
        print(f"[cv_pipeline] '{target_label}' not detected — returning original.")
        return image_path, False

    original_np = np.array(original)
    fill_np     = np.full_like(original_np, fill_color)
    mask_3ch    = np.stack([combined_mask] * 3, axis=-1)
    result_np   = np.where(mask_3ch == 1, fill_np, original_np).astype(np.uint8)

    output_path = _save(Image.fromarray(result_np), image_path, output_suffix)
    print(f"[cv_pipeline] Removed '{target_label}' → '{os.path.basename(output_path)}'")
    return output_path, True


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    TEST_IMAGE = sys.argv[1] if len(sys.argv) > 1 else "output/test.png"
    TEST_LABEL = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else "person"

    if not os.path.exists(TEST_IMAGE):
        print(f"Image not found: {TEST_IMAGE}")
        print("Usage: python3 cv_pipeline.py <image_path> <label>")
        sys.exit(1)

    print(f"\n── Step 1: Detecting objects in '{TEST_IMAGE}'")
    objects = list_detected_objects(TEST_IMAGE)
    for obj in objects:
        print(f"  {obj['label']:20s} confidence: {obj['confidence']:.2f}")

    print(f"\n── Step 2: Blurring '{TEST_LABEL}'")
    out_path, success = detect_and_blur(TEST_IMAGE, TEST_LABEL)
    if success:
        print(f"  ✅ Saved to: {out_path}")
    else:
        print(f"  ⚠️  '{TEST_LABEL}' not found. Available: {[o['label'] for o in objects]}")