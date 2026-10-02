"""
inpaint.py
──────────
Quick local test for the CV pipeline.
Scans your output/ folder, lets you pick an image,
shows what's detected, then blurs or removes a chosen object.

Usage:
    python3 inpaint.py
    python3 inpaint.py --image output/my_image.png
    python3 inpaint.py --image output/my_image.png --label person
    python3 inpaint.py --image output/my_image.png --label person --mode remove
"""

import os
import sys
import argparse
from cv_pipeline import list_detected_objects, detect_and_blur, detect_and_remove, detect_and_inpaint

OUTPUT_DIR = "output"


def pick_image() -> str:
    """Lists all images in output/ and lets user pick one interactively."""
    images = [
        f for f in os.listdir(OUTPUT_DIR)
        if f.endswith(".png") and "_blurred" not in f and "_removed" not in f
    ]

    if not images:
        print(f"No images found in {OUTPUT_DIR}/")
        sys.exit(1)

    images.sort()
    print("\n── Available images ──────────────────────────────")
    for i, name in enumerate(images):
        print(f"  [{i}] {name}")

    choice = input("\nPick image number: ").strip()
    try:
        return os.path.join(OUTPUT_DIR, images[int(choice)])
    except (ValueError, IndexError):
        print("Invalid choice.")
        sys.exit(1)


def pick_label(detected: list[dict]) -> str:
    """Shows detected objects and lets user pick one to blur/remove."""
    # Deduplicate labels while preserving order
    seen   = set()
    unique = []
    for obj in detected:
        if obj["label"] not in seen:
            seen.add(obj["label"])
            unique.append(obj)

    print("\n── Detected objects ──────────────────────────────")
    for i, obj in enumerate(unique):
        warning = f"  ⚠️  {obj['warning']}" if obj.get("warning") else ""
        print(f"  [{i}] {obj['label']:25s} confidence: {obj['confidence']:.2f}{warning}")

    choice = input("\nPick object to blur/remove: ").strip()
    try:
        return unique[int(choice)]["label"]
    except (ValueError, IndexError):
        print("Invalid choice.")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Test CV pipeline on local images")
    parser.add_argument("--image", type=str, default=None,
                        help="Path to image (skips interactive selection)")
    parser.add_argument("--label", type=str, default=None,
                        help="Object label to blur/remove (skips interactive selection)")
    parser.add_argument("--mode",  type=str, default="blur",
                        choices=["blur", "inpaint", "remove"],
                        help="'blur' Gaussian blur, 'inpaint' removes object cleanly, 'remove' fills with white (default: blur)")
    parser.add_argument("--radius", type=int, default=20,
                        help="Blur radius, only used in blur mode (default: 20)")
    parser.add_argument("--confidence", type=float, default=0.4,
                        help="Minimum detection confidence threshold (default: 0.4)")
    args = parser.parse_args()

    # ── Step 1: Pick image ────────────────────────────────────────────────────
    image_path = args.image if args.image else pick_image()

    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        sys.exit(1)

    print(f"\n── Image: {image_path}")

    # ── Step 2: Detect objects ────────────────────────────────────────────────
    print("\n── Running YOLO detection…")
    detected = list_detected_objects(image_path, min_confidence=args.confidence)

    if not detected:
        print(f"No objects detected above confidence {args.confidence}.")
        print("Try lowering --confidence, e.g. --confidence 0.2")
        sys.exit(0)

    # ── Step 3: Pick label ────────────────────────────────────────────────────
    label = args.label if args.label else pick_label(detected)

    # ── Step 3b: Pick mode interactively if not passed as argument ────────────
    if "--mode" not in sys.argv:
        print("\n── What do you want to do?")
        print("  [0] blur    — Gaussian blur over the object")
        print("  [1] inpaint — remove object, fill with background (OpenCV)")
        print("  [2] remove  — fill with white (shows exact pixel mask)")
        choice = input("\nPick mode: ").strip()
        mode = {"1": "inpaint", "2": "remove"}.get(choice, "blur")
    else:
        mode = args.mode

    print(f"\n── Target: '{label}' | Mode: {mode}")

    # ── Step 4: Apply blur or remove ──────────────────────────────────────────
    if mode == "blur":
        output_path, success = detect_and_blur(image_path, label, blur_radius=args.radius)
    elif mode == "inpaint":
        output_path, success = detect_and_inpaint(image_path, label)
    else:
        output_path, success = detect_and_remove(image_path, label)

    # ── Step 5: Report result ─────────────────────────────────────────────────
    print()
    if success:
        print(f"✅ Done! Output saved to: {output_path}")
        print(f"   Open it to verify the result.")
    else:
        available = list(set(d["label"] for d in detected))
        print(f"⚠️  '{label}' was not detected in this image.")
        print(f"   Available labels: {available}")


if __name__ == "__main__":
    main()