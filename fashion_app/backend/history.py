"""
history.py
──────────
Thread-safe helpers for reading and writing session history JSON files.
"""

import os
import json
import threading
from datetime import datetime

from config import HISTORY_DIR

_history_lock = threading.Lock()


def save_history(kind: str, entry: dict) -> None:
    """Appends entry to history/{kind}.json — thread-safe."""
    os.makedirs(HISTORY_DIR, exist_ok=True)
    path = os.path.join(HISTORY_DIR, f"{kind}.json")
    with _history_lock:
        history = []
        if os.path.exists(path):
            with open(path) as f:
                history = json.load(f)
        entry["timestamp"] = datetime.now().isoformat()
        history.append(entry)
        with open(path, "w") as f:
            json.dump(history, f, indent=2)


def load_history(kind: str) -> list:
    """Used only for the history panel in the UI — never for generation."""
    path = os.path.join(HISTORY_DIR, f"{kind}.json")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f)