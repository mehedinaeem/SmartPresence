"""Optional body appearance descriptors; never uses the FaceNet identity model."""
from __future__ import annotations

from typing import Protocol, Sequence

import cv2
import numpy as np


def normalize_embedding(value: np.ndarray | None) -> np.ndarray | None:
    """Return a finite L2 unit vector, or None for unusable features."""
    if value is None:
        return None
    vector = np.asarray(value, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if not vector.size or not np.isfinite(vector).all() or not np.isfinite(norm) or norm < 1e-12:
        return None
    return vector / norm


class AppearanceEncoder(Protocol):
    def extract(self, frame: np.ndarray, bbox: Sequence[float]) -> np.ndarray | None:
        """Extract an optional body descriptor for a clamped bounding box."""
        ...


class HistogramAppearanceEncoder:
    """Lightweight HSV torso histogram, not a learned person-ReID model.

    This deliberately independent encoder avoids unsupported tracker feature access.
    Matching clothing colors is weak evidence and always needs geometric gating.
    """

    def extract(self, frame: np.ndarray, bbox: Sequence[float]) -> np.ndarray | None:
        if frame is None or frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
            return None
        try:
            box = np.asarray(bbox, dtype=float).reshape(-1)
        except (TypeError, ValueError):
            return None
        if box.size != 4 or not np.isfinite(box).all():
            return None
        x1, y1, x2, y2 = box
        if x2 <= x1 or y2 <= y1:
            return None
        height, width = frame.shape[:2]
        left = max(0, min(width, int(x1 + .2 * (x2 - x1))))
        right = max(0, min(width, int(x2 - .2 * (x2 - x1))))
        top = max(0, min(height, int(y1 + .2 * (y2 - y1))))
        bottom = max(0, min(height, int(y2 - .2 * (y2 - y1))))
        if right - left < 4 or bottom - top < 4:
            return None
        hsv = cv2.cvtColor(frame[top:bottom, left:right], cv2.COLOR_BGR2HSV)
        histogram = cv2.calcHist([hsv], [0, 1, 2], None, [12, 4, 4], [0, 180, 0, 256, 0, 256])
        return normalize_embedding(histogram)


def aggregate_embeddings(history: Sequence[np.ndarray]) -> np.ndarray | None:
    """Normalize the mean of recent normalized descriptors before cosine matching."""
    return normalize_embedding(np.mean(history, axis=0)) if len(history) else None


def cosine_similarity(left: np.ndarray | None, right: np.ndarray | None) -> float | None:
    left, right = normalize_embedding(left), normalize_embedding(right)
    if left is None or right is None or left.shape != right.shape:
        return None
    return float(np.clip(np.dot(left, right), -1.0, 1.0))
