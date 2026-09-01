"""Substitutable face-detector interface and MTCNN implementation."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol
import numpy as np

@dataclass(frozen=True)
class FaceDetection:
    x: int; y: int; width: int; height: int; confidence: float
    @property
    def area(self) -> int: return self.width * self.height
    @property
    def box(self) -> tuple[int, int, int, int]: return self.x, self.y, self.width, self.height

class Detector(Protocol):
    def detect(self, image_rgb: np.ndarray) -> list[FaceDetection]: ...

class MTCNNFaceDetector:
    """MTCNN adapter. Input must be an RGB full-resolution image."""
    def __init__(self, device: str = "CPU:0") -> None:
        from mtcnn import MTCNN
        self._detector = MTCNN(device=device)

    def detect(self, image_rgb: np.ndarray) -> list[FaceDetection]:
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError(f"MTCNN expects RGB, got {image_rgb.shape}")
        result = []
        for item in self._detector.detect_faces(image_rgb):
            x, y, width, height = map(int, item["box"])
            result.append(FaceDetection(x, y, width, height, float(item["confidence"])))
        return result

    def detect_batch(self, images_rgb: list[np.ndarray]) -> list[list[FaceDetection]]:
        if not images_rgb: return []
        raw = self._detector.detect_faces(images_rgb)
        output = []
        for detections in raw:
            converted = []
            for item in detections:
                x, y, width, height = map(int, item["box"])
                converted.append(FaceDetection(x, y, width, height, float(item["confidence"])))
            output.append(converted)
        return output

def select_primary_face(detections: list[FaceDetection], confidence_threshold: float, min_width: int, min_height: int, dominance_ratio: float) -> tuple[FaceDetection | None, str | None, int]:
    confident = [d for d in detections if d.confidence >= confidence_threshold and d.width >= min_width and d.height >= min_height]
    if not detections: return None, "no_face_detected", 0
    if not confident:
        best = max(detections, key=lambda d: d.confidence)
        return None, ("detection_confidence_below_threshold" if best.confidence < confidence_threshold else "face_too_small"), len(detections)
    confident.sort(key=lambda d: (d.area, d.confidence), reverse=True)
    if len(confident) == 1 or confident[0].area >= confident[1].area * dominance_ratio:
        return confident[0], None, len(detections)
    return None, "multiple_ambiguous_faces", len(detections)
