from __future__ import annotations

import numpy as np


class FaceNetEmbedder:
    """Lazy DeepFace FaceNet adapter; the FaceNet variant emits 128-D vectors."""

    def __init__(self, model_name: str = "Facenet") -> None:
        self.model_name = model_name

    def embed(self, face_bgr: np.ndarray) -> np.ndarray:
        from deepface import DeepFace

        result = DeepFace.represent(face_bgr, model_name=self.model_name, detector_backend="skip", enforce_detection=False)
        return np.asarray(result[0]["embedding"], dtype=np.float32)
