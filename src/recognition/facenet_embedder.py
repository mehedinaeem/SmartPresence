from __future__ import annotations

import numpy as np
from src.utils.paths import settings


class FaceNetEmbedder:
    """Lazy DeepFace FaceNet adapter; the FaceNet variant emits 128-D vectors."""

    def __init__(self, model_name: str = "Facenet") -> None:
        self.model_name = model_name

    def embed(self, face_bgr: np.ndarray) -> np.ndarray:
        from deepface import DeepFace

        result = DeepFace.represent(
            face_bgr,
            model_name=self.model_name,
            detector_backend="skip",
            enforce_detection=False,
            normalization=settings()["RECOGNITION"]["embedding_normalization"],
        )
        vector = np.asarray(result[0]["embedding"], dtype=np.float32)
        if vector.shape != (settings()["RECOGNITION"]["embedding_dimension"],):
            raise ValueError(f"Unexpected FaceNet embedding shape: {vector.shape}")
        return vector
