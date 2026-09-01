from __future__ import annotations

import joblib
import numpy as np

from src.recognition.facenet_embedder import FaceNetEmbedder
from src.utils.paths import path, settings


class FaceRecognizer:
    def __init__(self) -> None:
        self.classifier = joblib.load(path("classifier")); self.encoder = joblib.load(path("label_encoder"))
        self.embedder = FaceNetEmbedder(settings()["RECOGNITION"]["model_name"])

    def recognize(self, face) -> tuple[str, float]:
        vector = self.embedder.embed(face).reshape(1, -1)
        probabilities = self.classifier.predict_proba(vector)[0]
        index = int(np.argmax(probabilities))
        return str(self.encoder.inverse_transform([index])[0]), float(probabilities[index])
