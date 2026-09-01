from __future__ import annotations

import json
import pickle
from datetime import datetime, timezone

import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

from src.utils.paths import path, settings


def train_model() -> dict:
    cfg = settings()
    with path("embeddings").open("rb") as stream:
        data = pickle.load(stream)
    encoder = LabelEncoder(); encoded = encoder.fit_transform(data["labels"])
    x_train, x_test, y_train, y_test = train_test_split(data["embeddings"], encoded, test_size=cfg["TRAIN_TEST_SPLIT"], random_state=cfg["RANDOM_STATE"], stratify=encoded)
    classifier = SVC(kernel=cfg["RECOGNITION"]["svm_kernel"], probability=True, random_state=cfg["RANDOM_STATE"])
    classifier.fit(x_train, y_train)
    joblib.dump(classifier, path("classifier", create_parent=True)); joblib.dump(encoder, path("label_encoder", create_parent=True))
    split_path = path("embeddings").parent / "evaluation_split.pkl"
    with split_path.open("wb") as stream:
        pickle.dump({"x_test": x_test, "y_test": y_test}, stream)
    info = {"model_name": "FaceNet + SVM", "facenet_model": cfg["RECOGNITION"]["model_name"], "embedding_dimension": int(data["embeddings"].shape[1]), "svm_kernel": cfg["RECOGNITION"]["svm_kernel"], "number_of_classes": len(encoder.classes_), "number_of_training_samples": len(x_train), "number_of_testing_samples": len(x_test), "train_test_split": cfg["TRAIN_TEST_SPLIT"], "random_state": cfg["RANDOM_STATE"], "training_date": datetime.now(timezone.utc).isoformat()}
    path("model_info", create_parent=True).write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info

