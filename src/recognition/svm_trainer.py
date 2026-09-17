"""Explicit future training; imported classifier artifacts are never overwritten."""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, Normalizer
from sklearn.svm import SVC

from src.utils.paths import settings
from src.utils.research import classifier_config, create_run, file_hash, students


def build_classifier(config: dict | None = None) -> Pipeline:
    parameters = classifier_config(config)
    return Pipeline([
        ("normalizer", Normalizer(norm=parameters["normalization"])),
        ("svc", SVC(**{key: parameters[key] for key in
                     ("kernel", "C", "gamma", "class_weight", "random_state")}, probability=True)),
    ])


def train_model(embeddings_path: Path, run_id: str | None = None) -> dict:
    cfg = settings()
    with Path(embeddings_path).open("rb") as stream:
        data = pickle.load(stream)
    if "dataset_manifest" not in data or "embedding_config" not in data:
        raise ValueError("Embeddings lack provenance. Explicitly run 02_extract_embeddings.py for a new research run.")
    expected = {key: cfg["RECOGNITION"][key] for key in
                ("model_name", "embedding_dimension", "embedding_normalization")}
    if data["embedding_config"] != expected:
        raise ValueError("Embedding configuration does not match current research configuration")
    vectors = np.asarray(data["embeddings"])
    labels = np.asarray(data["labels"])
    if vectors.ndim != 2 or vectors.shape[1] != expected["embedding_dimension"] or len(vectors) != len(labels):
        raise ValueError("Embeddings must be an N × 128 matrix with one label per row")
    if not np.isfinite(vectors).all():
        raise ValueError("Embeddings contain non-finite values")
    expected_rolls = {student["roll_number"] for student in students() if student["face_status"] == "uncovered"}
    if set(labels) != expected_rolls:
        raise ValueError("Training labels must contain exactly the 36 uncovered participants")
    manifest = data["dataset_manifest"]
    if len(labels) != manifest["number_of_images"] or len(data.get("sources", [])) != len(labels):
        raise ValueError("Embedding counts/sources disagree with dataset provenance")
    if data["sources"] != [entry["path"] for entry in manifest["files"]]:
        raise ValueError("Embedding source ordering disagrees with dataset manifest")
    if any(label != source.split("/")[0] for label, source in zip(labels, data["sources"])):
        raise ValueError("Embedding labels disagree with source folders")
    encoder = LabelEncoder()
    encoded = encoder.fit_transform(labels)
    parameters = classifier_config(cfg)
    indices = np.arange(len(labels))
    train_indices, test_indices = train_test_split(indices, test_size=parameters["test_size"],
                                                   random_state=parameters["random_state"], stratify=encoded)
    classifier = build_classifier(cfg)
    output = create_run("training", data["dataset_manifest"], run_id, cfg)
    classifier.fit(vectors[train_indices], encoded[train_indices])
    joblib.dump(classifier, output / "face_classifier.pkl")
    joblib.dump(encoder, output / "label_encoder.pkl")
    with (output / "evaluation_split.pkl").open("wb") as stream:
        pickle.dump({"x_test": vectors[test_indices], "y_test": encoded[test_indices]}, stream)
    split = {"train_indices": train_indices.tolist(), "test_indices": test_indices.tolist(),
             "sources": data.get("sources", [])}
    (output / "split_indices.json").write_text(json.dumps(split, indent=2) + "\n")
    info = {"run_id": output.name, "facenet_model": expected["model_name"],
            "embedding_dimension": int(vectors.shape[1]), "number_of_classes": len(encoder.classes_),
            "number_of_training_samples": len(train_indices), "number_of_testing_samples": len(test_indices),
            "classifier": parameters, "source_embeddings": str(Path(embeddings_path).resolve()),
            "source_embeddings_sha256": file_hash(Path(embeddings_path)),
            "dataset_hash": data["dataset_manifest"]["dataset_hash"]}
    (output / "model_info.json").write_text(json.dumps(info, indent=2) + "\n")
    return {"run_directory": str(output), **info}
