from __future__ import annotations

import pickle

import cv2
import numpy as np

from src.recognition.facenet_embedder import FaceNetEmbedder
from src.utils.helpers import image_files
from src.utils.paths import path, settings


def extract_embeddings() -> dict[str, int]:
    root = path("dataset_processed")
    embedder = FaceNetEmbedder(settings()["RECOGNITION"]["model_name"])
    vectors, labels, sources = [], [], []
    for student_dir in sorted(item for item in root.iterdir() if item.is_dir()):
        for image_path in image_files(student_dir):
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            vectors.append(embedder.embed(image)); labels.append(student_dir.name); sources.append(str(image_path.relative_to(root)))
    if not vectors:
        raise RuntimeError("No processed images found. Run preprocessing first.")
    output = path("embeddings", create_parent=True)
    with output.open("wb") as stream:
        pickle.dump({"embeddings": np.stack(vectors), "labels": np.asarray(labels), "sources": sources}, stream)
    return {"samples": len(labels), "classes": len(set(labels))}

