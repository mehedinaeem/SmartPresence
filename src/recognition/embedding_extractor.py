"""Extract processed crops into a new, provenance-bearing embedding run."""
from __future__ import annotations

import pickle

import cv2
import numpy as np

from src.recognition.facenet_embedder import FaceNetEmbedder
from src.utils.paths import path, settings
from src.utils.research import create_run, dataset_manifest, students


def extract_embeddings(run_id: str | None = None) -> dict:
    root = path("dataset_processed")
    cfg = settings()
    manifest = dataset_manifest(root)
    allowed = {s["roll_number"] for s in students() if s["face_status"] == "uncovered"}
    if set(manifest["rolls"]) != allowed:
        raise ValueError("Processed images must cover exactly the uncovered participants")
    output = create_run("embeddings", manifest, run_id, cfg)
    embedder = FaceNetEmbedder(cfg["RECOGNITION"]["model_name"])
    vectors, labels, sources = [], [], []
    for entry in manifest["files"]:
        source = entry["path"]
        image = cv2.imread(str(root / source))
        if image is None:
            raise ValueError(f"Unreadable image: {source}; run left incomplete")
        vectors.append(embedder.embed(image))
        labels.append(source.split("/")[0])
        sources.append(source)
    if dataset_manifest(root)["dataset_hash"] != manifest["dataset_hash"]:
        raise RuntimeError("Dataset changed during extraction; no embeddings saved")
    with (output / "embeddings.pkl").open("wb") as stream:
        pickle.dump({"embeddings": np.stack(vectors), "labels": np.asarray(labels), "sources": sources,
                     "dataset_manifest": manifest,
                     "embedding_config": {key: cfg["RECOGNITION"][key] for key in
                                          ("model_name", "embedding_dimension", "embedding_normalization")}}, stream)
    return {"run_directory": str(output), "samples": len(labels), "classes": len(set(labels))}
