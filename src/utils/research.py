"""Research configuration and content-addressed manifests for future runs."""
from __future__ import annotations

import csv
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform
import re
import subprocess
from datetime import datetime, timezone
from uuid import uuid4

from src.utils.helpers import image_files
from src.utils.paths import PROJECT_ROOT, path, settings

PACKAGES = ("deepface", "tensorflow", "tf-keras", "mtcnn", "numpy", "scipy", "scikit-learn",
            "joblib", "opencv-python", "pandas", "Pillow", "ultralytics", "torch", "PyYAML", "pytest")


def classifier_config(config: dict | None = None) -> dict:
    cfg = settings() if config is None else config
    recognition = cfg["RECOGNITION"]
    result = {
        "kernel": recognition["svm_kernel"], "C": float(recognition["svm_C"]),
        "gamma": recognition["svm_gamma"], "class_weight": recognition["svm_class_weight"],
        "normalization": recognition["normalization"],
        "random_state": int(cfg["RANDOM_STATE"]), "test_size": float(cfg["TRAIN_TEST_SPLIT"]),
    }
    if result["kernel"] != "rbf" or result["normalization"] != "l2":
        raise ValueError("Research training requires an L2 Normalizer and RBF SVM")
    if result["C"] <= 0 or not 0 < result["test_size"] < 1:
        raise ValueError("SVM C must be positive and test_size must be between 0 and 1")
    return result


def students() -> list[dict]:
    with path("students_metadata").open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def environment_versions() -> dict:
    packages = {}
    for name in PACKAGES:
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": packages}


def file_hash(source: Path) -> str:
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dataset_manifest(root: Path) -> dict:
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {root}")
    files = [{"path": p.relative_to(root).as_posix(), "sha256": file_hash(p)} for p in image_files(root)]
    if not files:
        raise ValueError(f"No dataset images in {root}")
    payload = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    rolls = sorted({Path(item["path"]).parts[0] for item in files})
    return {"dataset_root": str(root), "dataset_hash": hashlib.sha256(payload).hexdigest(),
            "hash_algorithm": "sha256 of canonical JSON sorted relative-path/content-sha256 records; v1",
            "number_of_images": len(files), "number_of_students": len(rolls), "rolls": rolls, "files": files}


def create_run(stage: str, dataset: dict, run_id: str | None = None,
               config: dict | None = None, runs_root: Path | None = None) -> Path:
    """Reserve a new directory; existing runs are never reused or overwritten."""
    cfg = settings() if config is None else config
    run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_") + uuid4().hex[:8]
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run_id):
        raise ValueError("run_id must be a single safe directory name")
    participants = students()
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=PROJECT_ROOT, text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    manifest = {
        "run_id": run_id, "stage": stage, "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit, "git_dirty": dirty,
        **{k: v for k, v in dataset.items() if k != "files"},
        "number_of_participants": len(participants),
        "covered_students": [r["roll_number"] for r in participants if r["face_status"] == "covered"],
        "uncovered_students": [r["roll_number"] for r in participants if r["face_status"] == "uncovered"],
        "facenet_model": cfg["RECOGNITION"]["model_name"],
        "embedding_dimension": cfg["RECOGNITION"]["embedding_dimension"],
        "embedding_normalization": cfg["RECOGNITION"]["embedding_normalization"],
        "classifier": classifier_config(cfg), "attendance_threshold": cfg["ATTENDANCE_THRESHOLD"],
        "video_sampling_interval": cfg["EXPERIMENT_FRAME_INTERVAL_SECONDS"],
        "realtime_interval": cfg["REALTIME_CCTV_FRAME_INTERVAL_SECONDS"],
        "environment": environment_versions(), "settings": cfg,
    }
    destination = (runs_root or path("experiments")) / run_id
    destination.mkdir(parents=True, exist_ok=False)
    for name, value in (("experiment_config.json", manifest), ("dataset_manifest.json", dataset)):
        (destination / name).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    (destination / "dataset_hash.txt").write_text(dataset["dataset_hash"] + "\n", encoding="utf-8")
    (destination / "README.md").write_text(
        f"# {run_id}\n\nStage: {stage}. See experiment_config.json for provenance.\n"
        "A directory alone does not establish successful completion; inspect stage artifacts.\n"
        "No metrics are implied for embedding or training-only runs.\n", encoding="utf-8")
    return destination
