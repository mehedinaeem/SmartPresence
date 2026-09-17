"""Read-only repository checks; never loads model binaries or downloads weights."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess

from src.fingerprint.fingerprint_database import COVERED_ROLLS, EXPECTED_ROLLS, FingerprintDatabase
from src.utils.helpers import image_files
from src.utils.paths import PROJECT_ROOT, path, settings
from src.utils.research import classifier_config, students


def validate_repository() -> list[tuple[str, str]]:
    findings = []

    def check(condition, message):
        findings.append(("PASS" if condition else "FAIL", message))

    try:
        cfg = settings()
        records = students()
        covered = {r["roll_number"] for r in records if r["face_status"] == "covered"}
        uncovered = {r["roll_number"] for r in records if r["face_status"] == "uncovered"}
        check(len(records) == 40, "40 participant records")
        check(len(uncovered) == 36, "36 uncovered participants")
        check(len(covered) == 4, "4 covered participants")
        check(covered == COVERED_ROLLS, "covered rolls match expected list")
        check({r["roll_number"] for r in records} == set(EXPECTED_ROLLS), "continuous roll range 22102001–22102040")
        check(cfg["ATTENDANCE_THRESHOLD"] == cfg["TESTING"]["attendance_threshold"] == .75, "attendance threshold = 75%")
        check(cfg["EXPERIMENT_FRAME_INTERVAL_SECONDS"] == cfg["TESTING"]["video_frame_interval_seconds"] == 3, "experiment sampling = 3 s")
        check(cfg["REALTIME_CCTV_FRAME_INTERVAL_SECONDS"] == 180, "realtime sampling = 180 s")
        check(cfg["RECOGNITION"]["model_name"] == "Facenet" and cfg["RECOGNITION"]["embedding_dimension"] == 128, "FaceNet dimension = 128")
        check(cfg["TESTING"]["normalization"] == cfg["RECOGNITION"]["embedding_normalization"] == "Facenet", "explicit FaceNet input normalization")
        params = classifier_config(cfg)
        check(params == {"kernel": "rbf", "C": 10., "gamma": "scale", "class_weight": "balanced", "normalization": "l2", "random_state": 42, "test_size": .2}, "future training = L2 Normalizer + configured RBF SVM")
        check(cfg["TRACKING"]["tracker"] == "bytetrack.yaml" and cfg["TRACKING"]["person_class_id"] == 0, "ByteTrack / person class 0")
        check(Path(cfg["TRACKING"]["model"]).name == "yolo11n.pt", "YOLO11n model configured")
        check((PROJECT_ROOT / cfg["TRACKING"]["model"]).is_file(), "YOLO model file available")
        check(cfg["TESTING"]["unknown_confidence_threshold"] == .55, "prototype unknown threshold = 0.55 (not calibrated)")
        for key in ("classifier", "label_encoder", "model_info", "students_metadata", "dataset_summary", "face_visibility", "fingerprint_database"):
            check(path(key).is_file(), f"{key} available")
        for name in ("preprocessing_summary.csv", "student_folder_mapping.csv", "processed_images.csv", "rejected_images.csv"):
            check((path("students_metadata").parent / name).is_file(), f"metadata/{name} available")
        FingerprintDatabase()
        check(True, "40 unique static fingerprint mappings validated against students.csv")
        info = json.loads(path("model_info").read_text())
        check(info.get("embedding_dimension") == 128, "saved metadata dimension = 128")
        check(info.get("student_count") == len(uncovered), "saved metadata describes 36 face classes")
        check(info.get("svm") == {key: params[key] for key in ("kernel", "C", "gamma", "class_weight")}, "saved SVM metadata matches future training")
        check(info.get("test_size") == params["test_size"] and info.get("random_state") == params["random_state"], "saved split and seed match configuration")
        check(info.get("train_samples", 0) + info.get("test_samples", 0) == info.get("successful_embeddings"), "saved embedding and split counts agree")
        with path("dataset_summary").open() as stream:
            snapshot_count = sum(int(row["processed_images"]) for row in csv.DictReader(stream))
        current_count = len(image_files(path("dataset_processed")))
        if snapshot_count != info.get("successful_embeddings"):
            findings.append(("WARNING", f"local preprocessing snapshot ({snapshot_count}) differs from imported Colab embeddings ({info.get('successful_embeddings')}); these are distinct histories"))
        if current_count != snapshot_count:
            findings.append(("WARNING", f"current processed files ({current_count}) differ from preprocessing metadata ({snapshot_count}); freeze/reconcile provenance before training"))
        if "embedding_normalization" not in info:
            findings.append(("WARNING", "imported metadata does not explicitly record embedding normalization; future runs record it, historical equivalence is unverified"))
        if not path("classifier").is_file() or not path("label_encoder").is_file():
            findings.append(("WARNING", "artifact existence checks do not prove serialized model compatibility"))
        for report in sorted(path("experiments").glob("*/metrics.json")):
            metrics = json.loads(report.read_text())
            findings.append(("PASS" if metrics else "WARNING", f"{report.relative_to(PROJECT_ROOT)}: {'metrics available' if metrics else 'empty historical placeholder, not a result'}"))
        check((PROJECT_ROOT / "outputs/reports/preprocessing_report.md").is_file(), "existing preprocessing report available")
        if not (PROJECT_ROOT / "LICENSE").exists():
            findings.append(("WARNING", "no LICENSE assigned; owner must choose code and data terms before release"))
        try:
            tracked = subprocess.check_output(["git", "ls-files", "-z", "dataset/processed"], cwd=PROJECT_ROOT).split(b"\0")
            count = sum(bool(name) for name in tracked)
            if count:
                findings.append(("WARNING", f"{count} processed files already tracked by Git; .gitignore does not remove them or their history"))
        except (OSError, subprocess.CalledProcessError):
            findings.append(("WARNING", "Git tracking/privacy audit unavailable"))
    except (OSError, ValueError, KeyError, TypeError) as error:
        findings.append(("FAIL", f"Repository validation could not complete: {error}"))
    return findings
