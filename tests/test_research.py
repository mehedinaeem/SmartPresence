"""Lightweight reproducibility tests: no training of real models or downloads."""
import copy
import json
import pickle
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from src.attendance.attendance_calculator import calculate
from src.recognition.facenet_embedder import FaceNetEmbedder
from src.recognition.svm_trainer import build_classifier, train_model
from src.tracking.identity_mapper import IdentityMapper
from src.utils.paths import path, settings
from src.utils.research import classifier_config, create_run, dataset_manifest, students

pytestmark = pytest.mark.unit


def test_research_defaults_and_saved_metadata():
    cfg = settings()
    info = json.loads(path("model_info").read_text())
    params = classifier_config()
    assert params == {"kernel": "rbf", "C": 10., "gamma": "scale", "class_weight": "balanced", "normalization": "l2", "random_state": 42, "test_size": .2}
    assert info["svm"] == {k: params[k] for k in ("kernel", "C", "gamma", "class_weight")}
    assert info["embedding_dimension"] == cfg["RECOGNITION"]["embedding_dimension"] == 128
    assert info["train_samples"] + info["test_samples"] == info["successful_embeddings"]
    assert info["student_count"] == len([s for s in students() if s["face_status"] == "uncovered"]) == 36
    assert cfg["TRACKING"]["person_class_id"] == 0
    assert cfg["TRACKING"]["tracker"] == "bytetrack.yaml"
    assert Path(cfg["TRACKING"]["model"]).name == "yolo11n.pt"
    assert cfg["TESTING"]["attendance_threshold"] == cfg["ATTENDANCE_THRESHOLD"] == .75
    assert cfg["TESTING"]["video_frame_interval_seconds"] == cfg["EXPERIMENT_FRAME_INTERVAL_SECONDS"] == 3
    assert cfg["REALTIME_CCTV_FRAME_INTERVAL_SECONDS"] == 180
    assert cfg["TESTING"]["unknown_confidence_threshold"] == .55


def test_classifier_pipeline():
    classifier = build_classifier()
    assert classifier.named_steps["normalizer"].norm == "l2"
    svc = classifier.named_steps["svc"]
    assert (svc.kernel, svc.C, svc.gamma, svc.class_weight, svc.random_state, svc.probability) == ("rbf", 10., "scale", "balanced", 42, True)
    cfg = copy.deepcopy(settings())
    cfg["RECOGNITION"]["svm_kernel"] = "linear"
    with pytest.raises(ValueError, match="RBF"):
        classifier_config(cfg)


def test_embedding_contract_without_deepface(monkeypatch):
    represent = Mock(return_value=[{"embedding": [2.] * 128}])
    monkeypatch.setitem(sys.modules, "deepface", SimpleNamespace(DeepFace=SimpleNamespace(represent=represent)))
    result = FaceNetEmbedder().embed(np.zeros((160, 160, 3)))
    assert result.shape == (128,)
    assert result[0] == 2  # no premature L2 normalization
    assert represent.call_args.kwargs == {"model_name": "Facenet", "detector_backend": "skip", "enforce_detection": False, "normalization": "Facenet"}
    represent.return_value = [{"embedding": [1.] * 512}]
    with pytest.raises(ValueError, match="shape"):
        FaceNetEmbedder().embed(np.zeros((160, 160, 3)))


def test_existing_inference_does_not_double_normalize(monkeypatch):
    from src.recognition.face_recognizer import FaceRecognizer
    from sklearn.preprocessing import LabelEncoder
    recognizer = FaceRecognizer.__new__(FaceRecognizer)
    recognizer.expected_dimension = 128
    recognizer.classifier_normalizes = True
    recognizer.unknown_threshold = .55
    recognizer.encoder = LabelEncoder().fit(["22102001"])
    recognizer.classifier = SimpleNamespace(classes_=np.array([0]), predict_proba=Mock(return_value=np.array([[.9]])))
    assert recognizer._predict(np.full(128, 2.))[0] == "22102001"
    assert np.all(recognizer.classifier.predict_proba.call_args.args[0] == 2.)
    with pytest.raises(ValueError, match="128"):
        recognizer._predict(np.ones(512))


def test_attendance_percentage_and_boundary():
    data = pd.DataFrame({"session_id": ["a"] * 8, "roll_number": ["22102001"] * 4 + ["22102003"] * 4, "observed": [1, 1, 1, 0, 1, 1, 0, 0]})
    result = calculate(data).set_index("roll_number")
    assert result.loc["22102001", "presence_percentage"] == .75
    assert result.loc["22102001", "attendance_status"] == "present"
    assert result.loc["22102003", "presence_percentage"] == .5
    assert result.loc["22102003", "attendance_status"] == "absent"
    with pytest.raises(ValueError, match="Missing columns"):
        calculate(data.drop(columns="observed"))


def test_temporary_track_identity():
    mapper = IdentityMapper()
    assert mapper.resolve(7) is None
    record = mapper.associate(7, "22102030", "verified-session")
    assert record.roll_number == "22102030"  # never inferred as 22102007
    assert record.track_id == 7 and record.session_id == "verified-session"
    assert mapper.export()[0]["roll_number"] == "22102030"
    assert IdentityMapper().resolve(7) is None  # stream/session-local lifetime
    with pytest.raises(ValueError):
        mapper.associate(8, "8", "session")


def test_manifest_content_hash_and_immutable_runs(tmp_path):
    root = tmp_path / "dataset" / "22102001"
    root.mkdir(parents=True)
    (root / "a.jpg").write_bytes(b"synthetic bytes for hash test, not a face image")
    first = dataset_manifest(root.parent)
    assert dataset_manifest(root.parent) == first
    output = create_run("unit-test", first, "test_run", runs_root=tmp_path / "runs")
    record = json.loads((output / "experiment_config.json").read_text())
    assert record["dataset_hash"] == first["dataset_hash"]
    assert record["number_of_images"] == 1 and record["number_of_participants"] == 40
    assert len(record["covered_students"]) == 4 and len(record["uncovered_students"]) == 36
    assert record["environment"]["python"] and record["classifier"]["normalization"] == "l2"
    before = (output / "experiment_config.json").read_bytes()
    with pytest.raises(FileExistsError):
        create_run("unit-test", first, "test_run", runs_root=tmp_path / "runs")
    assert (output / "experiment_config.json").read_bytes() == before
    (root / "a.jpg").write_bytes(b"changed test bytes")
    assert dataset_manifest(root.parent)["dataset_hash"] != first["dataset_hash"]
    with pytest.raises(ValueError):
        create_run("unit-test", first, "../escape", runs_root=tmp_path / "runs")


def test_training_rejects_unprovenanced_embeddings(tmp_path):
    source = tmp_path / "old.pkl"
    source.write_bytes(pickle.dumps({"embeddings": np.zeros((10, 128)), "labels": ["22102001"] * 10}))
    with pytest.raises(ValueError, match="provenance"):
        train_model(source)


def test_unknown_calibration_requires_both_inputs(monkeypatch, capsys):
    from scripts.evaluate_unknown_threshold import main
    monkeypatch.setattr(sys, "argv", ["evaluate_unknown_threshold.py"])
    assert main() == 1
    assert "No calibration performed" in capsys.readouterr().out


def test_numbered_image_cli_help_without_models():
    import subprocess
    for name in ("05_test_single_image.py", "06_test_group_image.py", "03_train_model.py", "04_evaluate_model.py"):
        result = subprocess.run([sys.executable, f"scripts/{name}", "--help"], text=True, capture_output=True)
        assert result.returncode == 0, result.stderr


def test_future_training_output_is_isolated(tmp_path, monkeypatch):
    import src.recognition.svm_trainer as trainer
    labels = [row["roll_number"] for row in students() if row["face_status"] == "uncovered" for _ in range(5)]
    sources = [f"{roll}/{i}.jpg" for i, roll in enumerate(labels)]
    cfg = settings()["RECOGNITION"]
    payload = {"embeddings": np.ones((len(labels), 128)), "labels": labels, "sources": sources,
               "embedding_config": {k: cfg[k] for k in ("model_name", "embedding_dimension", "embedding_normalization")},
               "dataset_manifest": {"number_of_images": len(labels), "dataset_hash": "unit-test-only", "files": [{"path": s} for s in sources]}}
    source = tmp_path / "embeddings.pkl"
    source.write_bytes(pickle.dumps(payload))
    run = tmp_path / "training"
    run.mkdir()
    create = Mock(return_value=run)
    classifier = Mock()
    monkeypatch.setattr(trainer, "create_run", create)
    monkeypatch.setattr(trainer, "build_classifier", Mock(return_value=classifier))
    dump = Mock()
    monkeypatch.setattr(trainer.joblib, "dump", dump)
    result = trainer.train_model(source, "unit-test")
    assert result["number_of_classes"] == 36
    assert result["number_of_training_samples"] == 144
    assert result["number_of_testing_samples"] == 36
    classifier.fit.assert_called_once()  # mocked; no model training occurs
    assert all(call.args[1].parent == run for call in dump.call_args_list)
    split = json.loads((run / "split_indices.json").read_text())
    assert not set(split["train_indices"]) & set(split["test_indices"])
    assert sorted(split["train_indices"] + split["test_indices"]) == list(range(180))
    assert not (run / "metrics.json").exists()


def test_validator_detects_policy_drift(monkeypatch):
    import src.utils.repository_validation as validation
    cfg = copy.deepcopy(settings())
    cfg["TESTING"]["video_frame_interval_seconds"] = 9
    monkeypatch.setattr(validation, "settings", lambda: cfg)
    findings = validation.validate_repository()
    assert ("FAIL", "experiment sampling = 3 s") in findings


def test_tracking_processes_consecutive_frames_without_model_download(monkeypatch):
    import src.tracking.person_tracker as tracking
    yolo = Mock()
    yolo.track.return_value = []
    constructor = Mock(return_value=yolo)
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=constructor))
    tracker = tracking.PersonTracker()
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    for _ in range(4):
        tracker.track(frame)
    assert yolo.track.call_count == 4
    assert yolo.track.call_args.kwargs["persist"] is True
    assert yolo.track.call_args.kwargs["tracker"] == "bytetrack.yaml"
    assert yolo.track.call_args.kwargs["classes"] == [0]


def test_evaluation_uses_saved_split_and_refuses_overwrite(tmp_path, monkeypatch):
    import src.recognition.svm_evaluator as evaluator
    from src.utils.research import file_hash
    source = tmp_path / "training"
    source.mkdir()
    (source / "experiment_config.json").write_text(json.dumps({"stage": "training", "settings": settings()}))
    (source / "dataset_manifest.json").write_text(json.dumps({"dataset_hash": "test-only", "number_of_images": 2}))
    (source / "evaluation_split.pkl").write_bytes(pickle.dumps({"x_test": np.ones((2, 128)), "y_test": np.array([0, 1])}))
    for name in ("face_classifier.pkl", "label_encoder.pkl", "model_info.json"):
        (source / name).write_bytes(b"mock artifact, not a trained model")
    before = {p.name: file_hash(p) for p in source.iterdir()}
    predictor = Mock(return_value=np.array([0, 1]))
    monkeypatch.setattr(evaluator.joblib, "load", lambda p: SimpleNamespace(predict=predictor) if p.name == "face_classifier.pkl" else SimpleNamespace(classes_=np.array(["22102001", "22102003"])))
    monkeypatch.setattr(evaluator, "create_run", lambda stage, data, run_id, cfg: create_run(stage, data, run_id, cfg, tmp_path / "runs"))
    result = evaluator.evaluate_model(source, "evaluation")
    output = Path(result["run_directory"])
    assert (output / "metrics.json").exists()
    assert (output / "classification_report.csv").exists()
    assert json.loads((output / "source_run.json").read_text())["artifacts"]["evaluation_split.pkl"] == before["evaluation_split.pkl"]
    metrics = (output / "metrics.json").read_bytes()
    with pytest.raises(FileExistsError):
        evaluator.evaluate_model(source, "evaluation")
    assert (output / "metrics.json").read_bytes() == metrics
    assert before == {p.name: file_hash(p) for p in source.iterdir()}
