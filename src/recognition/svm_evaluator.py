"""Evaluate an explicitly selected future training run into a fresh run directory."""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support

from src.utils.research import create_run, file_hash


def evaluate_model(training_run: Path, run_id: str | None = None) -> dict:
    source = Path(training_run)
    provenance = json.loads((source / "experiment_config.json").read_text())
    if provenance["stage"] != "training":
        raise ValueError("Select a training run, not an embedding/evaluation directory")
    dataset = json.loads((source / "dataset_manifest.json").read_text())
    classifier = joblib.load(source / "face_classifier.pkl")
    encoder = joblib.load(source / "label_encoder.pkl")
    with (source / "evaluation_split.pkl").open("rb") as stream:
        split = pickle.load(stream)
    predicted, truth = classifier.predict(split["x_test"]), split["y_test"]
    labels = list(range(len(encoder.classes_)))
    report = classification_report(truth, predicted, labels=labels, target_names=encoder.classes_, output_dict=True, zero_division=0)
    overall = {"accuracy": float(accuracy_score(truth, predicted))}
    for average in ("macro", "weighted"):
        precision, recall, f1, _ = precision_recall_fscore_support(truth, predicted, average=average, zero_division=0)
        overall.update({f"{average}_precision": float(precision), f"{average}_recall": float(recall), f"{average}_f1": float(f1)})
    output = create_run("evaluation", dataset, run_id, provenance["settings"])
    pd.DataFrame(report).transpose().to_csv(output / "classification_report.csv")
    pd.DataFrame(confusion_matrix(truth, predicted, labels=labels), index=encoder.classes_, columns=encoder.classes_).to_csv(output / "confusion_matrix.csv")
    (output / "metrics.json").write_text(json.dumps(overall, indent=2) + "\n")
    (output / "source_run.json").write_text(json.dumps({"training_run": str(source.resolve()),
        "artifacts": {name: file_hash(source / name) for name in
                      ("face_classifier.pkl", "label_encoder.pkl", "evaluation_split.pkl", "model_info.json")}}, indent=2) + "\n")
    return {"run_directory": str(output), **overall}
