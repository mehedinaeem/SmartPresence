from __future__ import annotations

import json
import pickle

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support

from src.utils.paths import PROJECT_ROOT, path


def evaluate_model(experiment: str = "exp_001_baseline") -> dict:
    classifier, encoder = joblib.load(path("classifier")), joblib.load(path("label_encoder"))
    with (path("embeddings").parent / "evaluation_split.pkl").open("rb") as stream:
        split = pickle.load(stream)
    predicted = classifier.predict(split["x_test"]); truth = split["y_test"]
    report = classification_report(truth, predicted, target_names=encoder.classes_, output_dict=True, zero_division=0)
    overall = {"accuracy": accuracy_score(truth, predicted)}
    for average in ("macro", "weighted"):
        precision, recall, f1, _ = precision_recall_fscore_support(truth, predicted, average=average, zero_division=0)
        overall.update({f"{average}_precision": precision, f"{average}_recall": recall, f"{average}_f1": f1})
    output_dir = PROJECT_ROOT / "experiments" / experiment; output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(report).transpose().to_csv(output_dir / "classification_report.csv")
    pd.DataFrame(confusion_matrix(truth, predicted), index=encoder.classes_, columns=encoder.classes_).to_csv(output_dir / "confusion_matrix.csv")
    (output_dir / "metrics.json").write_text(json.dumps(overall, indent=2), encoding="utf-8")
    return overall

