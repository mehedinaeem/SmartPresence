"""Threshold trade-offs from supplied face-validation scores, not fingerprint metrics.

Supply independent known/unknown validation CSVs with one face per row and a
confidence column (maximum SVM probability before Unknown rejection). This
measures known/unknown acceptance only, not correct student identification.
"""
import argparse
import csv
import math
from pathlib import Path


def scores(source: Path) -> list[float]:
    with source.open(newline="") as stream:
        result = [float(row["confidence"]) for row in csv.DictReader(stream)]
    if not result or any(not math.isfinite(value) or not 0 <= value <= 1 for value in result):
        raise ValueError(f"{source}: provide nonempty validation confidence scores in [0, 1]")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--known", type=Path)
    parser.add_argument("--unknown", type=Path)
    args = parser.parse_args()
    if args.known is None or args.unknown is None:
        print("Provide --known and --unknown validation-score CSVs. No calibration performed; inference remains unchanged.")
        return 1
    try:
        known, unknown = scores(args.known), scores(args.unknown)
    except (OSError, ValueError, KeyError) as error:
        print(f"Validation data error: {error}")
        return 1
    print("Face known/unknown acceptance only; no threshold is selected or saved.")
    print("threshold,precision,recall,false_accept_rate,false_reject_rate")
    for threshold in [i / 100 for i in range(0, 101, 5)]:
        tp = sum(value >= threshold for value in known)
        fp = sum(value >= threshold for value in unknown)
        precision = f"{tp / (tp + fp):.6f}" if tp + fp else "undefined"
        print(f"{threshold:.2f},{precision},{tp / len(known):.6f},{fp / len(unknown):.6f},{1 - tp / len(known):.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
