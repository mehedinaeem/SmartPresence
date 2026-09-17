"""Explicit future research run; existing runs are preserved."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", help="New run directory name; generated if omitted")
    parser.add_argument("--training-run", required=True, type=Path)
    args = parser.parse_args()
    from src.recognition.svm_evaluator import evaluate_model
    print(evaluate_model(args.training_run, run_id=args.run_id))
