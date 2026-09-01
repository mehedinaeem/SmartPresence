import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.recognition.svm_evaluator import evaluate_model

if __name__ == "__main__": print(evaluate_model())
