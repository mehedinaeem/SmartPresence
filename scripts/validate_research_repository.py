"""Read-only research checks; warnings never modify repository files."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.repository_validation import validate_repository

if __name__ == "__main__":
    print("SmartPresence Research Repository Validation\n")
    findings = validate_repository()
    for level, message in findings:
        print(f"[{level}] {message}")
    print("\nMetadata and existence checks only; model binaries were not loaded.")
    raise SystemExit(int(any(level == "FAIL" for level, _ in findings)))
