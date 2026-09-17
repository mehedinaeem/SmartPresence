"""Compatibility entry point; delegates to the existing implementation."""
from pathlib import Path
import runpy

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("08_calculate_attendance.py")), run_name="__main__")
