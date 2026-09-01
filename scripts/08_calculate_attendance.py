import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from src.attendance.attendance_calculator import calculate
from src.utils.paths import path

if __name__ == "__main__":
    source=path("attendance_logs")/"presence_log.csv"
    result=calculate(pd.read_csv(source)); target=path("attendance_logs")/"final_attendance.csv"; result.to_csv(target,index=False); print(target)
