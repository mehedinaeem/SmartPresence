from __future__ import annotations

import pandas as pd

from src.attendance.decision_engine import decide


def calculate(presence_log: pd.DataFrame) -> pd.DataFrame:
    required = {"session_id", "roll_number", "observed"}
    missing = required - set(presence_log.columns)
    if missing: raise ValueError(f"Missing columns: {sorted(missing)}")
    summary = presence_log.groupby(["session_id", "roll_number"], as_index=False).agg(presence_count=("observed", "sum"), total_observations=("observed", "count"))
    summary["presence_percentage"] = summary["presence_count"] / summary["total_observations"]
    summary["attendance_status"] = summary["presence_percentage"].map(lambda value: decide(value).attendance_status)
    return summary

