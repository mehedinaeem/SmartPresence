from src.utils.helpers import append_csv, utc_now
from src.utils.paths import path

FIELDS = ["session_id", "roll_number", "track_id", "timestamp", "observed", "source", "confidence"]


def log_presence(session_id: str, roll_number: str, observed: bool, source: str, track_id: int | None = None, confidence: float | None = None) -> None:
    append_csv(path("attendance_logs") / "presence_log.csv", FIELDS, {"session_id": session_id, "roll_number": roll_number, "track_id": track_id, "timestamp": utc_now(), "observed": int(observed), "source": source, "confidence": confidence})

