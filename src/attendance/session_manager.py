from __future__ import annotations

import uuid

from src.utils.helpers import append_csv, utc_now, validate_roll_number
from src.utils.paths import path

FIELDS = ["session_id", "roll_number", "fingerprint_id", "entry_time", "exit_time", "verification_status"]


def start_session(roll_number: str, fingerprint_id: str) -> str:
    session_id = uuid.uuid4().hex
    append_csv(path("attendance_logs") / "session_records.csv", FIELDS, {"session_id": session_id, "roll_number": validate_roll_number(roll_number), "fingerprint_id": fingerprint_id, "entry_time": utc_now(), "verification_status": "verified"})
    return session_id


def record_exit(session_id: str, roll_number: str, fingerprint_id: str) -> None:
    append_csv(path("attendance_logs") / "session_records.csv", FIELDS, {"session_id": session_id, "roll_number": validate_roll_number(roll_number), "fingerprint_id": fingerprint_id, "exit_time": utc_now(), "verification_status": "exit_verified"})

