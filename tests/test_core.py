import csv

from src.attendance.decision_engine import decide
from src.fingerprint.fingerprint_database import FingerprintDatabase
from src.utils.paths import path, settings
from src.video.frame_sampler import interval_for


def test_configuration():
    assert settings()["ATTENDANCE_THRESHOLD"] == 0.75
    assert interval_for("experiment") == 3
    assert interval_for("realtime") == 180


def test_metadata_has_expected_rolls():
    with path("students_metadata").open() as stream:
        rows = list(csv.DictReader(stream))
    assert [row["roll_number"] for row in rows] == [str(value) for value in range(22102001, 22102041)]
    assert FingerprintDatabase().roll_for("FP001") == "22102001"


def test_attendance_boundary():
    assert decide(.75).attendance_status == "present"
    assert decide(.749).attendance_status == "absent"
