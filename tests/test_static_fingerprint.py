"""Static identity lookup and isolated session tests; no vision models required."""
import csv
import runpy
import subprocess
import sys

import pytest

from scripts.test_static_fingerprint import main
from src.attendance import session_manager
from src.fingerprint.fingerprint_database import (
    COVERED_ROLLS, FIELDS, FingerprintDatabase, generate_fingerprint_database,
)
from src.fingerprint.verification import verify_fingerprint
from src.utils.paths import PROJECT_ROOT, path


@pytest.mark.parametrize("index", [1, 2, 17, 30, 31, 33, 40])
def test_known_identity(index):
    result = verify_fingerprint(f"FP{index:03d}")
    roll = str(22102000 + index)
    assert result == {
        "fingerprint_id": f"FP{index:03d}", "roll_number": roll,
        "template_id": f"T{index:03d}", "finger": "right_index",
        "face_status": "covered" if roll in COVERED_ROLLS else "uncovered",
        "status": "active", "verified": True,
    }


@pytest.mark.parametrize("value", ["FP999", "", "   "])
def test_unknown_identity(value):
    result = verify_fingerprint(value)
    assert result["verified"] is False
    assert result["reason"] == "Fingerprint ID not found"
    assert FingerprintDatabase().roll_for(value) is None


def test_normalization_and_reverse_lookup():
    database = FingerprintDatabase()
    assert database.verify(" fp030 \n")["roll_number"] == "22102030"
    assert database.roll_for(" fp017 ") == "22102017"
    assert database.fingerprint_for("22102017") == "FP017"
    record = database.student_for("FP017")
    record["status"] = "inactive"
    assert database.verify("FP017")["verified"]  # callers cannot mutate stored records
    assert database.student_for("FP999") is None


def test_generated_database(tmp_path):
    generated = generate_fingerprint_database(tmp_path / "fingerprints.csv")
    with generated.open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 40
    for key in ("fingerprint_id", "roll_number", "template_id"):
        assert len({row[key] for row in rows}) == 40
    assert {row["fingerprint_id"] for row in rows if row["face_status"] == "covered"} == {"FP002", "FP030", "FP031", "FP033"}
    assert FingerprintDatabase(generated).records == FingerprintDatabase().records


@pytest.fixture
def records():
    with path("fingerprint_database").open() as stream:
        return list(csv.DictReader(stream))


def write_records(destination, rows, fields=FIELDS):
    with destination.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return destination


@pytest.mark.parametrize("field,value", [
    ("fingerprint_id", "FP002"), ("fingerprint_id", "FP999"),
    ("roll_number", "22102002"), ("roll_number", "22102041"),
    ("template_id", "T002"), ("template_id", "T999"),
    ("face_status", "covered"), ("face_status", "unknown"),
    ("status", "enabled"), ("finger", "left_index"),
])
def test_rejects_invalid_database(tmp_path, records, field, value):
    records[0][field] = value
    with pytest.raises(ValueError, match="Fingerprint database"):
        FingerprintDatabase(write_records(tmp_path / "bad.csv", records))


def test_rejects_wrong_count_and_swapped_mapping(tmp_path, records):
    with pytest.raises(ValueError, match="40 rows"):
        FingerprintDatabase(write_records(tmp_path / "short.csv", records[:-1]))
    records[0]["fingerprint_id"], records[1]["fingerprint_id"] = records[1]["fingerprint_id"], records[0]["fingerprint_id"]
    with pytest.raises(ValueError, match="incorrect ID mapping"):
        FingerprintDatabase(write_records(tmp_path / "swapped.csv", records))


def test_missing_columns(tmp_path):
    file = tmp_path / "bad.csv"
    file.write_text("fingerprint_id\nFP001\n")
    with pytest.raises(ValueError, match="missing required columns"):
        FingerprintDatabase(file)


def test_invalid_authoritative_metadata(tmp_path):
    with path("students_metadata").open() as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        students = list(reader)
    source = tmp_path / "students.csv"
    with pytest.raises(ValueError, match="40 unique rolls"):
        generate_fingerprint_database(tmp_path / "out.csv", write_records(source, students[:-1], fields))
    students[1]["face_status"] = "uncovered"
    with pytest.raises(ValueError, match="incorrect face_status"):
        FingerprintDatabase(students_path=write_records(source, students, fields))


def test_inactive_database_record(tmp_path, records, monkeypatch, capsys):
    records[29]["status"] = "inactive"
    database = FingerprintDatabase(write_records(tmp_path / "inactive.csv", records))
    assert database.student_for("FP030")["status"] == "inactive"
    assert database.roll_for("FP030") is None
    assert database.fingerprint_for("22102030") is None
    monkeypatch.setattr("scripts.test_static_fingerprint.verify_fingerprint", database.verify)
    monkeypatch.setattr("scripts.test_static_fingerprint.start_session", lambda *args: pytest.fail("Inactive record created a session"))
    assert main(["FP030", "--create-session"]) == 1
    assert "Student fingerprint record is inactive" in capsys.readouterr().out


def test_inactive_authoritative_student(tmp_path):
    with path("students_metadata").open() as stream:
        reader = csv.DictReader(stream)
        fields, students = reader.fieldnames, list(reader)
    students[0]["active"] = "0"
    source = write_records(tmp_path / "students.csv", students, fields)
    assert not FingerprintDatabase(students_path=source).verify("FP001")["verified"]
    generated = generate_fingerprint_database(tmp_path / "db.csv", source)
    assert FingerprintDatabase(generated, source).student_for("FP001")["status"] == "inactive"


def test_optional_sessions_append(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(session_manager, "path", lambda key: tmp_path)
    assert main(["FP017"]) == 0
    assert not (tmp_path / "session_records.csv").exists()
    for value in ("FP017", " fp030 "):
        assert main([value, "--create-session"]) == 0
    target = tmp_path / "session_records.csv"
    with target.open() as stream:
        rows = list(csv.DictReader(stream))
    assert [row["roll_number"] for row in rows] == ["22102017", "22102030"]
    assert [row["fingerprint_id"] for row in rows] == ["FP017", "FP030"]
    assert len({row["session_id"] for row in rows}) == 2
    from uuid import UUID
    for row in rows:
        UUID(row["session_id"])
        assert row["verification_status"] == "verified"
    previous = target.read_bytes()
    assert main(["FP999", "--create-session"]) == 1
    assert main(["", "--create-session"]) == 1
    assert target.read_bytes() == previous
    output = capsys.readouterr().out
    assert "FaceNet + SVM" in output and "YOLO + ByteTrack" in output


def test_interactive_demo(monkeypatch, capsys):
    monkeypatch.setattr("builtins.input", lambda prompt: "FP030")
    assert main([]) == 0
    output = capsys.readouterr().out
    assert "22102030" in output and "Covered" in output and "YOLO + ByteTrack" in output


def test_clear_validation_error(monkeypatch, capsys):
    def invalid(value):
        raise ValueError("Fingerprint database: duplicate IDs")
    monkeypatch.setattr("scripts.test_static_fingerprint.verify_fingerprint", invalid)
    assert main(["FP030"]) == 2
    assert "duplicate IDs" in capsys.readouterr().err


def test_direct_cli_without_vision_imports():
    result = subprocess.run([sys.executable, "scripts/test_static_fingerprint.py", "FP017"], cwd=PROJECT_ROOT, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert "FaceNet + SVM" in result.stdout


def test_existing_session_entrypoint(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(session_manager, "path", lambda key: tmp_path)
    monkeypatch.setattr(sys, "argv", ["scripts/10_run_full_system.py", "FP030"])
    runpy.run_path(str(PROJECT_ROOT / "scripts/10_run_full_system.py"), run_name="__main__")
    with (tmp_path / "session_records.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1 and rows[0]["roll_number"] == "22102030"
    assert "22102030" in capsys.readouterr().out
