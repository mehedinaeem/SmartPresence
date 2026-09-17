"""Validated static fingerprint identity mapping; no biometric matching."""
from __future__ import annotations

import csv
from pathlib import Path

from src.utils.paths import path

FIELDS = ["fingerprint_id", "roll_number", "finger", "template_id", "face_status", "status"]
EXPECTED_ROLLS = [str(value) for value in range(22102001, 22102041)]
COVERED_ROLLS = {"22102002", "22102030", "22102031", "22102033"}


def normalize_fingerprint_id(fingerprint_id: str) -> str:
    return fingerprint_id.strip().upper()


def _read_csv(source: Path, required: set[str]) -> list[dict[str, str]]:
    with source.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{source}: missing required columns {sorted(required - set(reader.fieldnames or []))}")
        rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError(f"{source}: malformed CSV row")
    return rows


def _students(source: Path) -> dict[str, dict[str, str]]:
    rows = _read_csv(source, {"roll_number", "face_status", "active"})
    if len(rows) != 40 or {row["roll_number"] for row in rows} != set(EXPECTED_ROLLS):
        raise ValueError("students.csv must contain exactly 40 unique rolls 22102001–22102040")
    for row in rows:
        expected_face = "covered" if row["roll_number"] in COVERED_ROLLS else "uncovered"
        if row["face_status"] != expected_face:
            raise ValueError(f"students.csv: incorrect face_status for {row['roll_number']}; expected {expected_face}")
        if row["active"] not in {"0", "1"}:
            raise ValueError(f"students.csv: invalid active value for {row['roll_number']}")
    return {row["roll_number"]: row for row in rows}


def generate_fingerprint_database(destination: Path | None = None, students_path: Path | None = None) -> Path:
    """Explicitly generate the prototype CSV from authoritative student metadata."""
    students = _students(students_path or path("students_metadata"))
    destination = destination or path("fingerprint_database")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        for index, roll in enumerate(EXPECTED_ROLLS, 1):
            student = students[roll]
            writer.writerow({
                "fingerprint_id": f"FP{index:03d}", "roll_number": roll,
                "finger": "right_index", "template_id": f"T{index:03d}",
                "face_status": student["face_status"],
                "status": "active" if student["active"] == "1" else "inactive",
            })
    return destination


class FingerprintDatabase:
    def __init__(self, database_path: Path | None = None, students_path: Path | None = None) -> None:
        self.students = _students(students_path or path("students_metadata"))
        rows = _read_csv(database_path or path("fingerprint_database"), set(FIELDS))
        if len(rows) != 40:
            raise ValueError("Fingerprint database must contain exactly 40 rows")
        expected_sets = {
            "fingerprint_id": {f"FP{i:03d}" for i in range(1, 41)},
            "roll_number": set(EXPECTED_ROLLS),
            "template_id": {f"T{i:03d}" for i in range(1, 41)},
        }
        for field, expected in expected_sets.items():
            if {row[field] for row in rows} != expected:
                raise ValueError(f"Fingerprint database: {field} must contain all 40 expected unique values (no duplicates or missing IDs)")
        for row in rows:
            roll = row["roll_number"]
            index = EXPECTED_ROLLS.index(roll) + 1
            if row["fingerprint_id"] != f"FP{index:03d}" or row["template_id"] != f"T{index:03d}":
                raise ValueError(f"Fingerprint database: incorrect ID mapping for {roll}")
            if row["finger"] != "right_index":
                raise ValueError(f"Fingerprint database: invalid finger for {roll}; expected right_index")
            if row["face_status"] not in {"covered", "uncovered"} or row["face_status"] != self.students[roll]["face_status"]:
                raise ValueError(f"Fingerprint database: face_status disagrees with students.csv for {roll}")
            if row["status"] not in {"active", "inactive"}:
                raise ValueError(f"Fingerprint database: invalid status for {roll}")
        self.records = {row["fingerprint_id"]: row for row in rows}
        # Preserve the existing active-ID and reverse-lookup interfaces.
        self.by_fingerprint = {
            key: row["roll_number"] for key, row in self.records.items()
            if row["status"] == "active" and self.students[row["roll_number"]]["active"] == "1"
        }
        self.by_roll = {roll: key for key, roll in self.by_fingerprint.items()}

    def student_for(self, fingerprint_id: str) -> dict[str, str] | None:
        record = self.records.get(normalize_fingerprint_id(fingerprint_id))
        return dict(record) if record is not None else None

    def verify(self, fingerprint_id: str) -> dict:
        key = normalize_fingerprint_id(fingerprint_id)
        record = self.student_for(key)
        if record is None:
            return {"fingerprint_id": key, "verified": False, "reason": "Fingerprint ID not found"}
        if key not in self.by_fingerprint:
            return {**record, "verified": False, "reason": "Student fingerprint record is inactive"}
        return {**record, "verified": True}

    def roll_for(self, fingerprint_id: str) -> str | None:
        return self.by_fingerprint.get(normalize_fingerprint_id(fingerprint_id))

    def fingerprint_for(self, roll_number: str) -> str | None:
        return self.by_roll.get(str(roll_number))
