from __future__ import annotations

import csv

from src.utils.paths import path


class FingerprintDatabase:
    def __init__(self) -> None:
        with path("students_metadata").open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        self.by_fingerprint = {row["fingerprint_id"]: row["roll_number"] for row in rows if row["active"] == "1"}
        self.by_roll = {roll: fingerprint for fingerprint, roll in self.by_fingerprint.items()}

    def roll_for(self, fingerprint_id: str) -> str | None:
        return self.by_fingerprint.get(fingerprint_id)

    def fingerprint_for(self, roll_number: str) -> str | None:
        return self.by_roll.get(str(roll_number))

