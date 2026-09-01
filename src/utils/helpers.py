from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def image_files(folder: Path) -> list[Path]:
    return sorted(item for item in folder.rglob("*") if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS)


def append_csv(file_path: Path, fieldnames: list[str], row: dict) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    exists = file_path.exists() and file_path.stat().st_size > 0
    with file_path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fieldnames})


def validate_roll_number(value: str) -> str:
    value = str(value).strip()
    if len(value) != 8 or not value.isdigit():
        raise ValueError(f"Invalid roll number: {value!r}")
    return value

