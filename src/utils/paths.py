from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def settings() -> dict[str, Any]:
    with (PROJECT_ROOT / "config/settings.yaml").open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


@lru_cache(maxsize=1)
def configured_paths() -> dict[str, Path]:
    with (PROJECT_ROOT / "config/paths.yaml").open(encoding="utf-8") as stream:
        values = yaml.safe_load(stream)
    return {key: PROJECT_ROOT / value for key, value in values.items()}


def path(key: str, *, create_parent: bool = False) -> Path:
    result = configured_paths()[key]
    if create_parent:
        result.parent.mkdir(parents=True, exist_ok=True)
    return result

