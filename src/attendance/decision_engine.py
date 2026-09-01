from dataclasses import dataclass

from src.utils.paths import settings


@dataclass(frozen=True)
class AttendanceDecision:
    presence_percentage: float
    attendance_status: str


def decide(presence_percentage: float, threshold: float | None = None) -> AttendanceDecision:
    cutoff = settings()["ATTENDANCE_THRESHOLD"] if threshold is None else threshold
    value = max(0.0, min(1.0, float(presence_percentage)))
    return AttendanceDecision(value, "present" if value >= cutoff else "absent")

