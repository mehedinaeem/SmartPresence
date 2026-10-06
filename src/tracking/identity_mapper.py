"""Explicit, session-scoped associations only: track 7 never implies roll 22102007.

Callers must verify the student/session before association. This in-memory mapping
is not authentication, cross-camera re-identification, or automatic identity inference.
Use a fresh mapper for each session/stream; IDs can be reused by trackers.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import re

from src.utils.helpers import utc_now, validate_roll_number


@dataclass(frozen=True)
class TrackIdentity:
    track_id: int
    roll_number: str
    entry_time: str
    face_status: str
    session_id: str


class IdentityMapper:
    def __init__(self) -> None:
        self._tracks: dict[int, TrackIdentity] = {}

    def associate(self, track_id: int, roll_number: str, session_id: str, face_status: str = "covered") -> TrackIdentity:
        identity = TrackIdentity(int(track_id), validate_roll_number(roll_number), utc_now(), face_status, session_id)
        self._tracks[identity.track_id] = identity
        return identity

    def resolve(self, track_id: int) -> TrackIdentity | None:
        return self._tracks.get(int(track_id))

    def export(self) -> list[dict]:
        return [asdict(item) for item in self._tracks.values()]



@dataclass(frozen=True)
class StableStudentIdentity:
    stable_id: str | int
    roll_number: str
    session_id: str
    entry_time: str


class StableIdentityMapper:
    """Explicit verified bindings for ONE stream/session; never accepts raw IDs.

    The caller must select the visible stable identity belonging to the verified
    student. A fingerprint lookup alone cannot determine which box is theirs.
    Create a fresh mapper whenever PersonTracker.reset() starts another stream.
    """

    def __init__(self, session_id: str) -> None:
        if not session_id.strip():
            raise ValueError("A verified session_id is required")
        self.session_id = session_id
        self._identities: dict[str | int, StableStudentIdentity] = {}

    def associate_verified(self, stable_id: str | int, verification: dict) -> StableStudentIdentity:
        if verification.get("verified") is not True:
            raise ValueError("Student identity must be explicitly verified")
        if not ((isinstance(stable_id, str) and re.fullmatch(r"S[0-9]{3,}", stable_id) and int(stable_id[1:]) > 0)
                or (type(stable_id) is int and stable_id > 0)):
            raise ValueError("An S001-style stable_id (or legacy positive integer) is required")
        roll = validate_roll_number(verification["roll_number"])
        existing = self._identities.get(stable_id)
        if existing:
            if existing.roll_number != roll:
                raise ValueError("Stable identity already belongs to another student")
            return existing
        if any(item.roll_number == roll for item in self._identities.values()):
            raise ValueError("Student already has a stable identity; explicit re-verification/rebinding is required")
        identity = StableStudentIdentity(stable_id, roll, self.session_id, utc_now())
        self._identities[stable_id] = identity
        return identity

    def resolve(self, stable_id: str | int) -> StableStudentIdentity | None:
        return self._identities.get(stable_id)

    def export(self) -> list[dict]:
        return [asdict(item) for item in self._identities.values()]
