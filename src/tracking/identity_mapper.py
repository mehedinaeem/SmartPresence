from __future__ import annotations

from dataclasses import asdict, dataclass

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

