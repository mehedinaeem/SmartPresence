"""Compatibility import for the original stable tracker; IDs now use S001 format."""
from src.tracking.stable_identity_manager import StableIdentityManager, StableIdentitySettings


class StableIdentityTracker(StableIdentityManager):
    def __init__(self, max_gap_frames: int = 120) -> None:
        super().__init__(StableIdentitySettings(max_missing_frames=max_gap_frames,
                                               max_reassociation_gap_frames=max_gap_frames))
