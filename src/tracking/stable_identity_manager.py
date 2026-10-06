"""Session-level visual continuity above raw IDs; conservative, not biometric proof."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math
import time
from typing import Any

import numpy as np

from src.tracking.appearance_encoder import (
    AppearanceEncoder, HistogramAppearanceEncoder, aggregate_embeddings,
    cosine_similarity, normalize_embedding,
)
from src.tracking.identity_mapper import StableIdentityMapper

BBox = tuple[float, float, float, float]


@dataclass(frozen=True)
class StableIdentitySettings:
    enabled: bool = True
    max_missing_frames: int = 120
    max_reassociation_gap_frames: int = 120
    min_iou_for_reassociation: float = .30
    max_center_distance_ratio: float = .35
    min_size_ratio: float = .65
    max_size_ratio: float = 1.55
    overlap_iou_threshold: float = .20
    appearance_enabled: bool = True
    appearance_similarity_threshold: float = .90
    appearance_history_size: int = 20
    min_appearance_samples: int = 2
    min_appearance_confidence: float = .45
    motion_enabled: bool = True
    motion_prediction_frames: int = 5
    # Much stricter fallback when appearance is disabled or unavailable.
    geometry_only_max_gap_frames: int = 2
    geometry_only_min_iou: float = .70
    geometry_only_max_center_distance_ratio: float = .15

    def __post_init__(self) -> None:
        for name in ('max_missing_frames', 'max_reassociation_gap_frames', 'appearance_history_size',
                     'min_appearance_samples', 'motion_prediction_frames', 'geometry_only_max_gap_frames'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        if self.min_appearance_samples > self.appearance_history_size:
            raise ValueError('min_appearance_samples exceeds appearance_history_size')
        for name in ('min_iou_for_reassociation', 'overlap_iou_threshold', 'appearance_similarity_threshold',
                     'min_appearance_confidence', 'geometry_only_min_iou'):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f'{name} must be between 0 and 1')
        for name in ('max_center_distance_ratio', 'min_size_ratio', 'max_size_ratio',
                     'geometry_only_max_center_distance_ratio'):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be finite and positive')
        if self.min_size_ratio > 1 or self.max_size_ratio < 1:
            raise ValueError('Size ratio bounds must include 1')


@dataclass
class StableIdentity:
    stable_id: str
    current_track_id: int
    previous_track_ids: list[int]
    first_seen_frame: int
    last_seen_frame: int
    first_seen_time: float
    last_seen_time: float
    last_bbox: BBox
    confidence: float
    roll_number: str | None = None
    session_id: str | None = None
    lost_frame_count: int = 0
    active: bool = True
    expired: bool = False
    frames_visible: int = 0
    number_of_raw_id_changes: int = 0
    reassociation_count: int = 0
    longest_gap_frames: int = 0
    velocity: tuple[float, float] = (0., 0.)
    appearance_history: deque[np.ndarray] = field(default_factory=deque, repr=False)

    @property
    def appearance_embedding(self) -> np.ndarray | None:
        return aggregate_embeddings(self.appearance_history)

    @property
    def raw_track_ids(self) -> list[int]:
        return list(dict.fromkeys([*self.previous_track_ids, self.current_track_id]))


def box_iou(a: BBox, b: BBox) -> float:
    intersection = max(0., min(a[2], b[2]) - max(a[0], b[0])) * max(0., min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union > 0 else 0.


def center(box: BBox) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


class StableIdentityManager:
    """Retain lost candidates without inventing observations or forcing matches.

    Every update represents one consecutive frame, including empty detections.
    Timestamps are seconds in the stream (video frame/FPS or camera monotonic time).
    Without an explicit timestamp, use measured monotonic time since the first update.
    A fresh independent stream must call reset, clearing all student associations.
    """

    def __init__(self, settings: StableIdentitySettings | None = None,
                 appearance_encoder: AppearanceEncoder | None = None) -> None:
        self.settings = settings or StableIdentitySettings()
        self.encoder = (appearance_encoder or HistogramAppearanceEncoder()) if self.settings.appearance_enabled else None
        self.reset()

    def reset(self) -> None:
        self.frame_number = 0
        self.next_id = 1
        self.states: dict[str, StableIdentity] = {}
        self.raw_to_stable: dict[int, str] = {}
        self._mappers: dict[str, StableIdentityMapper] = {}
        self._last_timestamp: float | None = None
        self._clock_start: float | None = None

    def associate_roll(self, stable_id: str, roll_number: str, session_id: str, *, verified: bool = False) -> None:
        """Bind an explicitly selected visible person AFTER identity verification.

        There is no validated automatic fingerprint-to-box association. `verified`
        is a caller assertion, not authentication; use associate_verified with the
        existing fingerprint lookup result. Never pass a raw ID here.
        """
        self.associate_verified(stable_id, {'verified': verified, 'roll_number': roll_number}, session_id)

    def associate_verified(self, stable_id: str, verification: dict, session_id: str) -> None:
        identity = self.states.get(stable_id)
        if identity is None or not identity.active or identity.expired:
            raise ValueError('Select an existing currently visible stable identity')
        if identity.session_id is not None and identity.session_id != session_id:
            raise ValueError('Stable identity already belongs to another session')
        roll = verification.get('roll_number')
        if any(item.stable_id != stable_id and item.roll_number == roll for item in self.states.values()):
            raise ValueError('Student already belongs to another stable identity; do not automatically merge')
        mapper = self._mappers.get(session_id) or StableIdentityMapper(session_id)
        binding = mapper.associate_verified(stable_id, verification)
        self._mappers[session_id] = mapper
        identity.roll_number, identity.session_id = binding.roll_number, binding.session_id

    def update(self, frame: np.ndarray, tracks: list[dict[str, Any]],
               timestamp_seconds: float | None = None) -> list[dict[str, Any]]:
        if timestamp_seconds is None:
            now = time.monotonic()
            if self._clock_start is None:
                self._clock_start = now
            timestamp = now - self._clock_start
        else:
            timestamp = float(timestamp_seconds)
        if not math.isfinite(timestamp) or (self._last_timestamp is not None and timestamp < self._last_timestamp):
            raise ValueError('Stream timestamps must be finite and nondecreasing')
        raw_ids = [int(track['track_id']) for track in tracks]
        if len(raw_ids) != len(set(raw_ids)):
            raise ValueError('Duplicate raw tracker IDs in one frame')
        self.frame_number += 1
        self._last_timestamp = timestamp
        self._age()
        if not self.settings.enabled:
            return [{**track, 'stable_id': None, 'roll_number': None, 'session_id': None} for track in tracks]
        boxes = [tuple(float(track[k]) for k in ('x1', 'y1', 'x2', 'y2')) for track in tracks]
        crowded = [any(box_iou(box, other) > self.settings.overlap_iou_threshold
                       for j, other in enumerate(boxes) if i != j) for i, box in enumerate(boxes)]
        features = [self._extract(frame, box, tracks[i], crowded[i]) for i, box in enumerate(boxes)]
        assigned: dict[int, str] = {}
        used: set[str] = set()
        for i, raw_id in enumerate(raw_ids):
            sid = self.raw_to_stable.get(raw_id)
            state = self.states.get(sid)
            if state is None or state.expired:
                continue
            # Backend continuity is already evidence when the SAME raw ID returns
            # within its retention window. Sanity-check it, but do not demand a
            # second independent ReID decision just because detections were missed.
            compatible = self._compatible(state, boxes[i], features[i], recovery=False)
            if compatible:
                assigned[i] = sid
                used.add(sid)
            else:
                # Reject stale/reused raw aliases; keep the historical identity for reporting.
                self.raw_to_stable.pop(raw_id, None)
                state.expired = True
                state.appearance_history.clear()
        candidates: dict[int, list[str]] = {}
        for i, box in enumerate(boxes):
            if i in assigned or crowded[i]:
                continue
            candidates[i] = [sid for sid, state in self.states.items()
                             if sid not in used and not state.expired and state.current_track_id not in raw_ids
                             and self._compatible(state, box, features[i], recovery=True)]
        for i, options in candidates.items():
            if len(options) == 1 and sum(options[0] in values for values in candidates.values()) == 1:
                assigned[i] = options[0]
                used.add(options[0])
        result = []
        for i, track in enumerate(tracks):
            state = self.states.get(assigned.get(i))
            if state is None:
                sid = f'S{self.next_id:03d}'
                self.next_id += 1
                state = StableIdentity(sid, raw_ids[i], [], self.frame_number, self.frame_number,
                                       timestamp, timestamp, boxes[i], float(track['confidence']),
                                       appearance_history=deque(maxlen=self.settings.appearance_history_size))
                self.states[sid] = state
            self._observe(state, track, boxes[i], features[i], timestamp)
            result.append({**track, 'stable_id': state.stable_id,
                           'roll_number': state.roll_number, 'session_id': state.session_id})
        return result

    def _age(self) -> None:
        for state in self.states.values():
            state.active = False
            state.lost_frame_count = self.frame_number - state.last_seen_frame
            if state.lost_frame_count > self.settings.max_missing_frames:
                state.expired = True
                state.appearance_history.clear()
                if self.raw_to_stable.get(state.current_track_id) == state.stable_id:
                    self.raw_to_stable.pop(state.current_track_id)

    def _extract(self, frame: np.ndarray, box: BBox, track: dict, crowded: bool) -> np.ndarray | None:
        if self.encoder is None or crowded or float(track['confidence']) < self.settings.min_appearance_confidence:
            return None
        return normalize_embedding(self.encoder.extract(frame, box))

    def _compatible(self, state: StableIdentity, box: BBox, feature: np.ndarray | None, recovery: bool) -> bool:
        cfg, old = self.settings, state.last_bbox
        if not all(math.isfinite(value) for value in (*old, *box)):
            return False
        width, height = box[2] - box[0], box[3] - box[1]
        old_width, old_height = old[2] - old[0], old[3] - old[1]
        if min(width, height, old_width, old_height) <= 0:
            return False
        if not (cfg.min_size_ratio <= width / old_width <= cfg.max_size_ratio
                and cfg.min_size_ratio <= height / old_height <= cfg.max_size_ratio):
            return False
        gap = self.frame_number - state.last_seen_frame
        scale = math.hypot(old_width, old_height)
        distance = math.dist(center(box), center(old)) / scale
        if distance > (cfg.max_center_distance_ratio if recovery else max(.6, cfg.max_center_distance_ratio)):
            return False
        similarity = cosine_similarity(state.appearance_embedding, feature)
        if not recovery:
            return similarity is None or similarity >= .5
        if gap > cfg.max_reassociation_gap_frames or box_iou(old, box) < cfg.min_iou_for_reassociation:
            return False
        if cfg.motion_enabled and state.frames_visible >= 2:
            horizon = min(gap, cfg.motion_prediction_frames)
            old_center = center(old)
            predicted = (old_center[0] + state.velocity[0] * horizon, old_center[1] + state.velocity[1] * horizon)
            if math.dist(center(box), predicted) / scale > cfg.max_center_distance_ratio:
                return False
        # A contradictory appearance must never be bypassed by geometry-only matching.
        if similarity is not None and similarity < cfg.appearance_similarity_threshold:
            return False
        if similarity is not None and len(state.appearance_history) >= cfg.min_appearance_samples:
            return True
        return (gap <= cfg.geometry_only_max_gap_frames and box_iou(old, box) >= cfg.geometry_only_min_iou
                and distance <= cfg.geometry_only_max_center_distance_ratio)

    def _observe(self, state: StableIdentity, track: dict, box: BBox, feature: np.ndarray | None, timestamp: float) -> None:
        raw_id = int(track['track_id'])
        gap = self.frame_number - state.last_seen_frame
        if state.frames_visible:
            state.longest_gap_frames = max(state.longest_gap_frames, max(0, gap - 1))
            if gap > 1 or raw_id != state.current_track_id:
                state.reassociation_count += 1
            if gap == 1:
                old_center, new_center = center(state.last_bbox), center(box)
                state.velocity = tuple(.5 * v + .5 * (b - a) for v, a, b in zip(state.velocity, old_center, new_center))
            else:
                state.velocity = (0., 0.)
        if raw_id != state.current_track_id:
            self.raw_to_stable.pop(state.current_track_id, None)
            state.previous_track_ids.append(state.current_track_id)
            state.number_of_raw_id_changes += 1
        state.current_track_id = raw_id
        state.last_bbox, state.confidence = box, float(track['confidence'])
        state.last_seen_frame, state.last_seen_time = self.frame_number, timestamp
        state.lost_frame_count, state.active = 0, True
        state.frames_visible += 1
        if feature is not None:
            if state.appearance_history and feature.shape != state.appearance_history[0].shape:
                state.appearance_history.clear()
            state.appearance_history.append(feature)
        self.raw_to_stable[raw_id] = state.stable_id
