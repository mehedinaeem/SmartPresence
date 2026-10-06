"""Reusable YOLO person detector with persistent raw tracks and session-level visual identities."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2

from src.tracking.stable_identity_manager import StableIdentityManager, StableIdentitySettings


@dataclass(frozen=True)
class PersonTrack:
    """One tracked-person observation in a video frame."""

    track_id: int
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


class PersonTracker:
    """Detect people with a configurable backend and a separate stable manager."""

    def __init__(
        self,
        model_path: str | Path = str(Path(__file__).resolve().parents[2] / "models/tracking/yolo11s.pt"),
        confidence_threshold: float = 0.10,
        tracker_config: str = str(Path(__file__).resolve().parents[2] / "config/bytetrack_smartpresence.yaml"),
        person_class_id: int = 0,
        inference_size: int = 960,
        stable_max_gap_frames: int = 120,
        stable_settings: StableIdentitySettings | None = None,
        require_local_model: bool = False,
        cpu_threads: int | None = None,
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "Ultralytics is not installed. Install the pinned requirements with: pip install -r requirements.txt"
            ) from exc

        if not 0 < confidence_threshold <= 1 or inference_size < 32:
            raise ValueError("Tracking confidence must be in (0, 1]; inference size must be >= 32")
        if cpu_threads is not None and cpu_threads < 1:
            raise ValueError("cpu_threads must be positive")
        self.cpu_threads = cpu_threads
        self.identity_manager = StableIdentityManager(stable_settings or StableIdentitySettings(
            max_missing_frames=stable_max_gap_frames, max_reassociation_gap_frames=stable_max_gap_frames))
        self.stable_tracker = self.identity_manager  # Compatibility alias; never recreated per frame.
        self.model_path = str(model_path)
        self.confidence_threshold = confidence_threshold
        self.tracker_config = tracker_config
        self.person_class_id = person_class_id
        self.inference_size = inference_size
        if require_local_model and not Path(self.model_path).exists():
            raise RuntimeError(
                f"Tracking model '{self.model_path}' is unavailable locally. "
                "Provide existing weights with --model, for example models/tracking/yolo11n.pt. "
                "No detector download was attempted."
            )
        try:
            self.model = YOLO(self.model_path)
        except Exception as exc:
            raise RuntimeError(
                f"Could not load tracking model '{self.model_path}': {exc}. "
                "For offline or slower CPUs, select existing weights with --model models/tracking/yolo11n.pt. "
                "No automatic model fallback is applied."
            ) from exc

        if cpu_threads is not None:
            # Supported public callback: backend setup may reset PyTorch's thread
            # pool, so apply the explicit limit before each prediction begins.
            self.model.add_callback("on_predict_start", self._configure_cpu_threads)

    def _configure_cpu_threads(self, predictor: Any) -> None:
        import torch
        if getattr(getattr(predictor, "device", None), "type", None) == "cpu":
            torch.set_num_threads(self.cpu_threads)

    def reset(self) -> None:
        """Reset the selected backend and stable/student state for a new stream."""
        self.stable_tracker.reset()
        predictor = getattr(self.model, "predictor", None)
        for tracker in getattr(predictor, "trackers", []) or []:
            reset = getattr(tracker, "reset", None)
            if callable(reset):
                reset()

    def track(self, frame: Any, timestamp_seconds: float | None = None) -> list[dict[str, Any]]:
        """Track people in one consecutive frame and return serializable records."""
        try:
            results = self.model.track(
                frame, persist=True, classes=[self.person_class_id],
                conf=self.confidence_threshold, imgsz=self.inference_size,
                tracker=self.tracker_config, verbose=False,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Tracking failed with '{self.tracker_config}': {exc}. "
                "Install the pinned requirements; ByteTrack is available via --tracker bytetrack."
            ) from exc
        if not results or results[0].boxes is None:
            return self.identity_manager.update(frame, [], timestamp_seconds)

        boxes = results[0].boxes
        if boxes.id is None:
            return self.identity_manager.update(frame, [], timestamp_seconds)

        coordinates = boxes.xyxy.detach().cpu().tolist()
        confidences = boxes.conf.detach().cpu().tolist()
        track_ids = boxes.id.detach().cpu().tolist()
        classes = boxes.cls.detach().cpu().tolist()

        tracks: list[dict[str, int | float]] = []
        for xyxy, confidence, track_id, class_id in zip(
            coordinates, confidences, track_ids, classes
        ):
            if int(class_id) != self.person_class_id:
                continue
            x1, y1, x2, y2 = (int(round(value)) for value in xyxy)
            tracks.append(
                PersonTrack(
                    track_id=int(track_id),
                    confidence=float(confidence),
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                ).to_dict()
            )
        return self.identity_manager.update(frame, tracks, timestamp_seconds)

    @staticmethod
    def annotate(frame: Any, tracks: list[dict[str, Any]]) -> Any:
        """Return a copy of the frame with clean tracking labels and boxes."""
        annotated = frame.copy()
        color = (50, 220, 50)
        for track in tracks:
            x1, y1 = int(track["x1"]), int(track["y1"])
            x2, y2 = int(track["x2"]), int(track["y2"])
            visual = track.get("stable_id")
            raw = f'T{int(track["track_id"])}'
            identity = f'{track["roll_number"]} | {visual}' if track.get("roll_number") else f'{visual} | {raw}' if visual else raw
            label = f'{identity} | {float(track["confidence"]):.1%}'
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            (text_width, text_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
            )
            label_top = max(0, y1 - text_height - 10)
            cv2.rectangle(
                annotated, (x1, label_top), (x1 + text_width + 8, y1), color, -1
            )
            cv2.putText(
                annotated,
                label,
                (x1 + 4, max(text_height, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )
        return annotated
