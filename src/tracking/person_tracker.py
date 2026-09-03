"""Reusable YOLO person detector with persistent ByteTrack identities."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2


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
    """Detect COCO people and maintain temporary IDs with ByteTrack."""

    def __init__(
        self,
        model_path: str | Path = "yolo11n.pt",
        confidence_threshold: float = 0.40,
        tracker_config: str = "bytetrack.yaml",
        person_class_id: int = 0,
        inference_size: int = 640,
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "Ultralytics is not installed. Run: pip install ultralytics"
            ) from exc

        self.model_path = str(model_path)
        self.confidence_threshold = confidence_threshold
        self.tracker_config = tracker_config
        self.person_class_id = person_class_id
        self.inference_size = inference_size
        try:
            self.model = YOLO(self.model_path)
        except Exception as exc:
            raise RuntimeError(
                f"Could not load tracking model '{self.model_path}': {exc}"
            ) from exc

    def reset(self) -> None:
        """Reset ByteTrack state before starting a different video stream."""
        predictor = getattr(self.model, "predictor", None)
        for tracker in getattr(predictor, "trackers", []) or []:
            reset = getattr(tracker, "reset", None)
            if callable(reset):
                reset()

    def track(self, frame: Any) -> list[dict[str, int | float]]:
        """Track people in one consecutive frame and return serializable records."""
        results = self.model.track(
            frame,
            persist=True,
            classes=[self.person_class_id],
            conf=self.confidence_threshold,
            imgsz=self.inference_size,
            tracker=self.tracker_config,
            verbose=False,
        )
        if not results or results[0].boxes is None:
            return []

        boxes = results[0].boxes
        if boxes.id is None:
            return []

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
        return tracks

    @staticmethod
    def annotate(frame: Any, tracks: list[dict[str, int | float]]) -> Any:
        """Return a copy of the frame with clean tracking labels and boxes."""
        annotated = frame.copy()
        color = (50, 220, 50)
        for track in tracks:
            x1, y1 = int(track["x1"]), int(track["y1"])
            x2, y2 = int(track["x2"]), int(track["y2"])
            label = f'ID {int(track["track_id"])} | {float(track["confidence"]):.1%}'
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
