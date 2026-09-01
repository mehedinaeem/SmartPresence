from __future__ import annotations

import cv2


class PersonDetector:
    """OpenCV HOG baseline; replace with a validated research detector if needed."""

    def __init__(self) -> None:
        self.hog = cv2.HOGDescriptor(); self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def detect(self, frame) -> list[tuple[int, int, int, int]]:
        boxes, _ = self.hog.detectMultiScale(frame, winStride=(8, 8))
        return [tuple(map(int, box)) for box in boxes]

