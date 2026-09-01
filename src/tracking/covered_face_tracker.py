from src.tracking.person_detector import PersonDetector
from src.tracking.tracker import CentroidTracker


class CoveredFaceTracker:
    def __init__(self) -> None:
        self.detector, self.tracker = PersonDetector(), CentroidTracker()

    def process(self, frame):
        return self.tracker.update(self.detector.detect(frame))

